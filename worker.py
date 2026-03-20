import os
import json
import time
import tempfile
import logging
import shutil
import subprocess
from typing import Any, Dict, List, Optional

import boto3
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# CONFIG
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

AWS_BUCKET = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

if not AWS_BUCKET:
    raise RuntimeError("Missing AWS_BUCKET")

s3 = boto3.client("s3", region_name=AWS_REGION)

PENDING = "jobs/pending/"
PROCESSING = "jobs/processing/"
FINISHED = "jobs/finished/"
FAILED = "jobs/failed/"

REPORT_MODES = frozenset(
    {"report", "report_bundle", "report_th_en", "report_generator"}
)


# -----------------------------
# SAFE HELPERS
# -----------------------------
def safe_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except Exception:
        return -1


def retry(func, retries=2, delay=1.0):
    for i in range(retries + 1):
        try:
            return func()
        except Exception as e:
            if i == retries:
                raise
            logging.warning("Retry %s failed: %s", i + 1, e)
            time.sleep(delay)


def s3_object_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=AWS_BUCKET, Key=key)
        return True
    except Exception:
        return False


# -----------------------------
# S3
# -----------------------------
def s3_read_json(key: str) -> Dict[str, Any]:
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def s3_write_json(key: str, payload: Dict[str, Any]) -> None:
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json; charset=utf-8",
    )


def s3_upload_file(path: str, key: str, content_type: str) -> None:
    s3.upload_file(path, AWS_BUCKET, key, ExtraArgs={"ContentType": content_type})


def list_pending(limit: int = 200, max_scan: int = 800) -> List[str]:
    """
    Dots jobs first, then skeleton/other. Report jobs excluded entirely.
    Paginates pending prefix so we are not limited to the first S3 page blindly.
    """
    raw: List[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=PENDING):
        for item in page.get("Contents", []):
            k = str(item.get("Key") or "")
            if k.endswith(".json"):
                raw.append(k)
                if len(raw) >= max_scan:
                    break
        if len(raw) >= max_scan:
            break

    dots_keys: List[str] = []
    other_keys: List[str] = []
    for k in raw:
        try:
            job = s3_read_json(k)
            mode = str(job.get("mode") or "").strip().lower()
            if mode in REPORT_MODES:
                continue
            if mode == "dots":
                dots_keys.append(k)
            else:
                other_keys.append(k)
        except Exception:
            other_keys.append(k)

    return (dots_keys + other_keys)[:limit]


def claim_job(key: str) -> Optional[str]:
    name = key.split("/")[-1]
    new_key = PROCESSING + name
    try:
        s3.copy_object(
            Bucket=AWS_BUCKET,
            CopySource={"Bucket": AWS_BUCKET, "Key": key},
            Key=new_key,
            ContentType="application/json; charset=utf-8",
            MetadataDirective="REPLACE",
        )
        s3.delete_object(Bucket=AWS_BUCKET, Key=key)
        return new_key
    except Exception:
        return None


def move_job(key: str, prefix: str) -> str:
    name = key.split("/")[-1]
    new_key = prefix + name
    s3.copy_object(
        Bucket=AWS_BUCKET,
        CopySource={"Bucket": AWS_BUCKET, "Key": key},
        Key=new_key,
        ContentType="application/json; charset=utf-8",
        MetadataDirective="REPLACE",
    )
    s3.delete_object(Bucket=AWS_BUCKET, Key=key)
    return new_key


# -----------------------------
# VIDEO CORE
# -----------------------------
def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    return cap


def write_mp4(path: str, fps: float, w: int, h: int) -> cv2.VideoWriter:
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps or 25.0, (w, h))
    if not vw.isOpened():
        raise RuntimeError("VideoWriter failed")
    return vw


def validate_video_file(path: str) -> None:
    if not os.path.exists(path):
        raise RuntimeError("File missing")
    if os.path.getsize(path) <= 0:
        raise RuntimeError("Empty video")
    cap = cv2.VideoCapture(path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    cap.release()
    if frames <= 0:
        raise RuntimeError("No frames")
    if fps <= 0:
        raise RuntimeError("Invalid fps")


# -----------------------------
# FFMPEG
# -----------------------------
def run_ffmpeg(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or b"").decode()[-400:])


def transcode(input_path: str, output_path: str) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        input_path,
        "-vf",
        "scale='if(gt(iw,854),854,iw)':'if(gt(ih,480),480,ih)':force_original_aspect_ratio=decrease,"
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=24,format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-movflags",
        "+faststart",
        "-an",
        output_path,
    ]
    run_ffmpeg(cmd)
    validate_video_file(output_path)


# -----------------------------
# GENERATE (MediaPipe)
# -----------------------------
try:
    from mediapipe.python.solutions.pose import Pose
except ImportError:
    import mediapipe as mp

    Pose = mp.solutions.pose.Pose


def generate_dots_video(input_path: str, out_path: str) -> None:
    cap = open_video(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    vw = write_mp4(out_path, fps, w, h)

    with Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if res.pose_landmarks:
                for lm in res.pose_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(canvas, (x, y), 3, (255, 255, 255), -1)

            vw.write(canvas)

    vw.release()
    cap.release()


def generate_skeleton_video(input_path: str, out_path: str) -> None:
    cap = open_video(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    vw = write_mp4(out_path, fps, w, h)

    with Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if res.pose_landmarks:
                for lm in res.pose_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)

            vw.write(frame)

    vw.release()
    cap.release()


# -----------------------------
# JOB PROCESSOR
# -----------------------------
def process_job(job: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(job.get("mode", "")).lower()
    input_key = job["input_key"]

    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "input.mp4")
        s3.download_file(AWS_BUCKET, input_key, in_path)

        if mode == "dots":
            raw = os.path.join(td, "dots_raw.mp4")
            out = os.path.join(td, "dots.mp4")

            logging.info("DOTS START job_id=%s", job.get("job_id"))
            generate_dots_video(in_path, raw)
            validate_video_file(raw)

            retry(lambda: transcode(raw, out))
            retry(lambda: s3_upload_file(out, job["output_key"], "video/mp4"))

            if not s3_object_exists(job["output_key"]):
                raise RuntimeError("DOTS upload failed")

            logging.info("DOTS DONE job_id=%s", job.get("job_id"))
            return {"ok": True}

        if mode == "skeleton":
            raw = os.path.join(td, "skeleton_raw.mp4")
            out = os.path.join(td, "skeleton.mp4")

            logging.info("SKELETON START job_id=%s", job.get("job_id"))
            generate_skeleton_video(in_path, raw)
            validate_video_file(raw)

            retry(lambda: transcode(raw, out))
            retry(lambda: s3_upload_file(out, job["output_key"], "video/mp4"))

            if not s3_object_exists(job["output_key"]):
                raise RuntimeError("SKELETON upload failed")

            logging.info("SKELETON DONE job_id=%s", job.get("job_id"))
            return {"ok": True}

        raise RuntimeError(f"Unknown mode: {mode}")


# -----------------------------
# MAIN LOOP
# -----------------------------
def main() -> None:
    logging.info("Worker started bucket=%s", AWS_BUCKET)

    while True:
        keys = list_pending()

        for key in keys:
            try:
                job_data = s3_read_json(key)
            except Exception:
                continue

            mode = str(job_data.get("mode", "")).lower()

            if mode in REPORT_MODES:
                continue

            processing_key = claim_job(key)
            if not processing_key:
                continue

            try:
                job = s3_read_json(processing_key)
                mode = str(job.get("mode", "")).lower()

                if mode in REPORT_MODES:
                    logging.info(
                        "Skip report job_id=%s — moving back to pending (report_worker)",
                        job.get("job_id"),
                    )
                    move_job(processing_key, PENDING)
                    continue

                logging.info("Processing job_id=%s mode=%s", job.get("job_id"), mode)

                result = process_job(job)

                job["status"] = "finished"
                job["result"] = result
                s3_write_json(processing_key, job)

                move_job(processing_key, FINISHED)

                logging.info("Finished job_id=%s mode=%s", job.get("job_id"), mode)

            except Exception as e:
                logging.exception("FAILED job_id=%s error=%s", job.get("job_id"), str(e))

                try:
                    job = s3_read_json(processing_key)
                    job["status"] = "failed"
                    job["message"] = str(e)
                    job["failed_at"] = time.time()
                    s3_write_json(processing_key, job)
                except Exception:
                    pass

                move_job(processing_key, FAILED)

            break

        time.sleep(1)


if __name__ == "__main__":
    main()
