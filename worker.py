import os
import json
import time
import logging
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import boto3
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

try:
    from mediapipe.python.solutions.pose import Pose, PoseLandmark
except ImportError:
    import mediapipe as mp

    Pose = mp.solutions.pose.Pose
    PoseLandmark = mp.solutions.pose.PoseLandmark

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
    {"report", "report_th_en", "report_generator", "report_bundle"}
)

# Body skeleton only (no face)
SKELETON_JOINT_IDS = [
    PoseLandmark.LEFT_SHOULDER,
    PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.LEFT_ELBOW,
    PoseLandmark.RIGHT_ELBOW,
    PoseLandmark.LEFT_WRIST,
    PoseLandmark.RIGHT_WRIST,
    PoseLandmark.LEFT_HIP,
    PoseLandmark.RIGHT_HIP,
    PoseLandmark.LEFT_KNEE,
    PoseLandmark.RIGHT_KNEE,
    PoseLandmark.LEFT_ANKLE,
    PoseLandmark.RIGHT_ANKLE,
]

SKELETON_EDGES = [
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
    (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
    (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
    (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
    (PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),
    (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
    (PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE),
]


# -----------------------------
# HELPERS
# -----------------------------
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


def list_pending(limit: int = 200) -> List[str]:
    """
    Pending video jobs: dots first, then skeleton. Report jobs omitted (report_worker).
    """
    keys: List[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=PENDING):
        for item in page.get("Contents", []):
            k = str(item.get("Key") or "")
            if k.endswith(".json"):
                keys.append(k)
                if len(keys) >= limit:
                    break
        if len(keys) >= limit:
            break

    dots_keys: List[str] = []
    other_keys: List[str] = []
    for k in keys:
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

    return dots_keys + other_keys


def claim_job(pending_key: str) -> Optional[str]:
    name = pending_key.split("/")[-1]
    new_key = PROCESSING + name
    try:
        s3.copy_object(
            Bucket=AWS_BUCKET,
            CopySource={"Bucket": AWS_BUCKET, "Key": pending_key},
            Key=new_key,
            ContentType="application/json; charset=utf-8",
            MetadataDirective="REPLACE",
        )
        s3.delete_object(Bucket=AWS_BUCKET, Key=pending_key)
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
    if not os.path.exists(path) or os.path.getsize(path) <= 0:
        raise RuntimeError("Invalid output video")
    cap = cv2.VideoCapture(path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    cap.release()
    if frames <= 0 or fps <= 0:
        raise RuntimeError("Output video has no valid frames/fps")


def run_ffmpeg(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or "")[-500:])


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


def _lm_to_px_spread(lm, w: int, h: int, spread: float) -> Tuple[int, int]:
    nx = 0.5 + (float(lm.x) - 0.5) * spread
    ny = 0.5 + (float(lm.y) - 0.5) * spread
    return int(nx * w), int(ny * h)


def generate_dots_video(input_path: str, out_path: str) -> None:
    cap = open_video(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = write_mp4(out_path, fps, w, h)

    scale = 2
    w2, h2 = w * scale, h * scale
    dots_pose_spread = 0.76
    dot_radius = 4
    glow = (220, 220, 220)
    core = (255, 255, 255)

    with Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            canvas = np.zeros((h2, w2, 3), dtype=np.uint8)
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                for lm in res.pose_landmarks.landmark:
                    cx, cy = _lm_to_px_spread(lm, w2, h2, dots_pose_spread)
                    if 0 <= cx < w2 and 0 <= cy < h2:
                        cv2.circle(canvas, (cx, cy), dot_radius + 1, glow, -1)
                        cv2.circle(canvas, (cx, cy), dot_radius, core, -1)
            out = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_LINEAR)
            vw.write(out)

    vw.release()
    cap.release()


def generate_skeleton_video(input_path: str, out_path: str) -> None:
    cap = open_video(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = write_mp4(out_path, fps, w, h)

    white = (255, 255, 255)
    sk_spread = 0.76
    scale = 2
    w2, h2 = w * scale, h * scale
    line_thick = 2
    joint_r = 2

    with Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark
                frame2 = cv2.resize(frame, (w2, h2), interpolation=cv2.INTER_LINEAR)
                for a, b in SKELETON_EDGES:
                    la, lb = lms[a], lms[b]
                    if la.visibility < 0.5 or lb.visibility < 0.5:
                        continue
                    xa, ya = _lm_to_px_spread(la, w2, h2, sk_spread)
                    xb, yb = _lm_to_px_spread(lb, w2, h2, sk_spread)
                    cv2.line(frame2, (xa, ya), (xb, yb), white, line_thick, cv2.LINE_8)
                for pid in SKELETON_JOINT_IDS:
                    lm = lms[pid]
                    if lm.visibility < 0.5:
                        continue
                    x, y = _lm_to_px_spread(lm, w2, h2, sk_spread)
                    cv2.circle(frame2, (x, y), joint_r, white, -1)
                frame = cv2.resize(frame2, (w, h), interpolation=cv2.INTER_LINEAR)
            vw.write(frame)

    vw.release()
    cap.release()


def process_job(job: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(job.get("mode") or "").strip()
    input_key = job["input_key"]

    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "input.mp4")
        s3.download_file(AWS_BUCKET, input_key, in_path)

        if mode == "dots":
            raw = os.path.join(td, "raw.mp4")
            out = os.path.join(td, "out.mp4")
            logging.info("DOTS start job_id=%s", job.get("job_id"))
            generate_dots_video(in_path, raw)
            validate_video_file(raw)
            retry(lambda: transcode(raw, out))
            retry(lambda: s3_upload_file(out, job["output_key"], "video/mp4"))
            if not s3_object_exists(job["output_key"]):
                raise RuntimeError("Upload failed")
            return {"ok": True, "mode": "dots"}

        if mode == "skeleton":
            raw = os.path.join(td, "raw.mp4")
            out = os.path.join(td, "out.mp4")
            logging.info("SKELETON start job_id=%s", job.get("job_id"))
            generate_skeleton_video(in_path, raw)
            validate_video_file(raw)
            retry(lambda: transcode(raw, out))
            retry(lambda: s3_upload_file(out, job["output_key"], "video/mp4"))
            if not s3_object_exists(job["output_key"]):
                raise RuntimeError("Upload failed")
            return {"ok": True, "mode": "skeleton"}

        raise RuntimeError(f"Unknown mode: {mode}")


def main() -> None:
    logging.info("Worker started bucket=%s", AWS_BUCKET)

    while True:
        keys = list_pending()
        if not keys:
            time.sleep(3)
            continue

        for pending_key in keys:
            pending_job: Optional[Dict[str, Any]] = None
            last_err: Optional[Exception] = None
            for attempt in range(4):
                try:
                    pending_job = s3_read_json(pending_key)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    err_s = str(e).lower()
                    if attempt < 3 and (
                        "nosuchkey" in err_s or "404" in err_s or "does not exist" in err_s
                    ):
                        time.sleep(0.35)
                        continue
                    pending_job = None
                    break

            if pending_job is None:
                logging.warning(
                    "Peek failed %s (%s). Re-listing queue (another worker may have claimed dots).",
                    pending_key,
                    last_err,
                )
                break

            mode = str(pending_job.get("mode") or "").strip().lower()
            if mode in REPORT_MODES:
                continue

            processing_key = claim_job(pending_key)
            if not processing_key:
                continue

            try:
                job = s3_read_json(processing_key)
                jid = job.get("job_id")
                gid = job.get("group_id")
                jmode = str(job.get("mode") or "").strip()
                logging.info("Processing job_id=%s group_id=%s mode=%s", jid, gid, jmode)

                result = process_job(job)
                job["status"] = "finished"
                job["result"] = result
                s3_write_json(processing_key, job)
                move_job(processing_key, FINISHED)
                logging.info("Finished job_id=%s mode=%s", jid, jmode)
            except Exception as e:
                logging.exception("Job failed: %s", e)
                try:
                    job = s3_read_json(processing_key)
                    job["status"] = "failed"
                    job["message"] = str(e)
                    s3_write_json(processing_key, job)
                except Exception:
                    pass
                move_job(processing_key, FAILED)

            break

        time.sleep(0.2)


if __name__ == "__main__":
    main()
