import os
import io
import json
import time
import math
import tempfile
import logging
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import boto3
import cv2
import numpy as np

from dotenv import load_dotenv
load_dotenv()

# MediaPipe
try:
    from mediapipe.python.solutions import pose as mp_pose_module
    from mediapipe.python.solutions.pose import Pose, PoseLandmark
except ImportError:
    import mediapipe as mp
    mp_pose_module = mp.solutions.pose
    Pose = mp_pose_module.Pose
    PoseLandmark = mp_pose_module.PoseLandmark


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


# -----------------------------
# HELPERS
# -----------------------------
def safe_size(p):
    try:
        return os.path.getsize(p)
    except:
        return -1


def retry(func, retries=2, delay=1):
    for i in range(retries + 1):
        try:
            return func()
        except Exception as e:
            if i == retries:
                raise
            logging.warning("Retry %s failed: %s", i+1, e)
            time.sleep(delay)


def s3_object_exists(key):
    try:
        s3.head_object(Bucket=AWS_BUCKET, Key=key)
        return True
    except:
        return False


def s3_read_json(key):
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode())


def s3_write_json(key, payload):
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=json.dumps(payload).encode(),
        ContentType="application/json"
    )


def s3_upload_file(path, key, content_type):
    s3.upload_file(path, AWS_BUCKET, key, ExtraArgs={"ContentType": content_type})


def list_pending(limit=200):
    keys = []
    res = s3.list_objects_v2(Bucket=AWS_BUCKET, Prefix=PENDING)
    for obj in res.get("Contents", []):
        k = obj["Key"]
        if k.endswith(".json"):
            keys.append(k)
    return keys[:limit]


def claim_job(key):
    name = key.split("/")[-1]
    new_key = PROCESSING + name
    try:
        s3.copy_object(Bucket=AWS_BUCKET, CopySource={"Bucket": AWS_BUCKET, "Key": key}, Key=new_key)
        s3.delete_object(Bucket=AWS_BUCKET, Key=key)
        return new_key
    except:
        return None


def move_job(key, prefix):
    name = key.split("/")[-1]
    new_key = prefix + name
    s3.copy_object(Bucket=AWS_BUCKET, CopySource={"Bucket": AWS_BUCKET, "Key": key}, Key=new_key)
    s3.delete_object(Bucket=AWS_BUCKET, Key=key)
    return new_key


# -----------------------------
# VIDEO CORE
# -----------------------------
def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    return cap


def write_mp4(path, fps, w, h):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps or 25, (w, h))
    if not vw.isOpened():
        raise RuntimeError("VideoWriter failed")
    return vw


def validate_video_file(path):
    if not os.path.exists(path):
        raise RuntimeError("File missing")
    if os.path.getsize(path) <= 0:
        raise RuntimeError("Empty file")

    cap = cv2.VideoCapture(path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if frames <= 0:
        raise RuntimeError("No frames")
    if fps <= 0:
        raise RuntimeError("Invalid fps")


# -----------------------------
# TRANSCODE
# -----------------------------
def run_ffmpeg(cmd):
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode()[-500:])


def transcode(input_path, output_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")

    cmd = [
        ffmpeg, "-y", "-i", input_path,
        "-vf", "scale=854:-2,fps=24",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "28",
        "-movflags", "+faststart",
        output_path
    ]
    run_ffmpeg(cmd)
    validate_video_file(output_path)


# -----------------------------
# GENERATE
# -----------------------------
def generate_dots_video(input_path, out_path):
    cap = open_video(input_path)
    fps = cap.get(5)
    w = int(cap.get(3))
    h = int(cap.get(4))

    vw = write_mp4(out_path, fps, w, h)

    with Pose() as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            img = np.zeros((h, w, 3), dtype=np.uint8)
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if res.pose_landmarks:
                for lm in res.pose_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (x, y), 3, (255,255,255), -1)

            vw.write(img)

    vw.release()
    cap.release()


def generate_skeleton_video(input_path, out_path):
    cap = open_video(input_path)
    fps = cap.get(5)
    w = int(cap.get(3))
    h = int(cap.get(4))

    vw = write_mp4(out_path, fps, w, h)

    with Pose() as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                for lm in res.pose_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x,y), 2, (255,255,255), -1)

            vw.write(frame)

    vw.release()
    cap.release()


# -----------------------------
# JOB
# -----------------------------
def process_job(job):
    mode = job.get("mode")
    input_key = job["input_key"]

    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "input.mp4")
        s3.download_file(AWS_BUCKET, input_key, in_path)

        if mode == "dots":
            raw = os.path.join(td, "raw.mp4")
            out = os.path.join(td, "out.mp4")

            logging.info("DOTS start")
            generate_dots_video(in_path, raw)
            validate_video_file(raw)

            retry(lambda: transcode(raw, out))
            retry(lambda: s3_upload_file(out, job["output_key"], "video/mp4"))

            if not s3_object_exists(job["output_key"]):
                raise RuntimeError("Upload failed")

            return {"ok": True}

        if mode == "skeleton":
            raw = os.path.join(td, "raw.mp4")
            out = os.path.join(td, "out.mp4")

            logging.info("SKELETON start")
            generate_skeleton_video(in_path, raw)
            validate_video_file(raw)

            retry(lambda: transcode(raw, out))
            retry(lambda: s3_upload_file(out, job["output_key"], "video/mp4"))

            if not s3_object_exists(job["output_key"]):
                raise RuntimeError("Upload failed")

            return {"ok": True}

        raise RuntimeError("Unknown mode")


# -----------------------------
# LOOP
# -----------------------------
def main():
    logging.info("Worker started")

    while True:
        keys = list_pending()

        for key in keys:
            job_data = s3_read_json(key)

            mode = job_data.get("mode","").lower()

            # skip report jobs
            if mode in ("report", "report_bundle"):
                continue

            processing_key = claim_job(key)
            if not processing_key:
                continue

            try:
                job = s3_read_json(processing_key)

                logging.info("Processing %s", job.get("job_id"))
                result = process_job(job)

                job["status"] = "finished"
                job["result"] = result
                s3_write_json(processing_key, job)

                move_job(processing_key, FINISHED)

            except Exception as e:
                logging.exception("FAILED %s", e)

                try:
                    job = s3_read_json(processing_key)
                    job["status"] = "failed"
                    job["message"] = str(e)
                    s3_write_json(processing_key, job)
                except:
                    pass

                move_job(processing_key, FAILED)

            break

        time.sleep(1)


if __name__ == "__main__":
    main()