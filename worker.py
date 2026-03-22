import os
import json
import time
import tempfile
import logging
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

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

# Pending listing: S3 returns keys in lexicographic order. If the queue is large, dots jobs can
# sit beyond the first N keys; a small cap causes "skeleton works but dots never runs".
WORKER_PENDING_MAX_SCAN = max(800, int(os.getenv("WORKER_PENDING_MAX_SCAN", "25000") or "25000"))


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


def list_pending(limit: int = 200, max_scan: Optional[int] = None) -> List[str]:
    """
    Dots jobs first, then skeleton/other. Report jobs are skipped.

    Scan skips report JSONs while searching so dots/skeleton are found even when many report
    jobs sort earlier lexicographically under jobs/pending/.
    """
    cap = max(max_scan if max_scan is not None else WORKER_PENDING_MAX_SCAN, 1)
    dots_keys: List[str] = []
    other_keys: List[str] = []
    scanned = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=PENDING):
        for item in page.get("Contents", []):
            k = str(item.get("Key") or "")
            if not k.endswith(".json"):
                continue
            scanned += 1
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
            if len(dots_keys) + len(other_keys) >= limit:
                return (dots_keys + other_keys)[:limit]
            if scanned >= cap:
                break
        if scanned >= cap:
            break

    merged = (dots_keys + other_keys)[:limit]
    if not merged and scanned >= cap:
        logging.warning(
            "list_pending: read %s pending JSON keys (cap=%s), found no dots/skeleton — increase WORKER_PENDING_MAX_SCAN if jobs exist.",
            scanned,
            cap,
        )
    elif not dots_keys and other_keys:
        logging.info(
            "list_pending: no dots in this batch (%s other video keys of %s JSON scanned)",
            len(other_keys),
            scanned,
        )
    return merged


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
    except Exception as exc:
        logging.warning("claim_job failed pending_key=%s err=%s", key, exc)
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
    if not cap.isOpened():
        raise RuntimeError("Cannot open video for validation")
    frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps_meta = float(cap.get(cv2.CAP_PROP_FPS) or 0)

    # OpenCV/mp4v often reports frame_count=0 and/or fps=0 for sparse synthetic video
    # (dots-on-black) while skeleton-on-camera muxes fine — count frames if metadata lies.
    if frames_meta <= 0:
        n = 0
        while n < 2_000_000:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
        cap.release()
        if n <= 0:
            raise RuntimeError("No frames")
        logging.info("validate_video_file: counted %s frames (metadata count was 0) %s", n, path)
        return

    cap.release()
    if fps_meta <= 0:
        logging.warning(
            "validate_video_file: metadata fps missing/invalid (%.4f), accepting file with %s frames %s",
            fps_meta,
            frames_meta,
            path,
        )


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
    from mediapipe.python.solutions.pose import Pose, PoseLandmark
except ImportError:
    import mediapipe as mp

    Pose = mp.solutions.pose.Pose
    PoseLandmark = mp.solutions.pose.PoseLandmark


# Body-only skeleton (no face): shoulders–ankles only — no nose/eyes/ears.
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

# 1.0 = draw at MediaPipe landmark positions (follows the real body). Lower values pull toward
# frame center and look artificially “shrunk”; use ~0.92–0.98 only if you want a slight inset.
SKELETON_POSE_SPREAD = 1.0
# Below ~0.5, arms/hands often disappear from the overlay; too high looks "broken" on motion.
SKELETON_VISIBILITY_MIN = 0.35
# Longest side (px) for MediaPipe input; 0 = full frame (slow on 4K). 1280 keeps 1:1 mapping to output.
SKELETON_POSE_MAX_SIDE = int(os.getenv("SKELETON_POSE_MAX_SIDE", "1280") or "1280")
# Neon style: cyan #00FFFF glow (BGR) + white core — body joints/lines only (see SKELETON_* above).
SKELETON_CYAN_BGR = (255, 255, 0)
SKELETON_WHITE_BGR = (255, 255, 255)
SKELETON_GLOW_LINE_THICK = 11
SKELETON_CORE_LINE_THICK = 2
SKELETON_GLOW_JOINT_RADIUS = 7
SKELETON_CORE_JOINT_RADIUS = 3
SKELETON_GLOW_BLUR_SIGMA = 4.0
SKELETON_GLOW_ALPHA = 0.52


def _lm_to_px_spread(lm: Any, w: int, h: int, spread: float) -> Tuple[int, int]:
    nx = 0.5 + (float(lm.x) - 0.5) * spread
    ny = 0.5 + (float(lm.y) - 0.5) * spread
    return int(nx * w), int(ny * h)


def _frame_for_pose(frame: np.ndarray, max_width: int = 640) -> np.ndarray:
    """Downscale for MediaPipe only; normalized coords still map to full frame when drawing."""
    if frame is None:
        return frame
    fh, fw = frame.shape[:2]
    if fw <= max_width or max_width <= 0:
        return frame
    new_h = max(1, int(round(fh * (float(max_width) / float(fw)))))
    return cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_AREA)


def _prepare_pose_input_uniform(
    frame_bgr: np.ndarray, max_side: int
) -> Tuple[np.ndarray, int, int]:
    """
    Resize frame uniformly for pose (same aspect as output). Returns (rgb, pose_w, pose_h).
    Landmarks from MediaPipe are normalized to (pose_w, pose_h); map to output with
    x_out = lm.x * out_w (uniform scale preserves 1:1 alignment with the video frame).
    """
    if frame_bgr is None:
        raise ValueError("empty frame")
    fh, fw = frame_bgr.shape[:2]
    if max_side <= 0 or max(fw, fh) <= max_side:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return rgb, fw, fh
    scale = max_side / float(max(fw, fh))
    pw = max(1, int(round(fw * scale)))
    ph = max(1, int(round(fh * scale)))
    small = cv2.resize(frame_bgr, (pw, ph), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB), pw, ph


def _lm_norm_to_canvas(
    lm: Any, out_w: int, out_h: int, spread: float
) -> Tuple[int, int]:
    """Normalized landmark [0,1] on pose image → pixel on output canvas (same aspect)."""
    nx = 0.5 + (float(lm.x) - 0.5) * spread
    ny = 0.5 + (float(lm.y) - 0.5) * spread
    return int(nx * out_w), int(ny * out_h)


def generate_dots_video(input_path: str, out_path: str) -> None:
    cap = open_video(input_path)
    ok0, frame0 = cap.read()
    if not ok0 or frame0 is None:
        cap.release()
        raise RuntimeError("Cannot read first frame for dots")
    h, w = frame0.shape[:2]
    if w <= 0 or h <= 0:
        cap.release()
        raise RuntimeError("Invalid frame size for dots")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0

    vw = write_mp4(out_path, fps, w, h)

    def draw_dots(frame: np.ndarray) -> None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            for lm in res.pose_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(canvas, (x, y), 3, (255, 255, 255), -1)
        vw.write(canvas)

    with Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
        draw_dots(frame0)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            draw_dots(frame)

    vw.release()
    cap.release()


def generate_skeleton_video(input_path: str, out_path: str) -> None:
    """
    Body-only (no face): cyan glow + thin white bones and joint highlights at MediaPipe positions.
    Spread=1.0 follows the real figure; lower SKELETON_POSE_SPREAD insets the whole pose.
    """
    cap = open_video(input_path)
    ok0, frame0 = cap.read()
    if not ok0 or frame0 is None:
        cap.release()
        raise RuntimeError("Cannot read first frame for skeleton")
    h0, w0 = frame0.shape[:2]
    if w0 <= 0 or h0 <= 0:
        cap.release()
        raise RuntimeError("Invalid frame size for skeleton")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    vw = write_mp4(out_path, fps, w0, h0)

    spread = SKELETON_POSE_SPREAD
    vis_min = SKELETON_VISIBILITY_MIN
    cyan = SKELETON_CYAN_BGR
    white = SKELETON_WHITE_BGR
    glow_line = SKELETON_GLOW_LINE_THICK
    core_line = SKELETON_CORE_LINE_THICK
    glow_r = SKELETON_GLOW_JOINT_RADIUS
    core_r = SKELETON_CORE_JOINT_RADIUS
    glow_sigma = SKELETON_GLOW_BLUR_SIGMA
    glow_alpha = SKELETON_GLOW_ALPHA
    scale = 2
    w2, h2 = w0 * scale, h0 * scale
    kblur = max(3, int(round(glow_sigma * 4)) | 1)
    # Reusing landmarks on skipped frames makes the wireframe "float" or shrink vs the body when
    # the person moves (clapping, stepping). Run pose every frame.
    pose_max_side = max(0, SKELETON_POSE_MAX_SIDE)
    model_cx = max(0, min(2, int(os.getenv("SKELETON_MODEL_COMPLEXITY", "2") or "2")))
    last_landmarks: Optional[Any] = None

    def draw_skeleton_overlay(base: np.ndarray) -> np.ndarray:
        if not last_landmarks:
            return base
        lms = last_landmarks
        hi = cv2.resize(base, (w2, h2), interpolation=cv2.INTER_LINEAR)
        glow_layer = np.zeros_like(hi, dtype=np.uint8)

        def edge_segment(a: int, b: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
            la, lb = lms[a], lms[b]
            if float(la.visibility) < vis_min or float(lb.visibility) < vis_min:
                return None
            pa = _lm_norm_to_canvas(la, w2, h2, spread)
            pb = _lm_norm_to_canvas(lb, w2, h2, spread)
            return pa, pb

        for a, b in SKELETON_EDGES:
            seg = edge_segment(a, b)
            if not seg:
                continue
            (xa, ya), (xb, yb) = seg
            cv2.line(glow_layer, (xa, ya), (xb, yb), cyan, glow_line, cv2.LINE_AA)

        for pid in SKELETON_JOINT_IDS:
            lm = lms[pid]
            if float(lm.visibility) < vis_min:
                continue
            x, y = _lm_norm_to_canvas(lm, w2, h2, spread)
            cv2.circle(glow_layer, (x, y), glow_r, cyan, -1, lineType=cv2.LINE_AA)

        glow_soft = cv2.GaussianBlur(glow_layer, (kblur, kblur), glow_sigma)
        out = cv2.addWeighted(hi, 1.0, glow_soft, glow_alpha, 0.0)

        for a, b in SKELETON_EDGES:
            seg = edge_segment(a, b)
            if not seg:
                continue
            (xa, ya), (xb, yb) = seg
            cv2.line(out, (xa, ya), (xb, yb), white, core_line, cv2.LINE_AA)

        for pid in SKELETON_JOINT_IDS:
            lm = lms[pid]
            if float(lm.visibility) < vis_min:
                continue
            x, y = _lm_norm_to_canvas(lm, w2, h2, spread)
            cv2.circle(out, (x, y), core_r, white, -1, lineType=cv2.LINE_AA)

        return cv2.resize(out, (w0, h0), interpolation=cv2.INTER_LINEAR)

    with Pose(
        static_image_mode=False,
        model_complexity=model_cx,
        enable_segmentation=False,
    ) as pose:
        def process_frame(frame: np.ndarray) -> None:
            nonlocal last_landmarks
            fh, fw = frame.shape[:2]
            if fw != w0 or fh != h0:
                frame = cv2.resize(frame, (w0, h0), interpolation=cv2.INTER_LINEAR)

            rgb, _pw, _ph = _prepare_pose_input_uniform(frame, pose_max_side)
            res = pose.process(rgb)
            if res.pose_landmarks:
                last_landmarks = res.pose_landmarks.landmark

            vw.write(draw_skeleton_overlay(frame))

        process_frame(frame0)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            process_frame(frame)

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
