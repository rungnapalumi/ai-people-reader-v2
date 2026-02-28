import os
import io
import json

from dotenv import load_dotenv
load_dotenv()
import time
import math
import tempfile
import logging
import shutil
import subprocess
import smtplib
import re
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from email.mime.text import MIMEText

import boto3

# Heavy libs
import cv2
import numpy as np
try:
    # Try new MediaPipe API first (0.10.8+)
    from mediapipe.python.solutions import pose as mp_pose_module
    from mediapipe.python.solutions.pose import Pose, PoseLandmark
except ImportError:
    # Fall back to old API
    import mediapipe as mp
    mp_pose_module = mp.solutions.pose
    Pose = mp_pose_module.Pose
    PoseLandmark = mp_pose_module.PoseLandmark

from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

AWS_BUCKET = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

if not AWS_BUCKET:
    raise RuntimeError("Missing AWS_BUCKET (or S3_BUCKET)")

s3 = boto3.client("s3", region_name=AWS_REGION)
SES_REGION = os.getenv("SES_REGION", AWS_REGION).strip() or AWS_REGION
SES_FROM_EMAIL = (os.getenv("SES_FROM_EMAIL") or "").strip()
SMTP_HOST = (os.getenv("SMTP_HOST") or "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USERNAME = (os.getenv("SMTP_USERNAME") or "").strip()
SMTP_PASSWORD = (os.getenv("SMTP_PASSWORD") or "").strip()
SMTP_USE_TLS = str(os.getenv("SMTP_USE_TLS", "false")).strip().lower() in ("1", "true", "yes", "y")
SMTP_USE_SSL = str(os.getenv("SMTP_USE_SSL", "true")).strip().lower() in ("1", "true", "yes", "y")
SMTP_FROM_EMAIL = (os.getenv("SMTP_FROM_EMAIL") or SES_FROM_EMAIL).strip()

PENDING = "jobs/pending/"
PROCESSING = "jobs/processing/"
FINISHED = "jobs/finished/"
FAILED = "jobs/failed/"


# -----------------------------
# S3 helpers
# -----------------------------
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def s3_read_json(key: str) -> Dict[str, Any]:
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    raw = obj["Body"].read().decode("utf-8")
    return json.loads(raw)


def s3_write_json(key: str, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json; charset=utf-8",
    )


def s3_download_to_file(key: str, path: str) -> None:
    s3.download_file(AWS_BUCKET, key, path)


def s3_upload_file(path: str, key: str, content_type: str) -> None:
    s3.upload_file(path, AWS_BUCKET, key, ExtraArgs={"ContentType": content_type})


def s3_put_bytes(key: str, data: bytes, content_type: str) -> None:
    s3.put_object(Bucket=AWS_BUCKET, Key=key, Body=data, ContentType=content_type)


def presigned_get_url(key: str, expires: int = 86400) -> str:
    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": AWS_BUCKET, "Key": key},
        ExpiresIn=max(300, int(expires)),
    )


def is_valid_email(value: str) -> bool:
    return bool(EMAIL_RE.match(str(value or "").strip()))


def parse_email_list(value: str) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for token in re.split(r"[,\s;]+", str(value or "").strip()):
        email = token.strip()
        if not email:
            continue
        key = email.lower()
        if key in seen:
            continue
        if is_valid_email(email):
            seen.add(key)
            out.append(email)
    return out


def _send_email_via_ses(to_email: str, subject: str, body: str) -> Tuple[bool, str]:
    if not SES_FROM_EMAIL:
        return False, "ses_from_not_configured"
    try:
        ses = boto3.client("ses", region_name=SES_REGION)
        ses.send_email(
            Source=SES_FROM_EMAIL,
            Destination={"ToAddresses": [to_email]},
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {"Text": {"Data": body, "Charset": "UTF-8"}},
            },
        )
        return True, "sent_via_ses"
    except Exception as e:
        return False, f"ses_failed:{e}"


def _send_email_via_smtp(to_email: str, subject: str, body: str) -> Tuple[bool, str]:
    if not (SMTP_HOST and SMTP_USERNAME and SMTP_PASSWORD and SMTP_FROM_EMAIL):
        return False, "smtp_not_configured"
    try:
        msg = MIMEText(body, _charset="utf-8")
        msg["Subject"] = subject
        msg["From"] = SMTP_FROM_EMAIL
        msg["To"] = to_email
        if SMTP_USE_SSL:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=20) as server:
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.sendmail(SMTP_FROM_EMAIL, [to_email], msg.as_string())
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
                if SMTP_USE_TLS:
                    server.starttls()
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.sendmail(SMTP_FROM_EMAIL, [to_email], msg.as_string())
        return True, "sent_via_smtp"
    except Exception as e:
        return False, f"smtp_failed:{e}"


def send_mode_ready_email(job: Dict[str, Any], result: Dict[str, Any]) -> Tuple[bool, str]:
    mode = str(result.get("mode") or "").strip().lower()
    if mode not in ("dots", "skeleton"):
        return False, "skip_non_video_mode"
    recipients = parse_email_list(str(job.get("notify_email") or ""))
    if not recipients:
        return False, "skip_no_valid_recipients"
    output_key = str(result.get("output_key") or "").strip()
    if not output_key:
        return False, "skip_missing_output_key"
    try:
        url = presigned_get_url(output_key, expires=86400)
    except Exception as e:
        return False, f"skip_presign_failed:{e}"

    group_id = str(job.get("group_id") or "").strip()
    job_id = str(job.get("job_id") or "").strip()
    mode_title = "Dots Video" if mode == "dots" else "Skeleton Video"
    subject = f"AI People Reader - {mode_title} Ready ({group_id or job_id})"
    body = (
        f"Your {mode_title.lower()} is ready.\n\n"
        f"Group ID: {group_id}\n"
        f"Job ID: {job_id}\n"
        f"Download link (valid for 24 hours):\n{url}\n"
    )

    statuses: List[str] = []
    sent_any = False
    for to_email in recipients:
        sent, status = _send_email_via_ses(to_email, subject, body)
        if not sent:
            sent, status = _send_email_via_smtp(to_email, subject, body)
        statuses.append(f"{to_email}:{status}")
        sent_any = sent_any or sent
    return sent_any, " | ".join(statuses)


def list_pending(limit: int = 200) -> List[str]:
    """
    List more pending keys to avoid starvation when queue is crowded with
    report jobs (handled by report_worker). This worker then picks non-report
    jobs like skeleton/dots from the same pending queue.
    """
    keys: List[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=PENDING):
        for item in page.get("Contents", []):
            key = str(item.get("Key") or "")
            if key.endswith(".json"):
                keys.append(key)
                if len(keys) >= limit:
                    return keys
    return keys


def claim_job(pending_key: str) -> Optional[str]:
    """
    Atomic-ish claim: copy pending -> processing then delete pending.
    If copy/delete fails (race), return None.
    """
    name = pending_key.split("/")[-1]
    processing_key = PROCESSING + name
    try:
        s3.copy_object(
            Bucket=AWS_BUCKET,
            CopySource={"Bucket": AWS_BUCKET, "Key": pending_key},
            Key=processing_key,
            ContentType="application/json; charset=utf-8",
            MetadataDirective="REPLACE",
        )
        s3.delete_object(Bucket=AWS_BUCKET, Key=pending_key)
        return processing_key
    except Exception:
        return None


def move_job(current_key: str, dest_prefix: str) -> str:
    name = current_key.split("/")[-1]
    dest_key = dest_prefix + name
    s3.copy_object(
        Bucket=AWS_BUCKET,
        CopySource={"Bucket": AWS_BUCKET, "Key": current_key},
        Key=dest_key,
        ContentType="application/json; charset=utf-8",
        MetadataDirective="REPLACE",
    )
    s3.delete_object(Bucket=AWS_BUCKET, Key=current_key)
    return dest_key


# -----------------------------
# Video processing
# -----------------------------
def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    return cap


def write_mp4(out_path: str, fps: float, w: int, h: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps if fps > 0 else 25.0, (w, h))
    if not vw.isOpened():
        raise RuntimeError("Cannot open VideoWriter (mp4v)")
    return vw


def validate_video_file(path: str) -> None:
    """Fail fast if output video is empty/corrupt before upload."""
    if not os.path.exists(path):
        raise RuntimeError(f"Output video file does not exist: {path}")
    size = os.path.getsize(path)
    if size <= 0:
        raise RuntimeError(f"Output video file is empty: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open output video: {path}")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    duration = (frames / fps) if fps > 0 else 0.0
    cap.release()
    if frames <= 0:
        raise RuntimeError(f"Output video has no frames: {path}")
    if fps <= 0:
        raise RuntimeError(f"Output video has invalid fps: {path}")
    if duration <= 0.2:
        raise RuntimeError(f"Output video duration too short ({duration:.3f}s): {path}")


def _run_ffmpeg_transcode(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg transcode failed: {proc.stderr[-400:]}")


def transcode_dots_mp4(input_path: str, output_path: str) -> None:
    """Ultra-compatible dots profile for browser playback from shared links."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg not found. Install ffmpeg to enable browser-compatible MP4 output.")

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        input_path,
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-vf",
        (
            "scale='if(gt(iw,854),854,iw)':'if(gt(ih,480),480,ih)':"
            "force_original_aspect_ratio=decrease,"
            "scale=trunc(iw/2)*2:trunc(ih/2)*2,"
            "fps=24,format=yuv420p"
        ),
        "-c:v",
        "libx264",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        "-x264-params",
        "bframes=0:ref=1:cabac=0:keyint=48:min-keyint=48:scenecut=0",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-maxrate",
        "1200k",
        "-bufsize",
        "2400k",
        "-g",
        "48",
        "-keyint_min",
        "48",
        "-sc_threshold",
        "0",
        "-movflags",
        "+faststart",
        "-vsync",
        "cfr",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-shortest",
        "-r",
        "24",
        output_path,
    ]
    _run_ffmpeg_transcode(cmd)
    validate_video_file(output_path)


def transcode_skeleton_mp4(input_path: str, output_path: str) -> None:
    """Ultra-compatible skeleton profile for browser playback from links."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg not found. Install ffmpeg to enable browser-compatible MP4 output.")

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        input_path,
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-vf",
        (
            "scale='if(gt(iw,854),854,iw)':'if(gt(ih,480),480,ih)':"
            "force_original_aspect_ratio=decrease,"
            "scale=trunc(iw/2)*2:trunc(ih/2)*2,"
            "fps=24,format=yuv420p"
        ),
        "-c:v",
        "libx264",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        "-x264-params",
        "bframes=0:ref=1:cabac=0:keyint=48:min-keyint=48:scenecut=0",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-maxrate",
        "1200k",
        "-bufsize",
        "2400k",
        "-g",
        "48",
        "-keyint_min",
        "48",
        "-sc_threshold",
        "0",
        "-movflags",
        "+faststart",
        "-vsync",
        "cfr",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-shortest",
        "-r",
        "24",
        output_path,
    ]
    _run_ffmpeg_transcode(cmd)
    validate_video_file(output_path)


# -----------------------------
# Dots / Skeleton overlay (simple, stable)
# -----------------------------
POSE_LANDMARK_IDS = [
    PoseLandmark.NOSE,
    PoseLandmark.LEFT_EYE,
    PoseLandmark.RIGHT_EYE,
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


def _lm_to_px(lm, w: int, h: int) -> Tuple[int, int]:
    return int(lm.x * w), int(lm.y * h)


def generate_dots_video(input_path: str, out_path: str) -> None:
    """Generate dot motion video with white dots on black background - matches original implementation"""
    cap = open_video(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    vw = write_mp4(out_path, fps, w, h)
    
    dot_size = 5  # 5 pixels as per user requirement

    with Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Create black background (key difference from old code)
            output = np.zeros((h, w, 3), dtype=np.uint8)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_landmarks:
                # Draw ALL landmarks (not just subset) - matching original code
                for lm in res.pose_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    
                    # Ensure coordinates are within bounds
                    if 0 <= cx < w and 0 <= cy < h:
                        # Draw white dot on black background
                        cv2.circle(
                            output,
                            (cx, cy),
                            radius=dot_size,
                            color=(255, 255, 255),  # white in BGR format
                            thickness=-1
                        )

            vw.write(output)  # Write black background with white dots

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

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark

                # draw edges
                for a, b in SKELETON_EDGES:
                    la, lb = lms[a], lms[b]
                    if la.visibility < 0.5 or lb.visibility < 0.5:
                        continue
                    xa, ya = _lm_to_px(la, w, h)
                    xb, yb = _lm_to_px(lb, w, h)
                    cv2.line(frame, (xa, ya), (xb, yb), (0, 255, 0), 3)

                # joints
                for pid in POSE_LANDMARK_IDS:
                    lm = lms[pid]
                    if lm.visibility < 0.5:
                        continue
                    x, y = _lm_to_px(lm, w, h)
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            vw.write(frame)

    vw.release()
    cap.release()


# -----------------------------
# First Impression (REAL from video via MediaPipe Pose)
# -----------------------------
@dataclass
class FirstImpressionResult:
    eye_contact_pct: float
    upright_pct: float
    stance_stability: float
    notes: Dict[str, str]


def analyze_first_impression(input_path: str, sample_every_n: int = 3, max_frames: int = 300) -> FirstImpressionResult:
    """
    Simple + stable:
    - Eye Contact proxy: nose centered between eyes & eyes visible
    - Uprightness: torso vector close to vertical
    - Stance stability: ankles distance variance (lower variance = more stable)
    """
    cap = open_video(input_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total = 0
    eye_ok = 0
    upright_ok = 0
    ankle_dist: List[float] = []

    with Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i % sample_every_n != 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if not res.pose_landmarks:
                continue

            lms = res.pose_landmarks.landmark
            nose = lms[PoseLandmark.NOSE]
            leye = lms[PoseLandmark.LEFT_EYE]
            reye = lms[PoseLandmark.RIGHT_EYE]
            lsh = lms[PoseLandmark.LEFT_SHOULDER]
            rsh = lms[PoseLandmark.RIGHT_SHOULDER]
            lhip = lms[PoseLandmark.LEFT_HIP]
            rhip = lms[PoseLandmark.RIGHT_HIP]
            lank = lms[PoseLandmark.LEFT_ANKLE]
            rank = lms[PoseLandmark.RIGHT_ANKLE]

            # count only if main landmarks are visible enough
            if min(nose.visibility, leye.visibility, reye.visibility, lsh.visibility, rsh.visibility, lhip.visibility, rhip.visibility) < 0.5:
                continue

            total += 1

            # Eye contact proxy: nose x between eye x
            minx = min(leye.x, reye.x)
            maxx = max(leye.x, reye.x)
            if (minx <= nose.x <= maxx):
                eye_ok += 1

            # Uprightness: torso vector midHip->midShoulder angle to vertical
            mid_sh = np.array([(lsh.x + rsh.x) / 2.0, (lsh.y + rsh.y) / 2.0])
            mid_hip = np.array([(lhip.x + rhip.x) / 2.0, (lhip.y + rhip.y) / 2.0])
            v = mid_sh - mid_hip  # up direction (y smaller is up in image coords, but angle works)
            # angle to vertical axis (0, -1) in image space
            vert = np.array([0.0, -1.0])
            v_norm = np.linalg.norm(v) + 1e-9
            cosang = float(np.dot(v / v_norm, vert))
            ang = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
            if ang <= 15.0:
                upright_ok += 1

            # stance: ankle distance (normalized)
            if min(lank.visibility, rank.visibility) >= 0.5:
                dx = (lank.x - rank.x)
                dy = (lank.y - rank.y)
                ankle_dist.append(math.sqrt(dx*dx + dy*dy))

            if total >= max_frames:
                break

    cap.release()

    if total == 0:
        return FirstImpressionResult(
            eye_contact_pct=0.0,
            upright_pct=0.0,
            stance_stability=0.0,
            notes={"error": "insufficient_pose_frames"},
        )

    eye_pct = 100.0 * (eye_ok / total)
    upright_pct = 100.0 * (upright_ok / total)

    if len(ankle_dist) >= 10:
        std = float(np.std(np.array(ankle_dist)))
        # convert std to stability score (0..100) : lower std => higher score
        stability = max(0.0, min(100.0, 100.0 * (1.0 - (std / 0.20))))  # heuristic
    else:
        stability = 0.0

    notes = {}
    return FirstImpressionResult(
        eye_contact_pct=eye_pct,
        upright_pct=upright_pct,
        stance_stability=stability,
        notes=notes,
    )


# -----------------------------
# DOCX reports (EN/TH)
# -----------------------------
def _add_title(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(16)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def _add_h2(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(13)


def _fmt_band(pct: float) -> str:
    if pct >= 70:
        return "Strong"
    if pct >= 40:
        return "Moderate"
    return "Needs improvement"


def _fmt_band_th(pct: float) -> str:
    if pct >= 70:
        return "ดีมาก"
    if pct >= 40:
        return "ปานกลาง"
    return "ควรพัฒนา"


def build_report_en(fi: FirstImpressionResult, meta: Dict[str, Any]) -> bytes:
    doc = Document()
    _add_title(doc, "AI People Reader — Report (EN)")

    doc.add_paragraph(f"Group: {meta.get('group_id','')}")
    doc.add_paragraph(f"User: {meta.get('user_name','')}")
    doc.add_paragraph(" ")

    _add_h2(doc, "First Impression (from uploaded video)")
    doc.add_paragraph(f"Eye Contact: {fi.eye_contact_pct:.1f}%  — {_fmt_band(fi.eye_contact_pct)}")
    doc.add_paragraph(f"Uprightness (Posture & Upper-Body Alignment): {fi.upright_pct:.1f}%  — {_fmt_band(fi.upright_pct)}")
    doc.add_paragraph(f"Stance (Lower-Body Stability & Grounding): {fi.stance_stability:.1f}/100  — {_fmt_band(fi.stance_stability)}")

    if fi.notes:
        _add_h2(doc, "Notes")
        for k, v in fi.notes.items():
            doc.add_paragraph(f"- {k}: {v}")

    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()


def build_report_th(fi: FirstImpressionResult, meta: Dict[str, Any]) -> bytes:
    doc = Document()
    _add_title(doc, "AI People Reader — รายงาน (TH)")

    doc.add_paragraph(f"Group: {meta.get('group_id','')}")
    doc.add_paragraph(f"ผู้ใช้: {meta.get('user_name','')}")
    doc.add_paragraph(" ")

    _add_h2(doc, "First Impression (วิเคราะห์จากวิดีโอจริงที่อัปโหลด)")
    doc.add_paragraph(f"Eye Contact: {fi.eye_contact_pct:.1f}%  — {_fmt_band_th(fi.eye_contact_pct)}")
    doc.add_paragraph(f"Uprightness (แนวลำตัว/การวางช่วงบน): {fi.upright_pct:.1f}%  — {_fmt_band_th(fi.upright_pct)}")
    doc.add_paragraph(f"Stance (ความมั่นคงช่วงล่าง/การยืน): {fi.stance_stability:.1f}/100  — {_fmt_band_th(fi.stance_stability)}")

    if fi.notes:
        _add_h2(doc, "หมายเหตุ")
        for k, v in fi.notes.items():
            doc.add_paragraph(f"- {k}: {v}")

    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()


# -----------------------------
# Job processor
# -----------------------------
def process_job(job: Dict[str, Any]) -> Dict[str, Any]:
    mode = (job.get("mode") or "").strip()
    group_id = job.get("group_id", "")
    user_name = job.get("user_name", "")
    input_key = job.get("input_key", "")

    if not input_key:
        raise RuntimeError("Missing input_key")

    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "input.mp4")
        s3_download_to_file(input_key, in_path)

        if mode == "dots":
            out_key = job["output_key"]
            raw_path = os.path.join(td, "dots_raw.mp4")
            out_path = os.path.join(td, "dots.mp4")
            generate_dots_video(in_path, raw_path)
            validate_video_file(raw_path)
            transcode_dots_mp4(raw_path, out_path)
            s3_upload_file(out_path, out_key, "video/mp4")
            return {"ok": True, "mode": "dots", "output_key": out_key}

        if mode == "skeleton":
            out_key = job["output_key"]
            raw_path = os.path.join(td, "skeleton_raw.mp4")
            out_path = os.path.join(td, "skeleton.mp4")
            generate_skeleton_video(in_path, raw_path)
            validate_video_file(raw_path)
            transcode_skeleton_mp4(raw_path, out_path)
            s3_upload_file(out_path, out_key, "video/mp4")
            return {"ok": True, "mode": "skeleton", "output_key": out_key}

        if mode == "report_bundle":
            out_en_key = job["output_en_key"]
            out_th_key = job["output_th_key"]
            debug_key = job.get("output_debug_key", "")

            logging.info("Report bundle: EN=%s TH=%s", out_en_key, out_th_key)

            fi = analyze_first_impression(in_path)

            logging.info("First impression: eye_contact=%.1f%% upright=%.1f%% stance=%.1f", 
                        fi.eye_contact_pct, fi.upright_pct, fi.stance_stability)

            meta = {"group_id": group_id, "user_name": user_name}
            en_bytes = build_report_en(fi, meta)
            th_bytes = build_report_th(fi, meta)

            logging.info("Built reports: EN=%d bytes TH=%d bytes", len(en_bytes), len(th_bytes))

            s3_put_bytes(out_en_key, en_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            s3_put_bytes(out_th_key, th_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

            logging.info("Uploaded reports to S3")

            if debug_key:
                dbg = {
                    "group_id": group_id,
                    "user_name": user_name,
                    "first_impression": {
                        "eye_contact_pct": fi.eye_contact_pct,
                        "upright_pct": fi.upright_pct,
                        "stance_stability": fi.stance_stability,
                        "notes": fi.notes,
                    },
                }
                s3_put_bytes(debug_key, json.dumps(dbg, ensure_ascii=False, indent=2).encode("utf-8"), "application/json; charset=utf-8")

            return {
                "ok": True,
                "mode": "report_bundle",
                "output_en_key": out_en_key,
                "output_th_key": out_th_key,
                "debug_key": debug_key,
                "first_impression": {
                    "eye_contact_pct": fi.eye_contact_pct,
                    "upright_pct": fi.upright_pct,
                    "stance_stability": fi.stance_stability,
                    "notes": fi.notes,
                },
            }

        raise RuntimeError(f"Unknown mode: {mode}")


def main_loop(poll_seconds: int = 3) -> None:
    logging.info("Worker started. Bucket=%s region=%s", AWS_BUCKET, AWS_REGION)

    while True:
        # Scan deeper in pending queue so skeleton/dots are not starved
        # by many report jobs.
        keys = list_pending(limit=200)
        if not keys:
            time.sleep(poll_seconds)
            continue

        for pending_key in keys:
            # Peek pending job mode first, so this worker does not steal report jobs.
            try:
                pending_job = s3_read_json(pending_key)
                pending_mode = (pending_job.get("mode") or "").strip().lower()
                if pending_mode in ("report", "report_th_en", "report_generator"):
                    logging.info("Skipping pending report job %s mode=%s (reserved for report_worker)", pending_job.get("job_id"), pending_mode)
                    continue
            except Exception as e:
                logging.info("Skip pending key %s while peeking mode: %s", pending_key, e)
                continue

            processing_key = claim_job(pending_key)
            if not processing_key:
                continue

            try:
                job = s3_read_json(processing_key)
                mode = (job.get("mode") or "").strip()
                
                # Extra safety: keep report jobs for report_worker if one slips through.
                if mode in ("report", "report_th_en", "report_generator"):
                    logging.info("Skipping report mode job %s (handled by report_worker), moving back to pending", job.get("job_id"))
                    move_job(processing_key, PENDING)
                    continue
                
                job["status"] = "processing"
                s3_write_json(processing_key, job)

                logging.info("Processing job %s mode=%s", job.get("job_id"), mode)
                result = process_job(job)
                email_sent, email_status = send_mode_ready_email(job, result)

                job["status"] = "finished"
                job["result"] = result
                job["notification"] = {
                    "notify_email": str(job.get("notify_email") or "").strip(),
                    "sent": bool(email_sent),
                    "status": email_status,
                }
                s3_write_json(processing_key, job)
                move_job(processing_key, FINISHED)

                logging.info(
                    "Finished job %s mode=%s email_sent=%s email_status=%s",
                    job.get("job_id"),
                    mode,
                    email_sent,
                    email_status,
                )

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

        time.sleep(0.2)


if __name__ == "__main__":
    main_loop()
