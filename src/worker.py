import os
import io
import json
import time
import math
import tempfile
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import boto3

# Heavy libs
import cv2
import numpy as np
import mediapipe as mp
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

AWS_BUCKET = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

if not AWS_BUCKET:
    raise RuntimeError("Missing AWS_BUCKET (or S3_BUCKET)")

s3 = boto3.client("s3", region_name=AWS_REGION)

PENDING = "jobs/pending/"
PROCESSING = "jobs/processing/"
FINISHED = "jobs/finished/"
FAILED = "jobs/failed/"


# -----------------------------
# S3 helpers
# -----------------------------
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


def list_pending(limit: int = 10) -> List[str]:
    resp = s3.list_objects_v2(Bucket=AWS_BUCKET, Prefix=PENDING, MaxKeys=limit)
    return [x["Key"] for x in resp.get("Contents", []) if x["Key"].endswith(".json")]


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


# -----------------------------
# Dots / Skeleton overlay (simple, stable)
# -----------------------------
mp_pose = mp.solutions.pose

POSE_LANDMARK_IDS = [
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_EYE,
    mp_pose.PoseLandmark.RIGHT_EYE,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
]

SKELETON_EDGES = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
]


def _lm_to_px(lm, w: int, h: int) -> Tuple[int, int]:
    return int(lm.x * w), int(lm.y * h)


def generate_dots_video(input_path: str, out_path: str) -> None:
    cap = open_video(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    vw = write_mp4(out_path, fps, w, h)

    with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_landmarks:
                for pid in POSE_LANDMARK_IDS:
                    lm = res.pose_landmarks.landmark[pid]
                    if lm.visibility < 0.5:
                        continue
                    x, y = _lm_to_px(lm, w, h)
                    cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)

            vw.write(frame)

    vw.release()
    cap.release()


def generate_skeleton_video(input_path: str, out_path: str) -> None:
    cap = open_video(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    vw = write_mp4(out_path, fps, w, h)

    with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
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

    with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
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
            nose = lms[mp_pose.PoseLandmark.NOSE]
            leye = lms[mp_pose.PoseLandmark.LEFT_EYE]
            reye = lms[mp_pose.PoseLandmark.RIGHT_EYE]
            lsh = lms[mp_pose.PoseLandmark.LEFT_SHOULDER]
            rsh = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            lhip = lms[mp_pose.PoseLandmark.LEFT_HIP]
            rhip = lms[mp_pose.PoseLandmark.RIGHT_HIP]
            lank = lms[mp_pose.PoseLandmark.LEFT_ANKLE]
            rank = lms[mp_pose.PoseLandmark.RIGHT_ANKLE]

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
            out_path = os.path.join(td, "dots.mp4")
            generate_dots_video(in_path, out_path)
            s3_upload_file(out_path, out_key, "video/mp4")
            return {"ok": True, "mode": "dots", "output_key": out_key}

        if mode == "skeleton":
            out_key = job["output_key"]
            out_path = os.path.join(td, "skeleton.mp4")
            generate_skeleton_video(in_path, out_path)
            s3_upload_file(out_path, out_key, "video/mp4")
            return {"ok": True, "mode": "skeleton", "output_key": out_key}

        if mode == "report_bundle":
            out_en_key = job["output_en_key"]
            out_th_key = job["output_th_key"]
            debug_key = job.get("output_debug_key", "")

            fi = analyze_first_impression(in_path)

            meta = {"group_id": group_id, "user_name": user_name}
            en_bytes = build_report_en(fi, meta)
            th_bytes = build_report_th(fi, meta)

            s3_put_bytes(out_en_key, en_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            s3_put_bytes(out_th_key, th_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

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
        keys = list_pending(limit=10)
        if not keys:
            time.sleep(poll_seconds)
            continue

        for pending_key in keys:
            processing_key = claim_job(pending_key)
            if not processing_key:
                continue

            try:
                job = s3_read_json(processing_key)
                job["status"] = "processing"
                s3_write_json(processing_key, job)

                logging.info("Processing job %s mode=%s", job.get("job_id"), job.get("mode"))
                result = process_job(job)

                job["status"] = "finished"
                job["result"] = result
                s3_write_json(processing_key, job)
                move_job(processing_key, FINISHED)

                logging.info("Finished job %s", job.get("job_id"))

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
