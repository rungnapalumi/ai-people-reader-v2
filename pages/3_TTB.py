# pages/2_Submit_Job.py — Video Analysis (วิเคราะห์วิดีโอ)
# Upload once (shared key) -> get downloads:
#   1) Dots video
#   2) Skeleton video
#   3) English report (DOCX)
#   4) Thai report (DOCX)
#
# Uses LEGACY queue:
#   jobs/pending/<job_id>.json
#
# Output paths (fixed):
#   jobs/output/groups/<group_id>/dots.mp4
#   jobs/output/groups/<group_id>/skeleton.mp4
#   jobs/output/groups/<group_id>/report_en.docx
#   jobs/output/groups/<group_id>/report_th.docx
#
# Key point:
# - DOCX only (no PDF)
# - Can paste any group_id to retrieve outputs (no session loss problem)

import os
import json
import uuid
import re
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import streamlit as st
import boto3


# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="Video Analysis (วิเคราะห์วิดีโอ)", layout="wide")


# -------------------------
# Env / S3
# -------------------------
AWS_BUCKET = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
SES_REGION = os.getenv("SES_REGION", AWS_REGION)
SES_FROM_EMAIL = os.getenv("SES_FROM_EMAIL", "").strip()

if not AWS_BUCKET:
    st.error("Missing AWS_BUCKET (or S3_BUCKET) environment variable in Render.")
    st.stop()

s3 = boto3.client("s3", region_name=AWS_REGION)
ses = boto3.client("ses", region_name=SES_REGION)

JOBS_PENDING_PREFIX = "jobs/pending/"
JOBS_PROCESSING_PREFIX = "jobs/processing/"
JOBS_FINISHED_PREFIX = "jobs/finished/"
JOBS_FAILED_PREFIX = "jobs/failed/"

JOBS_OUTPUT_PREFIX = "jobs/output/"
JOBS_GROUP_PREFIX = "jobs/groups/"


# -------------------------
# Helpers
# -------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_job_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:5]
    return f"{ts}__{rand}"


def new_group_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:6]
    return f"{ts}_{rand}"


def guess_content_type(filename: str) -> str:
    fn = (filename or "").lower()
    if fn.endswith(".mp4"):
        return "video/mp4"
    if fn.endswith(".mov"):
        return "video/quicktime"
    if fn.endswith(".m4v"):
        return "video/x-m4v"
    if fn.endswith(".webm"):
        return "video/webm"
    if fn.endswith(".docx"):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if fn.endswith(".json"):
        return "application/json"
    return "application/octet-stream"


def s3_put_bytes(key: str, data: bytes, content_type: str) -> None:
    s3.put_object(Bucket=AWS_BUCKET, Key=key, Body=data, ContentType=content_type)


def s3_put_json(key: str, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json; charset=utf-8",
    )


def s3_key_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=AWS_BUCKET, Key=key)
        return True
    except Exception:
        return False


def s3_read_json(key: str) -> Optional[Dict[str, Any]]:
    try:
        obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
        raw = obj["Body"].read().decode("utf-8")
        return json.loads(raw)
    except Exception:
        return None


def presigned_get_url(key: str, expires: int = 3600, filename: Optional[str] = None) -> str:
    params: Dict[str, Any] = {"Bucket": AWS_BUCKET, "Key": key}
    if filename:
        params["ResponseContentDisposition"] = f'attachment; filename="{filename}"'
        params["ResponseContentType"] = guess_content_type(filename)

    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params=params,
        ExpiresIn=expires,
    )


def s3_read_bytes(key: str) -> bytes:
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    return obj["Body"].read()


def is_valid_email(value: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", (value or "").strip()))


def send_test_email(group_id: str, to_email: str, outputs: Dict[str, str]) -> None:
    if not SES_FROM_EMAIL:
        raise RuntimeError("SES_FROM_EMAIL is not configured in environment.")
    if not is_valid_email(to_email):
        raise ValueError("Invalid recipient email.")

    dots_key = outputs.get("dots_video", "")
    skeleton_key = outputs.get("skeleton_video", "")
    report_en_key = outputs.get("report_en_docx", "")
    report_th_key = outputs.get("report_th_docx", "")

    msg = MIMEMultipart()
    msg["Subject"] = f"AI People Reader - Test Email ({group_id})"
    msg["From"] = SES_FROM_EMAIL
    msg["To"] = to_email

    lines = [
        f"Test email for group: {group_id}",
        "",
        "Video links:",
        f"- Dots: {presigned_get_url(dots_key, expires=604800, filename='dots.mp4')}",
        f"- Skeleton: {presigned_get_url(skeleton_key, expires=604800, filename='skeleton.mp4')}",
        "",
        "DOCX files are attached. Backup links:",
        f"- Report EN: {presigned_get_url(report_en_key, expires=604800, filename='report_en.docx')}",
        f"- Report TH: {presigned_get_url(report_th_key, expires=604800, filename='report_th.docx')}",
    ]
    msg.attach(MIMEText("\n".join(lines), "plain", "utf-8"))

    en_part = MIMEApplication(
        s3_read_bytes(report_en_key),
        _subtype="vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    en_part.add_header("Content-Disposition", "attachment", filename="report_en.docx")
    msg.attach(en_part)

    th_part = MIMEApplication(
        s3_read_bytes(report_th_key),
        _subtype="vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    th_part.add_header("Content-Disposition", "attachment", filename="report_th.docx")
    msg.attach(th_part)

    ses.send_raw_email(
        Source=SES_FROM_EMAIL,
        Destinations=[to_email],
        RawMessage={"Data": msg.as_bytes()},
    )


def enqueue_legacy_job(job: Dict[str, Any]) -> str:
    job_id = str(job["job_id"])
    job_json_key = f"{JOBS_PENDING_PREFIX}{job_id}.json"
    s3_put_json(job_json_key, job)
    return job_json_key


def safe_slug(text: str, fallback: str = "user") -> str:
    t = (text or "").strip()
    if not t:
        return fallback
    out = []
    for ch in t:
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        elif ch.isspace():
            out.append("_")
    s = "".join(out).strip("_")
    return s if s else fallback


def build_output_keys(group_id: str) -> Dict[str, str]:
    base = f"{JOBS_OUTPUT_PREFIX}groups/{group_id}/"
    return {
        "dots_video": base + "dots.mp4",
        "skeleton_video": base + "skeleton.mp4",
        "report_en_docx": base + "report_en.docx",
        "report_th_docx": base + "report_th.docx",
        "debug_en": base + "debug_en.json",
        "debug_th": base + "debug_th.json",
    }

def enqueue_report_only_job(group_id: str, client_name: str, report_style: str = "full") -> str:
    input_key = f"{JOBS_GROUP_PREFIX}{group_id}/input/input.mp4"
    if not s3_key_exists(input_key):
        raise RuntimeError(f"Input video not found for group_id={group_id}")

    created_at = utc_now_iso()
    job_report = {
        "job_id": new_job_id(),
        "group_id": group_id,
        "created_at": created_at,
        "status": "pending",
        "mode": "report",
        "input_key": input_key,
        "client_name": client_name or "Anonymous",
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "languages": ["th", "en"],
        "output_prefix": f"{JOBS_GROUP_PREFIX}{group_id}",
        "analysis_mode": "real",
        "sample_fps": 5,
        "max_frames": 300,
        "report_style": report_style,
    }
    return enqueue_legacy_job(job_report)


def get_report_outputs_from_job(group_id: str) -> Dict[str, str]:
    """Find the finished report job and extract actual output paths"""
    found: Dict[str, str] = {}
    try:
        # Search in both finished and failed jobs
        prefixes = [JOBS_FINISHED_PREFIX, JOBS_FAILED_PREFIX]
        
        for prefix in prefixes:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=prefix):
                for item in page.get("Contents", []):
                    key = item["Key"]
                    if not key.endswith(".json"):
                        continue
                    
                    # Read job JSON
                    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
                    job_data = json.loads(obj["Body"].read().decode("utf-8"))
                    
                    # Check if this job belongs to our group and is a report job
                    if (job_data.get("group_id") == group_id and 
                        job_data.get("mode") in ("report", "report_th_en", "report_generator")):
                        
                        # Extract output paths from job
                        outputs = job_data.get("outputs", {})
                        reports = outputs.get("reports", {})
                        
                        # If outputs exist, return them
                        if reports:
                            en_key = reports.get("EN", {}).get("docx_key", "")
                            th_key = reports.get("TH", {}).get("docx_key", "")
                            if en_key:
                                found["report_en_docx"] = en_key
                            if th_key:
                                found["report_th_docx"] = th_key
                            if found.get("report_en_docx") and found.get("report_th_docx"):
                                return found
                        
                        # If no outputs structure, try to construct paths from output_prefix
                        output_prefix = job_data.get("output_prefix", "")
                        if output_prefix:
                            # Try to find the actual files by listing S3
                            scanned = find_report_files_in_s3(output_prefix)
                            if scanned.get("report_en_docx"):
                                found["report_en_docx"] = scanned["report_en_docx"]
                            if scanned.get("report_th_docx"):
                                found["report_th_docx"] = scanned["report_th_docx"]
                            if found.get("report_en_docx") and found.get("report_th_docx"):
                                return found
            
    except Exception as e:
        # Silent fail - don't show error to customers
        pass

    # Fallback: scan common prefixes directly, even when job JSON is not found yet.
    fallback_prefixes = [
        f"{JOBS_GROUP_PREFIX}{group_id}",
        f"{JOBS_OUTPUT_PREFIX}groups/{group_id}",
    ]
    for pfx in fallback_prefixes:
        scanned = find_report_files_in_s3(pfx)
        if scanned.get("report_en_docx") and not found.get("report_en_docx"):
            found["report_en_docx"] = scanned["report_en_docx"]
        if scanned.get("report_th_docx") and not found.get("report_th_docx"):
            found["report_th_docx"] = scanned["report_th_docx"]
        if found.get("report_en_docx") and found.get("report_th_docx"):
            break

    return found


def find_report_files_in_s3(prefix: str) -> Dict[str, str]:
    """Find report files by scanning S3 with prefix"""
    try:
        result = {"report_en_docx": "", "report_th_docx": ""}
        paginator = s3.get_paginator("list_objects_v2")
        
        for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if key.endswith(".docx"):
                    if "_EN.docx" in key:
                        result["report_en_docx"] = key
                    elif "_TH.docx" in key:
                        result["report_th_docx"] = key
        
        return result
    except Exception:
        return {"report_en_docx": "", "report_th_docx": ""}


def ensure_session_defaults() -> None:
    if "last_group_id" not in st.session_state:
        st.session_state["last_group_id"] = ""
    if "last_outputs" not in st.session_state:
        st.session_state["last_outputs"] = {}
    if "last_jobs" not in st.session_state:
        st.session_state["last_jobs"] = {}
    if "last_job_json_keys" not in st.session_state:
        st.session_state["last_job_json_keys"] = {}


def find_job_json(job_id: str) -> Optional[str]:
    candidates = [
        f"{JOBS_PENDING_PREFIX}{job_id}.json",
        f"{JOBS_PROCESSING_PREFIX}{job_id}.json",
        f"{JOBS_FINISHED_PREFIX}{job_id}.json",
        f"{JOBS_FAILED_PREFIX}{job_id}.json",
    ]
    for k in candidates:
        if s3_key_exists(k):
            return k
    return None


def infer_job_bucket_status(job_key: str) -> str:
    if job_key.startswith(JOBS_PENDING_PREFIX):
        return "pending"
    if job_key.startswith(JOBS_PROCESSING_PREFIX):
        return "processing"
    if job_key.startswith(JOBS_FINISHED_PREFIX):
        return "finished"
    if job_key.startswith(JOBS_FAILED_PREFIX):
        return "failed"
    return "unknown"


# -------------------------
# UI
# -------------------------
ensure_session_defaults()

st.markdown("# Video Analysis (วิเคราะห์วิดีโอ)")
st.caption("Upload your video once, then click **Run Analysis** to generate dots + skeleton + reports (EN/TH). (DOCX only)")

with st.expander("Optional: User Name (ชื่อผู้ใช้) — ใช้สำหรับติดตามงาน", expanded=False):
    user_name = st.text_input("Enter User Name", value="", placeholder="e.g., Rung / Founder / Co-Founder")
    notify_email = st.text_input(
        "Notification Email (สำหรับส่งผลลัพธ์อัตโนมัติ)",
        value="",
        placeholder="name@example.com",
    )

report_type_ui = st.selectbox(
    "Report Type",
    options=["Full", "Simple"],
    index=0,
    help="Full = include graphs + detailed numbers, Simple = no graphs and no detection totals.",
)

uploaded = st.file_uploader(
    "Video (MP4/MOV/M4V/WEBM)",
    type=["mp4", "mov", "m4v", "webm"],
    accept_multiple_files=False,
)

colA, colB, colC = st.columns([1, 1, 2])
with colA:
    run = st.button("🎬 Run Analysis", type="primary", use_container_width=True)
with colB:
    refresh = st.button("🔄 Refresh", use_container_width=True)

with colC:
    manual_group = st.text_input(
        "Paste group_id to load old results (กันหายเวลา refresh/deploy)",
        value=st.session_state.get("last_group_id", ""),
        placeholder="e.g., 20260207_164107_3d658__user",
    )

note = st.empty()

# -------------------------
# Submit jobs
# -------------------------
if run:
    if not uploaded:
        note.error("Please upload a video first.")
        st.stop()

    base_user = safe_slug(user_name, fallback="user")
    group_id = f"{new_group_id()}__{base_user}"
    input_key = f"{JOBS_GROUP_PREFIX}{group_id}/input/input.mp4"

    try:
        s3_put_bytes(
            key=input_key,
            data=uploaded.getvalue(),
            content_type=guess_content_type(uploaded.name or "input.mp4"),
        )
    except Exception as e:
        note.error(f"Upload to S3 failed: {e}")
        st.stop()

    outputs = build_output_keys(group_id)
    created_at = utc_now_iso()

    job_dots = {
        "job_id": new_job_id(),
        "group_id": group_id,
        "created_at": created_at,
        "status": "pending",
        "mode": "dots",
        "input_key": input_key,
        "output_key": outputs["dots_video"],
        "user_name": user_name or "",
    }

    job_skel = {
        "job_id": new_job_id(),
        "group_id": group_id,
        "created_at": created_at,
        "status": "pending",
        "mode": "skeleton",
        "input_key": input_key,
        "output_key": outputs["skeleton_video"],
        "user_name": user_name or "",
    }

    # Report job - handled by report_worker.py
    job_report = {
        "job_id": new_job_id(),
        "group_id": group_id,
        "created_at": created_at,
        "status": "pending",
        "mode": "report",  # report_worker.py handles this
        "input_key": input_key,
        "client_name": user_name or "Anonymous",
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "languages": ["th", "en"],
        "output_prefix": f"{JOBS_GROUP_PREFIX}{group_id}",
        "analysis_mode": "real",  # Use real MediaPipe analysis
        "sample_fps": 5,
        "max_frames": 300,
        "report_style": "simple" if report_type_ui == "Simple" else "full",
        "notify_email": (notify_email or "").strip(),
    }

    try:
        k1 = enqueue_legacy_job(job_dots)
        k2 = enqueue_legacy_job(job_skel)
        k3 = enqueue_legacy_job(job_report)
    except Exception as e:
        note.error(f"Enqueue job failed: {e}")
        st.stop()

    st.session_state["last_group_id"] = group_id
    st.session_state["last_outputs"] = outputs
    st.session_state["last_jobs"] = {
        "dots": job_dots["job_id"],
        "skeleton": job_skel["job_id"],
        "report": job_report["job_id"],
    }
    st.session_state["last_job_json_keys"] = {
        "dots": k1,
        "skeleton": k2,
        "report": k3,
    }

    note.success(f"Submitted! group_id = {group_id}")
    st.info("Wait a bit, then press Refresh. หรือ copy group_id เก็บไว้ แล้ว paste กลับมาได้เสมอ")


# -------------------------
# Download section
# -------------------------
st.divider()
st.subheader("Downloads (ผลลัพธ์สำหรับดาวน์โหลด)")

group_id = (manual_group or "").strip()
if group_id:
    outputs = build_output_keys(group_id)
    # Get actual report paths from finished job JSON
    report_outputs = get_report_outputs_from_job(group_id)
    # Only override when a real key is discovered; never blank out defaults.
    if report_outputs.get("report_en_docx"):
        outputs["report_en_docx"] = report_outputs["report_en_docx"]
    if report_outputs.get("report_th_docx"):
        outputs["report_th_docx"] = report_outputs["report_th_docx"]
else:
    st.caption("No group_id yet. Upload a video and click **Run Analysis**.")
    st.stop()

st.caption(f"Group: `{group_id}`")


def download_block(title: str, key: str, filename: str) -> None:
    if not key:
        st.write(f"- {title}: (missing key)")
        return
    ready = s3_key_exists(key)
    if ready:
        url = presigned_get_url(key, expires=3600, filename=filename)
        st.success(f"✅ {title} ready")
        st.link_button(f"Download {title}", url, use_container_width=True)
        st.code(key, language="text")
    else:
        st.warning(f"⏳ {title} not ready yet")
        st.code(key, language="text")


# --- Downloads ---
c1, c2 = st.columns(2)

with c1:
    st.markdown("### Videos")
    download_block("Dots video", outputs.get("dots_video", ""), "dots.mp4")
    download_block("Skeleton video", outputs.get("skeleton_video", ""), "skeleton.mp4")

with c2:
    st.markdown("### Reports")
    st.markdown("**English**")
    download_block("Report EN (DOCX)", outputs.get("report_en_docx", ""), "report_en.docx")

    st.markdown("**Thai**")
    download_block("Report TH (DOCX)", outputs.get("report_th_docx", ""), "report_th.docx")

reports_ready = bool(outputs.get("report_en_docx")) and bool(outputs.get("report_th_docx")) and s3_key_exists(outputs.get("report_en_docx", "")) and s3_key_exists(outputs.get("report_th_docx", ""))
videos_ready = bool(outputs.get("dots_video")) and bool(outputs.get("skeleton_video")) and s3_key_exists(outputs.get("dots_video", "")) and s3_key_exists(outputs.get("skeleton_video", ""))

if videos_ready and not reports_ready:
    st.divider()
    st.warning("Reports are still not ready. You can re-run report generation for this group.")
    if st.button("Re-run report generation", use_container_width=False):
        try:
            guessed_name = group_id.split("__", 1)[1] if "__" in group_id else "Anonymous"
            new_report_key = enqueue_report_only_job(
                group_id=group_id,
                client_name=guessed_name,
                report_style="full",
            )
            st.success(f"Queued report job again: {new_report_key}")
        except Exception as e:
            st.error(f"Cannot re-queue report job: {e}")

# --- Manual email test (only when all files are ready) ---
all_ready = all(
    s3_key_exists(k)
    for k in [
        outputs.get("dots_video", ""),
        outputs.get("skeleton_video", ""),
        outputs.get("report_en_docx", ""),
        outputs.get("report_th_docx", ""),
    ]
    if k
) and all(
    bool(outputs.get(k))
    for k in ["dots_video", "skeleton_video", "report_en_docx", "report_th_docx"]
)

if all_ready:
    st.divider()
    st.subheader("Email Test")
    test_email = st.text_input(
        "Recipient email for test send",
        value=(notify_email or "").strip(),
        placeholder="name@example.com",
    )
    if st.button("Send test email now", use_container_width=False):
        try:
            send_test_email(group_id=group_id, to_email=test_email, outputs=outputs)
            st.success("Test email sent.")
        except Exception as e:
            st.error(f"Send email failed: {e}")

st.caption("Tip: ถ้า refresh แล้วไม่ขึ้น ให้ paste group_id ที่ใช้งานจริง แล้วกด Refresh ใหม่ได้ตลอด")
