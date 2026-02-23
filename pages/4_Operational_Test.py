import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import boto3
import streamlit as st
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

SUPPORT_CONTACT_TEXT = "หากพบปัญหากรุณาติดต่อ 0817008484"


st.set_page_config(page_title="Operational Test", layout="wide")

AWS_BUCKET = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

if not AWS_BUCKET:
    st.error("Missing AWS_BUCKET (or S3_BUCKET) environment variable.")
    st.stop()

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    config=Config(signature_version="s3v4"),
)
S3_UPLOAD_CONFIG = TransferConfig(
    multipart_threshold=8 * 1024 * 1024,
    multipart_chunksize=8 * 1024 * 1024,
    max_concurrency=4,
    use_threads=True,
)

JOBS_PENDING_PREFIX = "jobs/pending/"
JOBS_PROCESSING_PREFIX = "jobs/processing/"
JOBS_FINISHED_PREFIX = "jobs/finished/"
JOBS_FAILED_PREFIX = "jobs/failed/"
JOBS_GROUP_PREFIX = "jobs/groups/"
OPERATION_TEST_RECIPIENTS = [
    "rungnapa@imagematters.at",
    "alisa@imagematters.at",
]

BANNER_PATHS = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "top_banner.png"),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "banner.png"),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "Header.png"),
]


def render_banner() -> None:
    for path in BANNER_PATHS:
        if os.path.exists(path):
            st.image(path, width="stretch")
            return


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_job_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}__{uuid.uuid4().hex[:5]}"


def new_group_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:6]}"


def safe_slug(text: str, fallback: str = "user") -> str:
    cleaned = []
    for ch in str(text or "").strip():
        if ch.isalnum() or ch in ("_", "-"):
            cleaned.append(ch)
        elif ch.isspace():
            cleaned.append("_")
    out = "".join(cleaned).strip("_")
    return out or fallback


def guess_content_type(filename: str) -> str:
    fn = str(filename or "").lower()
    if fn.endswith(".mp4"):
        return "video/mp4"
    if fn.endswith(".mov"):
        return "video/quicktime"
    if fn.endswith(".m4v"):
        return "video/x-m4v"
    if fn.endswith(".webm"):
        return "video/webm"
    if fn.endswith(".pdf"):
        return "application/pdf"
    if fn.endswith(".json"):
        return "application/json"
    return "application/octet-stream"


def s3_upload_stream(key: str, file_obj: Any, content_type: str) -> None:
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    s3.upload_fileobj(
        Fileobj=file_obj,
        Bucket=AWS_BUCKET,
        Key=key,
        ExtraArgs={"ContentType": content_type},
        Config=S3_UPLOAD_CONFIG,
    )


def s3_put_json(key: str, payload: Dict[str, Any]) -> None:
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
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
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return None


def presigned_get_url(key: str, expires: int = 3600, filename: Optional[str] = None) -> str:
    params: Dict[str, Any] = {"Bucket": AWS_BUCKET, "Key": key}
    if filename:
        params["ResponseContentDisposition"] = f'attachment; filename="{filename}"'
        params["ResponseContentType"] = guess_content_type(filename)
    return s3.generate_presigned_url("get_object", Params=params, ExpiresIn=expires)


def enqueue_report_job(job: Dict[str, Any]) -> str:
    key = f"{JOBS_PENDING_PREFIX}{job['job_id']}.json"
    s3_put_json(key, job)
    return key


def is_valid_email_format(value: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", str(value or "").strip()))


def get_job_status_from_key(key: str) -> str:
    if key.startswith(JOBS_PENDING_PREFIX):
        return "pending"
    if key.startswith(JOBS_PROCESSING_PREFIX):
        return "processing"
    if key.startswith(JOBS_FINISHED_PREFIX):
        return "finished"
    if key.startswith(JOBS_FAILED_PREFIX):
        return "failed"
    return "unknown"


def get_latest_operation_test_job(group_id: str) -> Dict[str, Any]:
    latest_job: Dict[str, Any] = {}
    latest_sort: Optional[tuple] = None
    status_rank = {
        "finished": 3,
        "processing": 2,
        "pending": 1,
        "failed": 0,
    }
    prefixes = [JOBS_PENDING_PREFIX, JOBS_PROCESSING_PREFIX, JOBS_FINISHED_PREFIX, JOBS_FAILED_PREFIX]
    try:
        for prefix in prefixes:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=prefix):
                for item in page.get("Contents", []):
                    key = str(item.get("Key") or "")
                    if not key.endswith(".json"):
                        continue
                    payload = s3_read_json(key) or {}
                    if str(payload.get("group_id") or "").strip() != group_id:
                        continue
                    if str(payload.get("mode") or "").strip().lower() != "report":
                        continue
                    status = get_job_status_from_key(key)
                    last_modified = item.get("LastModified")
                    last_ts = (
                        float(last_modified.timestamp())
                        if hasattr(last_modified, "timestamp")
                        else 0.0
                    )
                    created_at = str(payload.get("created_at") or "")
                    sort_key = (last_ts, created_at, status_rank.get(status, -1), key)
                    if latest_sort is None or sort_key > latest_sort:
                        latest_sort = sort_key
                        latest_job = payload
                        latest_job["_job_bucket_key"] = key
        if latest_job:
            pass
        return latest_job
    except Exception:
        return {}


def get_pdf_key_from_job(job: Dict[str, Any]) -> str:
    reports = (job.get("outputs") or {}).get("reports") or {}
    th_pdf = ((reports.get("TH") or {}).get("pdf_key") or "").strip()
    en_pdf = ((reports.get("EN") or {}).get("pdf_key") or "").strip()
    return th_pdf or en_pdf


def get_pdf_keys_from_job(job: Dict[str, Any]) -> Dict[str, str]:
    reports = (job.get("outputs") or {}).get("reports") or {}
    th_pdf = ((reports.get("TH") or {}).get("pdf_key") or "").strip()
    en_pdf = ((reports.get("EN") or {}).get("pdf_key") or "").strip()
    return {"TH": th_pdf, "EN": en_pdf}


def get_latest_finished_operation_test_job(group_id: str) -> Dict[str, Any]:
    latest_job: Dict[str, Any] = {}
    latest_ts = 0.0
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=JOBS_FINISHED_PREFIX):
            for item in page.get("Contents", []):
                key = str(item.get("Key") or "")
                if not key.endswith(".json"):
                    continue
                payload = s3_read_json(key) or {}
                if str(payload.get("group_id") or "").strip() != group_id:
                    continue
                if str(payload.get("mode") or "").strip().lower() != "report":
                    continue
                last_modified = item.get("LastModified")
                ts = float(last_modified.timestamp()) if hasattr(last_modified, "timestamp") else 0.0
                if ts >= latest_ts:
                    latest_ts = ts
                    latest_job = payload
                    latest_job["_job_bucket_key"] = key
        return latest_job
    except Exception:
        return {}


def get_latest_failed_operation_test_job(group_id: str) -> Dict[str, Any]:
    latest_job: Dict[str, Any] = {}
    latest_ts = 0.0
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=JOBS_FAILED_PREFIX):
            for item in page.get("Contents", []):
                key = str(item.get("Key") or "")
                if not key.endswith(".json"):
                    continue
                payload = s3_read_json(key) or {}
                if str(payload.get("group_id") or "").strip() != group_id:
                    continue
                if str(payload.get("mode") or "").strip().lower() != "report":
                    continue
                last_modified = item.get("LastModified")
                ts = float(last_modified.timestamp()) if hasattr(last_modified, "timestamp") else 0.0
                if ts >= latest_ts:
                    latest_ts = ts
                    latest_job = payload
                    latest_job["_job_bucket_key"] = key
        return latest_job
    except Exception:
        return {}


def find_latest_group_pdf_key(group_id: str) -> str:
    prefixes = [
        f"{JOBS_GROUP_PREFIX}{group_id}/",
        f"jobs/output/groups/{group_id}/",
    ]
    latest_key = ""
    latest_ts = 0.0
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for prefix in prefixes:
            for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=prefix):
                for item in page.get("Contents", []):
                    key = str(item.get("Key") or "")
                    if not key.lower().endswith(".pdf"):
                        continue
                    last_modified = item.get("LastModified")
                    ts = float(last_modified.timestamp()) if hasattr(last_modified, "timestamp") else 0.0
                    if ts >= latest_ts:
                        latest_ts = ts
                        latest_key = key
    except Exception:
        return ""
    return latest_key


def find_pdf_key_from_finished_jobs(group_id: str) -> str:
    latest_key = ""
    latest_ts = 0.0
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=JOBS_FINISHED_PREFIX):
            for item in page.get("Contents", []):
                key = str(item.get("Key") or "")
                if not key.endswith(".json"):
                    continue
                payload = s3_read_json(key) or {}
                if str(payload.get("group_id") or "").strip() != group_id:
                    continue
                pdf_key = get_pdf_key_from_job(payload)
                if not pdf_key:
                    continue
                last_modified = item.get("LastModified")
                ts = float(last_modified.timestamp()) if hasattr(last_modified, "timestamp") else 0.0
                if ts >= latest_ts:
                    latest_ts = ts
                    latest_key = pdf_key
    except Exception:
        return ""
    return latest_key


def prettify_level(level: str) -> str:
    text = str(level or "").strip().lower()
    if text.startswith("low"):
        return "ต่ำ"
    if text.startswith("moderate"):
        return "กลาง"
    if text.startswith("high"):
        return "สูง"
    return "-"


def ensure_session_defaults() -> None:
    if "operation_test_group_id" not in st.session_state:
        st.session_state["operation_test_group_id"] = ""
    if "operation_test_upload_nonce" not in st.session_state:
        st.session_state["operation_test_upload_nonce"] = 0


def read_group_id_from_url() -> str:
    try:
        val = st.query_params.get("group_id", "")
        if isinstance(val, list):
            val = val[0] if val else ""
        return str(val or "").strip()
    except Exception:
        return ""


def persist_group_id_to_url(group_id: str) -> None:
    gid = str(group_id or "").strip()
    try:
        if gid:
            st.query_params["group_id"] = gid
        elif "group_id" in st.query_params:
            del st.query_params["group_id"]
    except Exception:
        pass


ensure_session_defaults()
render_banner()

st.markdown("# Operational Test")
st.caption("Upload one video to analyze First Impression only: Eye Contact, Uprightness, and Stance.")
st.caption("Result format: PDF only.")
manual_group_id = st.text_input(
    "Check result by Group ID (optional)",
    value=read_group_id_from_url(),
    placeholder="Paste group_id from email to fetch result on this page",
).strip()

name = st.text_input("Name (optional)", value="", placeholder="e.g., John Doe")
st.markdown("### Email Recipients")
recipient_mode = st.radio(
    "Send report to",
    options=["One recipient", "Both recipients"],
    horizontal=True,
    index=1,
)
if recipient_mode == "One recipient":
    selected_recipients = [
        st.selectbox(
            "Select recipient",
            options=OPERATION_TEST_RECIPIENTS,
            index=0,
        )
    ]
else:
    selected_recipients = OPERATION_TEST_RECIPIENTS[:]
notify_email = ",".join(selected_recipients)

st.markdown("### Report Language")
language_mode = st.radio(
    "Choose report language",
    options=["Thai only", "English only", "Thai + English"],
    horizontal=True,
    index=2,
)
if language_mode == "Thai only":
    selected_languages = ["th"]
elif language_mode == "English only":
    selected_languages = ["en"]
else:
    selected_languages = ["th", "en"]

uploaded = st.file_uploader(
    "Video (MP4/MOV/M4V/WEBM)",
    type=["mp4", "mov", "m4v", "webm"],
    accept_multiple_files=False,
    key=f"operation_test_uploader_{st.session_state['operation_test_upload_nonce']}",
)

run = st.button("Analyze First Impression", type="primary", width="stretch")
st.caption(SUPPORT_CONTACT_TEXT)
notice = st.empty()

if run:
    if not uploaded:
        notice.error("Please upload a video first.")
        st.stop()
    if not selected_recipients:
        notice.error("Please select at least one recipient email.")
        st.stop()

    base_user = safe_slug(name or selected_recipients[0] or "user", fallback="user")
    group_id = f"{new_group_id()}__{base_user}"
    input_key = f"{JOBS_GROUP_PREFIX}{group_id}/input/input.mp4"

    try:
        s3_upload_stream(
            key=input_key,
            file_obj=uploaded,
            content_type=guess_content_type(uploaded.name or "input.mp4"),
        )
    except Exception as e:
        notice.error(f"Upload to S3 failed: {e}")
        st.warning(SUPPORT_CONTACT_TEXT)
        st.stop()

    report_job = {
        "job_id": new_job_id(),
        "group_id": group_id,
        "created_at": utc_now_iso(),
        "status": "pending",
        "mode": "report",
        "input_key": input_key,
        "client_name": (name or "Anonymous").strip() or "Anonymous",
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "languages": selected_languages,
        "output_prefix": f"{JOBS_GROUP_PREFIX}{group_id}",
        "analysis_mode": "real",
        "sample_fps": 5,
        "max_frames": 300,
        "report_style": "operation_test",
        "report_format": "pdf",
        "notify_email": notify_email.strip(),
        "enterprise_folder": "operation_test",
        "expect_dots": False,
    }

    try:
        enqueue_key = enqueue_report_job(report_job)
    except Exception as e:
        notice.error(f"Enqueue job failed: {e}")
        st.warning(SUPPORT_CONTACT_TEXT)
        st.stop()

    st.session_state["operation_test_group_id"] = group_id
    st.session_state["operation_test_upload_nonce"] = int(st.session_state.get("operation_test_upload_nonce") or 0) + 1
    persist_group_id_to_url(group_id)
    notice.success("Submit to the worker. Load for the next vdo.")
    st.caption(f"group_id = {group_id}")
    st.caption(f"Queued job key: {enqueue_key}")
    st.rerun()

active_group_id = manual_group_id or st.session_state.get("operation_test_group_id") or read_group_id_from_url()
if active_group_id:
    persist_group_id_to_url(active_group_id)
    st.session_state["operation_test_group_id"] = active_group_id
    st.divider()
    st.subheader("Operational Test Result")
    st.caption(f"Group: `{active_group_id}`")

    latest_job = get_latest_operation_test_job(active_group_id)
    if not latest_job:
        st.info("Waiting for job registration. Please refresh in a moment.")
        st.stop()

    job_key = str(latest_job.get("_job_bucket_key") or "")
    status = get_job_status_from_key(job_key)

    st.markdown(f"**Job status:** `{status}`")
    if st.button("Refresh Result", width="content"):
        st.rerun()
    notification = latest_job.get("notification") or {}
    notification_status = str(notification.get("status") or "").strip()
    if notification_status:
        st.caption(f"Email status: `{notification_status}`")

    summary = latest_job.get("first_impression_summary") or {}
    if isinstance(summary, dict) and summary:
        st.markdown("### First Impression (ต่ำ / กลาง / สูง)")
        c1, c2, c3 = st.columns(3)
        eye = summary.get("eye_contact") or {}
        up = summary.get("uprightness") or {}
        stance = summary.get("stance") or {}
        c1.metric("Eye Contact", prettify_level(eye.get("level")))
        c2.metric("Uprightness", prettify_level(up.get("level")))
        c3.metric("Stance", prettify_level(stance.get("level")))

    pdf_keys = get_pdf_keys_from_job(latest_job)
    pdf_key = pdf_keys.get("TH") or pdf_keys.get("EN") or ""
    if not pdf_key:
        finished_job = get_latest_finished_operation_test_job(active_group_id)
        pdf_keys = get_pdf_keys_from_job(finished_job)
        pdf_key = pdf_keys.get("TH") or pdf_keys.get("EN") or ""
        if finished_job:
            notif = (finished_job.get("notification") or {})
            notif_status = str(notif.get("status") or "").strip()
            if notif_status:
                st.caption(f"Email status: `{notif_status}`")
    if not pdf_key:
        pdf_key = find_pdf_key_from_finished_jobs(active_group_id)
    failed_job = get_latest_failed_operation_test_job(active_group_id)
    failed_error = str((failed_job or {}).get("error") or "").strip()
    if failed_error:
        st.error(f"Latest failed error: {failed_error}")
    if not pdf_key:
        pdf_key = find_latest_group_pdf_key(active_group_id)

    if pdf_key and s3_key_exists(pdf_key):
        st.success("PDF is ready.")
        th_key = (pdf_keys.get("TH") or "").strip()
        en_key = (pdf_keys.get("EN") or "").strip()
        th_ready = bool(th_key) and s3_key_exists(th_key)
        en_ready = bool(en_key) and s3_key_exists(en_key)
        if th_ready:
            st.link_button(
                "Download Operational Test PDF (TH)",
                presigned_get_url(th_key, expires=3600, filename="operational_test_report_th.pdf"),
                width="stretch",
            )
            st.code(th_key, language="text")
        if en_ready:
            st.link_button(
                "Download Operational Test PDF (EN)",
                presigned_get_url(en_key, expires=3600, filename="operational_test_report_en.pdf"),
                width="stretch",
            )
            st.code(en_key, language="text")
        if (not th_ready) and (not en_ready):
            st.link_button(
                "Download Operational Test PDF",
                presigned_get_url(pdf_key, expires=3600, filename="operational_test_report.pdf"),
                width="stretch",
            )
            st.code(pdf_key, language="text")
    elif status == "failed":
        st.error("Analysis failed. Please upload again and retry.")
    else:
        st.info("Processing. PDF is not ready yet. Refresh this page shortly.")
else:
    st.caption("Upload a video and click Analyze First Impression to start.")

