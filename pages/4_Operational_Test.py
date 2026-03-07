import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import boto3
import streamlit as st
import streamlit.components.v1 as components
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
JOBS_OUTPUT_PREFIX = "jobs/output/groups/"
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


def enqueue_job(job: Dict[str, Any]) -> str:
    """Enqueue any job (dots, skeleton, report) to pending queue."""
    key = f"{JOBS_PENDING_PREFIX}{job['job_id']}.json"
    s3_put_json(key, job)
    return key


def verify_pending_jobs_exist(keys: List[str]) -> List[str]:
    retries = int(os.getenv("PENDING_VERIFY_RETRIES", "5") or "5")
    delay_seconds = float(os.getenv("PENDING_VERIFY_DELAY_SECONDS", "0.5") or "0.5")
    pending = list(keys)
    for attempt in range(max(1, retries)):
        missing: List[str] = []
        for key in pending:
            if not s3_key_exists(key):
                missing.append(key)
        if not missing:
            return []
        pending = missing
        if attempt < retries - 1:
            time.sleep(delay_seconds)
    return pending


def build_output_keys(group_id: str) -> Dict[str, str]:
    base = f"{JOBS_OUTPUT_PREFIX}{group_id}/"
    return {
        "dots_video": base + "dots.mp4",
        "skeleton_video": base + "skeleton.mp4",
    }


def find_dots_skeleton_keys(group_id: str) -> Dict[str, str]:
    """Find dots.mp4 and skeleton.mp4 under group output paths."""
    found: Dict[str, str] = {}
    prefixes = [
        f"{JOBS_OUTPUT_PREFIX}{group_id}/",
        f"{JOBS_GROUP_PREFIX}{group_id}/",
    ]
    for prefix in prefixes:
        try:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=prefix):
                for item in page.get("Contents", []):
                    key = str(item.get("Key") or "")
                    if key.endswith("dots.mp4") and not found.get("dots_video"):
                        found["dots_video"] = key
                    if key.endswith("skeleton.mp4") and not found.get("skeleton_video"):
                        found["skeleton_video"] = key
        except Exception:
            pass
    return found


def is_valid_email_format(value: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", str(value or "").strip()))

def format_submit_error_message(err: Exception) -> str:
    message = str(err or "").strip() or "Unknown error"
    lowered = message.lower()
    if "axioserror" in lowered or "status code 400" in lowered:
        return f"{message}\n\nกรุณา upload วีดีโอใหม่"
    return message


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


def get_docx_keys_from_job(job: Dict[str, Any]) -> Dict[str, str]:
    reports = (job.get("outputs") or {}).get("reports") or {}
    th_docx = ((reports.get("TH") or {}).get("docx_key") or "").strip()
    en_docx = ((reports.get("EN") or {}).get("docx_key") or "").strip()
    return {"TH": th_docx, "EN": en_docx}


def get_html_keys_from_job(job: Dict[str, Any]) -> Dict[str, str]:
    reports = (job.get("outputs") or {}).get("reports") or {}
    th_html = ((reports.get("TH") or {}).get("html_key") or "").strip()
    en_html = ((reports.get("EN") or {}).get("html_key") or "").strip()
    return {"TH": th_html, "EN": en_html}


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


def find_docx_keys_from_finished_jobs(group_id: str) -> Dict[str, str]:
    result: Dict[str, str] = {"TH": "", "EN": ""}
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
                reports = (payload.get("outputs") or {}).get("reports") or {}
                th_docx = ((reports.get("TH") or {}).get("docx_key") or "").strip()
                en_docx = ((reports.get("EN") or {}).get("docx_key") or "").strip()
                if th_docx:
                    result["TH"] = th_docx
                if en_docx:
                    result["EN"] = en_docx
                if result["TH"] and result["EN"]:
                    return result
    except Exception:
        pass
    return result


def find_report_keys_in_s3(group_id: str) -> Dict[str, str]:
    result: Dict[str, str] = {
        "pdf_th": "",
        "pdf_en": "",
        "docx_th": "",
        "docx_en": "",
        "html_th": "",
        "html_en": "",
    }
    prefixes = [
        f"{JOBS_GROUP_PREFIX}{group_id}/",
        f"{JOBS_OUTPUT_PREFIX}{group_id}/",
    ]
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for prefix in prefixes:
            for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=prefix):
                for item in page.get("Contents", []):
                    key = str(item.get("Key") or "")
                    lo = key.lower()
                    if lo.endswith("_th.pdf"):
                        result["pdf_th"] = key
                    elif lo.endswith("_en.pdf"):
                        result["pdf_en"] = key
                    elif lo.endswith("_th.docx"):
                        result["docx_th"] = key
                    elif lo.endswith("_en.docx"):
                        result["docx_en"] = key
                    elif lo.endswith("_th.html"):
                        result["html_th"] = key
                    elif lo.endswith("_en.html"):
                        result["html_en"] = key
    except Exception:
        pass
    return result


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
    if "clear_upload_counter" not in st.session_state:
        st.session_state["clear_upload_counter"] = 0


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
components.html(
    """
    <script>
    (function () {
      try {
        var topLoc = window.top.location;
        var path = topLoc.pathname || "/";
        if (path !== "/" && !path.startsWith("/_stcore")) {
          topLoc.replace(topLoc.origin + "/" + (topLoc.search || ""));
        }
      } catch (e) {}
    })();
    </script>
    """,
    height=0,
)

st.markdown("# Operational Test")

st.divider()
st.caption("Select specific functions to test. Useful when debugging — run only what you need.")
_op_clear = st.session_state.get("clear_upload_counter", 0)

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

st.markdown("### Functions to test")
st.caption("เลือกฟังก์ชันที่ต้องการทดสอบ (เลือกได้หลายอัน) — ไม่ต้องรันทุกอย่าง")
enable_dots = st.checkbox("Dots video", value=False, help="Generate dots overlay video only")
enable_skeleton = st.checkbox("Skeleton video", value=False, help="Generate skeleton overlay video only")
enable_th_report = st.checkbox("Thai report (PDF)", value=False, help="Generate Thai report PDF only")
enable_en_report = st.checkbox("English report (PDF)", value=False, help="Generate English report PDF only")
enable_th_docx = st.checkbox("Thai report (DOCX)", value=False, help="Generate Thai report DOCX only")
enable_en_docx = st.checkbox("English report (DOCX)", value=False, help="Generate English report DOCX only")
enable_th_html = st.checkbox("Thai report (HTML)", value=False, help="Generate Thai report HTML only")
enable_en_html = st.checkbox("English report (HTML)", value=False, help="Generate English report HTML only")

uploaded = st.file_uploader(
    "Video (MP4/MOV/M4V/WEBM)",
    type=["mp4", "mov", "m4v", "webm"],
    accept_multiple_files=False,
    key=f"operation_test_uploader_{_op_clear}_{st.session_state.get('operation_test_upload_nonce', 0)}",
)

if uploaded is not None:
    uploaded_name = str(uploaded.name or "input.mp4")
    uploaded_size_mb = float((uploaded.size or 0) / (1024 * 1024))
    st.caption(f"Selected file: `{uploaded_name}` ({uploaded_size_mb:.2f} MB)")
run = st.button("Run selected functions", type="primary", width="stretch", disabled=(uploaded is None))
st.caption(SUPPORT_CONTACT_TEXT)
notice = st.empty()

op_top_group_id = str(st.session_state.get("operation_test_group_id") or read_group_id_from_url() or "").strip()
op_group_hint = op_top_group_id
if run and not uploaded:
    st.warning("ยังไม่ได้เลือกไฟล์วิดีโอ กรุณาเลือกไฟล์ก่อนกด Run selected functions")
elif run and uploaded is not None:
    st.info("กำลังตรวจสอบฟังก์ชันที่เลือกและส่งงานเข้าคิว...")
elif uploaded is not None:
    st.info("เลือกไฟล์แล้ว กด Run selected functions เพื่อเริ่มทำงาน")
elif op_group_hint:
    st.caption(f"งานล่าสุด: `{op_group_hint}` (เลื่อนลงไปดูผลลัพธ์/ดาวน์โหลดได้)")
else:
    st.caption("ยังไม่เริ่มอัปโหลด กรุณาเลือกไฟล์และกด Run selected functions")

# --- ดาวน์โหลดผลจากอีเมล (แสดงด้านล่างปุ่ม Run selected functions) ---
st.divider()
st.markdown("### 📥 ดาวน์โหลดผลลัพธ์ (ได้รับอีเมลแล้ว)")
st.caption("วาง Group ID จากอีเมล (บรรทัด 'Group ID: ...') ด้านล่าง แล้วกดรีเฟรช")
manual_group_id = st.text_input(
    "Submission ID / Group ID (วางจากอีเมลได้)",
    value=read_group_id_from_url(),
    placeholder="Paste submission_id from email to fetch result on this page",
    key=f"operation_test_manual_group_id_{_op_clear}",
).strip()
if manual_group_id:
    st.session_state["operation_test_group_id"] = manual_group_id
    persist_group_id_to_url(manual_group_id)
_op_btn1, _op_btn2 = st.columns(2)
with _op_btn1:
    if st.button("🔄 รีเฟรชสถานะผลลัพธ์", key="optest_refresh_status_top", width="content"):
        st.rerun()
with _op_btn2:
    _op_has_prev = bool(manual_group_id or st.session_state.get("operation_test_group_id") or read_group_id_from_url())
    if _op_has_prev and st.button("🗑️ ล้างผลลัพธ์เพื่ออัปโหลดวิดีโอใหม่", key="optest_clear_results_top", type="secondary"):
        for _k in ("operation_test_group_id", "operation_test_queued", "operation_test_queued_keys", "last_uploaded_filename"):
            st.session_state.pop(_k, None)
        st.session_state["clear_upload_counter"] = st.session_state.get("clear_upload_counter", 0) + 1
        try:
            if "group_id" in st.query_params:
                del st.query_params["group_id"]
        except Exception:
            pass
        st.rerun()

op_top_group_id = str(manual_group_id or st.session_state.get("operation_test_group_id") or read_group_id_from_url() or "").strip()
if op_top_group_id:
    st.caption(f"กำลังตรวจไฟล์จาก Submission ID: `{op_top_group_id}`")
    op_outputs = build_output_keys(op_top_group_id)
    op_found_videos = find_dots_skeleton_keys(op_top_group_id)
    op_found_reports = find_report_keys_in_s3(op_top_group_id)
    op_dots_key = op_outputs.get("dots_video", "") or op_found_videos.get("dots_video", "")
    op_skel_key = op_outputs.get("skeleton_video", "") or op_found_videos.get("skeleton_video", "")
    op_th_key = op_found_reports.get("pdf_th") or op_found_reports.get("docx_th", "")
    op_en_key = op_found_reports.get("pdf_en") or op_found_reports.get("docx_en", "")
    op_items = [
        ("Dots Video", op_dots_key, "dots.mp4"),
        ("Skeleton Video", op_skel_key, "skeleton.mp4"),
        ("Report TH (PDF/DOCX)", op_th_key, "report_th.pdf" if (op_found_reports.get("pdf_th") or "").strip() else "report_th.docx"),
        ("Report EN (PDF/DOCX)", op_en_key, "report_en.pdf" if (op_found_reports.get("pdf_en") or "").strip() else "report_en.docx"),
    ]
    op_ready = any(key and s3_key_exists(key) for _, key, _ in op_items)
    if op_ready:
        st.success("พบไฟล์พร้อมดาวน์โหลด")
    else:
        st.caption("หากได้รับอีเมลแล้ว ลิงก์ด้านล่างจะทำงานเมื่อไฟล์พร้อม (กดรีเฟรชหลัง 2–5 นาที)")
    st.markdown("**ลิงก์ดาวน์โหลด**")
    for label, key, fn in op_items:
        if key:
            fn_use = fn if fn in ("dots.mp4", "skeleton.mp4") else (os.path.basename(key) or fn)
            try:
                url = presigned_get_url(key, expires=3600, filename=fn_use)
                st.link_button(f"⬇️ {label}", url, width="stretch")
            except Exception:
                pass
elif not op_top_group_id:
    st.caption("กรุณาวาง Group ID จากอีเมลด้านบนเพื่อดูปุ่มดาวน์โหลด")

st.divider()
if run:
    if not uploaded:
        notice.error("Please upload a video first.")
        st.stop()
    if not (
        enable_dots
        or enable_skeleton
        or enable_th_report
        or enable_en_report
        or enable_th_docx
        or enable_en_docx
        or enable_th_html
        or enable_en_html
    ):
        notice.error("Please select at least one function to test.")
        st.stop()

    report_pdf_languages = []
    if enable_th_report:
        report_pdf_languages.append("th")
    if enable_en_report:
        report_pdf_languages.append("en")

    report_docx_languages = []
    if enable_th_docx:
        report_docx_languages.append("th")
    if enable_en_docx:
        report_docx_languages.append("en")

    report_html_languages = []
    if enable_th_html:
        report_html_languages.append("th")
    if enable_en_html:
        report_html_languages.append("en")

    if (
        enable_th_report
        or enable_en_report
        or enable_th_docx
        or enable_en_docx
        or enable_th_html
        or enable_en_html
    ) and not selected_recipients:
        notice.error("Please select at least one recipient email for report.")
        st.stop()

    base_user = safe_slug(name or (selected_recipients[0] if selected_recipients else "user"), fallback="user")
    group_id = f"{new_group_id()}__{base_user}"
    input_key = f"{JOBS_GROUP_PREFIX}{group_id}/input/input.mp4"
    outputs = build_output_keys(group_id)
    created_at = utc_now_iso()

    try:
        s3_upload_stream(
            key=input_key,
            file_obj=uploaded,
            content_type=guess_content_type(uploaded.name or "input.mp4"),
        )
    except Exception as e:
        notice.error(f"Upload to S3 failed: {format_submit_error_message(e)}")
        st.warning(SUPPORT_CONTACT_TEXT)
        st.stop()

    queued: list[str] = []
    queued_job_keys: List[str] = []
    try:
        if enable_dots:
            job_dots = {
                "job_id": new_job_id(),
                "group_id": group_id,
                "created_at": created_at,
                "status": "pending",
                "mode": "dots",
                "input_key": input_key,
                "output_key": outputs["dots_video"],
                "user_name": (name or "Anonymous").strip() or "Anonymous",
                "notify_email": notify_email.strip(),
            }
            queued_job_keys.append(enqueue_job(job_dots))
            queued.append("dots")
        if enable_skeleton:
            job_skeleton = {
                "job_id": new_job_id(),
                "group_id": group_id,
                "created_at": created_at,
                "status": "pending",
                "mode": "skeleton",
                "input_key": input_key,
                "output_key": outputs["skeleton_video"],
                "user_name": (name or "Anonymous").strip() or "Anonymous",
                "notify_email": notify_email.strip(),
            }
            queued_job_keys.append(enqueue_job(job_skeleton))
            queued.append("skeleton")
        if report_pdf_languages:
            report_job = {
                "job_id": new_job_id(),
                "group_id": group_id,
                "created_at": created_at,
                "status": "pending",
                "mode": "report",
                "input_key": input_key,
                "client_name": (name or "").strip(),
                "analysis_date": datetime.now().strftime("%Y-%m-%d"),
                "languages": report_pdf_languages,
                "output_prefix": f"{JOBS_GROUP_PREFIX}{group_id}",
                "analysis_mode": "real",
                "sample_fps": 3,
                "max_frames": 150,
                "report_style": "operation_test",
                "report_format": "pdf",
                "notify_email": notify_email.strip(),
                "enterprise_folder": "operation_test",
                "expect_dots": enable_dots,
                "expect_skeleton": enable_skeleton,
            }
            queued_job_keys.append(enqueue_job(report_job))
            queued.append("report_pdf")
        if report_docx_languages:
            report_job = {
                "job_id": new_job_id(),
                "group_id": group_id,
                "created_at": created_at,
                "status": "pending",
                "mode": "report",
                "input_key": input_key,
                "client_name": (name or "").strip(),
                "analysis_date": datetime.now().strftime("%Y-%m-%d"),
                "languages": report_docx_languages,
                "output_prefix": f"{JOBS_GROUP_PREFIX}{group_id}",
                "analysis_mode": "real",
                "sample_fps": 3,
                "max_frames": 150,
                "report_style": "operation_test",
                "report_format": "docx",
                "notify_email": notify_email.strip(),
                "enterprise_folder": "operation_test",
                "expect_dots": enable_dots,
                "expect_skeleton": enable_skeleton,
            }
            queued_job_keys.append(enqueue_job(report_job))
            queued.append("report_docx")
        # HTML can be selected standalone. If PDF or DOCX report job is already queued,
        # HTML will be produced from that job and no extra enqueue is needed.
        if report_html_languages and (not report_pdf_languages) and (not report_docx_languages):
            report_job = {
                "job_id": new_job_id(),
                "group_id": group_id,
                "created_at": created_at,
                "status": "pending",
                "mode": "report",
                "input_key": input_key,
                "client_name": (name or "").strip(),
                "analysis_date": datetime.now().strftime("%Y-%m-%d"),
                "languages": report_html_languages,
                "output_prefix": f"{JOBS_GROUP_PREFIX}{group_id}",
                "analysis_mode": "real",
                "sample_fps": 3,
                "max_frames": 150,
                "report_style": "operation_test",
                "report_format": "html",
                "notify_email": notify_email.strip(),
                "enterprise_folder": "operation_test",
                "expect_dots": enable_dots,
                "expect_skeleton": enable_skeleton,
            }
            queued_job_keys.append(enqueue_job(report_job))
            queued.append("report_html")
    except Exception as e:
        notice.error(f"Enqueue job failed: {format_submit_error_message(e)}")
        st.warning(SUPPORT_CONTACT_TEXT)
        st.stop()

    missing_pending = verify_pending_jobs_exist(queued_job_keys)
    if missing_pending:
        notice.error(
            "Queued job verification failed: some pending job files were not found in S3.\n"
            f"Missing keys: {', '.join(missing_pending)}"
        )
        st.warning(SUPPORT_CONTACT_TEXT)
        st.stop()

    st.session_state["operation_test_group_id"] = group_id
    st.session_state["operation_test_upload_nonce"] = int(st.session_state.get("operation_test_upload_nonce") or 0) + 1
    st.session_state["operation_test_queued"] = queued
    st.session_state["operation_test_queued_keys"] = queued_job_keys
    persist_group_id_to_url(group_id)
    notice.success(f"Queued: {', '.join(queued)}. group_id = {group_id}")
    st.caption(f"Queued job keys: {', '.join(queued_job_keys)}")
    st.rerun()

active_group_id = manual_group_id or st.session_state.get("operation_test_group_id") or read_group_id_from_url()
if active_group_id:
    persist_group_id_to_url(active_group_id)
    st.session_state["operation_test_group_id"] = active_group_id
    st.divider()
    st.subheader("Operational Test Result")
    st.caption(f"Group: `{active_group_id}`")

    latest_job = get_latest_operation_test_job(active_group_id)
    queued_funcs = st.session_state.get("operation_test_queued") or []
    has_report_queued = "report_pdf" in queued_funcs or "report_docx" in queued_funcs
    has_video_queued = "dots" in queued_funcs or "skeleton" in queued_funcs

    if not latest_job and not has_video_queued:
        st.info("No report job found for this group. If you ran dots/skeleton only, check below.")

    if latest_job:
        job_key = str(latest_job.get("_job_bucket_key") or "")
        status = get_job_status_from_key(job_key)
        st.markdown(f"**Report job status:** `{status}`")
    if st.button("Refresh Result", width="content"):
        st.rerun()
    notification = (latest_job or {}).get("notification") or {}
    notification_status = str(notification.get("status") or "").strip()
    if notification_status:
        st.caption(f"Email status: `{notification_status}`")

    summary = (latest_job or {}).get("first_impression_summary") or {}
    if isinstance(summary, dict) and summary:
        st.markdown("### First Impression (ต่ำ / กลาง / สูง)")
        c1, c2, c3 = st.columns(3)
        eye = summary.get("eye_contact") or {}
        up = summary.get("uprightness") or {}
        stance = summary.get("stance") or {}
        c1.metric("Eye Contact", prettify_level(eye.get("level")))
        c2.metric("Uprightness", prettify_level(up.get("level")))
        c3.metric("Stance", prettify_level(stance.get("level")))

    pdf_keys = get_pdf_keys_from_job(latest_job or {})
    docx_keys = get_docx_keys_from_job(latest_job or {})
    html_keys = get_html_keys_from_job(latest_job or {})
    pdf_key = pdf_keys.get("TH") or pdf_keys.get("EN") or ""
    finished_job = get_latest_finished_operation_test_job(active_group_id)
    if finished_job:
        pdf_keys = get_pdf_keys_from_job(finished_job)
        docx_keys = get_docx_keys_from_job(finished_job)
        html_keys = get_html_keys_from_job(finished_job)
        pdf_key = pdf_key or pdf_keys.get("TH") or pdf_keys.get("EN") or ""
        notif = (finished_job.get("notification") or {})
        notif_status = str(notif.get("status") or "").strip()
        if notif_status:
            st.caption(f"Email status: `{notif_status}`")
    if not pdf_key:
        pdf_key = find_pdf_key_from_finished_jobs(active_group_id)
    if not docx_keys.get("TH") and not docx_keys.get("EN"):
        found_docx = find_docx_keys_from_finished_jobs(active_group_id)
        docx_keys = found_docx
    # Robust fallback: scan report artifacts directly in S3 by group prefix.
    scanned_keys = find_report_keys_in_s3(active_group_id)
    if not pdf_keys.get("TH") and scanned_keys.get("pdf_th"):
        pdf_keys["TH"] = str(scanned_keys.get("pdf_th") or "").strip()
    if not pdf_keys.get("EN") and scanned_keys.get("pdf_en"):
        pdf_keys["EN"] = str(scanned_keys.get("pdf_en") or "").strip()
    if not docx_keys.get("TH") and scanned_keys.get("docx_th"):
        docx_keys["TH"] = str(scanned_keys.get("docx_th") or "").strip()
    if not docx_keys.get("EN") and scanned_keys.get("docx_en"):
        docx_keys["EN"] = str(scanned_keys.get("docx_en") or "").strip()
    if not html_keys.get("TH") and scanned_keys.get("html_th"):
        html_keys["TH"] = str(scanned_keys.get("html_th") or "").strip()
    if not html_keys.get("EN") and scanned_keys.get("html_en"):
        html_keys["EN"] = str(scanned_keys.get("html_en") or "").strip()
    failed_job = get_latest_failed_operation_test_job(active_group_id)
    failed_error = str((failed_job or {}).get("error") or "").strip()
    if failed_error:
        st.error(f"Latest failed error: {failed_error}")
    if not pdf_key:
        pdf_key = find_latest_group_pdf_key(active_group_id)

    # Dots & Skeleton downloads
    queued_funcs = st.session_state.get("operation_test_queued") or []
    outputs = build_output_keys(active_group_id)
    found_videos = find_dots_skeleton_keys(active_group_id)
    dots_key = outputs.get("dots_video", "") or found_videos.get("dots_video", "")
    skel_key = outputs.get("skeleton_video", "") or found_videos.get("skeleton_video", "")
    if dots_key or skel_key:
        st.markdown("### Dots & Skeleton videos")
        if dots_key and s3_key_exists(dots_key):
            st.link_button(
                "Download Dots video",
                presigned_get_url(dots_key, expires=3600, filename="dots.mp4"),
                width="stretch",
            )
        elif "dots" in queued_funcs:
            st.caption("Dots: processing...")
        if skel_key and s3_key_exists(skel_key):
            st.link_button(
                "Download Skeleton video",
                presigned_get_url(skel_key, expires=3600, filename="skeleton.mp4"),
                width="stretch",
            )
        elif "skeleton" in queued_funcs:
            st.caption("Skeleton: processing...")
        st.divider()

    th_docx = (docx_keys.get("TH") or "").strip()
    en_docx = (docx_keys.get("EN") or "").strip()
    th_docx_ready = bool(th_docx) and s3_key_exists(th_docx)
    en_docx_ready = bool(en_docx) and s3_key_exists(en_docx)
    if th_docx_ready or en_docx_ready:
        st.markdown("### DOCX reports")
        if th_docx_ready:
            st.link_button(
                "Download Operational Test DOCX (TH)",
                presigned_get_url(th_docx, expires=3600, filename="operational_test_report_th.docx"),
                width="stretch",
            )
        if en_docx_ready:
            st.link_button(
                "Download Operational Test DOCX (EN)",
                presigned_get_url(en_docx, expires=3600, filename="operational_test_report_en.docx"),
                width="stretch",
            )
        st.divider()

    th_html_key = (html_keys.get("TH") or "").strip()
    en_html_key = (html_keys.get("EN") or "").strip()
    th_html_ready = bool(th_html_key) and s3_key_exists(th_html_key)
    en_html_ready = bool(en_html_key) and s3_key_exists(en_html_key)
    if th_html_ready or en_html_ready:
        st.markdown("### HTML reports (debug/preview)")
        if th_html_ready:
            st.link_button(
                "Download Operational Test HTML (TH)",
                presigned_get_url(th_html_key, expires=3600, filename="operational_test_report_th.html"),
                width="stretch",
            )
        if en_html_ready:
            st.link_button(
                "Download Operational Test HTML (EN)",
                presigned_get_url(en_html_key, expires=3600, filename="operational_test_report_en.html"),
                width="stretch",
            )
        st.divider()

    th_pdf_key = (pdf_keys.get("TH") or "").strip()
    en_pdf_key = (pdf_keys.get("EN") or "").strip()
    th_pdf_ready = bool(th_pdf_key) and s3_key_exists(th_pdf_key)
    en_pdf_ready = bool(en_pdf_key) and s3_key_exists(en_pdf_key)
    generic_pdf_ready = bool(pdf_key) and s3_key_exists(pdf_key)

    ready_now = []
    if dots_key and s3_key_exists(dots_key):
        ready_now.append(("Dots video", dots_key, "dots.mp4"))
    if skel_key and s3_key_exists(skel_key):
        ready_now.append(("Skeleton video", skel_key, "skeleton.mp4"))
    if th_docx_ready:
        ready_now.append(("Operational Test DOCX (TH)", th_docx, "operational_test_report_th.docx"))
    if en_docx_ready:
        ready_now.append(("Operational Test DOCX (EN)", en_docx, "operational_test_report_en.docx"))
    if th_pdf_ready:
        ready_now.append(("Operational Test PDF (TH)", th_pdf_key, "operational_test_report_th.pdf"))
    if en_pdf_ready:
        ready_now.append(("Operational Test PDF (EN)", en_pdf_key, "operational_test_report_en.pdf"))
    if generic_pdf_ready and (not th_pdf_ready) and (not en_pdf_ready):
        ready_now.append(("Operational Test PDF", pdf_key, "operational_test_report.pdf"))
    if th_html_ready:
        ready_now.append(("Operational Test HTML (TH)", th_html_key, "operational_test_report_th.html"))
    if en_html_ready:
        ready_now.append(("Operational Test HTML (EN)", en_html_key, "operational_test_report_en.html"))

    st.markdown("### Ready to Download Now")
    if st.button("Refresh output status", key="operation_test_refresh_ready_downloads", width="content"):
        st.rerun()
    if ready_now:
        for label, key, filename in ready_now:
            st.success(f"✅ {label} ready")
            st.link_button(label, presigned_get_url(key, expires=3600, filename=filename), width="stretch")
    else:
        st.info("No files are ready yet. Files will appear here immediately when each one is done.")
    st.caption("Each file appears as soon as it is ready. No need to wait for all outputs.")
    st.divider()

    if pdf_key and s3_key_exists(pdf_key):
        st.success("PDF is ready.")
        if th_pdf_ready:
            st.link_button(
                "Download Operational Test PDF (TH)",
                presigned_get_url(th_pdf_key, expires=3600, filename="operational_test_report_th.pdf"),
                width="stretch",
            )
            st.code(th_pdf_key, language="text")
        if en_pdf_ready:
            st.link_button(
                "Download Operational Test PDF (EN)",
                presigned_get_url(en_pdf_key, expires=3600, filename="operational_test_report_en.pdf"),
                width="stretch",
            )
            st.code(en_pdf_key, language="text")
        if (not th_pdf_ready) and (not en_pdf_ready):
            st.link_button(
                "Download Operational Test PDF",
                presigned_get_url(pdf_key, expires=3600, filename="operational_test_report.pdf"),
                width="stretch",
            )
            st.code(pdf_key, language="text")
    elif (latest_job and get_job_status_from_key(str(latest_job.get("_job_bucket_key") or "")) == "failed") or failed_error:
        st.error("Analysis failed. Please upload again and retry.")
    else:
        st.info("Processing. PDF is not ready yet. Refresh this page shortly.")
else:
    st.caption("Upload a video and click Analyze First Impression to start.")

