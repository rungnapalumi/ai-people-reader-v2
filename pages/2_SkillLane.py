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
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import streamlit as st
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config


# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="วิเคราะห์วิดีโอ", layout="wide")

THEME_CSS = """
<style>
:root {
  --bg-main: #2c2723;
  --bg-soft: #3a332d;
  --bg-card: #473e36;
  --text-main: #e6d9c8;
  --text-dim: #ccbda8;
  --accent: #c9a67a;
  --accent-strong: #b48d5f;
  --border: #6d5c4e;
}

.stApp {
  background: var(--bg-main);
  color: var(--text-main);
}

[data-testid="stSidebar"] {
  background: #28231f;
  border-right: 1px solid var(--border);
}

[data-testid="stSidebarNav"] a,
[data-testid="stSidebarNav"] button,
[data-testid="stSidebarNav"] a p,
[data-testid="stSidebarNav"] a span,
[data-testid="stSidebarNav"] button p,
[data-testid="stSidebarNav"] button span {
  color: #ffffff !important;
}

h1, h2, h3, h4, h5, h6 {
  color: #f0e4d4 !important;
}

p, label, span, div {
  color: var(--text-main);
}

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea {
  background: var(--bg-soft) !important;
  color: var(--text-main) !important;
  border: 1px solid var(--border) !important;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
  background: var(--bg-soft) !important;
  color: var(--text-main) !important;
  border: 1px solid var(--border) !important;
}

[data-testid="stFileUploader"] section {
  background: var(--bg-soft) !important;
  border: 1px dashed var(--border) !important;
}

[data-testid="stFileUploader"] section button {
  font-size: 0 !important;
}

[data-testid="stFileUploader"] section button::after {
  content: "Browse File";
  font-size: 1.1rem;
}

.stButton > button,
.stDownloadButton > button,
.stLinkButton > a {
  background: linear-gradient(180deg, var(--accent), var(--accent-strong)) !important;
  color: #231d17 !important;
  border: 0 !important;
  font-weight: 600 !important;
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--border);
  border-radius: 10px;
}

[data-testid="stAlert"] {
  background: var(--bg-card) !important;
  color: var(--text-main) !important;
  border: 1px solid var(--border) !important;
}

.stCaption {
  color: var(--text-dim) !important;
}
</style>
"""


def apply_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)

BANNER_PATH_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "top_banner.png"),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "banner.png"),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "Header.png"),
]


# -------------------------
# Env / S3
# -------------------------
AWS_BUCKET = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

if not AWS_BUCKET:
    st.error("Missing AWS_BUCKET (or S3_BUCKET) environment variable in Render.")
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

JOBS_OUTPUT_PREFIX = "jobs/output/"
JOBS_GROUP_PREFIX = "jobs/groups/"
ORG_SETTINGS_PREFIX = "jobs/config/organizations/"
EMPLOYEE_REGISTRY_PREFIX = "jobs/config/employees/"
TRAINING_VIDEOS = [
    {
        "title": "คลิปแนะนำการใช้ AI People Reader",
        "key": "Training/คลิปแนะนำการใช้ AI People Reader.mp4",
    },
    {
        "title": "คลิปแนะนำผู้สอนและการคิดค้น AI People Reader",
        "key": "Training/คลิปแนะนำผู้สอนและการคิดค้น AI People Reader.mp4",
    },
]


# -------------------------
# Helpers
# -------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_valid_email_format(value: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", (value or "").strip()))

def is_blocked_typo_domain(value: str) -> bool:
    email = (value or "").strip().lower()
    if "@" not in email:
        return False
    domain = email.split("@", 1)[1]
    blocked_domains = {
        "gmail.co",
        "gmai.com",
        "gnail.com",
        "yahoo.co",
        "yaho.com",
        "hotmail.co",
        "hotmai.com",
        "outlook.co",
    }
    return domain in blocked_domains


def render_top_banner() -> None:
    for path in BANNER_PATH_CANDIDATES:
        if os.path.exists(path):
            st.image(path, width="stretch")
            return


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

def s3_upload_stream(key: str, file_obj: Any, content_type: str) -> None:
    # Stream multipart upload instead of loading full file bytes into app memory.
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


def normalize_org_name(name: str) -> str:
    text = (name or "").strip().lower()
    if not text:
        return ""
    out = []
    for ch in text:
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        elif ch.isspace():
            out.append("_")
    normalized = "".join(out).strip("_")
    return normalized


def org_settings_key(org_name: str) -> str:
    org_id = normalize_org_name(org_name)
    return f"{ORG_SETTINGS_PREFIX}{org_id}.json"


def get_org_settings(org_name: str) -> Dict[str, str]:
    """Return admin-configured report defaults for organization, if any."""
    org_id = normalize_org_name(org_name)
    if not org_id:
        return {}
    payload = s3_read_json(org_settings_key(org_name)) or {}
    style = str(payload.get("report_style") or "").strip().lower()
    fmt = str(payload.get("report_format") or "").strip().lower()
    if style not in ("full", "simple") or fmt not in ("docx", "pdf"):
        return {}
    return {
        "organization_name": str(payload.get("organization_name") or org_name).strip(),
        "organization_id": org_id,
        "report_style": style,
        "report_format": fmt,
        "default_page": str(payload.get("default_page") or "").strip().lower(),
        "updated_at": str(payload.get("updated_at") or ""),
    }


def list_org_settings() -> list:
    rows = []
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=ORG_SETTINGS_PREFIX):
            for item in page.get("Contents", []):
                key = str(item.get("Key") or "")
                if not key.endswith(".json"):
                    continue
                payload = s3_read_json(key) or {}
                style = str(payload.get("report_style") or "").strip().lower()
                fmt = str(payload.get("report_format") or "").strip().lower()
                if style not in ("full", "simple") or fmt not in ("docx", "pdf"):
                    continue
                rows.append(payload)
    except Exception:
        return []
    return rows


def get_default_org_for_page(page_key: str) -> str:
    page_key = str(page_key or "").strip().lower()
    if not page_key:
        return ""
    settings = list_org_settings()
    # Prefer latest updated record when multiple organizations target same page.
    settings.sort(key=lambda x: str(x.get("updated_at") or ""), reverse=True)
    for s in settings:
        if str(s.get("default_page") or "").strip().lower() == page_key:
            return str(s.get("organization_name") or "").strip()
    return ""


def employee_registry_key(employee_id: str) -> str:
    emp = safe_slug(employee_id, fallback="")
    return f"{EMPLOYEE_REGISTRY_PREFIX}{emp}.json" if emp else ""


def save_employee_registry(
    employee_id: str,
    employee_email: str,
    employee_password: str,
    organization_name: str,
) -> None:
    key = employee_registry_key(employee_id)
    if not key:
        return
    payload = {
        "employee_id": (employee_id or "").strip(),
        "employee_email": (employee_email or "").strip(),
        "employee_password": (employee_password or "").strip(),
        "organization_name": (organization_name or "").strip(),
        "updated_at": utc_now_iso(),
    }
    s3_put_json(key, payload)


def build_output_keys(group_id: str) -> Dict[str, str]:
    base = f"{JOBS_OUTPUT_PREFIX}groups/{group_id}/"
    return {
        "dots_video": base + "dots.mp4",
        "skeleton_video": base + "skeleton.mp4",
        "report_en_docx": base + "report_en.docx",
        "report_th_docx": base + "report_th.docx",
        "report_en_pdf": base + "report_en.pdf",
        "report_th_pdf": base + "report_th.pdf",
        "debug_en": base + "debug_en.json",
        "debug_th": base + "debug_th.json",
    }

def enqueue_report_only_job(
    group_id: str,
    client_name: str,
    report_style: str = "full",
    report_format: str = "docx",
    enterprise_folder: str = "",
    notify_email: str = "",
) -> str:
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
        "languages": ["th"],
        "output_prefix": f"{JOBS_GROUP_PREFIX}{group_id}",
        "analysis_mode": "real",
        "sample_fps": 5,
        "max_frames": 300,
        "report_style": report_style,
        "report_format": report_format,
        "priority": 1,
        "enterprise_folder": (enterprise_folder or "").strip(),
        "notify_email": (notify_email or "").strip(),
    }
    return enqueue_legacy_job(job_report)

def get_report_style_for_group(group_id: str) -> str:
    """Best-effort lookup of previously requested report_style for this group."""
    try:
        prefixes = [JOBS_PENDING_PREFIX, JOBS_PROCESSING_PREFIX, JOBS_FINISHED_PREFIX, JOBS_FAILED_PREFIX]
        for prefix in prefixes:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=prefix):
                for item in page.get("Contents", []):
                    key = item["Key"]
                    if not key.endswith(".json"):
                        continue
                    job_data = s3_read_json(key) or {}
                    if (
                        job_data.get("group_id") == group_id
                        and str(job_data.get("mode") or "").strip().lower() in ("report", "report_th_en", "report_generator")
                    ):
                        style = str(job_data.get("report_style") or "").strip().lower()
                        if style in ("simple", "full"):
                            return style
    except Exception:
        pass
    return "full"

def get_report_format_for_group(group_id: str) -> str:
    """Best-effort lookup of previously requested report_format for this group."""
    try:
        prefixes = [JOBS_PENDING_PREFIX, JOBS_PROCESSING_PREFIX, JOBS_FINISHED_PREFIX, JOBS_FAILED_PREFIX]
        for prefix in prefixes:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=prefix):
                for item in page.get("Contents", []):
                    key = item["Key"]
                    if not key.endswith(".json"):
                        continue
                    job_data = s3_read_json(key) or {}
                    if (
                        job_data.get("group_id") == group_id
                        and str(job_data.get("mode") or "").strip().lower() in ("report", "report_th_en", "report_generator")
                    ):
                        fmt = str(job_data.get("report_format") or "").strip().lower()
                        if fmt in ("docx", "pdf"):
                            return fmt
    except Exception:
        pass
    return "docx"


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
                            en_pdf = reports.get("EN", {}).get("pdf_key", "")
                            th_pdf = reports.get("TH", {}).get("pdf_key", "")
                            if en_key:
                                found["report_en_docx"] = en_key
                            if th_key:
                                found["report_th_docx"] = th_key
                            if en_pdf:
                                found["report_en_pdf"] = en_pdf
                            if th_pdf:
                                found["report_th_pdf"] = th_pdf
                            if (found.get("report_en_docx") and found.get("report_th_docx")) or (found.get("report_en_pdf") and found.get("report_th_pdf")):
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
                            if scanned.get("report_en_pdf"):
                                found["report_en_pdf"] = scanned["report_en_pdf"]
                            if scanned.get("report_th_pdf"):
                                found["report_th_pdf"] = scanned["report_th_pdf"]
                            if (found.get("report_en_docx") and found.get("report_th_docx")) or (found.get("report_en_pdf") and found.get("report_th_pdf")):
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
        if scanned.get("report_en_pdf") and not found.get("report_en_pdf"):
            found["report_en_pdf"] = scanned["report_en_pdf"]
        if scanned.get("report_th_pdf") and not found.get("report_th_pdf"):
            found["report_th_pdf"] = scanned["report_th_pdf"]
        if (found.get("report_en_docx") and found.get("report_th_docx")) or (found.get("report_en_pdf") and found.get("report_th_pdf")):
            break

    return found


def get_report_notification_status(group_id: str) -> Dict[str, Any]:
    """Read latest report job notification status for this group."""
    latest_key = ""
    latest_job: Dict[str, Any] = {}
    try:
        prefixes = [JOBS_PENDING_PREFIX, JOBS_PROCESSING_PREFIX, JOBS_FINISHED_PREFIX, JOBS_FAILED_PREFIX]
        for prefix in prefixes:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=prefix):
                for item in page.get("Contents", []):
                    key = item["Key"]
                    if not key.endswith(".json"):
                        continue
                    job_data = s3_read_json(key) or {}
                    if (
                        job_data.get("group_id") == group_id
                        and str(job_data.get("mode") or "").strip().lower() in ("report", "report_th_en", "report_generator")
                    ):
                        if key > latest_key:
                            latest_key = key
                            latest_job = job_data
    except Exception:
        return {}

    if not latest_job:
        return {}

    notif = latest_job.get("notification") or {}
    return {
        "job_status": str(latest_job.get("status") or ""),
        "notify_email": str(notif.get("notify_email") or latest_job.get("notify_email") or ""),
        "sent": bool(notif.get("sent")),
        "status": str(notif.get("status") or ""),
        "updated_at": str(notif.get("updated_at") or latest_job.get("updated_at") or ""),
    }


def find_report_files_in_s3(prefix: str) -> Dict[str, str]:
    """Find report files by scanning S3 with prefix"""
    try:
        result = {"report_en_docx": "", "report_th_docx": "", "report_en_pdf": "", "report_th_pdf": ""}
        paginator = s3.get_paginator("list_objects_v2")
        
        for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if key.endswith(".docx"):
                    if "_EN.docx" in key:
                        result["report_en_docx"] = key
                    elif "_TH.docx" in key:
                        result["report_th_docx"] = key
                elif key.endswith(".pdf"):
                    if "_EN.pdf" in key:
                        result["report_en_pdf"] = key
                    elif "_TH.pdf" in key:
                        result["report_th_pdf"] = key
        
        return result
    except Exception:
        return {"report_en_docx": "", "report_th_docx": "", "report_en_pdf": "", "report_th_pdf": ""}


def ensure_session_defaults() -> None:
    if "last_group_id" not in st.session_state:
        st.session_state["last_group_id"] = ""
    if "last_outputs" not in st.session_state:
        st.session_state["last_outputs"] = {}
    if "last_jobs" not in st.session_state:
        st.session_state["last_jobs"] = {}
    if "last_job_json_keys" not in st.session_state:
        st.session_state["last_job_json_keys"] = {}

def _read_group_id_from_url() -> str:
    try:
        val = st.query_params.get("group_id", "")
        if isinstance(val, list):
            val = val[0] if val else ""
        return str(val or "").strip()
    except Exception:
        return ""

def _persist_group_id_to_url(group_id: str) -> None:
    gid = str(group_id or "").strip()
    try:
        if gid:
            st.query_params["group_id"] = gid
        elif "group_id" in st.query_params:
            del st.query_params["group_id"]
    except Exception:
        pass


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
apply_theme()
render_top_banner()

url_group_id = _read_group_id_from_url()
if url_group_id and not st.session_state.get("last_group_id"):
    st.session_state["last_group_id"] = url_group_id

st.markdown("# วิเคราะห์วิดีโอ")
st.caption("อัปโหลดวิดีโอ 1 ครั้ง แล้วกด **เริ่มวิเคราะห์** เพื่อสร้าง dots + skeleton + รายงาน")

page_default_org = get_default_org_for_page("ai_people_reader")
enterprise_folder = st.text_input(
    "ชื่อองค์กร",
    value=page_default_org,
    placeholder="เช่น TTB / ACME Group",
    disabled=bool(page_default_org),
)
if page_default_org:
    st.caption(f"ใช้องค์กรเริ่มต้นจากหน้าผู้ดูแลระบบ: {page_default_org}")
user_name = st.text_input(
    "อีเมลผู้ใช้งาน",
    value="",
    placeholder="name@example.com",
    help="ใช้เป็นชื่อโฟลเดอร์งาน และเป็นอีเมลสำหรับส่งผลลัพธ์",
)
st.caption("กรุณาตรวจสอบการพิมพ์ e-mail ให้ถูกต้องก่อนกดเริ่มวิเคราะห์")
notify_email = (user_name or "").strip()
if notify_email:
    if not is_valid_email_format(notify_email):
        st.warning("กรุณาตรวจสอบ e-mail ให้ถูกต้องอีกครั้ง (Please check your e-mail format).")
    elif is_blocked_typo_domain(notify_email):
        st.warning("รูปแบบโดเมนอีเมลอาจพิมพ์ผิด กรุณาตรวจสอบ e-mail อีกครั้ง (เช่น .com)")
employee_id = st.text_input(
    "ชื่อที่ใช้ในการรายงานผล",
    value="",
    placeholder="e.g., คุณสมชาย / Somchai",
)
employee_password = st.text_input(
    "รหัสผ่าน",
    value="",
    type="password",
    placeholder="กรอกรหัสผ่านผู้ใช้งาน",
)

org_settings = get_org_settings(enterprise_folder)

uploaded = st.file_uploader(
    "วิดีโอ (MP4/MOV/M4V/WEBM)",
    type=["mp4", "mov", "m4v", "webm"],
    accept_multiple_files=False,
)

run = st.button("🎬 เริ่มวิเคราะห์", type="primary", width="stretch")

st.markdown("## ข้อแนะนำในการอัดวีดีโอ")
st.markdown(
    """
1. วิดีโอควรเป็นแนวตั้ง เห็นเต็มตัวหัวจรดเท้า
2. กรุณาถ่ายวิดีโอเป็น **Full HD (1080 x 1920)**, format **MP4 or MOV**
3. กรุณาอย่าใส่สีขาว ดำ หรือสีเดียวกับผนัง
4. กรุณาเลือกยืนหน้าผนังสีอ่อน หรือในกรณีที่ผนังสีเข้ม คุณควรใส่เสื้อผ้าสีอ่อน
5. กรุณาเคลื่อนไหวและใช้ภาษามือเป็นธรรมชาติ
6. โปรดตรวจสอบให้แน่ใจว่าวิดีโอของคุณมีความยาวอย่างน้อย 2 นาที
"""
)

for training_video in TRAINING_VIDEOS:
    title = str(training_video.get("title") or "").strip()
    training_key = str(training_video.get("key") or "").strip()
    if title:
        st.markdown(f"### {title}")
    if training_key and s3_key_exists(training_key):
        st.video(presigned_get_url(training_key, expires=3600))
    else:
        st.warning(f"ไม่พบวิดีโอใน S3: {training_key}")

active_group_id = url_group_id or st.session_state.get("last_group_id", "")
if active_group_id:
    st.session_state["last_group_id"] = active_group_id
    _persist_group_id_to_url(active_group_id)

note = st.empty()

# -------------------------
# Submit jobs
# -------------------------
if run:
    if not uploaded:
        note.error("กรุณาอัปโหลดวิดีโอก่อน")
        st.stop()
    if not notify_email:
        note.error("กรุณากรอกอีเมลผู้ใช้งาน")
        st.stop()
    if (not is_valid_email_format(notify_email)) or is_blocked_typo_domain(notify_email):
        note.error("รูปแบบ e-mail ไม่ถูกต้อง กรุณาตรวจสอบ e-mail อีกครั้ง")
        st.stop()
    if not employee_id.strip():
        note.error("กรุณากรอกชื่อที่ใช้ในการรายงานผล")
        st.stop()
    if not employee_password.strip():
        note.error("กรุณากรอกรหัสผ่าน")
        st.stop()

    effective_report_style = org_settings.get("report_style") if org_settings else "full"
    effective_report_format = org_settings.get("report_format") if org_settings else "docx"

    base_user = safe_slug(user_name, fallback="user")
    group_id = f"{new_group_id()}__{base_user}"
    input_key = f"{JOBS_GROUP_PREFIX}{group_id}/input/input.mp4"

    try:
        s3_upload_stream(
            key=input_key,
            file_obj=uploaded,
            content_type=guess_content_type(uploaded.name or "input.mp4"),
        )
    except Exception as e:
        note.error(f"อัปโหลดไป S3 ไม่สำเร็จ: {e}")
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
        "employee_id": (employee_id or "").strip(),
        "employee_email": notify_email,
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
        "employee_id": (employee_id or "").strip(),
        "employee_email": notify_email,
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
        "languages": ["th"],
        "output_prefix": f"{JOBS_GROUP_PREFIX}{group_id}",
        "analysis_mode": "real",  # Use real MediaPipe analysis
        "sample_fps": 5,
        "max_frames": 300,
        "report_style": effective_report_style,
        "report_format": effective_report_format,
        "notify_email": notify_email,
        "enterprise_folder": (enterprise_folder or "").strip(),
        "employee_id": (employee_id or "").strip(),
        "employee_email": notify_email,
    }

    try:
        save_employee_registry(
            employee_id=employee_id,
            employee_email=notify_email,
            employee_password=employee_password,
            organization_name=enterprise_folder,
        )
        k1 = enqueue_legacy_job(job_dots)
        k2 = enqueue_legacy_job(job_skel)
        k3 = enqueue_legacy_job(job_report)
    except Exception as e:
        note.error(f"ส่งงานเข้าคิวไม่สำเร็จ: {e}")
        st.stop()

    st.session_state["last_group_id"] = group_id
    _persist_group_id_to_url(group_id)
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

    note.success(
        f"ส่งงานเรียบร้อย! group_id = {group_id} | report_style={effective_report_style}, report_format={effective_report_format}"
    )
    st.info("เริ่มประมวลผลแล้ว สามารถอยู่หน้านี้เพื่อรอติดตามสถานะได้")


# -------------------------
# Download section
# -------------------------
st.divider()
st.subheader("ผลลัพธ์สำหรับดาวน์โหลด")

group_id = active_group_id
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
    st.caption("ยังไม่มี group_id กรุณาอัปโหลดวิดีโอแล้วกด **เริ่มวิเคราะห์**")
    st.stop()

st.caption(f"กลุ่มงาน: `{group_id}`")


def download_block(title: str, key: str, filename: str) -> None:
    if not key:
        st.write(f"- {title}: (ยังไม่มี key)")
        return
    ready = s3_key_exists(key)
    if ready:
        url = presigned_get_url(key, expires=3600, filename=filename)
        st.success(f"✅ {title} พร้อมดาวน์โหลด")
        st.link_button(f"ดาวน์โหลด {title}", url, width="stretch")
        st.code(key, language="text")
    else:
        st.warning(f"⏳ {title} ยังไม่พร้อม")
        st.code(key, language="text")


# --- Downloads ---
c1, c2 = st.columns(2)

with c1:
    st.markdown("### วิดีโอ")
    download_block("วิดีโอ Dots", outputs.get("dots_video", ""), "dots.mp4")
    download_block("วิดีโอ Skeleton", outputs.get("skeleton_video", ""), "skeleton.mp4")

with c2:
    st.markdown("### รายงาน")
    selected_pdf = (get_report_format_for_group(group_id) == "pdf")
    en_key = outputs.get("report_en_pdf", "") if selected_pdf else outputs.get("report_en_docx", "")
    th_key = outputs.get("report_th_pdf", "") if selected_pdf else outputs.get("report_th_docx", "")
    en_name = "report_en.pdf" if selected_pdf else "report_en.docx"
    th_name = "report_th.pdf" if selected_pdf else "report_th.docx"
    report_label = "PDF" if selected_pdf else "DOCX"
    st.markdown("**ภาษาอังกฤษ**")
    download_block(f"รายงาน EN ({report_label})", en_key, en_name)

    st.markdown("**ภาษาไทย**")
    download_block(f"รายงาน TH ({report_label})", th_key, th_name)

reports_ready = bool(en_key) and bool(th_key) and s3_key_exists(en_key) and s3_key_exists(th_key)
videos_ready = bool(outputs.get("dots_video")) and bool(outputs.get("skeleton_video")) and s3_key_exists(outputs.get("dots_video", "")) and s3_key_exists(outputs.get("skeleton_video", ""))
dots_ready = bool(outputs.get("dots_video")) and s3_key_exists(outputs.get("dots_video", ""))
skeleton_ready = bool(outputs.get("skeleton_video")) and s3_key_exists(outputs.get("skeleton_video", ""))
en_report_ready = bool(en_key) and s3_key_exists(en_key)
th_report_ready = bool(th_key) and s3_key_exists(th_key)
primary_done = th_report_ready

st.divider()
st.subheader("สถานะการประมวลผล")
status_items = [
    ("รายงาน TH (หลัก)", th_report_ready),
    ("รายงาน EN (ตามหลัง)", en_report_ready),
    ("วิดีโอ Dots (ตามหลัง)", dots_ready),
    ("วิดีโอ Skeleton", skeleton_ready),
]
if primary_done:
    overall_pct = 100
else:
    overall_pct = int(round((sum(1 for _, ready in status_items if ready) / len(status_items)) * 100))
st.progress(overall_pct, text=f"ความคืบหน้าโดยรวม: {overall_pct}%")
for label, ready in status_items:
    item_pct = 100 if ready else 0
    st.progress(item_pct, text=f"{label}: {'พร้อมแล้ว' if ready else 'กำลังประมวลผล'} ({item_pct}%)")

# Clear step guidance for users while waiting.
if th_report_ready and dots_ready and skeleton_ready and en_report_ready:
    current_step = "ผลลัพธ์ทั้งหมดพร้อมแล้ว"
    next_step = "ดาวน์โหลดวิดีโอ/รายงานด้านล่างได้เลย ระบบจะส่งอีเมลครบในไม่ช้า"
elif th_report_ready:
    current_step = "ผลลัพธ์หลักพร้อมแล้ว: รายงานภาษาไทย"
    next_step = "งานเสร็จแล้ว รายงาน EN และ dots จะส่งตามทางอีเมลภายหลัง"
else:
    current_step = "กำลังสร้างรายงานภาษาไทย"
    next_step = "กรุณารอให้รายงานภาษาไทยเสร็จ (หลัก)"

st.info(f"ขั้นตอนปัจจุบัน: {current_step}")
st.caption(f"ขั้นตอนถัดไป: {next_step}")
if th_report_ready:
    st.success("ขอบคุณที่ใช้ AI People Reader การวิเคราะห์ทั้งหมดจะถูกส่งไปในเมล์ของคุณหลังจากนี้")

if videos_ready and not th_report_ready:
    st.divider()
    st.warning("รายงานภาษาไทยยังไม่พร้อม คุณสามารถสั่งสร้างรายงานใหม่สำหรับกลุ่มนี้ได้")
    if st.button("สั่งสร้างรายงานใหม่", width="content"):
        try:
            guessed_name = group_id.split("__", 1)[1] if "__" in group_id else "Anonymous"
            rerun_style = get_report_style_for_group(group_id)
            rerun_format = get_report_format_for_group(group_id)
            rerun_org_cfg = get_org_settings(enterprise_folder)
            if rerun_org_cfg:
                rerun_style = str(rerun_org_cfg.get("report_style") or rerun_style)
                rerun_format = str(rerun_org_cfg.get("report_format") or rerun_format)
            rerun_email = notify_email
            if not rerun_email:
                prev_notif = get_report_notification_status(group_id)
                rerun_email = str(prev_notif.get("notify_email") or "").strip()
            if rerun_email and ((not is_valid_email_format(rerun_email)) or is_blocked_typo_domain(rerun_email)):
                st.error("ไม่สามารถส่งงานสร้างรายงานใหม่ได้: รูปแบบ e-mail ไม่ถูกต้อง")
                st.stop()
            new_report_key = enqueue_report_only_job(
                group_id=group_id,
                client_name=guessed_name,
                report_style=rerun_style,
                report_format=rerun_format,
                enterprise_folder=(enterprise_folder or "").strip(),
                notify_email=rerun_email,
            )
            st.success(f"ส่งงานสร้างรายงานใหม่เข้าคิวแล้ว ({rerun_style}, {rerun_format}): {new_report_key}")
        except Exception as e:
            st.error(f"ไม่สามารถส่งงานสร้างรายงานใหม่เข้าคิวได้: {e}")

notification = get_report_notification_status(group_id)
if notification:
    st.divider()
    st.subheader("สถานะอีเมล")
    email_to = notification.get("notify_email", "")
    status = notification.get("status", "")
    if notification.get("sent"):
        st.success(f"ส่งอีเมลแล้วไปที่: {email_to}")
    elif status == "waiting_for_all_outputs":
        st.info(f"อีเมลอยู่ในคิว: รอผลลัพธ์ครบทั้งหมด (ปลายทาง: {email_to})")
    elif status in ("sending", "queued"):
        st.info(f"กำลังส่งอีเมล... (ปลายทาง: {email_to})")
    elif status == "skipped_no_notify_email":
        st.caption("งานนี้ไม่ได้ระบุอีเมลสำหรับแจ้งผล")
    elif status == "disabled_by_config":
        st.caption("ระบบปิดการส่งอีเมลไว้ตามการตั้งค่า")
    elif status:
        st.warning(f"สถานะอีเมล: {status} (ปลายทาง: {email_to})")

st.caption("สถานะจะอัปเดตอัตโนมัติ แนะนำให้เปิดหน้านี้ไว้เพื่อติดตามความคืบหน้า")
