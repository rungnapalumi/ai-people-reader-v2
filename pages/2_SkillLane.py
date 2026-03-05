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
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import streamlit as st
import streamlit.components.v1 as components
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

SUPPORT_CONTACT_TEXT = "หากพบปัญหากรุณาติดต่อ 0817008484"


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
  color: #ffffff !important;
  background: #4a4038 !important;
  border: 1px solid var(--border) !important;
}

[data-testid="stFileUploader"] section button::after {
  content: "Browse File";
  font-size: 1.1rem;
  color: #ffffff !important;
}

[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"],
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] div {
  color: #ffffff !important;
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

def format_submit_error_message(err: Exception) -> str:
    message = str(err or "").strip() or "Unknown error"
    lowered = message.lower()
    if "axioserror" in lowered or "status code 400" in lowered:
        return f"{message}\n\nกรุณา upload วีดีโอใหม่"
    return message

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


def presigned_put_url(key: str, content_type: str = "video/mp4", expires: int = 3600) -> str:
    """Generate presigned PUT URL for direct browser-to-S3 upload (skips server)."""
    return s3.generate_presigned_url(
        ClientMethod="put_object",
        Params={
            "Bucket": AWS_BUCKET,
            "Key": key,
            "ContentType": content_type,
        },
        ExpiresIn=expires,
    )


def enqueue_legacy_job(job: Dict[str, Any]) -> str:
    job_id = str(job["job_id"])
    job_json_key = f"{JOBS_PENDING_PREFIX}{job_id}.json"
    s3_put_json(job_json_key, job)
    return job_json_key


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


def get_org_settings(org_name: str) -> Dict[str, Any]:
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
        "enable_report_th": bool(payload.get("enable_report_th", True)),
        "enable_report_en": bool(payload.get("enable_report_en", True)),
        "enable_skeleton": bool(payload.get("enable_skeleton", True)),
        "enable_dots": bool(payload.get("enable_dots", True)),
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
    organization_name: str,
) -> None:
    key = employee_registry_key(employee_id)
    if not key:
        return
    payload = {
        "employee_id": (employee_id or "").strip(),
        "employee_email": (employee_email or "").strip(),
        "organization_name": (organization_name or "").strip(),
        "updated_at": utc_now_iso(),
    }
    s3_put_json(key, payload)


def normalize_email(value: Any) -> str:
    return str(value or "").strip().lower()


def get_employee_registry(employee_id: str) -> Dict[str, str]:
    key = employee_registry_key(employee_id)
    if not key:
        return {}
    payload = s3_read_json(key) or {}
    return {
        "employee_id": str(payload.get("employee_id") or "").strip(),
        "employee_email": normalize_email(payload.get("employee_email")),
    }


def is_employee_identity_verified(employee_id: str, employee_email: str) -> bool:
    reg = get_employee_registry(employee_id)
    if not reg:
        return False
    return normalize_email(employee_email) == reg.get("employee_email", "")


def is_group_owned_by_employee(group_id: str, employee_id: str, employee_email: str) -> bool:
    gid = str(group_id or "").strip()
    eid = str(employee_id or "").strip().lower()
    eml = normalize_email(employee_email)
    if not gid or not eid or not eml:
        return False

    prefixes = [JOBS_PENDING_PREFIX, JOBS_PROCESSING_PREFIX, JOBS_FINISHED_PREFIX, JOBS_FAILED_PREFIX]
    try:
        for prefix in prefixes:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=prefix):
                for item in page.get("Contents", []):
                    key = str(item.get("Key") or "")
                    if not key.endswith(".json"):
                        continue
                    job_data = s3_read_json(key) or {}
                    if str(job_data.get("group_id") or "").strip() != gid:
                        continue
                    job_eid = str(job_data.get("employee_id") or "").strip().lower()
                    job_eml = normalize_email(job_data.get("employee_email") or job_data.get("notify_email"))
                    if job_eid == eid and job_eml == eml:
                        return True
    except Exception:
        return False
    return False


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
        "expect_skeleton": False,
        "expect_dots": False,
        "priority": 1,
        "enterprise_folder": (enterprise_folder or "").strip(),
        "notify_email": (notify_email or "").strip(),
    }
    return enqueue_legacy_job(job_report)


def enqueue_video_only_job(
    group_id: str,
    mode: str,
    notify_email: str = "",
    employee_id: str = "",
    enterprise_folder: str = "SkillLane",
) -> str:
    mode_norm = str(mode or "").strip().lower()
    if mode_norm not in ("dots", "skeleton"):
        raise ValueError("mode must be 'dots' or 'skeleton'")
    input_key = f"{JOBS_GROUP_PREFIX}{group_id}/input/input.mp4"
    if not s3_key_exists(input_key):
        raise RuntimeError(f"Input video not found for group_id={group_id}")
    output_key = f"{JOBS_OUTPUT_PREFIX}groups/{group_id}/{mode_norm}.mp4"
    job = {
        "job_id": new_job_id(),
        "group_id": group_id,
        "created_at": utc_now_iso(),
        "status": "pending",
        "mode": mode_norm,
        "input_key": input_key,
        "output_key": output_key,
        "notify_email": (notify_email or "").strip(),
        "enterprise_folder": (enterprise_folder or "").strip() or "SkillLane",
        "employee_id": (employee_id or "").strip(),
        "employee_email": (notify_email or "").strip(),
    }
    return enqueue_legacy_job(job)

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
                    if (
                        job_data.get("group_id") == group_id
                        and str(job_data.get("mode") or "").strip().lower() in ("report", "report_th_en", "report_generator")
                    ):
                        
                        # Extract output paths from job
                        outputs = job_data.get("outputs", {})
                        reports = outputs.get("reports", {})
                        
                        # If outputs exist, return them
                        if reports:
                            en_key = reports.get("EN", {}).get("docx_key", "")
                            th_key = reports.get("TH", {}).get("docx_key", "")
                            en_pdf = reports.get("EN", {}).get("pdf_key", "")
                            th_pdf = reports.get("TH", {}).get("pdf_key", "")
                            en_html = reports.get("EN", {}).get("html_key", "")
                            th_html = reports.get("TH", {}).get("html_key", "")
                            if en_key:
                                found["report_en_docx"] = en_key
                            if th_key:
                                found["report_th_docx"] = th_key
                            if en_pdf:
                                found["report_en_pdf"] = en_pdf
                            if th_pdf:
                                found["report_th_pdf"] = th_pdf
                            if en_html:
                                found["report_en_html"] = en_html
                            if th_html:
                                found["report_th_html"] = th_html
                            # Keep scanning to also collect HTML/PDF variants when available.
                        
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
                            if scanned.get("report_en_html"):
                                found["report_en_html"] = scanned["report_en_html"]
                            if scanned.get("report_th_html"):
                                found["report_th_html"] = scanned["report_th_html"]
                            # Keep scanning to also collect HTML/PDF variants when available.
            
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
        if scanned.get("report_en_html") and not found.get("report_en_html"):
            found["report_en_html"] = scanned["report_en_html"]
        if scanned.get("report_th_html") and not found.get("report_th_html"):
            found["report_th_html"] = scanned["report_th_html"]
        # Keep scanning both prefixes to collect all report variants.

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
        result = {
            "report_en_docx": "",
            "report_th_docx": "",
            "report_en_pdf": "",
            "report_th_pdf": "",
            "report_en_html": "",
            "report_th_html": "",
        }
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
                elif key.endswith(".html"):
                    if "_EN.html" in key:
                        result["report_en_html"] = key
                    elif "_TH.html" in key:
                        result["report_th_html"] = key
        
        return result
    except Exception:
        return {
            "report_en_docx": "",
            "report_th_docx": "",
            "report_en_pdf": "",
            "report_th_pdf": "",
            "report_en_html": "",
            "report_th_html": "",
        }


def _direct_upload_html(
    presigned_url: str,
    group_id: str,
    notify_email: str,
    employee_id: str,
    content_type: str = "video/mp4",
) -> str:
    """HTML/JS for direct browser-to-S3 upload with progress bar."""
    redirect_params = (
        f"group_id={quote(group_id)}"
        f"&upload_done=1"
        f"&notify_email={quote(notify_email)}"
        f"&employee_id={quote(employee_id)}"
    )
    return f"""
<div style="font-family: inherit; color: #e6d9c8;">
  <p style="margin-bottom: 12px;">เลือกวิดีโอเพื่ออัปโหลดตรงไปยัง S3 (ไม่ผ่านเซิร์ฟเวอร์ — เร็วกว่า)</p>
  <input type="file" id="videoInput" accept="video/mp4,video/quicktime,video/x-m4v,video/webm,.mp4,.mov,.m4v,.webm" style="margin-bottom: 12px; color: #e6d9c8;" />
  <div id="progressWrap" style="display: none; margin: 12px 0;">
    <div style="background: #3a332d; border-radius: 8px; overflow: hidden; height: 24px;">
      <div id="progressBar" style="background: linear-gradient(180deg, #c9a67a, #b48d5f); height: 100%; width: 0%%; transition: width 0.2s;"></div>
    </div>
    <p id="progressText" style="margin-top: 6px; font-size: 14px;">0%%</p>
  </div>
  <p id="statusText" style="margin-top: 8px; font-size: 14px;"></p>
</div>
<script>
(function() {{
  var input = document.getElementById('videoInput');
  var progressWrap = document.getElementById('progressWrap');
  var progressBar = document.getElementById('progressBar');
  var progressText = document.getElementById('progressText');
  var statusText = document.getElementById('statusText');

  input.addEventListener('change', function() {{
    var file = input.files[0];
    if (!file) return;

    progressWrap.style.display = 'block';
    progressBar.style.width = '0%';
    progressText.textContent = '0%';
    statusText.textContent = 'กำลังอัปโหลด...';

    var xhr = new XMLHttpRequest();
    xhr.open('PUT', '{presigned_url}');
    xhr.setRequestHeader('Content-Type', '{content_type}');

    xhr.upload.addEventListener('progress', function(e) {{
      if (e.lengthComputable) {{
        var pct = Math.round((e.loaded / e.total) * 100);
        progressBar.style.width = pct + '%';
        progressText.textContent = pct + '%';
      }}
    }});

    xhr.onload = function() {{
      if (xhr.status >= 200 && xhr.status < 300) {{
        progressBar.style.width = '100%';
        progressText.textContent = '100%';
        statusText.textContent = 'อัปโหลดสำเร็จ! กำลังส่งงาน...';
        var qs = '{redirect_params}' + '&uploaded_file=' + encodeURIComponent(file.name || 'input.mp4');
        // Always return to app root to avoid broken /SkillLane subpath requests.
        window.top.location.href = window.top.location.origin + '/?' + qs;
      }} else {{
        statusText.textContent = 'อัปโหลดล้มเหลว (รหัส: ' + xhr.status + ')';
        statusText.style.color = '#e74c3c';
      }}
    }};

    xhr.onerror = function() {{
      statusText.textContent = 'เกิดข้อผิดพลาดในการอัปโหลด';
      statusText.style.color = '#e74c3c';
    }};

    xhr.send(file);
  }});
}})();
</script>
"""


def ensure_session_defaults() -> None:
    if "last_group_id" not in st.session_state:
        st.session_state["last_group_id"] = ""
    if "last_outputs" not in st.session_state:
        st.session_state["last_outputs"] = {}
    if "last_jobs" not in st.session_state:
        st.session_state["last_jobs"] = {}
    if "last_job_json_keys" not in st.session_state:
        st.session_state["last_job_json_keys"] = {}
    if "last_notify_email" not in st.session_state:
        st.session_state["last_notify_email"] = ""
    if "skilllane_submission_id_override" not in st.session_state:
        st.session_state["skilllane_submission_id_override"] = ""

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

def list_jobs_for_group(group_id: str) -> list:
    gid = str(group_id or "").strip()
    if not gid:
        return []
    rows = []
    prefixes = [JOBS_PENDING_PREFIX, JOBS_PROCESSING_PREFIX, JOBS_FINISHED_PREFIX, JOBS_FAILED_PREFIX]
    try:
        for prefix in prefixes:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=prefix):
                for item in page.get("Contents", []):
                    key = str(item.get("Key") or "")
                    if not key.endswith(".json"):
                        continue
                    job_data = s3_read_json(key) or {}
                    if str(job_data.get("group_id") or "").strip() != gid:
                        continue
                    rows.append(
                        {
                            "status": infer_job_bucket_status(key),
                            "mode": str(job_data.get("mode") or "").strip() or "-",
                            "job_id": str(job_data.get("job_id") or "").strip() or "-",
                            "job_key": key,
                            "updated_at": str(job_data.get("updated_at") or job_data.get("created_at") or "").strip() or "-",
                        }
                    )
    except Exception:
        return []
    rows.sort(key=lambda x: (str(x.get("updated_at") or ""), str(x.get("job_key") or "")), reverse=True)
    return rows


# -------------------------
# UI
# -------------------------
ensure_session_defaults()
apply_theme()
render_top_banner()
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

url_group_id = _read_group_id_from_url()
if url_group_id and not st.session_state.get("last_group_id"):
    st.session_state["last_group_id"] = url_group_id

st.markdown("# วิเคราะห์วิดีโอ")
st.caption("อัปโหลดวิดีโอ 1 ครั้ง แล้วกด **เริ่มวิเคราะห์** เพื่อสร้าง dots + skeleton + รายงาน")

page_default_org = "SkillLane"
enterprise_folder = st.text_input(
    "ชื่อองค์กร",
    value=page_default_org,
    placeholder="เช่น TTB / ACME Group",
    disabled=True,
)
st.caption("หน้านี้กำหนดชื่อองค์กรเป็น SkillLane อัตโนมัติและไม่สามารถแก้ไขได้")
user_name = st.text_input(
    "อีเมลผู้ใช้งาน",
    value=str(st.session_state.get("last_notify_email") or ""),
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
# Email-only flow: reuse email as stable identity key.
employee_id = notify_email
if notify_email:
    st.session_state["last_notify_email"] = notify_email
org_settings = get_org_settings(enterprise_folder)

# Read direct-upload callback flags before rendering upload widget.
upload_done = str(st.query_params.get("upload_done", "") or "").strip() == "1"
manual_direct_done = bool(st.session_state.pop("direct_upload_done_manual", False))
direct_ready = bool(st.session_state.get("direct_upload_ready"))
last_group_hint = str(st.session_state.get("last_group_id") or "").strip()

# Local-safe upload mode:
# - auto: enable direct upload on Render, disable on local/dev by default.
# - on/off: force behavior via env.
direct_upload_mode = str(os.getenv("SKILLLANE_DIRECT_UPLOAD_MODE", "auto") or "auto").strip().lower()
if direct_upload_mode in ("on", "true", "1", "yes"):
    use_direct_upload = True
elif direct_upload_mode in ("off", "false", "0", "no"):
    use_direct_upload = False
else:
    is_render_runtime = bool(
        os.getenv("RENDER")
        or os.getenv("RENDER_SERVICE_ID")
        or os.getenv("RENDER_EXTERNAL_URL")
    )
    use_direct_upload = is_render_runtime

# If direct mode is disabled (e.g., local), clear stale direct-upload state.
if not use_direct_upload:
    for _k in (
        "direct_upload_ready",
        "direct_upload_presigned_url",
        "direct_upload_group_id",
        "direct_upload_input_key",
        "direct_upload_notify_email",
        "direct_upload_employee_id",
        "direct_upload_enterprise_folder",
        "direct_upload_user_name",
    ):
        st.session_state.pop(_k, None)
    direct_ready = False
    upload_done = False
    manual_direct_done = False

# Auto-recovery: if browser callback/redirect fails but file is already in S3,
# continue enqueue flow automatically from session state.
if direct_ready and (not upload_done) and (not manual_direct_done):
    pending_input_key = str(st.session_state.get("direct_upload_input_key") or "").strip()
    if pending_input_key:
        try:
            if s3_key_exists(pending_input_key):
                manual_direct_done = True
        except Exception:
            pass

# Always show visible status so users know current step.
st.markdown("### สถานะการอัปโหลด/ส่งงาน")
if st.button("🔄 รีเฟรชสถานะผลลัพธ์", key="refresh_status_top", width="content"):
    st.rerun()
submission_id_override = st.text_input(
    "Submission ID (วางจากอีเมลได้)",
    value=str(st.session_state.get("skilllane_submission_id_override") or ""),
    placeholder="เช่น 20260305_062908_93a975__rungnapaimagemattersat",
    help="หากปุ่มดาวน์โหลดยังไม่ขึ้น ให้วาง Submission ID จากอีเมล แล้วระบบจะดึงผลลัพธ์ตามรหัสนี้",
).strip()
if submission_id_override:
    st.session_state["skilllane_submission_id_override"] = submission_id_override
    st.session_state["last_group_id"] = submission_id_override
    _persist_group_id_to_url(submission_id_override)

# Always-visible direct download by Submission ID (independent from session/state blocks).
direct_group_id = str(submission_id_override or _read_group_id_from_url() or st.session_state.get("last_group_id") or "").strip()
st.markdown("#### Direct download by Submission ID")
if direct_group_id:
    st.caption(f"กำลังตรวจไฟล์จาก Submission ID: `{direct_group_id}`")
    direct_outputs_top = build_output_keys(direct_group_id)
    direct_reports_top = find_report_files_in_s3(f"{JOBS_GROUP_PREFIX}{direct_group_id}")
    if not direct_reports_top.get("report_en_docx") and not direct_reports_top.get("report_th_docx"):
        direct_reports_top = find_report_files_in_s3(f"{JOBS_OUTPUT_PREFIX}groups/{direct_group_id}")
    for _k in ("report_en_docx", "report_th_docx", "report_en_pdf", "report_th_pdf", "report_en_html", "report_th_html"):
        if direct_reports_top.get(_k):
            direct_outputs_top[_k] = str(direct_reports_top.get(_k) or "").strip()
    direct_items_top = [
        ("Dots Video", str(direct_outputs_top.get("dots_video") or "").strip(), "dots.mp4"),
        ("Skeleton Video", str(direct_outputs_top.get("skeleton_video") or "").strip(), "skeleton.mp4"),
        ("Report TH (PDF)", str(direct_outputs_top.get("report_th_pdf") or "").strip(), "report_th.pdf"),
        ("Report EN (PDF)", str(direct_outputs_top.get("report_en_pdf") or "").strip(), "report_en.pdf"),
    ]
    # Extra-robust fallback: list actual files under group prefixes and show all supported outputs.
    discovered_direct_items: List[tuple[str, str, str]] = []
    discovered_seen: set[str] = set()
    for _pfx in (f"{JOBS_GROUP_PREFIX}{direct_group_id}/", f"{JOBS_OUTPUT_PREFIX}groups/{direct_group_id}/"):
        try:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=_pfx):
                for item in page.get("Contents", []):
                    k = str(item.get("Key") or "").strip()
                    if not k or k in discovered_seen:
                        continue
                    kl = k.lower()
                    if not (kl.endswith(".mp4") or kl.endswith(".pdf")):
                        continue
                    if "/input/" in kl:
                        continue
                    fname = os.path.basename(k)
                    label = f"File: {fname}"
                    if "dots.mp4" in kl:
                        label = "Dots Video"
                    elif "skeleton.mp4" in kl:
                        label = "Skeleton Video"
                    elif "_th.pdf" in kl:
                        label = "Report TH (PDF)"
                    elif "_en.pdf" in kl:
                        label = "Report EN (PDF)"
                    else:
                        continue
                    discovered_direct_items.append((label, k, fname))
                    discovered_seen.add(k)
        except Exception:
            pass
    if discovered_direct_items:
        direct_items_top.extend(discovered_direct_items)
    # Dedupe by exact key while preserving order.
    _tmp_items: List[tuple[str, str, str]] = []
    _tmp_seen: set[str] = set()
    for _label, _key, _fn in direct_items_top:
        if not _key or _key in _tmp_seen:
            continue
        _tmp_items.append((_label, _key, _fn))
        _tmp_seen.add(_key)
    direct_items_top = _tmp_items
    ready_direct_top = [(label, key, fn) for (label, key, fn) in direct_items_top if key and s3_key_exists(key)]
    if ready_direct_top:
        for label, key, fn in ready_direct_top:
            st.link_button(f"Direct Download: {label}", presigned_get_url(key, expires=3600, filename=fn), width="stretch")
    else:
        st.caption("ยังไม่พบไฟล์ของ Submission ID นี้ใน S3")

    # Direct force requeue by Submission ID (useful when only one video output is missing).
    dots_ready_top = bool(direct_outputs_top.get("dots_video")) and s3_key_exists(str(direct_outputs_top.get("dots_video") or ""))
    skel_ready_top = bool(direct_outputs_top.get("skeleton_video")) and s3_key_exists(str(direct_outputs_top.get("skeleton_video") or ""))
    if (not dots_ready_top) or (not skel_ready_top):
        direct_rows = list_jobs_for_group(direct_group_id)
        dots_active_top = any(
            str(r.get("mode") or "").strip().lower() == "dots"
            and str(r.get("status") or "").strip().lower() in ("pending", "processing")
            for r in direct_rows
        )
        skel_active_top = any(
            str(r.get("mode") or "").strip().lower() == "skeleton"
            and str(r.get("status") or "").strip().lower() in ("pending", "processing")
            for r in direct_rows
        )
        st.warning("ยังไม่ครบทุกวิดีโอ สามารถกดส่งงานซ้ำเฉพาะรายการที่ขาดได้")
        d1, d2 = st.columns(2)
        with d1:
            if not dots_ready_top:
                if st.button(
                    "Force ส่งงาน Dots" if dots_active_top else "ส่งงาน Dots ใหม่",
                    key=f"direct_force_dots_{direct_group_id}",
                    width="stretch",
                ):
                    try:
                        new_key = enqueue_video_only_job(
                            group_id=direct_group_id,
                            mode="dots",
                            notify_email=notify_email,
                            employee_id=employee_id,
                            enterprise_folder=enterprise_folder,
                        )
                        st.success(f"ส่งงาน Dots แล้ว: {new_key}")
                    except Exception as e:
                        st.error(f"ส่งงาน Dots ไม่สำเร็จ: {format_submit_error_message(e)}")
        with d2:
            if not skel_ready_top:
                if st.button(
                    "Force ส่งงาน Skeleton" if skel_active_top else "ส่งงาน Skeleton ใหม่",
                    key=f"direct_force_skeleton_{direct_group_id}",
                    width="stretch",
                ):
                    try:
                        new_key = enqueue_video_only_job(
                            group_id=direct_group_id,
                            mode="skeleton",
                            notify_email=notify_email,
                            employee_id=employee_id,
                            enterprise_folder=enterprise_folder,
                        )
                        st.success(f"ส่งงาน Skeleton แล้ว: {new_key}")
                    except Exception as e:
                        st.error(f"ส่งงาน Skeleton ไม่สำเร็จ: {format_submit_error_message(e)}")
else:
    st.caption("วาง Submission ID ด้านบนเพื่อแสดงปุ่มดาวน์โหลดทันที")
if direct_ready and not upload_done and not manual_direct_done:
    pass
elif upload_done or manual_direct_done:
    st.success("อัปโหลดไป S3 สำเร็จแล้ว กำลังส่งงานเข้าคิววิเคราะห์...")
elif last_group_hint:
    st.caption(f"งานล่าสุด: `{last_group_hint}` (สามารถเลื่อนลงไปดูผลลัพธ์/ดาวน์โหลดได้)")
else:
    if use_direct_upload:
        st.caption("ยังไม่เริ่มอัปโหลด กรุณากดปุ่ม 'เลือกวิดีโอและอัปโหลด'")
    else:
        st.caption("โหมด local: ใช้อัปโหลดแบบสำรองด้านล่าง แล้วกด 'เริ่มวิเคราะห์'")

# Quick download/status block at top so users do not need to scroll to the bottom.
quick_group_id = str(submission_id_override or last_group_hint or _read_group_id_from_url() or "").strip()
if quick_group_id:
    st.caption(f"งานล่าสุด: `{quick_group_id}` (สามารถเลื่อนลงไปดูผลลัพธ์/ดาวน์โหลดได้)")
    quick_outputs = build_output_keys(quick_group_id)
    quick_report_outputs = get_report_outputs_from_job(quick_group_id)
    for _k in (
        "report_en_docx",
        "report_th_docx",
        "report_en_pdf",
        "report_th_pdf",
        "report_en_html",
        "report_th_html",
    ):
        if quick_report_outputs.get(_k):
            quick_outputs[_k] = str(quick_report_outputs.get(_k) or "").strip()

    quick_items = [
        ("วิดีโอ Dots", quick_outputs.get("dots_video", ""), "dots.mp4"),
        ("วิดีโอ Skeleton", quick_outputs.get("skeleton_video", ""), "skeleton.mp4"),
        ("รายงาน TH (PDF)", quick_outputs.get("report_th_pdf", ""), "report_th.pdf"),
        ("รายงาน EN (PDF)", quick_outputs.get("report_en_pdf", ""), "report_en.pdf"),
    ]
    ready_quick = [(label, key, fn) for (label, key, fn) in quick_items if key and s3_key_exists(key)]
    if ready_quick:
        st.markdown("#### พร้อมดาวน์โหลดตอนนี้ (งานล่าสุด)")
        for label, key, fn in ready_quick:
            st.link_button(f"ดาวน์โหลด {label}", presigned_get_url(key, expires=3600, filename=fn), width="stretch")

    # Allow one-click requeue for missing dots/skeleton from the latest group.
    # Do not gate by report-ready state, because some jobs may have partial outputs.
    dots_ready_quick = bool(quick_outputs.get("dots_video")) and s3_key_exists(str(quick_outputs.get("dots_video") or ""))
    skel_ready_quick = bool(quick_outputs.get("skeleton_video")) and s3_key_exists(str(quick_outputs.get("skeleton_video") or ""))
    if (not dots_ready_quick) or (not skel_ready_quick):
        quick_rows = list_jobs_for_group(quick_group_id)
        dots_active = any(str(r.get("mode") or "").strip().lower() == "dots" and str(r.get("status") or "").strip().lower() in ("pending", "processing") for r in quick_rows)
        skel_active = any(str(r.get("mode") or "").strip().lower() == "skeleton" and str(r.get("status") or "").strip().lower() in ("pending", "processing") for r in quick_rows)
        st.warning("ยังมีผลลัพธ์วิดีโอไม่ครบ (Dots/Skeleton) สามารถกดส่งงานใหม่เฉพาะรายการได้")
        c_requeue1, c_requeue2 = st.columns(2)
        with c_requeue1:
            if not dots_ready_quick:
                if dots_active:
                    st.caption("Dots มีงานค้างในคิวอยู่ แต่สามารถกดส่งซ้ำได้")
                if st.button(
                    "ส่งงาน Dots ใหม่" if not dots_active else "ส่งงาน Dots ซ้ำ (force)",
                    key=f"requeue_dots_top_{quick_group_id}",
                    width="stretch",
                ):
                    try:
                        k = enqueue_video_only_job(
                            group_id=quick_group_id,
                            mode="dots",
                            notify_email=notify_email,
                            employee_id=employee_id,
                            enterprise_folder=enterprise_folder,
                        )
                        st.success(f"ส่งงาน Dots ใหม่แล้ว: {k}")
                    except Exception as e:
                        st.error(f"ส่งงาน Dots ใหม่ไม่สำเร็จ: {format_submit_error_message(e)}")
        with c_requeue2:
            if not skel_ready_quick:
                if skel_active:
                    st.caption("Skeleton มีงานค้างในคิวอยู่ แต่สามารถกดส่งซ้ำได้")
                if st.button(
                    "ส่งงาน Skeleton ใหม่" if not skel_active else "ส่งงาน Skeleton ซ้ำ (force)",
                    key=f"requeue_skeleton_top_{quick_group_id}",
                    width="stretch",
                ):
                    try:
                        k = enqueue_video_only_job(
                            group_id=quick_group_id,
                            mode="skeleton",
                            notify_email=notify_email,
                            employee_id=employee_id,
                            enterprise_folder=enterprise_folder,
                        )
                        st.success(f"ส่งงาน Skeleton ใหม่แล้ว: {k}")
                    except Exception as e:
                        st.error(f"ส่งงาน Skeleton ใหม่ไม่สำเร็จ: {format_submit_error_message(e)}")

    # Direct download block: fetch links by Submission ID from S3 keys immediately.
    st.markdown("#### Direct download by Submission ID")
    direct_outputs = build_output_keys(quick_group_id)
    direct_reports = find_report_files_in_s3(f"{JOBS_GROUP_PREFIX}{quick_group_id}")
    if not direct_reports.get("report_en_docx") and not direct_reports.get("report_th_docx"):
        # Fallback prefix where some outputs may be written.
        direct_reports = find_report_files_in_s3(f"{JOBS_OUTPUT_PREFIX}groups/{quick_group_id}")
    for _k in ("report_en_docx", "report_th_docx", "report_en_pdf", "report_th_pdf", "report_en_html", "report_th_html"):
        if direct_reports.get(_k):
            direct_outputs[_k] = str(direct_reports.get(_k) or "").strip()

    direct_items = [
        ("Dots Video", str(direct_outputs.get("dots_video") or "").strip(), "dots.mp4"),
        ("Skeleton Video", str(direct_outputs.get("skeleton_video") or "").strip(), "skeleton.mp4"),
        ("Report TH (PDF)", str(direct_outputs.get("report_th_pdf") or "").strip(), "report_th.pdf"),
        ("Report EN (PDF)", str(direct_outputs.get("report_en_pdf") or "").strip(), "report_en.pdf"),
    ]
    ready_direct = [(label, key, fn) for (label, key, fn) in direct_items if key and s3_key_exists(key)]
    if ready_direct:
        for label, key, fn in ready_direct:
            st.link_button(f"Direct Download: {label}", presigned_get_url(key, expires=3600, filename=fn), width="stretch")
    else:
        st.caption("ยังไม่พบไฟล์สำหรับ Submission ID นี้ใน S3 (ลองกดรีเฟรชอีกครั้ง)")

# -------------------------
# Direct S3 upload (browser -> S3, skips server for speed)
# -------------------------
if use_direct_upload:
    if direct_ready and not upload_done and not manual_direct_done:
        presigned = st.session_state.get("direct_upload_presigned_url", "")
        gid = st.session_state.get("direct_upload_group_id", "")
        nem = st.session_state.get("direct_upload_notify_email", "")
        eid = st.session_state.get("direct_upload_employee_id", "")
        ikey = st.session_state.get("direct_upload_input_key", "")
        if presigned and gid:
            st.info("กำลังรออัปโหลดไฟล์ไป S3: เลือกไฟล์ในกรอบด้านล่าง แล้วรอจนเสร็จ จากนั้นกดปุ่มส่งงานต่อ")
            st.caption("อัปโหลดตรงไปยัง S3 (เร็วกว่า ไม่ผ่านเซิร์ฟเวอร์)")
            components.html(
                _direct_upload_html(presigned, gid, nem, eid),
                height=220,
                scrolling=False,
            )
            if st.button("✅ อัปโหลดเสร็จแล้ว - ส่งงานต่อ", key="direct_upload_continue"):
                if ikey and s3_key_exists(str(ikey)):
                    st.session_state["direct_upload_done_manual"] = True
                    st.rerun()
                else:
                    st.warning("ยังไม่พบไฟล์ใน S3 กรุณารอสักครู่แล้วกดปุ่มนี้อีกครั้ง")
            if st.button("← กลับ", key="direct_upload_back"):
                for k in ("direct_upload_ready", "direct_upload_presigned_url", "direct_upload_group_id",
                          "direct_upload_notify_email", "direct_upload_employee_id", "direct_upload_enterprise_folder"):
                    st.session_state.pop(k, None)
                st.rerun()
            st.caption(SUPPORT_CONTACT_TEXT)
            upload_clicked = False
        else:
            st.error("ไม่สามารถเตรียมลิงก์อัปโหลดได้ กรุณากดปุ่มอัปโหลดใหม่อีกครั้ง")
            if st.button("ล้างสถานะอัปโหลดแล้วเริ่มใหม่", key="reset_direct_upload_state"):
                for k in (
                    "direct_upload_ready",
                    "direct_upload_presigned_url",
                    "direct_upload_group_id",
                    "direct_upload_input_key",
                    "direct_upload_notify_email",
                    "direct_upload_employee_id",
                    "direct_upload_enterprise_folder",
                    "direct_upload_user_name",
                ):
                    st.session_state.pop(k, None)
                st.rerun()
            upload_clicked = False
    else:
        upload_clicked = st.button("📤 เลือกวิดีโอและอัปโหลด", type="primary", width="stretch", key="upload_video_btn")
    st.caption("อัปโหลดตรงไปยัง S3 — เร็วกว่าแบบเดิม (หากอัปโหลดล้มเหลว ให้ใช้แบบสำรองด้านล่าง)")
else:
    upload_clicked = False
    st.info("Local mode: ปิด direct-to-S3 ชั่วคราว เพื่อเลี่ยงปัญหา CORS/อัปโหลดค้าง")

uploaded = st.file_uploader(
    "วิดีโอ (MP4/MOV/M4V/WEBM) — แบบสำรอง",
    type=["mp4", "mov", "m4v", "webm"],
    accept_multiple_files=False,
    key="skilllane_file_uploader",
)
st.caption("หากอัปโหลดล้มเหลวหรือขึ้นสถานะ 400 กรุณา upload วีดีโอใหม่")
if uploaded is not None:
    uploaded_name = str(uploaded.name or "input.mp4")
    uploaded_size_mb = float((uploaded.size or 0) / (1024 * 1024))
    st.caption(f"Selected file: `{uploaded_name}` ({uploaded_size_mb:.2f} MB)")

run = st.button(
    "🎬 เริ่มวิเคราะห์",
    type="primary",
    width="stretch",
    key="run_analyze",
    disabled=(
        (uploaded is None)
        or (not notify_email)
        or (not is_valid_email_format(notify_email))
        or is_blocked_typo_domain(notify_email)
    ),
)
if use_direct_upload:
    st.caption("หรือใช้แบบสำรอง: เลือกไฟล์ด้านบนแล้วกดปุ่มนี้")
if not notify_email:
    st.warning("กรุณากรอกอีเมลผู้ใช้งานก่อนอัปโหลด/เริ่มวิเคราะห์")
elif (not is_valid_email_format(notify_email)) or is_blocked_typo_domain(notify_email):
    st.warning("รูปแบบอีเมลไม่ถูกต้อง กรุณาตรวจสอบก่อนเริ่มวิเคราะห์")
    st.caption(SUPPORT_CONTACT_TEXT)

# Handle "Upload Video" click -> prepare direct upload
if use_direct_upload and upload_clicked:
    if not notify_email:
        st.error("กรุณากรอกอีเมลผู้ใช้งาน")
    elif (not is_valid_email_format(notify_email)) or is_blocked_typo_domain(notify_email):
        st.error("รูปแบบ e-mail ไม่ถูกต้อง กรุณาตรวจสอบ e-mail อีกครั้ง")
    else:
        base_user = safe_slug(user_name, fallback="user")
        group_id = f"{new_group_id()}__{base_user}"
        input_key = f"{JOBS_GROUP_PREFIX}{group_id}/input/input.mp4"
        presigned_url = presigned_put_url(input_key, content_type="video/mp4", expires=3600)
        st.session_state["direct_upload_ready"] = True
        st.session_state["direct_upload_presigned_url"] = presigned_url
        st.session_state["direct_upload_group_id"] = group_id
        st.session_state["direct_upload_input_key"] = input_key
        st.session_state["direct_upload_notify_email"] = notify_email
        st.session_state["direct_upload_employee_id"] = employee_id
        st.session_state["direct_upload_enterprise_folder"] = enterprise_folder
        st.session_state["direct_upload_user_name"] = user_name
        st.rerun()

# Handle redirect after direct upload (upload_done=1)
url_upload_group = str(st.query_params.get("group_id", "") or "").strip()
url_upload_notify = str(st.query_params.get("notify_email", "") or "").strip()
url_upload_employee = str(st.query_params.get("employee_id", "") or "").strip()
url_upload_filename = str(st.query_params.get("uploaded_file", "") or "").strip()
if manual_direct_done and not upload_done:
    upload_done = True
    url_upload_group = str(st.session_state.get("direct_upload_group_id") or "").strip()
    url_upload_notify = str(st.session_state.get("direct_upload_notify_email") or "").strip()
    url_upload_employee = str(st.session_state.get("direct_upload_employee_id") or "").strip()
# Fallback from current form/session when callback query params are incomplete.
if upload_done and url_upload_group:
    if not url_upload_notify:
        url_upload_notify = str(
            st.session_state.get("direct_upload_notify_email")
            or notify_email
            or ""
        ).strip()
    if not url_upload_employee:
        url_upload_employee = str(
            st.session_state.get("direct_upload_employee_id")
            or employee_id
            or ""
        ).strip()

if upload_done and url_upload_group:
    for k in (
        "direct_upload_ready",
        "direct_upload_presigned_url",
        "direct_upload_group_id",
        "direct_upload_input_key",
        "direct_upload_notify_email",
        "direct_upload_employee_id",
        "direct_upload_enterprise_folder",
        "direct_upload_user_name",
    ):
        st.session_state.pop(k, None)
    run = True
    group_id = url_upload_group
    notify_email = url_upload_notify
    employee_id = url_upload_employee
    user_name = url_upload_notify
    enterprise_folder = "SkillLane"
    uploaded = None
    if url_upload_filename:
        st.session_state["last_uploaded_filename"] = url_upload_filename
    org_settings = get_org_settings(enterprise_folder)
elif upload_done and (not url_upload_group):
    st.error("อัปโหลดสำเร็จแต่ไม่พบ group_id สำหรับส่งงานต่อ กรุณากดอัปโหลดใหม่อีกครั้ง")

has_identity_input = bool(employee_id.strip() and notify_email)
identity_verified = False
if has_identity_input:
    identity_verified = is_employee_identity_verified(employee_id, notify_email)

candidate_group_id = (
    submission_id_override
    or st.session_state.get("last_group_id", "")
    or url_group_id
)
active_group_id = ""
blocked_group_id = ""
if candidate_group_id:
    # Email-only UX: always show latest accessible group from current session/url.
    active_group_id = candidate_group_id
    st.session_state["last_group_id"] = active_group_id
    _persist_group_id_to_url(active_group_id)

note = st.empty()

# -------------------------
# Submit jobs
# -------------------------
if run:
    uploaded_filename_for_status = ""
    input_key = f"{JOBS_GROUP_PREFIX}{group_id}/input/input.mp4" if upload_done else None
    if not upload_done and not uploaded:
        note.error("กรุณา upload วีดีโอใหม่ หรือกด 'เลือกวิดีโอและอัปโหลด'")
        st.stop()
    if not notify_email:
        note.error("กรุณากรอกอีเมลผู้ใช้งาน")
        st.stop()
    if (not is_valid_email_format(notify_email)) or is_blocked_typo_domain(notify_email):
        note.error("รูปแบบ e-mail ไม่ถูกต้อง กรุณาตรวจสอบ e-mail อีกครั้ง")
        st.stop()
    # Page policy: SkillLane always uses full report style.
    effective_report_style = "full"
    # Deliver report as PDF.
    effective_report_format = "pdf"
    # SkillLane page policy: always enqueue TH+EN report jobs.
    # Admin toggles can still control optional video outputs, but report should always run.
    enable_report_th = True
    enable_report_en = True
    # SkillLane page policy: always enqueue video outputs too.
    enable_skeleton = True
    enable_dots = True
    report_languages = []
    if enable_report_th:
        report_languages.append("th")
    if enable_report_en:
        report_languages.append("en")
    if not (enable_dots or enable_skeleton or report_languages):
        note.error("องค์กรนี้ยังไม่ได้เปิดการส่งออกผลลัพธ์ใดๆ กรุณาตั้งค่าจากหน้า Admin ก่อน")
        st.stop()

    if not upload_done:
        base_user = safe_slug(user_name, fallback="user")
        group_id = f"{new_group_id()}__{base_user}"
        input_key = f"{JOBS_GROUP_PREFIX}{group_id}/input/input.mp4"
        uploaded_filename_for_status = str((uploaded.name if uploaded is not None else "") or "input.mp4")

        try:
            s3_upload_stream(
                key=input_key,
                file_obj=uploaded,
                content_type=guess_content_type(uploaded_filename_for_status),
            )
        except Exception as e:
            note.error(f"อัปโหลดไป S3 ไม่สำเร็จ: {format_submit_error_message(e)}")
            st.warning(SUPPORT_CONTACT_TEXT)
            st.stop()
    else:
        input_key = f"{JOBS_GROUP_PREFIX}{group_id}/input/input.mp4"
        uploaded_filename_for_status = str(
            st.session_state.get("last_uploaded_filename")
            or url_upload_filename
            or "input.mp4"
        )
        if not s3_key_exists(input_key):
            note.error("ไม่พบวิดีโอใน S3 กรุณาอัปโหลดใหม่อีกครั้ง")
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
        "notify_email": notify_email,
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
        "notify_email": notify_email,
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
        "languages": report_languages,
        "output_prefix": f"{JOBS_GROUP_PREFIX}{group_id}",
        "analysis_mode": "real",  # Use real MediaPipe analysis
        "sample_fps": 5,
        "max_frames": 300,
        "report_style": effective_report_style,
        "report_format": effective_report_format,
        "expect_skeleton": bool(enable_skeleton),
        "expect_dots": bool(enable_dots),
        "notify_email": notify_email,
        "enterprise_folder": (enterprise_folder or "").strip(),
        "employee_id": (employee_id or "").strip(),
        "employee_email": notify_email,
    }

    try:
        save_employee_registry(
            employee_id=employee_id,
            employee_email=notify_email,
            organization_name=enterprise_folder,
        )
        queued_job_ids: Dict[str, str] = {}
        queued_job_keys: Dict[str, str] = {}
        if enable_dots:
            queued_job_keys["dots"] = enqueue_legacy_job(job_dots)
            queued_job_ids["dots"] = job_dots["job_id"]
        if enable_skeleton:
            queued_job_keys["skeleton"] = enqueue_legacy_job(job_skel)
            queued_job_ids["skeleton"] = job_skel["job_id"]
        if report_languages:
            queued_job_keys["report"] = enqueue_legacy_job(job_report)
            queued_job_ids["report"] = job_report["job_id"]
    except Exception as e:
        note.error(f"ส่งงานเข้าคิวไม่สำเร็จ: {format_submit_error_message(e)}")
        st.warning(SUPPORT_CONTACT_TEXT)
        st.stop()

    missing_pending = verify_pending_jobs_exist(list(queued_job_keys.values()))
    if missing_pending:
        note.error(
            "ระบบตรวจสอบคิวงานไม่ผ่าน: ไม่พบไฟล์งานบางรายการใน jobs/pending\n"
            f"Missing keys: {', '.join(missing_pending)}"
        )
        st.warning(SUPPORT_CONTACT_TEXT)
        st.stop()

    st.session_state["last_group_id"] = group_id
    _persist_group_id_to_url(group_id)
    active_group_id = group_id
    st.session_state["last_outputs"] = outputs
    st.session_state["last_jobs"] = queued_job_ids
    st.session_state["last_job_json_keys"] = queued_job_keys
    st.session_state["last_uploaded_filename"] = uploaded_filename_for_status

    note.success(
        f"ส่งงานเรียบร้อย! submission_id = {group_id} | report_style={effective_report_style}, report_format={effective_report_format}, "
        f"outputs={','.join(list(queued_job_ids.keys())) or '-'}"
    )
    st.caption(f"Queued job keys: {', '.join(queued_job_keys.values())}")
    if uploaded_filename_for_status:
        st.success(f"Uploaded to S3: {uploaded_filename_for_status}")
    st.caption(f"Submission ID: `{group_id}`")
    st.info("ระบบได้ทำการวิเคราะห์แล้ว ท่านจะได้รับ e-mail แจ้งหลังจากนี้ ขอบคุณที่ใช้ AI People Reader")

    if upload_done:
        try:
            for p in ("upload_done", "notify_email", "uploaded_file"):
                if p in st.query_params:
                    del st.query_params[p]
        except Exception:
            pass

# Keep showing the latest submitted group in this session even before ownership index catches up.
if not active_group_id:
    recent_group_id = str(st.session_state.get("last_group_id") or "").strip()
    recent_jobs = st.session_state.get("last_jobs") or {}
    if recent_group_id and recent_jobs:
        active_group_id = recent_group_id
        _persist_group_id_to_url(active_group_id)

group_id = active_group_id
if group_id:
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
    if report_outputs.get("report_en_pdf"):
        outputs["report_en_pdf"] = report_outputs["report_en_pdf"]
    if report_outputs.get("report_th_pdf"):
        outputs["report_th_pdf"] = report_outputs["report_th_pdf"]
    if report_outputs.get("report_en_html"):
        outputs["report_en_html"] = report_outputs["report_en_html"]
    if report_outputs.get("report_th_html"):
        outputs["report_th_html"] = report_outputs["report_th_html"]
else:
    if has_identity_input and not identity_verified:
        st.caption("กรุณากรอกอีเมลให้ถูกต้อง เพื่อดูเฉพาะงานของตนเอง")
    else:
        st.caption("ยังไม่พบ Submission ID ที่เข้าถึงได้สำหรับบัญชีนี้ กรุณาอัปโหลดวิดีโอแล้วกด **เริ่มวิเคราะห์**")
    st.divider()
    st.link_button(
        "กลับไปสู่บทเรียนออนไลน์ (SkillLane)",
        "https://www.skilllane.com/courses/8076",
        width="stretch",
    )
    st.stop()

st.caption(f"Submission ID: `{group_id}`")

with st.expander("ตรวจสถานะ Submission ID นี้ (Debug)", expanded=True):
    debug_group_id = st.text_input(
        "Submission ID ที่ต้องการตรวจสถานะ",
        value=group_id,
        key="skilllane_debug_group_id",
        help="ตรวจว่าแต่ละ job ของ submission นี้อยู่ใน pending/processing/finished/failed",
    ).strip()
    if debug_group_id:
        debug_rows = list_jobs_for_group(debug_group_id)
        if debug_rows:
            status_counts = {"pending": 0, "processing": 0, "finished": 0, "failed": 0}
            for row in debug_rows:
                state = str(row.get("status") or "").strip().lower()
                if state in status_counts:
                    status_counts[state] += 1
            st.caption(
                " | ".join(
                    [
                        f"pending={status_counts['pending']}",
                        f"processing={status_counts['processing']}",
                        f"finished={status_counts['finished']}",
                        f"failed={status_counts['failed']}",
                    ]
                )
            )
            st.dataframe(debug_rows, width="stretch", hide_index=True)
        else:
            st.warning("ไม่พบ job ของ Submission ID นี้ในคิว pending/processing/finished/failed")


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
        st.caption(SUPPORT_CONTACT_TEXT)
        st.code(key, language="text")


# --- Downloads ---
c1, c2 = st.columns(2)

with c1:
    st.markdown("### วิดีโอ")
    download_block("วิดีโอ Dots", outputs.get("dots_video", ""), "dots.mp4")
    download_block("วิดีโอ Skeleton", outputs.get("skeleton_video", ""), "skeleton.mp4")

with c2:
    st.markdown("### รายงาน")
    en_pdf_key = str(outputs.get("report_en_pdf", "") or "").strip()
    th_pdf_key = str(outputs.get("report_th_pdf", "") or "").strip()

    st.markdown("**ภาษาอังกฤษ**")
    download_block("รายงาน EN (PDF)", en_pdf_key, "report_en.pdf")

    st.markdown("**ภาษาไทย**")
    download_block("รายงาน TH (PDF)", th_pdf_key, "report_th.pdf")

    # Primary report ready state (for progress/status + quick download) - PDF only
    en_key = en_pdf_key
    th_key = th_pdf_key
    en_name = "report_en.pdf"
    th_name = "report_th.pdf"

videos_ready = bool(outputs.get("dots_video")) and bool(outputs.get("skeleton_video")) and s3_key_exists(outputs.get("dots_video", "")) and s3_key_exists(outputs.get("skeleton_video", ""))
dots_ready = bool(outputs.get("dots_video")) and s3_key_exists(outputs.get("dots_video", ""))
skeleton_ready = bool(outputs.get("skeleton_video")) and s3_key_exists(outputs.get("skeleton_video", ""))
en_report_ready = bool(en_key) and s3_key_exists(en_key)
th_report_ready = bool(th_key) and s3_key_exists(th_key)
primary_done = th_report_ready

# Show ready artifacts immediately without waiting for all outputs.
ready_now = []
if th_report_ready and th_key:
    ready_now.append(("รายงาน TH", th_key, th_name))
if en_report_ready and en_key:
    ready_now.append(("รายงาน EN", en_key, en_name))
if dots_ready:
    ready_now.append(("วิดีโอ Dots", outputs.get("dots_video", ""), "dots.mp4"))
if skeleton_ready:
    ready_now.append(("วิดีโอ Skeleton", outputs.get("skeleton_video", ""), "skeleton.mp4"))

st.divider()
st.subheader("พร้อมดาวน์โหลดตอนนี้")
if st.button("รีเฟรชสถานะผลลัพธ์", key="refresh_ready_downloads", width="content"):
    st.rerun()
if ready_now:
    for label, key, filename in ready_now:
        url = presigned_get_url(key, expires=3600, filename=filename)
        st.success(f"✅ {label} พร้อมแล้ว")
        st.link_button(f"ดาวน์โหลด {label}", url, width="stretch")
else:
    st.info("ยังไม่มีไฟล์ที่พร้อมดาวน์โหลดตอนนี้ ระบบจะทยอยขึ้นให้ทันทีเมื่อไฟล์นั้นเสร็จ")
st.caption("ถ้าไฟล์ไหนพร้อม จะขึ้นให้ก่อนทันที ไม่ต้องรอ Report และ Video ให้เสร็จพร้อมกัน")

st.divider()
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
    next_step = "ไฟล์อื่น (EN/Dots/Skeleton) จะทยอยพร้อมและดาวน์โหลดได้ทันทีที่เสร็จ"
else:
    current_step = "กำลังสร้างรายงานภาษาไทย"
    next_step = "ระบบจะโชว์ปุ่มดาวน์โหลดทันทีเมื่อแต่ละไฟล์พร้อม"

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
            rerun_style = "full"
            rerun_format = "pdf"
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
            st.error(f"ไม่สามารถส่งงานสร้างรายงานใหม่เข้าคิวได้: {format_submit_error_message(e)}")

st.divider()
st.link_button(
    "กลับไปสู่บทเรียนออนไลน์ (SkillLane)",
    "https://www.skilllane.com/courses/8076",
    width="stretch",
)
