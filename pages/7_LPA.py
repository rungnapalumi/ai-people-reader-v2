# pages/7_LPA.py — LPA portal (same flow as TTB)
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

from dotenv import load_dotenv

# Multipage apps run this file without executing app.py — load repo-root .env for local dev.
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

import streamlit as st
import streamlit.components.v1 as components
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

SUPPORT_CONTACT_TEXT = "หากพบปัญหากรุณาติดต่อ 0817008484"


# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="LPA — Video Analysis (วิเคราะห์วิดีโอ)", layout="wide")

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
    st.error(
        "Missing **AWS_BUCKET** (or **S3_BUCKET**).\n\n"
        "- **Local:** Copy `.env.example` → `.env` in the project root, set `AWS_BUCKET=...`, then restart Streamlit.\n"
        "- **Render:** Add the variable under the service **Environment** tab."
    )
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
            st.image(path, use_container_width=True)
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
    settings.sort(key=lambda x: str(x.get("updated_at") or ""), reverse=True)
    for s in settings:
        if str(s.get("default_page") or "").strip().lower() == page_key:
            return str(s.get("organization_name") or "").strip()
    return ""


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
        "sample_fps": 3,
        "max_frames": 150,
        "report_style": report_style,
        "report_format": report_format,
        "expect_skeleton": False,
        "expect_dots": False,
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
                            if scanned.get("report_en_html"):
                                found["report_en_html"] = scanned["report_en_html"]
                            if scanned.get("report_th_html"):
                                found["report_th_html"] = scanned["report_th_html"]
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
        if scanned.get("report_en_html") and not found.get("report_en_html"):
            found["report_en_html"] = scanned["report_en_html"]
        if scanned.get("report_th_html") and not found.get("report_th_html"):
            found["report_th_html"] = scanned["report_th_html"]
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
        "report_th_sent": bool(notif.get("report_th_sent") or notif.get("report_sent")),
        "report_en_sent": bool(notif.get("report_en_sent")),
        "skeleton_sent": bool(notif.get("skeleton_sent")),
        "dots_sent": bool(notif.get("dots_sent")),
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
                kl = key.lower()
                if key.endswith(".docx"):
                    if "report_en.docx" in kl or "_en.docx" in kl:
                        result["report_en_docx"] = key
                    elif "report_th.docx" in kl or "_th.docx" in kl:
                        result["report_th_docx"] = key
                elif key.endswith(".pdf"):
                    if "report_en.pdf" in kl or "_en.pdf" in kl:
                        result["report_en_pdf"] = key
                    elif "report_th.pdf" in kl or "_th.pdf" in kl:
                        result["report_th_pdf"] = key
                elif key.endswith(".html"):
                    if "report_en.html" in kl or "_en.html" in kl:
                        result["report_en_html"] = key
                    elif "report_th.html" in kl or "_th.html" in kl:
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


def ensure_session_defaults() -> None:
    if "last_group_id" not in st.session_state:
        st.session_state["last_group_id"] = ""
    if "last_outputs" not in st.session_state:
        st.session_state["last_outputs"] = {}
    if "last_jobs" not in st.session_state:
        st.session_state["last_jobs"] = {}
    if "last_job_json_keys" not in st.session_state:
        st.session_state["last_job_json_keys"] = {}
    if "clear_upload_counter" not in st.session_state:
        st.session_state["clear_upload_counter"] = 0
    if "lpa_submission_id_override" not in st.session_state:
        st.session_state["lpa_submission_id_override"] = ""

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


def enqueue_video_only_job(
    group_id: str,
    mode: str,
    notify_email: str = "",
    employee_id: str = "",
    enterprise_folder: str = "",
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
        "enterprise_folder": (enterprise_folder or "").strip() or "LPA",
        "employee_id": (employee_id or "").strip(),
        "employee_email": (notify_email or "").strip(),
    }
    return enqueue_legacy_job(job)


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
if url_group_id:
    if not st.session_state.get("last_group_id"):
        st.session_state["last_group_id"] = url_group_id
    if not st.session_state.get("lpa_submission_id_override"):
        st.session_state["lpa_submission_id_override"] = url_group_id

st.markdown("# LPA — Video Analysis (วิเคราะห์วิดีโอ)")

st.divider()
st.caption("อัปโหลดวิดีโอ 1 ครั้ง แล้วกด **Run Analysis** เพื่อสร้างผลลัพธ์")

page_default_org = get_default_org_for_page("lpa")
enterprise_folder = st.text_input(
    "Organization Name",
    value=page_default_org,
    placeholder="e.g., LPA / ACME Group",
    disabled=bool(page_default_org),
)
if page_default_org:
    st.caption(f"Default organization from admin page setting: {page_default_org}")
user_name = st.text_input(
    "User Name (Email Address)",
    value="",
    placeholder="name@example.com",
    help="ใช้เป็นชื่อโฟลเดอร์งาน และเป็นอีเมลสำหรับส่งผลลัพธ์",
)
st.caption("กรุณาตรวจสอบการพิมพ์ e-mail ให้ถูกต้องก่อนกด Run Analysis (Please double-check your e-mail).")
notify_email = (user_name or "").strip()
if notify_email:
    if not is_valid_email_format(notify_email):
        st.warning("กรุณาตรวจสอบ e-mail ให้ถูกต้องอีกครั้ง (Please check your e-mail format).")
    elif is_blocked_typo_domain(notify_email):
        st.warning("รูปแบบโดเมนอีเมลอาจพิมพ์ผิด กรุณาตรวจสอบ e-mail อีกครั้ง (เช่น .com)")
# Email-only flow: reuse email as stable identity key.
employee_id = notify_email
org_settings = get_org_settings(enterprise_folder)

uploaded = st.file_uploader(
    "Video (MP4/MOV/M4V/WEBM)",
    type=["mp4", "mov", "m4v", "webm"],
    accept_multiple_files=False,
    key=f"lpa_file_uploader_{st.session_state.get('clear_upload_counter', 0)}",
)
if uploaded is not None:
    uploaded_name = str(uploaded.name or "input.mp4")
    uploaded_size_mb = float((uploaded.size or 0) / (1024 * 1024))
    st.caption(f"Selected file: `{uploaded_name}` ({uploaded_size_mb:.2f} MB)")

run = st.button("🎬 Run Analysis", type="primary", width="stretch", disabled=(uploaded is None))
st.caption(SUPPORT_CONTACT_TEXT)

lpa_direct_group_id = str(st.session_state.get("lpa_submission_id_override") or _read_group_id_from_url() or st.session_state.get("last_group_id") or "").strip()
last_group_hint = str(st.session_state.get("last_group_id") or url_group_id or lpa_direct_group_id or "").strip()
if run and not uploaded:
    st.warning("ยังไม่ได้เลือกไฟล์วิดีโอ กรุณาเลือกไฟล์ก่อนกด Run Analysis")
elif run and uploaded is not None:
    st.caption("กำลังอัปโหลดและส่งคิว — รอสักครู่ (ไฟล์ใหญ่ใช้เวลาหลายนาที)")
elif uploaded is not None:
    st.info("เลือกไฟล์แล้ว กด Run Analysis เพื่อเริ่มวิเคราะห์")
elif last_group_hint:
    st.caption(f"งานล่าสุด: `{last_group_hint}` (เลื่อนลงไปดูผลลัพธ์/ดาวน์โหลดได้)")
else:
    st.caption("ยังไม่เริ่มอัปโหลด กรุณาเลือกไฟล์และกด Run Analysis")

note = st.empty()
# Submit ก่อนส่วนดาวน์โหลด: เดิมสแกน S3 หลายรอบก่อนอัปโหลด ทำให้ค้างที่ "กำลังตรวจสอบ..." และ worker เห็น pending=0 ช้า
if run:
    if not uploaded:
        note.error("Please upload a video first.")
        st.stop()
    if not notify_email:
        note.error("Please enter User Name (Email Address).")
        st.stop()
    if (not is_valid_email_format(notify_email)) or is_blocked_typo_domain(notify_email):
        note.error("รูปแบบ e-mail ไม่ถูกต้อง กรุณาตรวจสอบ e-mail อีกครั้ง")
        st.stop()
    effective_report_style = "simple"
    effective_report_format = org_settings.get("report_format") if org_settings else "pdf"
    enable_report_th = bool(org_settings.get("enable_report_th", True)) if org_settings else True
    enable_report_en = bool(org_settings.get("enable_report_en", True)) if org_settings else True
    enable_skeleton = bool(org_settings.get("enable_skeleton", True)) if org_settings else True
    report_languages: List[str] = []
    if enable_report_th:
        report_languages.append("th")
    if enable_report_en:
        report_languages.append("en")
    if not (enable_skeleton or report_languages):
        note.error("This organization has no enabled outputs. Please update organization settings in Admin page.")
        st.stop()

    base_user = safe_slug(user_name, fallback="user")
    group_id_submit = f"{new_group_id()}__{base_user}"
    input_key_submit = f"{JOBS_GROUP_PREFIX}{group_id_submit}/input/input.mp4"

    with st.spinner("กำลังอัปโหลดวิดีโอไป S3 และส่งคิววิเคราะห์ (ไฟล์ใหญ่ 30MB+ อาจใช้ 3–15 นาทีบน Render)..."):
        try:
            s3_upload_stream(
                key=input_key_submit,
                file_obj=uploaded,
                content_type=guess_content_type(uploaded.name or "input.mp4"),
            )
        except Exception as e:
            note.error(f"Upload to S3 failed: {format_submit_error_message(e)}")
            st.warning(SUPPORT_CONTACT_TEXT)
            st.stop()

        outputs_submit = build_output_keys(group_id_submit)
        created_at_submit = utc_now_iso()

        job_skel = {
            "job_id": new_job_id(),
            "group_id": group_id_submit,
            "created_at": created_at_submit,
            "status": "pending",
            "mode": "skeleton",
            "input_key": input_key_submit,
            "output_key": outputs_submit["skeleton_video"],
            "user_name": user_name or "",
            "employee_id": (employee_id or "").strip(),
            "notify_email": notify_email,
            "employee_email": notify_email,
        }

        job_report = {
            "job_id": new_job_id(),
            "group_id": group_id_submit,
            "created_at": created_at_submit,
            "status": "pending",
            "mode": "report",
            "input_key": input_key_submit,
            "client_name": user_name or "Anonymous",
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "languages": report_languages,
            "output_prefix": f"{JOBS_GROUP_PREFIX}{group_id_submit}",
            "analysis_mode": "real",
            "sample_fps": 3,
            "max_frames": 150,
            "report_style": effective_report_style,
            "report_format": effective_report_format,
            "expect_skeleton": bool(enable_skeleton),
            "notify_email": notify_email,
            "enterprise_folder": (enterprise_folder or "").strip(),
            "expect_dots": False,
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
            if enable_skeleton:
                queued_job_keys["skeleton"] = enqueue_legacy_job(job_skel)
                queued_job_ids["skeleton"] = job_skel["job_id"]
            if report_languages:
                queued_job_keys["report"] = enqueue_legacy_job(job_report)
                queued_job_ids["report"] = job_report["job_id"]
        except Exception as e:
            note.error(f"Enqueue job failed: {format_submit_error_message(e)}")
            st.warning(SUPPORT_CONTACT_TEXT)
            st.stop()

        missing_pending = verify_pending_jobs_exist(list(queued_job_keys.values()))
        if missing_pending:
            note.error(
                "Queued job verification failed: some jobs were not found in jobs/pending.\n"
                f"Missing keys: {', '.join(missing_pending)}"
            )
            st.warning(SUPPORT_CONTACT_TEXT)
            st.stop()

    st.session_state["last_group_id"] = group_id_submit
    _persist_group_id_to_url(group_id_submit)
    st.session_state["last_outputs"] = outputs_submit
    st.session_state["last_jobs"] = queued_job_ids
    st.session_state["last_job_json_keys"] = queued_job_keys

    note.success(
        f"Submitted! submission_id = {group_id_submit} | report_style={effective_report_style}, report_format={effective_report_format}, "
        f"outputs={','.join(list(queued_job_ids.keys())) or '-'}"
    )
    st.caption(f"Queued job keys: {', '.join(queued_job_keys.values())}")
    st.info("ระบบได้ทำการวิเคราะห์แล้ว ท่านจะได้รับ e-mail แจ้งหลังจากนี้ ขอบคุณที่ใช้ AI People Reader")

# --- ดาวน์โหลดผลจากอีเมล (แสดงด้านล่างปุ่ม Run Analysis) ---
st.divider()
st.markdown("### 📥 ดาวน์โหลดผลลัพธ์ (ได้รับอีเมลแล้ว)")
st.caption("วาง Group ID จากอีเมล (บรรทัด 'Group ID: ...') ด้านล่าง แล้วกดรีเฟรช")
_lpa_sub_id_key = f"lpa_submission_id_top_{st.session_state.get('clear_upload_counter', 0)}"
lpa_submission_id_override = st.text_input(
    "Submission ID / Group ID (วางจากอีเมลได้)",
    value=str(st.session_state.get("lpa_submission_id_override") or ""),
    placeholder="เช่น 20260306_094309_e7d062_user",
    key=_lpa_sub_id_key,
    help="คัดลอก Group ID จากอีเมล วางตรงนี้ แล้วกดรีเฟรช",
).strip()
if lpa_submission_id_override:
    st.session_state["lpa_submission_id_override"] = lpa_submission_id_override
    st.session_state["last_group_id"] = lpa_submission_id_override
    _persist_group_id_to_url(lpa_submission_id_override)
_lpa_btn1, _lpa_btn2 = st.columns(2)
with _lpa_btn1:
    if st.button("🔄 รีเฟรชสถานะผลลัพธ์", key="lpa_refresh_status_top", width="content"):
        st.rerun()
with _lpa_btn2:
    _lpa_has_prev = bool(
        lpa_submission_id_override
        or st.session_state.get("lpa_submission_id_override")
        or st.session_state.get("last_group_id")
        or _read_group_id_from_url()
    )
    if _lpa_has_prev and st.button("🗑️ ล้างผลลัพธ์เพื่ออัปโหลดวิดีโอใหม่", key="lpa_clear_results_top", type="secondary"):
        for _k in ("lpa_submission_id_override", "last_group_id", "last_outputs", "last_jobs", "last_job_json_keys", "last_uploaded_filename"):
            st.session_state.pop(_k, None)
        st.session_state["clear_upload_counter"] = st.session_state.get("clear_upload_counter", 0) + 1
        try:
            if "group_id" in st.query_params:
                del st.query_params["group_id"]
        except Exception:
            pass
        st.rerun()

lpa_direct_group_id = str(lpa_submission_id_override or _read_group_id_from_url() or st.session_state.get("last_group_id") or "").strip()
if lpa_direct_group_id:
    st.caption(f"กำลังตรวจไฟล์จาก Submission ID: `{lpa_direct_group_id}`")
    lpa_direct_outputs = build_output_keys(lpa_direct_group_id)
    lpa_job_report_outputs = get_report_outputs_from_job(lpa_direct_group_id)
    lpa_direct_reports = find_report_files_in_s3(f"{JOBS_OUTPUT_PREFIX}groups/{lpa_direct_group_id}")
    if not lpa_direct_reports.get("report_en_pdf") or not lpa_direct_reports.get("report_th_pdf"):
        lpa_scanned = find_report_files_in_s3(f"{JOBS_GROUP_PREFIX}{lpa_direct_group_id}")
        for k in ("report_en_docx", "report_th_docx", "report_en_pdf", "report_th_pdf", "report_en_html", "report_th_html"):
            if lpa_scanned.get(k) and not lpa_direct_reports.get(k):
                lpa_direct_reports[k] = lpa_scanned[k]
    for _k in ("report_en_docx", "report_th_docx", "report_en_pdf", "report_th_pdf", "report_en_html", "report_th_html"):
        if lpa_job_report_outputs.get(_k):
            lpa_direct_outputs[_k] = str(lpa_job_report_outputs.get(_k) or "").strip()
        elif lpa_direct_reports.get(_k):
            lpa_direct_outputs[_k] = str(lpa_direct_reports.get(_k) or "").strip()
    lpa_discovered_keys: set = set()
    for _pfx in (f"{JOBS_GROUP_PREFIX}{lpa_direct_group_id}/", f"{JOBS_OUTPUT_PREFIX}groups/{lpa_direct_group_id}/"):
        try:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=_pfx):
                for item in page.get("Contents", []):
                    k = str(item.get("Key") or "").strip()
                    kl = k.lower() if k else ""
                    if k and (kl.endswith(".mp4") or kl.endswith(".pdf") or kl.endswith(".docx")) and "/input/" not in kl:
                        lpa_discovered_keys.add(k)
        except Exception:
            pass
    lpa_ready = [
        (label, key, fn)
        for label, key, fn in [
            ("Dots Video", str(lpa_direct_outputs.get("dots_video") or "").strip(), "dots.mp4"),
            ("Skeleton Video", str(lpa_direct_outputs.get("skeleton_video") or "").strip(), "skeleton.mp4"),
            ("Report TH", str(lpa_direct_outputs.get("report_th_pdf") or lpa_direct_outputs.get("report_th_docx") or "").strip(), "report_th.pdf"),
            ("Report EN", str(lpa_direct_outputs.get("report_en_pdf") or lpa_direct_outputs.get("report_en_docx") or "").strip(), "report_en.pdf"),
        ]
        if key and (key in lpa_discovered_keys or s3_key_exists(key))
    ]
    if lpa_ready:
        st.success("พบไฟล์พร้อมดาวน์โหลด")
    else:
        st.caption("หากได้รับอีเมลแล้ว ลิงก์ด้านล่างจะทำงานเมื่อไฟล์พร้อม (กดรีเฟรชหลัง 2–5 นาที)")
    st.markdown("**ลิงก์ดาวน์โหลด**")
    for label, key, fn in [
        ("Dots Video", str(lpa_direct_outputs.get("dots_video") or "").strip(), "dots.mp4"),
        ("Skeleton Video", str(lpa_direct_outputs.get("skeleton_video") or "").strip(), "skeleton.mp4"),
        ("Report TH (PDF/DOCX)", str(lpa_direct_outputs.get("report_th_pdf") or lpa_direct_outputs.get("report_th_docx") or "").strip(), "report_th.pdf"),
        ("Report EN (PDF/DOCX)", str(lpa_direct_outputs.get("report_en_pdf") or lpa_direct_outputs.get("report_en_docx") or "").strip(), "report_en.pdf"),
    ]:
        if key:
            fn_use = os.path.basename(key) or fn
            try:
                url = presigned_get_url(key, expires=3600, filename=fn_use)
                st.link_button(f"⬇️ {label}", url, width="stretch")
            except Exception:
                pass
    lpa_skel_ready = bool(lpa_direct_outputs.get("skeleton_video")) and (str(lpa_direct_outputs.get("skeleton_video") or "") in lpa_discovered_keys or s3_key_exists(str(lpa_direct_outputs.get("skeleton_video") or "")))
    if not lpa_skel_ready:
        _lpa_notify = str(st.session_state.get("last_notify_email") or "").strip()
        lpa_rows = list_jobs_for_group(lpa_direct_group_id)
        skel_active = any(str(r.get("mode") or "").strip().lower() == "skeleton" and str(r.get("status") or "").strip().lower() in ("pending", "processing") for r in lpa_rows)
        st.warning("ยังไม่ครบทุกวิดีโอ สามารถกดส่งงานซ้ำเฉพาะรายการที่ขาดได้")
        if st.button("Force ส่งงาน Skeleton" if skel_active else "ส่งงาน Skeleton ใหม่", key=f"lpa_force_skeleton_{lpa_direct_group_id}", width="stretch"):
            try:
                new_key = enqueue_video_only_job(group_id=lpa_direct_group_id, mode="skeleton", notify_email=_lpa_notify, employee_id=_lpa_notify, enterprise_folder=get_default_org_for_page("lpa") or "LPA")
                st.success(f"ส่งงาน Skeleton แล้ว: {new_key}")
            except Exception as e:
                st.error(f"ส่งงาน Skeleton ไม่สำเร็จ: {format_submit_error_message(e)}")
elif not lpa_direct_group_id:
    st.caption("กรุณาวาง Group ID จากอีเมลด้านบนเพื่อดูปุ่มดาวน์โหลด")

st.divider()
has_identity_input = bool(employee_id.strip() and notify_email)
identity_verified = False
if has_identity_input:
    identity_verified = is_employee_identity_verified(employee_id, notify_email)

candidate_group_id = st.session_state.get("last_group_id", "") or url_group_id or lpa_direct_group_id
active_group_id = ""
blocked_group_id = ""
if candidate_group_id:
    # Email-only UX: always show latest accessible group from current session/url.
    active_group_id = candidate_group_id
    st.session_state["last_group_id"] = active_group_id
    _persist_group_id_to_url(active_group_id)

group_id = active_group_id
if group_id:
    notification = get_report_notification_status(group_id)
    if notification:
        st.divider()
        st.subheader("Email Status")
        email_to = notification.get("notify_email", "")
        status = notification.get("status", "")
        report_th_sent = bool(notification.get("report_th_sent"))
        report_en_sent = bool(notification.get("report_en_sent"))
        skeleton_sent = bool(notification.get("skeleton_sent"))
        dots_sent = bool(notification.get("dots_sent"))
        st.caption(
            f"Report TH: {'yes' if report_th_sent else 'no'} | "
            f"Report EN: {'yes' if report_en_sent else 'no'} | "
            f"Skeleton: {'yes' if skeleton_sent else 'no'} | "
            f"Dots: {'yes' if dots_sent else 'no'}"
        )
        if report_th_sent and report_en_sent and skeleton_sent:
            st.success(f"All emails sent to: {email_to}")
        elif report_th_sent and not (report_en_sent and skeleton_sent):
            st.info(f"Thai report email sent to: {email_to} | English report and skeleton will send via mail soon.")
        elif report_en_sent or skeleton_sent or dots_sent:
            st.info(f"Partial email sent to: {email_to} | remaining files will send via mail soon.")
        elif status.startswith("waiting_for_"):
            st.info(f"Email queued: {status} (to: {email_to})")
        elif status in ("sending", "queued"):
            st.info(f"Email is being sent... (to: {email_to})")
        elif status == "skipped_no_notify_email":
            st.caption("No notification email provided for this job.")
        elif status == "disabled_by_config":
            st.caption("Email sending is disabled by config.")
        elif status:
            st.warning(f"Email status: {status} (to: {email_to})")
        st.caption("Status updates are automatic. Keep this page open to follow progress.")


# -------------------------
# Download section
# -------------------------
st.divider()
st.subheader("Downloads (ผลลัพธ์สำหรับดาวน์โหลด)")

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
else:
    if has_identity_input and not identity_verified:
        st.caption("Please enter the correct email to view only your own jobs.")
    else:
        st.caption("No accessible Submission ID for this account yet. Upload a video and click **Run Analysis**.")
    st.stop()

st.caption(f"Submission ID: `{group_id}`")


def download_block(title: str, key: str, filename: str) -> None:
    if not key:
        st.write(f"- {title}: (missing key)")
        return
    ready = s3_key_exists(key)
    if ready:
        url = presigned_get_url(key, expires=3600, filename=filename)
        st.success(f"✅ {title} ready")
        st.link_button(f"Download {title}", url, width="stretch")
        st.code(key, language="text")
    else:
        st.warning(f"⏳ {title} not ready yet")
        st.caption(SUPPORT_CONTACT_TEXT)
        st.code(key, language="text")


# --- Downloads ---
c1, c2 = st.columns(2)

with c1:
    st.markdown("### Videos")
    download_block("Skeleton video", outputs.get("skeleton_video", ""), "skeleton.mp4")

with c2:
    st.markdown("### Reports")
    en_pdf_candidate = outputs.get("report_en_pdf", "")
    en_docx_candidate = outputs.get("report_en_docx", "")
    th_pdf_candidate = outputs.get("report_th_pdf", "")
    th_docx_candidate = outputs.get("report_th_docx", "")

    en_pdf_ready = bool(en_pdf_candidate) and s3_key_exists(en_pdf_candidate)
    en_docx_ready = bool(en_docx_candidate) and s3_key_exists(en_docx_candidate)
    th_pdf_ready = bool(th_pdf_candidate) and s3_key_exists(th_pdf_candidate)
    th_docx_ready = bool(th_docx_candidate) and s3_key_exists(th_docx_candidate)

    en_key = en_pdf_candidate if en_pdf_ready else (en_docx_candidate if en_docx_ready else (en_pdf_candidate or en_docx_candidate))
    th_key = th_pdf_candidate if th_pdf_ready else (th_docx_candidate if th_docx_ready else (th_pdf_candidate or th_docx_candidate))
    en_name = "report_en.pdf" if en_pdf_ready else "report_en.docx"
    th_name = "report_th.pdf" if th_pdf_ready else "report_th.docx"
    report_label = "PDF" if (en_pdf_ready or th_pdf_ready) else ("DOCX" if (en_docx_ready or th_docx_ready) else "PDF/DOCX")
    st.markdown("**English**")
    download_block(f"Report EN ({report_label})", en_key, en_name)

    st.markdown("**Thai**")
    download_block(f"Report TH ({report_label})", th_key, th_name)

    st.markdown("**HTML (Debug/Preview)**")
    en_html_key = str(report_outputs.get("report_en_html") or "").strip()
    th_html_key = str(report_outputs.get("report_th_html") or "").strip()
    if en_html_key:
        download_block("Report EN (HTML)", en_html_key, "report_en.html")
    else:
        st.caption("EN HTML: not available yet (generated when report format is PDF).")
    if th_html_key:
        download_block("Report TH (HTML)", th_html_key, "report_th.html")
    else:
        st.caption("TH HTML: not available yet (generated when report format is PDF).")

reports_ready = bool(en_key) and bool(th_key) and s3_key_exists(en_key) and s3_key_exists(th_key)
skeleton_ready = bool(outputs.get("skeleton_video")) and s3_key_exists(outputs.get("skeleton_video", ""))
en_report_ready = bool(en_key) and s3_key_exists(en_key)
th_report_ready = bool(th_key) and s3_key_exists(th_key)
primary_done = th_report_ready

# Show ready artifacts immediately without waiting for all outputs.
ready_now = []
if th_report_ready and th_key:
    ready_now.append(("Report TH", th_key, th_name))
if en_report_ready and en_key:
    ready_now.append(("Report EN", en_key, en_name))
if skeleton_ready:
    ready_now.append(("Skeleton video", outputs.get("skeleton_video", ""), "skeleton.mp4"))

st.divider()
st.subheader("Ready to Download Now")
if st.button("Refresh output status", key="lpa_refresh_ready_downloads", width="content"):
    st.rerun()
if ready_now:
    for label, key, filename in ready_now:
        url = presigned_get_url(key, expires=3600, filename=filename)
        st.success(f"✅ {label} ready")
        st.link_button(f"Download {label}", url, width="stretch")
else:
    st.info("No files are ready yet. Files will appear here immediately when each one is done.")
st.caption("Each file appears as soon as it is ready. No need to wait for all outputs.")

st.divider()
st.subheader("Processing Status")
status_items = [
    ("Report TH (primary)", th_report_ready),
    ("Report EN (follow-up)", en_report_ready),
    ("Skeleton Video (follow-up)", skeleton_ready),
]
if primary_done:
    overall_pct = 100
else:
    overall_pct = int(round((sum(1 for _, ready in status_items if ready) / len(status_items)) * 100))
st.progress(overall_pct, text=f"Overall progress: {overall_pct}%")
for label, ready in status_items:
    item_pct = 100 if ready else 0
    st.progress(item_pct, text=f"{label}: {'ready' if ready else 'processing'} ({item_pct}%)")

# Clear step guidance for users while waiting.
if th_report_ready and en_report_ready and skeleton_ready:
    current_step = "All outputs are ready."
    next_step = "Download files below. Email delivery (report/skeleton) should complete shortly."
elif th_report_ready:
    current_step = "Primary result is ready: Report TH."
    next_step = "Other files (EN/Skeleton) will appear for download immediately when each one is done."
else:
    current_step = "Generating Report TH."
    next_step = "Download buttons will appear immediately once each file is ready."

st.info(f"Current step: {current_step}")
st.caption(f"Next step: {next_step}")
if th_report_ready:
    st.success("ขอบคุณที่ใช้ AI People Reader การวิเคราะห์ทั้งหมดจะถูกส่งไปในเมล์ของคุณหลังจากนี้")

if skeleton_ready and not th_report_ready:
    st.warning("Thai report is still not ready. You can re-run report generation for this group. (รายงานภาษาไทยยังไม่พร้อม สามารถสั่งสร้างรายงานใหม่ได้)")
    if st.button("Re-run report generation", width="content"):
        try:
            guessed_name = group_id.split("__", 1)[1] if "__" in group_id else "Anonymous"
            rerun_style = "simple"
            rerun_format = get_report_format_for_group(group_id)
            rerun_email = notify_email
            if not rerun_email:
                prev_notif = get_report_notification_status(group_id)
                rerun_email = str(prev_notif.get("notify_email") or "").strip()
            if rerun_email and ((not is_valid_email_format(rerun_email)) or is_blocked_typo_domain(rerun_email)):
                st.error("Cannot re-queue report job: invalid e-mail format. Please check e-mail again.")
                st.stop()
            new_report_key = enqueue_report_only_job(
                group_id=group_id,
                client_name=guessed_name,
                report_style=rerun_style,
                report_format=rerun_format,
                enterprise_folder=(enterprise_folder or "").strip(),
                notify_email=rerun_email,
            )
            st.success(f"Queued report job again ({rerun_style}, {rerun_format}): {new_report_key}")
        except Exception as e:
            st.error(f"Cannot re-queue report job: {format_submit_error_message(e)}")
