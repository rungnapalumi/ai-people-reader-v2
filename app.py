import os
import json
import base64
from datetime import datetime, timezone
from typing import Any, Dict, List

import boto3
import streamlit as st
from zoneinfo import ZoneInfo

st.set_page_config(page_title="AI People Reader V2", layout="wide")


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

h1, h2, h3, h4, h5, h6 {
  color: #f0e4d4 !important;
}

p, label, span, div {
  color: var(--text-main);
}

[data-testid="stMarkdownContainer"] p {
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

.stButton > button,
.stDownloadButton > button {
  background: linear-gradient(180deg, var(--accent), var(--accent-strong)) !important;
  color: #231d17 !important;
  border: 0 !important;
  font-weight: 600 !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
  filter: brightness(1.05);
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


def _apply_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)

# Sidebar page names can be customized here.
HOME_PAGE_TITLE = "Admin"
SUBMIT_PAGE_TITLE = "AI People Reader"
TTB_PAGE_TITLE = "TTB"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "0108"
JOBS_PENDING_PREFIX = "jobs/pending/"
JOBS_PROCESSING_PREFIX = "jobs/processing/"
JOBS_FINISHED_PREFIX = "jobs/finished/"
JOBS_FAILED_PREFIX = "jobs/failed/"
JOBS_EMAIL_PENDING_PREFIX = "jobs/email_pending/"
JOBS_OUTPUT_GROUPS_PREFIX = "jobs/output/groups/"
ORG_SETTINGS_PREFIX = "jobs/config/organizations/"
EMPLOYEE_REGISTRY_PREFIX = "jobs/config/employees/"
AWS_BUCKET = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
APP_LOCAL_TIMEZONE = os.getenv("APP_LOCAL_TIMEZONE", "Asia/Bangkok")

BANNER_PATH_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "assets", "top_banner.png"),
    os.path.join(os.path.dirname(__file__), "assets", "banner.png"),
    os.path.join(os.path.dirname(__file__), "Header.png"),
]
TTB_SIDEBAR_LOGO_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "assets", "ttb_logo.png"),
    os.path.join(os.path.dirname(__file__), "assets", "ttb.png"),
]


def _get_s3_client():
    return boto3.client("s3", region_name=AWS_REGION)


def _render_top_banner() -> None:
    for path in BANNER_PATH_CANDIDATES:
        if os.path.exists(path):
            st.image(path, width="stretch")
            return


def _inject_ttb_sidebar_logo() -> None:
    logo_path = ""
    for path in TTB_SIDEBAR_LOGO_CANDIDATES:
        if os.path.exists(path):
            logo_path = path
            break
    if not logo_path:
        return

    try:
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode("ascii")
    except Exception:
        return

    st.markdown(
        f"""
<style>
[data-testid="stSidebarNav"] a[aria-label="TTB"] p,
[data-testid="stSidebarNav"] a[aria-label="TTB"] span,
[data-testid="stSidebarNav"] button[aria-label="TTB"] p,
[data-testid="stSidebarNav"] button[aria-label="TTB"] span {{
  font-size: 0 !important;
  line-height: 0 !important;
}}

[data-testid="stSidebarNav"] a[aria-label="TTB"] p::before,
[data-testid="stSidebarNav"] a[aria-label="TTB"] span::before,
[data-testid="stSidebarNav"] button[aria-label="TTB"] p::before,
[data-testid="stSidebarNav"] button[aria-label="TTB"] span::before {{
  content: "";
  display: inline-block;
  width: 60px;
  height: 34px;
  background-image: url("data:image/png;base64,{logo_b64}");
  background-repeat: no-repeat;
  background-size: contain;
  background-position: left center;
  vertical-align: middle;
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def _get_local_tz():
    try:
        return ZoneInfo(APP_LOCAL_TIMEZONE)
    except Exception:
        return timezone.utc


def _to_local_time_display(value) -> str:
    """
    Convert datetimes from S3/job JSON to local timezone display.
    """
    if value is None:
        return ""

    dt = None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        # Handle common ISO variants, including trailing Z.
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return text
    else:
        return str(value)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local_dt = dt.astimezone(_get_local_tz())
    return local_dt.strftime("%Y-%m-%d %H:%M:%S %Z")


def _normalize_org_name(name: str) -> str:
    text = str(name or "").strip().lower()
    if not text:
        return ""
    out = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        elif ch.isspace():
            out.append("_")
    normalized = "".join(out).strip("_")
    return normalized


def _org_settings_key(org_name: str) -> str:
    org_id = _normalize_org_name(org_name)
    return f"{ORG_SETTINGS_PREFIX}{org_id}.json"


def _get_org_settings(org_name: str) -> Dict[str, str]:
    if not AWS_BUCKET:
        return {}
    org_id = _normalize_org_name(org_name)
    if not org_id:
        return {}
    s3 = _get_s3_client()
    try:
        obj = s3.get_object(Bucket=AWS_BUCKET, Key=_org_settings_key(org_name))
        payload = json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return {}

    style = str(payload.get("report_style") or "").strip().lower()
    fmt = str(payload.get("report_format") or "").strip().lower()
    if style not in ("full", "simple"):
        style = "full"
    if fmt not in ("docx", "pdf"):
        fmt = "docx"
    return {
        "organization_name": str(payload.get("organization_name") or org_name).strip(),
        "organization_id": org_id,
        "report_style": style,
        "report_format": fmt,
        "default_page": str(payload.get("default_page") or "").strip().lower(),
        "updated_at": str(payload.get("updated_at") or ""),
    }


def _save_org_settings(org_name: str, report_style: str, report_format: str, default_page: str = "") -> str:
    if not AWS_BUCKET:
        raise RuntimeError("Missing AWS_BUCKET (or S3_BUCKET)")

    org_id = _normalize_org_name(org_name)
    if not org_id:
        raise ValueError("Organization name is required")

    style = str(report_style or "").strip().lower()
    fmt = str(report_format or "").strip().lower()
    if style not in ("full", "simple"):
        raise ValueError("Invalid report_style")
    if fmt not in ("docx", "pdf"):
        raise ValueError("Invalid report_format")

    payload = {
        "organization_name": str(org_name).strip(),
        "organization_id": org_id,
        "report_style": style,
        "report_format": fmt,
        "default_page": str(default_page or "").strip().lower(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    key = _org_settings_key(org_name)
    s3 = _get_s3_client()
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json; charset=utf-8",
    )
    return key


def _list_org_settings(limit: int = 200) -> List[Dict[str, str]]:
    if not AWS_BUCKET:
        return []
    s3 = _get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    rows: List[Dict[str, str]] = []
    for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=ORG_SETTINGS_PREFIX):
        for item in page.get("Contents", []):
            key = str(item.get("Key") or "")
            if not key.endswith(".json"):
                continue
            try:
                obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
                payload = json.loads(obj["Body"].read().decode("utf-8"))
            except Exception:
                continue
            rows.append(
                {
                    "organization_name": str(payload.get("organization_name") or ""),
                    "organization_id": str(payload.get("organization_id") or ""),
                    "report_style": str(payload.get("report_style") or ""),
                    "report_format": str(payload.get("report_format") or ""),
                    "default_page": str(payload.get("default_page") or ""),
                    "updated_at": _to_local_time_display(payload.get("updated_at")),
                }
            )
    rows.sort(key=lambda x: x.get("organization_name", "").lower())
    return rows[: max(1, int(limit))]


def _list_employee_registry(limit: int = 500) -> List[Dict[str, str]]:
    if not AWS_BUCKET:
        return []
    s3 = _get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    rows: List[Dict[str, str]] = []
    for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=EMPLOYEE_REGISTRY_PREFIX):
        for item in page.get("Contents", []):
            key = str(item.get("Key") or "")
            if not key.endswith(".json"):
                continue
            try:
                obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
                payload = json.loads(obj["Body"].read().decode("utf-8"))
            except Exception:
                continue
            rows.append(
                {
                    "employee_id": str(payload.get("employee_id") or ""),
                    "employee_email": str(payload.get("employee_email") or ""),
                    "employee_password": str(payload.get("employee_password") or ""),
                    "organization_name": str(payload.get("organization_name") or ""),
                    "updated_at": _to_local_time_display(payload.get("updated_at")),
                }
            )
    rows.sort(key=lambda x: (x.get("employee_id") or "").lower())
    return rows[: max(1, int(limit))]


def _list_pending_jobs() -> List[Dict[str, str]]:
    if not AWS_BUCKET:
        return []
    s3 = _get_s3_client()
    rows: List[Dict[str, str]] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=JOBS_PENDING_PREFIX):
        for obj in page.get("Contents", []):
            key = str(obj.get("Key") or "")
            if not key.endswith(".json"):
                continue
            rows.append(
                {
                    "key": key,
                    "last_modified": _to_local_time_display(obj.get("LastModified")),
                    "size_bytes": f"{int(obj.get('Size') or 0):,}",
                }
            )
    rows.sort(key=lambda x: x["key"])
    return rows


def _clear_pending_jobs() -> int:
    """Delete all pending JSON jobs and return deleted count."""
    pending_rows = _list_pending_jobs()
    if not pending_rows or not AWS_BUCKET:
        return 0

    s3 = _get_s3_client()
    keys = [row["key"] for row in pending_rows]
    deleted = 0
    for i in range(0, len(keys), 1000):
        batch = keys[i : i + 1000]
        resp = s3.delete_objects(
            Bucket=AWS_BUCKET,
            Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True},
        )
        deleted += len(resp.get("Deleted", []))
    return deleted


def _list_finished_jobs_with_email(limit: int = 50) -> List[Dict[str, str]]:
    """
    Return recent finished jobs grouped by group_id for admin monitoring.
    """
    if not AWS_BUCKET:
        return []

    s3 = _get_s3_client()
    def _as_dt(v: Any) -> datetime:
        if isinstance(v, datetime):
            return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        text = str(v or "").strip()
        if not text:
            return datetime.min.replace(tzinfo=timezone.utc)
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    def _collect_json_objects(prefixes: List[str], max_items: int) -> List[Dict[str, str]]:
        paginator = s3.get_paginator("list_objects_v2")
        out: List[Dict[str, str]] = []
        for prefix in prefixes:
            for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = str(obj.get("Key") or "")
                    if not key.endswith(".json"):
                        continue
                    out.append(
                        {
                            "key": key,
                            "source_prefix": prefix,
                            "last_modified_raw": str(obj.get("LastModified") or ""),
                        }
                    )
                    if len(out) >= max_items:
                        break
                if len(out) >= max_items:
                    break
            if len(out) >= max_items:
                break
        out.sort(key=lambda x: _as_dt(x.get("last_modified_raw")), reverse=True)
        return out[:max_items]

    def _collect_output_group_summaries(max_groups: int) -> Dict[str, Dict[str, Any]]:
        paginator = s3.get_paginator("list_objects_v2")
        groups: Dict[str, Dict[str, Any]] = {}
        for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=JOBS_OUTPUT_GROUPS_PREFIX):
            for obj in page.get("Contents", []):
                key = str(obj.get("Key") or "")
                if key.endswith("/") or key == JOBS_OUTPUT_GROUPS_PREFIX:
                    continue
                parts = key.split("/")
                if len(parts) < 5:  # jobs/output/groups/<group_id>/<file>
                    continue
                group_id = str(parts[3] or "").strip()
                file_name = str(parts[-1] or "").lower()
                if not group_id:
                    continue
                g = groups.setdefault(
                    group_id,
                    {
                        "job_updated_at_raw": "",
                        "modes_done": set(),
                        "jobs_in_group": 0,
                    },
                )
                g["jobs_in_group"] = int(g.get("jobs_in_group") or 0) + 1
                mod_raw = str(obj.get("LastModified") or "")
                if _as_dt(mod_raw) > _as_dt(g.get("job_updated_at_raw")):
                    g["job_updated_at_raw"] = mod_raw
                if file_name == "dots.mp4":
                    g["modes_done"].add("dots")
                elif file_name == "skeleton.mp4":
                    g["modes_done"].add("skeleton")
                elif file_name.endswith(".docx") or file_name.endswith(".pdf"):
                    g["modes_done"].add("report")
                if len(groups) >= max_groups:
                    return groups
        return groups

    max_scan = max(200, min(2000, int(limit) * 10))
    all_json = _collect_json_objects(
        [
            JOBS_FINISHED_PREFIX,
            JOBS_PROCESSING_PREFIX,
            JOBS_PENDING_PREFIX,
            JOBS_FAILED_PREFIX,
            JOBS_EMAIL_PENDING_PREFIX,
        ],
        max_items=max_scan,
    )

    grouped: Dict[str, Dict[str, Any]] = {}
    for item in all_json:
        key = item["key"]
        source_prefix = str(item.get("source_prefix") or "")
        key_last_modified_raw = str(item.get("last_modified_raw") or "")
        payload: Dict[str, Any] = {}
        try:
            obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
            payload = json.loads(obj["Body"].read().decode("utf-8"))
        except Exception:
            # Keep legacy/corrupt rows visible in monitor instead of hiding all data.
            payload = {}

        group_id = str(payload.get("group_id") or payload.get("group") or "").strip()
        if not group_id:
            # Backward compatibility for older jobs that don't store group_id.
            fallback_job_id = str(payload.get("job_id") or "").strip()
            if fallback_job_id:
                group_id = fallback_job_id
            else:
                group_id = f"legacy::{key.split('/')[-1].replace('.json', '')}"

        g = grouped.setdefault(
            group_id,
            {
                "group_id": group_id,
                "employee_id": "",
                "employee_email": "",
                "job_status": "",
                "job_updated_at_raw": "",
                "modes_done": set(),
                "jobs_in_group": 0,
                "sent_to_email": "",
                "email_sent": "no",
                "email_status": "",
                "email_updated_at_raw": "",
            },
        )

        g["jobs_in_group"] = int(g.get("jobs_in_group") or 0) + 1
        mode = str(payload.get("mode") or "").strip().lower()
        if mode:
            g["modes_done"].add(mode)

        emp_id = str(payload.get("employee_id") or "").strip()
        emp_email = str(payload.get("employee_email") or payload.get("notify_email") or "").strip()
        if emp_id and not g["employee_id"]:
            g["employee_id"] = emp_id
        if emp_email and not g["employee_email"]:
            g["employee_email"] = emp_email

        status_text = str(payload.get("status") or "").strip().lower()
        if not status_text:
            if source_prefix.startswith(JOBS_FINISHED_PREFIX):
                status_text = "finished"
            elif source_prefix.startswith(JOBS_PROCESSING_PREFIX):
                status_text = "processing"
            elif source_prefix.startswith(JOBS_PENDING_PREFIX):
                status_text = "pending"
            elif source_prefix.startswith(JOBS_FAILED_PREFIX):
                status_text = "failed"
            elif source_prefix.startswith(JOBS_EMAIL_PENDING_PREFIX):
                status_text = "email_pending"

        updated_at_raw = str(payload.get("updated_at") or key_last_modified_raw or "")
        if _as_dt(updated_at_raw) >= _as_dt(g.get("job_updated_at_raw")):
            g["job_updated_at_raw"] = updated_at_raw
            if status_text:
                g["job_status"] = status_text

        if not str(g.get("job_status") or "").strip():
            if source_prefix.startswith(JOBS_FINISHED_PREFIX):
                g["job_status"] = "finished"
            elif source_prefix.startswith(JOBS_PROCESSING_PREFIX):
                g["job_status"] = "processing"
            elif source_prefix.startswith(JOBS_PENDING_PREFIX):
                g["job_status"] = "pending"
            elif source_prefix.startswith(JOBS_FAILED_PREFIX):
                g["job_status"] = "failed"
            elif source_prefix.startswith(JOBS_EMAIL_PENDING_PREFIX):
                g["job_status"] = "email_pending"
            else:
                g["job_status"] = "unknown"

        notification = payload.get("notification") or {}
        # Support both current notification object and legacy flat fields.
        legacy_notify_email = str(payload.get("sent_to_email") or "").strip()
        notify_email = str(notification.get("notify_email") or payload.get("notify_email") or legacy_notify_email).strip()
        if notify_email:
            g["sent_to_email"] = notify_email

        legacy_sent = payload.get("email_sent")
        legacy_sent_bool = str(legacy_sent).strip().lower() in ("1", "true", "yes", "y")
        if bool(notification.get("sent")) or legacy_sent_bool:
            g["email_sent"] = "yes"
        email_status = str(notification.get("status") or payload.get("email_status") or "").strip()
        if not email_status and source_prefix.startswith(JOBS_EMAIL_PENDING_PREFIX):
            email_status = "waiting_for_all_outputs"
        if email_status:
            g["email_status"] = email_status

        email_updated_raw = str(notification.get("updated_at") or payload.get("email_updated_at") or "")
        if _as_dt(email_updated_raw) > _as_dt(g.get("email_updated_at_raw")):
            g["email_updated_at_raw"] = email_updated_raw

    # Only scan output fallback when no job JSON rows are available.
    if not grouped:
        output_groups = _collect_output_group_summaries(max_groups=max(100, min(2000, int(limit) * 20)))
        for group_id, og in output_groups.items():
            g = grouped.setdefault(
                group_id,
                {
                    "group_id": group_id,
                    "employee_id": "",
                    "employee_email": "",
                    "job_status": "finished_from_outputs",
                    "job_updated_at_raw": "",
                    "modes_done": set(),
                    "jobs_in_group": 0,
                    "sent_to_email": "",
                    "email_sent": "unknown",
                    "email_status": "no_job_json_found",
                    "email_updated_at_raw": "",
                },
            )
            g["jobs_in_group"] = max(int(g.get("jobs_in_group") or 0), int(og.get("jobs_in_group") or 0))
            g["modes_done"].update(og.get("modes_done") or set())
            if _as_dt(og.get("job_updated_at_raw")) > _as_dt(g.get("job_updated_at_raw")):
                g["job_updated_at_raw"] = str(og.get("job_updated_at_raw") or "")
            if not str(g.get("job_status") or "").strip():
                g["job_status"] = "finished_from_outputs"

    rows: List[Dict[str, str]] = []
    for g in grouped.values():
        modes_done = sorted(list(g.get("modes_done") or []))
        rows.append(
            {
                "group_id": str(g.get("group_id") or ""),
                "jobs_in_group": str(g.get("jobs_in_group") or 0),
                "modes_done": ", ".join(modes_done),
                "employee_id": str(g.get("employee_id") or ""),
                "employee_email": str(g.get("employee_email") or ""),
                "job_status": str(g.get("job_status") or ""),
                "job_updated_at_raw": str(g.get("job_updated_at_raw") or ""),
                "job_updated_at": _to_local_time_display(g.get("job_updated_at_raw")),
                "sent_to_email": str(g.get("sent_to_email") or ""),
                "email_sent": str(g.get("email_sent") or "no"),
                "email_status": str(g.get("email_status") or ""),
                "email_updated_at": _to_local_time_display(g.get("email_updated_at_raw")),
            }
        )

    rows.sort(key=lambda x: _as_dt(x.get("job_updated_at_raw")), reverse=True)
    return rows[: max(1, int(limit))]


def _list_failed_jobs(limit: int = 50) -> List[Dict[str, str]]:
    """
    Return recent failed jobs for admin monitoring.
    """
    if not AWS_BUCKET:
        return []

    s3 = _get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    failed_objects: List[Dict[str, str]] = []
    for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=JOBS_FAILED_PREFIX):
        for obj in page.get("Contents", []):
            key = str(obj.get("Key") or "")
            if key.endswith(".json"):
                failed_objects.append(
                    {
                        "key": key,
                        "last_modified": _to_local_time_display(obj.get("LastModified")),
                    }
                )

    failed_objects.sort(key=lambda x: x["last_modified"], reverse=True)
    target = failed_objects[: max(1, int(limit))]

    rows: List[Dict[str, str]] = []
    for item in target:
        key = item["key"]
        try:
            obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
            payload = json.loads(obj["Body"].read().decode("utf-8"))
        except Exception:
            continue

        notification = payload.get("notification") or {}
        rows.append(
            {
                "job_id": str(payload.get("job_id") or key.split("/")[-1].replace(".json", "")),
                "employee_id": str(payload.get("employee_id") or ""),
                "employee_email": str(payload.get("employee_email") or payload.get("notify_email") or ""),
                "mode": str(payload.get("mode") or ""),
                "job_status": str(payload.get("status") or "failed"),
                "job_updated_at": _to_local_time_display(payload.get("updated_at") or item["last_modified"]),
                "error": str(payload.get("error") or ""),
                "sent_to_email": str(notification.get("notify_email") or payload.get("notify_email") or "").strip(),
                "email_status": str(notification.get("status") or ""),
                "email_updated_at": _to_local_time_display(notification.get("updated_at")),
            }
        )

    return rows


def _queue_prefix_snapshot(prefix: str, sample_limit: int = 3) -> Dict[str, Any]:
    """
    Return lightweight queue diagnostics for admin UI.
    """
    if not AWS_BUCKET:
        return {"prefix": prefix, "count": 0, "sample_keys": []}
    s3 = _get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    sample: List[str] = []
    for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=prefix):
        for item in page.get("Contents", []):
            key = str(item.get("Key") or "")
            if not key.endswith(".json"):
                continue
            count += 1
            if len(sample) < sample_limit:
                sample.append(key)
    return {"prefix": prefix, "count": count, "sample_keys": sample}


def _list_submission_inputs(limit: int = 300) -> List[Dict[str, str]]:
    """
    Show submitted input metadata across all job queues, grouped by group_id.
    """
    if not AWS_BUCKET:
        return []

    s3 = _get_s3_client()
    prefixes = [
        JOBS_FINISHED_PREFIX,
        JOBS_PROCESSING_PREFIX,
        JOBS_PENDING_PREFIX,
        JOBS_FAILED_PREFIX,
    ]
    paginator = s3.get_paginator("list_objects_v2")
    grouped: Dict[str, Dict[str, Any]] = {}

    def _prefix_status(prefix: str) -> str:
        if prefix.startswith(JOBS_FINISHED_PREFIX):
            return "finished"
        if prefix.startswith(JOBS_PROCESSING_PREFIX):
            return "processing"
        if prefix.startswith(JOBS_PENDING_PREFIX):
            return "pending"
        if prefix.startswith(JOBS_FAILED_PREFIX):
            return "failed"
        return "unknown"

    # Prevent OOM by scanning only a bounded number of most-recent keys.
    raw_items: List[Dict[str, str]] = []
    max_scan = max(300, min(3000, int(limit) * 12))
    for prefix in prefixes:
        for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=prefix):
            for item in page.get("Contents", []):
                key = str(item.get("Key") or "")
                if not key.endswith(".json"):
                    continue
                raw_items.append(
                    {
                        "prefix": prefix,
                        "key": key,
                        "last_modified_raw": str(item.get("LastModified") or ""),
                    }
                )
                if len(raw_items) >= max_scan:
                    break
            if len(raw_items) >= max_scan:
                break
        if len(raw_items) >= max_scan:
            break

    raw_items.sort(key=lambda x: str(x.get("last_modified_raw") or ""), reverse=True)
    target_items = raw_items[:max_scan]

    for item in target_items:
        prefix = str(item.get("prefix") or "")
        key = str(item.get("key") or "")
        last_modified_raw = str(item.get("last_modified_raw") or "")
        payload: Dict[str, Any] = {}
        try:
            obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
            payload = json.loads(obj["Body"].read().decode("utf-8"))
        except Exception:
            payload = {}

        group_id = str(payload.get("group_id") or "").strip()
        if not group_id:
            fallback_job_id = str(payload.get("job_id") or "").strip()
            group_id = fallback_job_id or f"legacy::{key.split('/')[-1].replace('.json', '')}"

        g = grouped.setdefault(
            group_id,
            {
                "group_id": group_id,
                "input_key": "",
                "organization_name": "",
                "notify_email": "",
                "employee_id": "",
                "employee_email": "",
                "status": "",
                "modes": set(),
                "updated_at_raw": "",
                "source_keys": 0,
            },
        )
        g["source_keys"] = int(g.get("source_keys") or 0) + 1

        mode = str(payload.get("mode") or "").strip().lower()
        if mode:
            g["modes"].add(mode)

        input_key = str(payload.get("input_key") or "").strip()
        if input_key and not g["input_key"]:
            g["input_key"] = input_key

        org_name = str(payload.get("enterprise_folder") or "").strip()
        if org_name and not g["organization_name"]:
            g["organization_name"] = org_name

        notify_email = str(payload.get("notify_email") or "").strip()
        if notify_email and not g["notify_email"]:
            g["notify_email"] = notify_email

        emp_id = str(payload.get("employee_id") or "").strip()
        if emp_id and not g["employee_id"]:
            g["employee_id"] = emp_id

        emp_email = str(payload.get("employee_email") or "").strip()
        if emp_email and not g["employee_email"]:
            g["employee_email"] = emp_email

        status = str(payload.get("status") or "").strip().lower() or _prefix_status(prefix)
        g["status"] = status

        updated_at_raw = str(payload.get("updated_at") or last_modified_raw or "")
        if updated_at_raw and updated_at_raw > str(g.get("updated_at_raw") or ""):
            g["updated_at_raw"] = updated_at_raw

    rows: List[Dict[str, str]] = []
    for g in grouped.values():
        rows.append(
            {
                "group_id": str(g.get("group_id") or ""),
                "organization_name": str(g.get("organization_name") or ""),
                "employee_id": str(g.get("employee_id") or ""),
                "employee_email": str(g.get("employee_email") or ""),
                "notify_email": str(g.get("notify_email") or ""),
                "input_key": str(g.get("input_key") or ""),
                "status": str(g.get("status") or ""),
                "modes": ", ".join(sorted(list(g.get("modes") or []))),
                "updated_at": _to_local_time_display(g.get("updated_at_raw")),
                "source_keys": str(g.get("source_keys") or 0),
            }
        )

    rows.sort(key=lambda x: str(x.get("updated_at") or ""), reverse=True)
    return rows[: max(1, int(limit))]


def _render_admin_panel() -> None:
    st.subheader("Admin")
    st.caption("Login to view and clear pending jobs.")

    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False

    if not st.session_state.admin_authenticated:
        with st.form("admin_login_form", clear_on_submit=False):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_clicked = st.form_submit_button("Login")
        if login_clicked:
            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                st.success("Login successful.")
                st.rerun()
            else:
                st.error("Invalid username or password.")
        return

    st.success("Logged in as admin")
    if st.button("Logout", width="stretch"):
        st.session_state.admin_authenticated = False
        st.rerun()

    if not AWS_BUCKET:
        st.error("Missing AWS_BUCKET (or S3_BUCKET) environment variable.")
        return

    st.markdown("---")
    st.markdown("### Organization Settings")
    st.caption("Create or update per-organization report defaults (used by AI People Reader submit page).")

    org_name = st.text_input(
        "Organization (search/create)",
        value=st.session_state.get("admin_org_name", ""),
        placeholder="e.g., TTB / ACME Group",
        key="admin_org_name",
    )

    existing_org_cfg: Dict[str, str] = {}
    if org_name.strip():
        existing_org_cfg = _get_org_settings(org_name)
        if existing_org_cfg:
            st.info(
                f"Found settings for `{existing_org_cfg.get('organization_name')}` "
                f"(updated: {_to_local_time_display(existing_org_cfg.get('updated_at'))})"
            )
        else:
            st.warning("No settings found for this organization yet. You can create it below.")

    default_style_ui = "Simple" if existing_org_cfg.get("report_style") == "simple" else "Full"
    default_format_ui = "PDF" if existing_org_cfg.get("report_format") == "pdf" else "DOCX"
    page_options = {
        "Any page": "",
        "AI People Reader page": "ai_people_reader",
        "TTB page": "ttb",
    }
    existing_page_value = str(existing_org_cfg.get("default_page") or "").strip().lower()
    existing_page_label = next((k for k, v in page_options.items() if v == existing_page_value), "Any page")

    with st.form("org_settings_form", clear_on_submit=False):
        report_style_ui = st.selectbox("Default Report Type", options=["Full", "Simple"], index=0 if default_style_ui == "Full" else 1)
        report_format_ui = st.selectbox("Default Report File", options=["DOCX", "PDF"], index=0 if default_format_ui == "DOCX" else 1)
        default_page_ui = st.selectbox(
            "Default Page",
            options=list(page_options.keys()),
            index=list(page_options.keys()).index(existing_page_label),
            help="If set, that page will auto-fill this organization as default.",
        )
        save_org = st.form_submit_button("Save Organization Settings", type="primary")
    if save_org:
        try:
            saved_key = _save_org_settings(
                org_name=org_name,
                report_style="simple" if report_style_ui == "Simple" else "full",
                report_format="pdf" if report_format_ui == "PDF" else "docx",
                default_page=page_options.get(default_page_ui, ""),
            )
            st.success(f"Saved organization settings: {saved_key}")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to save organization settings: {e}")

    with st.expander("View all organization settings", expanded=False):
        try:
            org_rows = _list_org_settings(limit=500)
        except Exception as e:
            st.error(f"Unable to load organization settings: {e}")
            org_rows = []
        if org_rows:
            st.dataframe(org_rows, width="stretch", hide_index=True)
        else:
            st.caption("No organization settings yet.")

    with st.expander("View employee ID / email / password table", expanded=False):
        try:
            employee_rows = _list_employee_registry(limit=1000)
        except Exception as e:
            st.error(f"Unable to load employee registry: {e}")
            employee_rows = []
        if employee_rows:
            st.dataframe(employee_rows, width="stretch", hide_index=True)
        else:
            st.caption("No employee data yet.")

    with st.expander("All Submission Inputs", expanded=False):
        input_limit = st.slider("Show recent submission input rows", min_value=50, max_value=1000, value=300, step=50)
        try:
            input_rows = _list_submission_inputs(limit=input_limit)
        except Exception as e:
            st.error(f"Unable to load submission inputs: {e}")
            input_rows = []
        if input_rows:
            st.dataframe(input_rows, width="stretch", hide_index=True)
        else:
            st.caption("No submission input rows found.")

    st.markdown("---")
    st.markdown("### Pending Jobs")
    try:
        pending_rows = _list_pending_jobs()
    except Exception as e:
        st.error(f"Unable to load pending jobs: {e}")
        return

    st.write(f"Total pending jobs: **{len(pending_rows)}**")
    if pending_rows:
        st.dataframe(pending_rows, width="stretch", hide_index=True)
    else:
        st.info("No pending jobs found.")

    st.markdown("---")
    st.markdown("### Finished Jobs & Email Monitor")
    st.caption(f"Time zone: {APP_LOCAL_TIMEZONE}")
    with st.expander("Monitor debug (job JSON queue snapshot)", expanded=False):
        try:
            snapshots = [
                _queue_prefix_snapshot(JOBS_FINISHED_PREFIX),
                _queue_prefix_snapshot(JOBS_PROCESSING_PREFIX),
                _queue_prefix_snapshot(JOBS_PENDING_PREFIX),
                _queue_prefix_snapshot(JOBS_FAILED_PREFIX),
                _queue_prefix_snapshot(JOBS_EMAIL_PENDING_PREFIX),
            ]
            st.dataframe(snapshots, width="stretch", hide_index=True)
        except Exception as e:
            st.error(f"Unable to load queue snapshot: {e}")
    monitor_limit = st.slider("Show recent finished jobs", min_value=10, max_value=200, value=50, step=10)
    try:
        finished_rows = _list_finished_jobs_with_email(limit=monitor_limit)
    except Exception as e:
        st.error(f"Unable to load finished jobs/email monitor: {e}")
        finished_rows = []

    if finished_rows:
        sent_count = sum(1 for row in finished_rows if row.get("email_sent") == "yes")
        st.write(f"Loaded jobs: **{len(finished_rows)}** | Email sent: **{sent_count}**")
        st.dataframe(finished_rows, width="stretch", hide_index=True)
    else:
        st.info("No finished jobs found.")

    st.markdown("---")
    st.markdown("### Failed Jobs Monitor")
    failed_limit = st.slider("Show recent failed jobs", min_value=10, max_value=200, value=50, step=10)
    try:
        failed_rows = _list_failed_jobs(limit=failed_limit)
    except Exception as e:
        st.error(f"Unable to load failed jobs monitor: {e}")
        failed_rows = []

    if failed_rows:
        st.write(f"Loaded failed jobs: **{len(failed_rows)}**")
        st.dataframe(failed_rows, width="stretch", hide_index=True)
    else:
        st.info("No failed jobs found.")

    st.markdown("---")
    st.markdown("### Clear Pending Jobs")
    st.warning("Danger zone: this will permanently delete all pending job JSON files.")
    confirm_checked = st.checkbox("I understand this action cannot be undone.")
    confirm_text = st.text_input('Type "CLEAR" to confirm')
    can_clear = confirm_checked and (confirm_text.strip().upper() == "CLEAR")

    if st.button(
        "Clear Pending Jobs",
        type="primary",
        width="stretch",
        disabled=not can_clear,
    ):
        try:
            deleted = _clear_pending_jobs()
            st.success(f"Cleared pending jobs: {deleted}")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to clear pending jobs: {e}")


def _render_home() -> None:
    _apply_theme()
    _render_top_banner()
    st.title("AI People Reader V2")
    st.caption("Go to left menu -> AI People Reader")
    _render_admin_panel()


if hasattr(st, "Page") and hasattr(st, "navigation"):
    nav = st.navigation(
        [
            st.Page(_render_home, title=HOME_PAGE_TITLE),
            st.Page("pages/2_SkillLane.py", title=SUBMIT_PAGE_TITLE),
            st.Page("pages/3_TTB.py", title=TTB_PAGE_TITLE),
        ]
    )
    _inject_ttb_sidebar_logo()
    nav.run()
else:
    # Backward-compatible fallback for older Streamlit versions.
    _render_home()
