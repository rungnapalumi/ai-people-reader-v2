import os
import json
from datetime import datetime, timezone
from typing import Dict, List

import boto3
import streamlit as st
from zoneinfo import ZoneInfo

st.set_page_config(page_title="AI People Reader V2", layout="wide")

# Sidebar page names can be customized here.
HOME_PAGE_TITLE = "admi"
SUBMIT_PAGE_TITLE = "AI People Reader"
TTB_PAGE_TITLE = "TTB"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "0108"
JOBS_PENDING_PREFIX = "jobs/pending/"
JOBS_FINISHED_PREFIX = "jobs/finished/"
JOBS_FAILED_PREFIX = "jobs/failed/"
ORG_SETTINGS_PREFIX = "jobs/config/organizations/"
EMPLOYEE_REGISTRY_PREFIX = "jobs/config/employees/"
AWS_BUCKET = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
APP_LOCAL_TIMEZONE = os.getenv("APP_LOCAL_TIMEZONE", "Asia/Bangkok")


def _get_s3_client():
    return boto3.client("s3", region_name=AWS_REGION)


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
        "updated_at": str(payload.get("updated_at") or ""),
    }


def _save_org_settings(org_name: str, report_style: str, report_format: str) -> str:
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
    Return recent finished jobs with notification/email status for admin monitoring.
    """
    if not AWS_BUCKET:
        return []

    s3 = _get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    finished_objects: List[Dict[str, str]] = []

    for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=JOBS_FINISHED_PREFIX):
        for obj in page.get("Contents", []):
            key = str(obj.get("Key") or "")
            if key.endswith(".json"):
                finished_objects.append(
                    {
                        "key": key,
                        "last_modified": _to_local_time_display(obj.get("LastModified")),
                    }
                )

    # Show newest first.
    finished_objects.sort(key=lambda x: x["last_modified"], reverse=True)
    target = finished_objects[: max(1, int(limit))]

    rows: List[Dict[str, str]] = []
    for item in target:
        key = item["key"]
        try:
            obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
            raw = obj["Body"].read().decode("utf-8")
            payload = json.loads(raw)
        except Exception:
            continue

        notification = payload.get("notification") or {}
        notify_email = str(notification.get("notify_email") or payload.get("notify_email") or "").strip()
        sent_flag = bool(notification.get("sent"))
        sent_display = "yes" if sent_flag else "no"

        rows.append(
            {
                "job_id": str(payload.get("job_id") or key.split("/")[-1].replace(".json", "")),
                "employee_id": str(payload.get("employee_id") or ""),
                "employee_email": str(payload.get("employee_email") or payload.get("notify_email") or ""),
                "job_status": str(payload.get("status") or "finished"),
                "job_updated_at": _to_local_time_display(payload.get("updated_at") or item["last_modified"]),
                "sent_to_email": notify_email,
                "email_sent": sent_display,
                "email_status": str(notification.get("status") or ""),
                "email_updated_at": _to_local_time_display(notification.get("updated_at")),
            }
        )

    return rows


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
    if st.button("Logout", use_container_width=True):
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
    with st.form("org_settings_form", clear_on_submit=False):
        report_style_ui = st.selectbox("Default Report Type", options=["Full", "Simple"], index=0 if default_style_ui == "Full" else 1)
        report_format_ui = st.selectbox("Default Report File", options=["DOCX", "PDF"], index=0 if default_format_ui == "DOCX" else 1)
        save_org = st.form_submit_button("Save Organization Settings", type="primary")
    if save_org:
        try:
            saved_key = _save_org_settings(
                org_name=org_name,
                report_style="simple" if report_style_ui == "Simple" else "full",
                report_format="pdf" if report_format_ui == "PDF" else "docx",
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
            st.dataframe(org_rows, use_container_width=True, hide_index=True)
        else:
            st.caption("No organization settings yet.")

    with st.expander("View employee ID / email / password table", expanded=False):
        try:
            employee_rows = _list_employee_registry(limit=1000)
        except Exception as e:
            st.error(f"Unable to load employee registry: {e}")
            employee_rows = []
        if employee_rows:
            st.dataframe(employee_rows, use_container_width=True, hide_index=True)
        else:
            st.caption("No employee data yet.")

    st.markdown("---")
    st.markdown("### Pending Jobs")
    try:
        pending_rows = _list_pending_jobs()
    except Exception as e:
        st.error(f"Unable to load pending jobs: {e}")
        return

    st.write(f"Total pending jobs: **{len(pending_rows)}**")
    if pending_rows:
        st.dataframe(pending_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No pending jobs found.")

    st.markdown("---")
    st.markdown("### Finished Jobs & Email Monitor")
    st.caption(f"Time zone: {APP_LOCAL_TIMEZONE}")
    monitor_limit = st.slider("Show recent finished jobs", min_value=10, max_value=200, value=50, step=10)
    try:
        finished_rows = _list_finished_jobs_with_email(limit=monitor_limit)
    except Exception as e:
        st.error(f"Unable to load finished jobs/email monitor: {e}")
        finished_rows = []

    if finished_rows:
        sent_count = sum(1 for row in finished_rows if row.get("email_sent") == "yes")
        st.write(f"Loaded jobs: **{len(finished_rows)}** | Email sent: **{sent_count}**")
        st.dataframe(finished_rows, use_container_width=True, hide_index=True)
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
        st.dataframe(failed_rows, use_container_width=True, hide_index=True)
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
        use_container_width=True,
        disabled=not can_clear,
    ):
        try:
            deleted = _clear_pending_jobs()
            st.success(f"Cleared pending jobs: {deleted}")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to clear pending jobs: {e}")


def _render_home() -> None:
    st.title("AI People Reader V2")
    st.caption("Go to left menu -> AI People Reader")
    _render_admin_panel()


if hasattr(st, "Page") and hasattr(st, "navigation"):
    nav = st.navigation(
        [
            st.Page(_render_home, title=HOME_PAGE_TITLE),
            st.Page("pages/2_Submit_Job.py", title=SUBMIT_PAGE_TITLE),
            st.Page("pages/3_TTB.py", title=TTB_PAGE_TITLE),
        ]
    )
    nav.run()
else:
    # Backward-compatible fallback for older Streamlit versions.
    _render_home()
