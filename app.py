import os
import json
from typing import Dict, List

import boto3
import streamlit as st

st.set_page_config(page_title="AI People Reader V2", layout="wide")

# Sidebar page names can be customized here.
HOME_PAGE_TITLE = "AI Peopple Reader 1"
SUBMIT_PAGE_TITLE = "AI people Reader 2"
TTB_PAGE_TITLE = "TTB"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "0108"
JOBS_PENDING_PREFIX = "jobs/pending/"
JOBS_FINISHED_PREFIX = "jobs/finished/"
AWS_BUCKET = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")


def _get_s3_client():
    return boto3.client("s3", region_name=AWS_REGION)


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
                    "last_modified": str(obj.get("LastModified") or ""),
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
                        "last_modified": str(obj.get("LastModified") or ""),
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
                "job_status": str(payload.get("status") or "finished"),
                "job_updated_at": str(payload.get("updated_at") or item["last_modified"]),
                "sent_to_email": notify_email,
                "email_sent": sent_display,
                "email_status": str(notification.get("status") or ""),
                "email_updated_at": str(notification.get("updated_at") or ""),
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
    st.caption("Go to left menu -> AI people Reader 2")
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
