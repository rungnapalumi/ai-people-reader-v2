## pages/8_People_Reader.py
# People Reader page (from Training flow)
# Flow: Upload -> S3 -> Queue -> Download from page
# No email dependency

import os
import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import boto3
import streamlit as st
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="People Reader", layout="wide")

PAGE_TITLE = "People Reader"
SUPPORT_TEXT = "If you need help, please contact 0817008484"

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

.stApp { background: var(--bg-main); color: var(--text-main); }
[data-testid="stSidebar"] { background: #28231f; border-right: 1px solid var(--border); }

h1, h2, h3, h4, h5, h6 { color: #f0e4d4 !important; }
p, label, span, div { color: var(--text-main); }

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea {
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
  background: #22c55e !important;
  color: #2563eb !important;
  border: 0 !important;
  font-weight: 600 !important;
}

[data-testid="stFileUploader"] section button::after {
  content: "Browse file";
  font-size: 1.1rem;
  color: #2563eb !important;
}

.stButton > button,
.stDownloadButton > button,
.stLinkButton > a {
  background: linear-gradient(180deg, var(--accent), var(--accent-strong)) !important;
  color: #231d17 !important;
  border: 0 !important;
  font-weight: 600 !important;
}

[data-testid="stAlert"] {
  background: var(--bg-card) !important;
  color: var(--text-main) !important;
  border: 1px solid var(--border) !important;
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


def render_top_banner() -> None:
    for path in BANNER_PATH_CANDIDATES:
        if os.path.exists(path):
            st.image(path, use_container_width=True)
            return


# -------------------------
# Env / S3
# -------------------------
AWS_BUCKET = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET") or "local"
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

if not AWS_BUCKET or AWS_BUCKET == "local":
    st.warning("Set AWS_BUCKET in .env for uploads.")

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
JOBS_FINISHED_PREFIX = "jobs/finished/"
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
    if fn.endswith(".pdf"):
        return "application/pdf"
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


def s3_read_json_key(key: str) -> Optional[Dict[str, Any]]:
    try:
        raw = s3_get_bytes(key)
        if not raw:
            return None
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def s3_output_ready(key: str) -> bool:
    """True if the object exists. Uses HeadObject, then list_objects_v2 exact key (some IAM allows List but not Head)."""
    k = (key or "").strip()
    if not k:
        return False
    if s3_key_exists(k):
        return True
    try:
        resp = s3.list_objects_v2(Bucket=AWS_BUCKET, Prefix=k, MaxKeys=5)
        for obj in resp.get("Contents", []):
            if str(obj.get("Key") or "") == k:
                return True
    except Exception:
        pass
    return False


def s3_get_bytes(key: str) -> Optional[bytes]:
    try:
        obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
        return obj["Body"].read()
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
    key = f"{JOBS_PENDING_PREFIX}{job_id}.json"
    s3_put_json(key, job)
    return key


def build_output_keys(group_id: str) -> Dict[str, str]:
    base = f"{JOBS_OUTPUT_PREFIX}groups/{group_id}/"
    return {
        "dots_video": base + "dots.mp4",
        "skeleton_video": base + "skeleton.mp4",
        "report_en_pdf": base + "report_en.pdf",
        "report_th_pdf": base + "report_th.pdf",
    }


def get_group_id_variants(group_id: str) -> List[str]:
    g = (group_id or "").strip()
    if not g:
        return []
    out = [g]
    # Fix missing "20" (year) — e.g. 0260319... → 20260319...
    if g.startswith("0") and len(g) >= 7 and g[1:7].isdigit():
        out.append("20" + g)
    if "__" in g:
        alt = g.replace("__", "_", 1)
        if alt != g:
            out.append(alt)
    elif "_" in g:
        parts = g.rsplit("_", 1)
        if len(parts) == 2 and len(parts[1]) > 2:
            alt = parts[0] + "__" + parts[1]
            if alt != g:
                out.append(alt)
    return out


def find_report_files_in_s3(prefix: str) -> Dict[str, str]:
    result = {
        "report_en_pdf": "",
        "report_th_pdf": "",
    }
    try:
        prefix = (prefix or "").rstrip("/") + "/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=prefix):
            for item in page.get("Contents", []):
                key = str(item.get("Key") or "")
                kl = key.lower()
                if kl.endswith(".pdf"):
                    if "report_en.pdf" in kl or "_en.pdf" in kl:
                        result["report_en_pdf"] = key
                    elif "report_th.pdf" in kl or "_th.pdf" in kl:
                        result["report_th_pdf"] = key
    except Exception:
        pass
    return result


def resolve_outputs(group_id: str) -> Dict[str, str]:
    outputs = build_output_keys(group_id)
    gid_variants = get_group_id_variants(group_id) or [group_id]

    for gid in gid_variants:
        for prefix in (f"{JOBS_GROUP_PREFIX}{gid}/", f"{JOBS_OUTPUT_PREFIX}groups/{gid}/"):
            try:
                paginator = s3.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=prefix):
                    for item in page.get("Contents", []):
                        key = str(item.get("Key") or "")
                        kl = key.lower()
                        if "dots.mp4" in kl:
                            outputs["dots_video"] = key
                        elif "skeleton.mp4" in kl:
                            outputs["skeleton_video"] = key
            except Exception:
                pass

        scanned = find_report_files_in_s3(f"{JOBS_OUTPUT_PREFIX}groups/{gid}")
        if scanned.get("report_en_pdf"):
            outputs["report_en_pdf"] = scanned["report_en_pdf"]
        if scanned.get("report_th_pdf"):
            outputs["report_th_pdf"] = scanned["report_th_pdf"]

        scanned = find_report_files_in_s3(f"{JOBS_GROUP_PREFIX}{gid}")
        if scanned.get("report_en_pdf") and not outputs.get("report_en_pdf"):
            outputs["report_en_pdf"] = scanned["report_en_pdf"]
        if scanned.get("report_th_pdf") and not outputs.get("report_th_pdf"):
            outputs["report_th_pdf"] = scanned["report_th_pdf"]

    return outputs


def apply_finished_job_output_keys(group_id: str, outputs: Dict[str, str]) -> None:
    """Fill dots/skeleton S3 keys from finished job JSON (authoritative output_key from worker)."""
    gid = (group_id or "").strip()
    if not gid:
        return
    variants = set(get_group_id_variants(gid) or [gid])
    m = st.session_state.get("people_reader_jobs_by_group") or {}
    entry: Dict[str, Any] = {}
    for g in [gid] + [x for x in variants if x != gid]:
        entry = m.get(g) or {}
        if entry:
            break

    def _merge_from_job(job_id: str, field: str) -> None:
        if not job_id:
            return
        fk = f"{JOBS_FINISHED_PREFIX}{job_id}.json"
        data = s3_read_json_key(fk)
        if not data:
            return
        gj = str(data.get("group_id") or "").strip()
        if gj not in variants and gj != gid:
            return
        if str(data.get("status") or "").lower() != "finished":
            return
        mode = str(data.get("mode") or "").strip().lower()
        want = "dots" if field == "dots_video" else "skeleton"
        if mode != want:
            return
        res = data.get("result") or {}
        if res.get("ok") is False:
            return
        out_key = str(res.get("output_key") or "").strip()
        if out_key:
            outputs[field] = out_key

    _merge_from_job(str(entry.get("dots_job_id") or "").strip(), "dots_video")
    _merge_from_job(str(entry.get("skeleton_job_id") or "").strip(), "skeleton_video")


def discover_video_outputs_from_finished_jobs(group_id: str, outputs: Dict[str, str], max_json: int = 400) -> None:
    """If session has no job ids, scan jobs/finished/YYYYMMDD* for finished dots/skeleton with matching group_id."""
    gid = (group_id or "").strip()
    if not gid:
        return
    variants = set(get_group_id_variants(gid) or [gid])
    date_prefix = gid[:8] if len(gid) >= 8 and gid[:8].isdigit() else ""
    scan_prefix = f"{JOBS_FINISHED_PREFIX}{date_prefix}" if date_prefix else JOBS_FINISHED_PREFIX
    seen = 0
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=scan_prefix):
            for item in page.get("Contents", []):
                key = str(item.get("Key") or "")
                if not key.endswith(".json"):
                    continue
                seen += 1
                if seen > max_json:
                    return
                data = s3_read_json_key(key)
                if not data:
                    continue
                gj = str(data.get("group_id") or "").strip()
                if gj not in variants:
                    continue
                if str(data.get("status") or "").lower() != "finished":
                    continue
                mode = str(data.get("mode") or "").strip().lower()
                if mode not in ("dots", "skeleton"):
                    continue
                res = data.get("result") or {}
                if res.get("ok") is False:
                    continue
                out_key = str(res.get("output_key") or "").strip()
                if not out_key:
                    continue
                if mode == "dots":
                    outputs["dots_video"] = out_key
                else:
                    outputs["skeleton_video"] = out_key
                if outputs.get("dots_video") and outputs.get("skeleton_video"):
                    return
    except Exception:
        pass


def ensure_session_defaults() -> None:
    defaults = {
        "people_reader_last_group_id": "",
        "people_reader_submission_id_override": "",
        "people_reader_last_name": "",
        "people_reader_audience_mode": "one",
        "people_reader_jobs_by_group": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def clear_state() -> None:
    for k in (
        "people_reader_last_group_id",
        "people_reader_submission_id_override",
        "people_reader_last_name",
        "people_reader_download_id",
        "people_reader_audience_mode",
        "people_reader_video_upload",
        "people_reader_jobs_by_group",
    ):
        st.session_state.pop(k, None)


def render_download_button(
    label: str, key: str, filename: str, mime: str, button_key: str, ready: bool = False
) -> None:
    if not key:
        st.caption(f"Waiting for {label}")
        return

    is_video = (mime or "").startswith("video/")
    if is_video:
        try:
            url = presigned_get_url(key, expires=3600, filename=filename)
            if ready:
                btn_html = (
                    '<a href="' + url + '" target="_blank" rel="noopener" '
                    'style="display: inline-block; padding: 0.5rem 1.25rem; background: #22c55e; '
                    'color: white !important; border-radius: 6px; text-decoration: none; font-weight: 600;">'
                    "✓ Download " + label + "</a>"
                )
                st.markdown(btn_html, unsafe_allow_html=True)
            else:
                st.link_button(f"Download {label}", url)
        except Exception:
            st.caption(f"Waiting for {label}")
        return

    data = s3_get_bytes(key)
    if data and len(data) < 50 * 1024 * 1024:
        st.download_button(
            label=f"✓ Download {label}" if ready else f"Download {label}",
            data=data,
            file_name=filename,
            mime=mime,
            key=button_key,
        )
    else:
        try:
            url = presigned_get_url(key, expires=3600, filename=filename)
            if ready:
                btn_html = (
                    '<a href="' + url + '" target="_blank" rel="noopener" '
                    'style="display: inline-block; padding: 0.5rem 1.25rem; background: #22c55e; '
                    'color: white !important; border-radius: 6px; text-decoration: none; font-weight: 600;">'
                    "✓ Download " + label + "</a>"
                )
                st.markdown(btn_html, unsafe_allow_html=True)
            else:
                st.link_button(f"Download {label}", url)
        except Exception:
            st.caption(f"Waiting for {label}")


# -------------------------
# UI
# -------------------------
ensure_session_defaults()
apply_theme()
render_top_banner()

st.markdown(f"# {PAGE_TITLE}")
st.caption(
    "Upload one video and download the outputs directly from this page. "
    "Reports match the standard layout (Engaging, Confidence, Authority, Effort/Shape graphs) "
    "plus **Adaptability** (Flexibility & Agility) — only for jobs from this page."
)
st.divider()

name_value = st.text_input(
    "Name",
    value=str(st.session_state.get("people_reader_last_name") or ""),
    placeholder="Enter your name",
)

if name_value:
    st.session_state["people_reader_last_name"] = name_value

audience_mode = st.radio(
    "Audience Type",
    options=["one", "many"],
    format_func=lambda x: "Presenting to 1 person" if x == "one" else "Presenting to multiple people",
    horizontal=True,
    key="people_reader_audience_mode",
)

uploaded = st.file_uploader(
    "Video (MP4 / MOV / M4V / WEBM)",
    type=["mp4", "mov", "m4v", "webm"],
    accept_multiple_files=False,
    key="people_reader_video_upload",
)

if uploaded is not None:
    uploaded_name = str(uploaded.name or "input.mp4")
    uploaded_size_mb = float((uploaded.size or 0) / (1024 * 1024))
    st.caption(f"Selected file: `{uploaded_name}` ({uploaded_size_mb:.2f} MB)")

run = st.button(
    "Start Analysis",
    type="primary",
    use_container_width=True,
    disabled=(uploaded is None or not AWS_BUCKET or AWS_BUCKET == "local"),
)

st.caption(SUPPORT_TEXT)

if run:
    if not AWS_BUCKET or AWS_BUCKET == "local":
        st.error("S3 is required. Set AWS_BUCKET in .env and restart.")
        st.stop()
    if uploaded is None:
        st.warning("Please upload a video first.")
        st.stop()
    if not name_value.strip():
        st.warning("Please enter your name first.")
        st.stop()

    base_user = safe_slug(name_value, fallback="user")
    group_id = f"{new_group_id()}__{base_user}"
    input_key = f"{JOBS_GROUP_PREFIX}{group_id}/input/input.mp4"

    with st.status("Uploading video to cloud...", expanded=True) as status:
        try:
            st.write("Uploading your video (this may take a minute for large files)...")
            s3_upload_stream(
                key=input_key,
                file_obj=uploaded,
                content_type=guess_content_type(uploaded.name or "input.mp4"),
            )
            st.write("✓ Upload complete.")
        except Exception as e:
            status.update(label="Upload failed", state="error")
            st.error(f"Upload failed: {e}")
            st.stop()

        # Do not block on HEAD after upload: upload_fileobj success means the object is there.
        # Extra HEAD often fails on misconfigured IAM (HeadObject denied) and looked like "Verification failed".
        st.write("✓ Video uploaded. Queuing analysis jobs...")
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
            "user_name": name_value,
            "employee_id": base_user,
            "employee_email": "",
            "notify_email": "",
        }

        job_skeleton = {
            "job_id": new_job_id(),
            "group_id": group_id,
            "created_at": created_at,
            "status": "pending",
            "mode": "skeleton",
            "input_key": input_key,
            "output_key": outputs["skeleton_video"],
            "user_name": name_value,
            "employee_id": base_user,
            "employee_email": "",
            "notify_email": "",
            "suppress_completion_email": True,
        }

        job_report = {
            "job_id": new_job_id(),
            "group_id": group_id,
            "created_at": created_at,
            "status": "pending",
            "mode": "report",
            "input_key": input_key,
            "client_name": name_value,
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "languages": ["th", "en"],
            "output_prefix": f"{JOBS_GROUP_PREFIX}{group_id}",
            "analysis_mode": "real",
            "sample_fps": 3,
            "max_frames": 150,
            "report_style": "people_reader",
            "report_format": "pdf",
            "expect_skeleton": True,
            "expect_dots": True,
            "notify_email": "rungnapa@imagematters.at",
            "enterprise_folder": PAGE_TITLE,
            "employee_id": base_user,
            "employee_email": "",
            "audience_mode": audience_mode,
        }

        try:
            enqueue_legacy_job(job_dots)
            enqueue_legacy_job(job_skeleton)
            enqueue_legacy_job(job_report)
        except Exception as e:
            status.update(label="Queue failed", state="error")
            st.error(f"Queue submission failed: {e}")
            st.stop()

        st.write("✓ Jobs queued.")
        status.update(label="Complete", state="complete")
        st.session_state["people_reader_last_group_id"] = group_id
        st.session_state["people_reader_submission_id_override"] = group_id
        if "people_reader_jobs_by_group" not in st.session_state:
            st.session_state["people_reader_jobs_by_group"] = {}
        st.session_state["people_reader_jobs_by_group"][group_id] = {
            "dots_job_id": job_dots["job_id"],
            "skeleton_job_id": job_skeleton["job_id"],
        }
        st.success(f"Submission received. Group ID: `{group_id}`")
        st.info("Please keep this Group ID and use Refresh to check when files are ready.")

    st.rerun()


# -------------------------
# Download section
# -------------------------
st.divider()
st.markdown("### Download Results")

def _sanitize_group_id(raw: str) -> str:
    """Extract actual group ID; strip 'Current Group ID:' etc if pasted by mistake."""
    s = str(raw or "").strip()
    for prefix in ("Current Group ID:", "Current Group ID：", "Group ID:"):
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):].strip()
    return s.strip("`'\"")

current_group_id = _sanitize_group_id(
    st.session_state.get("people_reader_submission_id_override")
    or st.session_state.get("people_reader_last_group_id")
    or ""
)

manual_group_id = _sanitize_group_id(
    st.text_input(
        "Group ID",
        value=current_group_id,
        placeholder="Paste your Group ID here (e.g. 20240319_075553_1a1b29__Name)",
        key="people_reader_download_id",
    )
)

if manual_group_id:
    current_group_id = manual_group_id
    st.session_state["people_reader_submission_id_override"] = manual_group_id
    st.session_state["people_reader_last_group_id"] = manual_group_id

col1, col2 = st.columns(2)

with col1:
    if st.button("Refresh", key="people_reader_refresh"):
        st.rerun()

with col2:
    has_prev = bool(
        st.session_state.get("people_reader_submission_id_override")
        or st.session_state.get("people_reader_last_group_id")
    )
    if has_prev and st.button("Clear and Start New Upload", key="people_reader_clear", type="secondary"):
        clear_state()
        st.rerun()

if current_group_id:
    st.info(f"Current Group ID: `{current_group_id}`")
    resolved = resolve_outputs(current_group_id)
    apply_finished_job_output_keys(current_group_id, resolved)
    dots_key = str(resolved.get("dots_video") or "").strip()
    skel_key = str(resolved.get("skeleton_video") or "").strip()
    dots_ready = bool(dots_key) and s3_output_ready(dots_key)
    skeleton_ready = bool(skel_key) and s3_output_ready(skel_key)
    if not dots_ready or not skeleton_ready:
        discover_video_outputs_from_finished_jobs(current_group_id, resolved)
        dots_key = str(resolved.get("dots_video") or "").strip()
        skel_key = str(resolved.get("skeleton_video") or "").strip()
        dots_ready = bool(dots_key) and s3_output_ready(dots_key)
        skeleton_ready = bool(skel_key) and s3_output_ready(skel_key)

    status_items = [
        ("Dots Video", dots_ready),
        ("Skeleton Video", skeleton_ready),
    ]

    overall_pct = int(round((sum(1 for _, ready in status_items if ready) / len(status_items)) * 100))
    st.progress(overall_pct, text=f"Overall progress: {overall_pct}%")

    if overall_pct < 100:
        try:
            @st.fragment(run_every=timedelta(seconds=8))
            def _auto_refresh():
                st.rerun()
        except Exception:
            pass

    for label, ready in status_items:
        st.progress(100 if ready else 0, text=f"{label}: {'Ready' if ready else 'Processing'}")

    st.markdown("---")
    st.subheader("Available Downloads")
    st.caption("Reports (TH/EN PDF) are sent to aipeoplereader.com")

    render_download_button(
        label="Dots Video",
        key=resolved.get("dots_video", ""),
        filename="dots.mp4",
        mime="video/mp4",
        button_key="people_reader_dl_dots",
        ready=dots_ready,
    )
    render_download_button(
        label="Skeleton Video",
        key=resolved.get("skeleton_video", ""),
        filename="skeleton.mp4",
        mime="video/mp4",
        button_key="people_reader_dl_skeleton",
        ready=skeleton_ready,
    )

    if not any([dots_ready, skeleton_ready]):
        st.caption("Files are not ready yet. Please wait a few minutes and click Refresh.")

else:
    st.caption("Upload a video above or paste an existing Group ID to download results.")

# -------------------------
# Recording guide
# -------------------------
st.divider()
st.markdown("## Recording Guidelines")
st.markdown(
    """
1. Record in portrait orientation and show the full body from head to toe  
2. Recommended video size: **720 x 1280** in **MP4** or **MOV**  
3. Avoid wearing white, black, or colors similar to the wall  
4. Stand in front of a light-colored wall, or wear light-colored clothes if the wall is dark  
5. Move naturally and use hand gestures naturally  
6. Make sure your video is at least 2 minutes long  
"""
)