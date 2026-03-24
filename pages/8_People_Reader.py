## pages/8_People_Reader.py
# People Reader page (from Training flow)
# Flow: Upload -> S3 -> Queue -> Download from page
# No email dependency

import os
import sys
import json
import uuid
import tempfile
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

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
JOBS_PROCESSING_PREFIX = "jobs/processing/"
JOBS_FINISHED_PREFIX = "jobs/finished/"
JOBS_FAILED_PREFIX = "jobs/failed/"
JOBS_OUTPUT_PREFIX = "jobs/output/"
JOBS_GROUP_PREFIX = "jobs/groups/"
MOVEMENT_CALIBRATION_S3_KEY = str(
    os.getenv("MOVEMENT_TYPE_CALIBRATION_S3_KEY") or "config/movement_type_calibration.json"
).strip()

# Values must match movement_type_classifier.TYPE_TEMPLATES keys + "auto"
PEOPLE_READER_MOVEMENT_TYPE_CHOICES: List[tuple] = [
    ("Auto — closest match from video", "auto"),
    ("Type 1 (Khun K)", "type_1"),
    ("Type 2 (Irene)", "type_2"),
    ("Type 3 (Khun Hongyok)", "type_3"),
    ("Type 4 (Boon)", "type_4"),
    ("Type 5 (Elisha)", "type_5"),
    ("Type 6 (Alisa)", "type_6"),
]


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


def _job_payload_output_key(data: Dict[str, Any]) -> str:
    """Worker stores target path on the job; older workers only returned ok=True in result."""
    if not isinstance(data, dict):
        return ""
    res = data.get("result") or {}
    return str(res.get("output_key") or data.get("output_key") or "").strip()


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
        out_key = _job_payload_output_key(data)
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
                out_key = _job_payload_output_key(data)
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


def scan_dots_skeleton_job_status(group_id: str, max_per_folder: int = 300) -> Dict[str, str]:
    """
    Read dots/skeleton job JSONs from S3 (same YYYYMMDD prefix as group_id) to show queue/processing/failed.
    """
    gid = (group_id or "").strip()
    out: Dict[str, str] = {}
    if not gid or not AWS_BUCKET or AWS_BUCKET == "local":
        return out
    variants = set(get_group_id_variants(gid) or [gid])
    date_prefix = gid[:8] if len(gid) >= 8 and gid[:8].isdigit() else ""
    if not date_prefix:
        return out

    # Higher rank wins (user-facing severity)
    best: Dict[str, Tuple[int, str]] = {}

    def note(mode: str, rank: int, text: str) -> None:
        cur = best.get(mode)
        if cur is None or rank > cur[0]:
            best[mode] = (rank, text)

    folder_meta = [
        (JOBS_FAILED_PREFIX, "failed"),
        (JOBS_PROCESSING_PREFIX, "processing"),
        (JOBS_PENDING_PREFIX, "pending"),
        (JOBS_FINISHED_PREFIX, "finished"),
    ]
    for folder, fname in folder_meta:
        scanned = 0
        prefix = f"{folder}{date_prefix}"
        try:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=prefix):
                for item in page.get("Contents", []):
                    key = str(item.get("Key") or "")
                    if not key.endswith(".json"):
                        continue
                    scanned += 1
                    if scanned > max_per_folder:
                        break
                    data = s3_read_json_key(key)
                    if not data:
                        continue
                    gj = str(data.get("group_id") or "").strip()
                    if gj not in variants:
                        continue
                    mode = str(data.get("mode") or "").strip().lower()
                    if mode not in ("dots", "skeleton"):
                        continue
                    st = str(data.get("status") or "").strip().lower()
                    msg = str(data.get("message") or "").strip()
                    if len(msg) > 500:
                        msg = msg[:497] + "..."
                    res_ok = (data.get("result") or {}).get("ok")

                    if fname == "failed" or st == "failed":
                        note(mode, 4, (f"Failed: {msg}" if msg else "Failed — check video worker logs on the server"))
                    elif res_ok is False:
                        note(mode, 4, (f"Failed: {msg}" if msg else "Worker reported result ok=false"))
                    elif fname == "processing" or st == "processing":
                        note(mode, 3, "Processing on video worker…")
                    elif fname == "pending" or st == "pending":
                        note(
                            mode,
                            2,
                            "Queued — waiting for a free worker. If the queue is large this can take many minutes.",
                        )
                    elif fname == "finished" or st == "finished":
                        note(mode, 1, "Worker finished — verifying file on S3…")
                if scanned > max_per_folder:
                    break
        except Exception:
            pass

    for mode, (_, text) in best.items():
        out[mode] = text
    return out


def _repo_root_and_src() -> Tuple[str, str]:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return root, os.path.join(root, "src")


def import_movement_calibration_modules() -> Tuple[Any, Any]:
    """movement_type_classifier at repo root; report_core in src/."""
    root, src = _repo_root_and_src()
    for p in (src, root):
        if p not in sys.path:
            sys.path.insert(0, p)
    import movement_type_classifier as mtc  # noqa: WPS433

    from report_core import extract_movement_type_frame_features_from_video  # noqa: WPS433

    return mtc, extract_movement_type_frame_features_from_video


def bundled_expected_to_session_overrides(bundled: Dict[str, Any]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    out: Dict[str, Dict[str, Tuple[float, float]]] = {}
    if not isinstance(bundled, dict):
        return out
    for tid, block in bundled.items():
        if not isinstance(block, dict):
            continue
        exp = block.get("expected")
        if not isinstance(exp, dict):
            continue
        inner: Dict[str, Tuple[float, float]] = {}
        for k, pair in exp.items():
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                inner[str(k)] = (float(pair[0]), float(pair[1]))
        if inner:
            out[str(tid)] = inner
    return out


def build_calibration_download_payload(bundled: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "version": 1,
        "updated_at": utc_now_iso(),
        "types": dict(bundled) if isinstance(bundled, dict) else {},
    }


def ensure_session_defaults() -> None:
    defaults = {
        "people_reader_last_group_id": "",
        "people_reader_submission_id_override": "",
        "people_reader_last_name": "",
        "people_reader_audience_mode": "one",
        "people_reader_jobs_by_group": {},
        "people_reader_calibration_bundled": {},
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
_git = (os.getenv("RENDER_GIT_COMMIT") or os.getenv("APP_GIT_SHA") or "").strip()
_build = f"`{_git[:10]}`" if len(_git) >= 7 else "`local`"
st.caption(
    f"**Deploy:** Streamlit app build {_build}. "
    "PDF/DOCX files are **not** rendered here — they are built by the separate **report worker** "
    "(`src/report_worker.py` + `src/report_core.py` on Render). After `git push`, redeploy **that** worker "
    "and use **Clear build cache** if reports still look old."
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

movement_type_mode = st.selectbox(
    "Movement type (report profile)",
    options=[c[1] for c in PEOPLE_READER_MOVEMENT_TYPE_CHOICES],
    format_func=lambda v: next(label for label, key in PEOPLE_READER_MOVEMENT_TYPE_CHOICES if key == v),
    index=0,
    key="people_reader_movement_type_mode",
    help="Auto: classify the video to the nearest type and blend scores into the report. "
    "Or choose a type to align Engaging, Confidence, Authority, Adaptability and first-impression cues to that profile.",
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
            "movement_type_mode": movement_type_mode,
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

    job_hints = scan_dots_skeleton_job_status(current_group_id)
    if overall_pct < 100:
        st.caption(
            "Dots and skeleton videos are produced by the **video worker** (`worker.py` on Render or your server), "
            "not by the report worker. If this stays at 0% for a long time, check that the worker is running and "
            "that the queue is not overloaded."
        )
        with st.expander("Job status from S3 (same day as Group ID)", expanded=True):
            st.write(f"**Dots:** {job_hints.get('dots', '— no matching job JSON found under jobs/pending|processing|finished|failed for this group —')}")
            st.write(f"**Skeleton:** {job_hints.get('skeleton', '— same —')}")
        if job_hints.get("dots", "").startswith("Failed") or job_hints.get("skeleton", "").startswith("Failed"):
            st.error(
                "At least one video job failed. Fix the error (often ffmpeg/MediaPipe on the worker), then upload again."
            )
        try:

            @st.fragment(run_every=timedelta(seconds=10))
            def _people_reader_auto_refresh():
                st.rerun()

            _people_reader_auto_refresh()
        except Exception:
            st.caption("Tip: click **Refresh** periodically to poll S3 for finished files.")

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
# Movement type calibration (reference videos → expected ranges → S3 for report worker)
# -------------------------
st.divider()
st.markdown("## Calibrate movement types")
with st.expander("Upload reference videos to tune the 6 type profiles (expected feature ranges)", expanded=False):
    st.caption(
        "For each type, upload a **reference** clip that best represents that profile. "
        "We extract the same summary features as the report worker and suggest `expected` min/max ranges. "
        "**Stage** each type, then **Save to S3** at `config/movement_type_calibration.json` (or your env key). "
        "The **report worker** merges this into classification on each job (when `movement_type_mode` is set)."
    )
    bundled_cal = st.session_state.get("people_reader_calibration_bundled")
    if not isinstance(bundled_cal, dict):
        bundled_cal = {}
        st.session_state["people_reader_calibration_bundled"] = bundled_cal

    st.write("**Staged types:**", ", ".join(sorted(bundled_cal.keys())) if bundled_cal else "— none —")
    if st.button("Clear staged calibration", key="people_reader_cal_clear_bundled"):
        st.session_state["people_reader_calibration_bundled"] = {}
        st.session_state.pop("people_reader_cal_last_suggested", None)
        st.session_state.pop("people_reader_cal_last_summary", None)
        st.session_state.pop("people_reader_cal_last_type", None)
        st.rerun()

    cal_type_pick = st.selectbox(
        "Type to calibrate",
        options=["type_1", "type_2", "type_3", "type_4", "type_5", "type_6"],
        format_func=lambda t: next((lbl for lbl, k in PEOPLE_READER_MOVEMENT_TYPE_CHOICES if k == t), t),
        key="people_reader_cal_type",
    )
    cal_audience = st.radio(
        "Audience (match your recordings)",
        options=["one", "many"],
        format_func=lambda x: "1 person" if x == "one" else "Multiple",
        horizontal=True,
        key="people_reader_cal_audience",
    )
    cal_margin = st.slider("Half-width ± around measured value", 0.03, 0.20, 0.08, 0.01, key="people_reader_cal_margin")
    cal_min_band = st.slider("Minimum band width", 0.04, 0.20, 0.06, 0.01, key="people_reader_cal_minband")
    cal_sample_n = st.number_input("Sample every N frames", 1, 10, 3, key="people_reader_cal_sample_n")
    cal_max_frames = st.number_input("Max frames", 50, 400, 200, 10, key="people_reader_cal_max_frames")

    cal_video = st.file_uploader(
        "Reference video for the selected type",
        type=["mp4", "mov", "m4v", "webm"],
        key="people_reader_cal_video",
    )

    if st.button("Run calibration on uploaded video", type="primary", key="people_reader_cal_run"):
        if cal_video is None:
            st.warning("Upload a reference video first.")
        else:
            try:
                mtc, extract_fn = import_movement_calibration_modules()
                tpl = mtc.TYPE_TEMPLATES.get(cal_type_pick)
                if tpl is None:
                    st.error("Unknown type.")
                else:
                    suf = os.path.splitext(cal_video.name or ".mp4")[1] or ".mp4"
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suf)
                    tfile.write(cal_video.getvalue())
                    tfile.close()
                    tpath = tfile.name
                    try:
                        feats = extract_fn(
                            tpath,
                            audience_mode=cal_audience,
                            sample_every_n=int(cal_sample_n),
                            max_frames=int(cal_max_frames),
                        )
                        sf_loc = mtc.build_summary_features_from_timeseries(feats)
                        suggested_loc = mtc.suggested_expected_from_reference_video_summary(
                            sf_loc,
                            tpl,
                            margin=float(cal_margin),
                            min_band=float(cal_min_band),
                        )
                        st.session_state["people_reader_cal_last_type"] = cal_type_pick
                        st.session_state["people_reader_cal_last_summary"] = sf_loc
                        st.session_state["people_reader_cal_last_suggested"] = suggested_loc
                        st.success("Done — review the table below and stage if you agree.")
                    finally:
                        try:
                            os.unlink(tpath)
                        except OSError:
                            pass
            except Exception as e:
                st.error(
                    f"Calibration failed: {e}. "
                    "The Streamlit server needs the same deps as analysis (OpenCV, MediaPipe, `src/report_core.py`)."
                )

    last_sug = st.session_state.get("people_reader_cal_last_suggested")
    last_tid = str(st.session_state.get("people_reader_cal_last_type") or "")
    if last_sug and isinstance(last_sug, dict) and last_tid:
        try:
            mtc2, _ = import_movement_calibration_modules()
            tpl2 = mtc2.TYPE_TEMPLATES.get(last_tid)
            sf_loc = st.session_state.get("people_reader_cal_last_summary") or {}
            if tpl2 is not None:
                st.subheader(f"Suggested ranges for `{last_tid}` — {tpl2.name}")
                rows_cal = []
                for fname in sorted(last_sug.keys()):
                    old_r = tpl2.expected.get(fname, (0.0, 0.0))
                    nlo, nhi = last_sug[fname]
                    v = float(sf_loc.get(fname, 0.0)) if isinstance(sf_loc, dict) else 0.0
                    rows_cal.append(
                        {
                            "feature": fname,
                            "measured": round(v, 4),
                            "current_low": old_r[0],
                            "current_high": old_r[1],
                            "suggested_low": nlo,
                            "suggested_high": nhi,
                        }
                    )
                st.dataframe(rows_cal, use_container_width=True)
                if st.button("Stage suggested ranges for this type", key="people_reader_cal_stage"):
                    bd = dict(st.session_state.get("people_reader_calibration_bundled") or {})

                    def _pair_to_list(v: Any) -> List[float]:
                        if isinstance(v, (list, tuple)) and len(v) >= 2:
                            return [float(v[0]), float(v[1])]
                        return [0.0, 0.0]

                    bd[last_tid] = {
                        "name": tpl2.name,
                        "expected": {str(k): _pair_to_list(last_sug[k]) for k in last_sug},
                    }
                    st.session_state["people_reader_calibration_bundled"] = bd
                    st.success(f"Staged `{last_tid}`. Repeat for other types, then save to S3.")
                    st.rerun()
        except Exception as e:
            st.warning(f"Could not render calibration table: {e}")

    bundled_cal = st.session_state.get("people_reader_calibration_bundled")
    if not isinstance(bundled_cal, dict):
        bundled_cal = {}
    payload_cal = build_calibration_download_payload(bundled_cal)
    st.download_button(
        "Download calibration JSON",
        data=json.dumps(payload_cal, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="movement_type_calibration.json",
        mime="application/json",
        key="people_reader_cal_dl",
        disabled=not bundled_cal,
    )

    c_save, c_load = st.columns(2)
    with c_save:
        if st.button("Save staged calibration to S3", key="people_reader_cal_save_s3"):
            if not AWS_BUCKET or AWS_BUCKET == "local":
                st.error("Configure AWS_BUCKET for S3.")
            elif not bundled_cal:
                st.warning("Stage at least one type first.")
            else:
                try:
                    blob = json.dumps(build_calibration_download_payload(bundled_cal), ensure_ascii=False, indent=2).encode(
                        "utf-8"
                    )
                    s3.put_object(
                        Bucket=AWS_BUCKET,
                        Key=MOVEMENT_CALIBRATION_S3_KEY,
                        Body=blob,
                        ContentType="application/json; charset=utf-8",
                    )
                    st.success(f"Saved `s3://{AWS_BUCKET}/{MOVEMENT_CALIBRATION_S3_KEY}`. Redeploy or wait for report worker ETag refresh.")
                except Exception as e:
                    st.error(str(e))
    with c_load:
        if st.button("Load calibration from S3 into session", key="people_reader_cal_load_s3"):
            if not AWS_BUCKET or AWS_BUCKET == "local":
                st.error("Configure AWS_BUCKET.")
            else:
                try:
                    raw = s3_get_bytes(MOVEMENT_CALIBRATION_S3_KEY)
                    if not raw:
                        st.error("Object missing or not readable.")
                    else:
                        data = json.loads(raw.decode("utf-8"))
                        types_block = data.get("types") if isinstance(data, dict) else None
                        if isinstance(types_block, dict):
                            st.session_state["people_reader_calibration_bundled"] = types_block
                            st.success("Loaded into session.")
                            st.rerun()
                        else:
                            st.error("Invalid JSON (expected top-level `types`).")
                except Exception as e:
                    st.error(str(e))

    st.markdown("**Preview:** classify a test clip using **staged** overrides only (session).")
    test_cal_clip = st.file_uploader(
        "Test video",
        type=["mp4", "mov", "m4v", "webm"],
        key="people_reader_cal_test_vid",
    )
    if st.button("Preview best match (staged overrides)", key="people_reader_cal_preview"):
        if not bundled_cal:
            st.warning("Stage at least one type.")
        elif test_cal_clip is None:
            st.warning("Upload a test video.")
        else:
            try:
                mtc3, extract_fn3 = import_movement_calibration_modules()
                suf3 = os.path.splitext(test_cal_clip.name or ".mp4")[1] or ".mp4"
                tf3 = tempfile.NamedTemporaryFile(delete=False, suffix=suf3)
                tf3.write(test_cal_clip.getvalue())
                tf3.close()
                p3 = tf3.name
                try:
                    feats3 = extract_fn3(
                        p3,
                        audience_mode=cal_audience,
                        sample_every_n=3,
                        max_frames=200,
                    )
                    sf3 = mtc3.build_summary_features_from_timeseries(feats3)
                    ov3 = bundled_expected_to_session_overrides(bundled_cal)
                    cls3 = mtc3.classify_movement_type(sf3, session_overrides=ov3)
                    best3 = cls3.get("best_match") or {}
                    st.write(
                        f"**Best match:** {best3.get('type_name')} "
                        f"(score={best3.get('score')}, confidence≈{best3.get('confidence_pct')}%)"
                    )
                    st.json({str(x.get("type_id")): x.get("score") for x in (cls3.get("scores") or [])})
                finally:
                    try:
                        os.unlink(p3)
                    except OSError:
                        pass
            except Exception as e:
                st.error(str(e))

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