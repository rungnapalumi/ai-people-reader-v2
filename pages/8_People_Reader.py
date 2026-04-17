## pages/8_People_Reader.py
# People Reader page (from Training flow)
# Flow: Upload -> S3 -> Queue -> Download from page
# No email dependency

import os
import json
import re
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import boto3
import streamlit as st
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError

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

/* Match primary actions (Start Analysis): accent gradient, no green/blue clash from Streamlit links inside button */
[data-testid="stFileUploader"] section button,
[data-testid="stFileUploader"] section [data-testid="stBaseButton-secondary"] {
  background: linear-gradient(180deg, var(--accent), var(--accent-strong)) !important;
  color: #231d17 !important;
  border: 0 !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
}
[data-testid="stFileUploader"] section button *,
[data-testid="stFileUploader"] section [data-testid="stBaseButton-secondary"] *,
[data-testid="stFileUploader"] section button a,
[data-testid="stFileUploader"] section [data-testid="stBaseButton-secondary"] a {
  color: #231d17 !important;
}
[data-testid="stFileUploader"] section button svg,
[data-testid="stFileUploader"] section [data-testid="stBaseButton-secondary"] svg {
  fill: #231d17 !important;
  stroke: #231d17 !important;
  color: #231d17 !important;
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

/* File uploader: keep the remove (X) button clickable even when error text overflows */
[data-testid="stFileUploaderFile"] {
  position: relative !important;
  overflow: visible !important;
}
[data-testid="stFileUploaderFile"] button[aria-label],
[data-testid="stFileUploaderFile"] button:last-of-type {
  z-index: 10 !important;
  position: relative !important;
  pointer-events: auto !important;
  min-width: 24px !important;
  min-height: 24px !important;
  flex-shrink: 0 !important;
}
[data-testid="stFileUploaderFile"] [data-testid="stFileUploaderFileErrorMessage"],
[data-testid="stFileUploaderFile"] small,
[data-testid="stFileUploaderFile"] .uploadedFileName + div {
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  white-space: nowrap !important;
  max-width: 60% !important;
  pointer-events: none !important;
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

# Values must match movement_type_classifier.TYPE_TEMPLATES keys + "auto"
PEOPLE_READER_MOVEMENT_TYPE_CHOICES: List[tuple] = [
    ("Auto — closest match from video", "auto"),
    ("Type 1 (Khun K)", "type_1"),
    ("Type 2 (Irene)", "type_2"),
    ("Type 3 (Khun Hongyok)", "type_3"),
    ("Type 4 (Boon)", "type_4"),
    ("Type 5 (Elisha)", "type_5"),
    ("Type 6 (Alisa)", "type_6"),
    ("Type 7 (Anne)", "type_7"),
    ("Type 8 (Panu)", "type_8"),
    ("Type 9 (Punlop)", "type_9"),
    ("Type 10 (R.)", "type_10"),
]


def combination_10_types_image_path() -> str:
    """Product rubric table (`Combination 10 types.png`); prefer assets copy without spaces."""
    root = os.path.dirname(os.path.dirname(__file__))
    for rel in (
        os.path.join("assets", "combination_10_types.png"),
        "Combination 10 types.png",
    ):
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            return p
    return ""


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
    return "".join(out).strip("_")


def people_reader_fixed_settings() -> Dict[str, Any]:
    """One path only: People Reader PDF in English (Adaptability + movement-type matching). No Admin/S3 org file."""
    return {
        "organization_name": PAGE_TITLE,
        "organization_id": normalize_org_name(PAGE_TITLE),
        "report_style": "people_reader",
        "report_format": "pdf",
        "languages": ["en"],
        "enable_report_th": False,
        "enable_report_en": True,
        "enable_skeleton": True,
        "enable_dots": True,
    }


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


def presigned_put_url(key: str, expires: int = 1800) -> str:
    return s3.generate_presigned_url(
        ClientMethod="put_object",
        Params={"Bucket": AWS_BUCKET, "Key": key},
        ExpiresIn=expires,
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
    """Same idea as report_worker: some IAM policies allow GetObject but not HeadObject."""
    k = str(key or "").strip()
    if not k:
        return False
    try:
        s3.head_object(Bucket=AWS_BUCKET, Key=k)
        return True
    except ClientError as e:
        code = (e.response.get("Error") or {}).get("Code") or ""
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        if code in ("403", "AccessDenied"):
            try:
                s3.get_object(Bucket=AWS_BUCKET, Key=k, Range="bytes=0-0")
                return True
            except ClientError as e2:
                c2 = (e2.response.get("Error") or {}).get("Code") or ""
                if c2 in ("404", "NoSuchKey", "NotFound"):
                    return False
    except Exception:
        pass
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
    # Typo: leading "0" instead of "20" on the date (common paste error).
    # e.g. 0260416_063310_... → 20260416_063310_...  (NOT "20"+g — that becomes 200260416_...)
    if re.match(r"^0\d{6}(?:_|$)", g):
        alt = "2" + g
        if alt not in out:
            out.append(alt)
    # Missing "20" century: 260416_063310_... → 20260416_... (must not match 0260416_ — that uses "2"+g above)
    if re.match(r"^2(?!0)\d{6}_", g):
        alt2 = "20" + g
        if alt2 not in out:
            out.append(alt2)
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


def _YYYYMMDD_prefix_from_group_variants(group_id: str) -> str:
    """Prefix for S3 listing jobs/finished|pending/YYYYMMDD… — works when user typos 0260416 → 20260416."""
    variants = set(get_group_id_variants(group_id) or [group_id])
    for v in variants:
        if len(v) >= 8 and v[:8].isdigit():
            return v[:8]
    return ""


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


def merge_report_en_pdf_from_finished_job(group_id: str, outputs: Dict[str, str]) -> None:
    """If S3 list did not find report_en.pdf, take EN pdf_key from finished report job (same source as email)."""
    if str(outputs.get("report_en_pdf") or "").strip():
        return
    job = find_finished_report_job_for_group(group_id)
    if not job:
        return
    outs = job.get("outputs")
    if isinstance(outs, dict):
        en = (outs.get("reports") or {}).get("EN") or {}
        if isinstance(en, dict):
            pdf_key = str(en.get("pdf_key") or "").strip()
            if pdf_key:
                outputs["report_en_pdf"] = pdf_key
                return
    for k in ("report_en_pdf_key", "report_en_key"):
        alt = str(job.get(k) or "").strip()
        if alt.lower().endswith(".pdf"):
            outputs["report_en_pdf"] = alt
            return


def merge_report_th_pdf_from_finished_job(group_id: str, outputs: Dict[str, str]) -> None:
    """If S3 list did not find report_th.pdf, take TH pdf_key from finished report job."""
    if str(outputs.get("report_th_pdf") or "").strip():
        return
    job = find_finished_report_job_for_group(group_id)
    if not job:
        return
    outs = job.get("outputs")
    if isinstance(outs, dict):
        th = (outs.get("reports") or {}).get("TH") or {}
        if isinstance(th, dict):
            pdf_key = str(th.get("pdf_key") or "").strip()
            if pdf_key:
                outputs["report_th_pdf"] = pdf_key
                return
    for k in ("report_th_pdf_key", "report_th_key"):
        alt = str(job.get(k) or "").strip()
        if alt.lower().endswith(".pdf"):
            outputs["report_th_pdf"] = alt
            return


def find_finished_report_job_for_group(group_id: str, max_json: int = 800) -> Optional[Dict[str, Any]]:
    """
    Locate the latest finished `mode=report` job JSON for this group_id under jobs/finished/.
    Used to show People Reader category levels on the page when email is delayed or missing.
    """
    gid = (group_id or "").strip()
    if not gid or not AWS_BUCKET or AWS_BUCKET == "local":
        return None
    variants = set(get_group_id_variants(gid) or [gid])
    # Fast path: exact report job id from this browser session (avoids S3 scan cap when queue is busy).
    try:
        m = st.session_state.get("people_reader_jobs_by_group") or {}
        rid = str((m.get(gid) or {}).get("report_job_id") or "").strip()
        if rid:
            fk = f"{JOBS_FINISHED_PREFIX}{rid}.json"
            data = s3_read_json_key(fk)
            if isinstance(data, dict):
                gj = str(data.get("group_id") or "").strip()
                if gj in variants and str(data.get("status") or "").lower() == "finished":
                    if str(data.get("mode") or "").strip().lower() == "report":
                        return data
    except Exception:
        pass

    date_prefix = _YYYYMMDD_prefix_from_group_variants(gid)
    scan_prefix = f"{JOBS_FINISHED_PREFIX}{date_prefix}" if date_prefix else JOBS_FINISHED_PREFIX
    matches: List[Dict[str, Any]] = []
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
                    break
                data = s3_read_json_key(key)
                if not data:
                    continue
                gj = str(data.get("group_id") or "").strip()
                if gj not in variants:
                    continue
                if str(data.get("status") or "").lower() != "finished":
                    continue
                if str(data.get("mode") or "").strip().lower() != "report":
                    continue
                matches.append(data)
            if seen > max_json:
                break
    except Exception:
        return None

    if not matches:
        return None

    def _jid(d: Dict[str, Any]) -> str:
        return str(d.get("job_id") or "")

    matches.sort(key=_jid, reverse=True)
    for d in matches:
        if (d.get("movement_type_info") or {}):
            return d
    return matches[0]


def _format_scale_en(level: Any) -> str:
    s = str(level or "").strip().lower()
    return {"low": "Low", "moderate": "Moderate", "high": "High"}.get(s, str(level or "—").title())


def _format_scale_th(level: Any) -> str:
    s = str(level or "").strip().lower()
    return {"low": "ต่ำ", "moderate": "กลาง", "high": "สูง"}.get(s, "—")


def discover_video_outputs_from_finished_jobs(group_id: str, outputs: Dict[str, str], max_json: int = 400) -> None:
    """If session has no job ids, scan jobs/finished/YYYYMMDD* for finished dots/skeleton with matching group_id."""
    gid = (group_id or "").strip()
    if not gid:
        return
    variants = set(get_group_id_variants(gid) or [gid])
    date_prefix = _YYYYMMDD_prefix_from_group_variants(gid)
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


def report_failure_user_hint(report_line: str) -> str:
    """Extra context when report worker failed (MediaPipe / OpenGL / headless)."""
    low = (report_line or "").lower()
    if any(
        x in low
        for x in (
            "nsopengl",
            "kgpuservice",
            "pixel format",
            "imagetotensorcalculator",
            "could not create",
        )
    ):
        return (
            "**สรุปปัญหา (อ่านทีละข้อ)**\n\n"
            "**1) Error นี้คืออะไร** — ข้อความ `kGpuService` / `NSOpenGLPixelFormat` / `ImageToTensorCalculator` แปลว่า "
            "**MediaPipe ต้องการกราฟิก (OpenGL)** แต่เครื่องที่รัน **report worker** ไม่มีจอหรือไม่มี GL ให้ใช้ "
            "(เช่น Mac ไม่มีจอ, SSH ไม่มี session กราฟิก, หรือบน Linux แต่ไม่ได้ห่อด้วย `xvfb-run`)\n\n"
            "**2) ทำอย่างไรให้ได้ PDF** — บน **Render** ให้รัน **Linux worker** และ `startCommand` ใช้ **`xvfb-run`** ตาม `render.yaml` "
            "อย่าใช้ macOS เป็นเครื่องประมวลผลรายงานจริง\n\n"
            "**3) ทำไมยัง Failed ใน UI** — โค้ดบน `main` พยายาม **fallback วิเคราะห์** และ retry ถ้าเป็น error แบบนี้ — "
            "ถ้ายัง Failed ให้ตรวจว่า **Deploy แล้วใช้ commit ล่าสุด** ดูล็อก `REPORT_CORE_VERSION` และข้อความใน `jobs/failed/<job_id>.json`\n\n"
            "**4) ล็อก `NoSuchKey` ตอนอ่าน `jobs/pending/`** — มักเป็น **ปกติ** เมื่อมี **report worker หลาย instance** ชนกัน "
            "(อีกตัวย้ายไฟล์ไป `processing/` ก่อน) — แก้ที่ปลายทาง: **ตั้ง scale report worker = 1** หรือยอมรับ noise ในล็อก"
        )
    return ""


def _pick_report_status_line(
    entries: List[Dict[str, Any]],
    preferred_report_job_id: Optional[str],
) -> str:
    """
    Multiple report JSONs can exist for the same group_id (retries / duplicate enqueue).
    Prefer the browser's report_job_id; otherwise pick the newest by created_at then job_id.
    (Old logic: scanning jobs/failed/ first made any old failure mask a newer pending job.)
    """
    if not entries:
        return ""
    pref = str(preferred_report_job_id or "").strip()
    if pref:
        for e in entries:
            if str(e.get("job_id") or "").strip() == pref:
                return str(e.get("text") or "")
    # Newest submission wins when we cannot match session (or session cleared).
    def _sort_key(e: Dict[str, Any]) -> Tuple[str, str]:
        return (str(e.get("created_at") or ""), str(e.get("job_id") or ""))

    chosen = max(entries, key=_sort_key)
    return str(chosen.get("text") or "")


def scan_dots_skeleton_job_status(
    group_id: str,
    max_per_folder: int = 300,
    preferred_report_job_id: Optional[str] = None,
) -> Dict[str, str]:
    """
    Read dots / skeleton / **report** job JSONs from S3 (same YYYYMMDD prefix as group_id).

    Video worker handles dots|skeleton; **report worker** (`src/report_worker.py`) handles mode=report.
    Without a Live report worker, report jobs stay in `jobs/pending/` and you will not get PDF or email.

    ``preferred_report_job_id`` should be the report job id from this browser session when available
    so the status line matches the job the user just queued (not an older failed attempt).
    """
    gid = (group_id or "").strip()
    out: Dict[str, str] = {}
    if not gid or not AWS_BUCKET or AWS_BUCKET == "local":
        return out
    variants = set(get_group_id_variants(gid) or [gid])
    date_prefix = _YYYYMMDD_prefix_from_group_variants(gid)
    if not date_prefix:
        return out

    # Higher rank wins for dots/skeleton (single job per mode expected).
    best: Dict[str, Tuple[int, str]] = {}
    report_entries: List[Dict[str, Any]] = []

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
                    if mode not in ("dots", "skeleton", "report"):
                        continue
                    st = str(data.get("status") or "").strip().lower()
                    # Prefer "error" (worker exception) over stale "message" from older clients.
                    msg = str(data.get("error") or data.get("message") or "").strip()
                    if len(msg) > 2500:
                        msg = msg[:2497] + "..."
                    res_ok = (data.get("result") or {}).get("ok")
                    is_report = mode == "report"
                    job_id_row = str(data.get("job_id") or "").strip()
                    created_at_row = str(data.get("created_at") or "").strip()

                    if fname == "failed" or st == "failed":
                        if is_report:
                            report_entries.append(
                                {
                                    "job_id": job_id_row,
                                    "created_at": created_at_row,
                                    "text": (
                                        f"Failed: {msg}"
                                        if msg
                                        else "Failed — open this job JSON in jobs/failed/ or check report worker logs"
                                    ),
                                }
                            )
                        else:
                            note(
                                mode,
                                4,
                                (f"Failed: {msg}" if msg else "Failed — check video worker logs on the server"),
                            )
                    elif res_ok is False:
                        if is_report:
                            report_entries.append(
                                {
                                    "job_id": job_id_row,
                                    "created_at": created_at_row,
                                    "text": (f"Failed: {msg}" if msg else "Worker reported result ok=false"),
                                }
                            )
                        else:
                            note(mode, 4, (f"Failed: {msg}" if msg else "Worker reported result ok=false"))
                    elif fname == "processing" or st == "processing":
                        if is_report:
                            report_entries.append(
                                {
                                    "job_id": job_id_row,
                                    "created_at": created_at_row,
                                    "text": "Processing on **report worker** (PDF generation — can take several minutes)…",
                                }
                            )
                        else:
                            note(mode, 3, "Processing on video worker…")
                    elif fname == "pending" or st == "pending":
                        if is_report:
                            report_entries.append(
                                {
                                    "job_id": job_id_row,
                                    "created_at": created_at_row,
                                    "text": (
                                        "Queued — waiting for **report worker**. Ensure the **ai-people-reader-v2-report-worker** "
                                        "service is **Live** on Render (separate from the video worker)."
                                    ),
                                }
                            )
                        else:
                            note(
                                mode,
                                2,
                                "Queued — waiting for a free worker. If the queue is large this can take many minutes.",
                            )
                    elif fname == "finished" or st == "finished":
                        if is_report:
                            report_entries.append(
                                {
                                    "job_id": job_id_row,
                                    "created_at": created_at_row,
                                    "text": "Finished — report worker saved outputs; PDF should be on S3 / emailed.",
                                }
                            )
                        else:
                            note(mode, 1, "Worker finished — verifying file on S3…")
                if scanned > max_per_folder:
                    break
        except Exception:
            pass

    rline = _pick_report_status_line(report_entries, preferred_report_job_id)
    if rline:
        out["report"] = rline

    for mode, (_, text) in best.items():
        out[mode] = text
    return out


def is_valid_email_format(value: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", str(value or "").strip()))


def ensure_session_defaults() -> None:
    defaults = {
        "people_reader_last_group_id": "",
        "people_reader_submission_id_override": "",
        "people_reader_last_name": "",
        "people_reader_audience_mode": "one",
        "people_reader_jobs_by_group": {},
        "people_reader_notify_email": (os.getenv("PEOPLE_READER_DEFAULT_NOTIFY_EMAIL") or "").strip(),
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
    try:
        url = presigned_get_url(key, expires=3600, filename=filename)
    except Exception:
        st.caption(f"Waiting for {label}")
        return

    # When ready: same green pill for video + PDF (avoid st.download_button theme = mismatched brown).
    if ready:
        btn_html = (
            '<a href="' + url + '" target="_blank" rel="noopener" '
            'style="display: inline-block; padding: 0.5rem 1.25rem; background: #22c55e; '
            'color: #ffffff !important; border-radius: 6px; text-decoration: none; font-weight: 600;">'
            "✓ Download " + label + "</a>"
        )
        st.markdown(btn_html, unsafe_allow_html=True)
        return

    if not is_video:
        data = s3_get_bytes(key)
        if data and len(data) < 50 * 1024 * 1024:
            st.download_button(
                label=f"Download {label}",
                data=data,
                file_name=filename,
                mime=mime,
                key=button_key,
            )
            return
    st.link_button(f"Download {label}", url)


# -------------------------
# UI
# -------------------------
ensure_session_defaults()
apply_theme()
render_top_banner()

st.markdown(f"# {PAGE_TITLE}")
st.caption(
    "รายงาน **ภาษาอังกฤษอย่างเดียว** (PDF) แบบ People Reader — **Adaptability** + movement-type matching. "
    "ไม่ต้องตั้งค่าองค์กรใน Admin"
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

report_notify_email = st.text_input(
    "Email for English PDF report",
    key="people_reader_notify_email",
    placeholder="you@company.com",
    help="The report worker emails the English People Reader PDF here (AWS SES/SMTP).",
)

audience_mode = st.radio(
    "Audience Type",
    options=["one", "many"],
    format_func=lambda x: "Presenting to 1 person" if x == "one" else "Presenting to multiple people",
    horizontal=True,
    key="people_reader_audience_mode",
)

_combo_img = combination_10_types_image_path()
if _combo_img:
    with st.expander("Movement type reference — 10 types (product table)", expanded=True):
        st.image(_combo_img, use_container_width=True)
        st.caption(
            "Columns: Eye contact · Stance · Uprightness · Engaging · Authority · Confidence · Adaptability "
            "(low / moderate / high). Types 7–10 use the same rubric; weighted Auto match uses `movement_type_classifier.py`."
        )

movement_type_mode = st.selectbox(
    "Movement type (report profile)",
    options=[c[1] for c in PEOPLE_READER_MOVEMENT_TYPE_CHOICES],
    format_func=lambda v: next(label for label, key in PEOPLE_READER_MOVEMENT_TYPE_CHOICES if key == v),
    index=0,
    key="people_reader_movement_type_mode",
    help="Auto: classify the video to the nearest of **10** profiles (7-dim match + legacy score) and blend scores into the report. "
    "Or choose a type to align Engaging, Confidence, Authority, Adaptability and first-impression cues to that profile.",
)

# --- Direct-to-S3 upload (browser → S3 via presigned URL, bypasses Streamlit server) ---
if "direct_upload_done" not in st.session_state:
    st.session_state["direct_upload_done"] = False
if "direct_upload_key" not in st.session_state:
    st.session_state["direct_upload_key"] = ""
if "direct_upload_group_id" not in st.session_state:
    st.session_state["direct_upload_group_id"] = ""
if "direct_upload_filename" not in st.session_state:
    st.session_state["direct_upload_filename"] = ""

ACCEPTED_VIDEO_TYPES = "video/mp4,video/quicktime,video/x-m4v,video/webm,.mp4,.mov,.m4v,.webm"

def _render_direct_uploader(presigned_url: str, upload_id: str) -> None:
    """Inject HTML/JS that uploads a file directly to S3 via presigned PUT URL."""
    html = f"""
    <div id="uploader-{upload_id}" style="font-family:sans-serif;">
      <input type="file" id="file-{upload_id}" accept="{ACCEPTED_VIDEO_TYPES}"
             style="margin-bottom:8px;color:#e6d9c8;" />
      <div id="info-{upload_id}" style="color:#ccbda8;font-size:0.85rem;margin-bottom:8px;"></div>
      <button id="btn-{upload_id}" disabled
              style="background:linear-gradient(180deg,#c9a67a,#b48d5f);color:#231d17;
                     border:0;padding:10px 24px;border-radius:6px;font-weight:600;font-size:1rem;
                     cursor:pointer;width:100%;opacity:0.5;">
        Upload &amp; Start Analysis
      </button>
      <div id="progress-wrap-{upload_id}" style="display:none;margin-top:10px;">
        <div style="background:#3a332d;border-radius:6px;overflow:hidden;height:22px;">
          <div id="bar-{upload_id}" style="background:linear-gradient(90deg,#c9a67a,#b48d5f);
               height:100%;width:0%;transition:width 0.2s;display:flex;align-items:center;
               justify-content:center;font-size:0.8rem;color:#231d17;font-weight:600;">0%</div>
        </div>
        <div id="speed-{upload_id}" style="color:#ccbda8;font-size:0.8rem;margin-top:4px;"></div>
      </div>
      <div id="status-{upload_id}" style="margin-top:8px;font-size:0.9rem;"></div>
    </div>
    <script>
    (function() {{
      const fInput = document.getElementById('file-{upload_id}');
      const btn = document.getElementById('btn-{upload_id}');
      const info = document.getElementById('info-{upload_id}');
      const bar = document.getElementById('bar-{upload_id}');
      const progWrap = document.getElementById('progress-wrap-{upload_id}');
      const speedEl = document.getElementById('speed-{upload_id}');
      const statusEl = document.getElementById('status-{upload_id}');
      const url = {json.dumps(presigned_url)};

      fInput.addEventListener('change', function() {{
        const f = fInput.files[0];
        if (!f) {{ btn.disabled = true; btn.style.opacity='0.5'; info.textContent=''; return; }}
        const mb = (f.size / 1048576).toFixed(2);
        info.textContent = f.name + ' (' + mb + ' MB)';
        btn.disabled = false;
        btn.style.opacity = '1';
      }});

      btn.addEventListener('click', function() {{
        const f = fInput.files[0];
        if (!f) return;
        btn.disabled = true;
        btn.style.opacity = '0.5';
        fInput.disabled = true;
        progWrap.style.display = 'block';
        statusEl.innerHTML = '<span style="color:#c9a67a;">Uploading directly to cloud...</span>';

        const startTime = Date.now();
        const xhr = new XMLHttpRequest();
        xhr.open('PUT', url, true);
        xhr.setRequestHeader('Content-Type', f.type || 'video/mp4');

        xhr.upload.addEventListener('progress', function(e) {{
          if (e.lengthComputable) {{
            const pct = Math.round(e.loaded / e.total * 100);
            bar.style.width = pct + '%';
            bar.textContent = pct + '%';
            const elapsed = (Date.now() - startTime) / 1000;
            if (elapsed > 0.5) {{
              const mbps = (e.loaded / 1048576 / elapsed).toFixed(1);
              const remain = ((e.total - e.loaded) / (e.loaded / elapsed) );
              speedEl.textContent = mbps + ' MB/s — ~' + Math.ceil(remain) + 's remaining';
            }}
          }}
        }});

        xhr.addEventListener('load', function() {{
          if (xhr.status >= 200 && xhr.status < 300) {{
            bar.style.width = '100%';
            bar.textContent = '100%';
            speedEl.textContent = '';
            statusEl.innerHTML = '<span style="color:#4ade80;">✓ Upload complete! Creating analysis jobs...</span>';
            // Signal Streamlit to proceed
            const data = {{ uploaded: true, filename: f.name, size: f.size }};
            window.parent.postMessage({{
              type: 'streamlit:setComponentValue',
              value: JSON.stringify(data)
            }}, '*');
            // Also set a marker on the iframe for polling
            document.getElementById('uploader-{upload_id}').dataset.done = 'true';
            document.getElementById('uploader-{upload_id}').dataset.filename = f.name;
            document.getElementById('uploader-{upload_id}').dataset.filesize = f.size;
          }} else {{
            statusEl.innerHTML = '<span style="color:#f87171;">Upload failed (HTTP ' + xhr.status + '). Please try again.</span>';
            btn.disabled = false;
            btn.style.opacity = '1';
            fInput.disabled = false;
          }}
        }});

        xhr.addEventListener('error', function() {{
          statusEl.innerHTML = '<span style="color:#f87171;">Network error. Please check your connection and try again.</span>';
          btn.disabled = false;
          btn.style.opacity = '1';
          fInput.disabled = false;
        }});

        xhr.send(f);
      }});
    }})();
    </script>
    """
    import streamlit.components.v1 as components
    components.html(html, height=200)


def _start_direct_upload_flow() -> None:
    """Pre-generate group_id + presigned URL, render the uploader, and wait."""
    if not name_value.strip():
        st.warning("กรุณากรอกชื่อก่อนอัปโหลดวิดีโอ")
        return
    if not str(report_notify_email or "").strip() or not is_valid_email_format(str(report_notify_email)):
        st.warning("กรุณากรอกอีเมลที่ถูกต้องก่อนอัปโหลดวิดีโอ")
        return

    base_user = safe_slug(name_value, fallback="user")
    group_id = st.session_state.get("direct_upload_group_id") or ""
    if not group_id:
        group_id = f"{new_group_id()}__{base_user}"
        st.session_state["direct_upload_group_id"] = group_id

    input_key = f"{JOBS_GROUP_PREFIX}{group_id}/input/input.mp4"
    st.session_state["direct_upload_key"] = input_key

    try:
        url = presigned_put_url(input_key, expires=1800)
    except Exception as e:
        st.error(f"ไม่สามารถสร้าง upload URL ได้: {e}")
        return

    upload_id = group_id.replace(".", "_").replace("-", "_")
    _render_direct_uploader(url, upload_id)


# Show the direct uploader if not yet uploaded
if not st.session_state["direct_upload_done"]:
    st.markdown("**Video (MP4 / MOV / M4V / WEBM)**")
    _start_direct_upload_flow()

    confirm = st.button(
        "✓ Upload finished — Create Jobs",
        type="primary",
        use_container_width=True,
        help="Click after the upload bar shows 100%",
    )
    st.caption(SUPPORT_TEXT)

    if confirm:
        input_key = st.session_state.get("direct_upload_key", "")
        if input_key and s3_key_exists(input_key):
            st.session_state["direct_upload_done"] = True
            st.rerun()
        else:
            st.error("วิดีโอยังอัปโหลดไม่เสร็จ กรุณารอให้ progress bar ถึง 100% ก่อนกดปุ่ม")
    st.stop()

# --- Upload confirmed — create jobs ---
group_id = st.session_state["direct_upload_group_id"]
input_key = st.session_state["direct_upload_key"]
uploaded_filename = st.session_state.get("direct_upload_filename") or "input.mp4"
st.session_state["direct_upload_done"] = False
st.session_state["direct_upload_group_id"] = ""
st.session_state["direct_upload_key"] = ""
st.session_state["direct_upload_filename"] = ""

if True:
    org_payload = people_reader_fixed_settings()
    if not name_value.strip():
        st.warning("ขั้นตอน 6 — ชื่อ: กรุณากรอกชื่อก่อน")
        st.stop()
    if not str(report_notify_email or "").strip():
        st.warning("ขั้นตอน 6 — อีเมล: กรุณากรอกอีเมลสำหรับส่งรายงาน PDF")
        st.stop()
    if not is_valid_email_format(str(report_notify_email)):
        st.warning("ขั้นตอน 6 — อีเมล: รูปแบบอีเมลไม่ถูกต้อง")
        st.stop()

    langs = [str(x).strip().lower() for x in (org_payload.get("languages") or ["en"]) if str(x).strip()]
    if not langs:
        langs = ["en"]

    report_fmt = str(org_payload.get("report_format") or "pdf").strip().lower()
    if report_fmt not in ("docx", "pdf"):
        report_fmt = "pdf"
    want_dots = bool(org_payload.get("enable_dots", True))
    want_skeleton = bool(org_payload.get("enable_skeleton", True))
    org_display = str(org_payload.get("organization_name") or PAGE_TITLE).strip()
    org_id = normalize_org_name(org_display)
    base_user = safe_slug(name_value, fallback="user")

    with st.status("Creating analysis jobs...", expanded=True) as status:
        st.write("✓ Video already uploaded directly to S3.")
        st.write("✓ Video uploaded. Queuing analysis jobs...")
        outputs = build_output_keys(group_id)
        created_at = utc_now_iso()

        org_meta = {
            "organization_name": org_display,
            "organization_id": org_id,
            "required_report_style": "people_reader",
        }

        job_dots_id = ""
        job_skeleton_id = ""
        job_report_id = ""

        if want_dots:
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
                # Report worker sends one bundle email (reports + dots + skeleton); skip worker.py per-mode mail.
                "suppress_completion_email": True,
                **org_meta,
            }
            job_dots_id = job_dots["job_id"]

        if want_skeleton:
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
                # Video worker: when dots is also queued, wait for dots.mp4 on S3 before encoding skeleton
                # (avoids skeleton finishing first when multiple video-worker replicas run in parallel).
                "require_dots_output": bool(want_dots),
                **org_meta,
            }
            job_skeleton_id = job_skeleton["job_id"]

        job_report = {
            "job_id": new_job_id(),
            "group_id": group_id,
            "created_at": created_at,
            "status": "pending",
            "mode": "report",
            "input_key": input_key,
            "client_name": name_value,
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "languages": langs,
            "output_prefix": f"{JOBS_GROUP_PREFIX}{group_id}",
            "analysis_mode": "real",
            "sample_fps": 3,
            "max_frames": 150,
            # Worker resolves layout via people_reader_job=True first (never drift to legacy "full").
            "people_reader_job": True,
            "report_style": "people_reader",
            "required_report_style": "people_reader",
            "report_format": report_fmt,
            "expect_skeleton": want_skeleton,
            "expect_dots": want_dots,
            "notify_email": str(report_notify_email or "").strip(),
            "enterprise_folder": org_display,
            "employee_id": base_user,
            "employee_email": "",
            "audience_mode": audience_mode,
            "movement_type_mode": movement_type_mode,
            "report_email_send_en_asap": False,
            "bundle_completion_email": True,
            **org_meta,
        }
        job_report_id = job_report["job_id"]

        try:
            if want_dots:
                enqueue_legacy_job(job_dots)
            if want_skeleton:
                enqueue_legacy_job(job_skeleton)
            enqueue_legacy_job(job_report)
        except Exception as e:
            status.update(label="Queue failed", state="error")
            st.error(f"ขั้นตอน 8 — คิวงาน: ล้มเหลว — {e}")
            st.stop()

        st.write("✓ Jobs queued.")
        status.update(label="Complete", state="complete")
        st.session_state["people_reader_last_group_id"] = group_id
        st.session_state["people_reader_submission_id_override"] = group_id
        if "people_reader_jobs_by_group" not in st.session_state:
            st.session_state["people_reader_jobs_by_group"] = {}
        st.session_state["people_reader_jobs_by_group"][group_id] = {
            "dots_job_id": job_dots_id,
            "skeleton_job_id": job_skeleton_id,
            "report_job_id": job_report_id,
        }
        if "people_reader_langs_by_group" not in st.session_state:
            st.session_state["people_reader_langs_by_group"] = {}
        st.session_state["people_reader_langs_by_group"][group_id] = list(langs)
        st.success(f"Submission received. Group ID: `{group_id}`")
        if job_report_id:
            st.caption(f"Report job queued: `{job_report_id}` (needs **report worker** running on Render).")
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
    merge_report_en_pdf_from_finished_job(current_group_id, resolved)
    merge_report_th_pdf_from_finished_job(current_group_id, resolved)
    dots_key = str(resolved.get("dots_video") or "").strip()
    skel_key = str(resolved.get("skeleton_video") or "").strip()
    report_en_pdf_key = str(resolved.get("report_en_pdf") or "").strip()
    report_th_pdf_key = str(resolved.get("report_th_pdf") or "").strip()
    dots_ready = bool(dots_key) and s3_output_ready(dots_key)
    skeleton_ready = bool(skel_key) and s3_output_ready(skel_key)
    en_report_ready = bool(report_en_pdf_key) and s3_output_ready(report_en_pdf_key)
    th_report_ready = bool(report_th_pdf_key) and s3_output_ready(report_th_pdf_key)
    if not dots_ready or not skeleton_ready:
        discover_video_outputs_from_finished_jobs(current_group_id, resolved)
        dots_key = str(resolved.get("dots_video") or "").strip()
        skel_key = str(resolved.get("skeleton_video") or "").strip()
        dots_ready = bool(dots_key) and s3_output_ready(dots_key)
        skeleton_ready = bool(skel_key) and s3_output_ready(skel_key)
    if not en_report_ready or not th_report_ready:
        merge_report_en_pdf_from_finished_job(current_group_id, resolved)
        merge_report_th_pdf_from_finished_job(current_group_id, resolved)
        report_en_pdf_key = str(resolved.get("report_en_pdf") or "").strip()
        report_th_pdf_key = str(resolved.get("report_th_pdf") or "").strip()
        en_report_ready = bool(report_en_pdf_key) and s3_output_ready(report_en_pdf_key)
        th_report_ready = bool(report_th_pdf_key) and s3_output_ready(report_th_pdf_key)

    report_job_for_langs = find_finished_report_job_for_group(current_group_id)
    _langs_sel = (st.session_state.get("people_reader_langs_by_group") or {}).get(current_group_id)
    if not _langs_sel and report_job_for_langs and report_job_for_langs.get("languages"):
        _raw = report_job_for_langs["languages"]
        if isinstance(_raw, str):
            _raw = [_raw]
        _langs_sel = [str(x).strip().lower() for x in _raw if str(x).strip()]
    if not _langs_sel:
        _langs_sel = ["en"]
    wants_th = any(x.startswith("th") for x in _langs_sel)
    wants_en = any(x.startswith("en") for x in _langs_sel)
    expect_dots = bool(report_job_for_langs.get("expect_dots", True)) if report_job_for_langs else True
    expect_skeleton = bool(report_job_for_langs.get("expect_skeleton", True)) if report_job_for_langs else True

    status_items: List[Tuple[str, bool]] = []
    if expect_dots:
        status_items.append(("Dots Video", dots_ready))
    if expect_skeleton:
        status_items.append(("Skeleton Video", skeleton_ready))
    if wants_th:
        status_items.append(("Thai Report (PDF)", th_report_ready))
    if wants_en:
        status_items.append(("English Report (PDF)", en_report_ready))
    if not status_items:
        status_items = [("Outputs", False)]

    n_items = len(status_items)
    overall_pct = int(round((sum(1 for _, ready in status_items if ready) / max(n_items, 1)) * 100))
    st.progress(overall_pct, text=f"Overall progress: {overall_pct}%")

    _jobs_by_group = st.session_state.get("people_reader_jobs_by_group") or {}
    _session_report_job_id = str((_jobs_by_group.get(current_group_id) or {}).get("report_job_id") or "").strip() or None
    job_hints = scan_dots_skeleton_job_status(
        current_group_id,
        preferred_report_job_id=_session_report_job_id,
    )
    _report_hint_line = str(job_hints.get("report") or "").strip()
    _report_status_says_finished = _report_hint_line.startswith("Finished")
    if overall_pct < 100:
        st.caption(
            "**Video worker** (`worker.py`) สร้าง dots + skeleton แยกงาน — แต่ละรายการจะขึ้น **Ready** เมื่อไฟล์ขึ้น S3 แล้ว "
            "**Report worker** (`src/report_worker.py`) สร้างรายงานตามภาษาที่เลือก — เป็น **บริการ Render แยก** ต้อง **Live**. "
            "เมื่อครบ **100%** ระบบจะส่ง **อีเมลเดียว** (รายงาน + dots + skeleton) ไปที่อีเมลที่กรอกไว้"
        )
        with st.expander("Job status from S3 (same day as Group ID)", expanded=True):
            st.write(
                f"**Dots:** {job_hints.get('dots', '— no matching job JSON found under jobs/pending|processing|finished|failed for this group —')}"
            )
            st.write(
                f"**Skeleton:** {job_hints.get('skeleton', '— same —')}"
            )
            st.write(
                f"**Report (PDF):** {job_hints.get('report', '— no report job JSON found — If you just submitted, wait; otherwise the report job may use a different date prefix or bucket.')}"
            )
            _rid = _session_report_job_id
            if _rid:
                st.caption(
                    f"Report **job_id** (stored when you submitted in this browser): `{_rid}` "
                    f"→ look on S3 for `jobs/pending/{_rid}.json` (or processing/finished/failed)."
                )
        if job_hints.get("dots", "").startswith("Failed") or job_hints.get("skeleton", "").startswith("Failed"):
            st.error(
                "At least one video job failed. Fix the error (often ffmpeg/MediaPipe on the worker), then upload again."
            )
        if job_hints.get("report", "").startswith("Failed"):
            st.error(
                "Report job failed — no PDF and no email. Read the error above, check **jobs/failed/** for this job_id, "
                "and report worker logs on Render."
            )
            _rh = report_failure_user_hint(job_hints.get("report", ""))
            if _rh:
                st.info(_rh)
        # Full reruns while waiting break st.file_uploader (HTTP 400 on /_stcore/upload_file) — keep polling opt-in.
        st.checkbox(
            "Auto-refresh every 15s while waiting for dots/skeleton/report (turn **off** before uploading videos)",
            value=False,
            key="people_reader_enable_s3_poll",
        )
        if st.session_state.get("people_reader_enable_s3_poll"):
            try:

                @st.fragment(run_every=timedelta(seconds=15))
                def _people_reader_auto_refresh():
                    st.rerun()

                _people_reader_auto_refresh()
            except Exception:
                st.caption("Polling unavailable in this Streamlit version — use **Refresh**.")
        else:
            st.caption("Tip: click **Refresh** to poll S3, or enable auto-refresh above (disable it before file upload).")

    for label, ready in status_items:
        if ready:
            bar_pct, bar_txt = 100, f"{label}: Ready to download"
        elif label == "English Report (PDF)" and wants_en and (not en_report_ready) and _report_status_says_finished and bool(
            report_en_pdf_key
        ):
            # Job JSON can show "Finished" while Streamlit IAM/HEAD has not yet seen the PDF — avoid "Finished" vs Processing clash.
            bar_pct, bar_txt = 50, f"{label}: Worker finished — verifying PDF on S3…"
        elif label == "Thai Report (PDF)" and wants_th and (not th_report_ready) and _report_status_says_finished and bool(
            report_th_pdf_key
        ):
            bar_pct, bar_txt = 50, f"{label}: Worker finished — verifying PDF on S3…"
        else:
            bar_pct, bar_txt = 0, f"{label}: Processing"
        st.progress(bar_pct, text=bar_txt)

    st.markdown("---")
    st.subheader("Available Downloads")
    st.caption(
        "แต่ละไฟล์จะกดดาวน์โหลดได้เมื่อขึ้น **Ready to download** ด้านบน — "
        "เมื่อครบ **100%** ระบบจะส่ง **อีเมลเดียว** (รายงาน + วิดีโอ) ไปที่อีเมลที่คุณกรอก"
    )

    if expect_dots:
        render_download_button(
            label="Dots Video",
            key=resolved.get("dots_video", ""),
            filename="dots.mp4",
            mime="video/mp4",
            button_key="people_reader_dl_dots",
            ready=dots_ready,
        )
    if expect_skeleton:
        render_download_button(
            label="Skeleton Video",
            key=resolved.get("skeleton_video", ""),
            filename="skeleton.mp4",
            mime="video/mp4",
            button_key="people_reader_dl_skeleton",
            ready=skeleton_ready,
        )
    if wants_th:
        render_download_button(
            label="Thai Report (PDF)",
            key=report_th_pdf_key,
            filename="People_Reader_report_TH.pdf",
            mime="application/pdf",
            button_key="people_reader_dl_report_th_pdf",
            ready=th_report_ready,
        )
    if wants_en:
        render_download_button(
            label="English Report (PDF)",
            key=report_en_pdf_key,
            filename="People_Reader_report_EN.pdf",
            mime="application/pdf",
            button_key="people_reader_dl_report_en_pdf",
            ready=en_report_ready,
        )
    if wants_en and (not en_report_ready) and report_en_pdf_key:
        st.caption("English report key found but the file is not visible yet — click **Refresh**.")
    if wants_th and (not th_report_ready) and report_th_pdf_key:
        st.caption("Thai report key found but the file is not visible yet — click **Refresh**.")

    st.markdown("---")
    st.subheader("Category levels in your report (Low / Moderate / High)")
    st.caption(
        "Same scales as the PDF/DOCX (Engaging, Confidence, Authority, Adaptability). "
        "Loaded from the finished report job on S3 — useful if email is slow or missing."
    )
    try:
        report_job = find_finished_report_job_for_group(current_group_id)
        mt = (report_job or {}).get("movement_type_info") if report_job else None
    except Exception as e:
        report_job = None
        mt = None
        st.caption(f"Could not read report job from S3: {e}")

    if report_job:
        rcv = str(report_job.get("report_core_version") or "").strip()
        gsha = str(report_job.get("report_worker_git_sha") or "").strip()
        if rcv or gsha:
            st.caption(
                f"**Report build on S3:** `report_core_version={rcv or '—'}` · "
                f"`git={gsha or '—'}` — if this does not match after deploy, the **report worker** "
                "may still be on old code (use Render **Clear build cache & deploy**)."
            )
    if report_job and isinstance(report_job.get("notification"), dict):
        with st.expander("Email delivery status (from finished report job on S3)", expanded=False):
            st.json(report_job["notification"])

    if mt and isinstance(mt, dict):
        levels = mt.get("seven_chosen_template_levels") or []
        if isinstance(levels, list) and len(levels) >= 7:
            eng = levels[3]
            auth = levels[4]
            conf = levels[5]
            adap = levels[6]
            rows = [
                ("Engaging & Connecting", "การมีส่วนร่วมและการเชื่อมโยง", eng),
                ("Confidence", "ความมั่นใจ", conf),
                ("Authority", "ความเป็นผู้นำและอำนาจ", auth),
                ("Adaptability", "ความยืดหยุ่นในการปรับตัว", adap),
            ]
            md_lines = [
                "| Category | หมวด (TH) | Scale | ระดับ |",
                "|---|---|:---:|---|",
            ]
            for en_label, th_label, lv in rows:
                md_lines.append(
                    f"| {en_label} | {th_label} | {_format_scale_en(lv)} | {_format_scale_th(lv)} |"
                )
            st.markdown("\n".join(md_lines))

            tname = str(mt.get("type_name") or "—")
            m7 = int(mt.get("seven_match_chosen_matches") or 0)
            p7 = int(mt.get("confidence_pct") or 0)
            mode_note = str(mt.get("mode_en") or mt.get("mode") or "")
            st.info(
                f"**Closest profile:** {tname} {mode_note}  \n"
                f"**7-dimension match:** {m7}/7 ({p7}%)"
            )

            top_en = str(mt.get("seven_match_line_en") or "").strip()
            if top_en:
                with st.expander("Top two profile matches (7 dimensions)", expanded=False):
                    st.write(top_en)
                    st.caption(str(mt.get("seven_match_line_th") or ""))
        else:
            st.warning(
                "Report job found but category levels are incomplete. Try **Refresh** after the report worker finishes."
            )
    elif report_job:
        is_people_reader_job = str(report_job.get("report_style") or "").strip().lower().startswith(
            "people_reader"
        ) or str(report_job.get("enterprise_folder") or "").strip().lower() == "people reader"
        if is_people_reader_job:
            rcv = str(report_job.get("report_core_version") or "").strip()
            if rcv:
                st.info(
                    "This report was built with a **current** worker (`report_core_version` is set on the job), "
                    "but **movement type details** are still missing. That usually means the movement-classification "
                    "step failed or was skipped (see report worker logs for `[movement_type]`). "
                    "The PDF can still show Engaging, Confidence, Authority, and Adaptability from analysis."
                )
            else:
                st.info(
                    "Report finished, but **movement type details** are not in this job file "
                    "(job processed by an **older** worker before this metadata was stored). "
                    "**Redeploy** the latest **report worker**, then run a **new** upload with a new Group ID to see "
                    "category levels here; the PDF may still list the four categories."
                )
        else:
            st.info(
                "Report job finished, but **movement type details** are missing "
                "(non–People Reader report). PDF may still list categories from analysis only."
            )
    else:
        st.info(
            "No finished **report** job found for this Group ID yet (or S3 scan limit reached). "
            "When the report worker finishes, click **Refresh** — category levels appear here even if email has not arrived."
        )

    _any_report_ready = (wants_en and en_report_ready) or (wants_th and th_report_ready)
    if not any([dots_ready, skeleton_ready, _any_report_ready]):
        st.caption("Files are not ready yet. Please wait a few minutes and click Refresh.")

else:
    st.caption("Upload a video above or paste an existing Group ID to download results.")

st.divider()
st.caption(
    "Movement type (Auto) ranks **10** profiles in **`movement_type_classifier.py`** by **weighted template score** "
    "(pose features vs each profile’s `expected` ranges), then uses **7-dimension agreement** as a tie-break. "
    "The seven rubric levels (`people_reader_seven`) are eye, stance, upright, engaging, authority, confidence, adaptability; "
    "video side uses pose-summary tertiles (1–4) and composites as high/low (5–7). Report bars follow the chosen profile. "
    "No S3 calibration."
)

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