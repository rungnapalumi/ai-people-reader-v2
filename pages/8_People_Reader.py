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
ORG_SETTINGS_PREFIX = "jobs/config/organizations/"

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


def org_settings_key(org_name: str) -> str:
    org_id = normalize_org_name(org_name)
    return f"{ORG_SETTINGS_PREFIX}{org_id}.json" if org_id else ""


def list_people_reader_organization_names() -> List[str]:
    """Organizations whose Admin setting `report_style` is `people_reader`."""
    if not AWS_BUCKET or AWS_BUCKET == "local":
        return []
    out: List[str] = []
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=ORG_SETTINGS_PREFIX):
            for item in page.get("Contents", []):
                key = str(item.get("Key") or "")
                if not key.endswith(".json"):
                    continue
                payload = s3_read_json(key) or {}
                if str(payload.get("report_style") or "").strip().lower() != "people_reader":
                    continue
                nm = str(payload.get("organization_name") or "").strip()
                if nm:
                    out.append(nm)
    except Exception:
        return []
    out.sort(key=lambda x: x.lower())
    # de-dupe preserve order
    seen = set()
    uniq: List[str] = []
    for n in out:
        k = n.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(n)
    return uniq


def default_org_hint_name() -> str:
    """Prefer org whose default_page is AI People Reader."""
    for nm in list_people_reader_organization_names():
        payload = s3_read_json(org_settings_key(nm)) or {}
        if str(payload.get("default_page") or "").strip().lower() == "ai_people_reader":
            return nm
    names = list_people_reader_organization_names()
    return names[0] if names else ""


def validate_organization_for_people_reader(org_name: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Returns (ok, error_message_with_step, org_payload).
    On success, org_payload is the S3 JSON for the organization.
    """
    on = (org_name or "").strip()
    if not on or on.startswith("—"):
        return (
            False,
            "ขั้นตอน 2 — เลือกองค์กร: กรุณาเลือกองค์กร (Organization) ที่ได้รับอนุญาตให้ใช้รายงาน **People Reader** เท่านั้น",
            {},
        )
    key = org_settings_key(on)
    if not key:
        return False, "ขั้นตอน 2 — ชื่อองค์กรไม่ถูกต้อง", {}
    payload = s3_read_json(key) or {}
    if not payload:
        return (
            False,
            "ขั้นตอน 3 — โหลดการตั้งค่า: ไม่พบไฟล์ตั้งค่าองค์กรในระบบ "
            f"(`{key}`). สร้างและบันทึกองค์กรใน **Admin → Organization Settings** ก่อน "
            "และตั้ง **Default Report Type = People Reader**",
            {},
        )
    style = str(payload.get("report_style") or "").strip().lower()
    if style != "people_reader":
        return (
            False,
            "ขั้นตอน 4 — ประเภทรายงาน: องค์กรนี้ตั้งค่าเป็น "
            f"**{style}** ไม่ใช่ **people_reader** — หน้านี้จะไม่ส่งงานไปทำรายงานแบบอื่นแทน "
            "แก้ใน Admin ให้เป็น **People Reader** เท่านั้น",
            {},
        )
    if not (bool(payload.get("enable_report_th", True)) or bool(payload.get("enable_report_en", True))):
        return (
            False,
            "ขั้นตอน 5 — รายงาน PDF: องค์กรนี้ปิดทั้งรายงานไทยและอังกฤษในการตั้งค่า "
            "เปิดอย่างน้อยหนึ่งภาษาใน Admin",
            {},
        )
    return True, "", payload


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


def find_finished_report_job_for_group(group_id: str, max_json: int = 800) -> Optional[Dict[str, Any]]:
    """
    Locate the latest finished `mode=report` job JSON for this group_id under jobs/finished/.
    Used to show People Reader category levels on the page when email is delayed or missing.
    """
    gid = (group_id or "").strip()
    if not gid or not AWS_BUCKET or AWS_BUCKET == "local":
        return None
    variants = set(get_group_id_variants(gid) or [gid])
    date_prefix = gid[:8] if len(gid) >= 8 and gid[:8].isdigit() else ""
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
    "เลือก **Organization** ที่ Admin ตั้งค่าเป็น **People Reader** เท่านั้น — ระบบจะไม่สร้างรายงาน Full/Simple/Operational Test แทน "
    "อัปโหลดวิดีโอแล้วดาวน์โหลดผลได้จากหน้านี้ (Engaging, Confidence, Authority, Effort/Shape + **Adaptability**)."
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

selected_org_label = ""
people_reader_orgs = list_people_reader_organization_names()
if not people_reader_orgs:
    st.error(
        "**ขั้นตอน 1 — องค์กร:** ยังไม่มีองค์กรที่ตั้งค่า **Default Report Type = People Reader** ใน "
        "**Admin → Organization Settings**. หน้านี้จะไม่ส่งงานไปทำรายงานแบบอื่น — ต้องตั้งค่าก่อน"
    )
    st.caption(
        "ใน Admin: ใส่ชื่อองค์กร → เลือก **People Reader** ใน Default Report Type → เปิดรายงานไทย/อังกฤษตามต้องการ → บันทึก"
    )
else:
    org_opts = ["— เลือกองค์กร —"] + people_reader_orgs
    hint = default_org_hint_name()
    _org_index = org_opts.index(hint) if hint and hint in org_opts else 0
    org_select_widget = st.selectbox(
        "Organization",
        options=org_opts,
        index=_org_index,
        key="people_reader_org_select",
        help="เฉพาะองค์กรที่ Admin ตั้งค่าเป็น **People Reader** เท่านั้น — ระบบจะไม่สลับไปรายงาน Full/Simple/Operational Test",
    )
    selected_org_label = (
        ""
        if (not org_select_widget or str(org_select_widget).startswith("—"))
        else str(org_select_widget).strip()
    )

name_value = st.text_input(
    "Name",
    value=str(st.session_state.get("people_reader_last_name") or ""),
    placeholder="Enter your name",
)

if name_value:
    st.session_state["people_reader_last_name"] = name_value

report_notify_email = st.text_input(
    "Email for report PDFs (Thai & English)",
    key="people_reader_notify_email",
    placeholder="you@company.com",
    help="The report worker sends both PDFs to this address (AWS SES/SMTP). Also check spam.",
)

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

report_email_send_en_asap = st.checkbox(
    "Email English PDF as soon as it is ready (do not wait for Thai PDF)",
    value=True,
    key="people_reader_report_email_send_en_asap",
    help="The report worker generates English first, uploads it to S3, then can email the English report "
    "before Thai finishes. Thai PDF follows in a separate email when ready.",
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
    disabled=(
        uploaded is None
        or not AWS_BUCKET
        or AWS_BUCKET == "local"
        or not people_reader_orgs
        or not selected_org_label
    ),
)

st.caption(SUPPORT_TEXT)

if run:
    if not AWS_BUCKET or AWS_BUCKET == "local":
        st.error("ขั้นตอน 1 — โครงสร้างพื้นฐาน: ต้องตั้งค่า S3 (AWS_BUCKET) ใน .env")
        st.stop()
    if not people_reader_orgs:
        st.error("ขั้นตอน 1 — องค์กร: ยังไม่มีองค์กร People Reader ใน Admin")
        st.stop()
    ok_org, org_err, org_payload = validate_organization_for_people_reader(selected_org_label)
    if not ok_org:
        st.error(org_err)
        st.stop()
    if uploaded is None:
        st.warning("ขั้นตอน 6 — วิดีโอ: กรุณาอัปโหลดไฟล์วิดีโอก่อน")
        st.stop()
    if not name_value.strip():
        st.warning("ขั้นตอน 6 — ชื่อ: กรุณากรอกชื่อก่อน")
        st.stop()
    if not str(report_notify_email or "").strip():
        st.warning("ขั้นตอน 6 — อีเมล: กรุณากรอกอีเมลสำหรับส่งรายงาน PDF")
        st.stop()
    if not is_valid_email_format(str(report_notify_email)):
        st.warning("ขั้นตอน 6 — อีเมล: รูปแบบอีเมลไม่ถูกต้อง")
        st.stop()

    langs: List[str] = []
    if bool(org_payload.get("enable_report_th", True)):
        langs.append("th")
    if bool(org_payload.get("enable_report_en", True)):
        langs.append("en")
    if not langs:
        st.error("ขั้นตอน 5 — รายงาน: องค์กรนี้ปิดรายงานทั้งสองภาษา — แก้ใน Admin")
        st.stop()

    report_fmt = str(org_payload.get("report_format") or "pdf").strip().lower()
    if report_fmt not in ("docx", "pdf"):
        report_fmt = "pdf"
    want_dots = bool(org_payload.get("enable_dots", True))
    want_skeleton = bool(org_payload.get("enable_skeleton", True))
    org_display = str(org_payload.get("organization_name") or selected_org_label).strip()
    org_id = normalize_org_name(org_display)

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
            st.error(f"ขั้นตอน 7 — อัปโหลดวิดีโอ: ล้มเหลว — {e}")
            st.stop()

        # Do not block on HEAD after upload: upload_fileobj success means the object is there.
        # Extra HEAD often fails on misconfigured IAM (HeadObject denied) and looked like "Verification failed".
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
            "report_email_send_en_asap": bool(report_email_send_en_asap),
            **org_meta,
        }

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
        # Full reruns while waiting break st.file_uploader (HTTP 400 on /_stcore/upload_file) — keep polling opt-in.
        st.checkbox(
            "Auto-refresh every 15s while waiting for dots/skeleton (turn **off** before uploading videos)",
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
        st.progress(100 if ready else 0, text=f"{label}: {'Ready' if ready else 'Processing'}")

    st.markdown("---")
    st.subheader("Available Downloads")
    st.caption(
        "Reports (TH/EN PDF) are emailed by the **report worker** to the address you entered at upload. "
        "You can also download PDFs here when they appear."
    )

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

    if not any([dots_ready, skeleton_ready]):
        st.caption("Files are not ready yet. Please wait a few minutes and click Refresh.")

else:
    st.caption("Upload a video above or paste an existing Group ID to download results.")

st.divider()
st.caption(
    "Movement type (Auto) ranks the six profiles in **`movement_type_classifier.py`** by **weighted template score** "
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