# report_worker.py — AI People Reader Report Worker (TH/EN DOCX/PDF)
#
# ✅ This worker:
#   - Polls S3 jobs/pending for *.json
#   - Processes jobs with mode="report" (or "report_th_en")
#   - Downloads input video from S3
#   - Generates Graph 1/2, DOCX TH+EN, PDF TH+EN (PDF TH requires Thai TTF in repo)
#   - Uploads outputs back to S3
#   - Moves job JSON to jobs/finished or jobs/failed
#
# Job JSON minimal example:
# {
#   "job_id": "20260125_010203__abc12",
#   "mode": "report",
#   "input_key": "jobs/uploads/<job_id>/input.mp4",
#   "client_name": "John Doe",
#   "languages": ["th", "en"],           # optional, default ["th","en"]
#   "output_prefix": "jobs/output/<job_id>/report",  # optional
#   "analysis_mode": "real",             # optional: "real" or "fallback"
#   "sample_fps": 5,                     # optional
#   "max_frames": 300                    # optional
# }

import os
import io
import json
import time
import logging
import tempfile
import re
import subprocess
import smtplib
import ssl
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Iterable, List, Tuple
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from html import escape

import boto3
from botocore.config import Config

# ------------------------------------------------------------
# IMPORTANT:
#   report_worker MUST import report generation logic from report_core.py
#   (do NOT import app.py, because Streamlit UI runs on import)
# ------------------------------------------------------------
try:
    from report_core import (
        ReportData,
        CategoryResult,
        FirstImpressionData,
        format_seconds_to_mmss,
        get_video_duration_seconds,
        analyze_video_mediapipe,
        analyze_video_placeholder,
        analyze_first_impression_from_video,
        generate_effort_graph,
        generate_shape_graph,
        build_docx_report,
        build_pdf_report,
        mp,  # mediapipe module or None
    )
except Exception as e:
    raise RuntimeError(
        "Cannot import report_core.py. Create report_core.py and move report logic there.\n"
        f"Import error: {e}"
    )

# -----------------------------------------
# Config & logger
# -----------------------------------------
AWS_BUCKET = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
SES_REGION = os.getenv("SES_REGION", AWS_REGION)
SES_FROM_EMAIL = os.getenv("SES_FROM_EMAIL", "").strip()
SES_CONFIGURATION_SET = os.getenv("SES_CONFIGURATION_SET", "").strip()
SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", "").strip()
SMTP_USE_TLS = str(os.getenv("SMTP_USE_TLS", "true")).strip().lower() in ("1", "true", "yes", "on")
SMTP_USE_SSL = str(os.getenv("SMTP_USE_SSL", "false")).strip().lower() in ("1", "true", "yes", "on")
EMAIL_LINK_EXPIRES_SECONDS = int(os.getenv("EMAIL_LINK_EXPIRES_SECONDS", "604800"))  # up to 7 days
MAX_DOCX_ATTACHMENT_BYTES = int(os.getenv("MAX_DOCX_ATTACHMENT_BYTES", "4194304"))  # 4MB per docx
ENABLE_EMAIL_NOTIFICATIONS = str(os.getenv("ENABLE_EMAIL_NOTIFICATIONS", "true")).strip().lower() in ("1", "true", "yes", "on")
POLL_INTERVAL = int(os.getenv("JOB_POLL_INTERVAL", "10"))
MAX_EMAIL_RETRY_ATTEMPTS = int(os.getenv("MAX_EMAIL_RETRY_ATTEMPTS", "10"))
EMAIL_SUSPEND_PREFIX = str(os.getenv("EMAIL_SUSPEND_PREFIX", "Backup/suspend")).strip().strip("/")
MAX_EMAIL_PENDING_JOB_AGE_HOURS = int(os.getenv("MAX_EMAIL_PENDING_JOB_AGE_HOURS", "24"))
OPERATION_TEST_EMAIL_PENDING_CLEANUP_HOURS = int(
    os.getenv("OPERATION_TEST_EMAIL_PENDING_CLEANUP_HOURS", "24")
)
EMAIL_QUEUE_MAX_ITEMS_WHEN_BUSY = int(os.getenv("EMAIL_QUEUE_MAX_ITEMS_WHEN_BUSY", "30"))
EMAIL_QUEUE_MAX_ITEMS_WHEN_IDLE = int(os.getenv("EMAIL_QUEUE_MAX_ITEMS_WHEN_IDLE", "30"))
EMAIL_PENDING_MAX_ITEMS_PER_ROUND = int(os.getenv("EMAIL_PENDING_MAX_ITEMS_PER_ROUND", "30"))
EMAIL_RETRY_BACKOFF_SECONDS = int(os.getenv("EMAIL_RETRY_BACKOFF_SECONDS", "1200"))  # 20 minutes
FORCED_NOTIFY_EMAILS = str(
    os.getenv(
        "FORCED_NOTIFY_EMAILS",
        "alisa@imagematters.at,rungnapa@imagematters.at",
    )
).strip()

# Defaults (can be overridden per-job)
DEFAULT_ANALYSIS_MODE = os.getenv("ANALYSIS_MODE", "real").strip().lower()  # "real" or "fallback"
DEFAULT_SAMPLE_FPS = float(os.getenv("SAMPLE_FPS", "5"))
DEFAULT_MAX_FRAMES = int(os.getenv("MAX_FRAMES", "300"))
DEFAULT_POSE_MODEL_COMPLEXITY = int(os.getenv("POSE_MODEL_COMPLEXITY", "1"))
DEFAULT_POSE_MIN_DET = float(os.getenv("POSE_MIN_DET", "0.5"))
DEFAULT_POSE_MIN_TRACK = float(os.getenv("POSE_MIN_TRACK", "0.5"))
DEFAULT_FACE_MIN_DET = float(os.getenv("FACE_MIN_DET", "0.5"))
DEFAULT_FACEMESH_MIN_DET = float(os.getenv("FACEMESH_MIN_DET", "0.5"))
DEFAULT_FACEMESH_MIN_TRACK = float(os.getenv("FACEMESH_MIN_TRACK", "0.5"))

JOBS_PREFIX = "jobs"
PENDING_PREFIX = f"{JOBS_PREFIX}/pending"
PROCESSING_PREFIX = f"{JOBS_PREFIX}/processing"
FINISHED_PREFIX = f"{JOBS_PREFIX}/finished"
FAILED_PREFIX = f"{JOBS_PREFIX}/failed"
OUTPUT_PREFIX = f"{JOBS_PREFIX}/output"
EMAIL_PENDING_PREFIX = f"{JOBS_PREFIX}/email_pending"

if not AWS_BUCKET:
    raise RuntimeError("Missing AWS_BUCKET (or S3_BUCKET) environment variable")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
)
logger = logging.getLogger("report_worker")

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    config=Config(signature_version="s3v4"),
)
ses = boto3.client("ses", region_name=SES_REGION)

def log_ses_runtime_context() -> None:
    """Log actual runtime sender context to avoid env/account confusion."""
    try:
        sts = boto3.client("sts")
        account_id = sts.get_caller_identity().get("Account", "unknown")
    except Exception as e:
        account_id = f"unknown ({e})"

    prod_enabled = "unknown"
    sending_enabled = "unknown"
    try:
        sesv2 = boto3.client("sesv2", region_name=SES_REGION)
        acc = sesv2.get_account()
        prod_enabled = str(acc.get("ProductionAccessEnabled"))
        sending_enabled = str(acc.get("SendingEnabled"))
    except Exception as e:
        prod_enabled = f"unknown ({e})"

    logger.info(
        "[email_context] aws_account=%s ses_region=%s aws_region=%s ses_from=%s production=%s sending=%s",
        account_id,
        SES_REGION,
        AWS_REGION,
        SES_FROM_EMAIL or "(empty)",
        prod_enabled,
        sending_enabled,
    )

def smtp_config_status() -> Dict[str, Any]:
    return {
        "host": bool(SMTP_HOST),
        "port": SMTP_PORT,
        "username": bool(SMTP_USERNAME),
        "password": bool(SMTP_PASSWORD),
        "from_email": bool(SMTP_FROM_EMAIL or SES_FROM_EMAIL),
        "use_tls": SMTP_USE_TLS,
        "use_ssl": SMTP_USE_SSL,
    }

def log_smtp_runtime_context() -> None:
    s = smtp_config_status()
    logger.info(
        "[smtp_context] host=%s port=%s username=%s password=%s from_email=%s tls=%s ssl=%s configured=%s",
        s["host"],
        s["port"],
        s["username"],
        s["password"],
        s["from_email"],
        s["use_tls"],
        s["use_ssl"],
        bool(s["host"] and s["username"] and s["password"] and s["from_email"]),
    )


# -----------------------------------------
# Small S3 helpers
# -----------------------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_job_id_datetime_utc(job_id: str) -> Optional[datetime]:
    text = str(job_id or "").strip()
    m = re.match(r"^(\d{8})_(\d{6})", text)
    if not m:
        return None
    ts = f"{m.group(1)}{m.group(2)}"
    try:
        return datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def parse_iso_datetime_utc(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def s3_get_json(key: str, log_key: bool = True) -> Dict[str, Any]:
    if log_key:
        logger.info("[s3_get_json] key=%s", key)
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    data = obj["Body"].read()
    return json.loads(data.decode("utf-8"))


def s3_put_json(key: str, payload: Dict[str, Any]) -> None:
    body_str = json.dumps(payload, ensure_ascii=False)
    logger.info("[s3_put_json] key=%s size=%d bytes", key, len(body_str))
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=body_str.encode("utf-8"),
        ContentType="application/json",
    )


def download_to_temp(key: str, suffix: str = ".mp4") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    logger.info("[s3_download] %s -> %s", key, path)
    with open(path, "wb") as f:
        s3.download_fileobj(AWS_BUCKET, key, f)
    return path


def _normalized_rotation_degrees(value: Any) -> int:
    try:
        deg = int(round(float(value)))
    except Exception:
        return 0
    deg = deg % 360
    if deg in (90, 180, 270):
        return deg
    return 0


def get_video_rotation_degrees(video_path: str) -> int:
    """Read rotation metadata using ffprobe. Returns one of 0/90/180/270."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream_tags=rotate:stream_side_data=rotation",
        "-of",
        "json",
        video_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as e:
        logger.warning("[orientation] ffprobe unavailable path=%s err=%s", video_path, e)
        return 0
    if proc.returncode != 0:
        logger.warning("[orientation] ffprobe failed path=%s rc=%s", video_path, proc.returncode)
        return 0
    try:
        payload = json.loads(proc.stdout or "{}")
    except Exception:
        return 0
    streams = payload.get("streams") or []
    if not streams:
        return 0
    stream0 = streams[0] or {}
    side_data = stream0.get("side_data_list") or []
    for item in side_data:
        deg = _normalized_rotation_degrees((item or {}).get("rotation"))
        if deg:
            return deg
    tags = stream0.get("tags") or {}
    return _normalized_rotation_degrees(tags.get("rotate"))


def auto_fix_video_orientation(video_path: str) -> str:
    """
    Normalize orientation before analysis when rotation metadata is present.
    Returns the new upright path, or the original path if no change/failure.
    """
    rotation = get_video_rotation_degrees(video_path)
    if rotation == 0:
        return video_path

    vf = {
        90: "transpose=1",
        180: "transpose=1,transpose=1",
        270: "transpose=2",
    }.get(rotation, "")
    if not vf:
        return video_path

    fd, out_path = tempfile.mkstemp(suffix="_upright.mp4")
    os.close(fd)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vf",
        vf,
        "-metadata:s:v:0",
        "rotate=0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        out_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            logger.warning(
                "[orientation] ffmpeg rotate failed rc=%s rotate=%s path=%s",
                proc.returncode,
                rotation,
                video_path,
            )
            try:
                os.remove(out_path)
            except Exception:
                pass
            return video_path
        logger.info("[orientation] normalized rotate=%s source=%s output=%s", rotation, video_path, out_path)
        return out_path
    except Exception as e:
        logger.warning("[orientation] ffmpeg unavailable path=%s err=%s", video_path, e)
        try:
            os.remove(out_path)
        except Exception:
            pass
        return video_path


def upload_bytes(key: str, data: bytes, content_type: str) -> None:
    logger.info("[s3_upload_bytes] key=%s size=%d", key, len(data))
    s3.put_object(Bucket=AWS_BUCKET, Key=key, Body=data, ContentType=content_type)


def upload_file(path: str, key: str, content_type: str) -> None:
    logger.info("[s3_upload_file] %s -> %s", path, key)
    with open(path, "rb") as f:
        s3.upload_fileobj(f, AWS_BUCKET, key, ExtraArgs={"ContentType": content_type})

def s3_key_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=AWS_BUCKET, Key=key)
        return True
    except Exception:
        return False

def guess_content_type(filename: str) -> str:
    fn = (filename or "").lower()
    if fn.endswith(".mp4"):
        return "video/mp4"
    if fn.endswith(".pdf"):
        return "application/pdf"
    if fn.endswith(".docx"):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return "application/octet-stream"


def presigned_get_url(key: str, expires: int = EMAIL_LINK_EXPIRES_SECONDS, filename: str = "") -> str:
    params: Dict[str, Any] = {"Bucket": AWS_BUCKET, "Key": key}
    if filename:
        params["ResponseContentType"] = guess_content_type(filename)
        # For MP4 links in email, forcing attachment is more reliable across clients.
        if str(filename).lower().endswith(".mp4"):
            params["ResponseContentDisposition"] = f'attachment; filename="{filename}"'
        else:
            params["ResponseContentDisposition"] = f'inline; filename="{filename}"'
    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params=params,
        ExpiresIn=expires,
    )

def is_valid_email(value: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", (value or "").strip()))


def parse_email_list(value: str) -> List[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    tokens = re.split(r"[,\s;]+", raw)
    out: List[str] = []
    seen = set()
    for token in tokens:
        email = str(token or "").strip()
        if not email:
            continue
        key = email.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(email)
    return out


def is_operation_test_style(report_style: str) -> bool:
    return str(report_style or "").strip().lower().startswith("operation_test")


def resolve_notification_recipients(notify_email: str, report_style: str = "") -> List[str]:
    if is_operation_test_style(report_style):
        allowed = parse_email_list(FORCED_NOTIFY_EMAILS)
        selected = parse_email_list(notify_email)
        if allowed:
            allowed_map = {e.lower(): e for e in allowed}
            merged = [allowed_map[s.lower()] for s in selected if s.lower() in allowed_map]
            if not merged:
                merged = allowed
        else:
            merged = selected
    else:
        merged = parse_email_list(notify_email)
    out: List[str] = []
    seen = set()
    for email in merged:
        key = email.lower()
        if key in seen:
            continue
        if not is_valid_email(email):
            continue
        seen.add(key)
        out.append(email)
    return out

def s3_read_bytes(key: str) -> bytes:
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    return obj["Body"].read()

def safe_s3_segment(value: str, fallback: str = "unknown") -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return fallback
    out = []
    for ch in raw:
        if ch.isalnum() or ch in (".", "_", "-"):
            out.append(ch)
        elif ch in ("@",):
            out.append("_at_")
        elif ch in (" ", "/", "\\", ":"):
            out.append("_")
    normalized = "".join(out).strip("._-")
    return normalized or fallback

def s3_copy_if_exists(src_key: str, dest_key: str) -> bool:
    if not src_key:
        return False
    if not s3_key_exists(src_key):
        return False
    filename = os.path.basename(str(dest_key or "").strip())
    content_type = guess_content_type(filename)
    extra_args: Dict[str, Any] = {
        "MetadataDirective": "REPLACE",
        "ContentType": content_type,
    }
    # Force browser-friendly behavior when opening MP4 directly from S3 console.
    if str(filename).lower().endswith(".mp4"):
        extra_args["ContentDisposition"] = f'inline; filename="{filename}"'
    s3.copy_object(
        Bucket=AWS_BUCKET,
        CopySource={"Bucket": AWS_BUCKET, "Key": src_key},
        Key=dest_key,
        **extra_args,
    )
    return True

def sync_enterprise_package(
    *,
    group_id: str,
    enterprise_folder: str,
    notify_email: str,
    dots_key: str,
    skeleton_key: str,
    report_en_key: str,
    report_th_key: str,
) -> Dict[str, Any]:
    """
    Keep a single S3 folder for enterprise handoff:
      jobs/customer_packages/<organization>/<user_email>/<group_id>/
    It contains dots/skeleton/videos + EN/TH report + manifest.json
    """
    gid = str(group_id or "").strip()
    enterprise = str(enterprise_folder or "").strip()
    email = str(notify_email or "").strip()
    if not gid:
        return {}

    customer_segment = safe_s3_segment(enterprise, fallback="unassigned_customer")
    user_segment = safe_s3_segment(email, fallback="unknown_user")
    package_prefix = f"{JOBS_PREFIX}/customer_packages/{customer_segment}/{user_segment}/{gid}"

    report_en_ext = ".pdf" if str(report_en_key or "").lower().endswith(".pdf") else ".docx"
    report_th_ext = ".pdf" if str(report_th_key or "").lower().endswith(".pdf") else ".docx"
    targets = {
        "dots_video": {"source": dots_key, "dest": f"{package_prefix}/dots.mp4"},
        "skeleton_video": {"source": skeleton_key, "dest": f"{package_prefix}/skeleton.mp4"},
        "report_en": {"source": report_en_key, "dest": f"{package_prefix}/report_en{report_en_ext}"},
        "report_th": {"source": report_th_key, "dest": f"{package_prefix}/report_th{report_th_ext}"},
    }

    copied: Dict[str, Any] = {}
    for label, item in targets.items():
        ok = s3_copy_if_exists(item["source"], item["dest"])
        copied[label] = {
            "source": item["source"],
            "dest": item["dest"],
            "ready": bool(ok),
        }

    manifest = {
        "group_id": gid,
        "enterprise_folder": enterprise,
        "notify_email": email,
        "customer_segment": customer_segment,
        "user_segment": user_segment,
        "package_prefix": package_prefix,
        "updated_at": utc_now_iso(),
        "files": copied,
    }
    manifest_key = f"{package_prefix}/manifest.json"
    s3_put_json(manifest_key, manifest)
    manifest["manifest_key"] = manifest_key
    return manifest

def _docx_filename_from_key(key: str, fallback: str) -> str:
    base = os.path.basename(str(key or "").strip())
    return base if base.lower().endswith(".docx") else fallback

def smtp_is_configured() -> bool:
    s = smtp_config_status()
    return bool(s["host"] and s["username"] and s["password"] and s["from_email"])

def send_with_smtp_fallback(msg: MIMEMultipart, to_email: str, group_id: str) -> Tuple[bool, str]:
    if not smtp_is_configured():
        return False, "smtp_not_configured"

    from_email = SMTP_FROM_EMAIL or SES_FROM_EMAIL
    try:
        if "From" in msg:
            msg.replace_header("From", from_email)
        else:
            msg["From"] = from_email
        if "To" in msg:
            msg.replace_header("To", to_email)
        else:
            msg["To"] = to_email

        if SMTP_USE_SSL:
            server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30)
        else:
            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30)
            server.ehlo()
            if SMTP_USE_TLS:
                server.starttls(context=ssl.create_default_context())
                server.ehlo()

        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(from_email, [to_email], msg.as_bytes())
        server.quit()
        logger.info("[email] sent via SMTP fallback to=%s group_id=%s", to_email, group_id)
        return True, "sent_via_smtp_fallback"
    except Exception as e:
        logger.exception("[email] SMTP fallback failed to=%s group_id=%s err=%s", to_email, group_id, e)
        return False, f"smtp_send_failed: {e}"

def send_result_email(
    job: Dict[str, Any],
    video_keys: Dict[str, str],
    report_docx_keys: Dict[str, str],
) -> Tuple[bool, str]:
    recipients = resolve_notification_recipients(
        str(job.get("notify_email") or ""),
        str(job.get("report_style") or ""),
    )
    if not recipients:
        logger.info("[email] skip: no valid recipients")
        return False, "invalid_or_missing_notify_email"
    if not SES_FROM_EMAIL and not smtp_is_configured():
        logger.warning("[email] skip: SES_FROM_EMAIL and SMTP fallback are not configured")
        return False, "missing_ses_from_email_and_smtp"

    job_id = str(job.get("job_id") or "").strip()
    group_id = str(job.get("group_id") or "").strip()
    subject = f"AI People Reader - Results Ready ({group_id or job_id})"
    has_video_links = any(bool(v) for v in video_keys.values())
    report_labels = [str(label) for label, key in report_docx_keys.items() if key]
    has_report_links = bool(report_labels)
    lines = [
        f"Job ID: {job_id}",
        f"Your analysis results are ready for group: {group_id}",
        "",
    ]
    if has_report_links:
        attached_lines = [f"- {label}" for label in report_labels]
        lines.extend(
            [
                "Attached files:",
                *attached_lines,
                "",
            ]
        )
    if has_video_links:
        lines.append("Video links:")
    def _video_filename_from_label(label: str) -> str:
        text = str(label or "").strip().lower()
        if "uploaded video" in text or "input" in text:
            return "input.mp4"
        if "dots" in text:
            return "dots.mp4"
        if "skeleton" in text:
            return "skeleton.mp4"
        return "video.mp4"

    for label, key in video_keys.items():
        if key:
            vname = _video_filename_from_label(label)
            lines.append(f"- {label}: {presigned_get_url(key, filename=vname)}")

    # Fallback links for report DOCX in case attachment cannot be included.
    report_link_lines: List[str] = []
    for label, key in report_docx_keys.items():
        if key:
            if str(key).lower().endswith(".pdf"):
                rname = "report_en.pdf" if "EN" in label else "report_th.pdf"
            else:
                rname = "report_en.docx" if "EN" in label else "report_th.docx"
            report_link_lines.append(f"- {label}: {presigned_get_url(key, filename=rname)}")

    if has_report_links:
        lines.extend([
            "",
            "Backup report links (if your email client blocks attachments):",
        ])
        lines.extend(report_link_lines if report_link_lines else ["- Not available"])
    lines.extend([
        "",
        "Links expire in 7 days.",
        "If a link expires, open Submit Job page and refresh by group_id.",
    ])
    body_text = "\n".join(lines)

    # Add HTML body to avoid URL wrapping issues in email clients.
    html_video_rows = []
    for label, key in video_keys.items():
        if key:
            vname = _video_filename_from_label(label)
            url = presigned_get_url(key, filename=vname)
            html_video_rows.append(f"<li><a href=\"{escape(url)}\">{escape(label)}</a></li>")
    html_report_rows = []
    for label, key in report_docx_keys.items():
        if key:
            if str(key).lower().endswith(".pdf"):
                rname = "report_en.pdf" if "EN" in label else "report_th.pdf"
            else:
                rname = "report_en.docx" if "EN" in label else "report_th.docx"
            url = presigned_get_url(key, filename=rname)
            html_report_rows.append(f"<li><a href=\"{escape(url)}\">{escape(label)}</a></li>")

    attached_html = ""
    if has_report_links:
        attached_html = "<p>Attached files:<br/>" + "<br/>".join([f"- {escape(label)}" for label in report_labels]) + "</p>"
    videos_html = f"<p><b>Video links</b></p><ul>{''.join(html_video_rows) if html_video_rows else '<li>Not available</li>'}</ul>" if has_video_links else ""
    reports_html = f"<p><b>Backup report links</b></p><ul>{''.join(html_report_rows) if html_report_rows else '<li>Not available</li>'}</ul>" if has_report_links else ""
    body_html = f"""
<html>
  <body>
    <p><b>Job ID:</b> {escape(job_id)}<br/>
       <b>Group ID:</b> {escape(group_id)}</p>
    {attached_html}
    {videos_html}
    {reports_html}
    <p>Links expire in 7 days.</p>
  </body>
</html>
"""
    def build_message_for_recipient(to_email: str) -> MIMEMultipart:
        msg = MIMEMultipart("mixed")
        msg["Subject"] = subject
        msg["From"] = SES_FROM_EMAIL or SMTP_FROM_EMAIL
        msg["To"] = to_email
        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(body_text, "plain", "utf-8"))
        alt.attach(MIMEText(body_html, "html", "utf-8"))
        msg.attach(alt)
        # Attach report files (DOCX/PDF) if available and not too large.
        for label, key in report_docx_keys.items():
            if not key:
                continue
            try:
                file_bytes = s3_read_bytes(key)
                if len(file_bytes) > MAX_DOCX_ATTACHMENT_BYTES:
                    logger.warning("[email] skip attachment too large label=%s key=%s size=%d", label, key, len(file_bytes))
                    continue
                is_pdf = str(key).lower().endswith(".pdf")
                fallback_name = "report_en.pdf" if (is_pdf and "EN" in label) else "report_th.pdf" if is_pdf else "report_en.docx" if "EN" in label else "report_th.docx"
                filename = os.path.basename(str(key or "").strip()) or fallback_name
                subtype = "pdf" if is_pdf else "vnd.openxmlformats-officedocument.wordprocessingml.document"
                part = MIMEApplication(file_bytes, _subtype=subtype)
                part.add_header("Content-Disposition", "attachment", filename=filename)
                msg.attach(part)
            except Exception as e:
                logger.warning("[email] cannot attach report label=%s key=%s err=%s", label, key, e)
        return msg

    statuses: List[str] = []
    sent_count = 0
    for to_email in recipients:
        msg = build_message_for_recipient(to_email)
        try:
            if not SES_FROM_EMAIL:
                raise RuntimeError("SES_FROM_EMAIL is empty")

            raw_params: Dict[str, Any] = {
                "Source": SES_FROM_EMAIL,
                "Destinations": [to_email],
                "RawMessage": {"Data": msg.as_bytes()},
            }
            if SES_CONFIGURATION_SET:
                raw_params["ConfigurationSetName"] = SES_CONFIGURATION_SET
            ses.send_raw_email(**raw_params)
            sent_count += 1
            statuses.append(f"{to_email}:sent")
            logger.info("[email] sent to=%s group_id=%s", to_email, group_id)
            continue
        except Exception as e:
            # Fallback: if Raw email is denied, send plain email without attachments.
            err_str = str(e)
            if "SendRawEmail" in err_str and "AccessDenied" in err_str:
                try:
                    fallback_params: Dict[str, Any] = {
                        "Source": SES_FROM_EMAIL,
                        "Destination": {"ToAddresses": [to_email]},
                        "Message": {
                            "Subject": {"Data": subject, "Charset": "UTF-8"},
                            "Body": {
                                "Text": {
                                    "Data": (
                                        body_text
                                        + "\n\nNote: Attachment could not be included due to SES permission. "
                                          "Please use backup links above for report files."
                                    ),
                                    "Charset": "UTF-8",
                                }
                            },
                        },
                    }
                    if SES_CONFIGURATION_SET:
                        fallback_params["ConfigurationSetName"] = SES_CONFIGURATION_SET
                    ses.send_email(**fallback_params)
                    sent_count += 1
                    statuses.append(f"{to_email}:sent_fallback_plain_email_no_attachment")
                    logger.warning(
                        "[email] raw denied, sent fallback plain email to=%s group_id=%s",
                        to_email,
                        group_id,
                    )
                    continue
                except Exception as e2:
                    logger.exception(
                        "[email] fallback send_email failed to=%s group_id=%s err=%s",
                        to_email,
                        group_id,
                        e2,
                    )
                    statuses.append(f"{to_email}:send_failed_raw_and_fallback")
                    continue

            smtp_sent, smtp_status = send_with_smtp_fallback(msg, to_email, group_id)
            if smtp_sent:
                sent_count += 1
                statuses.append(f"{to_email}:{smtp_status}")
                continue

            logger.exception("[email] send failed to=%s group_id=%s err=%s", to_email, group_id, e)
            statuses.append(f"{to_email}:send_failed")

    # Consider email delivery successful when at least one recipient is delivered.
    # This avoids retry loops/suspension when one recipient is temporarily rejected.
    any_sent = sent_count > 0
    return any_sent, " | ".join(statuses) if statuses else "no_recipient_processed"

def build_email_payload(job: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
    group_id = str(job.get("group_id") or "").strip()
    job_id = str(job.get("job_id") or "").strip()
    en_report = outputs.get("reports", {}).get("EN", {}) or {}
    th_report = outputs.get("reports", {}).get("TH", {}) or {}
    en_key = en_report.get("docx_key") or en_report.get("pdf_key") or ""
    th_key = th_report.get("docx_key") or th_report.get("pdf_key") or ""
    output_prefix = str(job.get("output_prefix") or f"{OUTPUT_PREFIX}/{job_id}").strip().rstrip("/")
    analysis_date = str(job.get("analysis_date") or datetime.now().strftime("%Y-%m-%d")).strip()
    report_format = str(job.get("report_format") or "docx").strip().lower()
    ext = "pdf" if report_format == "pdf" else "docx"
    raw_languages = job.get("languages") or ["th", "en"]
    if isinstance(raw_languages, str):
        raw_languages = [raw_languages]
    langs = [str(x).strip().lower() for x in raw_languages if str(x).strip()]
    wants_th = any(l.startswith("th") for l in langs)
    wants_en = any(l.startswith("en") for l in langs)
    if not langs:
        wants_th, wants_en = True, True
    report_style = str(job.get("report_style") or "").strip().lower()
    if is_operation_test_style(report_style):
        wants_th = True
        wants_en = False
        en_key = ""
    if output_prefix and wants_th and not th_key:
        th_key = f"{output_prefix}/Presentation_Analysis_Report_{analysis_date}_TH.{ext}"
    if output_prefix and wants_en and not en_key:
        en_key = f"{output_prefix}/Presentation_Analysis_Report_{analysis_date}_EN.{ext}"
    return {
        "job_id": job_id,
        "group_id": group_id,
        "enterprise_folder": str(job.get("enterprise_folder") or "").strip(),
        "notify_email": str(job.get("notify_email") or "").strip(),
        "report_style": report_style,
        "input_video_key": str(job.get("input_key") or "").strip(),
        "expect_dots": bool(job.get("expect_dots", True)),
        "dots_key": f"jobs/output/groups/{group_id}/dots.mp4" if group_id else "",
        "skeleton_key": f"jobs/output/groups/{group_id}/skeleton.mp4" if group_id else "",
        "report_en_key": en_key,
        "report_th_key": th_key,
        "report_th_email_sent": False,
        "report_en_email_sent": is_operation_test_style(report_style),
        "skeleton_email_sent": False,
        "dots_email_sent": False,
        "attempts": 0,
        "updated_at": utc_now_iso(),
    }

def email_payload_all_ready(payload: Dict[str, Any]) -> bool:
    # Dots is optional now; send email when skeleton + both reports are ready.
    required = [
        payload.get("skeleton_key", ""),
        payload.get("report_en_key", ""),
        payload.get("report_th_key", ""),
    ]
    required = [k for k in required if k]
    if not required:
        return False
    return all(s3_key_exists(k) for k in required)

def email_payload_report_ready(payload: Dict[str, Any]) -> bool:
    en_key = str(payload.get("report_en_key") or "").strip()
    th_key = str(payload.get("report_th_key") or "").strip()
    if not en_key or not th_key:
        return False
    return s3_key_exists(en_key) and s3_key_exists(th_key)

def email_payload_report_th_ready(payload: Dict[str, Any]) -> bool:
    th_key = str(payload.get("report_th_key") or "").strip()
    return bool(th_key) and s3_key_exists(th_key)

def email_payload_report_en_ready(payload: Dict[str, Any]) -> bool:
    en_key = str(payload.get("report_en_key") or "").strip()
    return bool(en_key) and s3_key_exists(en_key)

def email_payload_skeleton_ready(payload: Dict[str, Any]) -> bool:
    sk_key = str(payload.get("skeleton_key") or "").strip()
    return bool(sk_key) and s3_key_exists(sk_key)

def email_payload_dots_ready(payload: Dict[str, Any]) -> bool:
    dots_key = str(payload.get("dots_key") or "").strip()
    if not bool(payload.get("expect_dots", True)):
        return True
    return bool(dots_key) and s3_key_exists(dots_key)

def queue_email_pending(payload: Dict[str, Any]) -> str:
    job_id = str(payload.get("job_id") or "").strip()
    if not job_id:
        raise ValueError("Missing job_id for email pending queue")
    key = f"{EMAIL_PENDING_PREFIX}/{job_id}.json"
    s3_put_json(key, payload)
    return key


def move_email_pending_to_suspend(key: str, payload: Dict[str, Any], reason: str) -> str:
    """
    Archive stuck/invalid email-pending payloads for manual audit and remove from active queue.
    """
    base = os.path.basename(str(key or "").strip()) or f"{str(payload.get('job_id') or 'unknown')}.json"
    suspend_key = f"{EMAIL_SUSPEND_PREFIX}/{base}"
    archived_payload = dict(payload or {})
    archived_payload["suspend_reason"] = str(reason or "").strip()
    archived_payload["suspended_at"] = utc_now_iso()
    archived_payload["source_queue_key"] = key
    s3_put_json(suspend_key, archived_payload)
    s3.delete_object(Bucket=AWS_BUCKET, Key=key)
    return suspend_key

def update_finished_job_notification(
    job_id: str,
    sent: bool,
    status: str,
    report_th_sent: Optional[bool] = None,
    report_en_sent: Optional[bool] = None,
    skeleton_sent: Optional[bool] = None,
    dots_sent: Optional[bool] = None,
) -> None:
    if not job_id:
        return
    key = f"{FINISHED_PREFIX}/{job_id}.json"
    if not s3_key_exists(key):
        return
    job = s3_get_json(key, log_key=False)
    job["notification"] = {
        "notify_email": str(job.get("notify_email") or "").strip(),
        "sent": bool(sent),
        "status": status,
        "updated_at": utc_now_iso(),
    }
    if report_th_sent is not None:
        job["notification"]["report_th_sent"] = bool(report_th_sent)
    if report_en_sent is not None:
        job["notification"]["report_en_sent"] = bool(report_en_sent)
    if skeleton_sent is not None:
        job["notification"]["skeleton_sent"] = bool(skeleton_sent)
    if dots_sent is not None:
        job["notification"]["dots_sent"] = bool(dots_sent)
    s3_put_json(key, job)

def process_pending_email_queue(max_items: int = 10) -> None:
    max_items = max(1, min(int(max_items or 1), EMAIL_PENDING_MAX_ITEMS_PER_ROUND))
    scanned = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=EMAIL_PENDING_PREFIX):
        for item in page.get("Contents", []):
            key = item.get("Key", "")
            if not key.endswith(".json"):
                continue
            if scanned >= max_items:
                return
            scanned += 1

            try:
                payload = s3_get_json(key, log_key=False)
            except Exception:
                continue

            job_id = str(payload.get("job_id") or "").strip()
            notify_email = str(payload.get("notify_email") or "").strip()
            report_style = str(payload.get("report_style") or "").strip().lower()
            is_operation_test = is_operation_test_style(report_style)
            report_th_sent = bool(payload.get("report_th_email_sent"))
            report_en_sent = bool(payload.get("report_en_email_sent"))
            skeleton_sent = bool(payload.get("skeleton_email_sent"))
            dots_sent = bool(payload.get("dots_email_sent"))
            if is_operation_test:
                # Operation Test sends TH report only.
                payload["report_en_key"] = ""
                payload["report_en_email_sent"] = True
                report_en_sent = True
            expect_report_en = bool(str(payload.get("report_en_key") or "").strip())
            attempts = int(payload.get("attempts") or 0)
            if EMAIL_RETRY_BACKOFF_SECONDS > 0 and attempts > 0:
                updated_at = parse_iso_datetime_utc(payload.get("updated_at"))
                if updated_at is not None:
                    elapsed = (datetime.now(timezone.utc) - updated_at).total_seconds()
                    if elapsed < float(EMAIL_RETRY_BACKOFF_SECONDS):
                        continue
            job_created_at = parse_job_id_datetime_utc(job_id)
            if (
                is_operation_test
                and OPERATION_TEST_EMAIL_PENDING_CLEANUP_HOURS > 0
                and job_created_at is not None
            ):
                age_hours = (datetime.now(timezone.utc) - job_created_at).total_seconds() / 3600.0
                if age_hours >= float(OPERATION_TEST_EMAIL_PENDING_CLEANUP_HOURS):
                    update_finished_job_notification(
                        job_id,
                        bool(report_th_sent or report_en_sent or skeleton_sent or dots_sent),
                        f"cleaned_operation_test_email_pending_{OPERATION_TEST_EMAIL_PENDING_CLEANUP_HOURS}h",
                        report_th_sent=report_th_sent,
                        report_en_sent=report_en_sent,
                        skeleton_sent=skeleton_sent,
                        dots_sent=dots_sent,
                    )
                    suspended_key = move_email_pending_to_suspend(
                        key,
                        payload,
                        reason=f"operation_test_email_pending_cleanup_{OPERATION_TEST_EMAIL_PENDING_CLEANUP_HOURS}h",
                    )
                    logger.warning(
                        "[email_queue] cleaned stale operation_test payload key=%s suspended_key=%s age_hours=%.1f",
                        key,
                        suspended_key,
                        age_hours,
                    )
                    continue
            if (not is_operation_test) and MAX_EMAIL_PENDING_JOB_AGE_HOURS > 0 and job_created_at is not None:
                age_hours = (datetime.now(timezone.utc) - job_created_at).total_seconds() / 3600.0
                if age_hours >= float(MAX_EMAIL_PENDING_JOB_AGE_HOURS):
                    update_finished_job_notification(
                        job_id,
                        bool(report_th_sent or report_en_sent or skeleton_sent or dots_sent),
                        f"stopped_stale_email_job_{MAX_EMAIL_PENDING_JOB_AGE_HOURS}h",
                        report_th_sent=report_th_sent,
                        report_en_sent=report_en_sent,
                        skeleton_sent=skeleton_sent,
                        dots_sent=dots_sent,
                    )
                    suspended_key = move_email_pending_to_suspend(
                        key,
                        payload,
                        reason=f"stale_email_job_age_{MAX_EMAIL_PENDING_JOB_AGE_HOURS}h",
                    )
                    logger.warning("[email_queue] moved stale payload key=%s suspended_key=%s age_hours=%.1f", key, suspended_key, age_hours)
                    continue
            if (not is_operation_test) and attempts >= MAX_EMAIL_RETRY_ATTEMPTS:
                update_finished_job_notification(
                    job_id,
                    bool(report_th_sent or report_en_sent or skeleton_sent or dots_sent),
                    f"stopped_max_email_retries_{MAX_EMAIL_RETRY_ATTEMPTS}",
                    report_th_sent=report_th_sent,
                    report_en_sent=report_en_sent,
                    skeleton_sent=skeleton_sent,
                    dots_sent=dots_sent,
                )
                suspended_key = move_email_pending_to_suspend(
                    key,
                    payload,
                    reason=f"max_email_retries_{MAX_EMAIL_RETRY_ATTEMPTS}",
                )
                logger.warning("[email_queue] moved to suspend key=%s suspended_key=%s", key, suspended_key)
                continue
            if is_operation_test:
                recipients = resolve_notification_recipients(notify_email, report_style)
                if not recipients:
                    update_finished_job_notification(
                        job_id,
                        False,
                        "skipped_no_valid_recipients",
                        report_th_sent=report_th_sent,
                        report_en_sent=report_en_sent,
                        skeleton_sent=skeleton_sent,
                        dots_sent=dots_sent,
                    )
                    payload["attempts"] = int(payload.get("attempts") or 0) + 1
                    payload["updated_at"] = utc_now_iso()
                    s3_put_json(key, payload)
                    logger.warning("[email_queue] operation_test has no valid recipients yet; keep pending key=%s", key)
                    continue
            else:
                if not notify_email:
                    update_finished_job_notification(
                        job_id,
                        False,
                        "skipped_no_notify_email",
                        report_th_sent=report_th_sent,
                        report_en_sent=report_en_sent,
                        skeleton_sent=skeleton_sent,
                        dots_sent=dots_sent,
                    )
                    suspended_key = move_email_pending_to_suspend(key, payload, reason="missing_notify_email")
                    logger.warning("[email_queue] moved to suspend key=%s suspended_key=%s", key, suspended_key)
                    continue
                if not is_valid_email(notify_email):
                    update_finished_job_notification(
                        job_id,
                        False,
                        "skipped_invalid_notify_email",
                        report_th_sent=report_th_sent,
                        report_en_sent=report_en_sent,
                        skeleton_sent=skeleton_sent,
                        dots_sent=dots_sent,
                    )
                    suspended_key = move_email_pending_to_suspend(key, payload, reason="invalid_notify_email")
                    logger.warning("[email_queue] moved to suspend key=%s suspended_key=%s", key, suspended_key)
                    continue

            statuses: List[str] = []
            sent_any = False

            # Stage 1: TH report first.
            if (not report_th_sent) and email_payload_report_th_ready(payload):
                sent, status = send_result_email(
                    {
                        "job_id": job_id,
                        "group_id": payload.get("group_id", ""),
                        "notify_email": notify_email,
                        "report_style": report_style,
                    },
                    (
                        {"Uploaded video (MP4)": payload.get("input_video_key", "")}
                        if is_operation_test
                        else {}
                    ),
                    {
                        "Report TH": payload.get("report_th_key", ""),
                    },
                )
                statuses.append(f"report_th:{status}")
                if sent:
                    report_th_sent = True
                    payload["report_th_email_sent"] = True
                    sent_any = True

            # Stage 2: EN report later.
            if (not report_en_sent) and email_payload_report_en_ready(payload):
                sent, status = send_result_email(
                    {
                        "job_id": job_id,
                        "group_id": payload.get("group_id", ""),
                        "notify_email": notify_email,
                        "report_style": report_style,
                    },
                    (
                        {"Uploaded video (MP4)": payload.get("input_video_key", "")}
                        if is_operation_test
                        else {}
                    ),
                    {
                        "Report EN": payload.get("report_en_key", ""),
                    },
                )
                statuses.append(f"report_en:{status}")
                if sent:
                    report_en_sent = True
                    payload["report_en_email_sent"] = True
                    sent_any = True

            # Stage 3: Dots later, independent from report readiness.
            if (not dots_sent) and email_payload_dots_ready(payload):
                if bool(payload.get("expect_dots", True)):
                    sent, status = send_result_email(
                        {
                            "job_id": job_id,
                            "group_id": payload.get("group_id", ""),
                            "notify_email": notify_email,
                            "report_style": report_style,
                        },
                        (
                            {
                                **({"Uploaded video (MP4)": payload.get("input_video_key", "")} if is_operation_test else {}),
                                "Dots video (MP4)": payload.get("dots_key", ""),
                            }
                        ),
                        {},
                    )
                    statuses.append(f"dots:{status}")
                    if sent:
                        dots_sent = True
                        payload["dots_email_sent"] = True
                        sent_any = True
                else:
                    dots_sent = True
                    payload["dots_email_sent"] = True

            if not statuses:
                waiting = []
                if not report_th_sent:
                    waiting.append("report_th")
                if expect_report_en and not report_en_sent:
                    waiting.append("report_en")
                if bool(payload.get("expect_dots", True)) and not dots_sent:
                    waiting.append("dots")
                statuses.append("waiting_for_" + "_and_".join(waiting) if waiting else "waiting")

            # Keep the enterprise handoff folder in sync once all outputs are ready.
            try:
                package_info = sync_enterprise_package(
                    group_id=str(payload.get("group_id") or "").strip(),
                    enterprise_folder=str(payload.get("enterprise_folder") or "").strip(),
                    notify_email=notify_email,
                    dots_key=str(payload.get("dots_key") or "").strip(),
                    skeleton_key=str(payload.get("skeleton_key") or "").strip(),
                    report_en_key=str(payload.get("report_en_key") or "").strip(),
                    report_th_key=str(payload.get("report_th_key") or "").strip(),
                )
                if package_info:
                    logger.info("[enterprise_package] synced for job_id=%s prefix=%s", job_id, package_info.get("package_prefix", ""))
            except Exception as e:
                logger.warning("[enterprise_package] sync failed in email queue job_id=%s: %s", job_id, e)

            status_text = " | ".join(statuses)
            overall_sent = bool(report_th_sent or report_en_sent or dots_sent or sent_any)
            update_finished_job_notification(
                job_id,
                overall_sent,
                status_text,
                report_th_sent=report_th_sent,
                report_en_sent=report_en_sent,
                skeleton_sent=skeleton_sent,
                dots_sent=dots_sent,
            )

            report_en_done = report_en_sent or (not expect_report_en)
            # TH report is the primary delivery milestone. Once sent, stop keeping this job in email_pending.
            all_done = report_th_sent
            if all_done:
                s3.delete_object(Bucket=AWS_BUCKET, Key=key)
            else:
                payload["attempts"] = int(payload.get("attempts") or 0) + 1
                payload["updated_at"] = utc_now_iso()
                s3_put_json(key, payload)


def list_pending_json_keys() -> Iterable[str]:
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=PENDING_PREFIX):
        for item in page.get("Contents", []):
            k = item["Key"]
            if k.endswith(".json"):
                yield k


def find_one_pending_job_key() -> Optional[str]:
    picked_key: Optional[str] = None
    picked_priority = -1
    picked_created_at = ""

    for k in list_pending_json_keys():
        try:
            job = s3_get_json(k)
        except Exception as e:
            # Key may be consumed by another worker between list/get.
            logger.info("[find_one_pending_job_key] skip key=%s reason=%s", k, e)
            continue

        mode = str(job.get("mode") or "").strip().lower()
        if mode in ("report", "report_th_en", "report_generator"):
            priority = int(job.get("priority") or 0)
            created_at = str(job.get("created_at") or "")
            if (
                priority > picked_priority
                or (priority == picked_priority and created_at > picked_created_at)
            ):
                picked_key = k
                picked_priority = priority
                picked_created_at = created_at

        logger.info("[find_one_pending_job_key] ignore non-report key=%s mode=%s", k, mode)
    if picked_key:
        logger.info(
            "[find_one_pending_job_key] picked report key=%s priority=%s created_at=%s",
            picked_key,
            picked_priority,
            picked_created_at,
        )
    return picked_key


def move_json(old_key: str, new_key: str, payload: Dict[str, Any]) -> None:
    s3_put_json(new_key, payload)
    if old_key != new_key:
        logger.info("[s3_delete] key=%s", old_key)
        s3.delete_object(Bucket=AWS_BUCKET, Key=old_key)


def update_status(job: Dict[str, Any], status: str, error: Optional[str] = None) -> Dict[str, Any]:
    job["status"] = status
    job["updated_at"] = utc_now_iso()
    if error is not None:
        job["error"] = error
    return job


# -----------------------------------------
# Report generation helpers
# -----------------------------------------
def _t(lang: str, en: str, th: str) -> str:
    return th if (lang or "").strip().lower().startswith("th") else en


def _build_categories_from_result(result: Dict[str, Any], total: int) -> List[CategoryResult]:
    # Keep exactly the same 3 categories as your current app.py
    categories = [
        CategoryResult(
            name_en="Engaging & Connecting",
            name_th="การสร้างความเป็นมิตรและสร้างสัมพันธภาพ",
            score=int(result["engaging_score"]),
            scale=("moderate" if int(result["engaging_score"]) in [3, 4] else ("high" if int(result["engaging_score"]) >= 5 else "low")),
            positives=int(result["engaging_pos"]),
            total=int(total),
        ),
        CategoryResult(
            name_en="Confidence",
            name_th="ความมั่นใจ",
            score=int(result["convince_score"]),
            scale=("moderate" if int(result["convince_score"]) in [3, 4] else ("high" if int(result["convince_score"]) >= 5 else "low")),
            positives=int(result["convince_pos"]),
            total=int(total),
        ),
        CategoryResult(
            name_en="Authority",
            name_th="ความเป็นผู้นำและอำนาจ",
            score=int(result["authority_score"]),
            scale=("moderate" if int(result["authority_score"]) in [3, 4] else ("high" if int(result["authority_score"]) >= 5 else "low")),
            positives=int(result["authority_pos"]),
            total=int(total),
        ),
    ]
    return categories


def run_analysis(video_path: str, job: Dict[str, Any]) -> Dict[str, Any]:
    analysis_mode = str(job.get("analysis_mode") or DEFAULT_ANALYSIS_MODE).strip().lower()
    sample_fps = float(job.get("sample_fps") or DEFAULT_SAMPLE_FPS)
    max_frames = int(job.get("max_frames") or DEFAULT_MAX_FRAMES)

    want_real = analysis_mode.startswith("real")
    if want_real and (mp is not None):
        logger.info("[analysis] Using real mediapipe analysis (sample_fps=%s, max_frames=%s)", sample_fps, max_frames)
        return analyze_video_mediapipe(
            video_path=video_path,
            sample_fps=float(sample_fps),
            max_frames=int(max_frames),
            pose_model_complexity=int(job.get("pose_model_complexity") or DEFAULT_POSE_MODEL_COMPLEXITY),
            pose_min_detection_confidence=float(job.get("pose_min_det") or DEFAULT_POSE_MIN_DET),
            pose_min_tracking_confidence=float(job.get("pose_min_track") or DEFAULT_POSE_MIN_TRACK),
            face_min_detection_confidence=float(job.get("face_min_det") or DEFAULT_FACE_MIN_DET),
            facemesh_min_detection_confidence=float(job.get("facemesh_min_det") or DEFAULT_FACEMESH_MIN_DET),
            facemesh_min_tracking_confidence=float(job.get("facemesh_min_track") or DEFAULT_FACEMESH_MIN_TRACK),
        )

    logger.info("[analysis] Using fallback placeholder analysis")
    return analyze_video_placeholder(video_path=video_path, seed=42)


def _first_impression_level(value: float, metric: str = "") -> str:
    score = float(value or 0.0)
    name = str(metric or "").strip().lower()
    if name == "eye_contact":
        return "high"
    if name in ("stance", "uprightness"):
        if score >= 80.0:
            return "high"
        if score >= 50.0:
            return "moderate"
        return "low"
    if score >= 70.0:
        return "high"
    if score >= 40.0:
        return "moderate"
    return "low"


def generate_reports_for_lang(
    job: Dict[str, Any],
    result: Dict[str, Any],
    video_path: str,
    lang_code: str,
    out_dir: str,
) -> Tuple[bytes, Optional[bytes], Dict[str, str], Dict[str, Any]]:
    """
    Returns:
      docx_bytes, pdf_bytes(or None), uploaded_key_map (local keys suggestion)
    """
    analysis_date = str(job.get("analysis_date") or datetime.now().strftime("%Y-%m-%d")).strip()
    client_name = str(job.get("client_name") or "").strip()

    duration_str = format_seconds_to_mmss(float(result.get("duration_seconds") or get_video_duration_seconds(video_path)))
    total = int(result.get("total_indicators") or 0) or 1

    # Run First Impression analysis
    first_impression = analyze_first_impression_from_video(video_path, sample_every_n=5, max_frames=200)
    
    # Log the actual detected values for debugging
    logger.info("[first_impression] Eye Contact: %.1f%%, Uprightness: %.1f%%, Stance: %.1f%%", 
                first_impression.eye_contact_pct, first_impression.upright_pct, first_impression.stance_stability)

    categories = _build_categories_from_result(result, total=total)
    report = ReportData(
        client_name=client_name,
        analysis_date=analysis_date,
        video_length_str=duration_str,
        overall_score=int(round(float(sum([c.score for c in categories])) / max(1, len(categories)))),
        categories=categories,
        summary_comment=str(job.get("summary_comment") or "").strip(),
        generated_by=_t(lang_code, "Generated by AI People Reader™", "จัดทำโดย AI People Reader™"),
        first_impression=first_impression,
    )

    report_style = str(job.get("report_style") or "").strip().lower()
    enterprise_folder = str(job.get("enterprise_folder") or "").strip().lower()
    if is_operation_test_style(report_style):
        report_style = "operation_test"
        lang_code = "th"
    if not report_style:
        report_style = "full"
    report_format = str(job.get("report_format") or "docx").strip().lower()
    wants_pdf = report_format == "pdf"
    logger.info(
        "[report] render lang=%s style=%s format=%s enterprise_folder=%s",
        lang_code,
        report_style,
        report_format,
        enterprise_folder,
    )
    # Graphs are disabled for all report styles by request.
    graph1_path = ""
    graph2_path = ""

    # DOCX (in-memory)
    docx_bio = io.BytesIO()
    build_docx_report(
        report,
        docx_bio,
        graph1_path=graph1_path,
        graph2_path=graph2_path,
        lang=lang_code,
        report_style=report_style,
    )
    docx_bytes = docx_bio.getvalue()
    if not docx_bytes:
        raise RuntimeError("DOCX generation produced empty output")

    # PDF (file -> bytes) only when requested by job format.
    pdf_bytes = None
    pdf_out_path = ""
    if wants_pdf:
        pdf_out_path = os.path.join(out_dir, f"Presentation_Analysis_Report_{analysis_date}_{lang_code.upper()}.pdf")
        try:
            build_pdf_report(
                report,
                pdf_out_path,
                graph1_path=graph1_path,
                graph2_path=graph2_path,
                lang=lang_code,
                report_style=report_style,
            )
            if os.path.exists(pdf_out_path):
                with open(pdf_out_path, "rb") as f:
                    pdf_bytes = f.read()
        except Exception as e:
            logger.warning("[pdf] PDF build failed for lang=%s: %s", lang_code, e)
            pdf_bytes = None

    key_map = {
        "graph1_path": graph1_path,
        "graph2_path": graph2_path,
        "pdf_out_path": pdf_out_path,
    }
    first_impression_summary = {
        "eye_contact": {
            "value": round(float(first_impression.eye_contact_pct), 1),
            "level": _first_impression_level(first_impression.eye_contact_pct, metric="eye_contact"),
        },
        "uprightness": {
            "value": round(float(first_impression.upright_pct), 1),
            "level": _first_impression_level(first_impression.upright_pct, metric="uprightness"),
        },
        "stance": {
            "value": round(float(first_impression.stance_stability), 1),
            "level": _first_impression_level(first_impression.stance_stability, metric="stance"),
        },
    }
    return docx_bytes, pdf_bytes, key_map, first_impression_summary


def process_report_job(job: Dict[str, Any]) -> Dict[str, Any]:
    job_id = str(job.get("job_id") or "").strip()
    input_key = str(job.get("input_key") or "").strip()
    group_id = str(job.get("group_id") or "").strip()
    if not job_id:
        raise ValueError("Job JSON missing 'job_id'")
    if not input_key:
        raise ValueError("Job JSON missing 'input_key'")

    report_style = str(job.get("report_style") or "").strip().lower()
    enterprise_folder = str(job.get("enterprise_folder") or "").strip().lower()
    if is_operation_test_style(report_style):
        report_style = "operation_test"
        job["report_style"] = "operation_test"

    # languages: default TH only for fastest first delivery.
    languages = job.get("languages") or ["th"]
    if isinstance(languages, str):
        languages = [languages]
    languages = [str(x).strip().lower() for x in languages if str(x).strip()]
    if not languages:
        languages = ["th", "en"]
    report_format = str(job.get("report_format") or "docx").strip().lower()
    if report_format not in ("docx", "pdf"):
        report_format = "docx"

    # Operation Test must always produce Thai PDF-only output.
    if report_style == "operation_test":
        languages = ["th"]
        report_format = "pdf"
        job["languages"] = ["th"]
        job["report_format"] = "pdf"
        logger.info("[report] operation_test override: force languages=%s report_format=%s", languages, report_format)

    # output prefix
    output_prefix = str(job.get("output_prefix") or f"{OUTPUT_PREFIX}/{job_id}").strip().rstrip("/")
    if report_style == "operation_test" and group_id:
        # Keep Operation Test outputs under the same group folder as input for easy S3 lookup.
        output_prefix = f"jobs/groups/{group_id}"
        job["output_prefix"] = output_prefix
    # We'll store files under:
    #   <output_prefix>/report_TH.docx, report_EN.docx, report_TH.pdf, report_EN.pdf
    #   <output_prefix>/Graph_1_TH.png, Graph_2_TH.png, ...
    logger.info("[report] job_id=%s input_key=%s languages=%s output_prefix=%s", job_id, input_key, languages, output_prefix)

    # Download video
    video_suffix = os.path.splitext(input_key)[1] or ".mp4"
    video_path = download_to_temp(input_key, suffix=video_suffix)
    analysis_video_path = auto_fix_video_orientation(video_path)
    if analysis_video_path != video_path:
        job["input_video_normalized"] = True
    else:
        job["input_video_normalized"] = False

    out_dir = tempfile.mkdtemp(prefix=f"report_{job_id}_")
    try:
        # Analyze once (shared for both languages)
        result = run_analysis(analysis_video_path, job)

        outputs: Dict[str, Any] = {"reports": {}, "graphs": {}}

        for lang_code in languages:
            lang_code = "th" if lang_code.startswith("th") else "en"
            docx_bytes, pdf_bytes, local_paths, first_impression_summary = generate_reports_for_lang(
                job, result, analysis_video_path, lang_code, out_dir
            )
            if not isinstance(job.get("first_impression_summary"), dict):
                job["first_impression_summary"] = first_impression_summary

            # Graph uploads are disabled; keep keys null in outputs.
            g1_key = None
            g2_key = None

            # Upload DOCX when requested format is DOCX.
            analysis_date = str(job.get("analysis_date") or datetime.now().strftime("%Y-%m-%d")).strip()
            docx_key = None
            if report_format == "docx":
                docx_name = f"Presentation_Analysis_Report_{analysis_date}_{lang_code.upper()}.docx"
                docx_key = f"{output_prefix}/{docx_name}"
                upload_bytes(
                    docx_key,
                    docx_bytes,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )

            # Upload PDF when requested format is PDF.
            pdf_key = None
            if report_format == "pdf":
                if not pdf_bytes:
                    raise RuntimeError("PDF format requested but PDF generation failed")
                pdf_name = f"Presentation_Analysis_Report_{analysis_date}_{lang_code.upper()}.pdf"
                pdf_key = f"{output_prefix}/{pdf_name}"
                upload_bytes(pdf_key, pdf_bytes, "application/pdf")
                if report_style == "operation_test" and group_id:
                    # Canonical path for Operation Test so UI/S3 always find report in group folder.
                    canonical_pdf_key = f"jobs/groups/{group_id}/{pdf_name}"
                    if canonical_pdf_key != pdf_key:
                        upload_bytes(canonical_pdf_key, pdf_bytes, "application/pdf")
                    pdf_key = canonical_pdf_key

            outputs["graphs"][lang_code.upper()] = {"graph1_key": g1_key, "graph2_key": g2_key}
            outputs["reports"][lang_code.upper()] = {"docx_key": docx_key, "pdf_key": pdf_key}

        # Save structured outputs into job JSON
        job["output_prefix"] = output_prefix
        job["outputs"] = outputs
        job["analysis_engine"] = str(result.get("analysis_engine") or "unknown")
        job["duration_seconds"] = float(result.get("duration_seconds") or 0.0)
        job["analyzed_frames"] = int(result.get("analyzed_frames") or 0)

        # Enterprise package: one folder per (notify_email, group_id) for client handoff.
        group_id = str(job.get("group_id") or "").strip()
        enterprise_folder = str(job.get("enterprise_folder") or "").strip()
        notify_email = str(job.get("notify_email") or "").strip()
        if group_id:
            en_report = outputs.get("reports", {}).get("EN", {}) or {}
            th_report = outputs.get("reports", {}).get("TH", {}) or {}
            dots_key = f"jobs/output/groups/{group_id}/dots.mp4"
            skeleton_key = f"jobs/output/groups/{group_id}/skeleton.mp4"
            report_en_key = str(en_report.get("docx_key") or en_report.get("pdf_key") or "").strip()
            report_th_key = str(th_report.get("docx_key") or th_report.get("pdf_key") or "").strip()
            try:
                package_info = sync_enterprise_package(
                    group_id=group_id,
                    enterprise_folder=enterprise_folder,
                    notify_email=notify_email,
                    dots_key=dots_key,
                    skeleton_key=skeleton_key,
                    report_en_key=report_en_key,
                    report_th_key=report_th_key,
                )
                if package_info:
                    job["enterprise_package"] = package_info
            except Exception as e:
                logger.warning("[enterprise_package] initial sync failed for group_id=%s: %s", group_id, e)

        # Notification flow:
        # - send TH report as soon as ready (primary milestone)
        # - EN report and dots are follow-up deliveries
        if ENABLE_EMAIL_NOTIFICATIONS:
            payload = build_email_payload(job, outputs)
            report_style = str(payload.get("report_style") or "").strip().lower()
            is_operation_test = is_operation_test_style(report_style)
            notify_email_value = str(payload.get("notify_email") or "").strip()
            recipients = resolve_notification_recipients(notify_email_value, report_style)
            if recipients:
                statuses: List[str] = []
                report_th_sent = False
                report_en_sent = False
                skeleton_sent = False
                dots_sent = False
                expect_report_en = bool(str(payload.get("report_en_key") or "").strip())

                if email_payload_report_th_ready(payload):
                    sent, status = send_result_email(
                        {
                            "job_id": payload["job_id"],
                            "group_id": payload.get("group_id", ""),
                            "notify_email": payload["notify_email"],
                            "report_style": report_style,
                        },
                        (
                            {"Uploaded video (MP4)": payload.get("input_video_key", "")}
                            if is_operation_test
                            else {}
                        ),
                        {
                            "Report TH": payload.get("report_th_key", ""),
                        },
                    )
                    statuses.append(f"report_th:{status}")
                    report_th_sent = bool(sent)
                    payload["report_th_email_sent"] = report_th_sent

                if email_payload_report_en_ready(payload):
                    sent, status = send_result_email(
                        {
                            "job_id": payload["job_id"],
                            "group_id": payload.get("group_id", ""),
                            "notify_email": payload["notify_email"],
                            "report_style": report_style,
                        },
                        (
                            {"Uploaded video (MP4)": payload.get("input_video_key", "")}
                            if is_operation_test
                            else {}
                        ),
                        {
                            "Report EN": payload.get("report_en_key", ""),
                        },
                    )
                    statuses.append(f"report_en:{status}")
                    report_en_sent = bool(sent)
                    payload["report_en_email_sent"] = report_en_sent

                if bool(payload.get("expect_dots", True)) and email_payload_dots_ready(payload):
                    sent, status = send_result_email(
                        {
                            "job_id": payload["job_id"],
                            "group_id": payload.get("group_id", ""),
                            "notify_email": payload["notify_email"],
                            "report_style": report_style,
                        },
                        (
                            {
                                **({"Uploaded video (MP4)": payload.get("input_video_key", "")} if is_operation_test else {}),
                                "Dots video (MP4)": payload.get("dots_key", ""),
                            }
                        ),
                        {},
                    )
                    statuses.append(f"dots:{status}")
                    dots_sent = bool(sent)
                    payload["dots_email_sent"] = dots_sent
                elif not bool(payload.get("expect_dots", True)):
                    dots_sent = True
                    payload["dots_email_sent"] = True

                # Primary completion milestone: TH report delivered.
                primary_done = report_th_sent
                if not primary_done:
                    queue_email_pending(payload)
                    waiting = []
                    if not report_th_sent:
                        waiting.append("report_th")
                    statuses.append("waiting_for_" + "_and_".join(waiting) if waiting else "waiting")

                email_sent = bool(primary_done or report_en_sent or dots_sent)
                email_status = " | ".join(statuses) if statuses else "queued"
            else:
                if is_operation_test:
                    email_sent, email_status = False, "skipped_no_valid_recipients"
                elif notify_email_value:
                    email_sent, email_status = False, "skipped_invalid_notify_email"
                else:
                    email_sent, email_status = False, "skipped_no_notify_email"
                report_th_sent, report_en_sent, skeleton_sent, dots_sent = False, False, False, False
        else:
            email_sent, email_status = False, "disabled_by_config"
            report_th_sent, report_en_sent, skeleton_sent, dots_sent = False, False, False, False
        job["notification"] = {
            "notify_email": str(job.get("notify_email") or "").strip(),
            "sent": email_sent,
            "status": email_status,
            "updated_at": utc_now_iso(),
            "report_th_sent": bool(report_th_sent),
            "report_en_sent": bool(report_en_sent),
            "skeleton_sent": bool(skeleton_sent),
            "dots_sent": bool(dots_sent),
        }
        
        # Debug: Log the outputs structure
        logger.info("[report] Saving outputs to job: %s", json.dumps(outputs, indent=2))

        return job

    finally:
        # Cleanup temp files
        try:
            os.remove(video_path)
        except Exception:
            pass
        if analysis_video_path != video_path:
            try:
                os.remove(analysis_video_path)
            except Exception:
                pass


# -----------------------------------------
# Job processor
# -----------------------------------------
def process_job(job_json_key: str) -> None:
    try:
        raw_job = s3_get_json(job_json_key)
    except Exception as e:
        # Job might have been taken by another worker (race condition)
        if "NoSuchKey" in str(e) or "does not exist" in str(e):
            logger.info("[process_job] Job %s already taken by another worker, skipping", job_json_key)
            return
        else:
            raise  # Re-raise if it's a different error
    
    job_id = raw_job.get("job_id")
    mode = str(raw_job.get("mode") or "").strip().lower()

    if not job_id:
        raise ValueError("Job JSON missing 'job_id'")

    logger.info("[process_job] job_id=%s mode=%s key=%s", job_id, mode, job_json_key)

    # Check if this worker should handle this job type
    if mode not in ("report", "report_th_en", "report_generator"):
        logger.info("[process_job] Skipping job_id=%s mode=%s (not a report job)", job_id, mode)
        return  # Leave job in pending for other workers

    # Move to processing
    job = dict(raw_job)
    job = update_status(job, "processing", error=None)
    processing_key = f"{PROCESSING_PREFIX}/{job_id}.json"
    move_json(job_json_key, processing_key, job)

    try:
        # Process report job
        job = process_report_job(job)

        job = update_status(job, "finished", error=None)
        finished_key = f"{FINISHED_PREFIX}/{job_id}.json"
        move_json(processing_key, finished_key, job)
        logger.info("[process_job] job_id=%s finished", job_id)

    except Exception as exc:
        logger.exception("[process_job] job_id=%s FAILED: %s", job_id, exc)
        job = update_status(job, "failed", error=str(exc))
        failed_key = f"{FAILED_PREFIX}/{job_id}.json"
        move_json(processing_key, failed_key, job)


# -----------------------------------------
# Main loop
# -----------------------------------------
def main() -> None:
    logger.info("====== AI People Reader Report Worker (TH/EN) ======")
    logger.info("Using bucket: %s", AWS_BUCKET)
    logger.info("Region       : %s", AWS_REGION)
    logger.info("Poll every   : %s seconds", POLL_INTERVAL)
    log_ses_runtime_context()
    log_smtp_runtime_context()

    while True:
        try:
            # Always drain some email queue first so pending emails are delivered ASAP.
            if ENABLE_EMAIL_NOTIFICATIONS:
                process_pending_email_queue(max_items=EMAIL_QUEUE_MAX_ITEMS_WHEN_BUSY)

            job_key = find_one_pending_job_key()
            if job_key:
                process_job(job_key)
                # Keep draining email queue even when report jobs keep arriving continuously.
                if ENABLE_EMAIL_NOTIFICATIONS:
                    process_pending_email_queue(max_items=EMAIL_QUEUE_MAX_ITEMS_WHEN_BUSY)
            else:
                if ENABLE_EMAIL_NOTIFICATIONS:
                    process_pending_email_queue(max_items=EMAIL_QUEUE_MAX_ITEMS_WHEN_IDLE)
                time.sleep(POLL_INTERVAL)
        except Exception as exc:
            logger.exception("[main] Unexpected error: %s", exc)
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()