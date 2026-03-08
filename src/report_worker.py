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
import sys
import io
import json
import base64
import importlib.util

# Ensure report_core is loaded from src/ (same dir as this script) — critical for operation_test + remark points
_worker_dir = os.path.dirname(os.path.abspath(__file__))
_report_core_path = os.path.join(_worker_dir, "report_core.py")
if not os.path.exists(_report_core_path):
    raise RuntimeError(f"report_core.py not found at {_report_core_path}")
_spec = importlib.util.spec_from_file_location("report_core", _report_core_path)
if _spec is None:
    raise RuntimeError(f"Failed to create spec for report_core at {_report_core_path}")
_report_core = importlib.util.module_from_spec(_spec)
sys.modules["report_core"] = _report_core

from dotenv import load_dotenv
load_dotenv()
import time
import logging
import tempfile
import re
import subprocess
import shutil
from pathlib import Path
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
#   report_worker MUST import report generation logic from report_core.py (src/)
#   (do NOT import app.py, because Streamlit UI runs on import)
# ------------------------------------------------------------
try:
    if _spec.loader is None:
        raise RuntimeError("report_core spec has no loader")
    _spec.loader.exec_module(_report_core)
    ReportData = _report_core.ReportData
    CategoryResult = _report_core.CategoryResult
    FirstImpressionData = _report_core.FirstImpressionData
    format_seconds_to_mmss = _report_core.format_seconds_to_mmss
    get_video_duration_seconds = _report_core.get_video_duration_seconds
    analyze_video_mediapipe = _report_core.analyze_video_mediapipe
    analyze_video_placeholder = _report_core.analyze_video_placeholder
    analyze_first_impression_from_video = _report_core.analyze_first_impression_from_video
    generate_effort_graph = _report_core.generate_effort_graph
    generate_shape_graph = _report_core.generate_shape_graph
    build_docx_report = _report_core.build_docx_report
    build_pdf_report = _report_core.build_pdf_report
    first_impression_level = _report_core.first_impression_level
    mp = _report_core.mp
except Exception as e:
    raise RuntimeError(
        "Cannot import report_core.py from src/. Create src/report_core.py.\n"
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
EMERGENCY_THAI_DOCX_FALLBACK = str(
    os.getenv("EMERGENCY_THAI_DOCX_FALLBACK", "false")
).strip().lower() in ("1", "true", "yes", "on")
THAI_PDF_VIA_DOCX = str(os.getenv("THAI_PDF_VIA_DOCX", "true")).strip().lower() in ("1", "true", "yes", "on")
PDF_VIA_DOCX_FOR_ALL_LANGS = str(os.getenv("PDF_VIA_DOCX_FOR_ALL_LANGS", "true")).strip().lower() in ("1", "true", "yes", "on")
DOCX_TO_PDF_TIMEOUT_SECONDS = int(os.getenv("DOCX_TO_PDF_TIMEOUT_SECONDS", "180"))
USE_DOCX2PDF_FALLBACK = str(os.getenv("USE_DOCX2PDF_FALLBACK", "false")).strip().lower() in ("1", "true", "yes", "on")
PDF_VIA_HTML_FIRST = str(os.getenv("PDF_VIA_HTML_FIRST", "true")).strip().lower() in ("1", "true", "yes", "on")
HTML_PDF_ENGINE = str(os.getenv("HTML_PDF_ENGINE", "auto")).strip().lower()  # auto|chrome|libreoffice
HTML_TO_PDF_TIMEOUT_SECONDS = int(os.getenv("HTML_TO_PDF_TIMEOUT_SECONDS", str(DOCX_TO_PDF_TIMEOUT_SECONDS)))
PDF_HTML_STRICT_FOR_OPERATION_TEST = str(
    os.getenv("PDF_HTML_STRICT_FOR_OPERATION_TEST", "true")
).strip().lower() in ("1", "true", "yes", "on")
DOCX_FALLBACK_ON_PDF_FAIL = str(
    os.getenv("DOCX_FALLBACK_ON_PDF_FAIL", "true")
).strip().lower() in ("1", "true", "yes", "on")
THAI_PDF_IMAGE_CAPTURE = str(os.getenv("THAI_PDF_IMAGE_CAPTURE", "true")).strip().lower() in ("1", "true", "yes", "on")
THAI_PDF_IMAGE_DPI = int(os.getenv("THAI_PDF_IMAGE_DPI", "220"))
THAI_PDF_IMAGE_CAPTURE_STRICT = str(
    os.getenv("THAI_PDF_IMAGE_CAPTURE_STRICT", "true")
).strip().lower() in ("1", "true", "yes", "on")
THAI_CAPTURE_FROM_DOCX_DIRECT = str(
    os.getenv("THAI_CAPTURE_FROM_DOCX_DIRECT", "true")
).strip().lower() in ("1", "true", "yes", "on")
POLL_INTERVAL = int(os.getenv("JOB_POLL_INTERVAL", "3"))
PROCESSING_STALE_MINUTES = int(os.getenv("PROCESSING_STALE_MINUTES", "20"))
PROCESSING_RECOVERY_MAX_ITEMS = int(os.getenv("PROCESSING_RECOVERY_MAX_ITEMS", "50"))
PROCESSING_RECOVERY_INTERVAL_SECONDS = int(os.getenv("PROCESSING_RECOVERY_INTERVAL_SECONDS", "60"))
IDLE_HEARTBEAT_SECONDS = int(os.getenv("IDLE_HEARTBEAT_SECONDS", "60"))
MAX_EMAIL_RETRY_ATTEMPTS = int(os.getenv("MAX_EMAIL_RETRY_ATTEMPTS", "10"))
JOB_MAX_RETRIES = int(os.getenv("JOB_MAX_RETRIES", "3"))
EMAIL_SUSPEND_PREFIX = str(os.getenv("EMAIL_SUSPEND_PREFIX", "Backup/suspend")).strip().strip("/")
MAX_EMAIL_PENDING_JOB_AGE_HOURS = int(os.getenv("MAX_EMAIL_PENDING_JOB_AGE_HOURS", "24"))
PENDING_STALE_HOURS = float(os.getenv("PENDING_STALE_HOURS", "2"))  # Move to failed if input missing for this long
OPERATION_TEST_EMAIL_PENDING_CLEANUP_HOURS = int(
    os.getenv("OPERATION_TEST_EMAIL_PENDING_CLEANUP_HOURS", "24")
)
EMAIL_QUEUE_MAX_ITEMS_WHEN_BUSY = int(os.getenv("EMAIL_QUEUE_MAX_ITEMS_WHEN_BUSY", "30"))
EMAIL_QUEUE_MAX_ITEMS_WHEN_IDLE = int(os.getenv("EMAIL_QUEUE_MAX_ITEMS_WHEN_IDLE", "30"))
EMAIL_QUEUE_MAX_ITEMS_AFTER_JOB = int(os.getenv("EMAIL_QUEUE_MAX_ITEMS_AFTER_JOB", "30"))
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
DEFAULT_SAMPLE_FPS = float(os.getenv("SAMPLE_FPS", "3"))
DEFAULT_MAX_FRAMES = int(os.getenv("MAX_FRAMES", "150"))
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
    """Log actual runtime sender context. Non-blocking: SES permission errors are logged but do not affect worker."""
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
        # IAM may lack ses:GetAccount — this does NOT block report generation.
        logger.warning("[email_context] SES GetAccount skipped (no ses:GetAccount permission): %s", type(e).__name__)

    logger.info(
        "[email_context] aws_account=%s ses_region=%s ses_from=%s production=%s sending=%s",
        account_id,
        SES_REGION,
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


def _find_libreoffice_bin() -> str:
    for candidate in ("libreoffice", "soffice"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    for candidate in (
        # macOS (local dev)
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        # Linux
        "/usr/bin/libreoffice",
        "/usr/bin/soffice",
        "/usr/lib/libreoffice/program/soffice",
        "/usr/lib/libreoffice/program/soffice.bin",
        # Render buildpack
        "/opt/render/project/src/.apt/usr/bin/libreoffice",
        "/opt/render/project/src/.apt/usr/bin/soffice",
        "/opt/render/project/src/.apt/usr/lib/libreoffice/program/soffice",
        "/opt/render/project/src/.apt/usr/lib/libreoffice/program/soffice.bin",
    ):
        if os.path.exists(candidate):
            return candidate
    # Last-resort scan for buildpack-style installs where binary location may vary.
    try:
        import glob

        for pat in (
            "/Applications/LibreOffice.app/**/soffice",
            "/opt/render/project/src/.apt/**/soffice",
            "/opt/render/project/src/.apt/**/libreoffice",
        ):
            for p in glob.glob(pat, recursive=True):
                if os.path.isfile(p) and os.access(p, os.X_OK):
                    return p
    except Exception:
        pass
    return ""


def _find_chrome_bin() -> str:
    for candidate in (
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "chrome",
    ):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    for candidate in (
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/opt/render/project/src/.apt/usr/bin/chromium",
        "/opt/render/project/src/.apt/usr/bin/chromium-browser",
        "/opt/render/project/src/.apt/usr/bin/google-chrome",
        "/opt/render/project/src/.apt/usr/bin/google-chrome-stable",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ):
        if os.path.exists(candidate):
            return candidate
    # Render/buildpack fallback scan
    try:
        import glob
        for pat in (
            "/opt/render/project/src/.apt/**/chromium",
            "/opt/render/project/src/.apt/**/chromium-browser",
            "/opt/render/project/src/.apt/**/google-chrome",
            "/opt/render/project/src/.apt/**/google-chrome-stable",
        ):
            for p in glob.glob(pat, recursive=True):
                if os.path.isfile(p) and os.access(p, os.X_OK):
                    return p
    except Exception:
        pass
    return ""


def _convert_docx_to_pdf_via_libreoffice(
    docx_path: str, pdf_path: str, timeout: int = DOCX_TO_PDF_TIMEOUT_SECONDS
) -> None:
    """Convert DOCX to PDF using LibreOffice headless.
    Uses unique UserInstallation per run to avoid concurrent conversion conflicts
    (multiple instances sharing the same profile cause intermittent failures).
    """
    lo_bin = _find_libreoffice_bin()
    if not lo_bin:
        raise RuntimeError("LibreOffice binary not found (expected libreoffice/soffice)")
    out_dir = os.path.dirname(pdf_path)
    # Unique profile dir per conversion — prevents "sometimes works, sometimes doesn't"
    # when TH+EN or multiple jobs run in parallel.
    profile_dir = os.path.join(out_dir, f"lo_profile_{os.getpid()}_{id(docx_path)}")
    os.makedirs(profile_dir, exist_ok=True)
    try:
        profile_uri = Path(profile_dir).resolve().as_uri()  # e.g. file:///tmp/.../lo_profile_xxx
        cmd = [
            lo_bin,
            f"-env:UserInstallation={profile_uri}",
            "--headless",
            "--nologo",
            "--nofirststartwizard",
            "--convert-to",
            "pdf:writer_pdf_Export",
            "--outdir",
            out_dir,
            os.path.abspath(docx_path),
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=timeout
        )
    finally:
        try:
            shutil.rmtree(profile_dir, ignore_errors=True)
        except Exception:
            pass
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"LibreOffice failed (rc={proc.returncode}): {err[:500]}")
    stem = os.path.splitext(os.path.basename(docx_path))[0]
    expected = os.path.join(out_dir, stem + ".pdf")
    actual = expected if os.path.exists(expected) else None
    if not actual:
        for f in os.listdir(out_dir):
            if f.lower().endswith(".pdf"):
                actual = os.path.join(out_dir, f)
                break
    if not actual or not os.path.exists(actual):
        raise RuntimeError("LibreOffice succeeded but PDF output not found")
    if actual != pdf_path:
        os.rename(actual, pdf_path)


def _convert_docx_to_pdf_via_docx2pdf(docx_path: str, pdf_path: str) -> None:
    """Convert DOCX to PDF using docx2pdf (Windows only, requires Microsoft Word)."""
    try:
        from docx2pdf import convert as docx2pdf_convert
    except ImportError:
        raise RuntimeError("docx2pdf not installed. Run: pip install docx2pdf")
    docx2pdf_convert(os.path.abspath(docx_path), os.path.abspath(pdf_path))


DOCX_TO_PDF_RETRIES = int(os.getenv("DOCX_TO_PDF_RETRIES", "2"))  # Retry once on transient failure


def convert_docx_bytes_to_pdf_bytes(docx_bytes: bytes, filename_stem: str = "report") -> bytes:
    if not docx_bytes:
        raise RuntimeError("DOCX bytes are empty")

    with tempfile.TemporaryDirectory(prefix="docx2pdf_") as td:
        input_path = os.path.join(td, f"{filename_stem}.docx")
        pdf_path = os.path.join(td, f"{filename_stem}.pdf")
        with open(input_path, "wb") as f:
            f.write(docx_bytes)

        last_err = None
        for attempt in range(max(1, DOCX_TO_PDF_RETRIES)):
            try:
                _convert_docx_to_pdf_via_libreoffice(
                    input_path, pdf_path, timeout=DOCX_TO_PDF_TIMEOUT_SECONDS
                )
                last_err = None
                break
            except Exception as e:
                last_err = e
                if attempt + 1 < max(1, DOCX_TO_PDF_RETRIES):
                    logger.warning("[pdf] docx->pdf attempt %s failed, retrying: %s", attempt + 1, e)
                    time.sleep(1)  # Brief pause before retry

        if last_err:
            if USE_DOCX2PDF_FALLBACK:
                try:
                    _convert_docx_to_pdf_via_docx2pdf(input_path, pdf_path)
                except Exception as e2:
                    raise RuntimeError(f"LibreOffice failed: {last_err} | docx2pdf fallback failed: {e2}")
            else:
                raise last_err

        if not os.path.exists(pdf_path):
            raise RuntimeError("PDF conversion succeeded but output not found")

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        if not pdf_bytes:
            raise RuntimeError("Converted PDF is empty")
        return pdf_bytes


def _image_data_uri(path: str) -> str:
    if not path or (not os.path.exists(path)):
        return ""
    ext = os.path.splitext(path)[1].lower().lstrip(".") or "png"
    mime = "image/png" if ext == "png" else "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
    with open(path, "rb") as f:
        raw = f.read()
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _brand_asset_data_uri(filename: str) -> str:
    try:
        root = Path(__file__).resolve().parent.parent
        p = root / filename
        if p.exists():
            return _image_data_uri(str(p))
    except Exception:
        pass
    return ""


def _scale_display_for_lang(scale: str, lang_code: str) -> str:
    s = str(scale or "").strip().lower()
    is_th = str(lang_code or "").strip().lower().startswith("th")
    if s.startswith("high"):
        return "สูง" if is_th else "High"
    if s.startswith("moderate"):
        return "กลาง" if is_th else "Moderate"
    if s.startswith("low"):
        return "ต่ำ" if is_th else "Low"
    return "-"


def build_html_report_file(
    report: ReportData,
    out_html_path: str,
    graph1_path: str,
    graph2_path: str,
    lang_code: str,
    report_style: str = "full",
) -> None:
    """Build a simple HTML report that can be converted to PDF."""
    is_th = str(lang_code or "").strip().lower().startswith("th")
    style = str(report_style or "").strip().lower()
    is_operation_test = style.startswith("operation_test")
    title = (
        "รายงานการวิเคราะห์การนำเสนอด้วยการเคลื่อนไหว กับ AI People Reader"
        if is_th
        else "Movement in Communication with AI People Reader Report"
    )
    date_label = "วันที่วิเคราะห์" if is_th else "Analysis Date"
    client_label = "ชื่อลูกค้า" if is_th else "Client Name"
    duration_label = "ระยะเวลา" if is_th else "Duration"
    detailed_label = "รายละเอียดการวิเคราะห์" if is_th else "Detailed Analysis"
    note_label = "หมายเหตุ" if is_th else "Note"
    scale_label = "ระดับ" if is_th else "Scale"

    fi = report.first_impression
    if fi:
        eye_lv = _scale_display_for_lang(_first_impression_level(fi.eye_contact_pct, "eye_contact"), lang_code)
        up_lv = _scale_display_for_lang(_first_impression_level(fi.upright_pct, "uprightness"), lang_code)
        st_lv = _scale_display_for_lang(_first_impression_level(fi.stance_stability, "stance"), lang_code)
    else:
        eye_lv = up_lv = st_lv = "-"

    g1 = _image_data_uri(graph1_path)
    g2 = _image_data_uri(graph2_path)
    remark_text = (
        "ความรู้สึกที่เกิดจากความประทับใจแรกพบนั้นเป็นสิ่งที่มนุษย์หลีกเลี่ยงไม่ได้ และมักเกิดขึ้นภายใน 5 วินาทีแรกของการพบกัน"
        if is_th
        else "First impression forms quickly, usually within the first 5 seconds. After that, the overall movement and communication cues shape perception."
    )
    remark_extra_page1 = (
        """
      <div style="margin-top: 12px;"></div>
      <div><b>คำอธิบายการผสมผสาน:</b></div>
      <div style="margin-top: 6px;"></div>
      <div><b>1. การสบตาน้อย + ความตั้งตรงน้อย + การยืนและการวางเท้าต่ำ</b></div>
      <div>บุคคลมักดูไม่เป็นภัยและยืดหยุ่น แต่บุคคลอาจดูมีความมั่นใจและอำนาจในระดับต่ำ</div>"""
        if is_th
        else """
      <div style="margin-top: 12px;"></div>
      <div><b>Combination Explanation:</b></div>
      <div style="margin-top: 6px;"></div>
      <div><b>1. Low Eye Contact + Low Uprightness + Low Stance.</b></div>
      <div>The person tends to appear non-threatening and flexible. However, the person can also appear to possess low level of confidence and authority.</div>"""
    )
    remark_extra_page2 = (
        """
      <div><b>2. การสบตาปานกลาง + ความตั้งตรงปานกลาง + การยืนและการวางเท้าปานกลาง</b></div>
      <div>บุคคลมักดูเข้าถึงได้ง่าย และมีความมั่นใจและอำนาจในระดับที่เพียงพอ</div>
      <div style="margin-top: 8px;"><b>3. การสบตาสูง + ความตั้งตรงสูง + การยืนและการวางเท้าสูง</b></div>
      <div>บุคคลมักดูมีความมั่นใจและอำนาจในระดับสูง และอาจดูไม่เข้าถึงได้ง่ายหรือยืดหยุ่น</div>
      <div style="margin-top: 14px;"></div>"""
        if is_th
        else """
      <div><b>2. Moderate Eye Contact + Moderate Uprightness + Moderate Stance.</b></div>
      <div>The person tends to appear approachable, and has adequate level of confidence and authority.</div>
      <div style="margin-top: 8px;"><b>3. High Eye Contact + High Uprightness + High Stance.</b></div>
      <div>The person tends to appear to possess high level of confidence and authority, and may not appear approachable or flexible.</div>
      <div style="margin-top: 14px;"></div>"""
    )

    def cat_scale(i: int) -> str:
        if i < len(report.categories):
            return _scale_display_for_lang(report.categories[i].scale, lang_code)
        return "-"

    header_img = _brand_asset_data_uri("Header.png")
    footer_img = _brand_asset_data_uri("Footer.png")

    # Layout for browser print: fixed brand header/footer + explicit page sections.
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    @page {{ size: A4; margin: 24mm 13mm 22mm 13mm; }}
    body {{
      font-family: "Noto Sans Thai", "TH Sarabun New", "Sarabun", "Tahoma", Arial, Helvetica, sans-serif;
      font-size: 13px;
      line-height: {("1.55" if not is_th else "1.50")};
      color: #111;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
      text-rendering: geometricPrecision;
      margin: 0;
    }}
    .page-header {{
      position: fixed;
      top: -16mm;
      left: 0;
      right: 0;
      text-align: center;
      height: 18mm;
    }}
    .page-header img {{ max-height: 16mm; width: auto; }}
    .page-footer {{
      position: fixed;
      bottom: -14mm;
      left: 0;
      right: 0;
      text-align: center;
      height: 14mm;
      font-size: 10px;
      color: #666;
    }}
    .page-footer img {{ max-height: 10mm; width: auto; }}
    .page {{
      page-break-after: always;
      break-after: page;
    }}
    .page:last-of-type {{
      page-break-after: auto;
      break-after: auto;
    }}
    h1 {{ font-size: 28px; margin: 8mm 0 12px; color: #b24b45; text-align: center; }}
    h2 {{ font-size: 22px; margin: 0 0 12px; }}
    h3 {{ font-size: 18px; margin: 12px 0 8px; }}
    .meta {{ margin: 2px 0; }}
    .section {{ margin-top: 10px; }}
    .scale {{ margin-left: 26px; font-weight: 700; margin-bottom: 10px; }}
    ul {{ margin-top: 6px; margin-bottom: 6px; list-style-type: square; }}
    li {{ margin-bottom: 4px; }}
    .graph {{ margin-top: 4px; page-break-before: always; break-before: page; }}
    .graph img {{ width: 100%; height: auto; }}
  </style>
</head>
<body>
  <div class="page-header">{f'<img src="{header_img}" alt="header" />' if header_img else ''}</div>
  <div class="page-footer">{f'<img src="{footer_img}" alt="footer" />' if footer_img else ''}</div>

  <div class="page">
    <h1>PEOPLE READER</h1>
    <h2>{escape(title)}</h2>
    <div class="meta"><b>{escape(client_label)}:</b> {escape(str(report.client_name or '-'))}</div>
    <div class="meta"><b>{escape(date_label)}:</b> {escape(str(report.analysis_date or '-'))}</div>
    <div class="meta"><b>{escape(duration_label)}:</b> {escape(str(report.video_length_str or '-'))}</div>
    <h3>{escape(detailed_label)}</h3>
    <div class="section">
      <b>{"1. ความประทับใจแรกพบ" if is_th else "1. First impression"}</b>
      <ul>
        <li>{"การสบตา (Eye Contact)" if is_th else "Eye Contact"}</li>
      </ul>
      <div class="scale">{escape(scale_label)}: {escape(eye_lv)}</div>
      <ul>
        <li>{"ความตั้งตรงของร่างกาย (Uprightness)" if is_th else "Uprightness"}</li>
      </ul>
      <div class="scale">{escape(scale_label)}: {escape(up_lv)}</div>
      <ul>
        <li>{"การยืนและการวางเท้า (Stance)" if is_th else "Stance"}</li>
      </ul>
      <div class="scale">{escape(scale_label)}: {escape(st_lv)}</div>
      <div><b>{escape(note_label)}</b></div>
      <div>{escape(remark_text)}</div>
      {remark_extra_page1}
    </div>
  </div>

  <div class="page">
    <div class="section">
      {remark_extra_page2}
      <b>{"2. การสร้างความเป็นมิตรและสร้างสัมพันธภาพ" if is_th else "2. Engaging & Connecting"}</b>
      <ul>
        <li>{"ความเป็นกันเอง" if is_th else "Approachability"}</li>
        <li>{"ความเข้าถึงได้" if is_th else "Relatability"}</li>
        <li>{"การมีส่วนร่วม เชื่อมโยง และสร้างความคุ้นเคยกับทีมอย่างรวดเร็ว" if is_th else "Engagement, connect and build instant rapport with team"}</li>
      </ul>
      <div class="scale">{escape(scale_label)}: {escape(cat_scale(0))}</div>
      <b>{"3. ความมั่นใจ" if is_th else "3. Confidence"}</b>
      <ul>
        <li>{"บุคลิกภาพเชิงบวก" if is_th else "Optimistic Presence"}</li>
        <li>{"ความมีสมาธิ" if is_th else "Focus"}</li>
        <li>{"ความสามารถในการโน้มน้าวและยืนหยัดในจุดยืนเพื่อให้ผู้อื่นคล้อยตาม" if is_th else "Ability to persuade and stand one's ground, in order to convince others."}</li>
      </ul>
      <div class="scale">{escape(scale_label)}: {escape(cat_scale(1))}</div>
      <b>{"4. ความเป็นผู้นำและความดูมีอำนาจ" if is_th else "4. Authority"}</b>
      <ul>
        <li>{"แสดงให้เห็นถึงความสำคัญและความเร่งด่วนของประเด็น" if is_th else "Showing sense of importance and urgency in subject matter"}</li>
        <li>{"ผลักดันให้เกิดการลงมือทำ" if is_th else "Pressing for action"}</li>
      </ul>
      <div class="scale">{escape(scale_label)}: {escape(cat_scale(2))}</div>
    </div>
  </div>
  {f'<div class="graph"><h3>{"ผลการวิเคราะห์ Effort" if is_th else "Effort Motion Detection Results"}</h3><img src="{g1}" /></div>' if (is_operation_test and g1) else ''}
  {f'<div class="graph"><h3>{"ผลการวิเคราะห์ Shape" if is_th else "Shape Motion Detection Results"}</h3><img src="{g2}" /></div>' if (is_operation_test and g2) else ''}
</body>
</html>
"""
    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(html)


def _convert_html_to_pdf_via_chrome(html_path: str, filename_stem: str = "report") -> bytes:
    if not html_path or (not os.path.exists(html_path)):
        raise RuntimeError("HTML file not found for PDF conversion")
    chrome_bin = _find_chrome_bin()
    if not chrome_bin:
        raise RuntimeError("Chrome/Chromium binary not found")
    logger.info("[pdf] chrome binary=%s", chrome_bin)

    with tempfile.TemporaryDirectory(prefix="html2pdf_chrome_") as td:
        out_pdf_path = os.path.join(td, f"{filename_stem}.pdf")
        html_uri = Path(html_path).resolve().as_uri()
        attempts = [
            [
                "--headless=new",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--allow-file-access-from-files",
                "--run-all-compositor-stages-before-draw",
                "--print-to-pdf-no-header",
                f"--print-to-pdf={out_pdf_path}",
                html_uri,
            ],
            [
                "--headless",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--allow-file-access-from-files",
                "--no-pdf-header-footer",
                f"--print-to-pdf={out_pdf_path}",
                html_uri,
            ],
        ]
        errors: List[str] = []
        for idx, flags in enumerate(attempts, start=1):
            try:
                if os.path.exists(out_pdf_path):
                    os.remove(out_pdf_path)
            except Exception:
                pass
            cmd = [chrome_bin, *flags]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=HTML_TO_PDF_TIMEOUT_SECONDS,
                env={**os.environ, "HOME": td},
            )
            if proc.returncode == 0 and os.path.exists(out_pdf_path):
                with open(out_pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                if pdf_bytes:
                    logger.info("[pdf] chrome html->pdf succeeded attempt=%s", idx)
                    return pdf_bytes
            err = (proc.stderr or proc.stdout or "").strip()
            errors.append(f"attempt={idx} rc={proc.returncode} err={err[:280]}")
        raise RuntimeError("Chrome HTML->PDF failed " + " | ".join(errors))


def _convert_html_to_pdf_via_libreoffice(html_path: str, filename_stem: str = "report") -> bytes:
    """Convert HTML file to PDF via LibreOffice."""
    if not html_path or (not os.path.exists(html_path)):
        raise RuntimeError("HTML file not found for PDF conversion")
    lo_bin = _find_libreoffice_bin()
    if not lo_bin:
        raise RuntimeError("LibreOffice binary not found (expected libreoffice/soffice)")

    with tempfile.TemporaryDirectory(prefix="html2pdf_") as td:
        input_path = os.path.join(td, f"{filename_stem}.html")
        expected_pdf_path = os.path.join(td, f"{filename_stem}.pdf")
        shutil.copyfile(html_path, input_path)
        cmd = [
            lo_bin,
            "--headless",
            "--nologo",
            "--nofirststartwizard",
            "--convert-to",
            "pdf",
            "--outdir",
            td,
            input_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=DOCX_TO_PDF_TIMEOUT_SECONDS)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"LibreOffice HTML->PDF failed rc={proc.returncode} err={err[:500]}")
        pdf_path = expected_pdf_path
        if not os.path.exists(pdf_path):
            cands = [p for p in os.listdir(td) if p.lower().endswith(".pdf")]
            if cands:
                pdf_path = os.path.join(td, cands[0])
        if not os.path.exists(pdf_path):
            raise RuntimeError("HTML->PDF conversion succeeded but PDF output not found")
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        if not pdf_bytes:
            raise RuntimeError("Converted PDF is empty")
        return pdf_bytes


def convert_html_file_to_pdf_bytes(html_path: str, filename_stem: str = "report") -> bytes:
    """
    Convert HTML file to PDF with configurable engine.
    - auto: Chrome/Chromium first, then LibreOffice fallback
    - chrome: Chrome/Chromium only
    - libreoffice: LibreOffice only
    """
    engine = HTML_PDF_ENGINE if HTML_PDF_ENGINE in ("auto", "chrome", "libreoffice") else "auto"
    logger.info("[pdf] html_to_pdf engine=%s file=%s", engine, os.path.basename(html_path or ""))
    if engine == "chrome":
        return _convert_html_to_pdf_via_chrome(html_path, filename_stem)
    if engine == "libreoffice":
        return _convert_html_to_pdf_via_libreoffice(html_path, filename_stem)
    try:
        return _convert_html_to_pdf_via_chrome(html_path, filename_stem)
    except Exception as chrome_err:
        logger.warning("[pdf] chrome html->pdf failed, falling back to libreoffice: %s", chrome_err)
        return _convert_html_to_pdf_via_libreoffice(html_path, filename_stem)


def rasterize_pdf_bytes_to_image_pdf_bytes(pdf_bytes: bytes, dpi: int = 220) -> bytes:
    """
    Convert each PDF page into an image and rebuild a PDF from those images.
    This avoids text shaping differences in downstream PDF viewers at the cost of
    non-selectable text.
    """
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        raise RuntimeError(f"PyMuPDF is required for PDF image capture mode: {e}")

    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    if src.page_count == 0:
        raise RuntimeError("Cannot rasterize empty PDF")

    out = fitz.open()
    use_dpi = max(120, min(int(dpi or 220), 400))
    zoom = float(use_dpi) / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for page in src:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        img_pdf = fitz.open("png", img_bytes).convert_to_pdf()
        img_doc = fitz.open("pdf", img_pdf)
        out.insert_pdf(img_doc)
        img_doc.close()

    result = out.tobytes(deflate=True, garbage=3)
    out.close()
    src.close()
    if not result:
        raise RuntimeError("Rasterized PDF output is empty")
    return result


def convert_docx_bytes_to_image_pdf_bytes(docx_bytes: bytes, filename_stem: str = "report", dpi: int = 220) -> bytes:
    """
    Convert DOCX to PNG pages via LibreOffice, then merge PNG pages into a PDF.
    This captures Writer layout directly from DOCX before any reportlab path.
    """
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        raise RuntimeError(f"PyMuPDF is required for DOCX image capture mode: {e}")

    if not docx_bytes:
        raise RuntimeError("DOCX bytes are empty")
    lo_bin = _find_libreoffice_bin()
    if not lo_bin:
        raise RuntimeError("LibreOffice binary not found (expected libreoffice/soffice)")

    with tempfile.TemporaryDirectory(prefix="docx2imgpdf_") as td:
        input_path = os.path.join(td, f"{filename_stem}.docx")
        with open(input_path, "wb") as f:
            f.write(docx_bytes)

        # LibreOffice writer export to PNG (one file per page in most builds).
        cmd = [
            lo_bin,
            "--headless",
            "--nologo",
            "--nofirststartwizard",
            "--convert-to",
            "png",
            "--outdir",
            td,
            input_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=DOCX_TO_PDF_TIMEOUT_SECONDS)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"LibreOffice DOCX->PNG failed rc={proc.returncode} err={err[:500]}")

        png_paths = sorted(
            [
                os.path.join(td, name)
                for name in os.listdir(td)
                if str(name).lower().endswith(".png")
            ]
        )
        if not png_paths:
            raise RuntimeError("DOCX->PNG produced no images")

        out = fitz.open()
        for p in png_paths:
            with open(p, "rb") as f:
                img_bytes = f.read()
            # Optional upscale by dpi via pixmap resample if needed.
            if dpi and int(dpi) > 96:
                img_doc = fitz.open("png", img_bytes)
                pix = img_doc[0].get_pixmap(matrix=fitz.Matrix(float(dpi) / 96.0, float(dpi) / 96.0), alpha=False)
                img_bytes = pix.tobytes("png")
                img_doc.close()
            img_pdf = fitz.open("png", img_bytes).convert_to_pdf()
            page_doc = fitz.open("pdf", img_pdf)
            out.insert_pdf(page_doc)
            page_doc.close()

        result = out.tobytes(deflate=True, garbage=3)
        out.close()
        if not result:
            raise RuntimeError("DOCX image-capture PDF output is empty")
        return result

def s3_key_exists(key: str, retries: int = 3) -> bool:
    """Check if S3 key exists. Retries on transient errors (eventual consistency, network)."""
    last_err = None
    for attempt in range(retries):
        try:
            s3.head_object(Bucket=AWS_BUCKET, Key=key)
            return True
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(0.3 * (attempt + 1))
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
        # Prefer recipients explicitly provided by the user.
        # Fallback to forced list only when no recipient is provided.
        merged = selected if selected else allowed
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
    # Email policy: only send PDF reports (exclude DOCX and HTML).
    report_pdf_keys = {
        label: key
        for label, key in (report_docx_keys or {}).items()
        if key and str(key).strip().lower().endswith(".pdf")
    }
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
    report_labels = [str(label) for label, key in report_pdf_keys.items() if key]
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

    # Fallback links for report PDF in case attachment cannot be included.
    report_link_lines: List[str] = []
    for label, key in report_pdf_keys.items():
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
    for label, key in report_pdf_keys.items():
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
       <b>Group ID / Submission ID:</b> {escape(group_id)}<br/>
       <small>(วางรหัสนี้ในหน้าเว็บเพื่อดาวน์โหลดผลลัพธ์)</small></p>
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
        # Attach report PDF files only (exclude DOCX/HTML).
        for label, key in report_pdf_keys.items():
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
    en_docx_key = str(en_report.get("docx_key") or "").strip()
    en_pdf_key = str(en_report.get("pdf_key") or "").strip()
    en_html_key = str(en_report.get("html_key") or "").strip()
    th_docx_key = str(th_report.get("docx_key") or "").strip()
    th_pdf_key = str(th_report.get("pdf_key") or "").strip()
    th_html_key = str(th_report.get("html_key") or "").strip()
    output_prefix = str(job.get("output_prefix") or f"{OUTPUT_PREFIX}/{job_id}").strip().rstrip("/")
    analysis_date = str(job.get("analysis_date") or datetime.now().strftime("%Y-%m-%d")).strip()
    report_format = str(job.get("report_format") or "docx").strip().lower()
    # Email policy: only send PDF report links.
    ext = "pdf"
    en_key = en_pdf_key
    th_key = th_pdf_key
    raw_languages = job.get("languages") or ["th", "en"]
    if isinstance(raw_languages, str):
        raw_languages = [raw_languages]
    langs = [str(x).strip().lower() for x in raw_languages if str(x).strip()]
    wants_th = any(l.startswith("th") for l in langs)
    wants_en = any(l.startswith("en") for l in langs)
    if not langs:
        wants_th, wants_en = True, True
    report_style = str(job.get("report_style") or "").strip().lower()
    expect_skeleton = bool(job.get("expect_skeleton", False))
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
        "expect_skeleton": expect_skeleton,
        "expect_dots": bool(job.get("expect_dots", True)),
        "dots_key": f"jobs/output/groups/{group_id}/dots.mp4" if group_id else "",
        "skeleton_key": (f"jobs/output/groups/{group_id}/skeleton.mp4" if group_id and expect_skeleton else ""),
        "report_en_key": en_key,
        "report_th_key": th_key,
        "report_en_docx_key": en_docx_key,
        "report_en_pdf_key": en_pdf_key,
        "report_en_html_key": en_html_key,
        "report_th_docx_key": th_docx_key,
        "report_th_pdf_key": th_pdf_key,
        "report_th_html_key": th_html_key,
        "report_th_email_sent": False,
        "report_en_email_sent": False,
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
            expect_skeleton = bool(payload.get("expect_skeleton", False))
            expect_dots = bool(payload.get("expect_dots", True))
            expects_report_th = bool(str(payload.get("report_th_key") or "").strip())
            expect_report_en = bool(str(payload.get("report_en_key") or "").strip())
            attempts = int(payload.get("attempts") or 0)
            # Keep early retries responsive; apply backoff only after repeated failures.
            if EMAIL_RETRY_BACKOFF_SECONDS > 0 and attempts >= 2:
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

            # ส่งแยกกัน: Report 1 เมล์, Skeleton 1 เมล์, Dots 1 เมล์
            job_info = {
                "job_id": job_id,
                "group_id": payload.get("group_id", ""),
                "notify_email": notify_email,
                "report_style": report_style,
            }
            video_keys_report: Dict[str, str] = {}
            if is_operation_test and str(payload.get("input_video_key") or "").strip():
                video_keys_report["Uploaded video (MP4)"] = str(payload.get("input_video_key") or "").strip()

            # 1) Report email (TH + EN)
            report_links: Dict[str, str] = {}
            if (not report_th_sent) and email_payload_report_th_ready(payload):
                th_pdf = str(payload.get("report_th_pdf_key") or payload.get("report_th_key") or "").strip()
                if th_pdf and th_pdf.lower().endswith(".pdf"):
                    report_links["Report TH (PDF)"] = th_pdf
            if (not report_en_sent) and email_payload_report_en_ready(payload):
                en_pdf = str(payload.get("report_en_pdf_key") or payload.get("report_en_key") or "").strip()
                if en_pdf and en_pdf.lower().endswith(".pdf"):
                    report_links["Report EN (PDF)"] = en_pdf
            if report_links:
                sent, status = send_result_email(job_info, video_keys_report, report_links)
                statuses.append(f"reports:{status}")
                if sent:
                    report_th_sent = True
                    report_en_sent = True
                    payload["report_th_email_sent"] = True
                    payload["report_en_email_sent"] = True
                    sent_any = True

            # 2) Skeleton email แยก
            if (not skeleton_sent) and expect_skeleton and email_payload_skeleton_ready(payload):
                sk_key = str(payload.get("skeleton_key") or "").strip()
                sent, status = send_result_email(job_info, {"Skeleton video (MP4)": sk_key}, {})
                statuses.append(f"skeleton:{status}")
                if sent:
                    skeleton_sent = True
                    payload["skeleton_email_sent"] = True
                    sent_any = True

            # 3) Dots email แยก
            if (not dots_sent) and expect_dots and email_payload_dots_ready(payload):
                d_key = str(payload.get("dots_key") or "").strip()
                sent, status = send_result_email(job_info, {"Dots video (MP4)": d_key}, {})
                statuses.append(f"dots:{status}")
                if sent:
                    dots_sent = True
                    payload["dots_email_sent"] = True
                    sent_any = True

            if not statuses:
                waiting = []
                if expects_report_th and not report_th_sent:
                    waiting.append("report_th")
                if expect_report_en and not report_en_sent:
                    waiting.append("report_en")
                if expect_skeleton and not skeleton_sent:
                    waiting.append("skeleton")
                if expect_dots and not dots_sent:
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

            report_th_done = report_th_sent or (not expects_report_th)
            report_en_done = report_en_sent or (not expect_report_en)
            skeleton_done = skeleton_sent or (not expect_skeleton)
            dots_done = dots_sent or (not expect_dots)
            # Finish when all requested deliveries are done.
            all_done = report_th_done and report_en_done and skeleton_done and dots_done
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


def list_processing_json_items() -> Iterable[Dict[str, Any]]:
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=PROCESSING_PREFIX):
        for item in page.get("Contents", []):
            k = str(item.get("Key") or "")
            if k.endswith(".json"):
                yield item


def recover_stale_processing_jobs() -> int:
    """
    On startup, move stale jobs from processing -> pending so they can be retried.
    A job is stale when its last activity timestamp is older than PROCESSING_STALE_MINUTES.
    """
    stale_minutes = max(1, int(PROCESSING_STALE_MINUTES or 0))
    max_items = max(1, int(PROCESSING_RECOVERY_MAX_ITEMS or 1))
    cutoff_seconds = stale_minutes * 60.0
    now = datetime.now(timezone.utc)
    recovered = 0

    logger.info(
        "[startup_recovery] scanning processing jobs stale_minutes=%s max_items=%s",
        stale_minutes,
        max_items,
    )

    for item in list_processing_json_items():
        if recovered >= max_items:
            break
        key = str(item.get("Key") or "").strip()
        if not key:
            continue
        try:
            job = s3_get_json(key, log_key=False)
        except Exception as e:
            logger.warning("[startup_recovery] skip unreadable key=%s err=%s", key, e)
            continue

        mode = str(job.get("mode") or "").strip().lower()
        if mode not in ("report", "report_th_en", "report_generator"):
            continue

        # Prefer explicit timestamps from payload, fallback to S3 last modified.
        last_dt = (
            parse_iso_datetime_utc(job.get("updated_at"))
            or parse_iso_datetime_utc(job.get("created_at"))
        )
        if last_dt is None:
            last_mod = item.get("LastModified")
            if isinstance(last_mod, datetime):
                if last_mod.tzinfo is None:
                    last_mod = last_mod.replace(tzinfo=timezone.utc)
                last_dt = last_mod.astimezone(timezone.utc)
        if last_dt is None:
            logger.warning("[startup_recovery] skip key=%s reason=no_timestamp", key)
            continue

        age_seconds = (now - last_dt).total_seconds()
        if age_seconds < cutoff_seconds:
            continue

        job_id = str(job.get("job_id") or "").strip()
        group_id = str(job.get("group_id") or "").strip()
        pending_key = f"{PENDING_PREFIX}/{job_id}.json" if job_id else key.replace("/processing/", "/pending/")
        recovered_job = update_status(job, "pending", error=None)
        recovered_job["recovered_from_processing_at"] = utc_now_iso()
        recovered_job["recovered_from_processing_key"] = key
        move_json(key, pending_key, recovered_job)
        recovered += 1
        logger.warning(
            "[startup_recovery] requeued stale job_id=%s group_id=%s age_seconds=%.0f from=%s to=%s",
            job_id,
            group_id,
            age_seconds,
            key,
            pending_key,
        )

    logger.info("[startup_recovery] completed recovered=%s", recovered)
    return recovered


def count_processing_jobs() -> int:
    total = 0
    for _ in list_processing_json_items():
        total += 1
    return total


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
            # Skip jobs that are in retry backoff
            retry_after = job.get("retry_after")
            if retry_after is not None:
                try:
                    ts = float(retry_after)
                    if time.time() < ts:
                        continue
                except (TypeError, ValueError):
                    pass
            priority = int(job.get("priority") or 0)
            created_at = str(job.get("created_at") or "")
            if (
                priority > picked_priority
                or (priority == picked_priority and created_at > picked_created_at)
            ):
                picked_key = k
                picked_priority = priority
                picked_created_at = created_at
        else:
            logger.debug(
                "[find_one_pending_job_key] ignore non-report key=%s mode=%s",
                k,
                mode,
            )
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
    logger.info("[analysis] job_id=%s analysis_mode=%r want_real=%s mp=%s", str(job.get("job_id") or "").strip(), analysis_mode, want_real, "ok" if mp is not None else "None")
    mediapipe_runtime_failed = False
    if want_real and (mp is not None):
        logger.info("[analysis] Using real mediapipe analysis (sample_fps=%s, max_frames=%s)", sample_fps, max_frames)
        try:
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
        except Exception as e:
            logger.warning("[analysis] MediaPipe runtime failed, falling back to placeholder: %s", e)
            logger.exception("[analysis] MediaPipe full traceback:")
            mediapipe_runtime_failed = True

    if want_real and mp is None:
        logger.warning("[analysis] MediaPipe unavailable (mp=None); using placeholder. Install mediapipe for real analysis.")
    reason = "mediapipe_runtime_failed" if mediapipe_runtime_failed else ("mp=None" if (want_real and mp is None) else f"analysis_mode={analysis_mode!r} (not real)")
    logger.info("[analysis] Using fallback placeholder analysis job_id=%s reason=%s", str(job.get("job_id") or "").strip(), reason)
    job_id = str(job.get("job_id") or "").strip()
    return analyze_video_placeholder(video_path=video_path, job_id=job_id)


def _first_impression_level(value: float, metric: str = "") -> str:
    """Uses report_core.first_impression_level (eye contact: 60/35 thresholds)."""
    return first_impression_level(value, metric=metric).lower()


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
    if not client_name or client_name.lower() == "anonymous":
        input_key = str(job.get("input_key") or "").strip()
        if input_key:
            base = os.path.basename(input_key)
            client_name = (os.path.splitext(base)[0] or base).strip()
        if not client_name:
            client_name = os.path.splitext(os.path.basename(video_path))[0] or "video"

    duration_str = format_seconds_to_mmss(float(result.get("duration_seconds") or get_video_duration_seconds(video_path)))
    total = int(result.get("total_indicators") or 0) or 1

    # Run First Impression analysis (guard against MediaPipe runtime/import issues).
    audience_mode = str(job.get("audience_mode") or "one").strip().lower()
    if audience_mode not in ("one", "many"):
        audience_mode = "one"
    try:
        first_impression = analyze_first_impression_from_video(
            video_path, sample_every_n=3, max_frames=100, audience_mode=audience_mode
        )
    except Exception as e:
        logger.warning("[first_impression] analysis failed, using zero fallback: %s", e)
        first_impression = FirstImpressionData(
            eye_contact_pct=0.0,
            upright_pct=0.0,
            stance_stability=0.0,
        )
    
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
    if not report_style:
        report_style = "full"
    report_format = str(job.get("report_format") or "docx").strip().lower()
    # Generate PDF for every report job (DOCX -> PDF via LibreOffice path is supported).
    wants_pdf = True
    logger.info(
        "[report] render lang=%s style=%s format=%s enterprise_folder=%s",
        lang_code,
        report_style,
        report_format,
        enterprise_folder,
    )
    graph1_path = ""
    graph2_path = ""
    # Requirement update: operation_test must include graph pages in both TH/EN reports.
    should_generate_graphs = True
    if should_generate_graphs:
        try:
            effort_data = result.get("effort_detection") or {}
            shape_data = result.get("shape_detection") or {}
            if isinstance(effort_data, dict) and isinstance(shape_data, dict) and effort_data and shape_data:
                graph1_path = os.path.join(out_dir, f"Graph_1_{lang_code}.png")
                graph2_path = os.path.join(out_dir, f"Graph_2_{lang_code}.png")
                generate_effort_graph(effort_data, shape_data, graph1_path)
                generate_shape_graph(shape_data, graph2_path)
            else:
                logger.info("[report] skip graph generation: missing effort/shape data lang=%s", lang_code)
        except Exception as e:
            logger.warning("[report] graph generation failed for lang=%s: %s", lang_code, e)
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

    # Build HTML for every report job (docx/pdf) so UI can always offer
    # debug/preview download and operators can inspect layout issues quickly.
    pdf_bytes = None
    pdf_generation_mode = "not_requested"
    pdf_out_path = ""
    html_out_path = os.path.join(
        out_dir,
        f"Presentation_Analysis_Report_{analysis_date}_{lang_code.upper()}.html",
    )
    try:
        build_html_report_file(
            report=report,
            out_html_path=html_out_path,
            graph1_path=graph1_path,
            graph2_path=graph2_path,
            lang_code=lang_code,
            report_style=report_style,
        )
    except Exception as e:
        logger.warning("[report] html build failed for lang=%s: %s", lang_code, e)
        html_out_path = ""

    # PDF (file -> bytes) for every report job.
    if wants_pdf:
        pdf_generation_mode = "requested"
        strict_html_pdf = bool(
            PDF_HTML_STRICT_FOR_OPERATION_TEST and is_operation_test_style(report_style)
        )

        # Try DOCX -> PDF first (LibreOffice) as primary path.
        try:
            pdf_bytes = convert_docx_bytes_to_pdf_bytes(
                docx_bytes,
                filename_stem=f"Presentation_Analysis_Report_{analysis_date}_{lang_code.upper()}",
            )
            logger.info("[pdf] generated via docx->pdf conversion lang=%s", lang_code)
            pdf_generation_mode = "docx_libreoffice"
        except Exception as e:
            logger.warning("[pdf] docx->pdf conversion failed for lang=%s: %s", lang_code, e)
            pdf_bytes = None

        # When DOCX fails: try ReportLab first (reliable multi-page + sub-bullets).
        # HTML->PDF can produce single-page or missing sub-bullets due to Chrome print quirks.
        if not pdf_bytes:
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
                    if pdf_bytes:
                        pdf_generation_mode = "reportlab"
                        logger.info("[pdf] generated via reportlab fallback lang=%s (full layout)", lang_code)
            except Exception as e:
                logger.warning("[pdf] reportlab fallback failed for lang=%s: %s", lang_code, e)
                pdf_bytes = None

        if strict_html_pdf and (not pdf_bytes):
            raise RuntimeError(
                "operation_test strict mode: docx->pdf and reportlab both failed; "
                "html->pdf fallback is disabled"
            )

        # Last resort: HTML -> PDF (can have print/layout quirks).
        if (not pdf_bytes) and PDF_VIA_HTML_FIRST and html_out_path:
            try:
                pdf_bytes = convert_html_file_to_pdf_bytes(
                    html_path=html_out_path,
                    filename_stem=f"Presentation_Analysis_Report_{analysis_date}_{lang_code.upper()}",
                )
                logger.info("[pdf] generated via html->pdf conversion lang=%s (last resort)", lang_code)
                pdf_generation_mode = f"html_{HTML_PDF_ENGINE}"
            except Exception as e:
                logger.warning("[pdf] html->pdf conversion failed for lang=%s: %s", lang_code, e)
                pdf_bytes = None

    key_map = {
        "graph1_path": graph1_path,
        "graph2_path": graph2_path,
        "pdf_out_path": pdf_out_path,
        "html_out_path": html_out_path,
        "pdf_generation_mode": pdf_generation_mode,
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
    if is_operation_test_style(report_style) or enterprise_folder == "operation_test":
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
    if report_format not in ("docx", "pdf", "html"):
        report_format = "docx"

    # Operational Test uses dedicated layout.
    # Keep user-selected languages/formats (e.g., EN-only, HTML-only) when provided.
    if report_style == "operation_test":
        if not languages:
            languages = ["th", "en"]
        job["languages"] = languages
        job["report_format"] = report_format
        logger.info("[report] operation_test override: languages=%s report_format=%s", languages, report_format)

    # output prefix
    output_prefix = str(job.get("output_prefix") or f"{OUTPUT_PREFIX}/{job_id}").strip().rstrip("/")
    if report_style == "operation_test" and group_id:
        # Keep Operational Test outputs under the same group folder as input for easy S3 lookup.
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
        engine = str(result.get("analysis_engine") or "unknown")
        logger.info("[analysis] job_id=%s analysis_engine=%s (real=mediapipe, placeholder=fallback)", job_id, engine)

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

            # Emergency switch: Thai PDF glyph overlap can block delivery.
            # In this mode, keep Thai output as DOCX so customers receive readable files immediately.
            force_thai_docx = bool(EMERGENCY_THAI_DOCX_FALLBACK and lang_code == "th")
            # Upload PDF whenever we have generated bytes (policy: PDF-first delivery).
            wants_pdf_output = bool(pdf_bytes) and (not force_thai_docx)
            if force_thai_docx and report_format == "pdf":
                logger.warning("[report] emergency_thai_docx_fallback enabled; lang=th will upload DOCX instead of PDF")

            # Upload only the requested output format.
            analysis_date = str(job.get("analysis_date") or datetime.now().strftime("%Y-%m-%d")).strip()
            docx_key = None
            wants_docx_output = (
                (report_format == "docx")
                or force_thai_docx
                or ((report_format == "pdf") and DOCX_FALLBACK_ON_PDF_FAIL)
            )
            if wants_docx_output:
                docx_name = f"Presentation_Analysis_Report_{analysis_date}_{lang_code.upper()}.docx"
                docx_key = f"{output_prefix}/{docx_name}"
                upload_bytes(
                    docx_key,
                    docx_bytes,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
                if report_style == "operation_test" and group_id:
                    ui_docx = f"jobs/output/groups/{group_id}/report_{lang_code}.docx"
                    upload_bytes(
                        ui_docx,
                        docx_bytes,
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )

            # Upload HTML (when generated) for debug/preview.
            html_key = None
            html_path = str(local_paths.get("html_out_path") or "").strip()
            if html_path and os.path.exists(html_path):
                html_name = os.path.basename(html_path)
                html_key = f"{output_prefix}/{html_name}"
                upload_file(html_path, html_key, "text/html; charset=utf-8")
            if report_format == "html" and not html_key:
                raise RuntimeError("HTML format requested but HTML generation failed")

            # Upload PDF when requested and not overridden by Thai fallback.
            pdf_key = None
            pdf_render_mode = str(local_paths.get("pdf_generation_mode") or "disabled")
            if wants_pdf_output:
                if not pdf_bytes:
                    if DOCX_FALLBACK_ON_PDF_FAIL and docx_key:
                        logger.warning(
                            "[pdf] generation failed lang=%s; using DOCX fallback output only",
                            lang_code,
                        )
                        pdf_render_mode = "docx_fallback_only"
                        wants_pdf_output = False
                    else:
                        raise RuntimeError("PDF format requested but PDF generation failed")
            if wants_pdf_output:
                if pdf_bytes is None:
                    raise RuntimeError("PDF format requested but PDF bytes are empty after conversion")
                if lang_code == "th" and THAI_PDF_IMAGE_CAPTURE:
                    try:
                        if THAI_CAPTURE_FROM_DOCX_DIRECT:
                            pdf_bytes = convert_docx_bytes_to_image_pdf_bytes(
                                docx_bytes,
                                filename_stem=f"Presentation_Analysis_Report_{analysis_date}_{lang_code.upper()}",
                                dpi=THAI_PDF_IMAGE_DPI,
                            )
                            logger.info(
                                "[pdf] thai docx-direct image-capture enabled lang=%s dpi=%s",
                                lang_code,
                                THAI_PDF_IMAGE_DPI,
                            )
                            pdf_render_mode = "image_capture_docx_direct"
                        else:
                            pdf_bytes = rasterize_pdf_bytes_to_image_pdf_bytes(pdf_bytes, dpi=THAI_PDF_IMAGE_DPI)
                            logger.info("[pdf] thai image-capture pdf enabled lang=%s dpi=%s", lang_code, THAI_PDF_IMAGE_DPI)
                            pdf_render_mode = "image_capture"
                    except Exception as e:
                        err_text = str(e or "")
                        lo_missing = "libreoffice binary not found" in err_text.lower()
                        if THAI_PDF_IMAGE_CAPTURE_STRICT and not lo_missing:
                            raise RuntimeError(
                                f"thai image-capture conversion failed (strict mode): {e}"
                            ) from e
                        if lo_missing:
                            logger.warning(
                                "[pdf] thai image-capture skipped because LibreOffice is unavailable; using fallback pdf path"
                            )
                            pdf_render_mode = "docx_or_reportlab_fallback"
                        else:
                            logger.warning("[pdf] thai image-capture conversion failed lang=%s err=%s", lang_code, e)
                            pdf_render_mode = "docx_or_reportlab_fallback"
                pdf_name = f"Presentation_Analysis_Report_{analysis_date}_{lang_code.upper()}.pdf"
                pdf_key = f"{output_prefix}/{pdf_name}"
                upload_bytes(pdf_key, pdf_bytes, "application/pdf")
                if report_style == "operation_test" and group_id:
                    # Canonical path for Operational Test so UI/S3 always find report in group folder.
                    canonical_pdf_key = f"jobs/groups/{group_id}/{pdf_name}"
                    if canonical_pdf_key != pdf_key:
                        upload_bytes(canonical_pdf_key, pdf_bytes, "application/pdf")
                    pdf_key = canonical_pdf_key
                # Always upload to jobs/output/groups/ so local web and UI default paths (report_en.pdf, report_th.pdf) work
                if group_id:
                    ui_canonical = f"jobs/output/groups/{group_id}/report_{lang_code}.pdf"
                    upload_bytes(ui_canonical, pdf_bytes, "application/pdf")

            outputs["graphs"][lang_code.upper()] = {"graph1_key": g1_key, "graph2_key": g2_key}
            outputs["reports"][lang_code.upper()] = {
                "docx_key": docx_key,
                "html_key": html_key,
                "pdf_key": pdf_key,
                "pdf_render_mode": pdf_render_mode,
            }

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
            report_en_key = str(report_en_key or en_report.get("html_key") or "").strip()
            report_th_key = str(th_report.get("docx_key") or th_report.get("pdf_key") or "").strip()
            report_th_key = str(report_th_key or th_report.get("html_key") or "").strip()
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
                expect_skeleton = bool(payload.get("expect_skeleton", False))
                expect_dots = bool(payload.get("expect_dots", True))
                expects_report_th = bool(str(payload.get("report_th_key") or "").strip())
                expect_report_en = bool(str(payload.get("report_en_key") or "").strip())

                # ส่งแยกกัน: Report 1 เมล์, Skeleton 1 เมล์, Dots 1 เมล์
                job_info = {
                    "job_id": payload["job_id"],
                    "group_id": payload.get("group_id", ""),
                    "notify_email": payload["notify_email"],
                    "report_style": report_style,
                }
                video_keys_report: Dict[str, str] = {}
                if is_operation_test and str(payload.get("input_video_key") or "").strip():
                    video_keys_report["Uploaded video (MP4)"] = str(payload.get("input_video_key") or "").strip()

                # 1) Report email (TH + EN)
                th_ready = email_payload_report_th_ready(payload)
                en_ready = email_payload_report_en_ready(payload)
                report_links: Dict[str, str] = {}
                if th_ready and str(payload.get("report_th_pdf_key") or "").strip():
                    report_links["Report TH (PDF)"] = str(payload.get("report_th_pdf_key") or "").strip()
                if en_ready and str(payload.get("report_en_pdf_key") or "").strip():
                    report_links["Report EN (PDF)"] = str(payload.get("report_en_pdf_key") or "").strip()
                if report_links and (not report_th_sent or not report_en_sent):
                    sent, status = send_result_email(job_info, video_keys_report, report_links)
                    statuses.append(f"reports:{status}")
                    if sent:
                        report_th_sent = True
                        report_en_sent = True
                        payload["report_th_email_sent"] = True
                        payload["report_en_email_sent"] = True

                # 2) Skeleton email แยก
                if (not skeleton_sent) and expect_skeleton and email_payload_skeleton_ready(payload):
                    sk_key = str(payload.get("skeleton_key") or "").strip()
                    sent, status = send_result_email(
                        job_info,
                        {"Skeleton video (MP4)": sk_key},
                        {},
                    )
                    statuses.append(f"skeleton:{status}")
                    if sent:
                        skeleton_sent = True
                        payload["skeleton_email_sent"] = True

                # 3) Dots email แยก
                if (not dots_sent) and expect_dots and email_payload_dots_ready(payload):
                    d_key = str(payload.get("dots_key") or "").strip()
                    sent, status = send_result_email(
                        job_info,
                        {"Dots video (MP4)": d_key},
                        {},
                    )
                    statuses.append(f"dots:{status}")
                    if sent:
                        dots_sent = True
                        payload["dots_email_sent"] = True

                report_th_done = report_th_sent or (not expects_report_th)
                report_en_done = report_en_sent or (not expect_report_en)
                skeleton_done = skeleton_sent or (not expect_skeleton)
                dots_done = dots_sent or (not expect_dots)
                primary_done = report_th_done and report_en_done and skeleton_done and dots_done
                if not primary_done:
                    queue_email_pending(payload)
                    waiting = []
                    if expects_report_th and not report_th_sent:
                        waiting.append("report_th")
                    if expect_report_en and not report_en_sent:
                        waiting.append("report_en")
                    if expect_skeleton and not skeleton_sent:
                        waiting.append("skeleton")
                    if expect_dots and not dots_sent:
                        waiting.append("dots")
                    statuses.append("waiting_for_" + "_and_".join(waiting) if waiting else "waiting")

                email_sent = bool(primary_done or report_th_sent or report_en_sent or skeleton_sent or dots_sent)
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
    group_id = str(raw_job.get("group_id") or "").strip()

    if not job_id:
        raise ValueError("Job JSON missing 'job_id'")

    logger.info("[process_job] job_id=%s group_id=%s mode=%s key=%s", job_id, group_id, mode, job_json_key)

    # Check if this worker should handle this job type
    if mode not in ("report", "report_th_en", "report_generator"):
        logger.info("[process_job] Skipping job_id=%s group_id=%s mode=%s (not a report job)", job_id, group_id, mode)
        return  # Leave job in pending for other workers

    # Verify input video exists before taking the job (avoids processing when upload not ready)
    input_key = str(raw_job.get("input_key") or "").strip()
    if not input_key:
        job = update_status(dict(raw_job), "failed", error="missing input_key")
        failed_key = f"{FAILED_PREFIX}/{job_id}.json"
        move_json(job_json_key, failed_key, job)
        logger.warning("[process_job] job_id=%s missing input_key, moved to failed", job_id)
        return
    if not s3_key_exists(input_key):
        job_created = parse_job_id_datetime_utc(job_id) or parse_iso_datetime_utc(raw_job.get("created_at"))
        age_hours = (datetime.now(timezone.utc) - job_created).total_seconds() / 3600.0 if job_created else 999.0
        if age_hours >= PENDING_STALE_HOURS:
            err_msg = f"input_key not found in S3 after {age_hours:.0f}h: {input_key}"
            job = update_status(dict(raw_job), "failed", error=err_msg)
            failed_key = f"{FAILED_PREFIX}/{job_id}.json"
            move_json(job_json_key, failed_key, job)
            logger.warning("[process_job] job_id=%s input_key missing for %.0fh, moved to failed: %s", job_id, age_hours, input_key)
        else:
            logger.info("[process_job] job_id=%s input_key=%s not found (age=%.1fh), will retry", job_id, input_key, age_hours)
        return

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
        logger.info("[process_job] job_id=%s group_id=%s finished", job_id, group_id)

    except Exception as exc:
        logger.exception("[process_job] job_id=%s group_id=%s FAILED: %s", job_id, group_id, exc)
        retry_count = int(job.get("retry_count") or 0)
        if retry_count < JOB_MAX_RETRIES:
            job["retry_count"] = retry_count + 1
            job["last_error"] = str(exc)
            # Backoff: wait before retry (30s, 60s, 90s) so transient issues can resolve
            backoff_sec = 30 * (retry_count + 1)
            job["retry_after"] = (datetime.now(timezone.utc).timestamp() + backoff_sec)
            job = update_status(job, "pending", error=None)
            pending_key = f"{PENDING_PREFIX}/{job_id}.json"
            move_json(processing_key, pending_key, job)
            logger.info("[process_job] job_id=%s retry %d/%d, moved to pending (retry_after +%ds)", job_id, retry_count + 1, JOB_MAX_RETRIES, backoff_sec)
        else:
            job = update_status(job, "failed", error=str(exc))
            failed_key = f"{FAILED_PREFIX}/{job_id}.json"
            move_json(processing_key, failed_key, job)


# -----------------------------------------
# Main loop
# -----------------------------------------
def main() -> None:
    logger.info("====== AI People Reader Report Worker (TH/EN) ======")
    logger.info("report_core version: %s", getattr(_report_core, "REPORT_CORE_VERSION", "unknown"))
    mp_status = "available" if mp is not None else "UNAVAILABLE (placeholder only)"
    logger.info("MediaPipe: %s", mp_status)
    logger.info("DEFAULT_ANALYSIS_MODE: %s (env ANALYSIS_MODE=%s)", DEFAULT_ANALYSIS_MODE, os.getenv("ANALYSIS_MODE", "(not set)"))
    if not DEFAULT_ANALYSIS_MODE.startswith("real"):
        logger.warning(
            "ANALYSIS_MODE=%s → reports will use PLACEHOLDER (not real MediaPipe). Set ANALYSIS_MODE=real on Render for real analysis.",
            DEFAULT_ANALYSIS_MODE,
        )
    logger.info("Using bucket: %s", AWS_BUCKET)
    logger.info("Region       : %s", AWS_REGION)
    logger.info("Poll every   : %s seconds", POLL_INTERVAL)
    lo_bin = _find_libreoffice_bin()
    logger.info(
        "PDF config   : via_html_first=%s html_engine=%s thai_image_capture=%s thai_docx_direct=%s",
        PDF_VIA_HTML_FIRST,
        HTML_PDF_ENGINE,
        THAI_PDF_IMAGE_CAPTURE,
        THAI_CAPTURE_FROM_DOCX_DIRECT,
    )
    logger.info(
        "LibreOffice  : %s (DOCX->PDF primary path; if empty, will fallback to ReportLab)",
        lo_bin or "(not found - install LibreOffice for local: https://www.libreoffice.org/download)",
    )
    logger.info(
        "PDF strict   : html_strict_for_operation_test=%s",
        PDF_HTML_STRICT_FOR_OPERATION_TEST,
    )
    logger.info(
        "PDF fallback : docx_fallback_on_pdf_fail=%s",
        DOCX_FALLBACK_ON_PDF_FAIL,
    )
    logger.info(
        "Recovery cfg : processing_stale_minutes=%s pending_stale_hours=%s processing_recovery_max_items=%s",
        PROCESSING_STALE_MINUTES,
        PENDING_STALE_HOURS,
        PROCESSING_RECOVERY_MAX_ITEMS,
    )
    logger.info(
        "Loop cfg     : recovery_interval_seconds=%s idle_heartbeat_seconds=%s",
        PROCESSING_RECOVERY_INTERVAL_SECONDS,
        IDLE_HEARTBEAT_SECONDS,
    )
    log_ses_runtime_context()
    log_smtp_runtime_context()
    recover_stale_processing_jobs()
    last_recovery_ts = time.time()
    last_heartbeat_ts = time.time()

    while True:
        try:
            now_ts = time.time()
            if now_ts - last_recovery_ts >= max(10, PROCESSING_RECOVERY_INTERVAL_SECONDS):
                recover_stale_processing_jobs()
                last_recovery_ts = now_ts

            # Prioritize report generation first to avoid email_pending backlog starving new results.
            job_key = find_one_pending_job_key()
            if job_key:
                process_job(job_key)
                last_heartbeat_ts = time.time()
                if ENABLE_EMAIL_NOTIFICATIONS and EMAIL_QUEUE_MAX_ITEMS_AFTER_JOB > 0:
                    process_pending_email_queue(max_items=EMAIL_QUEUE_MAX_ITEMS_AFTER_JOB)
            else:
                if now_ts - last_heartbeat_ts >= max(10, IDLE_HEARTBEAT_SECONDS):
                    processing_count = count_processing_jobs()
                    logger.info(
                        "[heartbeat] worker_alive pending=0 processing=%s poll_interval=%ss",
                        processing_count,
                        POLL_INTERVAL,
                    )
                    last_heartbeat_ts = now_ts
                if ENABLE_EMAIL_NOTIFICATIONS:
                    process_pending_email_queue(max_items=EMAIL_QUEUE_MAX_ITEMS_WHEN_IDLE)
                time.sleep(POLL_INTERVAL)
        except Exception as exc:
            logger.exception("[main] Unexpected error: %s", exc)
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()