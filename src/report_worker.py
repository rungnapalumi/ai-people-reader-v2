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
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Iterable, List, Tuple
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

import boto3

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
EMAIL_LINK_EXPIRES_SECONDS = int(os.getenv("EMAIL_LINK_EXPIRES_SECONDS", "604800"))  # up to 7 days
MAX_DOCX_ATTACHMENT_BYTES = int(os.getenv("MAX_DOCX_ATTACHMENT_BYTES", "4194304"))  # 4MB per docx
ENABLE_EMAIL_NOTIFICATIONS = str(os.getenv("ENABLE_EMAIL_NOTIFICATIONS", "true")).strip().lower() in ("1", "true", "yes", "on")
POLL_INTERVAL = int(os.getenv("JOB_POLL_INTERVAL", "10"))

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

s3 = boto3.client("s3", region_name=AWS_REGION)
ses = boto3.client("ses", region_name=SES_REGION)


# -----------------------------------------
# Small S3 helpers
# -----------------------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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

def presigned_get_url(key: str, expires: int = EMAIL_LINK_EXPIRES_SECONDS) -> str:
    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": AWS_BUCKET, "Key": key},
        ExpiresIn=expires,
    )

def is_valid_email(value: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", (value or "").strip()))

def s3_read_bytes(key: str) -> bytes:
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    return obj["Body"].read()

def _docx_filename_from_key(key: str, fallback: str) -> str:
    base = os.path.basename(str(key or "").strip())
    return base if base.lower().endswith(".docx") else fallback

def send_result_email(
    job: Dict[str, Any],
    video_keys: Dict[str, str],
    report_docx_keys: Dict[str, str],
) -> Tuple[bool, str]:
    to_email = str(job.get("notify_email") or "").strip()
    if not is_valid_email(to_email):
        logger.info("[email] skip: invalid or missing notify_email=%s", to_email)
        return False, "invalid_or_missing_notify_email"
    if not SES_FROM_EMAIL:
        logger.warning("[email] skip: SES_FROM_EMAIL is not configured")
        return False, "missing_ses_from_email"

    job_id = str(job.get("job_id") or "").strip()
    group_id = str(job.get("group_id") or "").strip()
    subject = f"AI People Reader - Results Ready ({group_id or job_id})"
    lines = [
        f"Job ID: {job_id}",
        f"Your analysis results are ready for group: {group_id}",
        "",
        "Attached files:",
        "- Report EN (DOCX)",
        "- Report TH (DOCX)",
        "",
        "Video links:",
    ]
    for label, key in video_keys.items():
        if key:
            lines.append(f"- {label}: {presigned_get_url(key)}")

    # Fallback links for report DOCX in case attachment cannot be included.
    report_link_lines: List[str] = []
    for label, key in report_docx_keys.items():
        if key:
            report_link_lines.append(f"- {label}: {presigned_get_url(key)}")

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

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = SES_FROM_EMAIL
    msg["To"] = to_email
    msg.attach(MIMEText(body_text, "plain", "utf-8"))

    # Attach DOCX reports if available and not too large.
    for label, key in report_docx_keys.items():
        if not key:
            continue
        try:
            docx_bytes = s3_read_bytes(key)
            if len(docx_bytes) > MAX_DOCX_ATTACHMENT_BYTES:
                logger.warning("[email] skip attachment too large label=%s key=%s size=%d", label, key, len(docx_bytes))
                continue
            fallback_name = "report_en.docx" if "EN" in label else "report_th.docx"
            filename = _docx_filename_from_key(key, fallback=fallback_name)
            part = MIMEApplication(
                docx_bytes,
                _subtype="vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
            part.add_header("Content-Disposition", "attachment", filename=filename)
            msg.attach(part)
        except Exception as e:
            logger.warning("[email] cannot attach DOCX label=%s key=%s err=%s", label, key, e)

    try:
        ses.send_raw_email(
            Source=SES_FROM_EMAIL,
            Destinations=[to_email],
            RawMessage={"Data": msg.as_bytes()},
        )
        logger.info("[email] sent to=%s group_id=%s", to_email, group_id)
        return True, "sent"
    except Exception as e:
        # Fallback: if Raw email is denied, send plain email without attachments.
        err_str = str(e)
        if "SendRawEmail" in err_str and "AccessDenied" in err_str:
            try:
                ses.send_email(
                    Source=SES_FROM_EMAIL,
                    Destination={"ToAddresses": [to_email]},
                    Message={
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
                )
                logger.warning(
                    "[email] raw denied, sent fallback plain email to=%s group_id=%s",
                    to_email,
                    group_id,
                )
                return True, "sent_fallback_plain_email_no_attachment"
            except Exception as e2:
                logger.exception(
                    "[email] fallback send_email failed to=%s group_id=%s err=%s",
                    to_email,
                    group_id,
                    e2,
                )
                return False, f"send_failed_raw_and_fallback: {e2}"

        logger.exception("[email] send failed to=%s group_id=%s err=%s", to_email, group_id, e)
        return False, f"send_failed: {e}"

def build_email_payload(job: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
    group_id = str(job.get("group_id") or "").strip()
    return {
        "job_id": str(job.get("job_id") or "").strip(),
        "group_id": group_id,
        "notify_email": str(job.get("notify_email") or "").strip(),
        "dots_key": f"jobs/output/groups/{group_id}/dots.mp4" if group_id else "",
        "skeleton_key": f"jobs/output/groups/{group_id}/skeleton.mp4" if group_id else "",
        "report_en_key": outputs.get("reports", {}).get("EN", {}).get("docx_key", ""),
        "report_th_key": outputs.get("reports", {}).get("TH", {}).get("docx_key", ""),
        "attempts": 0,
        "updated_at": utc_now_iso(),
    }

def email_payload_all_ready(payload: Dict[str, Any]) -> bool:
    required = [
        payload.get("dots_key", ""),
        payload.get("skeleton_key", ""),
        payload.get("report_en_key", ""),
        payload.get("report_th_key", ""),
    ]
    required = [k for k in required if k]
    if not required:
        return False
    return all(s3_key_exists(k) for k in required)

def queue_email_pending(payload: Dict[str, Any]) -> str:
    job_id = str(payload.get("job_id") or "").strip()
    if not job_id:
        raise ValueError("Missing job_id for email pending queue")
    key = f"{EMAIL_PENDING_PREFIX}/{job_id}.json"
    s3_put_json(key, payload)
    return key

def update_finished_job_notification(job_id: str, sent: bool, status: str) -> None:
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
    s3_put_json(key, job)

def process_pending_email_queue(max_items: int = 10) -> None:
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
            if not notify_email:
                update_finished_job_notification(job_id, False, "skipped_no_notify_email")
                s3.delete_object(Bucket=AWS_BUCKET, Key=key)
                continue

            if not email_payload_all_ready(payload):
                # Keep waiting until all outputs are ready.
                update_finished_job_notification(job_id, False, "waiting_for_all_outputs")
                continue

            sent, status = send_result_email(
                {"job_id": job_id, "group_id": payload.get("group_id", ""), "notify_email": notify_email},
                {
                    "Dots video (MP4)": payload.get("dots_key", ""),
                    "Skeleton video (MP4)": payload.get("skeleton_key", ""),
                },
                {
                    "Report EN (DOCX)": payload.get("report_en_key", ""),
                    "Report TH (DOCX)": payload.get("report_th_key", ""),
                },
            )
            update_finished_job_notification(job_id, sent, status if sent else "send_failed")

            if sent or status in ("invalid_or_missing_notify_email", "missing_ses_from_email"):
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
    for k in list_pending_json_keys():
        try:
            job = s3_get_json(k)
        except Exception as e:
            # Key may be consumed by another worker between list/get.
            logger.info("[find_one_pending_job_key] skip key=%s reason=%s", k, e)
            continue

        mode = str(job.get("mode") or "").strip().lower()
        if mode in ("report", "report_th_en", "report_generator"):
            logger.info("[find_one_pending_job_key] picked report key=%s mode=%s", k, mode)
            return k

        logger.info("[find_one_pending_job_key] ignore non-report key=%s mode=%s", k, mode)
    return None


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


def generate_reports_for_lang(
    job: Dict[str, Any],
    result: Dict[str, Any],
    video_path: str,
    lang_code: str,
    out_dir: str,
) -> Tuple[bytes, Optional[bytes], Dict[str, str]]:
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

    report_style = str(job.get("report_style") or "full").strip().lower()
    is_simple = report_style.startswith("simple")

    # Graphs are only required for full report style.
    graph1_path = ""
    graph2_path = ""
    if not is_simple:
        graph1_path = os.path.join(out_dir, f"Graph_1_{lang_code}.png")
        graph2_path = os.path.join(out_dir, f"Graph_2_{lang_code}.png")
        generate_effort_graph(result.get("effort_detection", {}), result.get("shape_detection", {}), graph1_path)
        generate_shape_graph(result.get("shape_detection", {}), graph2_path)

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

    # PDF (file -> bytes). Thai PDF requires Thai TTF (your existing build_pdf_report already enforces that)
    pdf_bytes = None
    pdf_out_path = os.path.join(out_dir, f"Presentation_Analysis_Report_{analysis_date}_{lang_code.upper()}.pdf")
    try:
        build_pdf_report(report, pdf_out_path, graph1_path=graph1_path, graph2_path=graph2_path, lang=lang_code)
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
    return docx_bytes, pdf_bytes, key_map


def process_report_job(job: Dict[str, Any]) -> Dict[str, Any]:
    job_id = str(job.get("job_id") or "").strip()
    input_key = str(job.get("input_key") or "").strip()
    if not job_id:
        raise ValueError("Job JSON missing 'job_id'")
    if not input_key:
        raise ValueError("Job JSON missing 'input_key'")

    # languages: default TH+EN
    languages = job.get("languages") or ["th", "en"]
    if isinstance(languages, str):
        languages = [languages]
    languages = [str(x).strip().lower() for x in languages if str(x).strip()]
    if not languages:
        languages = ["th", "en"]

    # output prefix
    output_prefix = str(job.get("output_prefix") or f"{OUTPUT_PREFIX}/{job_id}").strip().rstrip("/")
    # We'll store files under:
    #   <output_prefix>/report_TH.docx, report_EN.docx, report_TH.pdf, report_EN.pdf
    #   <output_prefix>/Graph_1_TH.png, Graph_2_TH.png, ...
    logger.info("[report] job_id=%s input_key=%s languages=%s output_prefix=%s", job_id, input_key, languages, output_prefix)

    # Download video
    video_suffix = os.path.splitext(input_key)[1] or ".mp4"
    video_path = download_to_temp(input_key, suffix=video_suffix)

    out_dir = tempfile.mkdtemp(prefix=f"report_{job_id}_")
    try:
        # Analyze once (shared for both languages)
        result = run_analysis(video_path, job)

        outputs: Dict[str, Any] = {"reports": {}, "graphs": {}}

        for lang_code in languages:
            lang_code = "th" if lang_code.startswith("th") else "en"
            docx_bytes, pdf_bytes, local_paths = generate_reports_for_lang(job, result, video_path, lang_code, out_dir)

            # Upload graphs only for full style.
            g1_key = None
            g2_key = None
            if local_paths.get("graph1_path") and local_paths.get("graph2_path"):
                g1_key = f"{output_prefix}/Graph_1_{lang_code.upper()}.png"
                g2_key = f"{output_prefix}/Graph_2_{lang_code.upper()}.png"
                upload_file(local_paths["graph1_path"], g1_key, "image/png")
                upload_file(local_paths["graph2_path"], g2_key, "image/png")

            # Upload DOCX
            analysis_date = str(job.get("analysis_date") or datetime.now().strftime("%Y-%m-%d")).strip()
            docx_name = f"Presentation_Analysis_Report_{analysis_date}_{lang_code.upper()}.docx"
            docx_key = f"{output_prefix}/{docx_name}"
            upload_bytes(
                docx_key,
                docx_bytes,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

            # Upload PDF if built
            pdf_key = None
            if pdf_bytes:
                pdf_name = f"Presentation_Analysis_Report_{analysis_date}_{lang_code.upper()}.pdf"
                pdf_key = f"{output_prefix}/{pdf_name}"
                upload_bytes(pdf_key, pdf_bytes, "application/pdf")

            outputs["graphs"][lang_code.upper()] = {"graph1_key": g1_key, "graph2_key": g2_key}
            outputs["reports"][lang_code.upper()] = {"docx_key": docx_key, "pdf_key": pdf_key}

        # Save structured outputs into job JSON
        job["output_prefix"] = output_prefix
        job["outputs"] = outputs
        job["analysis_engine"] = str(result.get("analysis_engine") or "unknown")
        job["duration_seconds"] = float(result.get("duration_seconds") or 0.0)
        job["analyzed_frames"] = int(result.get("analyzed_frames") or 0)

        # Notification flow:
        # - report job finishes normally
        # - email is sent when all outputs (dots/skeleton/reports) are ready
        if ENABLE_EMAIL_NOTIFICATIONS:
            payload = build_email_payload(job, outputs)
            if str(payload.get("notify_email") or "").strip():
                if email_payload_all_ready(payload):
                    email_sent, email_status = send_result_email(
                        {"job_id": payload["job_id"], "group_id": payload.get("group_id", ""), "notify_email": payload["notify_email"]},
                        {
                            "Dots video (MP4)": payload.get("dots_key", ""),
                            "Skeleton video (MP4)": payload.get("skeleton_key", ""),
                        },
                        {
                            "Report EN (DOCX)": payload.get("report_en_key", ""),
                            "Report TH (DOCX)": payload.get("report_th_key", ""),
                        },
                    )
                else:
                    queue_email_pending(payload)
                    email_sent, email_status = False, "waiting_for_all_outputs"
            else:
                email_sent, email_status = False, "skipped_no_notify_email"
        else:
            email_sent, email_status = False, "disabled_by_config"
        job["notification"] = {
            "notify_email": str(job.get("notify_email") or "").strip(),
            "sent": email_sent,
            "status": email_status,
            "updated_at": utc_now_iso(),
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

    while True:
        try:
            job_key = find_one_pending_job_key()
            if job_key:
                process_job(job_key)
            else:
                if ENABLE_EMAIL_NOTIFICATIONS:
                    process_pending_email_queue(max_items=10)
                time.sleep(POLL_INTERVAL)
        except Exception as exc:
            logger.exception("[main] Unexpected error: %s", exc)
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()