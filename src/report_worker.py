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
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Iterable, List, Tuple

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

if not AWS_BUCKET:
    raise RuntimeError("Missing AWS_BUCKET (or S3_BUCKET) environment variable")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
)
logger = logging.getLogger("report_worker")

s3 = boto3.client("s3", region_name=AWS_REGION)


# -----------------------------------------
# Small S3 helpers
# -----------------------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def s3_get_json(key: str) -> Dict[str, Any]:
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


def list_pending_json_keys() -> Iterable[str]:
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=PENDING_PREFIX):
        for item in page.get("Contents", []):
            k = item["Key"]
            if k.endswith(".json"):
                yield k


def find_one_pending_job_key() -> Optional[str]:
    for k in list_pending_json_keys():
        logger.info("[find_one_pending_job_key] found %s", k)
        return k
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

    # Graphs
    graph1_path = os.path.join(out_dir, f"Graph_1_{lang_code}.png")
    graph2_path = os.path.join(out_dir, f"Graph_2_{lang_code}.png")

    generate_effort_graph(result.get("effort_detection", {}), result.get("shape_detection", {}), graph1_path)
    generate_shape_graph(result.get("shape_detection", {}), graph2_path)

    # DOCX (in-memory)
    docx_bio = io.BytesIO()
    build_docx_report(report, docx_bio, graph1_path=graph1_path, graph2_path=graph2_path, lang=lang_code)
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

            # Upload graphs (lang-specific copies are okay; keys stable)
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
                time.sleep(POLL_INTERVAL)
        except Exception as exc:
            logger.exception("[main] Unexpected error: %s", exc)
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()