#!/usr/bin/env python3
"""Batch-upload every video in a folder to the Render pipeline and collect
the final 7-category Presentation Analysis bands once each job finishes.

Equivalent to clicking Start Analysis on page 9 once per file, but from
the terminal so 29 videos can be queued with one command.

Usage
-----
    # 1) Upload + queue every mp4/mov in batch_videos/ (default folder).
    #    Prints one row per file and writes a queue CSV you can reuse later.
    python scripts/batch_upload_render.py batch_videos/

    # 2) Upload + queue + poll jobs/finished/*.json until every job is done
    #    (or timeout).  Writes batch_render_results.csv at the end.
    python scripts/batch_upload_render.py batch_videos/ --watch

    # 3) Skip upload (videos already queued) and just poll.  Reuses the
    #    queue CSV from the first run so you can resume.
    python scripts/batch_upload_render.py --poll-only --queue-csv batch_render_queue.csv

Env
---
    Uses the same AWS credentials / S3 bucket as pages/9_Presentation_Analysis.py.
    Reads .env from the repo root (via python-dotenv) if present.
    Required variables: AWS_BUCKET (or S3_BUCKET), AWS_REGION (default ap-southeast-1).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".avi"}

JOBS_PENDING_PREFIX = "jobs/pending/"
JOBS_FINISHED_PREFIX = "jobs/finished/"
JOBS_FAILED_PREFIX = "jobs/failed/"
JOBS_OUTPUT_PREFIX = "jobs/output/"
JOBS_GROUP_PREFIX = "jobs/groups/"

AWS_BUCKET = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET") or ""
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

QUEUE_CSV_DEFAULT = "batch_render_queue.csv"
RESULTS_CSV_DEFAULT = "batch_render_results.csv"

S3_UPLOAD_CONFIG = TransferConfig(
    multipart_threshold=8 * 1024 * 1024,
    multipart_chunksize=8 * 1024 * 1024,
    max_concurrency=4,
    use_threads=True,
)


def _s3_client():
    if not AWS_BUCKET:
        sys.exit(
            "AWS_BUCKET is not set. Add it to .env (same variable "
            "pages/9_Presentation_Analysis.py uses) and re-run."
        )
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        config=Config(signature_version="s3v4"),
    )


def _content_type(name: str) -> str:
    fn = (name or "").lower()
    if fn.endswith(".mp4"):
        return "video/mp4"
    if fn.endswith(".mov"):
        return "video/quicktime"
    if fn.endswith(".m4v"):
        return "video/x-m4v"
    if fn.endswith(".webm"):
        return "video/webm"
    return "application/octet-stream"


def _safe_slug(text: str, fallback: str = "user") -> str:
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
    return s or fallback


def _new_group_id(base_user: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:6]
    return f"{ts}_{rand}__{base_user}"


def _new_job_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:5]
    return f"{ts}__{rand}"


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_videos(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        and not p.name.startswith(".")
    )


def _upload_and_queue(
    s3,
    video: Path,
    audience_mode: str,
    movement_type_mode: str,
    notify_email: str,
    org_name: str,
    report_format: str,
    languages: List[str],
) -> Dict[str, Any]:
    base_user = _safe_slug(video.stem, fallback="user")
    group_id = _new_group_id(base_user)
    input_key = f"{JOBS_GROUP_PREFIX}{group_id}/input/input.mp4"

    size_mb = video.stat().st_size / (1024 * 1024)
    print(f"  → uploading {video.name} ({size_mb:.1f} MB) to s3://{AWS_BUCKET}/{input_key}")

    s3.upload_file(
        Filename=str(video),
        Bucket=AWS_BUCKET,
        Key=input_key,
        ExtraArgs={"ContentType": _content_type(video.name)},
        Config=S3_UPLOAD_CONFIG,
    )

    job_id = _new_job_id()
    created_at = _utc_iso()
    job_report = {
        "job_id": job_id,
        "group_id": group_id,
        "created_at": created_at,
        "status": "pending",
        "mode": "report",
        "input_key": input_key,
        "client_name": video.stem,
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "languages": languages,
        "output_prefix": f"{JOBS_GROUP_PREFIX}{group_id}",
        "analysis_mode": "real",
        "sample_fps": 3,
        "max_frames": 200,
        "people_reader_job": True,
        "presentation_analysis_job": True,
        "report_style": "people_reader",
        "required_report_style": "people_reader",
        "report_format": report_format,
        "expect_skeleton": False,
        "expect_dots": False,
        "notify_email": str(notify_email or "").strip(),
        "enterprise_folder": org_name,
        "employee_id": base_user,
        "employee_email": "",
        "audience_mode": audience_mode,
        "movement_type_mode": movement_type_mode,
        "report_email_send_en_asap": False,
        "bundle_completion_email": False,
        "organization_name": org_name,
        "organization_id": _safe_slug(org_name, fallback="presentation_analysis"),
    }

    pending_key = f"{JOBS_PENDING_PREFIX}{job_id}.json"
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=pending_key,
        Body=json.dumps(job_report).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"    queued job {job_id}  group={group_id}")
    return {
        "name": video.name,
        "group_id": group_id,
        "job_id": job_id,
        "input_key": input_key,
        "pending_key": pending_key,
        "uploaded_at": created_at,
    }


def _try_get_job_json(s3, prefix: str, job_id: str) -> Optional[Dict[str, Any]]:
    key = f"{prefix}{job_id}.json"
    try:
        obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        raise
    return json.loads(obj["Body"].read().decode("utf-8"))


def _poll_until_done(
    s3,
    rows: List[Dict[str, Any]],
    timeout_min: int,
    poll_sec: int,
) -> List[Dict[str, Any]]:
    """Wait until every job leaves jobs/pending/ (finished or failed)."""
    deadline = time.time() + timeout_min * 60
    remaining = {r["job_id"]: r for r in rows}
    results: Dict[str, Dict[str, Any]] = {}

    while remaining and time.time() < deadline:
        done_this_round = []
        for job_id, row in list(remaining.items()):
            finished = _try_get_job_json(s3, JOBS_FINISHED_PREFIX, job_id)
            if finished is not None:
                results[job_id] = {**row, "status": "finished", "job_json": finished}
                done_this_round.append(job_id)
                bands = finished.get("presentation_analysis_bands") or {}
                print(
                    f"  ✓ {row['name']:<30} Eye={bands.get('eye_contact','?')[:1]} "
                    f"Upr={bands.get('uprightness','?')[:1]} Stn={bands.get('stance','?')[:1]} "
                    f"Eng={bands.get('engaging','?')[:1]} Adp={bands.get('adaptability','?')[:1]} "
                    f"Cnf={bands.get('confidence','?')[:1]} Aut={bands.get('authority','?')[:1]}"
                )
                continue
            failed = _try_get_job_json(s3, JOBS_FAILED_PREFIX, job_id)
            if failed is not None:
                results[job_id] = {**row, "status": "failed", "job_json": failed}
                done_this_round.append(job_id)
                print(f"  ✗ {row['name']:<30} FAILED: {failed.get('error','(no error)')[:120]}")

        for jid in done_this_round:
            remaining.pop(jid, None)

        if remaining:
            left = len(remaining)
            total = len(rows)
            print(f"  ... waiting on {left}/{total} jobs  (poll again in {poll_sec}s)", flush=True)
            time.sleep(poll_sec)

    for job_id, row in remaining.items():
        results[job_id] = {**row, "status": "timeout", "job_json": {}}
        print(f"  ! {row['name']:<30} TIMEOUT (still pending after {timeout_min} min)")

    return [results[r["job_id"]] for r in rows]


def _write_queue_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = ["name", "group_id", "job_id", "input_key", "pending_key", "uploaded_at"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nqueue CSV saved → {path}")


def _read_queue_csv(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_results_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = [
        "name", "group_id", "job_id", "status",
        "eye_pct", "upright_pct", "stance_stability",
        "eye_contact", "uprightness", "stance",
        "engaging", "adaptability", "confidence", "authority",
        "error",
    ]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            jj = r.get("job_json") or {}
            bands = jj.get("presentation_analysis_bands") or {}
            fi = jj.get("presentation_analysis_first_impression_raw") or {}
            w.writerow({
                "name": r.get("name", ""),
                "group_id": r.get("group_id", ""),
                "job_id": r.get("job_id", ""),
                "status": r.get("status", ""),
                "eye_pct": round(float(fi.get("eye_contact_pct") or 0.0), 2),
                "upright_pct": round(float(fi.get("upright_pct") or 0.0), 2),
                "stance_stability": round(float(fi.get("stance_stability") or 0.0), 2),
                "eye_contact": bands.get("eye_contact", ""),
                "uprightness": bands.get("uprightness", ""),
                "stance": bands.get("stance", ""),
                "engaging": bands.get("engaging", ""),
                "adaptability": bands.get("adaptability", ""),
                "confidence": bands.get("confidence", ""),
                "authority": bands.get("authority", ""),
                "error": (jj.get("error") or jj.get("last_error") or "") if r.get("status") != "finished" else "",
            })
    print(f"results CSV saved → {path}")


def _print_results_table(rows: List[Dict[str, Any]]) -> None:
    L = {"High": "H", "Moderate": "M", "Low": "L", "": "-"}
    short = ["Eye", "Upr", "Stn", "Eng", "Adp", "Cnf", "Aut"]
    keys = ["eye_contact", "uprightness", "stance",
            "engaging", "adaptability", "confidence", "authority"]

    header = f"{'Name':<28} | {'status':>8} | " + " | ".join(f"{s:^3}" for s in short)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        jj = r.get("job_json") or {}
        bands = jj.get("presentation_analysis_bands") or {}
        band_str = " | ".join(f"{L.get(bands.get(k,''), '-'):^3}" for k in keys)
        print(f"{r.get('name',''):<28} | {r.get('status',''):>8} | {band_str}")
    print("-" * len(header))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", nargs="?", default="batch_videos",
                        help="Folder of videos to upload (default: batch_videos/)")
    parser.add_argument("--watch", action="store_true",
                        help="Poll jobs/finished/*.json after queueing and print band results")
    parser.add_argument("--poll-only", action="store_true",
                        help="Skip upload; just poll jobs listed in --queue-csv")
    parser.add_argument("--queue-csv", default=QUEUE_CSV_DEFAULT,
                        help=f"Queue CSV path (default: {QUEUE_CSV_DEFAULT})")
    parser.add_argument("--results-csv", default=RESULTS_CSV_DEFAULT,
                        help=f"Results CSV path (default: {RESULTS_CSV_DEFAULT})")
    parser.add_argument("--timeout-min", type=int, default=60,
                        help="Maximum minutes to wait for all jobs to finish (default: 60)")
    parser.add_argument("--poll-sec", type=int, default=30,
                        help="Seconds between finished/-bucket polls (default: 30)")
    parser.add_argument("--notify-email", default="",
                        help="Email to notify when each job finishes (optional)")
    parser.add_argument("--audience-mode", default="one", choices=["one", "many"])
    parser.add_argument("--movement-type-mode", default="auto",
                        help="movement_type_mode value for the job (default: auto)")
    parser.add_argument("--report-format", default="pdf", choices=["pdf", "docx"])
    parser.add_argument("--language", default="en",
                        help="Report language: en, th, or en,th (default: en)")
    parser.add_argument("--org-name", default="Presentation Analysis",
                        help="enterprise_folder / organization_name (default: Presentation Analysis)")
    args = parser.parse_args()

    queue_csv = Path(args.queue_csv)
    results_csv = Path(args.results_csv)
    languages = [part.strip() for part in args.language.split(",") if part.strip()] or ["en"]

    s3 = _s3_client()
    print(f"S3 bucket: {AWS_BUCKET}   region: {AWS_REGION}")

    if args.poll_only:
        if not queue_csv.exists():
            sys.exit(f"--poll-only requires {queue_csv} (create with a normal upload run first).")
        rows = _read_queue_csv(queue_csv)
        print(f"Polling {len(rows)} job(s) from {queue_csv}")
    else:
        folder = Path(args.folder)
        videos = _find_videos(folder)
        if not videos:
            sys.exit(f"No videos found in {folder}")
        print(f"Found {len(videos)} video(s) in {folder}")

        rows = []
        for i, video in enumerate(videos, 1):
            print(f"[{i}/{len(videos)}] {video.name}")
            try:
                row = _upload_and_queue(
                    s3, video,
                    audience_mode=args.audience_mode,
                    movement_type_mode=args.movement_type_mode,
                    notify_email=args.notify_email,
                    org_name=args.org_name,
                    report_format=args.report_format,
                    languages=languages,
                )
                rows.append(row)
            except Exception as exc:
                print(f"  !! upload/queue failed: {exc}")
        _write_queue_csv(rows, queue_csv)

    if args.watch or args.poll_only:
        print(f"\nPolling every {args.poll_sec}s (timeout {args.timeout_min} min)...")
        poll_rows = _poll_until_done(s3, rows, args.timeout_min, args.poll_sec)
        _print_results_table(poll_rows)
        _write_results_csv(poll_rows, results_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
