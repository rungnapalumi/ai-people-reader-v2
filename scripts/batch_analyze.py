#!/usr/bin/env python3
"""Batch-analyze every video in a folder with the Presentation Analysis
pipeline and produce one CSV + a pretty table.

Usage:
    python scripts/batch_analyze.py batch_videos/
    python scripts/batch_analyze.py batch_videos/ --cache scripts/.batch_cache.json
    python scripts/batch_analyze.py batch_videos/ --csv results.csv

Features are cached so re-running is instant for already-processed videos.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".avi"}


def find_videos(folder: Path) -> List[Path]:
    """Recursively list every video file in *folder*."""
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    vids = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        and not p.name.startswith(".")
    )
    return vids


def extract_features(video_path: Path) -> Dict[str, Any]:
    """Run MediaPipe once and return the features the scorer needs."""
    from src.report_core import (
        analyze_first_impression_from_video,
        analyze_video_mediapipe,
    )

    fi = analyze_first_impression_from_video(str(video_path), audience_mode="one")
    result = analyze_video_mediapipe(
        str(video_path), sample_fps=3, max_frames=200,
    )
    return {
        "first_impression": [fi.eye_contact_pct, fi.upright_pct, fi.stance_stability],
        "category_features": result.get("category_features", {}),
        "effort_counts": result.get("effort_counts", {}),
        "analyzed_frames": result.get("analyzed_frames", 0),
    }


def score_from_features(feats: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the full hybrid scorer (ML + lookup + tiebreaker) to cached
    features and return the 7 bands + key raw values."""
    from src.presentation_scorer import score_presentation
    fi = tuple(feats["first_impression"])
    cat_feats = feats.get("category_features", {})
    analysis = {
        "analyzed_frames": feats.get("analyzed_frames", 0),
        "effort_counts": feats.get("effort_counts", {}),
    }
    pred = score_presentation(
        first_impression_pct=fi,
        category_features=cat_feats,
        analysis_result=analysis,
    )
    return pred


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder",
        nargs="?",
        default="batch_videos",
        help="Folder containing videos (default: batch_videos/)",
    )
    parser.add_argument(
        "--cache",
        default="scripts/.batch_cache.json",
        help="JSON cache file (default: scripts/.batch_cache.json)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Output CSV path (default: <folder>_results.csv)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract features even if cached",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress MediaPipe warnings",
    )
    args = parser.parse_args()

    if args.quiet:
        os.environ.setdefault("GLOG_minloglevel", "3")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        logging.getLogger().setLevel(logging.ERROR)
    else:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")

    os.environ.setdefault("PRES_SCORER_MODE", "hybrid")

    # Pre-load ML models.
    from src.presentation_ml import load_models
    load_models(force_reload=True)

    folder = Path(args.folder)
    cache_path = Path(args.cache)
    csv_path = Path(args.csv) if args.csv else Path(f"{folder.name}_results.csv")

    videos = find_videos(folder)
    if not videos:
        print(f"No videos found in {folder}")
        return 2

    cache: Dict[str, Dict[str, Any]] = {}
    if cache_path.exists() and not args.force:
        with open(cache_path, "r", encoding="utf-8") as fh:
            cache = json.load(fh)

    print(f"Found {len(videos)} video(s) in {folder}")
    if cache:
        already = sum(1 for v in videos if v.name in cache)
        print(f"  {already} already cached, {len(videos) - already} to extract")

    rows: List[Dict[str, Any]] = []
    for i, video in enumerate(videos, 1):
        name = video.name
        t0 = time.time()

        if name in cache and not args.force:
            feats = cache[name]
            status = "cached"
        else:
            print(f"[{i}/{len(videos)}] {name} ... extracting", flush=True)
            try:
                feats = extract_features(video)
            except Exception as exc:
                print(f"  !! failed: {exc}")
                continue
            cache[name] = feats
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(cache, fh, indent=2)
            status = f"{time.time()-t0:.1f}s"

        pred = score_from_features(feats)
        fi = feats["first_impression"]
        rows.append({
            "name": name,
            "status": status,
            "eye_pct": round(fi[0], 2),
            "upright_pct": round(fi[1], 2),
            "stance_stability": round(fi[2], 2),
            "eye_contact": pred.get("eye_contact"),
            "uprightness": pred.get("uprightness"),
            "stance": pred.get("stance"),
            "engaging": pred.get("engaging"),
            "adaptability": pred.get("adaptability"),
            "confidence": pred.get("confidence"),
            "authority": pred.get("authority"),
        })

    # Pretty table
    L = {"High": "H", "Moderate": "M", "Low": "L", None: "-"}
    short = ["Eye", "Upr", "Stn", "Eng", "Adp", "Cnf", "Aut"]
    keys = ["eye_contact", "uprightness", "stance",
            "engaging", "adaptability", "confidence", "authority"]

    print("\n" + "=" * 100)
    print("Results")
    print("=" * 100)
    header = (
        f"{'Name':<26} | {'upright':>7} | {'stance':>6} | "
        + " | ".join(f"{s:^3}" for s in short)
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        bands = " | ".join(f"{L[r[k]]:^3}" for k in keys)
        print(
            f"{r['name']:<26} | {r['upright_pct']:>7.1f} | {r['stance_stability']:>6.1f} | {bands}"
        )
    print("-" * len(header))

    # CSV output
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"\nCSV saved → {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
