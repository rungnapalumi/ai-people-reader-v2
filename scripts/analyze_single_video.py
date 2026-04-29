#!/usr/bin/env python3
"""Extract + score a single video with the Presentation Analysis pipeline.

Usage: python scripts/analyze_single_video.py path/to/video.mov
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main(video_path: str) -> int:
    os.environ.setdefault("PRES_SCORER_MODE", "hybrid")

    from src.report_core import (
        analyze_first_impression_from_video,
        analyze_video_mediapipe,
    )
    from src.presentation_scorer import score_presentation, build_overview
    from src.presentation_ml import load_models

    load_models(force_reload=True)

    print(f"=== Analyzing: {video_path} ===\n")

    print("Step 1/3 — First Impression (5-second window)...")
    fi = analyze_first_impression_from_video(video_path, audience_mode="one")
    print(f"  eye_contact_pct   = {fi.eye_contact_pct:.2f}")
    print(f"  upright_pct       = {fi.upright_pct:.2f}")
    print(f"  stance_stability  = {fi.stance_stability:.2f}")

    print("\nStep 2/3 — Full-video MediaPipe (sample_fps=3, max_frames=200)...")
    result = analyze_video_mediapipe(video_path, sample_fps=3, max_frames=200)
    print(f"  analyzed_frames   = {result.get('analyzed_frames')}")
    print(f"  effort_counts     = {result.get('effort_counts')}")

    fi_tuple = (fi.eye_contact_pct, fi.upright_pct, fi.stance_stability)
    cat_feats = result.get("category_features", {})

    overview = build_overview(
        first_impression_pct=fi_tuple,
        category_features=cat_feats,
        analysis_result=result,
    )
    print("\nStep 3/3 — PresentationOverview (key signals):")
    for k in (
        "stance_stability", "upright_pct", "eye_contact_pct", "hip_sway_std",
        "hand_block_share", "hand_low_share", "hands_above_share",
        "distinct_hand_shapes", "strong_effort_share", "gesture_share",
    ):
        v = getattr(overview, k, None)
        if isinstance(v, float):
            print(f"  {k:<22} = {v:.3f}")
        else:
            print(f"  {k:<22} = {v}")

    pred = score_presentation(
        first_impression_pct=fi_tuple,
        category_features=cat_feats,
        analysis_result=result,
    )

    print("\n=== Presentation Analysis Bands ===")
    for k in ("eye_contact", "uprightness", "stance", "engaging",
              "adaptability", "confidence", "authority"):
        print(f"  {k:<14} = {pred.get(k)}")
    print(f"  engine         = {pred.get('engine')}")

    print("\n=== Rationale per category ===")
    for cat, reasons in (pred.get("rationale") or {}).items():
        reasons_s = "; ".join(reasons) if reasons else "(none)"
        print(f"  {cat:<14} {reasons_s}")

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/analyze_single_video.py path/to/video.mov")
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1]))
