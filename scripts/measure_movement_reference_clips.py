#!/usr/bin/env python3
"""
Measure summary_features for movement-type reference videos (same settings as report worker).

Usage (from repo root):
  python scripts/measure_movement_reference_clips.py
  python scripts/measure_movement_reference_clips.py --json calibration_measured_summaries.json

Reference files (gitignored *.mov):
  Type 1.mov … Type 6.mov, K.Type 7.mov, Type 8.mov, Type 9.mov, Type 10.mov

Update TYPE_TEMPLATES `expected` in movement_type_classifier.py from these numbers (with bands),
then bump REPORT_CORE_VERSION in src/report_core.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# type_id -> filename under REPO_ROOT
REFERENCE_FILES = {
    "type_1": "Type 1.mov",
    "type_2": "Type 2.mov",
    "type_3": "Type 3.mov",
    "type_4": "Type 4.mov",
    "type_5": "Type 5.mov",
    "type_6": "Type 6.mov",
    "type_7": "K.Type 7.mov",
    "type_8": "Type 8.mov",
    "type_9": "Type 9.mov",
    "type_10": "Type 10.mov",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json",
        default="",
        help="Write merged summary dict to this path (e.g. calibration_measured_summaries.json)",
    )
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
    import report_core as rc  # noqa: E402
    import movement_type_classifier as mtc  # noqa: E402

    out: dict = {}
    for tid, name in REFERENCE_FILES.items():
        path = os.path.join(REPO_ROOT, name)
        if not os.path.isfile(path):
            print(f"[skip] {tid}: missing {path}", file=sys.stderr)
            continue
        print(f"[run] {tid} <- {name}", file=sys.stderr)
        feats = rc.extract_movement_type_frame_features_from_video(
            path,
            audience_mode="one",
            sample_every_n=3,
            max_frames=300,
        )
        sf = mtc.build_summary_features_from_timeseries(feats)
        out[tid] = {k: float(sf[k]) for k in sorted(sf.keys())}

    print(json.dumps(out, indent=2))

    if args.json:
        out_path = args.json if os.path.isabs(args.json) else os.path.join(REPO_ROOT, args.json)
        merged = {}
        if os.path.isfile(out_path):
            with open(out_path, encoding="utf-8") as f:
                merged = json.load(f)
        if not isinstance(merged, dict):
            merged = {}
        merged.update(out)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
            f.write("\n")
        print(f"[write] {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
