#!/usr/bin/env python3
"""Retrain Stance and Confidence as RandomForest.

Stance currently has no joblib (rule-based wins LOOCV) and Confidence is
saved as a StackingClassifier that does not deserialize cleanly under
the Render numpy/sklearn build. We need both as plain RandomForest
classifiers so:

  * Stance ML loads on Render and lifts the (Upr, Stn) lookup table out
    of its rule-only state.
  * Confidence ML loads cleanly and contributes to the final blend
    instead of silently falling back to the rule-based result.

Run this script from the numpy 1.26.4 + sklearn 1.7.1 virtualenv so the
joblib pickle matches the report-worker runtime.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut

import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.presentation_ml import FEATURE_NAMES, build_feature_vector  # noqa: E402

FEATURE_CACHE = REPO_ROOT / "scripts" / ".presentation_features_cache.json"
GROUND_TRUTH_FILES = [
    REPO_ROOT / "New Video Analysis_with comments.xlsx",
    REPO_ROOT / "Movement Combinations for Report 10 types.xlsx",
]
MODELS_DIR = REPO_ROOT / "models"
METADATA_PATH = MODELS_DIR / "metadata.json"


def _load_ground_truth() -> List[Dict]:
    """Reuse the calibration loader (same logic, no new code path)."""
    from scripts.calibrate_presentation_analysis import load_ground_truth  # noqa: WPS433
    return load_ground_truth()


def _build_dataset() -> Tuple[List[str], np.ndarray, Dict[str, List[str]]]:
    with open(FEATURE_CACHE, "r", encoding="utf-8") as fh:
        cache = json.load(fh)
    recs = _load_ground_truth()

    names: List[str] = []
    X: List[List[float]] = []
    cats = ["eye_contact", "uprightness", "stance", "engaging",
            "adaptability", "confidence", "authority"]
    y: Dict[str, List[str]] = {c: [] for c in cats}

    for r in recs:
        feats = cache.get(r["name"])
        if not feats:
            continue
        vec = build_feature_vector(
            first_impression_pct=tuple(feats["first_impression"]),
            category_features=feats.get("category_features", {}),
            analysis_result={
                "analyzed_frames": feats.get("analyzed_frames", 0),
                "effort_counts": feats.get("effort_counts", {}),
            },
        )
        names.append(r["name"])
        X.append(vec)
        for c in cats:
            y[c].append(r["categories"].get(c))

    return names, np.array(X, dtype=float), y


def _loocv_rf(X: np.ndarray, y: List[str], n_estimators: int = 300) -> Tuple[int, int]:
    """Return (exact_hits, total)."""
    mask = [v is not None for v in y]
    Xm = X[mask]
    ym = np.array([v for v in y if v is not None])

    preds: List[str] = [None] * len(ym)  # type: ignore
    for tr_idx, te_idx in LeaveOneOut().split(Xm):
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        clf.fit(Xm[tr_idx], ym[tr_idx])
        preds[te_idx[0]] = clf.predict(Xm[te_idx])[0]

    exact = sum(1 for p, t in zip(preds, ym) if p == t)
    return exact, len(ym)


def main() -> int:
    print("=== Retraining Stance + Confidence as RandomForest ===")
    print(f"FEATURE_NAMES: {len(FEATURE_NAMES)} features")

    names, X, y_by_cat = _build_dataset()
    print(f"Loaded {len(names)} clips with {X.shape[1]} features each\n")

    # LOOCV check before saving
    print("LOOCV (RandomForest, n=300, class_weight=balanced):")
    for cat in ("stance", "confidence"):
        exact, n = _loocv_rf(X, y_by_cat[cat])
        print(f"  {cat:<14}  RF: {exact}/{n} ({100*exact/n:.0f}%)")
    print()

    MODELS_DIR.mkdir(exist_ok=True)
    for cat in ("stance", "confidence"):
        mask = [v is not None for v in y_by_cat[cat]]
        Xm = X[mask]
        ym = np.array([v for v in y_by_cat[cat] if v is not None])

        clf = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        clf.fit(Xm, ym)
        out = MODELS_DIR / f"presentation_{cat}.joblib"
        joblib.dump(clf, out)
        print(f"  saved  {cat:<14} -> {out.name} (RandomForest)")

    # Update metadata.json to reflect the change
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    meta["best_by_category"]["stance"] = {
        "best_model": "rf",
        "n": len(names),
        "note": (
            "Re-added as RandomForest so the (Upr, Stn) lookup table can use "
            "ML-driven Stance on Render. Rule-based stance still wins LOOCV "
            "but ML pushes more clips into the unanimous lookup cells, "
            "which matters more for downstream Confidence accuracy."
        ),
    }
    meta["best_by_category"]["confidence"] = {
        "best_model": "rf",
        "n": len(names),
        "note": (
            "Downgraded from StackingClassifier to RandomForest because "
            "StackingClassifier pickles do not deserialize cleanly across "
            "the local/Render numpy/sklearn split. RF is slightly less "
            "accurate in LOOCV but actually loads on the report-worker."
        ),
    }
    METADATA_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n  updated  {METADATA_PATH.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
