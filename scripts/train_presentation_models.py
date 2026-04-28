#!/usr/bin/env python3
"""Train 7 RandomForest classifiers for the Presentation Analysis scorer.

Workflow
--------
1. Read the 18-clip ground truth from ``New Video Analysis_with comments.xlsx``.
2. Load cached features from ``scripts/.presentation_features_cache.json``
   (produced by ``calibrate_presentation_analysis.py --extract``).
3. Build a feature matrix using :func:`src.presentation_ml.build_feature_vector`
   so training and inference share the same layout.
4. For each of the 7 categories:
     a. Run Leave-One-Out cross-validation with a RandomForestClassifier.
        Report exact accuracy + adjacent (±1 band) accuracy and compare to
        the rule-based scorer's accuracy on the same clips.
     b. Retrain on all 18 clips and save to ``models/presentation_{cat}.joblib``.
5. Persist a ``models/metadata.json`` with feature names, hyperparameters,
   per-category LOOCV accuracy, and the rule-based baseline for comparison.

Usage::

    python3.11 scripts/train_presentation_models.py --train
    python3.11 scripts/train_presentation_models.py --train --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.presentation_ml import (  # noqa: E402
    CATEGORIES,
    FEATURE_NAMES,
    build_feature_vector,
)
from src.presentation_scorer import score_presentation  # noqa: E402
from scripts.calibrate_presentation_analysis import (  # noqa: E402
    FEATURE_CACHE,
    load_ground_truth,
)

MODELS_DIR = os.path.join(REPO_ROOT, "models")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")

BAND_ORDER = ["Low", "Moderate", "High"]


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_training_data() -> Tuple[List[str], List[List[float]], Dict[str, List[str]]]:
    """Return (names, X, y_by_category).

    ``names`` is a parallel list of clip names. ``X`` is shape (N, F) with
    features in :data:`FEATURE_NAMES` order. ``y_by_category`` maps each
    category to a list of length N with "Low"/"Moderate"/"High" labels (or
    ``None`` for clips missing that label).
    """
    with open(FEATURE_CACHE, "r", encoding="utf-8") as fh:
        cache = json.load(fh)

    records = load_ground_truth()
    names: List[str] = []
    X: List[List[float]] = []
    y_by_cat: Dict[str, List[str]] = {c: [] for c in CATEGORIES}

    for rec in records:
        name = rec["name"]
        feats = cache.get(name)
        if not feats:
            logging.warning("  no cached features for %s — skipping", name)
            continue

        vec = build_feature_vector(
            first_impression_pct=tuple(feats["first_impression"]),
            category_features=feats.get("category_features", {}),
            analysis_result={
                "analyzed_frames": feats.get("analyzed_frames", 0),
                "effort_counts": feats.get("effort_counts", {}),
            },
        )
        names.append(name)
        X.append(vec)

        for cat in CATEGORIES:
            y_by_cat[cat].append(rec["categories"].get(cat))

    return names, X, y_by_cat


def rule_based_predictions(
    names: List[str],
    cache: Dict[str, Any],
) -> Dict[str, List[str]]:
    """Run the existing rule-based scorer for baseline comparison."""
    by_cat: Dict[str, List[str]] = {c: [] for c in CATEGORIES}
    for name in names:
        feats = cache[name]
        pred = score_presentation(
            first_impression_pct=tuple(feats["first_impression"]),
            category_features=feats.get("category_features", {}),
            analysis_result={
                "analyzed_frames": feats.get("analyzed_frames", 0),
                "effort_counts": feats.get("effort_counts", {}),
            },
        )
        for cat in CATEGORIES:
            by_cat[cat].append(pred.get(cat))
    return by_cat


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def _adjacent(t: str, p: str) -> bool:
    if t not in BAND_ORDER or p not in BAND_ORDER:
        return False
    return abs(BAND_ORDER.index(t) - BAND_ORDER.index(p)) <= 1


def _accuracy(y_true: List[str], y_pred: List[str]) -> Tuple[int, int, int]:
    """Return (exact_hits, adjacent_hits, total_labeled)."""
    exact = adj = total = 0
    for t, p in zip(y_true, y_pred):
        if t is None:
            continue
        total += 1
        if t == p:
            exact += 1
        if _adjacent(t or "", p or ""):
            adj += 1
    return exact, adj, total


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train_and_evaluate(
    names: List[str],
    X: List[List[float]],
    y_by_cat: Dict[str, List[str]],
    cache: Dict[str, Any],
    verbose: bool = False,
    save: bool = True,
) -> Dict[str, Any]:
    """Run LOOCV + retrain-on-all for each category.

    Returns a metadata dict with per-category accuracy (RF and rule-based).
    """
    try:
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import LeaveOneOut
        import joblib
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn / joblib are required for training. "
            "Install with `pip install scikit-learn joblib`."
        ) from exc

    os.makedirs(MODELS_DIR, exist_ok=True)

    X_np = np.array(X, dtype=float)
    N = len(names)

    rule_preds = rule_based_predictions(names, cache)

    results: Dict[str, Any] = {
        "feature_names": FEATURE_NAMES,
        "n_samples": N,
        "categories": {},
        "hyperparams": {
            "n_estimators": 400,
            "max_depth": 6,
            "min_samples_leaf": 1,
            "class_weight": "balanced",
            "random_state": 42,
        },
    }

    print(f"\nLeave-One-Out Cross-Validation on {N} clips ({len(FEATURE_NAMES)} features)\n")
    print(f"{'Category':<14} | {'RF Exact':>10} | {'RF Adj':>8} | {'Rule Exact':>12} | {'Rule Adj':>10} | Δ exact")
    print("-" * 84)

    for cat in CATEGORIES:
        y_cat = y_by_cat[cat]

        # Filter out missing labels.
        mask = [v is not None for v in y_cat]
        X_cat = X_np[mask]
        y_cat_clean = [y for y in y_cat if y is not None]
        names_cat = [n for n, m in zip(names, mask) if m]
        y_arr = np.array(y_cat_clean)

        if len(set(y_cat_clean)) < 2:
            # Can't train a classifier with only one class. Majority-vote.
            majority = Counter(y_cat_clean).most_common(1)[0][0]
            preds_loocv = [majority] * len(y_cat_clean)
            results["categories"][cat] = {
                "degenerate_single_class": majority,
                "n": len(y_cat_clean),
            }
            # Save a constant model by training on the data anyway (sklearn
            # will handle it; but to keep things simple we just skip saving).
            rf_exact = sum(1 for a, b in zip(y_cat_clean, preds_loocv) if a == b)
            rf_adj = sum(1 for a, b in zip(y_cat_clean, preds_loocv) if _adjacent(a, b))
        else:
            preds_loocv: List[str] = [None] * len(y_cat_clean)  # type: ignore
            loo = LeaveOneOut()
            for train_idx, test_idx in loo.split(X_cat):
                clf = RandomForestClassifier(
                    n_estimators=400,
                    max_depth=6,
                    min_samples_leaf=1,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                )
                clf.fit(X_cat[train_idx], y_arr[train_idx])
                preds_loocv[int(test_idx[0])] = str(clf.predict(X_cat[test_idx])[0])

            rf_exact, rf_adj, rf_total = _accuracy(y_cat_clean, preds_loocv)

            # Retrain on the full (labeled) dataset and persist.
            final_clf = RandomForestClassifier(
                n_estimators=400,
                max_depth=6,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            final_clf.fit(X_cat, y_arr)
            if save:
                joblib.dump(
                    final_clf,
                    os.path.join(MODELS_DIR, f"presentation_{cat}.joblib"),
                )

        # Rule-based baseline on the same clips (filter to non-None labels).
        rule_preds_cat = [
            rule_preds[cat][i] for i, m in enumerate(mask) if m
        ]
        rb_exact, rb_adj, rb_total = _accuracy(y_cat_clean, rule_preds_cat)

        n_total = len(y_cat_clean)
        rf_exact_pct = 100.0 * rf_exact / max(1, n_total)
        rf_adj_pct = 100.0 * rf_adj / max(1, n_total)
        rb_exact_pct = 100.0 * rb_exact / max(1, n_total)
        rb_adj_pct = 100.0 * rb_adj / max(1, n_total)
        delta = rf_exact - rb_exact

        print(
            f"{cat:<14} | "
            f"{rf_exact}/{n_total} {rf_exact_pct:4.1f}% | "
            f"{rf_adj}/{n_total} {rf_adj_pct:4.1f}% | "
            f"{rb_exact}/{n_total} {rb_exact_pct:6.1f}% | "
            f"{rb_adj}/{n_total} {rb_adj_pct:5.1f}% | "
            f"{delta:+d}"
        )

        cat_meta = results["categories"].setdefault(cat, {})
        cat_meta.update({
            "n": n_total,
            "class_distribution": dict(Counter(y_cat_clean)),
            "rf_loocv": {
                "exact": rf_exact,
                "adj": rf_adj,
                "exact_pct": round(rf_exact_pct, 2),
                "adj_pct": round(rf_adj_pct, 2),
            },
            "rule_based": {
                "exact": rb_exact,
                "adj": rb_adj,
                "exact_pct": round(rb_exact_pct, 2),
                "adj_pct": round(rb_adj_pct, 2),
            },
        })

        if verbose:
            print("    per-clip (truth / loocv / rule):")
            for i, nm in enumerate(names_cat):
                t = y_cat_clean[i]
                r = preds_loocv[i]
                rb = rule_preds_cat[i]
                tag_r = "✓" if t == r else ("~" if _adjacent(t, r) else "✗")
                tag_rb = "✓" if t == rb else ("~" if _adjacent(t, rb) else "✗")
                print(f"      {nm:<12}  truth={t:<9} rf={r:<9}{tag_r}  rule={rb:<9}{tag_rb}")

    # Totals
    print("-" * 84)
    rf_total_exact = sum(results["categories"][c].get("rf_loocv", {}).get("exact", 0) for c in CATEGORIES)
    rf_total_adj = sum(results["categories"][c].get("rf_loocv", {}).get("adj", 0) for c in CATEGORIES)
    rb_total_exact = sum(results["categories"][c].get("rule_based", {}).get("exact", 0) for c in CATEGORIES)
    rb_total_adj = sum(results["categories"][c].get("rule_based", {}).get("adj", 0) for c in CATEGORIES)
    total_n = sum(results["categories"][c].get("n", 0) for c in CATEGORIES)
    if total_n:
        print(
            f"{'TOTAL':<14} | "
            f"{rf_total_exact}/{total_n} {100*rf_total_exact/total_n:4.1f}% | "
            f"{rf_total_adj}/{total_n} {100*rf_total_adj/total_n:4.1f}% | "
            f"{rb_total_exact}/{total_n} {100*rb_total_exact/total_n:6.1f}% | "
            f"{rb_total_adj}/{total_n} {100*rb_total_adj/total_n:5.1f}% | "
            f"{rf_total_exact - rb_total_exact:+d}"
        )
        results["total"] = {
            "rf_loocv_exact": rf_total_exact,
            "rf_loocv_adj": rf_total_adj,
            "rule_based_exact": rb_total_exact,
            "rule_based_adj": rb_total_adj,
            "n": total_n,
        }

    # --- Feature importance per model ---
    print("\nTop-5 feature importances per category (from retrained model):")
    for cat in CATEGORIES:
        model_path = os.path.join(MODELS_DIR, f"presentation_{cat}.joblib")
        if not os.path.exists(model_path):
            continue
        try:
            import joblib
            est = joblib.load(model_path)
            importances = list(zip(FEATURE_NAMES, est.feature_importances_))
            importances.sort(key=lambda kv: -kv[1])
            top5 = importances[:5]
            results["categories"][cat]["top_features"] = [
                {"name": n, "importance": round(float(v), 4)} for n, v in top5
            ]
            print(
                f"  {cat:<14}  " +
                "  ".join(f"{n}({v:.3f})" for n, v in top5)
            )
        except Exception as exc:
            print(f"  {cat:<14}  (failed to read: {exc})")

    if save:
        with open(METADATA_PATH, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nModels saved to {MODELS_DIR}/")
        print(f"Metadata saved to {METADATA_PATH}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="Train + save models")
    parser.add_argument("--eval", action="store_true", help="LOOCV only (no save)")
    parser.add_argument("--verbose", action="store_true", help="Print per-clip predictions")
    args = parser.parse_args()

    if not (args.train or args.eval):
        parser.print_help()
        return 1

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

    if not os.path.exists(FEATURE_CACHE):
        print(f"ERROR: cache not found at {FEATURE_CACHE}. "
              "Run scripts/calibrate_presentation_analysis.py --extract first.", file=sys.stderr)
        return 2

    with open(FEATURE_CACHE, "r", encoding="utf-8") as fh:
        cache = json.load(fh)

    names, X, y_by_cat = load_training_data()
    if not names:
        print("ERROR: no training data.", file=sys.stderr)
        return 3

    train_and_evaluate(names, X, y_by_cat, cache, verbose=args.verbose, save=args.train)
    return 0


if __name__ == "__main__":
    sys.exit(main())
