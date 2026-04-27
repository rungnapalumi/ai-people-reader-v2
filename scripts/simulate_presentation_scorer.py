#!/usr/bin/env python3
"""Sanity-check the presentation scorer on hand-estimated feature vectors
derived from the written comments in ``New Video Analysis_with comments.xlsx``.

This does NOT run MediaPipe — it feeds the scorer synthetic features built
to reflect the *verbal* descriptions (e.g. "Body swaying throughout") so we
can validate the rubric logic in isolation.

The real end-to-end accuracy is measured by
``calibrate_presentation_analysis.py --all`` on a box where MediaPipe runs.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.presentation_scorer import score_presentation  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic features: built to match the *descriptions* in the Excel.
#
# Each tuple is (first_impression, category_features, analysis_result) in
# the exact shape score_presentation() expects.
# ---------------------------------------------------------------------------

def fi(eye: float, up: float, st: float) -> Tuple[float, float, float]:
    return (eye, up, st)


def feats(
    *,
    hand_block: float = 0.35,
    hand_low: float = 0.15,
    hands_above: float = 0.20,
    hip_sway: float = 0.04,
    hip_advance: float = 0.0,
    distinct: int = 8,
) -> Dict[str, Any]:
    return {
        "hand_block_share": hand_block,
        "hand_low_share": hand_low,
        "hands_above_share": hands_above,
        "hip_sway_std": hip_sway,
        "hip_advance": hip_advance,
        "distinct_hand_shapes": distinct,
    }


def result(
    *,
    analyzed: int = 180,
    spreading: float = 0.0,
    enclosing: float = 0.0,
    gliding: float = 0.0,
    indirecting: float = 0.0,
    advancing: float = 0.0,
    pressing: float = 0.0,
    punching: float = 0.0,
    directing: float = 0.0,
    retreating: float = 0.0,
    flicking: float = 0.0,
    dabbing: float = 0.0,
) -> Dict[str, Any]:
    counts = {
        "Spreading": int(spreading * analyzed),
        "Enclosing": int(enclosing * analyzed),
        "Gliding": int(gliding * analyzed),
        "Indirecting": int(indirecting * analyzed),
        "Advancing": int(advancing * analyzed),
        "Pressing": int(pressing * analyzed),
        "Punching": int(punching * analyzed),
        "Directing": int(directing * analyzed),
        "Retreating": int(retreating * analyzed),
        "Flicking": int(flicking * analyzed),
        "Dabbing": int(dabbing * analyzed),
    }
    return {"analyzed_frames": analyzed, "effort_counts": counts}


# name -> (first_impression, category_features, result, ground_truth dict)
CLIPS: List[Tuple[str, Tuple[float, float, float], Dict[str, Any], Dict[str, Any], Dict[str, str]]] = [
    # --- Lea: H/L/H/M/L/L/L  — swaying, wide stance, blocking, few shapes
    (
        "Lea",
        fi(72, 55, 70),  # upright_pct ok but sway dominates → should be L
        feats(hand_block=0.55, hand_low=0.10, hands_above=0.15, hip_sway=0.075, distinct=6),
        result(spreading=0.14, enclosing=0.10, directing=0.08),
        {"eye_contact": "High", "uprightness": "Low", "stance": "High",
         "engaging": "Moderate", "adaptability": "Low", "confidence": "Low", "authority": "Low"},
    ),
    # --- Payu: H/H/M/H/H/M/M — upright, gentle sway, high variety spread+enclose
    (
        "Payu",
        fi(80, 78, 48),
        feats(hand_block=0.25, hand_low=0.10, hands_above=0.35, hip_sway=0.042, distinct=14),
        result(spreading=0.24, enclosing=0.20, gliding=0.06, directing=0.10, pressing=0.05),
        {"eye_contact": "High", "uprightness": "High", "stance": "Moderate",
         "engaging": "High", "adaptability": "High", "confidence": "Moderate", "authority": "Moderate"},
    ),
    # --- Sawitree: H/L/M/H/L/L/L — never upright, walks forward, low variety
    (
        "Sawitree",
        fi(75, 40, 45),
        feats(hand_block=0.30, hand_low=0.15, hands_above=0.30, hip_sway=0.070,
              hip_advance=0.06, distinct=5),
        result(spreading=0.28, enclosing=0.18, advancing=0.12),
        {"eye_contact": "High", "uprightness": "Low", "stance": "Moderate",
         "engaging": "High", "adaptability": "Low", "confidence": "Low", "authority": "Low"},
    ),
    # --- Sarinee: H/H/L/L/L/L/L — upright, feet together, nothing
    (
        "Sarinee",
        fi(78, 75, 20),
        feats(hand_block=0.45, hand_low=0.35, hands_above=0.10, hip_sway=0.038, distinct=4),
        result(spreading=0.03, enclosing=0.02, directing=0.05),
        {"eye_contact": "High", "uprightness": "High", "stance": "Low",
         "engaging": "Low", "adaptability": "Low", "confidence": "Low", "authority": "Low"},
    ),
    # --- Lisa1: H/H/M/M/L/H/H — extremely upright, weight-shifts, moderate gesture
    (
        "Lisa1",
        fi(82, 85, 48),
        feats(hand_block=0.30, hand_low=0.10, hands_above=0.20, hip_sway=0.028, distinct=6),
        result(spreading=0.14, enclosing=0.10, directing=0.15, pressing=0.06),
        {"eye_contact": "High", "uprightness": "High", "stance": "Moderate",
         "engaging": "Moderate", "adaptability": "Low", "confidence": "High", "authority": "High"},
    ),
    # --- Lisa 2: H/M/H/M/L/H/H — moderate upright, wide stance throughout
    (
        "Lisa2",
        fi(80, 62, 66),
        feats(hand_block=0.32, hand_low=0.10, hands_above=0.20, hip_sway=0.032, distinct=7),
        result(spreading=0.14, enclosing=0.10, directing=0.18, pressing=0.08),
        {"eye_contact": "High", "uprightness": "Moderate", "stance": "High",
         "engaging": "Moderate", "adaptability": "Low", "confidence": "High", "authority": "High"},
    ),
    # --- Chutima: H/M/M/M/L/M/M — mostly upright, weight-shifts
    (
        "Chutima",
        fi(78, 58, 45),
        feats(hand_block=0.35, hand_low=0.20, hands_above=0.20, hip_sway=0.045, hip_advance=0.03, distinct=7),
        result(spreading=0.16, enclosing=0.08, advancing=0.06, directing=0.10),
        {"eye_contact": "High", "uprightness": "Moderate", "stance": "Moderate",
         "engaging": "Moderate", "adaptability": "Low", "confidence": "Moderate", "authority": "Moderate"},
    ),
    # --- Aon: H/H/L/L/L/L/L — feet together, no spreading
    (
        "Aon",
        fi(80, 75, 15),
        feats(hand_block=0.50, hand_low=0.25, hands_above=0.10, hip_sway=0.030, distinct=5),
        result(spreading=0.02, enclosing=0.03, directing=0.08),
        {"eye_contact": "High", "uprightness": "High", "stance": "Low",
         "engaging": "Low", "adaptability": "Low", "confidence": "Low", "authority": "Low"},
    ),
    # --- Ann: M/L/L/M/M/L/L — eye low, swaying, feet together, moderate gesture
    (
        "Ann",
        fi(50, 42, 22),
        feats(hand_block=0.40, hand_low=0.20, hands_above=0.20, hip_sway=0.072, distinct=9),
        result(spreading=0.14, enclosing=0.10, directing=0.06, gliding=0.04, indirecting=0.04, pressing=0.03),
        {"eye_contact": "Moderate", "uprightness": "Low", "stance": "Low",
         "engaging": "Moderate", "adaptability": "Moderate", "confidence": "Low", "authority": "Low"},
    ),
    # --- Ches: H/H/H/M/L/H/H — extremely upright, same wide stance
    (
        "Ches",
        fi(80, 85, 70),
        feats(hand_block=0.30, hand_low=0.10, hands_above=0.20, hip_sway=0.028, distinct=6),
        result(spreading=0.16, enclosing=0.12, directing=0.18, pressing=0.06),
        {"eye_contact": "High", "uprightness": "High", "stance": "High",
         "engaging": "Moderate", "adaptability": "Low", "confidence": "High", "authority": "High"},
    ),
    # --- Candy: H/L/L/M/L/L/L — swaying, no stance
    (
        "Candy",
        fi(75, 40, 18),
        feats(hand_block=0.35, hand_low=0.20, hands_above=0.15, hip_sway=0.075, distinct=5),
        result(spreading=0.14, enclosing=0.10, directing=0.06),
        {"eye_contact": "High", "uprightness": "Low", "stance": "Low",
         "engaging": "Moderate", "adaptability": "Low", "confidence": "Low", "authority": "Low"},
    ),
    # --- Aom1: H/H/H/M/M/H/H — upright, wide stance, moderate gesture
    (
        "Aom1",
        fi(78, 78, 62),
        feats(hand_block=0.28, hand_low=0.10, hands_above=0.22, hip_sway=0.030, distinct=9),
        result(spreading=0.15, enclosing=0.10, directing=0.16, pressing=0.06),
        {"eye_contact": "High", "uprightness": "High", "stance": "High",
         "engaging": "Moderate", "adaptability": "Moderate", "confidence": "High", "authority": "High"},
    ),
    # --- Aom 2: H/M/M/H/H/M/M — moderate upright, high variety
    (
        "Aom2",
        fi(80, 60, 48),
        feats(hand_block=0.25, hand_low=0.10, hands_above=0.30, hip_sway=0.042, distinct=14),
        result(spreading=0.22, enclosing=0.18, gliding=0.06, directing=0.10, pressing=0.04),
        {"eye_contact": "High", "uprightness": "Moderate", "stance": "Moderate",
         "engaging": "High", "adaptability": "High", "confidence": "Moderate", "authority": "Moderate"},
    ),
    # --- Andre1: H/H/M/H/M/M/L — upright, high gesture, but weak drive (Auth downgrade)
    (
        "Andre1",
        fi(80, 72, 45),
        feats(hand_block=0.25, hand_low=0.35, hands_above=0.30, hip_sway=0.038, distinct=10),
        result(spreading=0.22, enclosing=0.14, gliding=0.04, directing=0.03),
        {"eye_contact": "High", "uprightness": "High", "stance": "Moderate",
         "engaging": "High", "adaptability": "Moderate", "confidence": "Moderate", "authority": "Low"},
    ),
    # --- Andre2: same as Andre1
    (
        "Andre2",
        fi(80, 72, 45),
        feats(hand_block=0.25, hand_low=0.35, hands_above=0.30, hip_sway=0.038, distinct=10),
        result(spreading=0.22, enclosing=0.14, gliding=0.04, directing=0.03),
        {"eye_contact": "High", "uprightness": "High", "stance": "Moderate",
         "engaging": "High", "adaptability": "Moderate", "confidence": "Moderate", "authority": "Low"},
    ),
    # --- Meiji1: H/H/H/L/L/H/H — extremely upright, wide stance, no gesture
    (
        "Meiji1",
        fi(80, 82, 68),
        feats(hand_block=0.40, hand_low=0.30, hands_above=0.10, hip_sway=0.028, distinct=5),
        result(spreading=0.04, enclosing=0.04, directing=0.18, pressing=0.08),
        {"eye_contact": "High", "uprightness": "High", "stance": "High",
         "engaging": "Low", "adaptability": "Low", "confidence": "High", "authority": "High"},
    ),
    # --- Meiji2: M/M/M/M/L/M/M
    (
        "Meiji2",
        fi(52, 58, 45),
        feats(hand_block=0.35, hand_low=0.20, hands_above=0.20, hip_sway=0.048, distinct=6),
        result(spreading=0.12, enclosing=0.08, directing=0.10, pressing=0.04),
        {"eye_contact": "Moderate", "uprightness": "Moderate", "stance": "Moderate",
         "engaging": "Moderate", "adaptability": "Low", "confidence": "Moderate", "authority": "Moderate"},
    ),
    # --- Meiji3: H/M/M/H/M/M/M
    (
        "Meiji3",
        fi(78, 60, 46),
        feats(hand_block=0.25, hand_low=0.15, hands_above=0.28, hip_sway=0.042, distinct=10),
        result(spreading=0.22, enclosing=0.16, gliding=0.05, directing=0.08, pressing=0.04),
        {"eye_contact": "High", "uprightness": "Moderate", "stance": "Moderate",
         "engaging": "High", "adaptability": "Moderate", "confidence": "Moderate", "authority": "Moderate"},
    ),
]


CATEGORIES = [
    "eye_contact", "uprightness", "stance",
    "engaging", "adaptability", "confidence", "authority",
]


def _letter(band: str) -> str:
    return {"High": "H", "Moderate": "M", "Low": "L"}.get(band, "-")


def _adj(t: str, p: str) -> bool:
    order = {"Low": 0, "Moderate": 1, "High": 2}
    return (t in order and p in order) and abs(order[t] - order[p]) <= 1


def main() -> int:
    totals = {c: {"hit": 0, "adj": 0, "n": 0} for c in CATEGORIES}
    rows = []
    for name, first_imp, cat_feats, analysis_res, truth in CLIPS:
        pred = score_presentation(first_imp, cat_feats, analysis_res)
        row = {"name": name}
        for c in CATEGORIES:
            t = truth[c]
            p = pred[c]
            row[f"t_{c}"] = t
            row[f"p_{c}"] = p
            totals[c]["n"] += 1
            if t == p:
                totals[c]["hit"] += 1
            if _adj(t, p):
                totals[c]["adj"] += 1
        rows.append(row)

    print()
    print("Name         | " + " | ".join(f"{c[:3]:<7}" for c in CATEGORIES))
    print("-" * (14 + len(CATEGORIES) * 10))
    for row in rows:
        cells = []
        for c in CATEGORIES:
            t = row[f"t_{c}"]
            p = row[f"p_{c}"]
            flag = "✓" if t == p else ("~" if _adj(t, p) else "✗")
            cells.append(f"{_letter(t)}/{_letter(p)} {flag}")
        print(f"{row['name']:<12} | " + " | ".join(f"{c:<7}" for c in cells))

    print()
    print(f"{'Category':<15} {'Exact':>10} {'Adj(±1)':>10}")
    print("-" * 40)
    total_hit = total_adj = total_n = 0
    for c in CATEGORIES:
        t = totals[c]
        n = t["n"]
        h = t["hit"]
        a = t["adj"]
        total_hit += h
        total_adj += a
        total_n += n
        print(f"{c:<15} {h}/{n} ({h/n*100:4.0f}%)  {a}/{n} ({a/n*100:4.0f}%)")
    print("-" * 40)
    print(
        f"{'TOTAL':<15} {total_hit}/{total_n} ({total_hit/total_n*100:4.0f}%)  "
        f"{total_adj}/{total_n} ({total_adj/total_n*100:4.0f}%)"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
