# movement_type_classifier.py
# -*- coding: utf-8 -*-
#
# People Reader / movement-type classification for this repo:
#   - Templates: TYPE_TEMPLATES + people_reader_seven rubric, 7-dim match, narratives.
#   - Production report worker: src/report_worker.py (loads src/report_core.py for pose features).
#   - Optional: people_reader_job.apply_movement_type_classification (same logic; for scripts/tests).
#
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# =========================================================
# Data structures
# =========================================================

@dataclass
class TypeTemplate:
    type_id: str
    name: str
    summary: str
    expected: Dict[str, Tuple[float, float]]
    weights: Dict[str, float]
    traits: Dict[str, str]
    # People Reader 7-dim rubric: eye, stance, upright, engaging, authority, confidence, adaptability
    people_reader_seven: Tuple[str, str, str, str, str, str, str]


# =========================================================
# 6 movement type templates
# Values are normalized to 0.0 - 1.0
# Expected ranges calibrated from project reference clips "Type 1.mov" … "Type 6.mov"
# (MediaPipe pose + report_core.extract_movement_type_frame_features_from_video, sample_every_n=3, max_frames=300).
# Note: weight_shift_raw is uniformly low across these six files in the current feature pipeline;
# discrimination leans on engagement_score, gesture_variation_score, stance_width_score, uprightness.
# People Reader matching/report uses `people_reader_seven` (product rubric); `expected` stays for weighted distance.
# =========================================================

TYPE_TEMPLATES: Dict[str, TypeTemplate] = {
    "type_1": TypeTemplate(
        type_id="type_1",
        name="Type 1 (Khun K)",
        summary="Stable, upright, controlled, confident and authoritative with low adaptability.",
        expected={
            "eye_contact": (0.90, 1.00),
            "uprightness": (0.86, 1.00),
            "stance_width_score": (0.35, 0.55),
            "weight_shift_score": (0.89, 1.00),   # high score = low weight shift (stable)
            "engagement_score": (0.00, 0.17),
            "gesture_variation_score": (0.00, 0.14),
            "chest_open_score": (0.18, 0.38),
            "rotation_control_score": (0.90, 1.00),
        },
        weights={
            "eye_contact": 1.2,
            "uprightness": 1.4,
            "stance_width_score": 1.2,
            "weight_shift_score": 1.4,
            "engagement_score": 0.9,
            "gesture_variation_score": 0.7,
            "chest_open_score": 0.8,
            "rotation_control_score": 1.0,
        },
        traits={
            "confidence": "high",
            "authority": "high",
            "adaptability": "low",
        },
        people_reader_seven=(
            "high",
            "high",
            "high",
            "moderate",
            "high",
            "high",
            "low",
        ),
    ),
    "type_2": TypeTemplate(
        type_id="type_2",
        name="Type 2 (Irene)",
        summary="Strong eye contact and posture, but lower engagement due to chest blocking; still confident and authoritative.",
        expected={
            "eye_contact": (0.90, 1.00),
            "uprightness": (0.78, 0.90),
            "stance_width_score": (0.40, 0.54),
            "weight_shift_score": (0.89, 1.00),
            "engagement_score": (0.10, 0.24),
            "gesture_variation_score": (0.01, 0.16),
            "chest_blocking_score": (0.00, 0.22),  # pipeline metric on reference clip
            "rotation_control_score": (0.90, 1.00),
        },
        weights={
            "eye_contact": 1.2,
            "uprightness": 1.2,
            "stance_width_score": 1.35,
            "weight_shift_score": 1.2,
            "engagement_score": 1.0,
            "gesture_variation_score": 0.8,
            "chest_blocking_score": 1.3,
            "rotation_control_score": 0.8,
        },
        traits={
            "confidence": "high",
            "authority": "high",
            "adaptability": "low",
        },
        people_reader_seven=(
            "high",
            "high",
            "high",
            "low",
            "high",
            "high",
            "low",
        ),
    ),
    "type_3": TypeTemplate(
        type_id="type_3",
        name="Type 3 (Khun Hongyok)",
        summary="Good eye contact but weaker stance, moderate uprightness with upper-body rotation, low confidence and authority.",
        expected={
            "eye_contact": (0.90, 1.00),
            "uprightness": (0.82, 1.00),
            "stance_width_score": (0.20, 0.40),
            "weight_shift_score": (0.89, 1.00),
            "engagement_score": (0.00, 0.19),
            "gesture_variation_score": (0.00, 0.16),
            "rotation_control_score": (0.90, 1.00),
        },
        weights={
            "eye_contact": 1.2,
            "uprightness": 1.0,
            "stance_width_score": 1.4,
            "weight_shift_score": 0.8,
            "engagement_score": 0.9,
            "gesture_variation_score": 0.8,
            "rotation_control_score": 1.2,
        },
        traits={
            "confidence": "low",
            "authority": "low",
            "adaptability": "low",
        },
        people_reader_seven=(
            "high",
            "low",
            "moderate",
            "moderate",
            "low",
            "low",
            "low",
        ),
    ),
    "type_4": TypeTemplate(
        type_id="type_4",
        name="Type 4 (Boon)",
        summary="Moderate eye contact and posture, clear weight shifting and varied engagement; highly adaptable but lower confidence and authority.",
        expected={
            "eye_contact": (0.90, 1.00),
            "uprightness": (0.82, 1.00),
            "stance_width_score": (0.38, 0.58),
            "weight_shift_raw": (0.00, 0.12),
            # Narrow vs type_5: reference Type 4 ~0.33 engagement, ~0.21 gesture variation (Type 5 ~0.27 / ~0.17).
            "engagement_score": (0.30, 0.40),
            "gesture_variation_score": (0.185, 0.24),
            "rotation_raw": (0.00, 0.11),
        },
        weights={
            "eye_contact": 1.0,
            "uprightness": 1.0,
            "stance_width_score": 0.8,
            "weight_shift_raw": 1.3,
            "engagement_score": 1.25,
            "gesture_variation_score": 1.45,
            "rotation_raw": 1.0,
        },
        traits={
            "confidence": "low",
            "authority": "low",
            "adaptability": "high",
        },
        people_reader_seven=(
            "moderate",
            "moderate",
            "moderate",
            "moderate",
            "low",
            "low",
            "high",
        ),
    ),
    "type_5": TypeTemplate(
        type_id="type_5",
        name="Type 5 (Elisha)",
        summary="High eye contact and engagement but unstable posture, frequent rotation and leg shuffling; high adaptability with low confidence and authority.",
        expected={
            "eye_contact": (0.89, 1.00),
            "uprightness": (0.86, 0.94),
            "stance_width_score": (0.38, 0.52),
            "weight_shift_raw": (0.00, 0.11),
            "engagement_score": (0.24, 0.30),
            "gesture_variation_score": (0.14, 0.18),
            "rotation_raw": (0.00, 0.10),
        },
        weights={
            "eye_contact": 1.1,
            "uprightness": 1.0,
            "stance_width_score": 1.2,
            "weight_shift_raw": 1.0,
            "engagement_score": 1.5,
            "gesture_variation_score": 1.5,
            "rotation_raw": 1.2,
        },
        traits={
            "confidence": "low",
            "authority": "low",
            "adaptability": "high",
        },
        people_reader_seven=(
            "high",
            "low",
            "moderate",
            "high",
            "low",
            "low",
            "high",
        ),
    ),
    "type_6": TypeTemplate(
        type_id="type_6",
        name="Type 6 (Alisa)",
        summary="Strong eye contact, upright posture, clear stance with some weight shifts, high engagement and adaptability, high confidence and authority.",
        expected={
            "eye_contact": (0.90, 1.00),
            "uprightness": (0.90, 0.98),
            "stance_width_score": (0.54, 0.66),
            "weight_shift_raw": (0.00, 0.11),
            "engagement_score": (0.18, 0.30),
            "gesture_variation_score": (0.12, 0.22),
            "rotation_control_score": (0.90, 1.00),
        },
        weights={
            "eye_contact": 1.2,
            "uprightness": 1.25,
            "stance_width_score": 1.45,
            "weight_shift_raw": 1.0,
            "engagement_score": 1.15,
            "gesture_variation_score": 1.2,
            "rotation_control_score": 1.0,
        },
        traits={
            "confidence": "high",
            "authority": "high",
            "adaptability": "high",
        },
        people_reader_seven=(
            "high",
            "high",
            "high",
            "high",
            "high",
            "high",
            "high",
        ),
    ),
}


# Optional overrides for `expected` ranges (e.g. tests or future tooling).
# Report worker clears this before each job so classification uses TYPE_TEMPLATES only.
# Keys: type_id -> feature_name -> (low, high)
_EXPECTED_RANGE_OVERRIDES: Dict[str, Dict[str, Tuple[float, float]]] = {}


def clear_expected_range_overrides() -> None:
    _EXPECTED_RANGE_OVERRIDES.clear()


def apply_calibration_json(payload: Any) -> None:
    """
    Merge calibration from JSON (e.g. hand-edited or tests). Report worker does not load this from S3.

    Supported shapes:
      { "types": { "type_1": { "expected": { "eye_contact": [0.7, 0.92], ... } } } }
      { "type_1": { "expected": { ... } }, ... }
    """
    clear_expected_range_overrides()
    if not isinstance(payload, dict):
        return
    root = payload.get("types") if isinstance(payload.get("types"), dict) else payload
    if not isinstance(root, dict):
        return
    for tid, block in root.items():
        tid_s = str(tid).strip()
        if tid_s not in TYPE_TEMPLATES:
            continue
        if not isinstance(block, dict):
            continue
        exp = block.get("expected")
        if not isinstance(exp, dict):
            continue
        parsed: Dict[str, Tuple[float, float]] = {}
        for fk, pair in exp.items():
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                lo, hi = float(pair[0]), float(pair[1])
                lo, hi = clamp01(lo), clamp01(hi)
                if lo > hi:
                    lo, hi = hi, lo
                parsed[str(fk)] = (lo, hi)
        if parsed:
            _EXPECTED_RANGE_OVERRIDES[tid_s] = parsed


def export_calibration_json_obj() -> Dict[str, Any]:
    """Current overrides suitable for saving to S3 (wrapped for version field)."""
    types_out: Dict[str, Any] = {}
    for tid, exp in _EXPECTED_RANGE_OVERRIDES.items():
        types_out[tid] = {
            "expected": {k: [round(v[0], 4), round(v[1], 4)] for k, v in exp.items()},
        }
    return {
        "version": 1,
        "types": types_out,
    }


def effective_expected_for_type(
    type_id: str,
    tpl: TypeTemplate,
    session_overrides: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
) -> Dict[str, Tuple[float, float]]:
    base = dict(tpl.expected)
    ov = _EXPECTED_RANGE_OVERRIDES.get(type_id)
    if ov:
        base.update(ov)
    if session_overrides:
        extra = session_overrides.get(type_id)
        if extra:
            base.update(extra)
    return base


def suggested_expected_from_reference_video_summary(
    summary_features: Dict[str, float],
    template: TypeTemplate,
    margin: float = 0.08,
    min_band: float = 0.06,
) -> Dict[str, Tuple[float, float]]:
    """
    From a reference recording for this type, propose (low, high) per template feature
    as value ± margin, clamped to [0,1], with a minimum band width.
    """
    m = max(0.02, min(0.30, float(margin)))
    bw = max(0.04, min(0.25, float(min_band)))
    out: Dict[str, Tuple[float, float]] = {}
    for feat_name in template.expected.keys():
        v = float(summary_features.get(feat_name, 0.0))
        lo = clamp01(v - m)
        hi = clamp01(v + m)
        if hi - lo < bw:
            mid = clamp01((lo + hi) / 2.0)
            lo = clamp01(mid - bw / 2.0)
            hi = clamp01(mid + bw / 2.0)
        if lo > hi:
            lo, hi = hi, lo
        out[str(feat_name)] = (round(lo, 4), round(hi, 4))
    return out


# =========================================================
# Helper functions
# =========================================================

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def safe_mean(values: List[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return default
    return sum(vals) / len(vals)


def safe_std(values: List[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return default
    try:
        return statistics.pstdev(vals)
    except Exception:
        return default


def normalize_linear(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clamp01((value - low) / (high - low))


def score_feature_against_range(value: float, low: float, high: float) -> float:
    """
    1.0 if inside range, then falls off linearly outside range.
    """
    value = float(value)

    if low <= value <= high:
        return 1.0

    width = max(0.10, high - low)

    if value < low:
        dist = low - value
    else:
        dist = value - high

    return clamp01(1.0 - (dist / width))


# =========================================================
# Summary feature builder
# report_worker can pass in either:
# 1) ready-made summary_features
# 2) frame_features time series
# =========================================================

def build_summary_features_from_timeseries(frame_features: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Expected optional keys in frame_features:
      eye_contact
      uprightness
      stance_width_ratio
      weight_shift
      gesture_energy
      gesture_variation
      chest_blocking
      chest_open
      rotation

    Returns normalized summary features.
    """

    eye_contact = clamp01(safe_mean(frame_features.get("eye_contact", []), 0.0))
    uprightness = clamp01(safe_mean(frame_features.get("uprightness", []), 0.0))

    stance_width_raw = safe_mean(frame_features.get("stance_width_ratio", []), 0.0)
    # shoulder-width-like stance around ~1.0 should score higher
    # 0.0-0.35 = weak, 0.35-0.75 = usable, 0.75-1.25 = strong, >1.5 maybe too wide
    if stance_width_raw <= 0.35:
        stance_width_score = normalize_linear(stance_width_raw, 0.0, 0.35) * 0.3
    elif stance_width_raw <= 1.25:
        stance_width_score = 0.3 + 0.7 * normalize_linear(stance_width_raw, 0.35, 1.25)
    else:
        stance_width_score = clamp01(1.0 - normalize_linear(stance_width_raw, 1.25, 2.0) * 0.4)

    weight_shift_raw = clamp01(safe_mean(frame_features.get("weight_shift", []), 0.0))
    weight_shift_score = clamp01(1.0 - weight_shift_raw)  # high score = stable

    gesture_energy_raw = clamp01(safe_mean(frame_features.get("gesture_energy", []), 0.0))
    gesture_variation_score = clamp01(safe_mean(frame_features.get("gesture_variation", []), 0.0))

    chest_blocking_score = clamp01(safe_mean(frame_features.get("chest_blocking", []), 0.0))
    chest_open_score = clamp01(safe_mean(frame_features.get("chest_open", []), 0.0))

    rotation_raw = clamp01(safe_mean(frame_features.get("rotation", []), 0.0))
    rotation_control_score = clamp01(1.0 - rotation_raw)

    # Engagement is a blend. Tune later.
    engagement_score = clamp01(
        0.45 * gesture_energy_raw +
        0.35 * gesture_variation_score +
        0.20 * chest_open_score -
        0.25 * chest_blocking_score
    )

    return {
        "eye_contact": eye_contact,
        "uprightness": uprightness,
        "stance_width_raw": clamp01(stance_width_raw),   # if already normalized upstream
        "stance_width_score": stance_width_score,
        "weight_shift_raw": weight_shift_raw,
        "weight_shift_score": weight_shift_score,
        "gesture_energy_raw": gesture_energy_raw,
        "gesture_variation_score": gesture_variation_score,
        "chest_blocking_score": chest_blocking_score,
        "chest_open_score": chest_open_score,
        "rotation_raw": rotation_raw,
        "rotation_control_score": rotation_control_score,
        "engagement_score": engagement_score,
    }


# =========================================================
# Classification
# =========================================================

def classify_movement_type(
    summary_features: Dict[str, float],
    session_overrides: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "best_match": {...},
        "alternatives": [...],
        "scores": [...],
        "summary_features": {...}
      }
    """
    scores: List[Dict[str, Any]] = []

    for type_id, tpl in TYPE_TEMPLATES.items():
        weighted_sum = 0.0
        total_weight = 0.0
        matched_features = []
        expected_map = effective_expected_for_type(type_id, tpl, session_overrides)

        for feat_name, feat_range in expected_map.items():
            value = float(summary_features.get(feat_name, 0.0))
            low, high = feat_range
            feat_score = score_feature_against_range(value, low, high)
            weight = float(tpl.weights.get(feat_name, 1.0))

            weighted_sum += feat_score * weight
            total_weight += weight

            matched_features.append({
                "feature": feat_name,
                "value": round(value, 4),
                "expected_range": [low, high],
                "score": round(feat_score, 4),
                "weight": round(weight, 4),
            })

        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        scores.append({
            "type_id": type_id,
            "type_name": tpl.name,
            "summary": tpl.summary,
            "score": round(final_score, 6),
            "traits": dict(tpl.traits),
            "matched_features": matched_features,
        })

    scores.sort(key=lambda x: x["score"], reverse=True)

    best_match = scores[0]
    alternatives = scores[1:3]

    best_score = best_match["score"]
    second_score = alternatives[0]["score"] if alternatives else 0.0

    # confidence = best score + separation
    confidence = clamp01((0.65 * best_score) + (0.35 * max(0.0, best_score - second_score)))

    return {
        "best_match": {
            **best_match,
            "confidence": round(confidence, 4),
            "confidence_pct": int(round(confidence * 100)),
        },
        "alternatives": alternatives,
        "scores": scores,
        "summary_features": summary_features,
    }


# =========================================================
# Narratives for report
# =========================================================

def _feature_phrase(summary_features: Dict[str, float]) -> List[str]:
    phrases: List[str] = []

    eye_contact = summary_features.get("eye_contact", 0.0)
    uprightness = summary_features.get("uprightness", 0.0)
    weight_shift_raw = summary_features.get("weight_shift_raw", 0.0)
    engagement_score = summary_features.get("engagement_score", 0.0)
    gesture_variation = summary_features.get("gesture_variation_score", 0.0)
    chest_blocking = summary_features.get("chest_blocking_score", 0.0)

    if eye_contact >= 0.70:
        phrases.append("strong eye contact")
    elif eye_contact >= 0.50:
        phrases.append("moderate eye contact")
    else:
        phrases.append("limited eye contact")

    if uprightness >= 0.75:
        phrases.append("upright posture")
    elif uprightness >= 0.45:
        phrases.append("moderately upright posture")
    else:
        phrases.append("less stable upright posture")

    if weight_shift_raw >= 0.70:
        phrases.append("frequent weight shifting")
    elif weight_shift_raw >= 0.30:
        phrases.append("some weight shifting")
    else:
        phrases.append("stable lower-body stance")

    if engagement_score >= 0.65:
        phrases.append("high engagement")
    elif engagement_score >= 0.40:
        phrases.append("moderate engagement")
    else:
        phrases.append("low engagement")

    if gesture_variation >= 0.60:
        phrases.append("varied hand and arm patterns")
    elif gesture_variation >= 0.25:
        phrases.append("repeated gesture patterns")
    else:
        phrases.append("limited gesture variety")

    if chest_blocking >= 0.65:
        phrases.append("noticeable chest-blocking behavior")

    return phrases


def generate_movement_type_narrative(classification: Dict[str, Any]) -> str:
    best = classification["best_match"]
    sf = classification["summary_features"]

    type_name = best["type_name"]
    confidence_pct = best["confidence_pct"]
    confidence = best["traits"].get("confidence", "unknown")
    authority = best["traits"].get("authority", "unknown")
    adaptability = best["traits"].get("adaptability", "unknown")
    phrases = _feature_phrase(sf)

    phrase_text = ", ".join(phrases[:5]) if phrases else "observable movement signals"

    return (
        f"The speaker most closely matches {type_name} with {confidence_pct}% confidence. "
        f"This pattern suggests {confidence} confidence, {authority} authority, and {adaptability} adaptability. "
        f"Key observable signals include {phrase_text}."
    )


def generate_movement_type_short_block(classification: Dict[str, Any]) -> Dict[str, Any]:
    best = classification["best_match"]
    alt = classification.get("alternatives", [])

    return {
        "movement_type": best["type_name"],
        "type_id": best["type_id"],
        "confidence_pct": best["confidence_pct"],
        "confidence": best["traits"].get("confidence"),
        "authority": best["traits"].get("authority"),
        "adaptability": best["traits"].get("adaptability"),
        "summary": best["summary"],
        "alternative_types": [
            {
                "type_id": x["type_id"],
                "type_name": x["type_name"],
                "score_pct": int(round(x["score"] * 100)),
            }
            for x in alt
        ],
    }


# =========================================================
# Public convenience API
# =========================================================

def analyze_movement_type(
    summary_features: Optional[Dict[str, float]] = None,
    frame_features: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Any]:
    """
    Main entry point.

    Use one of:
      analyze_movement_type(summary_features=...)
      analyze_movement_type(frame_features=...)

    Returns:
      {
        "summary_features": ...,
        "classification": ...,
        "report_block": ...,
        "narrative": ...
      }
    """
    if summary_features is None:
        summary_features = build_summary_features_from_timeseries(frame_features or {})

    classification = classify_movement_type(summary_features)
    narrative = generate_movement_type_narrative(classification)
    report_block = generate_movement_type_short_block(classification)

    return {
        "summary_features": classification["summary_features"],
        "classification": classification,
        "report_block": report_block,
        "narrative": narrative,
    }


# =========================================================
# People Reader (page 8): 7-dimension match vs all 6 TYPE_TEMPLATES
# Template side: each type’s `people_reader_seven` (product spreadsheet: low / moderate / high per dim).
# Video side: dims 1–4 = movement tertiles; dims 5–7 = high/low from movement composites (≥0.5 → high).
# Report category bars use the chosen type’s `people_reader_seven`. Weighted `classify_movement_type` still uses `expected`.
# =========================================================

PEOPLE_READER_SEVEN_DIM_LABELS_EN: Tuple[str, ...] = (
    "Eye contact",
    "Stance",
    "Uprightness",
    "Engaging",
    "Authority",
    "Confidence",
    "Adaptability",
)

PEOPLE_READER_SEVEN_DIM_LABELS_TH: Tuple[str, ...] = (
    "การสบตา",
    "ท่ายืน",
    "ความตั้งตรง",
    "การมีส่วนร่วม",
    "ความเป็นผู้นำและอำนาจ",
    "ความมั่นใจ",
    "ความยืดหยุ่นในการปรับตัว",
)


def people_reader_level_from_01(v: float) -> str:
    """Map a 0–1 value to low / moderate / high by equal tertiles."""
    x = clamp01(float(v))
    if x < 1.0 / 3.0:
        return "low"
    if x < 2.0 / 3.0:
        return "moderate"
    return "high"


def people_reader_high_low_from_signal01(v: float) -> str:
    """High vs low for video dims 5–7 (template rubric uses only high/low on those axes)."""
    return "high" if clamp01(float(v)) >= 0.5 else "low"


def normalize_people_reader_rubric_level(label: Any) -> str:
    """Normalize spreadsheet labels to low | moderate | high."""
    s = str(label or "").strip().lower()
    if s in ("low", "l"):
        return "low"
    if s in ("high", "h"):
        return "high"
    if s in ("moderate", "medium", "mod", "m", "med"):
        return "moderate"
    return "moderate"


def _video_authority_signal(sf: Dict[str, float]) -> float:
    up = clamp01(float(sf.get("uprightness", 0.0)))
    rot = clamp01(float(sf.get("rotation_control_score", 0.0)))
    stab = clamp01(float(sf.get("weight_shift_score", 0.0)))
    return clamp01(0.38 * up + 0.32 * rot + 0.30 * stab)


def _video_confidence_signal(sf: Dict[str, float]) -> float:
    ec = clamp01(float(sf.get("eye_contact", 0.0)))
    up = clamp01(float(sf.get("uprightness", 0.0)))
    gv = clamp01(float(sf.get("gesture_variation_score", 0.0)))
    return clamp01(0.40 * ec + 0.35 * up + 0.25 * gv)


def _video_adaptability_signal(sf: Dict[str, float]) -> float:
    gv = clamp01(float(sf.get("gesture_variation_score", 0.0)))
    eng = clamp01(float(sf.get("engagement_score", 0.0)))
    co = clamp01(float(sf.get("chest_open_score", 0.0)))
    return clamp01(0.50 * gv + 0.30 * eng + 0.20 * co)


def _mid_expected(exp: Dict[str, Tuple[float, float]], key: str, default: float = 0.5) -> float:
    pair = exp.get(key)
    if not pair or len(pair) < 2:
        return default
    lo, hi = float(pair[0]), float(pair[1])
    return (lo + hi) / 2.0


def video_seven_levels_people_reader(summary_features: Dict[str, float]) -> List[str]:
    """Movement-only: dims 1–4 tertiles; dims 5–7 high/low from composites (aligned with trait rubric)."""
    sf = summary_features
    return [
        people_reader_level_from_01(float(sf.get("eye_contact", 0.0))),
        people_reader_level_from_01(float(sf.get("stance_width_score", 0.0))),
        people_reader_level_from_01(float(sf.get("uprightness", 0.0))),
        people_reader_level_from_01(float(sf.get("engagement_score", 0.0))),
        people_reader_high_low_from_signal01(_video_authority_signal(sf)),
        people_reader_high_low_from_signal01(_video_confidence_signal(sf)),
        people_reader_high_low_from_signal01(_video_adaptability_signal(sf)),
    ]


def template_seven_levels_people_reader(
    tpl: TypeTemplate,
    session_overrides: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
) -> List[str]:
    """Template: explicit per-type rubric (`people_reader_seven`); session overrides affect `expected` only, not this."""
    _ = session_overrides
    return [normalize_people_reader_rubric_level(x) for x in tpl.people_reader_seven]


def people_reader_category_scales_from_template(
    tpl: TypeTemplate,
    session_overrides: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
) -> Dict[str, str]:
    """Report bars from `people_reader_seven` (dims 3–6 → engaging … adaptability)."""
    levels = template_seven_levels_people_reader(tpl, session_overrides=session_overrides)
    return {
        "engaging": levels[3],
        "authority": levels[4],
        "confidence": levels[5],
        "adaptability": levels[6],
    }


def people_reader_scale_to_category_score(scale: str) -> int:
    """Map scale label to 1–7 score for report bars (low→2, moderate→4, high→6)."""
    s = str(scale or "").strip().lower()
    if s == "low":
        return 2
    if s == "high":
        return 6
    return 4


def rank_people_reader_types_by_seven_match(
    summary_features: Dict[str, float],
    session_overrides: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
) -> List[Dict[str, Any]]:
    """
    Compare video vs each profile: movement tertiles (1–4) and composite high/low (5–7) vs template rubric
    `people_reader_seven` (no main analysis 1–7 scores).

    **Ranking (Auto mode):** primary = weighted `classify_movement_type` score (higher first);
    tie-break = 7-dimension match count, then `type_id`. This aligns reference clips with calibrated `expected` ranges
    while still using 7-dim agreement when scores are close.
    """
    v_levels = video_seven_levels_people_reader(summary_features)
    cls = classify_movement_type(summary_features, session_overrides=session_overrides)
    legacy_by_id = {str(x["type_id"]): float(x.get("score") or 0.0) for x in cls.get("scores") or []}

    ranked: List[Dict[str, Any]] = []
    for tid, tpl in TYPE_TEMPLATES.items():
        t_levels = template_seven_levels_people_reader(tpl, session_overrides=session_overrides)
        matches = sum(1 for a, b in zip(v_levels, t_levels) if a == b)
        ranked.append(
            {
                "type_id": tid,
                "type_name": tpl.name,
                "matches": int(matches),
                "match_pct": int(round(100.0 * float(matches) / 7.0)),
                "legacy_classifier_score": round(legacy_by_id.get(tid, 0.0), 6),
                "template_levels": t_levels,
            }
        )
    ranked.sort(
        key=lambda r: (-float(r["legacy_classifier_score"]), -int(r["matches"]), str(r["type_id"]))
    )
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
    return ranked


# =========================================================
# Simple result object (for job analysis dicts / external callers)
# =========================================================


@dataclass
class MovementTypeResult:
    """Compact view of `classify_movement_type` for wiring into analysis payloads."""

    type_code: str
    type_name: str
    score: float
    summary: str
    matched_rules: List[str]
    all_scores: List[Dict[str, Any]]


def classify_movement_type_result(
    summary_features: Dict[str, float],
    session_overrides: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
) -> MovementTypeResult:
    """
    Run `classify_movement_type` and return a flat result (similar to example integration snippets).

    - type_code: best-match type_id (e.g. type_3)
    - matched_rules: short strings per feature from the best match’s matched_features
    - all_scores: full ranked list from the classifier
    """
    data = classify_movement_type(summary_features, session_overrides=session_overrides)
    bm = data.get("best_match") or {}
    rules: List[str] = []
    for mf in bm.get("matched_features") or []:
        fn = str(mf.get("feature", ""))
        er = mf.get("expected_range") or [0, 0]
        rules.append(
            f"{fn} in [{float(er[0]):.3f},{float(er[1]):.3f}] score={mf.get('score')}"
        )
    return MovementTypeResult(
        type_code=str(bm.get("type_id", "")),
        type_name=str(bm.get("type_name", "")),
        score=float(bm.get("score") or 0.0),
        summary=str(bm.get("summary", "")),
        matched_rules=rules,
        all_scores=list(data.get("scores") or []),
    )