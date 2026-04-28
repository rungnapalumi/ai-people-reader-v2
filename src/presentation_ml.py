"""Presentation Analysis — ML scorer (RandomForest-based).

This module complements :mod:`src.presentation_scorer` (rule-based) with a
classifier-driven scorer. The ML path:

1. Takes the same ``(first_impression_pct, category_features, analysis_result)``
   triple the rule-based scorer uses.
2. Builds a fixed-length feature vector (see :data:`FEATURE_NAMES`).
3. Loads 7 pre-trained RandomForest classifiers (one per category) from the
   ``models/`` directory and returns per-category H/M/L bands.

If any model is missing or an exception occurs, callers should fall back to
the rule-based scorer.

Feature vector layout (stable across train / inference)
-------------------------------------------------------
The ordering below is the **canonical order** — training and inference MUST
produce vectors in the same order. Any changes require re-training.

    # First-impression primitives
    eye_contact_pct
    upright_pct
    stance_stability

    # Original category features
    hand_block_share
    hand_low_share
    hands_above_share
    hip_sway_std
    hip_advance
    distinct_hand_shapes

    # Effort shares (0..1)
    spreading_share
    enclosing_share
    gliding_share
    indirecting_share
    advancing_share
    pressing_share
    punching_share
    directing_share
    retreating_share
    dabbing_share
    flicking_share

    # Enriched presentation features (15 new)
    posture_uprightness
    torso_stability
    head_stability
    eye_direction_proxy
    shoulder_openness
    hand_openness
    gesture_range
    gesture_smoothness
    movement_intentionality
    hesitation_score
    rhythm_consistency
    energy_level
    center_presence
    body_sway                      # pass-through of hip_sway_std
    stance_stability_norm          # stance_stability / 100.0

Models are stored in ``models/presentation_{category}.joblib``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Canonical feature order. Shared between training and inference.
# --------------------------------------------------------------------------
FEATURE_NAMES: List[str] = [
    # First-impression primitives
    "eye_contact_pct",
    "upright_pct",
    "stance_stability",
    # Original category features
    "hand_block_share",
    "hand_low_share",
    "hands_above_share",
    "hip_sway_std",
    "hip_advance",
    "distinct_hand_shapes",
    # Effort shares (0..1)
    "spreading_share",
    "enclosing_share",
    "gliding_share",
    "indirecting_share",
    "advancing_share",
    "pressing_share",
    "punching_share",
    "directing_share",
    "retreating_share",
    "dabbing_share",
    "flicking_share",
    # Enriched features
    "posture_uprightness",
    "torso_stability",
    "head_stability",
    "eye_direction_proxy",
    "shoulder_openness",
    "hand_openness",
    "gesture_range",
    "gesture_smoothness",
    "movement_intentionality",
    "hesitation_score",
    "rhythm_consistency",
    "energy_level",
    "center_presence",
    "body_sway",
    "stance_stability_norm",
    # Holistic (face + hand) features
    "gaze_forward_ratio",
    "head_yaw_std",
    "head_pitch_mean",
    "face_detection_ratio",
    "left_hand_openness",
    "right_hand_openness",
    "hand_detection_ratio",
    "pointing_ratio",
    "finger_variation",
    # Audio features (librosa; language-agnostic)
    "audio_voiced_ratio",
    "audio_speech_ratio",
    "audio_pause_count_per_min",
    "audio_pause_mean_duration",
    "audio_longest_pause",
    "audio_pitch_mean",
    "audio_pitch_std",
    "audio_pitch_range",
    "audio_volume_mean",
    "audio_volume_std",
    "audio_volume_dynamic_range",
]

CATEGORIES: List[str] = [
    "eye_contact",
    "uprightness",
    "stance",
    "engaging",
    "adaptability",
    "confidence",
    "authority",
]

MODELS_DIR = os.getenv(
    "PRESENTATION_MODELS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"),
)


# --------------------------------------------------------------------------
# Feature vector construction
# --------------------------------------------------------------------------

def build_feature_vector(
    first_impression_pct: Tuple[float, float, float],
    category_features: Dict[str, Any],
    analysis_result: Dict[str, Any],
) -> List[float]:
    """Produce the canonical feature vector used for training and inference.

    Values are always returned in the order defined by :data:`FEATURE_NAMES`.
    Missing fields default to 0.0 so the vector is always the right length.
    """
    eye_pct, upright_pct, stance_stab = first_impression_pct
    cf = category_features or {}

    effort_counts: Dict[str, int] = dict(analysis_result.get("effort_counts") or {})
    analyzed = max(1, int(analysis_result.get("analyzed_frames") or 0))

    def _share(name: str) -> float:
        return float(effort_counts.get(name, 0)) / analyzed

    def _cf(name: str, default: float = 0.0) -> float:
        try:
            return float(cf.get(name, default))
        except (TypeError, ValueError):
            return default

    values: List[float] = [
        float(eye_pct),
        float(upright_pct),
        float(stance_stab),
        _cf("hand_block_share"),
        _cf("hand_low_share"),
        _cf("hands_above_share"),
        _cf("hip_sway_std"),
        _cf("hip_advance"),
        _cf("distinct_hand_shapes"),
        _share("Spreading"),
        _share("Enclosing"),
        _share("Gliding"),
        _share("Indirecting"),
        _share("Advancing"),
        _share("Pressing"),
        _share("Punching"),
        _share("Directing"),
        _share("Retreating"),
        _share("Dabbing"),
        _share("Flicking"),
        _cf("posture_uprightness"),
        _cf("torso_stability"),
        _cf("head_stability"),
        _cf("eye_direction_proxy"),
        _cf("shoulder_openness"),
        _cf("hand_openness"),
        _cf("gesture_range"),
        _cf("gesture_smoothness"),
        _cf("movement_intentionality"),
        _cf("hesitation_score"),
        _cf("rhythm_consistency"),
        _cf("energy_level"),
        _cf("center_presence"),
        _cf("body_sway", _cf("hip_sway_std")),
        float(stance_stab) / 100.0,
        _cf("gaze_forward_ratio"),
        _cf("head_yaw_std"),
        _cf("head_pitch_mean"),
        _cf("face_detection_ratio"),
        _cf("left_hand_openness"),
        _cf("right_hand_openness"),
        _cf("hand_detection_ratio"),
        _cf("pointing_ratio"),
        _cf("finger_variation"),
        _cf("audio_voiced_ratio"),
        _cf("audio_speech_ratio"),
        _cf("audio_pause_count_per_min"),
        _cf("audio_pause_mean_duration"),
        _cf("audio_longest_pause"),
        _cf("audio_pitch_mean"),
        _cf("audio_pitch_std"),
        _cf("audio_pitch_range"),
        _cf("audio_volume_mean"),
        _cf("audio_volume_std"),
        _cf("audio_volume_dynamic_range"),
    ]
    assert len(values) == len(FEATURE_NAMES), (
        f"feature vector length mismatch: {len(values)} vs {len(FEATURE_NAMES)}"
    )
    return values


# --------------------------------------------------------------------------
# Model loading (cached)
# --------------------------------------------------------------------------

_MODEL_CACHE: Dict[str, Any] = {}


def _model_path(category: str) -> str:
    return os.path.join(MODELS_DIR, f"presentation_{category}.joblib")


def load_models(force_reload: bool = False) -> Dict[str, Any]:
    """Load all 7 classifiers from disk. Cached in-process.

    Returns a dict {category: estimator}. If a model file is missing, that
    category is silently omitted from the dict (caller should fall back).
    """
    global _MODEL_CACHE
    if _MODEL_CACHE and not force_reload:
        return _MODEL_CACHE

    try:
        import joblib  # type: ignore
    except Exception as exc:
        logger.warning("[presentation_ml] joblib not available: %s", exc)
        return {}

    loaded: Dict[str, Any] = {}
    for cat in CATEGORIES:
        path = _model_path(cat)
        if not os.path.exists(path):
            logger.info("[presentation_ml] no model for %s at %s", cat, path)
            continue
        try:
            loaded[cat] = joblib.load(path)
        except Exception as exc:
            logger.warning("[presentation_ml] failed to load %s: %s", cat, exc)

    _MODEL_CACHE = loaded
    if loaded:
        logger.info(
            "[presentation_ml] loaded %d/%d models from %s",
            len(loaded), len(CATEGORIES), MODELS_DIR,
        )
    return loaded


def predict_bands(feature_vector: List[float]) -> Optional[Dict[str, str]]:
    """Run the 7 classifiers on a feature vector. Returns {cat: "High"/"Moderate"/"Low"}.

    Returns ``None`` if no models are loaded (caller should use rule-based).
    """
    models = load_models()
    if not models:
        return None

    out: Dict[str, str] = {}
    try:
        import numpy as np  # type: ignore
    except Exception:
        return None

    x = np.array([feature_vector], dtype=float)
    for cat, est in models.items():
        try:
            band = est.predict(x)[0]
            out[cat] = str(band)
        except Exception as exc:
            logger.warning("[presentation_ml] predict failed for %s: %s", cat, exc)

    return out or None


def predict_bands_with_proba(
    feature_vector: List[float],
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Same as :func:`predict_bands` but also returns class probabilities.

    Returns ``{category: {"band": str, "proba": {"Low": p, "Moderate": p, "High": p}}}``
    when available, or ``None`` if no models are loaded.
    """
    models = load_models()
    if not models:
        return None

    try:
        import numpy as np  # type: ignore
    except Exception:
        return None

    x = np.array([feature_vector], dtype=float)
    out: Dict[str, Dict[str, Any]] = {}
    for cat, est in models.items():
        try:
            band = str(est.predict(x)[0])
            entry: Dict[str, Any] = {"band": band}
            if hasattr(est, "predict_proba") and hasattr(est, "classes_"):
                proba = est.predict_proba(x)[0]
                entry["proba"] = {
                    str(c): float(p) for c, p in zip(est.classes_, proba)
                }
            out[cat] = entry
        except Exception as exc:
            logger.warning("[presentation_ml] predict failed for %s: %s", cat, exc)

    return out or None
