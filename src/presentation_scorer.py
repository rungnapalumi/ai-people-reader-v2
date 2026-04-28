"""
Presentation Analysis scorer (used by pages/9_Presentation_Analysis.py).

Design goals
------------
1. **Whole-video overview first**, *then* per-category scoring. All category
   decisions read the same shared "overview" dict of signals, so the rubric
   stays consistent across categories.
2. **Many independent criteria per category** (5-8 thresholds each), not a
   single score mapped through one band. Each criterion contributes a vote
   (+1 / -1) and the final band is the sum vs. two thresholds.
3. **Calibrated against the 18-clip ground truth** in
   `New Video Analysis_with comments.xlsx`. Every threshold has a one-line
   justification pointing to the clip(s) that drove it.
4. **Environment-variable overrides**. Every threshold has a
   `PRES_*` env var so tuning post-deploy does not require a redeploy.

The scorer consumes the raw output of
``src.report_core.analyze_video_mediapipe`` plus the already-computed
``FirstImpressionData``. It does **not** rerun pose estimation.

Output
------
A dict with keys:

- ``eye_contact``, ``uprightness``, ``stance`` — first-impression H/M/L.
- ``engaging``, ``adaptability``, ``confidence``, ``authority`` — category H/M/L.
- ``overview`` — the shared overview dict (for logging / report building).
- ``rationale`` — per-category breakdown of which criteria fired.

All bands are returned as the literal strings ``"High"``, ``"Moderate"``,
``"Low"`` so callers can drop them straight into the report template.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except Exception:
        return int(default)


def _band_from_votes(votes: int, high_at: int = 2, low_at: int = -1) -> str:
    """Aggregate ±1 criterion votes into H/M/L.

    ``votes >= high_at``  → High
    ``votes <= low_at``   → Low
    otherwise             → Moderate
    """
    if votes >= high_at:
        return "High"
    if votes <= low_at:
        return "Low"
    return "Moderate"


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Overview: compute all whole-video features up-front
# ---------------------------------------------------------------------------

@dataclass
class PresentationOverview:
    """All whole-video signals the scorer reads.

    Only field names that start with an underscore are derived; the rest come
    directly from the analyzer's output so they can be logged verbatim.
    """

    # First-impression (% or 0-100 stability)
    eye_contact_pct: float
    upright_pct: float
    stance_stability: float

    # Category features (fractions 0-1 / std)
    hand_block_share: float
    hand_low_share: float
    hands_above_share: float
    hip_sway_std: float
    hip_advance: float
    distinct_hand_shapes: int

    # Gesture / effort shares (fraction of analyzed frames)
    spreading_share: float
    enclosing_share: float
    gliding_share: float
    indirecting_share: float
    advancing_share: float
    pressing_share: float
    punching_share: float
    directing_share: float
    retreating_share: float

    # Derived composites
    gesture_share: float           # spreading + enclosing + gliding
    strong_effort_share: float     # pressing + punching + directing
    effort_variety: int            # number of efforts with share >= 3%

    # Misc
    analyzed_frames: int


def build_overview(
    first_impression_pct: Tuple[float, float, float],
    category_features: Dict[str, Any],
    analysis_result: Dict[str, Any],
) -> PresentationOverview:
    """Assemble a :class:`PresentationOverview` from analyzer outputs.

    Parameters
    ----------
    first_impression_pct : (eye_pct, upright_pct, stance_stability)
        Already computed by :func:`src.report_core.build_first_impression_from_video`.
    category_features : dict
        The ``category_features`` sub-dict from ``analyze_video_mediapipe``.
    analysis_result : dict
        The top-level return value of ``analyze_video_mediapipe`` — used for
        ``effort_counts`` / ``analyzed_frames``.
    """
    eye_pct, upright_pct, stance_stab = first_impression_pct

    cf = category_features or {}
    hand_block = float(cf.get("hand_block_share", 0.0))
    hand_low = float(cf.get("hand_low_share", 0.0))
    hands_above = float(cf.get("hands_above_share", 0.0))
    hip_sway = float(cf.get("hip_sway_std", 0.0))
    hip_adv = float(cf.get("hip_advance", 0.0))
    distinct = int(cf.get("distinct_hand_shapes", 0))

    effort_counts: Dict[str, int] = dict(analysis_result.get("effort_counts") or {})
    analyzed = max(1, int(analysis_result.get("analyzed_frames") or 0))

    def _share(name: str) -> float:
        return float(effort_counts.get(name, 0)) / analyzed

    spreading = _share("Spreading")
    enclosing = _share("Enclosing")
    gliding = _share("Gliding")
    indirecting = _share("Indirecting")
    advancing = _share("Advancing")
    pressing = _share("Pressing")
    punching = _share("Punching")
    directing = _share("Directing")
    retreating = _share("Retreating")

    gesture_share = spreading + enclosing + gliding
    strong_effort_share = pressing + punching + directing

    # Effort variety: how many distinct efforts appear with >= 3% frame share.
    variety_threshold = _env_float("PRES_EFFORT_VARIETY_MIN_SHARE", 0.03)
    effort_variety = sum(
        1
        for name in (
            "Spreading", "Enclosing", "Gliding", "Indirecting",
            "Advancing", "Pressing", "Punching", "Directing",
            "Retreating", "Flicking", "Dabbing",
        )
        if _share(name) >= variety_threshold
    )

    return PresentationOverview(
        eye_contact_pct=float(eye_pct),
        upright_pct=float(upright_pct),
        stance_stability=float(stance_stab),
        hand_block_share=hand_block,
        hand_low_share=hand_low,
        hands_above_share=hands_above,
        hip_sway_std=hip_sway,
        hip_advance=hip_adv,
        distinct_hand_shapes=distinct,
        spreading_share=spreading,
        enclosing_share=enclosing,
        gliding_share=gliding,
        indirecting_share=indirecting,
        advancing_share=advancing,
        pressing_share=pressing,
        punching_share=punching,
        directing_share=directing,
        retreating_share=retreating,
        gesture_share=gesture_share,
        strong_effort_share=strong_effort_share,
        effort_variety=effort_variety,
        analyzed_frames=analyzed,
    )


# ---------------------------------------------------------------------------
# Per-category scorers
# ---------------------------------------------------------------------------
#
# Each scorer returns (band, votes, reasons[]) where ``reasons`` is a list
# of "+kind" / "-kind" strings so we can log exactly which criteria fired
# for each clip. Thresholds are derived from the 18-clip Excel rubric.
# ---------------------------------------------------------------------------

def score_eye_contact(o: PresentationOverview) -> Tuple[str, List[str]]:
    """Eye Contact: 16/18 clips in the reference are H, two (Ann, Meiji2) M.

    Default H; only drop to M / L when eye_contact_pct is clearly low.
    """
    reasons: List[str] = []
    m_cut = _env_float("PRES_EYE_MODERATE_CUT", 55.0)
    l_cut = _env_float("PRES_EYE_LOW_CUT", 30.0)
    eye = o.eye_contact_pct
    if eye < l_cut:
        reasons.append(f"eye_pct<{l_cut}")
        return "Low", reasons
    if eye < m_cut:
        reasons.append(f"eye_pct<{m_cut}")
        return "Moderate", reasons
    reasons.append(f"eye_pct>={m_cut}")
    return "High", reasons


def score_uprightness(o: PresentationOverview) -> Tuple[str, List[str]]:
    """Uprightness: `upright_pct` + body sway (`hip_sway_std`).

    Ground-truth anchors:
      * Lea:  upright maybe ok but swaying throughout → L.
      * Lisa1/Ches:  extremely upright, barely swaying → H.
      * Chutima/Aom2/Meiji2/3:  moderately upright → M.
      * Ann/Candy:  swaying left-right throughout → L.
    """
    reasons: List[str] = []
    votes = 0

    up_high = _env_float("PRES_UPRIGHT_HIGH", 65.0)
    up_mod = _env_float("PRES_UPRIGHT_MOD", 45.0)
    sway_low = _env_float("PRES_UPRIGHT_SWAY_LOW", 0.040)
    sway_high = _env_float("PRES_UPRIGHT_SWAY_HIGH", 0.060)

    if o.upright_pct >= up_high:
        votes += 1
        reasons.append(f"+upright_pct>={up_high}")
    elif o.upright_pct < up_mod:
        votes -= 1
        reasons.append(f"-upright_pct<{up_mod}")

    if o.hip_sway_std <= sway_low:
        votes += 1
        reasons.append(f"+hip_sway<={sway_low}")
    elif o.hip_sway_std >= sway_high:
        votes -= 1
        reasons.append(f"-hip_sway>={sway_high}")

    if o.upright_pct >= 80.0 and o.hip_sway_std <= 0.035:
        votes += 1
        reasons.append("+extremely_upright_bonus")

    if o.hip_sway_std >= 0.075:
        votes -= 1
        reasons.append("-severe_sway_penalty")

    return _band_from_votes(votes), reasons


def score_stance(o: PresentationOverview) -> Tuple[str, List[str]]:
    """Stance: uses `stance_stability` which already bakes in feet-together
    cap (≤18) and wide-stable boost (+10).

    Ground-truth anchors:
      * Aon/Candy:  feet together throughout → L.
      * Ann:  feet together almost throughout → L.
      * Lea/Ches:  same wide stance throughout → H.
      * Lisa1/Payu/Chutima:  weight-shifts but stable → M.
    """
    reasons: List[str] = []
    votes = 0

    stance_high = _env_float("PRES_STANCE_HIGH", 60.0)
    stance_low = _env_float("PRES_STANCE_LOW", 30.0)

    if o.stance_stability >= stance_high:
        votes += 2
        reasons.append(f"+stance>={stance_high}")
    elif o.stance_stability <= stance_low:
        votes -= 2
        reasons.append(f"-stance<={stance_low}")

    # If sway is high even when stance_stability is middling, pull down
    # ("Ann":  stance ~ moderate number but feet-together visually → L).
    if o.stance_stability < 45.0 and o.hip_sway_std > 0.06:
        votes -= 1
        reasons.append("-unstable_mid_stance")

    return _band_from_votes(votes, high_at=2, low_at=-1), reasons


def score_engaging(o: PresentationOverview) -> Tuple[str, List[str]]:
    """Engagement & Connecting: spreading + enclosing + advance + open posture.

    Ground-truth anchors:
      * Payu/Sawitree/Aom2/Andre1/Andre2/Meiji3:  H — "spreading + enclosing".
      * Aon/Sarinee/Meiji1:  L — no spreading at all.
      * Most others:  M — moderate spreading/enclosing.
    """
    reasons: List[str] = []
    votes = 0

    g_high = _env_float("PRES_ENG_GESTURE_HIGH", 0.22)
    g_low = _env_float("PRES_ENG_GESTURE_LOW", 0.08)
    block_high = _env_float("PRES_ENG_BLOCK_HIGH", 0.70)
    block_low = _env_float("PRES_ENG_BLOCK_LOW", 0.35)
    hands_up_high = _env_float("PRES_ENG_HANDS_UP_HIGH", 0.25)

    # Gesture share (spreading + enclosing + gliding).
    if o.gesture_share >= g_high:
        votes += 1
        reasons.append(f"+gesture_share>={g_high}")
    elif o.gesture_share <= g_low:
        votes -= 1
        reasons.append(f"-gesture_share<={g_low}")

    # Hands blocking body torpedoes engagement.
    if o.hand_block_share >= block_high:
        votes -= 1
        reasons.append(f"-hand_block>={block_high}")
    elif o.hand_block_share <= block_low:
        votes += 1
        reasons.append(f"+hand_block<={block_low}")

    # Arms up expands the comms zone ("Payu").
    if o.hands_above_share >= hands_up_high:
        votes += 1
        reasons.append(f"+hands_above>={hands_up_high}")

    # Walking toward audience ("Sawitree").
    adv_thresh = _env_float("PRES_ENG_ADVANCE", 0.02)
    if o.hip_advance >= adv_thresh:
        votes += 1
        reasons.append(f"+hip_advance>={adv_thresh}")

    # Diversity of gestures ("Payu / Aom2 / Andre").
    distinct_high = _env_int("PRES_ENG_DISTINCT_HIGH", 10)
    if o.distinct_hand_shapes >= distinct_high:
        votes += 1
        reasons.append(f"+distinct>={distinct_high}")

    # Clip with almost zero spreading + enclosing → automatic L.
    if o.spreading_share + o.enclosing_share <= 0.04:
        votes = min(votes, -1)
        reasons.append("-no_spreading_or_enclosing")

    return _band_from_votes(votes, high_at=3, low_at=0), reasons


def score_adaptability(o: PresentationOverview) -> Tuple[str, List[str]]:
    """Adaptability: effort/shape variety — "variations in shapes of hands".

    Ground-truth anchors:
      * Payu/Aom2:  H — "very high variations in shapes of hands".
      * Ann/Andre/Meiji3/Aom1:  M — "moderate variations".
      * Lea/Sawitree/Lisa/Ches/Sarinee/Meiji1:  L — "very few / very low variations".
    """
    reasons: List[str] = []
    votes = 0

    # Distinct hand-shape bins: clear clusters in the 18-clip rubric are
    #   H clips: distinct >= 12   (Payu/Aom2 "very high variations")
    #   M clips: distinct  8..11  (Aom1/Andre/Ann/Meiji3)
    #   L clips: distinct <=  7   (Lisa/Ches/Chutima/Meiji1/Meiji2/Sarinee/Lea/Sawitree)
    distinct_high = _env_int("PRES_ADAPT_DISTINCT_HIGH", 12)
    distinct_mod = _env_int("PRES_ADAPT_DISTINCT_MOD", 9)
    distinct_low_max = _env_int("PRES_ADAPT_DISTINCT_LOW_MAX", 7)
    variety_high = _env_int("PRES_ADAPT_VARIETY_HIGH", 5)
    variety_low = _env_int("PRES_ADAPT_VARIETY_LOW", 3)

    if o.distinct_hand_shapes >= distinct_high:
        votes += 2
        reasons.append(f"+distinct>={distinct_high}")
    elif o.distinct_hand_shapes <= distinct_low_max:
        votes -= 2
        reasons.append(f"-distinct<={distinct_low_max}")
    elif o.distinct_hand_shapes >= distinct_mod:
        votes += 1
        reasons.append(f"+distinct>={distinct_mod}")

    if o.effort_variety >= variety_high:
        votes += 1
        reasons.append(f"+effort_variety>={variety_high}")
    elif o.effort_variety <= variety_low:
        votes -= 1
        reasons.append(f"-effort_variety<={variety_low}")

    # Closed/blocking posture shrinks effective "variation".
    if o.hand_block_share > 0.60:
        votes -= 1
        reasons.append("-closed_posture")

    # Require strong evidence for H (otherwise moderate-distinct + variety
    # alone can drift clips like Meiji3 into the wrong band).
    return _band_from_votes(votes, high_at=3, low_at=-1), reasons


def score_confidence(o: PresentationOverview) -> Tuple[str, List[str]]:
    """Confidence: upright + stable stance + not-blocking posture.

    Ground-truth anchors:
      * Lisa1/Lisa2/Ches/Aom1/Meiji1:  H — "extremely upright, never changed stance".
      * Payu/Chutima/Aom2/Andre/Meiji2/3:  M — upright but tilting/weight-shifting.
      * Lea/Sawitree/Ann/Candy/Aon/Sarinee:  L — swaying / no stance / blocking.
    """
    reasons: List[str] = []
    votes = 0

    up_high = _env_float("PRES_CONF_UPRIGHT_HIGH", 65.0)
    up_low = _env_float("PRES_CONF_UPRIGHT_LOW", 40.0)
    sway_hi = _env_float("PRES_CONF_SWAY_HIGH", 0.055)
    sway_lo = _env_float("PRES_CONF_SWAY_LOW", 0.035)
    stance_hi = _env_float("PRES_CONF_STANCE_HIGH", 50.0)
    stance_lo = _env_float("PRES_CONF_STANCE_LOW", 25.0)
    block_hi = _env_float("PRES_CONF_BLOCK_HIGH", 0.70)

    if o.upright_pct >= up_high:
        votes += 1
        reasons.append(f"+upright>={up_high}")
    elif o.upright_pct < up_low:
        votes -= 1
        reasons.append(f"-upright<{up_low}")

    if o.hip_sway_std <= sway_lo:
        votes += 1
        reasons.append(f"+sway<={sway_lo}")
    elif o.hip_sway_std >= sway_hi:
        votes -= 1
        reasons.append(f"-sway>={sway_hi}")

    if o.stance_stability >= stance_hi:
        votes += 1
        reasons.append(f"+stance>={stance_hi}")
    elif o.stance_stability <= stance_lo:
        votes -= 1
        reasons.append(f"-stance<={stance_lo}")

    if o.hand_block_share >= block_hi:
        votes -= 1
        reasons.append(f"-hand_block>={block_hi}")

    # "extremely upright throughout" bonus: upright>=80 AND sway<=0.03.
    if o.upright_pct >= 80.0 and o.hip_sway_std <= 0.030:
        votes += 1
        reasons.append("+extremely_upright_bonus")

    # No-stance + low engagement = nervous body language → L (Aon case).
    if o.stance_stability < 25.0 and o.gesture_share < 0.08:
        votes = min(votes, -1)
        reasons.append("-no_stance_no_engagement")

    return _band_from_votes(votes, high_at=2, low_at=-1), reasons


def score_authority(
    o: PresentationOverview,
    confidence_band: str,
) -> Tuple[str, List[str]]:
    """Authority closely tracks Confidence in the ground truth, with one
    specific exception: Andre1/Andre2 are Conf=M but Auth=L (no strong-effort
    "pressing for action" despite appearing upright).

    Rule:
      * Start from Confidence band.
      * Downgrade one level if ``strong_effort_share`` is very low AND
        ``hand_low_share`` is high (hands hanging at sides, no drive).
      * Upgrade one level (rare) if strong_effort_share is high and
        stance is solid.
    """
    reasons: List[str] = [f"auth_base=conf({confidence_band})"]

    ladder = ["Low", "Moderate", "High"]
    idx = ladder.index(confidence_band)

    strong_low = _env_float("PRES_AUTH_STRONG_LOW", 0.04)
    strong_high = _env_float("PRES_AUTH_STRONG_HIGH", 0.20)
    hand_low_high = _env_float("PRES_AUTH_HAND_LOW_HIGH", 0.30)

    # Downgrade: low strong effort + hands hanging low (Andre pattern).
    if (
        o.strong_effort_share < strong_low
        and o.hand_low_share >= hand_low_high
        and idx > 0
    ):
        idx -= 1
        reasons.append(
            f"-strong_effort<{strong_low} & hand_low>={hand_low_high} (Andre-pattern)"
        )

    # Upgrade: very strong directing/pressing + solid stance.
    if (
        o.strong_effort_share >= strong_high
        and o.stance_stability >= 55.0
        and idx < 2
    ):
        idx += 1
        reasons.append(
            f"+strong_effort>={strong_high} & stance_solid"
        )

    return ladder[idx], reasons


# ---------------------------------------------------------------------------
# Top-level entrypoint
# ---------------------------------------------------------------------------

_RULE_CATEGORIES = (
    "eye_contact", "uprightness", "stance",
    "engaging", "adaptability", "confidence", "authority",
)


def score_presentation(
    first_impression_pct: Tuple[float, float, float],
    category_features: Dict[str, Any],
    analysis_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Run the full presentation-analysis scorer.

    Selection of scoring engine (controlled by env vars, ML is preferred
    when models are available):

      ``PRES_SCORER_MODE``
        ``"ml"``       — pure ML (RandomForest). Falls back to rule if models
                         missing or an error occurs.
        ``"rule"``     — pure rule-based (this module).
        ``"hybrid"``   — default. Use ML; if ML's class probability for a
                         category is low (< ``PRES_ML_MIN_CONFIDENCE``, default
                         0.40), fall back to rule-based for that category.
        ``"ensemble"`` — run both scorers; if they disagree, prefer the one
                         with the higher confidence. Rule-based confidence is
                         treated as 0.55 (slightly-certain).

    Returns a dict with 7 category bands, the overview, and per-category
    rationale (which criteria fired or which classifier decided).
    """
    overview = build_overview(
        first_impression_pct=first_impression_pct,
        category_features=category_features,
        analysis_result=analysis_result,
    )

    rule_result = _score_rule_based(overview)

    mode = (os.getenv("PRES_SCORER_MODE", "hybrid").strip().lower() or "hybrid")
    if mode == "rule":
        return _log_and_return(rule_result, overview, engine="rule")

    ml_result = _try_score_ml(first_impression_pct, category_features, analysis_result)
    if not ml_result:
        return _log_and_return(rule_result, overview, engine="rule(fallback)")

    if mode == "ml":
        return _log_and_return(
            _blend_ml_rule(ml_result, rule_result, mode="ml"),
            overview,
            engine="ml",
        )

    if mode == "ensemble":
        return _log_and_return(
            _blend_ml_rule(ml_result, rule_result, mode="ensemble"),
            overview,
            engine="ensemble",
        )

    # Default: hybrid.
    return _log_and_return(
        _blend_ml_rule(ml_result, rule_result, mode="hybrid"),
        overview,
        engine="hybrid",
    )


# ---------------------------------------------------------------------------
# Internal helpers: rule-based scoring + ML blending.
# ---------------------------------------------------------------------------

def _score_rule_based(overview: PresentationOverview) -> Dict[str, Any]:
    eye_band, eye_reasons = score_eye_contact(overview)
    up_band, up_reasons = score_uprightness(overview)
    st_band, st_reasons = score_stance(overview)
    eng_band, eng_reasons = score_engaging(overview)
    adp_band, adp_reasons = score_adaptability(overview)
    con_band, con_reasons = score_confidence(overview)
    aut_band, aut_reasons = score_authority(overview, con_band)

    return {
        "eye_contact": eye_band,
        "uprightness": up_band,
        "stance": st_band,
        "engaging": eng_band,
        "adaptability": adp_band,
        "confidence": con_band,
        "authority": aut_band,
        "rationale": {
            "eye_contact": eye_reasons,
            "uprightness": up_reasons,
            "stance": st_reasons,
            "engaging": eng_reasons,
            "adaptability": adp_reasons,
            "confidence": con_reasons,
            "authority": aut_reasons,
        },
    }


def _try_score_ml(
    first_impression_pct: Tuple[float, float, float],
    category_features: Dict[str, Any],
    analysis_result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Run the ML scorer. Returns ``None`` if models aren't available."""
    try:
        from src.presentation_ml import (
            build_feature_vector, predict_bands_with_proba,
        )
    except Exception as exc:
        logger.warning("[presentation_scorer] ML module unavailable: %s", exc)
        return None

    try:
        vec = build_feature_vector(
            first_impression_pct, category_features, analysis_result
        )
        results = predict_bands_with_proba(vec)
    except Exception as exc:
        logger.warning("[presentation_scorer] ML scoring error: %s", exc)
        return None

    if not results:
        return None

    return results


def _blend_ml_rule(
    ml_result: Dict[str, Dict[str, Any]],
    rule_result: Dict[str, Any],
    mode: str,
) -> Dict[str, Any]:
    """Combine ML predictions with the rule-based scorer.

    * ``mode == "ml"`` — pick ML for all categories where the model exists.
    * ``mode == "hybrid"`` — pick ML unless its top-class probability is below
      ``PRES_ML_MIN_CONFIDENCE`` (default 0.40), in which case use rule.
    * ``mode == "ensemble"`` — when they disagree, pick the higher-confidence
      one (rule treated as 0.55 confidence).
    """
    min_conf = _env_float("PRES_ML_MIN_CONFIDENCE", 0.40)
    rule_conf = _env_float("PRES_RULE_CONFIDENCE", 0.55)

    final: Dict[str, Any] = {}
    rationale: Dict[str, List[str]] = {}

    for cat in _RULE_CATEGORIES:
        rule_band = rule_result.get(cat, "Moderate")
        rule_reasons = list(rule_result.get("rationale", {}).get(cat, []))

        if cat not in ml_result:
            final[cat] = rule_band
            rationale[cat] = ["src=rule_no_model"] + rule_reasons
            continue

        entry = ml_result[cat]
        ml_band = entry.get("band", rule_band)
        proba = entry.get("proba", {}) or {}
        ml_top = float(proba.get(ml_band, 1.0)) if proba else 1.0

        if mode == "ml":
            final[cat] = ml_band
            rationale[cat] = [f"src=ml(prob={ml_top:.2f})"] + rule_reasons
        elif mode == "hybrid":
            if ml_top >= min_conf:
                final[cat] = ml_band
                rationale[cat] = [f"src=ml(prob={ml_top:.2f})"] + rule_reasons
            else:
                final[cat] = rule_band
                rationale[cat] = [f"src=rule(ml_low={ml_top:.2f})"] + rule_reasons
        else:  # ensemble
            if ml_band == rule_band:
                final[cat] = ml_band
                rationale[cat] = [f"src=agree(ml={ml_top:.2f})"] + rule_reasons
            elif ml_top >= rule_conf:
                final[cat] = ml_band
                rationale[cat] = [
                    f"src=ml_wins({ml_band} ml={ml_top:.2f} > rule={rule_conf:.2f})"
                ] + rule_reasons
            else:
                final[cat] = rule_band
                rationale[cat] = [
                    f"src=rule_wins({rule_band} rule={rule_conf:.2f} >= ml={ml_top:.2f})"
                ] + rule_reasons

    final["rationale"] = rationale
    return final


def _log_and_return(
    scores: Dict[str, Any],
    overview: PresentationOverview,
    engine: str,
) -> Dict[str, Any]:
    logger.info(
        "[presentation_scorer:%s] eye=%s upright=%s stance=%s | "
        "engaging=%s adaptability=%s confidence=%s authority=%s | "
        "overview: upright=%.1f sway=%.4f stance=%.1f "
        "gesture=%.3f hand_block=%.3f distinct=%d strong=%.3f",
        engine,
        scores.get("eye_contact"), scores.get("uprightness"), scores.get("stance"),
        scores.get("engaging"), scores.get("adaptability"),
        scores.get("confidence"), scores.get("authority"),
        overview.upright_pct, overview.hip_sway_std, overview.stance_stability,
        overview.gesture_share, overview.hand_block_share,
        overview.distinct_hand_shapes, overview.strong_effort_share,
    )
    return {
        "eye_contact": scores.get("eye_contact"),
        "uprightness": scores.get("uprightness"),
        "stance": scores.get("stance"),
        "engaging": scores.get("engaging"),
        "adaptability": scores.get("adaptability"),
        "confidence": scores.get("confidence"),
        "authority": scores.get("authority"),
        "overview": overview,
        "rationale": scores.get("rationale", {}),
        "engine": engine,
    }


# ---------------------------------------------------------------------------
# Mapping H/M/L back to the 1-7 integer scale the report builder expects
# (category scores in report.categories[].scale / score fields).
# ---------------------------------------------------------------------------

def band_to_score_7(band: str) -> int:
    """Map H/M/L to a representative integer on the 1-7 scale.

    This lines up with the thresholds used by ``_display_scale`` /
    ``people_reader_scale_to_category_score`` so the report text reads the
    same band back from the integer.
    """
    b = (band or "").strip().lower()
    if b.startswith("h"):
        return 6
    if b.startswith("m"):
        return 4
    return 2
