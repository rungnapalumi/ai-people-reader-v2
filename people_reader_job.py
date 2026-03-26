# people_reader_job.py — People Reader movement-type blending (repo root)
#
# Used by report_worker.py (root) so it does not duplicate src/report_worker.py.
# Loads pose features from src/report_core.py (same pipeline as production worker).

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("report_worker")

_src_rc_mod: Optional[Any] = None


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _load_src_report_core():
    """MediaPipe movement features live in src/report_core.py (full build)."""
    global _src_rc_mod
    if _src_rc_mod is not None:
        return _src_rc_mod
    path = os.path.join(_repo_root(), "src", "report_core.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"src/report_core.py not found at {path}")
    spec = importlib.util.spec_from_file_location("report_core_src", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _src_rc_mod = mod
    return mod


def _import_movement_type_classifier():
    root = _repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    import movement_type_classifier as mtc  # noqa: WPS433

    return mtc


def apply_movement_type_classification(
    video_path: str,
    job: Dict[str, Any],
    result: Dict[str, Any],
    first_impression: Any,
) -> Tuple[Dict[str, Any], Any, Optional[Dict[str, Any]]]:
    """
    Same contract as src/report_worker.apply_movement_type_classification.
    Returns (updated_result, blended_first_impression, movement_type_info_or_none).
    Blended first impression uses the same dataclass type as `first_impression`.
    """
    mode_raw = str(job.get("movement_type_mode") or "").strip().lower()
    if not mode_raw:
        return result, first_impression, None
    try:
        mtc = _import_movement_type_classifier()
    except Exception as e:
        logger.warning("[movement_type] cannot import movement_type_classifier: %s", e)
        return result, first_impression, None

    mtc.clear_expected_range_overrides()

    rc = _load_src_report_core()
    extract_movement_type_frame_features_from_video = rc.extract_movement_type_frame_features_from_video

    audience_mode = str(job.get("audience_mode") or "one").strip().lower()
    if audience_mode not in ("one", "many"):
        audience_mode = "one"

    try:
        feats = extract_movement_type_frame_features_from_video(
            video_path,
            audience_mode=audience_mode,
            sample_every_n=3,
            max_frames=200,
        )
        sf = mtc.build_summary_features_from_timeseries(feats)
    except Exception as e:
        logger.warning("[movement_type] feature extraction failed: %s", e)
        sf = mtc.build_summary_features_from_timeseries({})

    ranked = mtc.rank_people_reader_types_by_seven_match(sf)
    if not ranked:
        logger.warning("[movement_type] empty seven-match ranking")
        return result, first_impression, None

    if mode_raw == "auto":
        chosen_id = str(ranked[0]["type_id"])
        mode_flag = "auto"
    elif mode_raw in mtc.TYPE_TEMPLATES:
        chosen_id = mode_raw
        mode_flag = "selected"
    else:
        logger.warning("[movement_type] unknown movement_type_mode=%r", mode_raw)
        return result, first_impression, None

    tpl_chosen = mtc.TYPE_TEMPLATES[chosen_id]
    chosen_row = next((r for r in ranked if str(r["type_id"]) == chosen_id), ranked[0])
    matches_chosen = int(chosen_row.get("matches") or 0)
    match_pct_chosen = int(chosen_row.get("match_pct") or 0)

    cat_scales = mtc.people_reader_category_scales_from_template(tpl_chosen)
    result = dict(result)
    result["engaging_score"] = max(
        1, min(7, mtc.people_reader_scale_to_category_score(cat_scales["engaging"]))
    )
    result["convince_score"] = max(
        1, min(7, mtc.people_reader_scale_to_category_score(cat_scales["confidence"]))
    )
    result["authority_score"] = max(
        1, min(7, mtc.people_reader_scale_to_category_score(cat_scales["authority"]))
    )
    result["adaptability_score"] = max(
        1, min(7, mtc.people_reader_scale_to_category_score(cat_scales["adaptability"]))
    )
    result["engaging_pos"] = int(result["engaging_score"] / 7 * 450)
    result["convince_pos"] = int(result["convince_score"] / 7 * 475)
    result["authority_pos"] = int(result["authority_score"] / 7 * 445)
    result["adaptability_pos"] = int(result["adaptability_score"] / 7 * 445)

    v_levels = mtc.video_seven_levels_people_reader(sf)
    classification_for_narrative = {
        "best_match": {
            "type_id": tpl_chosen.type_id,
            "type_name": tpl_chosen.name,
            "summary": tpl_chosen.summary,
            "traits": dict(tpl_chosen.traits),
            "score": round(matches_chosen / 7.0, 4),
            "confidence": max(0.0, min(1.0, float(match_pct_chosen) / 100.0)),
            "confidence_pct": match_pct_chosen,
            "matched_features": [],
        },
        "alternatives": [],
        "scores": [],
        "summary_features": sf,
    }
    narrative_en = str(mtc.generate_movement_type_narrative(classification_for_narrative) or "")
    traits = dict(tpl_chosen.traits)

    cls_fi = type(first_impression)
    fi2 = cls_fi(
        eye_contact_pct=float(
            0.42 * float(first_impression.eye_contact_pct)
            + 0.58 * (float(sf.get("eye_contact", 0.5)) * 100.0)
        ),
        upright_pct=float(
            0.42 * float(first_impression.upright_pct)
            + 0.58 * (float(sf.get("uprightness", 0.5)) * 100.0)
        ),
        stance_stability=float(
            0.42 * float(first_impression.stance_stability)
            + 0.58
            * (
                50.0
                + 50.0
                * (
                    0.55 * float(sf.get("stance_width_score", 0.5))
                    + 0.45 * float(sf.get("weight_shift_score", 0.5))
                )
            )
        ),
    )

    def traits_line_en(tr: Dict[str, Any]) -> str:
        return (
            f"Confidence: {tr.get('confidence', '-')}; "
            f"Authority: {tr.get('authority', '-')}; "
            f"Adaptability: {tr.get('adaptability', '-')}"
        )

    def traits_line_th(tr: Dict[str, Any]) -> str:
        m = {"high": "สูง", "low": "ต่ำ", "moderate": "กลาง"}

        def T(x: Any) -> str:
            return m.get(str(x).lower(), str(x))

        return (
            f"ความมั่นใจ: {T(tr.get('confidence'))}; "
            f"อำนาจ/ผู้นำ: {T(tr.get('authority'))}; "
            f"ความยืดหยุ่น: {T(tr.get('adaptability'))}"
        )

    r1 = ranked[0]
    r2 = ranked[1] if len(ranked) > 1 else r1
    seven_line_en = (
        f"1) {r1['type_name']} — {int(r1['matches'])}/7 ({int(r1['match_pct'])}%); "
        f"2) {r2['type_name']} — {int(r2['matches'])}/7 ({int(r2['match_pct'])}%)"
    )
    seven_line_th = (
        f"1) {r1['type_name']} — {int(r1['matches'])}/7 ({int(r1['match_pct'])}%); "
        f"2) {r2['type_name']} — {int(r2['matches'])}/7 ({int(r2['match_pct'])}%)"
    )

    info: Dict[str, Any] = {
        "type_name": str(tpl_chosen.name or ""),
        "type_id": str(tpl_chosen.type_id or ""),
        "summary": str(tpl_chosen.summary or ""),
        "confidence_pct": int(match_pct_chosen),
        "traits_line_en": traits_line_en(traits),
        "traits_line_th": traits_line_th(traits),
        "narrative_en": narrative_en,
        "narrative_th": narrative_en,
        "mode": mode_flag,
        "mode_en": (
            "(Auto-detected from video)"
            if mode_flag == "auto"
            else "(Manually selected — report aligned to this profile)"
        ),
        "mode_th": (
            "(วิเคราะห์อัตโนมัติจากวิดีโอ)"
            if mode_flag == "auto"
            else "(เลือกประเภทด้วยตนเอง — ปรับรายงานตามโปรไฟล์นี้)"
        ),
        "seven_match_chosen_matches": matches_chosen,
        "seven_match_chosen_out_of": 7,
        "seven_match_rank1_type_id": str(r1.get("type_id") or ""),
        "seven_match_rank1_type_name": str(r1.get("type_name") or ""),
        "seven_match_rank1_matches": int(r1.get("matches") or 0),
        "seven_match_rank1_pct": int(r1.get("match_pct") or 0),
        "seven_match_rank2_type_id": str(r2.get("type_id") or ""),
        "seven_match_rank2_type_name": str(r2.get("type_name") or ""),
        "seven_match_rank2_matches": int(r2.get("matches") or 0),
        "seven_match_rank2_pct": int(r2.get("match_pct") or 0),
        "seven_match_line_en": seven_line_en,
        "seven_match_line_th": seven_line_th,
        "seven_dim_labels_en": list(mtc.PEOPLE_READER_SEVEN_DIM_LABELS_EN),
        "seven_dim_labels_th": list(mtc.PEOPLE_READER_SEVEN_DIM_LABELS_TH),
        "seven_video_levels": list(v_levels),
        "seven_chosen_template_levels": list(chosen_row.get("template_levels") or []),
    }
    logger.info(
        "[movement_type] mode=%s chosen=%s seven_match=%s/7 rank1=%s/%s rank2=%s/%s",
        mode_flag,
        info.get("type_id"),
        matches_chosen,
        r1.get("type_id"),
        r1.get("matches"),
        r2.get("type_id"),
        r2.get("matches"),
    )
    return result, fi2, info
