from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass
class NarrativeBlock:
    title: str
    subtitle: str = ""
    level: str = ""
    bullets: List[str] = field(default_factory=list)
    impact_title: str = "Impact for clients:"
    impact_text: str = ""
    summary: str = ""


@dataclass
class NarrativeReport:
    first_impression_blocks: List[NarrativeBlock] = field(default_factory=list)
    category_blocks: List[NarrativeBlock] = field(default_factory=list)
    executive_summary: str = ""


# =========================================================
# HELPERS
# =========================================================

def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def get_float(data: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(data.get(key, default))
    except Exception:
        return default


def get_bool(data: Dict[str, Any], key: str, default: bool = False) -> bool:
    try:
        return bool(data.get(key, default))
    except Exception:
        return default


def qualitative_level(
    score: float,
    high_threshold: float = 0.75,
    moderate_threshold: float = 0.5,
) -> str:
    score = clamp(score)
    if score >= high_threshold:
        return "high"
    if score >= moderate_threshold:
        return "moderate"
    return "low"


def combine_scores(*values: float) -> float:
    vals = [clamp(v) for v in values if v is not None]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def pick_phrase(level: str, mapping: Dict[str, str], fallback: str = "") -> str:
    return mapping.get(level, fallback or mapping.get("moderate", ""))


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


# =========================================================
# INTERPRETATION LAYER
# =========================================================

def interpret_eye_contact(metrics: Dict[str, Any], audience_mode: str = "one") -> Dict[str, Any]:
    eye_contact_ratio = clamp(get_float(metrics, "eye_contact_ratio", 0.5))
    gaze_stability = clamp(get_float(metrics, "gaze_stability", 0.5))
    gaze_shift_control = clamp(get_float(metrics, "gaze_shift_control", 0.5))
    gaze_avoidance = clamp(get_float(metrics, "gaze_avoidance", 0.0))

    score = combine_scores(
        eye_contact_ratio,
        gaze_stability,
        gaze_shift_control,
        1.0 - gaze_avoidance,
    )
    level = qualitative_level(score)

    traits: List[str] = []
    if eye_contact_ratio >= 0.75:
        traits.append("audience-focused")
    elif eye_contact_ratio >= 0.5:
        traits.append("generally engaged")
    else:
        traits.append("inconsistent")

    if gaze_stability >= 0.75:
        traits.append("steady")
    elif gaze_stability >= 0.5:
        traits.append("mostly stable")
    else:
        traits.append("easily drifting")

    if gaze_shift_control >= 0.75:
        traits.append("purposeful")
    elif gaze_shift_control < 0.4:
        traits.append("somewhat reactive")

    if gaze_avoidance >= 0.45:
        traits.append("avoidant")

    strengths: List[str] = []
    risks: List[str] = []

    if eye_contact_ratio >= 0.7:
        strengths.append("maintains audience connection during key moments")
    else:
        risks.append("does not consistently hold audience focus")

    if gaze_stability >= 0.7:
        strengths.append("creates a calm and reliable visual presence")
    else:
        risks.append("may reduce perceived clarity or confidence")

    if gaze_shift_control >= 0.7:
        strengths.append("uses gaze shifts intentionally rather than randomly")
    else:
        risks.append("may appear distracted when shifting gaze")

    if gaze_avoidance >= 0.45:
        risks.append("shows signs of visual avoidance")

    return {
        "score": score,
        "level": level,
        "traits": dedupe_keep_order(traits),
        "strengths": strengths,
        "risks": risks,
        "audience_mode": audience_mode,
    }


def interpret_uprightness(metrics: Dict[str, Any]) -> Dict[str, Any]:
    posture_upright = clamp(get_float(metrics, "posture_upright", 0.5))
    head_alignment = clamp(get_float(metrics, "head_alignment", 0.5))
    shoulder_balance = clamp(get_float(metrics, "shoulder_balance", 0.5))
    chest_openness = clamp(get_float(metrics, "chest_openness", 0.5))
    slouch_ratio = clamp(get_float(metrics, "slouch_ratio", 0.0))

    score = combine_scores(
        posture_upright,
        head_alignment,
        shoulder_balance,
        chest_openness,
        1.0 - slouch_ratio,
    )
    level = qualitative_level(score)

    strengths: List[str] = []
    risks: List[str] = []

    if posture_upright >= 0.75:
        strengths.append("maintains a naturally upright posture")
    else:
        risks.append("upright alignment is not fully sustained")

    if head_alignment >= 0.7 and shoulder_balance >= 0.7:
        strengths.append("keeps the head and shoulders well aligned")
    else:
        risks.append("upper-body alignment can appear uneven at times")

    if chest_openness >= 0.7:
        strengths.append("projects openness and readiness")
    else:
        risks.append("may appear slightly closed or guarded")

    if slouch_ratio >= 0.35:
        risks.append("shows visible slouching or collapsing at moments")

    return {
        "score": score,
        "level": level,
        "strengths": strengths,
        "risks": risks,
    }


def interpret_stance(metrics: Dict[str, Any]) -> Dict[str, Any]:
    lower_body_stability = clamp(get_float(metrics, "lower_body_stability", 0.5))
    grounding = clamp(get_float(metrics, "grounding", 0.5))
    weight_shift_control = clamp(get_float(metrics, "weight_shift_control", 0.5))
    forward_orientation = clamp(get_float(metrics, "forward_orientation", 0.5))
    stance_symmetry = clamp(get_float(metrics, "stance_symmetry", 0.5))

    score = combine_scores(
        lower_body_stability,
        grounding,
        weight_shift_control,
        forward_orientation,
        stance_symmetry,
    )
    level = qualitative_level(score)

    strengths: List[str] = []
    risks: List[str] = []

    if grounding >= 0.7:
        strengths.append("appears grounded and physically settled")
    else:
        risks.append("may appear less settled or less rooted")

    if weight_shift_control >= 0.7:
        strengths.append("controls weight shifts without distraction")
    else:
        risks.append("shifting may slightly distract from message delivery")

    if forward_orientation >= 0.7:
        strengths.append("maintains clear audience orientation")
    else:
        risks.append("orientation toward the audience could be stronger")

    return {
        "score": score,
        "level": level,
        "strengths": strengths,
        "risks": risks,
    }


def interpret_engaging_connecting(metrics: Dict[str, Any]) -> Dict[str, Any]:
    facial_warmth = clamp(get_float(metrics, "facial_warmth", 0.5))
    openness = clamp(get_float(metrics, "openness", 0.5))
    expressive_variety = clamp(get_float(metrics, "expressive_variety", 0.5))
    audience_connection = clamp(get_float(metrics, "audience_connection", 0.5))

    score = combine_scores(
        facial_warmth,
        openness,
        expressive_variety,
        audience_connection,
    )
    level = qualitative_level(score)

    strengths = []
    risks = []

    if facial_warmth >= 0.7:
        strengths.append("appears approachable and easy to connect with")
    else:
        risks.append("could project more warmth in facial presence")

    if openness >= 0.7:
        strengths.append("signals receptiveness and accessibility")
    else:
        risks.append("may feel slightly guarded or formal")

    if audience_connection >= 0.7:
        strengths.append("creates rapport with the audience")
    else:
        risks.append("connection with the audience can be deepened")

    return {
        "score": score,
        "level": level,
        "strengths": strengths,
        "risks": risks,
    }


def interpret_confidence(metrics: Dict[str, Any]) -> Dict[str, Any]:
    voice_like_presence = clamp(get_float(metrics, "presence_score", 0.5))
    posture_strength = clamp(get_float(metrics, "posture_strength", 0.5))
    movement_control = clamp(get_float(metrics, "movement_control", 0.5))
    focus_consistency = clamp(get_float(metrics, "focus_consistency", 0.5))

    score = combine_scores(
        voice_like_presence,
        posture_strength,
        movement_control,
        focus_consistency,
    )
    level = qualitative_level(score)

    strengths = []
    risks = []

    if posture_strength >= 0.7:
        strengths.append("shows self-possession through body organization")
    else:
        risks.append("physical presence could appear more settled")

    if movement_control >= 0.7:
        strengths.append("moves with intention rather than nervousness")
    else:
        risks.append("movement may occasionally reduce authority")

    if focus_consistency >= 0.7:
        strengths.append("keeps a clear and directed presence")
    else:
        risks.append("focus may feel less sustained in key moments")

    return {
        "score": score,
        "level": level,
        "strengths": strengths,
        "risks": risks,
    }


def interpret_authority(metrics: Dict[str, Any]) -> Dict[str, Any]:
    verticality = clamp(get_float(metrics, "verticality", 0.5))
    decisiveness = clamp(get_float(metrics, "decisiveness", 0.5))
    grounded_power = clamp(get_float(metrics, "grounded_power", 0.5))
    urgency_signal = clamp(get_float(metrics, "urgency_signal", 0.5))

    score = combine_scores(
        verticality,
        decisiveness,
        grounded_power,
        urgency_signal,
    )
    level = qualitative_level(score)

    strengths = []
    risks = []

    if verticality >= 0.7:
        strengths.append("appears tall, composed, and structurally clear")
    else:
        risks.append("vertical presence could be stronger")

    if decisiveness >= 0.7:
        strengths.append("signals conviction and direction")
    else:
        risks.append("can sound less assertive in delivery")

    if grounded_power >= 0.7:
        strengths.append("combines stability with leadership presence")
    else:
        risks.append("leadership presence may feel lighter than intended")

    return {
        "score": score,
        "level": level,
        "strengths": strengths,
        "risks": risks,
    }


# =========================================================
# NARRATIVE GENERATORS
# =========================================================

def generate_eye_contact_block(insight: Dict[str, Any]) -> NarrativeBlock:
    level = insight["level"]
    traits = insight["traits"]
    strengths = insight["strengths"]
    risks = insight["risks"]
    audience_mode = insight.get("audience_mode", "one")

    trait_text = ", ".join(traits[:-1]) + f", and {traits[-1]}" if len(traits) > 1 else (traits[0] if traits else "engaged")
    bullets: List[str] = []

    if level == "high":
        bullets.append(f"Your eye contact is {trait_text}.")
        if audience_mode == "many":
            bullets.append("You scan the room in a controlled way, which helps you include multiple listeners without losing engagement.")
        else:
            bullets.append("You maintain direct gaze during key message points, which increases trust and clarity.")
        bullets.append("When you shift your gaze, it appears intentional rather than avoidant.")
        bullets.append("Overall, your eye contact supports confidence, sincerity, and credibility.")
        impact = "Strong eye contact signals presence, sincerity, and leadership confidence, making your message feel more reliable."
        summary = "Eye contact is one of your strongest first-impression signals."
    elif level == "moderate":
        bullets.append(f"Your eye contact appears {trait_text}.")
        bullets.append("You do connect with the audience, especially during clearer or more prepared moments.")
        bullets.append("At times, your gaze could stay longer on key points to strengthen trust and emphasis.")
        if risks:
            bullets.append("A more consistent gaze pattern would make your delivery feel calmer and more assured.")
        impact = "Improving gaze consistency would increase clarity, trust, and audience connection."
        summary = "Eye contact supports the message, but greater consistency would strengthen impact."
    else:
        bullets.append(f"Your eye contact currently appears {trait_text}.")
        bullets.append("Direct gaze is not yet sustained enough to fully support confidence and clarity.")
        bullets.append("At times, gaze may drift or disengage from the audience too quickly.")
        bullets.append("This can make the message feel less anchored, even when the content itself is strong.")
        impact = "A stronger eye-contact pattern would immediately improve trust, presence, and message authority."
        summary = "Eye contact is currently limiting the strength of the first impression."

    return NarrativeBlock(
        title="First Impression",
        subtitle="Eye Contact",
        level=level,
        bullets=bullets,
        impact_text=impact,
        summary=summary,
    )


def generate_uprightness_block(insight: Dict[str, Any]) -> NarrativeBlock:
    level = insight["level"]
    bullets: List[str] = []

    if level == "high":
        bullets.append("You maintain a naturally upright posture throughout the clip.")
        bullets.append("The chest stays open, shoulders relaxed, and head aligned — signaling balance, readiness, and authority.")
        bullets.append("Even when you gesture, your vertical alignment remains stable, showing good core control.")
        bullets.append("There is no visible slouching or collapsing, which supports a professional appearance.")
        impact = "Uprightness communicates self-assurance, clarity of thought, and emotional stability."
        summary = "Your posture strongly supports a credible and composed first impression."
    elif level == "moderate":
        bullets.append("Your posture is generally upright and supportive of a professional presence.")
        bullets.append("Most of the time, your upper-body alignment helps you look prepared and attentive.")
        bullets.append("There are occasional moments where the posture softens or loses some structural clarity.")
        bullets.append("A slightly more consistent vertical line would make your presence feel stronger.")
        impact = "A more stable upright posture would reinforce confidence and sharpen professional presence."
        summary = "Posture is supportive overall, with room for more consistency."
    else:
        bullets.append("Your upper-body posture does not yet remain consistently upright throughout the clip.")
        bullets.append("At times, the body line softens or collapses, which can reduce presence and clarity.")
        bullets.append("This may unintentionally weaken the impression of confidence and authority.")
        bullets.append("Strengthening upright alignment would immediately improve visual credibility.")
        impact = "Improving posture would make the overall delivery feel more assured and more leadership-oriented."
        summary = "Posture currently weakens the overall strength of first impression."

    return NarrativeBlock(
        title="First Impression",
        subtitle="Uprightness (Posture & Upper-Body Alignment)",
        level=level,
        bullets=bullets,
        impact_text=impact,
        summary=summary,
    )


def generate_stance_block(insight: Dict[str, Any]) -> NarrativeBlock:
    level = insight["level"]
    bullets: List[str] = []

    if level == "high":
        bullets.append("Your stance is symmetrical and grounded, with the body looking physically settled.")
        bullets.append("Weight shifts are controlled and minimal, preventing distraction and showing confidence.")
        bullets.append("You maintain good forward orientation toward the audience, reinforcing clarity and engagement.")
        bullets.append("The stance conveys both stability and a welcoming presence, suitable for instructional or coaching communication.")
        impact = "A grounded stance enhances authority, control, and smooth message delivery."
        summary = "Your lower-body stability helps the audience experience you as prepared and credible."
    elif level == "moderate":
        bullets.append("Your stance is generally stable and supportive of the message.")
        bullets.append("Most weight shifts are manageable and do not strongly distract from delivery.")
        bullets.append("There is some room to become more grounded and more consistently audience-facing.")
        bullets.append("A slightly stronger base would make the whole presentation feel calmer and more authoritative.")
        impact = "Greater grounding would improve stability, authority, and visual confidence."
        summary = "Stance is functional, but more grounding would increase impact."
    else:
        bullets.append("Your stance is not yet consistently grounded throughout the clip.")
        bullets.append("Shifts in the lower body may reduce the sense of stability and control.")
        bullets.append("This can make the delivery feel less anchored, even when your ideas are clear.")
        bullets.append("A stronger physical base would improve both confidence and perceived authority.")
        impact = "Improving lower-body grounding would make the message feel more stable and more credible."
        summary = "Stance currently reduces the sense of grounded confidence."

    return NarrativeBlock(
        title="First Impression",
        subtitle="Stance (Lower-Body Stability & Grounding)",
        level=level,
        bullets=bullets,
        impact_text=impact,
        summary=summary,
    )


def generate_engaging_block(insight: Dict[str, Any]) -> NarrativeBlock:
    level = insight["level"]

    if level == "high":
        bullets = [
            "You come across as approachable and easy to connect with.",
            "Your overall presence suggests openness, relatability, and a willingness to engage.",
            "The way you hold yourself supports quick rapport-building with the audience.",
            "This gives your delivery a human, warm, and collaborative quality.",
        ]
        impact = "This style of presence helps audiences feel welcomed, included, and more willing to listen."
        summary = "You naturally project an engaging and connecting presence."
    elif level == "moderate":
        bullets = [
            "You show some signs of warmth and connection in your delivery.",
            "There is a foundation of approachability, especially in more settled moments.",
            "At times, the connection could feel more open or more immediate.",
            "A bit more visible warmth would strengthen rapport and audience comfort.",
        ]
        impact = "More visible openness would help your message feel more relatable and more engaging."
        summary = "Your engaging quality is present, though not yet fully maximized."
    else:
        bullets = [
            "Your delivery currently feels more functional than relational.",
            "The audience may receive the message, but may not yet feel strongly connected to you as a speaker.",
            "A warmer and more open presence would help create rapport more quickly.",
            "This would make your communication feel more human, accessible, and memorable.",
        ]
        impact = "Increasing warmth and openness would improve connection, trust, and audience responsiveness."
        summary = "Engaging presence is currently under-expressed."

    return NarrativeBlock(
        title="Engaging & Connecting",
        level=level,
        bullets=bullets,
        impact_text=impact,
        summary=summary,
    )


def generate_confidence_block(insight: Dict[str, Any]) -> NarrativeBlock:
    level = insight["level"]

    if level == "high":
        bullets = [
            "You project optimistic presence and stable focus.",
            "Your body organization suggests self-possession rather than hesitation.",
            "Movement and posture work together to support a sense of certainty.",
            "Overall, the delivery feels persuasive because it appears internally grounded.",
        ]
        impact = "This kind of confidence increases trust and helps the audience believe in both the message and the messenger."
        summary = "Confidence is one of the strongest qualities in your delivery."
    elif level == "moderate":
        bullets = [
            "You show a reasonable degree of confidence in your presence.",
            "There are moments where your delivery feels steady and convincing.",
            "At other times, the confidence signal could be clearer or more sustained.",
            "More consistency in physical focus would strengthen persuasive impact.",
        ]
        impact = "A more sustained confidence signal would make the delivery feel more compelling and more persuasive."
        summary = "Confidence is present, but not yet fully consistent."
    else:
        bullets = [
            "Your current delivery does not yet project full confidence consistently.",
            "At times, the body appears less settled than the message may require.",
            "This can reduce persuasive force, even when the content is meaningful.",
            "A stronger organized presence would help the audience trust your message more quickly.",
        ]
        impact = "Stronger confidence signals would increase persuasion, clarity, and audience trust."
        summary = "Confidence is currently limiting persuasive strength."

    return NarrativeBlock(
        title="Confidence",
        level=level,
        bullets=bullets,
        impact_text=impact,
        summary=summary,
    )


def generate_authority_block(insight: Dict[str, Any]) -> NarrativeBlock:
    level = insight["level"]

    if level == "high":
        bullets = [
            "You show a clear sense of importance and direction in the subject matter.",
            "Your structure and presence support a leadership-oriented delivery.",
            "The body signals conviction without appearing rigid or overforced.",
            "This helps the audience read you as credible, prepared, and worthy of attention.",
        ]
        impact = "Authority helps the audience trust your leadership, respect your message, and follow your direction."
        summary = "Authority is strongly supported by your physical presence."
    elif level == "moderate":
        bullets = [
            "You show some authority in the way you hold the message.",
            "There are visible signs of structure, intention, and seriousness.",
            "However, the authority signal is not equally strong throughout the full clip.",
            "More decisiveness in body organization would increase leadership presence.",
        ]
        impact = "A stronger authority signal would help the audience experience you as more directive and more influential."
        summary = "Authority is present, though it could be reinforced further."
    else:
        bullets = [
            "Your delivery does not yet consistently project authority.",
            "The audience may understand the content without fully feeling your leadership behind it.",
            "This can reduce urgency, influence, and executive presence.",
            "A more grounded and decisive physical pattern would improve authority significantly.",
        ]
        impact = "Stronger authority cues would increase leadership impact, clarity, and persuasive power."
        summary = "Authority is currently weaker than the message likely requires."

    return NarrativeBlock(
        title="Authority",
        level=level,
        bullets=bullets,
        impact_text=impact,
        summary=summary,
    )


# =========================================================
# EXECUTIVE SUMMARY
# =========================================================

def generate_executive_summary(
    eye_contact_block: NarrativeBlock,
    uprightness_block: NarrativeBlock,
    stance_block: NarrativeBlock,
    engaging_block: NarrativeBlock,
    confidence_block: NarrativeBlock,
    authority_block: NarrativeBlock,
) -> str:
    strengths = []
    opportunities = []

    all_blocks = [
        eye_contact_block,
        uprightness_block,
        stance_block,
        engaging_block,
        confidence_block,
        authority_block,
    ]

    for block in all_blocks:
        if block.level == "high":
            strengths.append(block.title if block.title != "First Impression" else block.subtitle)
        elif block.level == "low":
            opportunities.append(block.title if block.title != "First Impression" else block.subtitle)

    if not strengths:
        strengths_text = "The delivery shows a useful foundation across several communication signals."
    else:
        strengths_text = "The strongest areas are " + ", ".join(strengths[:3]) + "."

    if not opportunities:
        opportunity_text = "Overall, the communication profile is well balanced and presentation-ready."
    else:
        opportunity_text = "The clearest development opportunities are " + ", ".join(opportunities[:3]).lower() + "."

    return f"{strengths_text} {opportunity_text} Overall, the presentation communicates meaningful potential, and with targeted refinement, it can become even more persuasive, credible, and audience-impactful."


# =========================================================
# MAIN ENGINE
# =========================================================

def build_narrative_report(
    first_impression_metrics: Dict[str, Any],
    category_metrics: Dict[str, Any],
    audience_mode: str = "one",
) -> NarrativeReport:
    eye_contact_insight = interpret_eye_contact(first_impression_metrics, audience_mode=audience_mode)
    uprightness_insight = interpret_uprightness(first_impression_metrics)
    stance_insight = interpret_stance(first_impression_metrics)

    engaging_insight = interpret_engaging_connecting(category_metrics)
    confidence_insight = interpret_confidence(category_metrics)
    authority_insight = interpret_authority(category_metrics)

    eye_contact_block = generate_eye_contact_block(eye_contact_insight)
    uprightness_block = generate_uprightness_block(uprightness_insight)
    stance_block = generate_stance_block(stance_insight)

    engaging_block = generate_engaging_block(engaging_insight)
    confidence_block = generate_confidence_block(confidence_insight)
    authority_block = generate_authority_block(authority_insight)

    executive_summary = generate_executive_summary(
        eye_contact_block,
        uprightness_block,
        stance_block,
        engaging_block,
        confidence_block,
        authority_block,
    )

    return NarrativeReport(
        first_impression_blocks=[
            eye_contact_block,
            uprightness_block,
            stance_block,
        ],
        category_blocks=[
            engaging_block,
            confidence_block,
            authority_block,
        ],
        executive_summary=executive_summary,
    )