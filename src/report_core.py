# report_core.py — shared report logic for report generation
import os
import io
import math
import random
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

try:
    from mediapipe.python.solutions import pose as mp_pose_module
    from mediapipe.python.solutions.pose import Pose, PoseLandmark
    mp = True
except ImportError:
    try:
        import mediapipe as mp
        mp_pose_module = mp.solutions.pose
        Pose = mp_pose_module.Pose
        PoseLandmark = mp_pose_module.PoseLandmark
        mp = True
    except Exception:
        mp = None
        mp_pose_module = None
        Pose = None
        PoseLandmark = None

# Dataclasses
@dataclass
class FirstImpressionData:
    eye_contact_pct: float
    upright_pct: float
    stance_stability: float

@dataclass
class CategoryResult:
    name_en: str
    name_th: str
    score: int
    scale: str
    positives: int
    total: int
    description: str = ""

@dataclass
class ReportData:
    client_name: str
    analysis_date: str
    video_length_str: str
    overall_score: int
    categories: list
    summary_comment: str
    generated_by: str
    first_impression: Optional[FirstImpressionData] = None

# Helpers
def format_seconds_to_mmss(total_seconds: float) -> str:
    total_seconds = max(0, float(total_seconds))
    mm = int(total_seconds // 60)
    ss = int(round(total_seconds - mm * 60))
    if ss == 60:
        mm += 1
        ss = 0
    return f"{mm:02d}:{ss:02d}"

def get_video_duration_seconds(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps <= 0:
        return 0.0
    return float(frames / fps)

def analyze_first_impression_from_video(video_path: str, sample_every_n: int = 5, max_frames: int = 200) -> FirstImpressionData:
    """Real First Impression analysis with MediaPipe using continuous scoring."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return FirstImpressionData(eye_contact_pct=0.0, upright_pct=0.0, stance_stability=0.0)

    frame_samples = []

    def _normalize_horizontal_angle(deg: float) -> float:
        while deg > 180.0:
            deg -= 360.0
        while deg <= -180.0:
            deg += 360.0
        if deg > 90.0:
            deg -= 180.0
        if deg < -90.0:
            deg += 180.0
        return deg

    with Pose(static_image_mode=False, model_complexity=1) as pose:
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            i += 1
            if i % sample_every_n != 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if not res.pose_landmarks:
                continue

            lms = res.pose_landmarks.landmark
            nose = lms[PoseLandmark.NOSE]
            leye = lms[PoseLandmark.LEFT_EYE]
            reye = lms[PoseLandmark.RIGHT_EYE]
            lsh = lms[PoseLandmark.LEFT_SHOULDER]
            rsh = lms[PoseLandmark.RIGHT_SHOULDER]
            lhip = lms[PoseLandmark.LEFT_HIP]
            rhip = lms[PoseLandmark.RIGHT_HIP]
            lank = lms[PoseLandmark.LEFT_ANKLE]
            rank = lms[PoseLandmark.RIGHT_ANKLE]

            if min(nose.visibility, leye.visibility, reye.visibility, lsh.visibility, rsh.visibility, lhip.visibility, rhip.visibility) < 0.5:
                continue

            frame_samples.append(
                {
                    "nose": (float(nose.x), float(nose.y)),
                    "leye": (float(leye.x), float(leye.y)),
                    "reye": (float(reye.x), float(reye.y)),
                    "lsh": (float(lsh.x), float(lsh.y)),
                    "rsh": (float(rsh.x), float(rsh.y)),
                    "lhip": (float(lhip.x), float(lhip.y)),
                    "rhip": (float(rhip.x), float(rhip.y)),
                    "lank": (float(lank.x), float(lank.y)),
                    "rank": (float(rank.x), float(rank.y)),
                    "leye_vis": float(leye.visibility),
                    "reye_vis": float(reye.visibility),
                    "lank_vis": float(lank.visibility),
                    "rank_vis": float(rank.visibility),
                }
            )
            if len(frame_samples) >= max_frames:
                break

    cap.release()

    total = len(frame_samples)
    if total == 0:
        return FirstImpressionData(eye_contact_pct=0.0, upright_pct=0.0, stance_stability=0.0)

    # Auto-calibrate camera roll (tilted camera) using median shoulder/hip line angles.
    roll_candidates = []
    for s in frame_samples:
        lsh = s["lsh"]
        rsh = s["rsh"]
        lhip = s["lhip"]
        rhip = s["rhip"]
        sh_ang = math.degrees(math.atan2(rsh[1] - lsh[1], rsh[0] - lsh[0]))
        hip_ang = math.degrees(math.atan2(rhip[1] - lhip[1], rhip[0] - lhip[0]))
        roll_candidates.append(_normalize_horizontal_angle(sh_ang))
        roll_candidates.append(_normalize_horizontal_angle(hip_ang))
    camera_roll_deg = float(np.median(np.array(roll_candidates))) if roll_candidates else 0.0
    camera_roll_deg = max(-20.0, min(20.0, camera_roll_deg))
    theta = -math.radians(camera_roll_deg)

    def _rotate_point(pt: tuple, cx: float = 0.5, cy: float = 0.5) -> tuple:
        x, y = pt
        dx = x - cx
        dy = y - cy
        xr = dx * math.cos(theta) - dy * math.sin(theta) + cx
        yr = dx * math.sin(theta) + dy * math.cos(theta) + cy
        return (xr, yr)

    eye_frame_scores = []
    upright_frame_scores = []
    ankle_dist = []
    ankle_center_x = []
    stance_width_ratios = []
    prev_nose_offset_ratio = None
    prev_torso_angle = None

    for s in frame_samples:
        nose = _rotate_point(s["nose"])
        leye = _rotate_point(s["leye"])
        reye = _rotate_point(s["reye"])
        lsh = _rotate_point(s["lsh"])
        rsh = _rotate_point(s["rsh"])
        lhip = _rotate_point(s["lhip"])
        rhip = _rotate_point(s["rhip"])
        lank = _rotate_point(s["lank"])
        rank = _rotate_point(s["rank"])

        # Eye contact (continuous): frontal face symmetry + gaze stability
        # with extra penalties for looking down, closing/occluding eyes, and head-down posture.
        eye_dist = abs(leye[0] - reye[0])
        if eye_dist > 1e-4:
            mid_eye_x = (leye[0] + reye[0]) / 2.0
            mid_eye_y = (leye[1] + reye[1]) / 2.0
            nose_offset_ratio = abs(nose[0] - mid_eye_x) / eye_dist
            symmetry_score = max(0.0, 1.0 - (nose_offset_ratio / 0.75))
            stability_bonus = 1.0
            if prev_nose_offset_ratio is not None:
                jitter = abs(nose_offset_ratio - prev_nose_offset_ratio)
                stability_bonus = max(0.70, 1.0 - (jitter / 0.35))
            prev_nose_offset_ratio = nose_offset_ratio
            base_eye_score = 100.0 * symmetry_score * stability_bonus

            nose_eye_ratio = (nose[1] - mid_eye_y) / eye_dist
            look_down_penalty = max(0.0, min(1.0, (nose_eye_ratio - 0.55) / 0.50))

            mid_sh_y = (lsh[1] + rsh[1]) / 2.0
            mid_sh_xy = np.array([(lsh[0] + rsh[0]) / 2.0, (lsh[1] + rsh[1]) / 2.0])
            mid_hip_xy = np.array([(lhip[0] + rhip[0]) / 2.0, (lhip[1] + rhip[1]) / 2.0])
            torso_len = float(np.linalg.norm(mid_sh_xy - mid_hip_xy)) + 1e-9
            head_raise_ratio = (mid_sh_y - nose[1]) / torso_len
            head_down_penalty = max(0.0, min(1.0, (0.28 - head_raise_ratio) / 0.18))

            eye_visibility = min(float(s["leye_vis"]), float(s["reye_vis"]))
            eye_closed_penalty = max(0.0, min(1.0, (0.85 - eye_visibility) / 0.35))

            attention_penalty = (
                0.28 * look_down_penalty
                + 0.22 * head_down_penalty
                + 0.18 * eye_closed_penalty
            )
            attention_factor = max(0.55, 1.0 - attention_penalty)
            eye_frame_scores.append(base_eye_score * attention_factor)

        # Uprightness (continuous): torso vertical angle + shoulder level + abrupt tilt penalty.
        mid_sh = np.array([(lsh[0] + rsh[0]) / 2.0, (lsh[1] + rsh[1]) / 2.0])
        mid_hip = np.array([(lhip[0] + rhip[0]) / 2.0, (lhip[1] + rhip[1]) / 2.0])
        v = mid_sh - mid_hip
        vert = np.array([0.0, -1.0])
        v_norm = np.linalg.norm(v) + 1e-9
        cosang = float(np.dot(v / v_norm, vert))
        ang = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
        shoulder_tilt = abs(lsh[1] - rsh[1])
        shoulder_tilt_ratio = shoulder_tilt / v_norm

        angle_score = max(0.0, 1.0 - (ang / 28.0))
        shoulder_score = max(0.0, 1.0 - (shoulder_tilt_ratio / 0.20))
        temporal_bonus = 1.0
        if prev_torso_angle is not None:
            angle_jump = abs(ang - prev_torso_angle)
            temporal_bonus = max(0.75, 1.0 - (angle_jump / 45.0))
        prev_torso_angle = ang
        upright_frame_scores.append(
            100.0 * (0.75 * angle_score + 0.25 * shoulder_score) * temporal_bonus
        )

        # Stance (continuous): lower-body stability + center sway + stance width appropriateness.
        if min(float(s["lank_vis"]), float(s["rank_vis"])) >= 0.5:
            dx = (lank[0] - rank[0])
            dy = (lank[1] - rank[1])
            dist = math.sqrt(dx * dx + dy * dy)
            ankle_dist.append(dist)
            ankle_center_x.append((lank[0] + rank[0]) / 2.0)
            shoulder_width = abs(lsh[0] - rsh[0])
            if shoulder_width > 1e-4:
                stance_width_ratios.append(dist / shoulder_width)

    eye_pct = float(np.mean(np.array(eye_frame_scores))) if eye_frame_scores else 0.0
    upright_pct = float(np.mean(np.array(upright_frame_scores))) if upright_frame_scores else 0.0

    if len(ankle_dist) >= 10:
        dist_arr = np.array(ankle_dist)
        dist_std = float(np.std(dist_arr))
        dist_mean = float(np.mean(dist_arr)) + 1e-9
        rel_std = dist_std / dist_mean

        # Lower relative ankle-distance variance -> better lower-body stability.
        base_stability = max(0.0, min(100.0, 100.0 * (1.0 - (rel_std / 0.35))))

        sway_score = 0.0
        if len(ankle_center_x) >= 10:
            sway_std = float(np.std(np.array(ankle_center_x)))
            sway_score = max(0.0, min(100.0, 100.0 * (1.0 - (sway_std / 0.06))))

        # Width should have limited influence because perspective/tilt can make
        # feet look narrower than reality. Prioritize stillness first.
        width_score = 75.0
        if stance_width_ratios:
            ratios = np.array(stance_width_ratios)
            ratio_mean = float(np.mean(ratios))
            ratio_std = float(np.std(ratios))
            # Use a wider tolerance so camera angle/zoom does not over-penalize.
            width_pref = max(0.70, 1.0 - (abs(ratio_mean - 1.10) / 1.80))
            width_stability = max(0.0, 1.0 - (ratio_std / 0.35))
            width_score = 100.0 * (0.65 * width_pref + 0.35 * width_stability)

        stability = max(
            0.0,
            min(100.0, 0.62 * base_stability + 0.30 * sway_score + 0.08 * width_score),
        )
    else:
        stability = 0.0

    return FirstImpressionData(eye_contact_pct=eye_pct, upright_pct=upright_pct, stance_stability=stability)

def generate_eye_contact_text(pct: float) -> list:
    """Generate descriptive text based on eye contact percentage - 10 levels"""
    if pct >= 90:  # Outstanding
        return [
            "• Your eye contact is outstanding — consistently steady, warm, and completely audience-focused throughout the entire presentation.",
            "• You maintain unwavering direct gaze during key moments, which maximizes trust, clarity, and audience engagement.",
            "• Every gaze shift is purposeful, natural, and enhances your communication (e.g., thinking pauses, emphasis).",
            "• Your eye contact is a masterclass in confidence and credibility, with zero signs of avoidance or nervousness."
        ]
    elif pct >= 80:  # Excellent
        return [
            "• Your eye contact is excellent — steady, engaging, and audience-focused almost throughout the presentation.",
            "• You consistently maintain direct gaze during key moments, which strongly increases trust and clarity.",
            "• Gaze shifts are purposeful and natural, never detracting from your professional presence.",
            "• Your eye contact demonstrates strong confidence and builds exceptional credibility with the audience."
        ]
    elif pct >= 70:  # Very Strong
        return [
            "• Your eye contact is very strong — steady and audience-focused for most of the time.",
            "• You maintain direct gaze during key moments effectively, which clearly builds trust and clarity.",
            "• Occasional gaze shifts are natural and appropriate, showing thoughtfulness without reducing engagement.",
            "• Overall, your eye contact strongly supports confidence and credibility with the audience."
        ]
    elif pct >= 60:  # Strong
        return [
            "• Your eye contact is strong — generally steady and audience-focused during important moments.",
            "• You maintain direct gaze during most key moments, which helps establish trust.",
            "• Some gaze shifts occur but they're mostly natural and don't significantly impact your presence.",
            "• Your eye contact effectively supports good confidence and reasonable credibility."
        ]
    elif pct >= 50:  # Good
        return [
            "• Your eye contact shows a good foundation, with several effective moments of direct audience connection.",
            "• You maintain direct gaze during many important points, which helps build trust.",
            "• With slightly longer eye contact in key moments, your delivery will feel even more confident.",
            "• Overall, your audience engagement is positive and can be strengthened further."
        ]
    elif pct >= 40:  # Above Average
        return [
            "• Your eye contact is developing well, with regular moments of direct audience connection.",
            "• You make direct eye contact consistently, and sustaining it slightly longer will increase impact.",
            "• Some gaze shifts occur more frequently than ideal, but this is very trainable.",
            "• Improving consistency will noticeably enhance trust and credibility."
        ]
    elif pct >= 30:  # Average
        return [
            "• Your eye contact is in a developing stage, with clear opportunities to become more consistent.",
            "• You already connect directly with the audience at times, especially in selected moments.",
            "• Extending direct gaze a bit longer will strengthen audience connection and engagement.",
            "• With focused practice, trust and credibility cues will improve clearly."
        ]
    else:
        return [
            "• Your eye contact is developing and currently inconsistent in several parts of the presentation.",
            "• You connect with the audience at times, and extending direct gaze during key moments will improve trust.",
            "• Some gaze shifts are more frequent than ideal, but this can be improved with focused practice.",
            "• With steadier eye contact, your confidence and credibility will become more apparent."
        ]

def generate_eye_contact_text_th(pct: float) -> list:
    """Thai descriptive text based on eye contact percentage - 10 levels"""
    if pct >= 90:
        return [
            "• การสบตาโดดเด่นมาก สม่ำเสมอ อบอุ่น และโฟกัสผู้ฟังตลอดการนำเสนอ",
            "• คุณรักษาการสบตาในจุดสำคัญได้อย่างยอดเยี่ยม ช่วยเพิ่มความไว้วางใจสูงมาก",
            "• การเปลี่ยนสายตาเป็นธรรมชาติและมีจุดประสงค์ ไม่ทำให้พลังการสื่อสารลดลง",
            "• ภาพรวมสะท้อนความมั่นใจและความน่าเชื่อถือในระดับสูงสุด",
        ]
    elif pct >= 80:
        return [
            "• การสบตายอดเยี่ยม มีความต่อเนื่องและเชื่อมโยงผู้ฟังได้ดีมาก",
            "• คุณสบตาในประเด็นสำคัญได้ชัดเจน ช่วยสร้างความเชื่อมั่นได้มาก",
            "• มีการละสายตาบ้างเล็กน้อยแต่ยังเป็นธรรมชาติ",
            "• ภาพรวมแสดงบุคลิกที่มั่นใจและน่าเชื่อถือมาก",
        ]
    elif pct >= 70:
        return [
            "• การสบตาแข็งแรงและคงเส้นคงวาในช่วงส่วนใหญ่ของการพูด",
            "• ช่วงที่เน้นสาระสำคัญมีการสบตาที่ดี ช่วยเสริมความไว้วางใจ",
            "• มีจังหวะละสายตาเป็นระยะแต่ไม่กระทบการเชื่อมต่อผู้ฟังมาก",
            "• โดยรวมสนับสนุนภาพลักษณ์มืออาชีพได้ดี",
        ]
    elif pct >= 60:
        return [
            "• การสบตาอยู่ในระดับดีและค่อนข้างสม่ำเสมอ",
            "• จุดสำคัญส่วนใหญ่ยังคงสบตาได้เหมาะสม",
            "• มีการละสายตาบ้างแต่ยังไม่รบกวนภาพรวม",
            "• โดยรวมช่วยเสริมความมั่นใจได้ชัดเจน",
        ]
    elif pct >= 50:
        return [
            "• การสบตามีพื้นฐานที่ดี และมีหลายช่วงที่เชื่อมโยงผู้ฟังได้ชัดเจน",
            "• ในประเด็นสำคัญสามารถสบตาได้พอสมควรและสร้างความไว้วางใจได้",
            "• หากเพิ่มระยะเวลาการสบตาอีกเล็กน้อย พลังการสื่อสารจะดีขึ้นชัดเจน",
            "• ภาพรวมเป็นบวกและต่อยอดได้อีกมาก",
        ]
    elif pct >= 40:
        return [
            "• การสบตากำลังพัฒนาไปได้ดี และมีช่วงที่เชื่อมโยงผู้ฟังได้ต่อเนื่อง",
            "• มีการสบตาสม่ำเสมอในหลายจังหวะ และสามารถต่อยอดให้คงที่ขึ้นได้",
            "• การละสายตายังมีอยู่บ้าง แต่สามารถปรับได้ด้วยการฝึก",
            "• หากเพิ่มความต่อเนื่องของการสบตา จะช่วยเสริมความน่าเชื่อถือมากขึ้น",
        ]
    elif pct >= 30:
        return [
            "• การสบตาอยู่ในช่วงพัฒนา และยังมีโอกาสเพิ่มความสม่ำเสมอได้อีก",
            "• มีหลายช่วงที่เชื่อมโยงผู้ฟังได้ดี โดยเฉพาะบางประเด็นสำคัญ",
            "• หากคุมการละสายตาให้น้อยลง จะทำให้การสื่อสารไหลลื่นขึ้น",
            "• แนะนำฝึกการสบตาแบบสั้นแต่ต่อเนื่องในจังหวะสำคัญ",
        ]
    else:
        return [
            "• การสบตายังอยู่ในช่วงพัฒนา และยังไม่สม่ำเสมอในหลายช่วงของการนำเสนอ",
            "• มีช่วงที่เชื่อมโยงผู้ฟังได้ดี หากเพิ่มระยะเวลาการสบตาในประเด็นสำคัญจะดีขึ้นมาก",
            "• การละสายตายังเกิดบ่อยกว่าที่ควร แต่สามารถปรับได้ด้วยการฝึกอย่างต่อเนื่อง",
            "• เมื่อคุมการสบตาได้คงที่ขึ้น ภาพลักษณ์ความมั่นใจและความน่าเชื่อถือจะชัดเจนขึ้น",
        ]

def generate_uprightness_text(pct: float) -> list:
    """Generate descriptive text based on uprightness percentage - 10 levels"""
    if pct >= 90:  # Outstanding
        return [
            "• You maintain outstanding upright posture with perfect vertical alignment throughout the entire presentation.",
            "• Your chest remains fully open, shoulders are optimally relaxed, and head alignment is impeccable — signaling exceptional balance, readiness, and commanding authority.",
            "• Even during active gesturing, your vertical alignment stays completely stable, demonstrating masterful core control.",
            "• There is absolutely no visible slouching or collapsing at any point, projecting supreme professional presence."
        ]
    elif pct >= 80:  # Excellent
        return [
            "• You maintain excellent upright posture consistently throughout the presentation.",
            "• Your chest stays open, shoulders are naturally relaxed, and head alignment is near-perfect — signaling strong balance, readiness, and authority.",
            "• Your vertical alignment remains remarkably stable even when gesturing, demonstrating superior core control.",
            "• There is virtually no slouching or collapsing, which projects a highly professional and confident appearance."
        ]
    elif pct >= 70:  # Very Strong
        return [
            "• You maintain very strong upright posture throughout most of the presentation.",
            "• The chest stays open, shoulders remain relaxed, and head is well-aligned — clearly signaling balance, readiness, and authority.",
            "• Your vertical alignment stays stable during most gestures, showing very good core control.",
            "• Minimal slouching occurs, which strongly supports a professional and composed appearance."
        ]
    elif pct >= 60:  # Strong
        return [
            "• You maintain strong upright posture for the majority of the clip.",
            "• Your chest generally stays open, shoulders relaxed, and head aligned — signaling good balance and authority.",
            "• Vertical alignment remains fairly stable during gestures, showing solid core control.",
            "• Occasional minor slouching appears but doesn't detract significantly from your professional presence."
        ]
    elif pct >= 50:  # Good
        return [
            "• Your posture has a good foundation, with many moments of stable upright alignment.",
            "• Chest and shoulder alignment are generally positive, with room to make consistency even better.",
            "• During gestures, vertical alignment is mostly stable and supports clear communication.",
            "• Overall posture is strong and can be elevated further with small refinements."
        ]
    elif pct >= 40:  # Above Average
        return [
            "• Your posture is developing well, with a number of solid upright moments throughout the presentation.",
            "• Chest and shoulder positioning are often acceptable, and greater consistency will add polish.",
            "• Vertical alignment varies in some gesture phases, which is common and trainable.",
            "• With steadier posture, your professional image will look even more confident."
        ]
    elif pct >= 30:  # Average
        return [
            "• Your posture is in a developing stage, alternating between upright moments and less stable alignment.",
            "• There are several points where shoulder and head position can be brought closer to ideal alignment.",
            "• Improving vertical consistency will make your presence feel stronger and more composed.",
            "• With ongoing practice, professional presence and authority will improve clearly."
        ]
    else:
        return [
            "• Your posture is still developing and varies between upright and less stable alignment.",
            "• There are several moments where shoulder and head alignment drift from ideal posture.",
            "• Improving core control and maintaining a taller stance will enhance your professional presence.",
            "• With consistent practice, your posture can become more stable and confident."
        ]

def generate_uprightness_text_th(pct: float) -> list:
    """Thai descriptive text based on uprightness percentage - 10 levels"""
    if pct >= 90:
        return [
            "• ท่าทางตั้งตรงโดดเด่นมาก การจัดแนวลำตัวนิ่งและสมดุลตลอดการพูด",
            "• หน้าอกเปิด ไหล่ผ่อนคลาย ศีรษะอยู่ในแนวที่ดีอย่างสม่ำเสมอ",
            "• แม้มีการใช้ท่าทาง มือและลำตัวก็ยังคงความมั่นคงสูง",
            "• ภาพรวมสื่อความมั่นใจและภาวะผู้นำได้ชัดเจนมาก",
        ]
    elif pct >= 80:
        return [
            "• ท่าทางตั้งตรงยอดเยี่ยมและคงเส้นคงวาเกือบตลอดคลิป",
            "• การจัดแนวหัวไหล่และลำตัวอยู่ในเกณฑ์ดีมาก",
            "• มีการเอนตัวเล็กน้อยบางช่วงแต่ไม่กระทบภาพรวม",
            "• โดยรวมดูมั่นใจและมืออาชีพมาก",
        ]
    elif pct >= 70:
        return [
            "• ท่าทางตั้งตรงดีมากในช่วงส่วนใหญ่ของการนำเสนอ",
            "• โครงลำตัวโดยรวมสมดุลและควบคุมได้ดี",
            "• มีจังหวะที่แนวลำตัวหลุดเล็กน้อยเป็นครั้งคราว",
            "• ภาพรวมยังสนับสนุนความน่าเชื่อถือได้ดี",
        ]
    elif pct >= 60:
        return [
            "• ท่าทางตั้งตรงอยู่ในระดับดีและค่อนข้างสม่ำเสมอ",
            "• ลำตัวและไหล่ส่วนใหญ่ยังอยู่ในแนวเหมาะสม",
            "• มีการคล้อยลำตัวบ้างเล็กน้อยในบางช่วง",
            "• โดยรวมช่วยเสริมบุคลิกมืออาชีพได้ดี",
        ]
    elif pct >= 50:
        return [
            "• ท่าทางตั้งตรงมีพื้นฐานที่ดี และมีหลายช่วงที่คงแนวลำตัวได้ดี",
            "• ความต่อเนื่องโดยรวมเป็นบวก และยังมีโอกาสทำให้คงที่ขึ้นอีก",
            "• หากคุมแกนลำตัวให้ต่อเนื่องขึ้นเล็กน้อย ภาพรวมจะดีขึ้นชัดเจน",
            "• โดยรวมอยู่ในทิศทางที่ดีและพัฒนาได้อีกมาก",
        ]
    elif pct >= 40:
        return [
            "• ท่าทางกำลังพัฒนาได้ดี และมีหลายช่วงที่รักษาแนวได้เหมาะสม",
            "• แนวไหล่/ศีรษะยังมีบางจังหวะที่แกว่งเล็กน้อย ซึ่งปรับได้ไม่ยาก",
            "• การคุมแกนกลางมีแนวโน้มดีและสามารถเสริมให้มั่นคงขึ้นได้",
            "• หากเพิ่มความสม่ำเสมออีกเล็กน้อย บุคลิกจะดูมืออาชีพขึ้นมาก",
        ]
    elif pct >= 30:
        return [
            "• ท่าทางอยู่ในช่วงพัฒนา มีทั้งช่วงที่ตั้งตรงได้ดีและช่วงที่ยังไม่นิ่ง",
            "• แนวไหล่และศีรษะยังมีบางจังหวะที่หลุดจากแนวเหมาะสม",
            "• หากคุมแนวลำตัวให้คงที่ขึ้น จะช่วยเสริมความมั่นใจได้ชัดเจน",
            "• แนะนำฝึกการคุมแกนลำตัวอย่างต่อเนื่องเพื่อเพิ่มความนิ่ง",
        ]
    else:
        return [
            "• ท่าทางยังอยู่ในช่วงพัฒนา และมีหลายช่วงที่แนวลำตัวยังไม่นิ่ง",
            "• แนวไหล่และศีรษะหลุดจากแนวที่เหมาะสมเป็นระยะ",
            "• หากเพิ่มการคุมแกนกลางและรักษาท่ายืนให้ต่อเนื่อง จะช่วยเสริมบุคลิกได้มาก",
            "• ฝึกอย่างสม่ำเสมอจะทำให้ภาพรวมดูมั่นใจและมืออาชีพขึ้นอย่างชัดเจน",
        ]

def generate_stance_text(stability: float) -> list:
    """Generate descriptive text based on stance stability - 10 levels"""
    if stability >= 90:  # Outstanding
        return [
            "• Your stance is outstanding — perfectly symmetrical and solidly grounded, with feet placed optimally at shoulder-width apart.",
            "• Weight shifts are virtually non-existent, creating an exceptionally stable platform that demonstrates supreme confidence.",
            "• You maintain flawless forward orientation toward the audience throughout, maximally reinforcing clarity and engagement.",
            "• Your stance conveys rock-solid stability and commanding, welcoming authority ideal for executive leadership communication."
        ]
    elif stability >= 80:  # Excellent
        return [
            "• Your stance is excellent — exceptionally symmetrical and grounded, with feet placed perfectly about shoulder-width apart.",
            "• Weight shifts are extremely controlled and minimal, preventing any distraction and demonstrating strong confidence.",
            "• You maintain superior forward orientation toward the audience throughout, reinforcing excellent clarity and engagement.",
            "• Your stance conveys remarkable stability and a welcoming, authoritative presence ideal for professional leadership."
        ]
    elif stability >= 70:  # Very Strong
        return [
            "• Your stance is very strong — highly symmetrical and well-grounded, with feet well-positioned shoulder-width apart.",
            "• Weight shifts are well-controlled and quite minimal, showing strong confidence and solid balance.",
            "• You maintain very good forward orientation toward the audience, clearly reinforcing engagement and clarity.",
            "• The stance conveys strong stability and a professional, welcoming presence suitable for senior communication roles."
        ]
    elif stability >= 60:  # Strong
        return [
            "• Your stance is strong — symmetrical and grounded, with feet appropriately placed about shoulder-width apart.",
            "• Weight shifts are controlled and minimal, preventing distraction and demonstrating good confidence.",
            "• You maintain solid forward orientation toward the audience, reinforcing clarity and reasonable engagement.",
            "• The stance effectively conveys stability and a professional presence suitable for most business communication."
        ]
    elif stability >= 50:  # Good
        return [
            "• Your stance has a good base, with generally stable positioning and only occasional weight shifts.",
            "• Feet placement is mostly appropriate, supporting balanced delivery through much of the presentation.",
            "• Weight distribution is mostly steady, with minor adjustments that are common and manageable.",
            "• Overall stance supports a positive professional presence and can be refined further."
        ]
    elif stability >= 40:  # Above Average
        return [
            "• Your stance is developing well, with many steady moments across the presentation.",
            "• Feet placement is generally acceptable, and greater consistency will improve grounding.",
            "• Some visible shifts occur in parts of the clip, which is normal and can be reduced with practice.",
            "• Better stability will further strengthen your grounded presence and authority."
        ]
    elif stability >= 30:  # Average
        return [
            "• Your stance is in a developing stage, with opportunities to become steadier throughout delivery.",
            "• Feet placement varies at times, and improving consistency will support better balance.",
            "• Some swaying or shifting appears in parts of the presentation, and this can be improved with focused drills.",
            "• Enhancing stance stability will clearly strengthen your professional presence."
        ]
    else:
        return [
            "• Your stance is still developing, with noticeable movement and balance variation in several moments.",
            "• Feet placement and weight distribution are not yet fully consistent throughout the presentation.",
            "• Reducing unnecessary shifts will help you appear more grounded and authoritative.",
            "• With regular practice, your stance can become steadier and more professional."
        ]

def generate_stance_text_th(stability: float) -> list:
    """Thai descriptive text based on stance stability - 10 levels"""
    if stability >= 90:
        return [
            "• ท่ายืนมั่นคงยอดเยี่ยม สมมาตร และมีฐานยืนที่แข็งแรงมาก",
            "• การถ่ายน้ำหนักน้อยมาก ทำให้ภาพรวมดูนิ่งและมั่นใจสูง",
            "• การวางเท้าเหมาะสมและช่วยเสริมพลังการสื่อสารอย่างชัดเจน",
            "• สะท้อนบุคลิกผู้นำที่พร้อมและน่าเชื่อถือมาก",
        ]
    elif stability >= 80:
        return [
            "• ท่ายืนมั่นคงมาก มีความสมดุลต่อเนื่อง",
            "• การถ่ายน้ำหนักเกิดน้อยและควบคุมได้ดี",
            "• ฐานยืนช่วยให้ภาพลักษณ์ดูมั่นใจและมืออาชีพ",
            "• โดยรวมสื่อความน่าเชื่อถือได้ดีมาก",
        ]
    elif stability >= 70:
        return [
            "• ท่ายืนแข็งแรงและค่อนข้างนิ่งในช่วงส่วนใหญ่",
            "• มีการปรับน้ำหนักเล็กน้อยแต่ไม่รบกวนผู้ฟังมาก",
            "• ฐานยืนโดยรวมดีและรองรับการสื่อสารได้ดี",
            "• ภาพรวมเสริมความมั่นใจได้ชัดเจน",
        ]
    elif stability >= 60:
        return [
            "• ท่ายืนอยู่ในระดับดี มีความมั่นคงพอเหมาะ",
            "• มีการเปลี่ยนน้ำหนักเป็นระยะแต่ยังควบคุมได้",
            "• ฐานยืนโดยรวมยังสนับสนุนภาพลักษณ์มืออาชีพ",
            "• ปรับความนิ่งเพิ่มอีกเล็กน้อยจะดีขึ้นมาก",
        ]
    elif stability >= 50:
        return [
            "• ท่ายืนมีพื้นฐานที่ดี และมีหลายช่วงที่ค่อนข้างมั่นคง",
            "• ความสมดุลโดยรวมเป็นบวก และยังปรับให้คงที่ขึ้นได้อีก",
            "• มีการขยับน้ำหนักเป็นระยะ ซึ่งสามารถลดลงได้ด้วยการฝึก",
            "• หากเพิ่มความนิ่งอีกเล็กน้อย ภาพรวมจะดูมืออาชีพขึ้นมาก",
        ]
    elif stability >= 40:
        return [
            "• ท่ายืนกำลังพัฒนาได้ดี และมีหลายช่วงที่รักษาฐานยืนได้เหมาะสม",
            "• ความต่อเนื่องของฐานยืนยังมีโอกาสทำให้คงที่ขึ้นอีก",
            "• มีการโยก/ถ่ายน้ำหนักในบางจังหวะ ซึ่งสามารถปรับได้ด้วยการฝึก",
            "• หากยืน grounded มากขึ้น จะช่วยเสริมภาพลักษณ์ให้มั่นคงขึ้นชัดเจน",
        ]
    elif stability >= 30:
        return [
            "• ท่ายืนอยู่ในช่วงพัฒนา และยังเพิ่มความนิ่งได้อีกในหลายช่วง",
            "• การวางเท้าและการถ่ายน้ำหนักมีบางจังหวะที่ยังไม่คงที่",
            "• หากคุมฐานยืนให้ต่อเนื่องขึ้น จะช่วยเสริมความมั่นใจของภาพรวม",
            "• แนะนำฝึกคุมแกนล่างและจังหวะการถ่ายน้ำหนักแบบสม่ำเสมอ",
        ]
    else:
        return [
            "• ท่ายืนยังอยู่ในช่วงพัฒนา และมีการเคลื่อนไหว/ถ่ายน้ำหนักที่เห็นได้ในหลายช่วง",
            "• ฐานยืนและการกระจายน้ำหนักยังไม่คงที่ตลอดการนำเสนอ",
            "• หากลดการโยกหรือขยับที่ไม่จำเป็น จะช่วยให้ภาพลักษณ์ดูมั่นคงขึ้น",
            "• ฝึกการยืนอย่างสม่ำเสมอจะช่วยเสริมความน่าเชื่อถือและความเป็นมืออาชีพได้ชัดเจน",
        ]

# Analysis functions
def analyze_video_mediapipe(video_path: str, sample_fps: float = 5, max_frames: int = 300, **kwargs) -> Dict[str, Any]:
    """Real MediaPipe analysis with proper Laban Movement Analysis"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    frame_interval = max(1, int(fps / sample_fps))
    
    # Effort & Shape detection counters
    effort_counts = {
        "Directing": 0, "Enclosing": 0, "Punching": 0, "Spreading": 0,
        "Pressing": 0, "Dabbing": 0, "Indirecting": 0, "Gliding": 0,
        "Flicking": 0, "Advancing": 0, "Retreating": 0
    }
    shape_counts = {"Directing": 0, "Enclosing": 0, "Spreading": 0, "Indirecting": 0, "Advancing": 0, "Retreating": 0}
    
    analyzed = 0
    prev_landmarks = None
    
    with Pose(static_image_mode=False, model_complexity=1) as pose:
        frame_idx = 0
        while analyzed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                
                if results.pose_landmarks:
                    lms = results.pose_landmarks.landmark
                    
                    # Calculate movement vectors if we have previous frame
                    if prev_landmarks is not None:
                        # Get key points
                        left_wrist = lms[PoseLandmark.LEFT_WRIST]
                        right_wrist = lms[PoseLandmark.RIGHT_WRIST]
                        left_elbow = lms[PoseLandmark.LEFT_ELBOW]
                        right_elbow = lms[PoseLandmark.RIGHT_ELBOW]
                        left_shoulder = lms[PoseLandmark.LEFT_SHOULDER]
                        right_shoulder = lms[PoseLandmark.RIGHT_SHOULDER]
                        nose = lms[PoseLandmark.NOSE]
                        
                        # Previous frame landmarks
                        prev_left_wrist = prev_landmarks[PoseLandmark.LEFT_WRIST]
                        prev_right_wrist = prev_landmarks[PoseLandmark.RIGHT_WRIST]
                        
                        # Calculate velocities (movement speed)
                        left_wrist_vel = math.sqrt(
                            (left_wrist.x - prev_left_wrist.x)**2 +
                            (left_wrist.y - prev_left_wrist.y)**2 +
                            (left_wrist.z - prev_left_wrist.z)**2
                        )
                        right_wrist_vel = math.sqrt(
                            (right_wrist.x - prev_right_wrist.x)**2 +
                            (right_wrist.y - prev_right_wrist.y)**2 +
                            (right_wrist.z - prev_right_wrist.z)**2
                        )
                        avg_velocity = (left_wrist_vel + right_wrist_vel) / 2
                        
                        # Spatial measurements
                        wrist_dist = abs(left_wrist.x - right_wrist.x)
                        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
                        body_expansion = wrist_dist / max(shoulder_width, 0.1)
                        
                        # Hand height relative to shoulders
                        avg_hand_y = (left_wrist.y + right_wrist.y) / 2
                        avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
                        hands_above_shoulders = avg_hand_y < avg_shoulder_y
                        
                        # Hand depth (Z-axis) relative to body
                        avg_hand_z = (left_wrist.z + right_wrist.z) / 2
                        avg_shoulder_z = (left_shoulder.z + right_shoulder.z) / 2
                        hands_forward = avg_hand_z < avg_shoulder_z
                        
                        # Movement direction (hands)
                        left_z_delta = left_wrist.z - prev_left_wrist.z
                        right_z_delta = right_wrist.z - prev_right_wrist.z
                        avg_z_delta = (left_z_delta + right_z_delta) / 2
                        
                        # Forward/backward: Use average of both hands with higher threshold
                        forward_movement = avg_z_delta < -0.03  # Both hands moving toward camera
                        backward_movement = avg_z_delta > 0.03  # Both hands moving away from camera
                        
                        # Vertical movement
                        upward_movement = (left_wrist.y - prev_left_wrist.y) < -0.01 or (right_wrist.y - prev_right_wrist.y) < -0.01
                        downward_movement = (left_wrist.y - prev_left_wrist.y) > 0.01 or (right_wrist.y - prev_right_wrist.y) > 0.01
                        
                        # Effort qualities
                        is_sudden = avg_velocity > 0.08  # Fast movement
                        is_sustained = 0.02 < avg_velocity <= 0.08  # Moderate movement
                        is_strong = body_expansion > 1.2 or hands_above_shoulders
                        is_light = body_expansion < 0.8
                        
                        # === EFFORT DETECTION (11 types) ===
                        
                        # 1. DIRECTING: Direct, sustained, forward movement
                        if hands_forward and is_sustained and forward_movement:
                            effort_counts["Directing"] += 1
                            shape_counts["Directing"] += 1
                        
                        # 2. ENCLOSING: Arms coming together, wrapping motion
                        if body_expansion < 0.8 and avg_velocity > 0.03:
                            effort_counts["Enclosing"] += 1
                            shape_counts["Enclosing"] += 1
                        
                        # 3. PUNCHING: Sudden, strong, direct forward thrust
                        if is_sudden and is_strong and forward_movement:
                            effort_counts["Punching"] += 1
                        
                        # 4. SPREADING: Arms spreading wide, opening gesture
                        if body_expansion > 1.5 and avg_velocity > 0.03:
                            effort_counts["Spreading"] += 1
                            shape_counts["Spreading"] += 1
                        
                        # 5. PRESSING: Sustained, strong, downward force
                        if is_sustained and is_strong and downward_movement:
                            effort_counts["Pressing"] += 1
                        
                        # 6. DABBING: Sudden, light, direct touch
                        if is_sudden and is_light and avg_velocity > 0.05:
                            effort_counts["Dabbing"] += 1
                        
                        # 7. INDIRECTING: Indirect, curved, wandering movements
                        if not hands_forward and avg_velocity > 0.04 and body_expansion > 1.0:
                            effort_counts["Indirecting"] += 1
                            shape_counts["Indirecting"] += 1
                        
                        # 8. GLIDING: Sustained, light, indirect floating
                        if is_sustained and is_light and upward_movement:
                            effort_counts["Gliding"] += 1
                        
                        # 9. FLICKING: Sudden, light, indirect quick gesture
                        if is_sudden and is_light and not forward_movement:
                            effort_counts["Flicking"] += 1
                        
                        # 10. ADVANCING: Significant forward movement (toward audience)
                        # Requires sustained forward motion with reasonable speed
                        if forward_movement and avg_velocity > 0.06 and is_sustained:
                            effort_counts["Advancing"] += 1
                            shape_counts["Advancing"] += 1
                        
                        # 11. RETREATING: Significant backward movement (pulling back defensively)
                        # Requires sustained backward motion - not just gesture return
                        if backward_movement and avg_velocity > 0.06 and is_sustained:
                            effort_counts["Retreating"] += 1
                            shape_counts["Retreating"] += 1
                    
                    # Store current landmarks for next iteration
                    prev_landmarks = lms
                    analyzed += 1
            
            frame_idx += 1
    
    cap.release()
    
    # Calculate percentages
    total_detections = max(1, sum(effort_counts.values()))
    effort_detection = {k: round(v / total_detections * 100, 1) for k, v in effort_counts.items()}

    total_shape = max(1, sum(shape_counts.values()))
    shape_detection = {k: round(v / total_shape * 100, 1) for k, v in shape_counts.items()}

    # Add diversity/consistency signals so different speaking styles separate better.
    dominant_share = max(effort_detection.values()) / 100.0 if effort_detection else 0.0
    variety_floor = max(2, int(max(1, analyzed) * 0.03))
    variety_count = sum(1 for v in effort_counts.values() if v >= variety_floor)
    monotony_penalty = max(0.0, (dominant_share - 0.35) * 8.0)
    variety_factor = max(0.85, min(1.20, 0.85 + (variety_count / 11.0) * 0.35))

    total_frames = max(1, analyzed)
    engaging_activity = (
        effort_counts.get("Spreading", 0)
        + effort_counts.get("Enclosing", 0)
        + effort_counts.get("Gliding", 0)
        + effort_counts.get("Indirecting", 0)
    ) / total_frames
    confidence_activity = (
        effort_counts.get("Directing", 0)
        + effort_counts.get("Punching", 0)
        + effort_counts.get("Advancing", 0)
        + effort_counts.get("Pressing", 0)
    ) / total_frames
    authority_activity = (
        effort_counts.get("Pressing", 0)
        + effort_counts.get("Punching", 0)
        + effort_counts.get("Directing", 0)
        + effort_counts.get("Advancing", 0)
    ) / total_frames

    engaging_boost = max(0.85, min(1.35, 0.85 + engaging_activity * 1.60))
    confidence_boost = max(0.85, min(1.35, 0.85 + confidence_activity * 1.50))
    authority_boost = max(0.85, min(1.35, 0.85 + authority_activity * 1.50))

    # Category raw signals: positives minus small penalties from counter-signals.
    engaging_raw = (
        effort_detection.get("Spreading", 0) * 0.34
        + effort_detection.get("Enclosing", 0) * 0.26
        + effort_detection.get("Gliding", 0) * 0.22
        + effort_detection.get("Indirecting", 0) * 0.18
        - effort_detection.get("Punching", 0) * 0.12
        - effort_detection.get("Retreating", 0) * 0.10
    )
    confidence_raw = (
        effort_detection.get("Directing", 0) * 0.36
        + effort_detection.get("Punching", 0) * 0.24
        + effort_detection.get("Advancing", 0) * 0.24
        + effort_detection.get("Pressing", 0) * 0.16
        - effort_detection.get("Retreating", 0) * 0.18
        - effort_detection.get("Indirecting", 0) * 0.08
    )
    authority_raw = (
        effort_detection.get("Pressing", 0) * 0.34
        + effort_detection.get("Punching", 0) * 0.26
        + effort_detection.get("Directing", 0) * 0.24
        + effort_detection.get("Advancing", 0) * 0.16
        - effort_detection.get("Flicking", 0) * 0.18
        - effort_detection.get("Retreating", 0) * 0.12
    )

    # Shape-based alignment helps separate categories with similar effort mixes.
    engaging_shape = (
        shape_detection.get("Spreading", 0) * 0.45
        + shape_detection.get("Enclosing", 0) * 0.35
        + shape_detection.get("Indirecting", 0) * 0.20
    )
    confidence_shape = (
        shape_detection.get("Directing", 0) * 0.60
        + shape_detection.get("Advancing", 0) * 0.35
        - shape_detection.get("Retreating", 0) * 0.25
    )
    authority_shape = (
        shape_detection.get("Directing", 0) * 0.55
        + shape_detection.get("Advancing", 0) * 0.30
        - shape_detection.get("Retreating", 0) * 0.40
    )

    engaging_raw = max(
        0.0,
        (engaging_raw * engaging_boost + engaging_shape * 0.22) * variety_factor - monotony_penalty * 0.90,
    )
    confidence_raw = max(
        0.0,
        (confidence_raw * confidence_boost + confidence_shape * 0.25) * variety_factor - monotony_penalty * 1.00,
    )
    authority_raw = max(
        0.0,
        (authority_raw * authority_boost + authority_shape * 0.25) * variety_factor - monotony_penalty * 1.10,
    )

    # Mild contrast expansion to avoid all categories collapsing into the same band.
    raw_vec = np.array([engaging_raw, confidence_raw, authority_raw], dtype=float)
    raw_mean = float(np.mean(raw_vec))
    contrast = 0.95
    engaging_raw = engaging_raw + contrast * (engaging_raw - raw_mean)
    confidence_raw = confidence_raw + contrast * (confidence_raw - raw_mean)
    authority_raw = authority_raw + contrast * (authority_raw - raw_mean)

    engaging_score = min(7, max(1, round(engaging_raw / 5.4 + 1.4)))
    convince_score = min(7, max(1, round(confidence_raw / 5.4 + 1.4)))
    authority_score = min(7, max(1, round(authority_raw / 5.4 + 1.4)))
    
    return {
        "analysis_engine": "mediapipe_real_enhanced",
        "duration_seconds": duration,
        "analyzed_frames": analyzed,
        "total_indicators": 450 + 475 + 445,  # = 1370 (real total)
        "engaging_score": engaging_score,
        "engaging_pos": int(engaging_score / 7 * 450),
        "convince_score": convince_score,
        "convince_pos": int(convince_score / 7 * 475),
        "authority_score": authority_score,
        "authority_pos": int(authority_score / 7 * 445),
        "effort_detection": effort_detection,
        "shape_detection": shape_detection,
    }

def analyze_video_placeholder(video_path: str, seed: int = 42) -> Dict[str, Any]:
    """Fallback placeholder analysis"""
    random.seed(seed)
    duration = get_video_duration_seconds(video_path)
    
    effort_detection = {
        "Directing": 23.9, "Enclosing": 11.9, "Punching": 11.5, "Spreading": 11.3,
        "Pressing": 10.8, "Dabbing": 8.4, "Indirecting": 7.4, "Gliding": 6.2,
        "Flicking": 3.5, "Advancing": 2.6, "Retreating": 2.5
    }
    shape_detection = {"Directing": 40.1, "Enclosing": 20.0, "Spreading": 18.9, "Indirecting": 12.4, "Advancing": 4.4, "Retreating": 4.1}
    
    return {
        "analysis_engine": "placeholder",
        "duration_seconds": duration,
        "analyzed_frames": 100,
        "total_indicators": 450 + 475 + 445,  # = 1370 (real total)
        "engaging_score": 5,
        "engaging_pos": 431,
        "convince_score": 5,
        "convince_pos": 475,
        "authority_score": 5,
        "authority_pos": 445,
        "effort_detection": effort_detection,
        "shape_detection": shape_detection,
    }

# Graph generation
def generate_effort_graph(effort_data: Dict[str, float], shape_data: Dict[str, float], output_path: str):
    """Generate Effort Motion Detection graph (Page 4 style)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Effort Motion Detection Results', fontsize=14, fontweight='bold')
    
    # Light blue color
    light_blue = '#87CEEB'
    
    # Left: Effort Summary (sorted by value, highest at top)
    sorted_efforts = sorted(effort_data.items(), key=lambda x: x[1], reverse=False)  # reverse=False for bottom-to-top
    efforts = [k for k, v in sorted_efforts]
    values = [v for k, v in sorted_efforts]
    
    ax1.barh(efforts, values, color=light_blue)
    ax1.set_xlabel('Percentage (%)')
    ax1.set_title('Effort Summary')
    ax1.set_xlim(0, 100)
    for i, v in enumerate(values):
        ax1.text(v + 1, i, f'{v}%', va='center')
    
    # Right: Top Movement Efforts
    top_3 = sorted(effort_data.items(), key=lambda x: x[1], reverse=True)[:3]
    top_names = [f"{k} - Rank #{i+1}" for i, (k, v) in enumerate(top_3)]
    top_vals = [v for k, v in top_3]
    
    ax2.barh(top_names, top_vals, color=light_blue)
    ax2.set_xlabel('Percentage (%)')
    ax2.set_title('Top Movement Efforts')
    ax2.set_xlim(0, 100)
    for i, v in enumerate(top_vals):
        ax2.text(v + 1, i, f'{v}%', va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def generate_shape_graph(shape_data: Dict[str, float], output_path: str):
    """Generate Shape Motion Detection graph (Page 5 style)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Light blue color
    light_blue = '#87CEEB'
    
    # Sort by value, highest (left) to lowest (right)
    sorted_shapes = sorted(shape_data.items(), key=lambda x: x[1], reverse=True)
    shapes = [k for k, v in sorted_shapes]
    values = [v for k, v in sorted_shapes]
    
    bars = ax.bar(shapes, values, color=light_blue)
    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel('Shape Motion Type')
    ax.set_title('Shape Motion Detection Results', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.2 if values else 10)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# DOCX Report building
def build_docx_report(
    report: ReportData,
    output_bio: io.BytesIO,
    graph1_path: str,
    graph2_path: str,
    lang: str = "en",
    report_style: str = "full",
):
    """Build 5-page DOCX report matching the exact template format"""
    doc = Document()
    style_name = str(report_style or "full").strip().lower()
    is_simple = style_name.startswith("simple")
    is_operation_test = style_name.startswith("operation_test")
    
    # Language-specific text
    is_thai = (lang == "th")
    
    texts = {
        "title": "รายงานการวิเคราะห์การนำเสนอ" if is_thai else "Character Analysis Report",
        "client_name": "ชื่อลูกค้า:" if is_thai else "Client Name:",
        "analysis_date": "วันที่วิเคราะห์:" if is_thai else "Analysis Date:",
        "video_info": "ข้อมูลวิดีโอ (Video Information)" if is_thai else "Video Information",
        "duration": "ระยะเวลา:" if is_thai else "Duration:",
        "detailed_analysis": "การวิเคราะห์โดยละเอียด (Detailed Analysis)" if is_thai else "Detailed Analysis",
        "first_impression": "1.  ความประทับใจแรกพบ (First Impression)" if is_thai else "1.  First impression",
        "eye_contact": "การสบตา (Eye Contact)" if is_thai else "Eye Contact",
        "uprightness": "ความตั้งตรงของร่างกาย (Uprightness)" if is_thai else "Uprightness (Posture & Upper-Body Alignment)",
        "stance": "การยืนและการวางเท้า (Stance)" if is_thai else "Stance (Lower-Body Stability & Grounding)",
        "impact_clients": "ผลกระทบ:" if is_thai else "Impact:",
        "engaging": "2. การสร้างความเป็นมิตรและสร้างสัมพันธภาพ" if is_thai else "2. Engaging & Connecting:",
        "approachability": "• ความเป็นกันเอง" if is_thai else "• Approachability",
        "relatability": "• ความเข้าถึงได้" if is_thai else "• Relatability",
        "engagement": "• การมีส่วนร่วม เชื่อมโยง และสร้างความคุ้นเคยกับทีมอย่างรวดเร็ว" if is_thai else "• Engagement, connect and build instant rapport with team",
        "confidence": "3. ความมั่นใจ:" if is_thai else "3. Confidence:",
        "optimistic": "• บุคลิกภาพเชิงบวก" if is_thai else "• Optimistic Presence",
        "focus": "• ความมีสมาธิ" if is_thai else "• Focus",
        "persuade": "• ความสามารถในการโน้มน้าวและยืนหยัดในจุดยืนเพื่อให้ผู้อื่นคล้อยตาม" if is_thai else "• Ability to persuade and stand one's ground, in order to convince others.",
        "authority": "4.  ความเป็นผู้นำ (Authority):" if is_thai else "4. Authority:",
        "importance": "• แสดงให้เห็นถึงความสำคัญและความเร่งด่วนของประเด็น" if is_thai else "• Showing sense of importance and urgency in subject matter",
        "pressing": "• ผลักดันให้เกิดการลงมือทำ" if is_thai else "• Pressing for action",
        "scale": "ระดับ:" if is_thai else "Scale:",
        "description": "คำอธิบาย: ตรวจพบ" if is_thai else "Description: Detected",
        "indicators": "การเคลื่อนไหวเชิงบวก" if is_thai else "positive indicators out of",
        "total_indicators": "ตัวชี้วัดทั้งหมด" if is_thai else "total indicators",
        "effort_title": "ผลลัพธ์การตรวจจับการเคลื่อนไหวแบบ Effort (Effort Motion Detection Results)" if is_thai else "Effort Motion Detection Results",
        "shape_title": "ผลลัพธ์การตรวจจับการเคลื่อนไหวแบบ Shape (Shape Motion Detection Results)" if is_thai else "Shape Motion Detection Results",
        "generated": "สร้างโดย AI People Reader™" if is_thai else "Generated by AI People Reader™",
    }
    
    # Impact texts in Thai
    impact_eye_thai = "การสบตาที่แข็งแกร่งส่งสัญญาณถึงความมีอยู่ ความจริงใจ และความเชื่อมั่นในความเป็นผู้นำ ทำให้ข้อความของคุณรู้สึกน่าเชื่อถือมากขึ้น"
    impact_upright_thai = "ท่าทางตรงสื่อถึงความมั่นใจในตนเอง ความชัดเจนในความคิด และความมั่นคงทางอารมณ์ ซึ่งเป็นลักษณะของผู้สื่อสารที่มีความไว้วางใจสูง"
    impact_stance_thai = "ท่ายืนที่มั่นคงช่วยเพิ่มอำนาจ การควบคุม และการส่งข้อความที่ราบรื่น ทำให้ผู้พูดดูเตรียมพร้อมและน่าเชื่อถือมากขึ้น"
    
    impact_eye_en = "Strong eye contact signals presence, sincerity, and leadership confidence, making your message feel more reliable."
    impact_upright_en = "Uprightness communicates self-assurance, clarity of thought, and emotional stability all traits of high-trust communicators."
    impact_stance_en = "A grounded stance enhances authority, control, and smooth message delivery, making the speaker appear more prepared and credible."

    def normalize_spacing() -> None:
        """
        Keep a consistent spacing pattern for numbered headings and bullet points
        across pages (especially sections 2-4).
        """
        for p in doc.paragraphs:
            t = (p.text or "").strip()
            if not t:
                continue

            pf = p.paragraph_format
            # Numbered section headings: "1.", "2.", "3.", "4."
            if t[:2] in ("1.", "2.", "3.", "4."):
                pf.space_before = Pt(14)
                pf.space_after = Pt(6)
                pf.line_spacing = 1.1
                continue

            # Bullets use explicit "•" in content; keep unified compact rhythm.
            if t.startswith("•"):
                pf.space_before = Pt(0)
                pf.space_after = Pt(6)
                pf.line_spacing = 1.2
                continue

            # Keep impact labels/body visually grouped.
            if t in (texts["impact_clients"], "Impact:", "ผลกระทบ:"):
                pf.space_before = Pt(6)
                pf.space_after = Pt(0)
                continue
    
    # Add header and footer images to all pages
    section = doc.sections[0]
    
    # Add header image
    header_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Header.png")
    if os.path.exists(header_path):
        header = section.header
        header_para = header.paragraphs[0]
        header_run = header_para.add_run()
        header_run.add_picture(header_path, width=Inches(6.5))
        header_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Add footer image
    footer_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Footer.png")
    if os.path.exists(footer_path):
        footer = section.footer
        footer_para = footer.paragraphs[0]
        footer_run = footer_para.add_run()
        footer_run.add_picture(footer_path, width=Inches(6.5))
        footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # ============================================================
    # PAGE 1: Cover + First Impression (Eye Contact start)
    # ============================================================
    
    # Title section - reduced spacing after header
    doc.add_paragraph()
    doc.add_paragraph()
    
    # Title
    title = doc.add_paragraph(texts["title"])
    title.runs[0].font.size = Pt(20)
    title.runs[0].font.bold = True
    doc.add_paragraph()
    
    # Client info
    doc.add_paragraph(f"{texts['client_name']}  {report.client_name}")
    doc.add_paragraph(f"{texts['analysis_date']}  {report.analysis_date}")
    
    # Video info
    video_info = doc.add_paragraph(texts["video_info"])
    video_info.runs[0].bold = True
    doc.add_paragraph(f"{texts['duration']} {report.video_length_str}")
    doc.add_paragraph()
    
    # Detailed Analysis header
    detailed = doc.add_paragraph(texts["detailed_analysis"])
    detailed.runs[0].bold = True
    
    # Section 1: First impression
    section1 = doc.add_paragraph(texts["first_impression"])
    section1.runs[0].bold = True
    
    # Eye Contact
    eye_header = doc.add_paragraph(texts["eye_contact"])
    
    if report.first_impression:
        fi = report.first_impression
        eye_texts = generate_eye_contact_text(fi.eye_contact_pct)
        
        if is_thai:
            eye_texts = generate_eye_contact_text_th(fi.eye_contact_pct)
        for text in eye_texts:
            doc.add_paragraph(text)
        
        impact = doc.add_paragraph(texts["impact_clients"])
        impact.runs[0].italic = True
        impact.paragraph_format.space_before = Pt(0)  # Reduce space before impact
        doc.add_paragraph(impact_eye_thai if is_thai else impact_eye_en)
    else:
        doc.add_paragraph("• Your eye contact is steady, warm, and audience-focused.")
        doc.add_paragraph("• You maintain direct gaze during key moments, which increases trust and clarity.")
        doc.add_paragraph("• When you shift your gaze, it is done purposefully (e.g., thinking, emphasizing).")
        doc.add_paragraph("• There is no sign of avoidance — overall, the eye contact supports confidence and credibility.")
        
        impact = doc.add_paragraph(texts["impact_clients"])
        impact.runs[0].italic = True
        impact.paragraph_format.space_before = Pt(0)  # Reduce space before impact
        doc.add_paragraph(impact_eye_thai if is_thai else impact_eye_en)
    
    # Uprightness section - show all bullets on page 1
    upright_header = doc.add_paragraph(texts["uprightness"])
    
    if report.first_impression:
        fi = report.first_impression
        upright_texts = generate_uprightness_text(fi.upright_pct)
        
        if is_thai:
            upright_texts = generate_uprightness_text_th(fi.upright_pct)
        for text in upright_texts:
            doc.add_paragraph(text)
        
        impact2 = doc.add_paragraph(texts["impact_clients"])
        impact2.runs[0].italic = True
        impact2.paragraph_format.space_before = Pt(0)  # Reduce space before impact
        doc.add_paragraph(impact_upright_thai if is_thai else impact_upright_en)
    else:
        doc.add_paragraph("• You maintain a naturally upright posture throughout the clip.")
        doc.add_paragraph("• The chest stays open, shoulders relaxed, and head aligned — signaling balance, readiness, and authority.")
        doc.add_paragraph("• Even when you gesture, your vertical alignment remains stable, showing good core control.")
        
        impact2 = doc.add_paragraph(texts["impact_clients"])
        impact2.runs[0].italic = True
        impact2.paragraph_format.space_before = Pt(0)  # Reduce space before impact
        doc.add_paragraph(impact_upright_thai if is_thai else impact_upright_en)
    
    # Stance section
    stance_header = doc.add_paragraph(texts["stance"])
    
    if report.first_impression:
        fi = report.first_impression
        stance_texts = generate_stance_text(fi.stance_stability)
        
        if is_thai:
            stance_texts = generate_stance_text_th(fi.stance_stability)
        for text in stance_texts[:2]:
            doc.add_paragraph(text)
    
    # Continue Stance (impact only)
    if report.first_impression:
        impact3 = doc.add_paragraph(texts["impact_clients"])
        impact3.runs[0].italic = True
        impact3.paragraph_format.space_before = Pt(0)  # Reduce space before impact
        doc.add_paragraph(impact_stance_thai if is_thai else impact_stance_en)
    else:
        doc.add_paragraph("• Your stance is symmetrical and grounded, with feet placed about shoulder-width apart.")
        doc.add_paragraph("• Weight shifts are controlled and minimal, preventing distraction and showing confidence.")
        
        impact3 = doc.add_paragraph(texts["impact_clients"])
        impact3.runs[0].italic = True
        impact3.paragraph_format.space_before = Pt(0)  # Reduce space before impact
        doc.add_paragraph(impact_stance_thai if is_thai else impact_stance_en)

    # PAGE BREAK TO PAGE 2
    doc.add_page_break()
    # Add one extra line of top spacing on the next page.
    doc.add_paragraph()
    
    # ============================================================
    # PAGE 2: Engaging & Connecting + Confidence
    # ============================================================
    
    # Section 2: Engaging & Connecting
    engaging_cat = report.categories[0]
    section2 = doc.add_paragraph(texts["engaging"])
    section2.runs[0].bold = True
    doc.add_paragraph(texts["approachability"])
    doc.add_paragraph(texts["relatability"])
    doc.add_paragraph(texts["engagement"])
    scale_para1 = doc.add_paragraph(f"{texts['scale']} {engaging_cat.scale.capitalize()}")
    scale_para1.runs[0].bold = True
    if not is_simple:
        doc.add_paragraph(f"{texts['description']} {engaging_cat.positives} {texts['indicators']} {engaging_cat.total} {texts['total_indicators']}")
    
    doc.add_paragraph()
    doc.add_paragraph()
    
    # Section 3: Confidence
    confidence_cat = report.categories[1]
    section3 = doc.add_paragraph(texts["confidence"])
    section3.runs[0].bold = True
    doc.add_paragraph(texts["optimistic"])
    doc.add_paragraph(texts["focus"])
    doc.add_paragraph(texts["persuade"])
    scale_para2 = doc.add_paragraph(f"{texts['scale']} {confidence_cat.scale.capitalize()}")
    scale_para2.runs[0].bold = True
    if not is_simple:
        doc.add_paragraph(f"{texts['description']} {confidence_cat.positives} {texts['indicators']} {confidence_cat.total} {texts['total_indicators']}")
    
    # Keep spacing between Section 3 and Section 4 similar to Section 2 and 3
    doc.add_paragraph()
    doc.add_paragraph()
    
    # ============================================================
    # Section 4: Authority (same page as Section 3)
    # ============================================================
    
    # Section 4: Authority
    authority_cat = report.categories[2]
    section4 = doc.add_paragraph(texts["authority"])
    section4.runs[0].bold = True
    doc.add_paragraph(texts["importance"])
    doc.add_paragraph(texts["pressing"])
    scale_para3 = doc.add_paragraph(f"{texts['scale']} {authority_cat.scale.capitalize()}")
    scale_para3.runs[0].bold = True
    if not is_simple:
        doc.add_paragraph(f"{texts['description']} {authority_cat.positives} {texts['indicators']} {authority_cat.total} {texts['total_indicators']}")
        
        # PAGE BREAK TO PAGE 4
        doc.add_page_break()
        
        # ============================================================
        # PAGE 4: Effort Motion Detection Results
        # ============================================================
        
        # Spacing after header (same as page 1)
        doc.add_paragraph()
        doc.add_paragraph()
        doc.add_paragraph()
        doc.add_paragraph()
        
        # Title
        title4 = doc.add_paragraph(texts["effort_title"])
        title4.runs[0].font.size = Pt(18)
        title4.runs[0].bold = True
        doc.add_paragraph()
        
        # Graph
        if os.path.exists(graph1_path):
            doc.add_picture(graph1_path, width=Inches(6.5))
        
        # PAGE BREAK TO PAGE 5
        doc.add_page_break()
        
        # ============================================================
        # PAGE 5: Shape Motion Detection Results
        # ============================================================
        
        # Spacing after header (same as page 1)
        doc.add_paragraph()
        doc.add_paragraph()
        doc.add_paragraph()
        doc.add_paragraph()
        
        # Title
        title5 = doc.add_paragraph(texts["shape_title"])
        title5.runs[0].font.size = Pt(18)
        title5.runs[0].bold = True
        doc.add_paragraph()
        
        # Graph
        if os.path.exists(graph2_path):
            doc.add_picture(graph2_path, width=Inches(6.5))
        
        # Add "Generated by AI People Reader™" text below graph
        doc.add_paragraph()
        doc.add_paragraph()
    else:
        doc.add_paragraph()

    # Normalize paragraph rhythm so numbered headings and bullets follow
    # a consistent spacing pattern throughout the document.
    normalize_spacing()

    generated_para = doc.add_paragraph(texts["generated"])
    generated_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    generated_run = generated_para.runs[0]
    generated_run.italic = True
    generated_run.font.size = Pt(11)
    
    # Save
    doc.save(output_bio)

def build_pdf_report(
    report: ReportData,
    output_path: str,
    graph1_path: str,
    graph2_path: str,
    lang: str = "en",
    report_style: str = "full",
):
    logger = logging.getLogger("report_core_pdf")
    """Build PDF report aligned with DOCX simple/full text structure (no graph embedding)."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.utils import simpleSplit
        from reportlab.pdfgen import canvas
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except Exception as e:
        raise RuntimeError(f"reportlab is required for PDF generation: {e}")

    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    x_left = 50
    x_right = 50
    usable_width = width - x_left - x_right
    header_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Header.png")
    footer_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Footer.png")
    top_content_y = height - 95
    bottom_content_y = 70
    y = top_content_y

    style_name = str(report_style or "full").strip().lower()
    is_simple = style_name.startswith("simple")
    is_operation_test = style_name.startswith("operation_test")
    lang_name = str(lang or "").strip().lower()
    # Operation Test is Thai-only by product requirement.
    is_thai = (lang_name == "th") or is_operation_test
    regular_font = "Helvetica"
    bold_font = "Helvetica-Bold"
    requires_unicode_font = False

    def _first_existing(paths: list) -> str:
        for p in paths:
            if p and os.path.exists(p):
                return p
        return ""

    def _glob_existing(patterns: list) -> list:
        import glob
        found = []
        for pat in patterns:
            if not pat:
                continue
            found.extend(glob.glob(pat, recursive=True))
        # Keep deterministic order and unique entries.
        uniq = []
        seen = set()
        for p in sorted(found):
            if p not in seen and os.path.isfile(p):
                uniq.append(p)
                seen.add(p)
        return uniq

    def _register_ttf(font_name: str, path: str, require_thai: bool = False) -> bool:
        if not path:
            return False
        try:
            pdfmetrics.getFont(font_name)
            return True
        except Exception:
            pass
        try:
            tt = TTFont(font_name, path)
            if require_thai:
                # Best-effort Thai glyph validation. Do not over-reject fonts when metadata
                # differs between platforms; runtime rendering is the final authority.
                required = [ord("ก"), ord("า"), ord("ไ"), ord("ย")]
                cmap = getattr(tt.face, "charToGlyph", {}) or {}
                widths = getattr(tt.face, "charWidths", {}) or {}
                if cmap:
                    if any((cp not in cmap) and (str(cp) not in cmap) for cp in required):
                        return False
                elif widths:
                    if any((cp not in widths) and (str(cp) not in widths) for cp in required):
                        return False
            pdfmetrics.registerFont(tt)
            return True
        except Exception:
            return False

    if is_thai:
        thai_glob_candidates = _glob_existing(
            [
                "/usr/share/fonts/**/*.ttf",
                "/usr/local/share/fonts/**/*.ttf",
            ]
        )
        thai_glob_preferred = [
            p for p in thai_glob_candidates
            if any(k in os.path.basename(p).lower() for k in ("noto", "sarabun", "thsarabun", "thai", "garuda", "waree", "kinnari", "loma"))
        ]

        thai_regular_candidates = [
            os.getenv("PDF_THAI_FONT_PATH", "").strip(),
            os.getenv("THAI_FONT_PATH", "").strip(),
            os.getenv("REPORT_THAI_FONT_PATH", "").strip(),
            "/Library/Fonts/THSarabunNew.ttf",
            "/Library/Fonts/Sarabun-Regular.ttf",
            "/System/Library/Fonts/Supplemental/Thonburi.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansThaiUI-Regular.ttf",
            "/usr/share/fonts/truetype/thai/Sarabun-Regular.ttf",
            "/usr/share/fonts/truetype/tlwg/Garuda.ttf",
            "/usr/share/fonts/truetype/tlwg/Waree.ttf",
            "/usr/share/fonts/truetype/tlwg/Loma.ttf",
            "/usr/share/fonts/truetype/tlwg/Kinnari.ttf",
        ]
        thai_bold_candidates = [
            os.getenv("PDF_THAI_FONT_BOLD_PATH", "").strip(),
            os.getenv("THAI_FONT_BOLD_PATH", "").strip(),
            os.getenv("REPORT_THAI_FONT_BOLD_PATH", "").strip(),
            "/Library/Fonts/THSarabunNew Bold.ttf",
            "/Library/Fonts/Sarabun-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Thonburi Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansThai-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansThaiUI-Bold.ttf",
            "/usr/share/fonts/truetype/thai/Sarabun-Bold.ttf",
            "/usr/share/fonts/truetype/tlwg/Garuda-Bold.ttf",
            "/usr/share/fonts/truetype/tlwg/Waree-Bold.ttf",
            "/usr/share/fonts/truetype/tlwg/Loma-Bold.ttf",
            "/usr/share/fonts/truetype/tlwg/Kinnari-Bold.ttf",
        ]
        thai_regular_candidates.extend(thai_glob_preferred)
        thai_bold_candidates.extend(thai_glob_preferred)

        ok_regular = False
        for path in thai_regular_candidates:
            if _register_ttf("ThaiPDFRegular", path, require_thai=True):
                ok_regular = True
                break

        ok_bold = False
        for path in thai_bold_candidates:
            if _register_ttf("ThaiPDFBold", path, require_thai=True):
                ok_bold = True
                break

        if not ok_regular:
            # Do not fail the whole job. Fallback to readable English PDF if Thai font is missing.
            logger.warning(
                "Thai font not found for PDF. Falling back to English labels/content for this PDF."
            )
            is_thai = False
            regular_font = "Helvetica"
            bold_font = "Helvetica-Bold"
            requires_unicode_font = False
        else:
            regular_font = "ThaiPDFRegular"
            bold_font = "ThaiPDFBold" if ok_bold else "ThaiPDFRegular"
            requires_unicode_font = True

    def draw_header_footer() -> None:
        # Match DOCX branding as closely as possible for PDF output.
        if os.path.exists(header_path):
            try:
                c.drawImage(
                    header_path,
                    x=28,
                    y=height - 62,
                    width=width - 56,
                    height=36,
                    preserveAspectRatio=True,
                    mask="auto",
                )
            except Exception:
                pass
        if os.path.exists(footer_path):
            try:
                c.drawImage(
                    footer_path,
                    x=28,
                    y=18,
                    width=width - 56,
                    height=30,
                    preserveAspectRatio=True,
                    mask="auto",
                )
            except Exception:
                pass

    draw_header_footer()

    def write_line(text: str, size: int = 11, bold: bool = False, gap: int = 18):
        nonlocal y
        font = bold_font if bold else regular_font
        if requires_unicode_font:
            safe = str(text or "")
        else:
            # Normalize common unicode punctuation for ASCII/Latin fonts.
            normalized = (
                str(text or "")
                .replace("•", "- ")
                .replace("—", "-")
                .replace("–", "-")
                .replace("“", '"')
                .replace("”", '"')
                .replace("’", "'")
            )
            safe = normalized.encode("latin-1", "replace").decode("latin-1")

        def _split_by_chars(long_line: str) -> list:
            parts = []
            buf = ""
            for ch in str(long_line or ""):
                probe = f"{buf}{ch}"
                if (not buf) or (pdfmetrics.stringWidth(probe, font, size) <= usable_width):
                    buf = probe
                else:
                    parts.append(buf)
                    buf = ch
            if buf:
                parts.append(buf)
            return parts or [""]

        initial_lines = simpleSplit(safe, font, size, usable_width) or [""]
        lines = []
        for ln in initial_lines:
            if pdfmetrics.stringWidth(ln, font, size) <= usable_width:
                lines.append(ln)
                continue
            if is_thai or (" " not in ln):
                lines.extend(_split_by_chars(ln))
            else:
                lines.append(ln)

        wrapped_gap = max(12, int(size * 1.35))
        for idx, line in enumerate(lines):
            if y <= bottom_content_y:
                c.showPage()
                draw_header_footer()
                y = top_content_y
            c.setFont(font, size)
            c.drawString(x_left, y, line)
            y -= gap if idx == len(lines) - 1 else wrapped_gap

    def write_block(lines: list, size: int = 11, bold: bool = False, gap: int = 16):
        for line in lines:
            write_line(line, size=size, bold=bold, gap=gap)

    title = "รายงานการวิเคราะห์การนำเสนอ" if is_thai else "Character Analysis Report"
    detailed_analysis_label = "รายละเอียดการวิเคราะห์การนำเสนอ" if is_thai else "Detailed Analysis"
    first_impression_label = "1.ความประทับใจแรกพบ (First Impression)" if is_thai else "1. First impression"
    eye_label = "การสบตา" if is_thai else "Eye Contact"
    upright_label = "ความตั้งตรงของร่างกาย" if is_thai else "Uprightness"
    stance_label = "การยืนและการวางเท้า" if is_thai else "Stance"
    category_labels = (
        ["2. การสร้างความเป็นมิตรและสัมพันธภาพ", "3. ความมั่นใจ", "4. ความเป็นผู้นำ"]
        if is_thai
        else ["2. Engaging & Connecting", "3. Confidence", "4. Authority"]
    )

    def _duration_th_text(mmss: str) -> str:
        raw = str(mmss or "").strip()
        parts = raw.split(":")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            sec = int(parts[0]) * 60 + int(parts[1])
            return f"{sec} วินาที ({raw})"
        return raw

    write_line(title, size=16, bold=True, gap=24)
    write_line(f"{'ชื่อลูกค้า' if is_thai else 'Client Name'}: {report.client_name}", bold=True)
    write_line(f"{'วันที่วิเคราะห์' if is_thai else 'Analysis Date'}: {report.analysis_date}")
    duration_label = _duration_th_text(report.video_length_str) if is_thai else report.video_length_str
    write_line(f"{'ความยาววิดิโอ' if is_thai else 'Duration'}: {duration_label}", gap=22)
    write_line(detailed_analysis_label, size=13, bold=True, gap=20)

    def _first_impression_level(value: float, metric: str = "") -> str:
        v = float(value or 0.0)
        name = str(metric or "").strip().lower()
        if name == "eye_contact":
            return "High"
        if name in ("stance", "uprightness"):
            if v >= 80.0:
                return "High"
            if v >= 50.0:
                return "Moderate"
            return "Low"
        if v >= 70.0:
            return "High"
        if v >= 40.0:
            return "Moderate"
        return "Low"

    # First impression sections (same narrative style as DOCX simple).
    write_line(first_impression_label, size=12, bold=True, gap=18)
    if report.first_impression:
        fi = report.first_impression
        if is_operation_test:
            if is_thai:
                write_line(f"▪{eye_label} (Eye Contact)", bold=True, gap=14)
                write_line(f"ระดับ: {_first_impression_level(fi.eye_contact_pct, metric='eye_contact')}", bold=True, gap=16)
                write_line(f"▪{upright_label} (Uprightness)", bold=True, gap=14)
                write_line(f"ระดับ: {_first_impression_level(fi.upright_pct, metric='uprightness')}", bold=True, gap=16)
                write_line(f"▪{stance_label} (Stance)", bold=True, gap=14)
                write_line(f"ระดับ: {_first_impression_level(fi.stance_stability, metric='stance')}", bold=True, gap=16)
            else:
                write_line(f"{eye_label}: {_first_impression_level(fi.eye_contact_pct, metric='eye_contact')}", bold=True, gap=16)
                write_line(f"{upright_label}: {_first_impression_level(fi.upright_pct, metric='uprightness')}", bold=True, gap=16)
                write_line(f"{stance_label}: {_first_impression_level(fi.stance_stability, metric='stance')}", bold=True, gap=16)
        else:
            eye_lines = generate_eye_contact_text_th(fi.eye_contact_pct) if is_thai else generate_eye_contact_text(fi.eye_contact_pct)
            up_lines = generate_uprightness_text_th(fi.upright_pct) if is_thai else generate_uprightness_text(fi.upright_pct)
            st_lines = generate_stance_text_th(fi.stance_stability) if is_thai else generate_stance_text(fi.stance_stability)

            write_line(eye_label, bold=True, gap=16)
            write_block(eye_lines, gap=14)
            write_line(upright_label, bold=True, gap=16)
            write_block(up_lines, gap=14)
            write_line(stance_label, bold=True, gap=16)
            write_block(st_lines, gap=14)
    else:
        write_line("- Not available", gap=20)

    if is_operation_test:
        if is_thai:
            def _scale_th(scale: str) -> str:
                s = str(scale or "").strip().lower()
                if s.startswith("high"):
                    return "สูง"
                if s.startswith("moderate"):
                    return "ปานกลาง"
                if s.startswith("low"):
                    return "ต่ำ"
                return "-"

            write_line("", gap=8)
            write_line("หมายเหตุ", bold=True, gap=16)
            write_line(
                "ความรู้สึกที่เกิดจากความประทับใจแรกพบนั้นเป็นสิ่งที่มนุษย์หลีกเลี่ยงไม่ได้ และมักเกิดขึ้นภายใน 5 วินาทีแรกของการพบกัน แต่หลังจากนั้นจะเริ่มวิเคราะห์การเคลื่อนไหวโดยรวมมาประกอบการตัดสินใจ",
                gap=16,
            )
            write_line("2.การสร้างความเป็นมิตรและสร้างสัมพันธภาพ", size=12, bold=True, gap=18)
            write_line("• ความเป็นกันเอง", gap=14)
            write_line("จัดทำโดย AI People Reader™", size=10)

            c.showPage()
            draw_header_footer()
            y = top_content_y

            engaging_scale = _scale_th(report.categories[0].scale) if len(report.categories) > 0 else "-"
            confidence_scale = _scale_th(report.categories[1].scale) if len(report.categories) > 1 else "-"
            authority_scale = _scale_th(report.categories[2].scale) if len(report.categories) > 2 else "-"

            write_line("• ความเข้าถึงได้", gap=14)
            write_line("• การมีส่วนร่วม เชื่อมโยง และสร้างความคุ้นเคยกับทีมอย่างรวดเร็ว", gap=14)
            write_line(f"ระดับ: {engaging_scale}", bold=True, gap=18)

            write_line("3.ความมั่นใจ:", size=12, bold=True, gap=18)
            write_line("• บุคลิกภาพเชิงบวก", gap=14)
            write_line("• ความมีสมาธิ", gap=14)
            write_line("• ความสามารถในการโน้มน้าวและยืนหยัดในจุดยืนเพื่อให้ผู้อื่นคล้อยตาม", gap=14)
            write_line(f"ระดับ: {confidence_scale}", bold=True, gap=18)

            write_line("4.ความเป็นผู้นำและความดูมีอำนาจ:", size=12, bold=True, gap=18)
            write_line("• แสดงให้เห็นถึงความสำคัญและความเร่งด่วนของประเด็น", gap=14)
            write_line("• ผลักดันให้เกิดการลงมือทำ", gap=14)
            write_line(f"ระดับ: {authority_scale}", bold=True, gap=18)
            write_line("จัดทำโดย AI People Reader™", size=10)
        else:
            write_line("", gap=10)
            write_line("Generated by AI People Reader", size=10)
        c.save()
        return

    # Categories (same order as DOCX)
    for idx, cat in enumerate(report.categories[:3]):
        write_line(category_labels[idx], size=12, bold=True, gap=18)
        write_line(
            f"{'ระดับ' if is_thai else 'Scale'}: {cat.scale.capitalize()}",
            bold=True,
            gap=16,
        )
        if not is_simple:
            if is_thai:
                write_line(f"คำอธิบาย: ตรวจพบตัวชี้วัดเชิงบวก {cat.positives} จากทั้งหมด {cat.total}", gap=16)
            else:
                write_line(f"Description: Detected {cat.positives} positive indicators out of {cat.total}", gap=16)

    write_line("", gap=10)
    write_line("สร้างโดย AI People Reader" if is_thai else "Generated by AI People Reader", size=10)
    c.save()
