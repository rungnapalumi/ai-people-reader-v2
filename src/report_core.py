# report_core.py — shared report logic for report generation
import os
import io
import math
import random
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
                        
                        # Movement direction
                        forward_movement = (left_wrist.z - prev_left_wrist.z) < -0.01 or (right_wrist.z - prev_right_wrist.z) < -0.01
                        backward_movement = (left_wrist.z - prev_left_wrist.z) > 0.01 or (right_wrist.z - prev_right_wrist.z) > 0.01
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
                        
                        # 10. ADVANCING: Moving body/hands forward in space
                        if forward_movement and avg_velocity > 0.05:
                            effort_counts["Advancing"] += 1
                            shape_counts["Advancing"] += 1
                        
                        # 11. RETREATING: Pulling back, withdrawing
                        if backward_movement and avg_velocity > 0.05:
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
    
    # Calculate category scores (based on dominant movements)
    # Engaging & Connecting: Spreading, Enclosing, Gliding (openness, warmth)
    engaging_score = min(7, max(1, int(
        (effort_detection.get("Spreading", 0) * 0.4 +
         effort_detection.get("Enclosing", 0) * 0.3 +
         effort_detection.get("Gliding", 0) * 0.3) / 10 + 2
    )))
    
    # Confidence: Directing, Punching, Advancing (assertiveness, clarity)
    convince_score = min(7, max(1, int(
        (effort_detection.get("Directing", 0) * 0.4 +
         effort_detection.get("Punching", 0) * 0.3 +
         effort_detection.get("Advancing", 0) * 0.3) / 10 + 2
    )))
    
    # Authority: Pressing, Punching, Directing (power, command)
    authority_score = min(7, max(1, int(
        (effort_detection.get("Pressing", 0) * 0.4 +
         effort_detection.get("Punching", 0) * 0.3 +
         effort_detection.get("Directing", 0) * 0.3) / 10 + 2
    )))
    
    return {
        "analysis_engine": "mediapipe_real_enhanced",
        "duration_seconds": duration,
        "analyzed_frames": analyzed,
        "total_indicators": 900,
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
        "total_indicators": 900,
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
    
    # Left: Effort Summary
    efforts = list(effort_data.keys())
    values = [effort_data[k] for k in efforts]
    
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
    
    shapes = list(shape_data.keys())
    values = [shape_data[k] for k in shapes]
    
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
def build_docx_report(report: ReportData, output_bio: io.BytesIO, graph1_path: str, graph2_path: str, lang: str = "en"):
    """Build complete DOCX report matching the template"""
    doc = Document()
    
    # Page 1: Header
    header = doc.add_paragraph()
    run = header.add_run("iPEOPLE READER")
    run.font.size = Pt(36)
    run.font.color.rgb = RGBColor(192, 0, 0)
    run.font.name = 'Arial'
    header.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_paragraph()
    
    title = doc.add_paragraph("Character Analysis Report")
    title.runs[0].font.size = Pt(20)
    title.runs[0].bold = True
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    
    doc.add_paragraph()
    doc.add_paragraph(f"Client Name: {report.client_name}")
    doc.add_paragraph(f"Analysis Date: {report.analysis_date}")
    doc.add_paragraph()
    
    doc.add_paragraph("Video Information").runs[0].bold = True
    doc.add_paragraph(f"Duration: {report.video_length_str}")
    doc.add_paragraph()
    
    doc.add_paragraph("Detailed Analysis").runs[0].bold = True
    doc.add_paragraph()
    
    # === SECTION 1: FIRST IMPRESSION ===
    section1 = doc.add_paragraph("1. First Impression")
    section1.runs[0].bold = True
    section1.runs[0].font.size = Pt(16)
    doc.add_paragraph()
    
    # 1.1 Eye Contact
    subsection = doc.add_paragraph("1.1 Eye Contact")
    subsection.runs[0].bold = True
    subsection.runs[0].font.size = Pt(14)
    doc.add_paragraph("• Your eye contact is steady, warm, and audience-focused.")
    doc.add_paragraph("• You maintain direct gaze during key message points, which increases trust and clarity.")
    doc.add_paragraph("• When you shift your gaze, it is done purposefully (e.g., thinking, emphasizing).")
    doc.add_paragraph("• There is no sign of avoidance — overall, the eye contact supports confidence and credibility.")
    
    impact = doc.add_paragraph("Impact for clients:")
    impact.runs[0].italic = True
    doc.add_paragraph("Strong eye contact signals presence, sincerity, and leadership confidence, making your message feel more reliable.")
    
    doc.add_paragraph()
    
    # 1.2 Uprightness
    subsection2 = doc.add_paragraph("1.2 Uprightness (Posture & Upper-Body Alignment)")
    subsection2.runs[0].bold = True
    subsection2.runs[0].font.size = Pt(14)
    doc.add_paragraph("• You maintain a naturally upright posture throughout the clip.")
    
    # Add page break
    doc.add_page_break()
    
    # Page 2: Continue Uprightness
    doc.add_paragraph("• The chest stays open, shoulders relaxed, and head aligned — signaling balance, readiness, and authority.")
    doc.add_paragraph("• Even when you gesture, your vertical alignment remains stable, showing good core control.")
    doc.add_paragraph("• There is no visible slouching or collapsing, which supports a professional appearance.")
    
    impact2 = doc.add_paragraph("Impact for clients:")
    impact2.runs[0].italic = True
    doc.add_paragraph("Uprightness communicates self-assurance, clarity of thought, and emotional stability all traits of high-trust communicators.")
    
    doc.add_paragraph()
    
    # 1.3 Stance
    subsection3 = doc.add_paragraph("1.3 Stance (Lower-Body Stability & Grounding)")
    subsection3.runs[0].bold = True
    subsection3.runs[0].font.size = Pt(14)
    doc.add_paragraph("• Your stance is symmetrical and grounded, with feet placed about shoulder-width apart.")
    doc.add_paragraph("• Weight shifts are controlled and minimal, preventing distraction and showing confidence.")
    doc.add_paragraph("• You maintain good forward orientation toward the audience, reinforcing clarity and engagement.")
    doc.add_paragraph("• The stance conveys both stability and a welcoming presence, suitable for instructional or coaching communication.")
    
    impact3 = doc.add_paragraph("Impact for clients:")
    impact3.runs[0].italic = True
    doc.add_paragraph("A grounded stance enhances authority, control, and smooth message delivery, making the speaker appear more prepared and credible.")
    
    doc.add_paragraph()
    doc.add_paragraph()
    
    # === SECTION 2: ENGAGING & CONNECTING ===
    section2 = doc.add_paragraph("2. Engaging & Connecting")
    section2.runs[0].bold = True
    section2.runs[0].font.size = Pt(16)
    doc.add_paragraph("• Approachability")
    doc.add_paragraph("• Relatability")
    doc.add_paragraph("• Engagement, connect and build instant rapport with team")
    
    # Add page break
    doc.add_page_break()
    
    # === PAGE 3: CATEGORY SCORES ===
    section3 = doc.add_paragraph("3. Category Analysis")
    section3.runs[0].bold = True
    section3.runs[0].font.size = Pt(16)
    doc.add_paragraph()
    
    for idx, cat in enumerate(report.categories, start=1):
        name = cat.name_en if lang == "en" else cat.name_th
        
        # Category header with score
        cat_header = doc.add_paragraph(f"3.{idx} {name} (Score: {cat.score}/7)")
        cat_header.runs[0].bold = True
        cat_header.runs[0].font.size = Pt(14)
        
        # Key indicators
        if "Engaging" in cat.name_en:
            doc.add_paragraph("• Optimistic Presence")
            doc.add_paragraph("• Focus")
            doc.add_paragraph("• Ability to persuade and stand one's ground, in order to convince others.")
        elif "Confidence" in cat.name_en:
            doc.add_paragraph("• Optimistic Presence")
            doc.add_paragraph("• Focus")
            doc.add_paragraph("• Ability to persuade and stand one's ground, in order to convince others.")
        elif "Authority" in cat.name_en:
            doc.add_paragraph("• Showing sense of importance and urgency in subject matter")
            doc.add_paragraph("• Pressing for action")
        
        doc.add_paragraph()
        
        # Scale and description
        scale_para = doc.add_paragraph(f"Scale: {cat.scale.capitalize()}")
        scale_para.runs[0].bold = True
        desc_text = f"Description: Detected {cat.positives} positive indicators out of {cat.total} total indicators"
        doc.add_paragraph(desc_text)
        doc.add_paragraph()
    
    # Add page break
    doc.add_page_break()
    
    # === PAGE 4: EFFORT MOTION DETECTION ===
    title4 = doc.add_paragraph("4. Effort Motion Detection Results")
    title4.runs[0].font.size = Pt(16)
    title4.runs[0].bold = True
    doc.add_paragraph()  # spacing
    if os.path.exists(graph1_path):
        doc.add_picture(graph1_path, width=Inches(6.5))
    
    # Add page break
    doc.add_page_break()
    
    # === PAGE 5: SHAPE MOTION DETECTION ===
    title5 = doc.add_paragraph("5. Shape Motion Detection Results")
    title5.runs[0].font.size = Pt(16)
    title5.runs[0].bold = True
    doc.add_paragraph()  # spacing
    if os.path.exists(graph2_path):
        doc.add_picture(graph2_path, width=Inches(6.5))
    
    # Footer
    doc.add_paragraph()
    footer = doc.add_paragraph(report.generated_by)
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.runs[0].italic = True
    
    doc.save(output_bio)

def build_pdf_report(report: ReportData, output_path: str, graph1_path: str, graph2_path: str, lang: str = "en"):
    """Build PDF report (placeholder - requires reportlab fonts setup)"""
    # For now, skip PDF generation as it requires Thai fonts
    # The DOCX is the primary output
    pass
