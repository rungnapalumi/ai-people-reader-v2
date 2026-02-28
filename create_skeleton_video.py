#!/usr/bin/env python3
"""
Create a skeleton overlay video from an input video file.
Usage:
  python create_skeleton_video.py <input_video.mp4>
  python create_skeleton_video.py <input_video.mp4> <output_skeleton.mp4>
"""
import os
import sys

import cv2
import numpy as np

try:
    from mediapipe.python.solutions.pose import Pose, PoseLandmark
except ImportError:
    import mediapipe as mp
    Pose = mp.solutions.pose.Pose
    PoseLandmark = mp.solutions.pose.PoseLandmark

# Skeleton structure (same as worker)
POSE_LANDMARK_IDS = [
    PoseLandmark.NOSE,
    PoseLandmark.LEFT_EYE,
    PoseLandmark.RIGHT_EYE,
    PoseLandmark.LEFT_SHOULDER,
    PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.LEFT_ELBOW,
    PoseLandmark.RIGHT_ELBOW,
    PoseLandmark.LEFT_WRIST,
    PoseLandmark.RIGHT_WRIST,
    PoseLandmark.LEFT_HIP,
    PoseLandmark.RIGHT_HIP,
    PoseLandmark.LEFT_KNEE,
    PoseLandmark.RIGHT_KNEE,
    PoseLandmark.LEFT_ANKLE,
    PoseLandmark.RIGHT_ANKLE,
]

SKELETON_EDGES = [
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
    (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
    (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
    (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
    (PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),
    (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
    (PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE),
]


def _lm_to_px(lm, w: int, h: int):
    return int(lm.x * w), int(lm.y * h)


def generate_skeleton_video(input_path: str, out_path: str) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not vw.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {out_path}")

    frame_idx = 0
    with Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark
                for a, b in SKELETON_EDGES:
                    la, lb = lms[a], lms[b]
                    if la.visibility < 0.5 or lb.visibility < 0.5:
                        continue
                    xa, ya = _lm_to_px(la, w, h)
                    xb, yb = _lm_to_px(lb, w, h)
                    cv2.line(frame, (xa, ya), (xb, yb), (0, 255, 0), 3)
                for pid in POSE_LANDMARK_IDS:
                    lm = lms[pid]
                    if lm.visibility < 0.5:
                        continue
                    x, y = _lm_to_px(lm, w, h)
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            vw.write(frame)
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx} frames...")

    vw.release()
    cap.release()
    print(f"Done. Output: {out_path} ({frame_idx} frames)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python create_skeleton_video.py <input_video.mp4> [output_skeleton.mp4]")
        print("\nExample:")
        print("  python create_skeleton_video.py my_video.mp4")
        print("  python create_skeleton_video.py my_video.mp4 skeleton_output.mp4")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.isfile(input_path):
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
    else:
        base, _ = os.path.splitext(input_path)
        out_path = f"{base}_skeleton.mp4"

    print(f"Input:  {input_path}")
    print(f"Output: {out_path}")
    print("Processing...")
    generate_skeleton_video(input_path, out_path)


if __name__ == "__main__":
    main()
