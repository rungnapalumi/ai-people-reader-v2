#!/usr/bin/env python3
"""
Benchmark: How long does MediaPipe Pose detection take to process a single frame?
Uses the same Pose config as the production worker (model_complexity=1, enable_segmentation=False).
"""
import time
import numpy as np

try:
    from mediapipe.python.solutions.pose import Pose
except ImportError:
    import mediapipe as mp
    Pose = mp.solutions.pose.Pose

# Same config as src/worker.py
MODEL_COMPLEXITY = 1
# Typical frame size from transcoded videos (854x480)
FRAME_WIDTH = 854
FRAME_HEIGHT = 480
NUM_WARMUP = 5
NUM_ITERS = 100


def main():
    # Synthetic RGB frame (same format as cv2.cvtColor produces)
    rgb = np.random.randint(0, 256, (FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    print(f"Benchmark: MediaPipe Pose per-frame latency")
    print(f"  Frame size: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"  Model complexity: {MODEL_COMPLEXITY}")
    print(f"  Warmup runs: {NUM_WARMUP}, timed runs: {NUM_ITERS}")
    print()

    with Pose(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        enable_segmentation=False,
    ) as pose:
        # Warmup (first runs are slower due to model init/JIT)
        for i in range(NUM_WARMUP):
            pose.process(rgb)
        print("  Warmup done.")

        # Timed runs
        times_ms = []
        for _ in range(NUM_ITERS):
            t0 = time.perf_counter()
            pose.process(rgb)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000)

    avg_ms = sum(times_ms) / len(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)
    # Median for robustness
    sorted_ms = sorted(times_ms)
    med_ms = sorted_ms[len(sorted_ms) // 2]

    print()
    print("Results (single frame):")
    print(f"  Average: {avg_ms:.2f} ms")
    print(f"  Median:  {med_ms:.2f} ms")
    print(f"  Min:    {min_ms:.2f} ms")
    print(f"  Max:    {max_ms:.2f} ms")
    print()
    print(f"  Frames per second (avg): {1000 / avg_ms:.1f} fps")
    print(f"  Frames per second (median): {1000 / med_ms:.1f} fps")


if __name__ == "__main__":
    main()
