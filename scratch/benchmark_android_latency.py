"""
SmartVisionAI - Phase 11 Android End-to-End Latency & Telemetry Benchmark
========================================================================
Measures performance across 30 continuous frames simulating live camera
capture at a 500ms cadence:
- YOLOv8n TFLite inference latency
- MiDaS Small ONNX depth latency
- DecisionEngine latency
- Indoor Safety latency
- Total pipeline latency: median, mean, P90, P95, min, max
- Headroom relative to 500ms frame arrival budget
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.processors.frame_processor import FrameProcessor


def generate_benchmark_frame(idx: int) -> np.ndarray:
    np.random.seed(idx)
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    noise = np.random.randint(-15, 16, (480, 640, 3)).astype(np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    frame[350:, :] = np.clip(frame[350:, :].astype(np.int16) - 30, 0, 255).astype(np.uint8)
    return frame


def run_benchmark():
    print("=" * 70)
    print("SMARTVISIONAI -- PHASE 11 ANDROID LATENCY & HEADROOM BENCHMARK")
    print("=" * 70)
    print("Initializing FrameProcessor (TFLite + ONNX)...")

    processor = FrameProcessor()

    num_frames = 30
    total_latencies = []
    yolo_latencies = []
    midas_latencies = []
    decision_latencies = []
    safety_latencies = []

    print(f"\nExecuting {num_frames} continuous simulated Android frames at ~500ms cadence...")

    # Warmup
    for w in range(3):
        warmup_frame = generate_benchmark_frame(w)
        processor.process_frame(warmup_frame)

    for i in range(1, num_frames + 1):
        frame = generate_benchmark_frame(i + 10)
        t_start = time.perf_counter()
        result = processor.process_frame(frame)
        t_total = (time.perf_counter() - t_start) * 1000.0

        breakdown = result.get("processing_breakdown_ms", {})

        total_latencies.append(t_total)
        yolo_latencies.append(breakdown.get("yolo", 0.0))
        midas_latencies.append(breakdown.get("midas", 0.0))
        decision_latencies.append(breakdown.get("decision", 0.0))
        safety_latencies.append(breakdown.get("indoor", 0.0))

        if i % 10 == 0 or i == num_frames:
            print(f"  Frame {i:02d}/{num_frames}: Total = {t_total:.1f}ms | YOLO = {breakdown.get('yolo', 0):.1f}ms | MiDaS = {breakdown.get('midas', 0):.1f}ms | Safety = {breakdown.get('indoor', 0):.1f}ms | Decision = {breakdown.get('decision', 0):.1f}ms")

    latencies_arr = np.array(total_latencies)
    median_lat = float(np.median(latencies_arr))
    p90_lat = float(np.percentile(latencies_arr, 90))
    p95_lat = float(np.percentile(latencies_arr, 95))
    min_lat = float(np.min(latencies_arr))
    max_lat = float(np.max(latencies_arr))
    mean_lat = float(np.mean(latencies_arr))

    median_yolo = float(np.median(yolo_latencies))
    median_midas = float(np.median(midas_latencies))
    median_dec = float(np.median(decision_latencies))
    median_safe = float(np.median(safety_latencies))

    budget_ms = 500.0
    headroom_ms = budget_ms - median_lat
    headroom_pct = (headroom_ms / budget_ms) * 100.0

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Total Frames Evaluated    : {num_frames}")
    print(f"  Target Android Cadence    : {budget_ms:.1f} ms")
    print(f"  Median Pipeline Latency   : {median_lat:.2f} ms")
    print(f"  Mean Pipeline Latency     : {mean_lat:.2f} ms")
    print(f"  90th Percentile Latency   : {p90_lat:.2f} ms")
    print(f"  95th Percentile Latency   : {p95_lat:.2f} ms")
    print(f"  Min / Max Latency         : {min_lat:.2f} ms / {max_lat:.2f} ms")
    print("----------------------------------------------------------------------")
    print("  Component Latency Breakdown (Median):")
    print(f"    - YOLOv8n TFLite        : {median_yolo:.2f} ms")
    print(f"    - MiDaS Small ONNX      : {median_midas:.2f} ms")
    print(f"    - Indoor Safety Manager : {median_safe:.2f} ms")
    print(f"    - Decision Engine       : {median_dec:.2f} ms")
    print("----------------------------------------------------------------------")
    print(f"  Median Headroom           : {headroom_ms:.2f} ms ({headroom_pct:.1f}% under budget)")
    print(f"  Cadence Target Compliance : {'PASS' if median_lat <= budget_ms else 'FAIL'}")
    print("=" * 70)

    assert median_lat <= budget_ms, f"Median latency {median_lat}ms exceeds 500ms budget!"
    print("\nALL BENCHMARK CRITERIA SATISFIED!")


if __name__ == "__main__":
    run_benchmark()
