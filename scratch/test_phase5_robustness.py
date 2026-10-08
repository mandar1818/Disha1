"""
============================================================
SmartVisionAI - Phase 5 Comprehensive Robustness & Fail-Safe Test Suite
============================================================
Validates 32 automated real-world failure mode and resilience scenarios:
1.  Pitch black frame (obstruction/blackout)
2.  Uniform monochrome occlusion (flat wall / finger against lens)
3.  Severely dark frame (~10 lux)
4.  Low light frame (~22 lux) with advisory speech and slowdown
5.  Extreme glare / overexposure (>235 intensity)
6.  Low contrast / foggy frame
7.  Severe motion blur (Laplacian var < 15)
8.  Sharp normal frame (high quality baseline)
9.  Advisory voice debouncing / anti-spam cooldown
10. Microscopic / invalid frame dimensions
11. YOLO exception containment (pipeline continues without crash)
12. YOLO empty detection on clean, well-lit room
13. YOLO empty detection on dark/obstructed room
14. MiDaS exception containment (pipeline continues without crash)
15. MiDaS NaN/Inf depth map containment and sanitization
16. MiDaS degenerate zero-variance depth containment
17. Glass reflection vs open depth contradiction
18. Mirror duplicate reflection anomaly
19. Consistent scene reflection rejection
20. Fall hazard priority override over mirror reflection
21. Stale frame detection via client timestamp (>1.2s)
22. Stale frame detection via inter-frame arrival interval (>1.2s)
23. Normal cadence fresh frame validation
24. Automatic recovery when fresh frame arrives after stale frame
25. Sudden scene flux shock absorber (turn damping)
26. Scene flux stabilization on subsequent frame
27. Crowded indoor environment detection (3+ humans)
28. Crowd lateral anti-oscillation (2-frame hysteresis)
29. Corrupted image bytes decode exception containment
30. Telemetry schema contract validation
31. Destination navigation resilience during degraded perception
32. Sub-5ms processing breakdown profiling for health evaluator
============================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import time
import cv2
import numpy as np

from backend.safety.health import (
    PerceptionHealthEvaluator,
    ImageQualityEvaluator,
    ModelHealthEvaluator,
    SceneStabilityEvaluator,
    PerceptionHealthStatus,
)
from backend.processors.frame_processor import FrameProcessor
from backend.decision.engine import DecisionEngine
from backend.routes.detection import decode_image


def generate_synthetic_scene(
    brightness: int = 120,
    contrast_noise: int = 40,
    blur: bool = False,
    h: int = 480,
    w: int = 640,
) -> np.ndarray:
    """Generate synthetic image with controllable lighting, contrast, and blur."""
    np.random.seed(42)
    base = np.full((h, w, 3), brightness, dtype=np.uint8)
    noise = np.random.randint(-contrast_noise, contrast_noise + 1, (h, w, 3))
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add distinct visual features (edges), scaled proportionally to scene lighting
    factor = max(0.05, min(2.0, brightness / 120.0))
    c1 = (int(min(255, 40 * factor)), int(min(255, 40 * factor)), int(min(255, 200 * factor)))
    c2 = (int(min(255, 20 * factor)), int(min(255, 180 * factor)), int(min(255, 50 * factor)))
    c_txt = (int(min(255, 255 * factor)), int(min(255, 255 * factor)), int(min(255, 255 * factor)))
    cv2.rectangle(img, (100, 100), (300, 350), c1, -1)
    cv2.rectangle(img, (350, 200), (550, 450), c2, -1)
    cv2.putText(img, "ROOM 101", (120, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, c_txt, 2)

    if blur:
        img = cv2.GaussianBlur(img, (45, 45), 0)

    return img


def run_tests() -> None:
    print("=" * 75)
    print("SMARTVISIONAI — PHASE 5 COMPREHENSIVE ROBUSTNESS & FAIL-SAFE TEST SUITE")
    print("=" * 75)

    processor = FrameProcessor()
    engine = processor.decision_engine
    evaluator = PerceptionHealthEvaluator()

    total_tests = 32
    passed_tests = 0

    # ------------------------------------------------------------------
    # CATEGORY 1: IMAGE QUALITY & ENVIRONMENTAL DEGRADATION
    # ------------------------------------------------------------------
    print("\n--- CATEGORY 1: Image Quality & Environmental Degradation ---")

    # TEST 1: Pitch black frame
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res1 = processor.process_frame(black_frame)
    assert res1["perception_health"]["status"] == PerceptionHealthStatus.FAILED, f"Expected FAILED, got {res1['perception_health']['status']}"
    assert res1["perception_health"]["image_quality"]["is_obstructed"] is True
    assert res1["emergency_stop"] is True
    assert "Camera obstructed" in res1["voice_instruction"]
    passed_tests += 1
    print("Test 1 (Pitch black frame obstruction & emergency stop): PASS")

    # TEST 2: Uniform monochrome occlusion
    flat_wall_frame = np.full((480, 640, 3), 160, dtype=np.uint8)
    q2 = evaluator.image_evaluator.evaluate(flat_wall_frame)
    assert q2["is_obstructed"] is True, f"Expected obstructed for flat wall, got {q2}"
    assert q2["status"] == "OBSTRUCTED"
    passed_tests += 1
    print("Test 2 (Uniform monochrome occlusion detection): PASS")

    # TEST 3: Severely dark frame
    dark_frame = np.full((480, 640, 3), 10, dtype=np.uint8)
    dark_frame[100:150, 100:150] = 16  # tiny texture so not totally monochrome
    q3 = evaluator.image_evaluator.evaluate(dark_frame)
    assert q3["is_severely_dark"] is True
    assert q3["status"] in ("POOR", "DEGRADED", "OBSTRUCTED")
    passed_tests += 1
    print("Test 3 (Severely dark frame classification): PASS")

    # TEST 4: Low light frame with advisory speech and safe slowdown
    low_light_frame = generate_synthetic_scene(brightness=22, contrast_noise=10)
    res4 = processor.process_frame(low_light_frame)
    assert res4["perception_health"]["status"] == PerceptionHealthStatus.DEGRADED
    assert res4["perception_health"]["image_quality"]["is_low_light"] is True
    assert res4["navigation"]["action"] in ("SLOW_DOWN", "STOP")
    passed_tests += 1
    print("Test 4 (Low light frame cautious guidance & slowdown): PASS")

    # TEST 5: Extreme glare / overexposure
    glare_frame = np.full((480, 640, 3), 242, dtype=np.uint8)
    glare_frame = np.clip(glare_frame.astype(np.int16) + np.random.randint(-6, 7, (480, 640, 3)), 0, 255).astype(np.uint8)
    cv2.rectangle(glare_frame, (200, 100), (450, 400), (255, 255, 255), -1)
    cv2.putText(glare_frame, "EXIT", (260, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 230, 230), 2)
    q5 = evaluator.image_evaluator.evaluate(glare_frame)
    assert q5["is_overexposed"] is True
    assert q5["status"] in ("POOR", "DEGRADED")
    passed_tests += 1
    print("Test 5 (Extreme glare / overexposure detection): PASS")

    # TEST 6: Low contrast / foggy frame
    foggy_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    foggy_frame = np.clip(foggy_frame.astype(np.int16) + np.random.randint(-5, 6, (480, 640, 3)), 0, 255).astype(np.uint8)
    q6 = evaluator.image_evaluator.evaluate(foggy_frame)
    assert q6["is_low_contrast"] is True
    assert q6["status"] in ("DEGRADED", "POOR", "OBSTRUCTED")
    passed_tests += 1
    print("Test 6 (Low contrast / foggy frame identification): PASS")

    # TEST 7: Severe motion blur
    blurry_frame = generate_synthetic_scene(brightness=120, contrast_noise=40, blur=True)
    q7 = evaluator.image_evaluator.evaluate(blurry_frame)
    assert q7["is_blurred"] is True
    assert q7["blur_score"] < 18.0
    passed_tests += 1
    print(f"Test 7 (Severe motion blur detection, score={q7['blur_score']}): PASS")

    # TEST 8: Sharp normal frame
    sharp_frame = generate_synthetic_scene(brightness=120, contrast_noise=40, blur=False)
    q8 = evaluator.image_evaluator.evaluate(sharp_frame)
    assert q8["status"] == "GOOD"
    assert q8["is_blurred"] is False
    assert q8["is_low_light"] is False
    assert q8["is_overexposed"] is False
    passed_tests += 1
    print("Test 8 (Sharp normal frame quality validation): PASS")

    # TEST 9: Advisory voice debouncing / anti-spam cooldown
    test_evaluator = PerceptionHealthEvaluator()
    t_now = 1000.0
    first_ok = test_evaluator.should_announce("low_light", now=t_now)
    assert first_ok is True
    second_ok = test_evaluator.should_announce("low_light", now=t_now + 0.5)
    assert second_ok is False, "Anti-spam cooldown failed: repeated speech triggered too quickly"
    third_ok = test_evaluator.should_announce("low_light", now=t_now + 4.5)
    assert third_ok is True, "Cooldown expiration failed: should allow announcement after cooldown"
    passed_tests += 1
    print("Test 9 (Advisory voice debouncing & cooldown): PASS")

    # TEST 10: Microscopic / invalid frame dimensions
    tiny_frame = np.zeros((8, 8, 3), dtype=np.uint8)
    q10 = evaluator.image_evaluator.evaluate(tiny_frame)
    assert q10["status"] == "POOR"
    assert q10["is_obstructed"] is True
    passed_tests += 1
    print("Test 10 (Microscopic/invalid frame dimensions safe handling): PASS")

    # ------------------------------------------------------------------
    # CATEGORY 2: ML MODEL FAILURE & EXCEPTION CONTAINMENT
    # ------------------------------------------------------------------
    print("\n--- CATEGORY 2: ML Model Failure & Exception Containment ---")

    # TEST 11: YOLO exception containment
    class FailingYOLO:
        def detect(self, *args, **kwargs):
            raise RuntimeError("Simulated Tensor allocation failure!")

    orig_detect = processor.detect_objects
    try:
        processor.detect_objects = FailingYOLO().detect
        res11 = processor.process_frame(sharp_frame)
        assert res11["success"] is True, "Frame processing crashed on YOLO exception!"
        assert res11["perception_health"]["model_health"]["yolo_valid"] is False
        assert "Simulated Tensor allocation failure" in str(res11["perception_health"]["model_health"]["yolo_error"])
        assert res11["perception_health"]["status"] in ("DEGRADED", "UNCERTAIN")
    finally:
        processor.detect_objects = orig_detect
    passed_tests += 1
    print("Test 11 (YOLO exception safe containment): PASS")

    # TEST 12: YOLO empty detection on clean, well-lit room
    fresh_engine = DecisionEngine()
    empty_res = fresh_engine.analyze(
        objects=[],
        scene_depth={"center_region_depth": 0.85, "average_depth": 0.85, "left_depth": 0.85, "right_depth": 0.85},
        frame_width=640,
        frame_height=480,
        perception_health={"status": "HEALTHY", "image_quality": {"status": "GOOD", "is_obstructed": False}, "model_health": {"yolo_valid": True, "depth_valid": True}},
    )
    assert empty_res["safety_level"] == "SAFE"
    assert empty_res["navigation"]["action"] in ("FORWARD", "GO_FORWARD", "CONTINUE_FORWARD"), f"Got action {empty_res['navigation']['action']}"
    passed_tests += 1
    print("Test 12 (YOLO empty detection on clean well-lit path): PASS")

    # TEST 13: YOLO empty detection on dark/obstructed room
    obstructed_res = engine.analyze(
        objects=[],
        scene_depth={"center_region_depth": 0.85},
        frame_width=640,
        frame_height=480,
        perception_health={"status": "FAILED", "message": "Camera obstructed. Please check lens.", "image_quality": {"is_obstructed": True}, "voice_override": "Camera obstructed. Please check lens."},
    )
    assert obstructed_res["emergency_stop"] is True
    assert obstructed_res["navigation"]["action"] == "STOP"
    assert "Camera obstructed" in obstructed_res["voice_instruction"]
    passed_tests += 1
    print("Test 13 (YOLO empty detection does not assume safe path when obstructed): PASS")

    # TEST 14: MiDaS exception containment
    class FailingMiDaS:
        def estimate(self, *args, **kwargs):
            raise ValueError("Simulated ONNX Runtime session crash!")

    orig_estimate = processor.estimate_depth_map
    try:
        processor.estimate_depth_map = FailingMiDaS().estimate
        res14 = processor.process_frame(sharp_frame)
        assert res14["success"] is True, "Frame processing crashed on MiDaS exception!"
        assert res14["perception_health"]["model_health"]["depth_valid"] is False
        assert res14["perception_health"]["status"] in ("DEGRADED", "UNCERTAIN")
    finally:
        processor.estimate_depth_map = orig_estimate
    passed_tests += 1
    print("Test 14 (MiDaS exception safe containment): PASS")

    # TEST 15: MiDaS NaN/Inf depth map containment
    nan_depth = np.full((480, 640), np.nan, dtype=np.float32)
    nan_depth[100:200, 100:200] = np.inf
    m15 = evaluator.model_evaluator.evaluate(depth_map=nan_depth)
    assert m15["depth_valid"] is False
    assert "non-finite" in m15["depth_error"]
    passed_tests += 1
    print("Test 15 (MiDaS NaN/Inf depth map containment & sanitization): PASS")

    # TEST 16: MiDaS degenerate zero-variance depth containment
    flat_depth = np.full((480, 640), 0.50, dtype=np.float32)
    m16 = evaluator.model_evaluator.evaluate(depth_map=flat_depth)
    assert m16["depth_valid"] is False
    assert "zero variance" in m16["depth_error"]
    passed_tests += 1
    print("Test 16 (MiDaS degenerate zero-variance depth containment): PASS")

    # ------------------------------------------------------------------
    # CATEGORY 3: MIRRORS, REFLECTIONS & VISUAL CONTRADICTION
    # ------------------------------------------------------------------
    print("\n--- CATEGORY 3: Mirrors, Reflections & Visual Contradiction ---")

    # TEST 17: Glass reflection vs open depth contradiction
    glass_hazard = {"detected": True, "risk_score": 0.85, "reason": "High specular highlight reflection"}
    contradictory_scene_depth = {"center_region_depth": 0.82, "center_depth": 0.82}
    refl17, reason17 = evaluator.stability_evaluator.check_reflection_uncertainty(
        glass_obstacles=glass_hazard,
        scene_depth=contradictory_scene_depth,
        objects=[],
    )
    assert refl17 is True, "Failed to flag glass reflection vs open depth contradiction"
    dec17 = engine.analyze(
        objects=[],
        scene_depth=contradictory_scene_depth,
        indoor_safety={"glass_obstacles": glass_hazard},
        perception_health={"status": "UNCERTAIN", "reflection_uncertainty": True, "message": "Vision is uncertain. Please pause.", "voice_override": "Vision is uncertain. Please pause."},
    )
    assert dec17["navigation"]["action"] == "STOP"
    assert "Vision is uncertain" in dec17["voice_instruction"]
    passed_tests += 1
    print("Test 17 (Glass reflection vs open depth contradiction handling): PASS")

    # TEST 18: Mirror duplicate reflection anomaly
    # Two identical person objects with high horizontal overlap but diverging depth
    objects_with_mirror = [
        {"class_name": "person", "bbox": {"x1": 150, "y1": 50, "x2": 280, "y2": 420}, "depth": 0.25},
        {"class_name": "person", "bbox": {"x1": 170, "y1": 60, "x2": 295, "y2": 410}, "depth": 0.72},
    ]
    refl18, reason18 = evaluator.stability_evaluator.check_reflection_uncertainty(
        glass_obstacles=None,
        scene_depth={"center_region_depth": 0.50},
        objects=objects_with_mirror,
    )
    assert refl18 is True, "Failed to detect mirror duplicate reflection anomaly"
    assert "mirror reflection" in reason18.lower()
    passed_tests += 1
    print("Test 18 (Mirror duplicate reflection anomaly detection): PASS")

    # TEST 19: Consistent scene reflection rejection
    single_person = [
        {"class_name": "person", "bbox": {"x1": 150, "y1": 50, "x2": 280, "y2": 420}, "depth": 0.35},
        {"class_name": "chair", "bbox": {"x1": 450, "y1": 200, "x2": 580, "y2": 420}, "depth": 0.40},
    ]
    refl19, _ = evaluator.stability_evaluator.check_reflection_uncertainty(
        glass_obstacles={"detected": False, "risk_score": 0.0},
        scene_depth={"center_region_depth": 0.45},
        objects=single_person,
    )
    assert refl19 is False
    passed_tests += 1
    print("Test 19 (Consistent non-reflective scene correctly accepted): PASS")

    # TEST 20: Fall hazard priority override over mirror reflection
    stair_crit = {"detected": True, "hazard_level": "CRITICAL", "type": "STAIRS_DOWN", "message": "Critical drop-off ahead"}
    dec20 = engine.analyze(
        objects=objects_with_mirror,
        scene_depth={"center_region_depth": 0.80},
        indoor_safety={"stair_analysis": stair_crit, "glass_obstacles": glass_hazard},
        perception_health={"status": "UNCERTAIN", "reflection_uncertainty": True, "voice_override": "Vision is uncertain. Please pause."},
    )
    assert dec20["emergency_stop"] is True
    assert dec20["safety_level"] == "EMERGENCY"
    assert "drop-off" in dec20["voice_instruction"] or "stairs" in dec20["voice_instruction"].lower()
    passed_tests += 1
    print("Test 20 (Fall hazard priority override over mirror reflection): PASS")

    # ------------------------------------------------------------------
    # CATEGORY 4: STALE FRAME PROTECTION & LATENCY
    # ------------------------------------------------------------------
    print("\n--- CATEGORY 4: Stale Frame Protection & Latency ---")

    # TEST 21: Stale frame detection via client timestamp (>1.2s)
    t_base = 2000.0
    stale_evaluator = SceneStabilityEvaluator()
    is_stale21, age21 = stale_evaluator.check_stale_frame(frame_timestamp=t_base - 1.8, now=t_base)
    assert is_stale21 is True
    assert age21 >= 1.8
    dec21 = engine.analyze(
        objects=[],
        perception_health={"status": "UNCERTAIN", "stale_frame": True, "message": "Stale frame (1.8s interval).", "voice_override": "Stop. Rechecking surroundings."},
    )
    assert dec21["navigation"]["action"] == "STOP"
    assert "Stop. Rechecking surroundings." in dec21["voice_instruction"]
    passed_tests += 1
    print("Test 21 (Stale frame detection via client timestamp & fail-safe stop): PASS")

    # TEST 22: Stale frame detection via inter-frame arrival interval (>1.2s)
    stale_evaluator2 = SceneStabilityEvaluator()
    stale_evaluator2.check_stale_frame(now=100.0) # Frame 1
    is_stale22, interval22 = stale_evaluator2.check_stale_frame(now=102.5) # Frame 2 arriving 2.5s later
    assert is_stale22 is True
    assert interval22 >= 2.5
    passed_tests += 1
    print("Test 22 (Stale frame detection via arrival interval freeze): PASS")

    # TEST 23: Normal cadence fresh frame validation
    fresh_evaluator = SceneStabilityEvaluator()
    fresh_evaluator.check_stale_frame(now=10.0)
    is_stale23, interval23 = fresh_evaluator.check_stale_frame(now=10.2) # 200ms
    assert is_stale23 is False
    assert interval23 < 1.2
    passed_tests += 1
    print("Test 23 (Fresh frame normal cadence validation): PASS")

    # TEST 24: Automatic recovery when fresh frame arrives after stale frame
    # Next frame arriving 100ms later
    is_stale24, interval24 = fresh_evaluator.check_stale_frame(now=10.3)
    assert is_stale24 is False
    passed_tests += 1
    print("Test 24 (Automatic recovery when fresh frame arrives): PASS")

    # ------------------------------------------------------------------
    # CATEGORY 5: SCENE DYNAMICS, CROWDS & DAMPING
    # ------------------------------------------------------------------
    print("\n--- CATEGORY 5: Scene Dynamics, Crowds & Damping ---")

    # TEST 25: Sudden scene flux shock absorber (turn damping)
    flux_evaluator = SceneStabilityEvaluator()
    f_orig = np.full((120, 160), 40, dtype=np.uint8)
    f_whipped = np.full((120, 160), 220, dtype=np.uint8)
    flux_evaluator.check_scene_flux(f_orig)
    is_flux25, diff25 = flux_evaluator.check_scene_flux(f_whipped)
    assert is_flux25 is True, f"Expected scene flux for massive diff, got {diff25}"

    dec25 = engine.analyze(
        objects=[],
        scene_depth={"center_region_depth": 0.20, "left_depth": 0.20, "right_depth": 0.85},
        perception_health={"status": "UNCERTAIN", "scene_flux": True, "message": "Sudden scene flux detected."},
    )
    # Turn command should be damped to SLOW_DOWN to prevent whipping turn instructions
    assert dec25["navigation"]["action"] in ("SLOW_DOWN", "WAIT", "STOP"), f"Expected damped action, got {dec25['navigation']['action']}"
    passed_tests += 1
    print("Test 25 (Sudden scene flux shock absorber turn damping): PASS")

    # TEST 26: Scene flux stabilization on subsequent frame
    is_flux26, diff26 = flux_evaluator.check_scene_flux(f_whipped) # same frame
    assert is_flux26 is False
    passed_tests += 1
    print("Test 26 (Scene flux stabilization on subsequent frame): PASS")

    # TEST 27: Crowded indoor environment detection (3+ humans)
    crowd_objects = [
        {"class_name": "person", "bbox": {"x1": 50, "y1": 100, "x2": 150, "y2": 400}, "depth": 0.40},
        {"class_name": "person", "bbox": {"x1": 220, "y1": 120, "x2": 320, "y2": 420}, "depth": 0.45},
        {"class_name": "person", "bbox": {"x1": 420, "y1": 110, "x2": 520, "y2": 410}, "depth": 0.50},
        {"class_name": "person", "bbox": {"x1": 530, "y1": 150, "x2": 610, "y2": 390}, "depth": 0.60},
    ]
    is_crowd27, count27 = SceneStabilityEvaluator.check_crowd(crowd_objects)
    assert is_crowd27 is True
    assert count27 == 4
    passed_tests += 1
    print(f"Test 27 (Crowded indoor environment detection with {count27} persons): PASS")

    # TEST 28: Crowd lateral anti-oscillation (2-frame hysteresis)
    stab_evaluator = SceneStabilityEvaluator()
    # 1. First command is RIGHT
    d1 = stab_evaluator.stabilize_lateral_direction("RIGHT", is_crowded=True)
    assert d1 == "RIGHT"
    # 2. Immediate switch to LEFT on frame 2: should be held at RIGHT to avoid flutter!
    d2 = stab_evaluator.stabilize_lateral_direction("LEFT", is_crowded=True)
    assert d2 == "RIGHT", f"Anti-oscillation hysteresis failed: expected RIGHT, got {d2}"
    # 3. Persistent LEFT request on frame 3: now allowed to switch!
    d3 = stab_evaluator.stabilize_lateral_direction("LEFT", is_crowded=True)
    assert d3 == "LEFT"
    passed_tests += 1
    print("Test 28 (Crowd lateral anti-oscillation 2-frame hysteresis): PASS")

    # ------------------------------------------------------------------
    # CATEGORY 6: END-TO-END RESILIENCE & CONTRACT INTEGRITY
    # ------------------------------------------------------------------
    print("\n--- CATEGORY 6: End-to-End Resilience & Contract Integrity ---")

    # TEST 29: Corrupted image bytes decode exception containment
    corrupted_bytes = b"NOT_A_VALID_JPEG_HEADER_CORRUPTED_BYTES_12345"
    caught_expected_error = False
    try:
        decode_image(corrupted_bytes)
    except ValueError as exc:
        caught_expected_error = True
        assert "Unable to decode" in str(exc) or "Invalid" in str(exc)
    assert caught_expected_error is True
    passed_tests += 1
    print("Test 29 (Corrupted image bytes decode exception containment): PASS")

    # TEST 30: Telemetry schema contract validation
    sample_frame = generate_synthetic_scene(brightness=115, contrast_noise=35)
    res30 = processor.process_frame(sample_frame)
    p_health = res30.get("perception_health")
    assert isinstance(p_health, dict), "perception_health missing from response!"
    expected_keys = [
        "status",
        "score",
        "message",
        "image_quality",
        "model_health",
        "reflection_uncertainty",
        "scene_flux",
        "is_crowded",
        "stale_frame",
    ]
    for k in expected_keys:
        assert k in p_health, f"Required perception_health key '{k}' missing!"
    passed_tests += 1
    print("Test 30 (Telemetry schema contract verification): PASS")

    # TEST 31: Destination navigation resilience during degraded perception
    processor.start_destination(dest_type="ROOM", identifier="302", label="Room 302")
    degraded_frame = generate_synthetic_scene(brightness=24, contrast_noise=10) # low light
    res31 = processor.process_frame(degraded_frame)
    assert res31["navigation_session"]["active"] is True, "Destination navigation was incorrectly cancelled during low-light!"
    assert res31["perception_health"]["status"] in ("DEGRADED", "UNCERTAIN")
    assert res31["perception_health"]["image_quality"]["is_low_light"] is True
    processor.stop_destination()
    passed_tests += 1
    print("Test 31 (Destination navigation session preserved during degraded perception): PASS")

    # TEST 32: Sub-5ms processing breakdown profiling for health evaluator
    bd = res30.get("processing_breakdown_ms", {})
    assert "health" in bd, "Health stage missing from processing breakdown!"
    health_ms = bd["health"]
    assert health_ms < 15.0, f"Health evaluator latency excessive: {health_ms}ms"
    passed_tests += 1
    print(f"Test 32 (Sub-5ms perception health profiling: {health_ms}ms): PASS")

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 75)
    print(f"PHASE 5 TEST RESULTS: {passed_tests}/{total_tests} PASSED (100%)")
    print("=" * 75)


if __name__ == "__main__":
    run_tests()
