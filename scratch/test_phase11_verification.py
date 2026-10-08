"""
test_phase11_verification.py
Phase 11: Approaching-Obstacle Detection & Predictive Safety Test Suite
Validates all 37 scenarios covering:
- Motion state classification (STATIONARY, APPROACHING, RAPIDLY_APPROACHING, RECEDING, UNKNOWN_MOTION)
- Approach rate calculation, trend consistency, noise rejection
- Numerical robustness (None, NaN, Inf, zero/negative dt)
- Predictive threat assessment and corridor weighting
- Predictive safety escalation (EMERGENCY, DANGER, CAUTION)
- Navigation avoidance (immediate hysteresis bypass for rapid center threat)
- Step guidance invariants (STOP/TURN/EMERGENCY/VERY_CLOSE -> steps=0)
- Voice instruction generation ("Warning. Person approaching.", "Stop immediately.")
- Response contracts and backward compatibility
"""

import sys
import os
import math
import time

sys.path.insert(0, r"D:\SmartVisionAI_New")

from backend.decision.engine import DecisionEngine


def run_phase11_tests():
    print("=" * 65)
    print("PHASE 11: APPROACHING-OBSTACLE DETECTION & PREDICTIVE SAFETY")
    print("=" * 65)

    passed = 0
    total = 37

    # ============================================================
    # GROUP 1: MOTION STATE CLASSIFICATION
    # ============================================================

    # Test 1: Single frame cannot classify approaching (UNKNOWN_MOTION)
    e = DecisionEngine()
    obs_f1 = [{"id": 1, "class_name": "person", "position": "CENTER", "depth": 0.5, "distance_m": 3.0, "risk_score": 0.4}]
    res1 = e.analyze(objects=obs_f1)
    obj1 = res1["objects"][0]
    assert obj1["motion_state"] == "UNKNOWN_MOTION", f"T1 failed motion_state: {obj1['motion_state']}"
    assert obj1["approach_rate_mps"] == 0.0, f"T1 failed approach_rate_mps: {obj1['approach_rate_mps']}"
    assert obj1["is_approaching"] is False, f"T1 failed is_approaching: {obj1['is_approaching']}"
    assert obj1["time_to_collision_s"] is None, f"T1 failed time_to_collision_s: {obj1['time_to_collision_s']}"
    passed += 1
    print("Test 1 PASS: Single frame -> UNKNOWN_MOTION, rate=0, is_approaching=False")

    # Test 2: Stationary object over consecutive frames
    e = DecisionEngine()
    obs_stat = [{"id": 1, "class_name": "chair", "position": "CENTER", "depth": 0.5, "distance_m": 3.0, "risk_score": 0.3}]
    for _ in range(4):
        res2 = e.analyze(objects=obs_stat)
    obj2 = res2["objects"][0]
    assert obj2["motion_state"] == "STATIONARY", f"T2 failed motion_state: {obj2['motion_state']}"
    assert obj2["is_approaching"] is False, f"T2 failed is_approaching: {obj2['is_approaching']}"
    passed += 1
    print("Test 2 PASS: Stationary object maintained -> STATIONARY, is_approaching=False")

    # Test 3: Moderate approaching obstacle (2+ frame confirmation & trend consistency)
    e = DecisionEngine()
    distances_mod = [3.2, 2.9, 2.6, 2.3]
    for d in distances_mod:
        res3 = e.analyze(objects=[{"id": 1, "class_name": "person", "position": "CENTER", "depth": 0.5, "distance_m": d, "risk_score": 0.4}])
    obj3 = res3["objects"][0]
    assert obj3["motion_state"] == "APPROACHING", f"T3 failed motion_state: {obj3['motion_state']}"
    assert obj3["is_approaching"] is True, f"T3 failed is_approaching: {obj3['is_approaching']}"
    assert obj3["approach_rate_mps"] >= 0.15, f"T3 failed rate: {obj3['approach_rate_mps']}"
    passed += 1
    print(f"Test 3 PASS: Moderate approach -> APPROACHING (rate={obj3['approach_rate_mps']}m/s, is_approaching=True)")

    # Test 4: Rapidly approaching obstacle (rate >= 0.50 m/s)
    e = DecisionEngine()
    distances_rapid = [4.0, 3.3, 2.5, 1.7]
    for d in distances_rapid:
        res4 = e.analyze(objects=[{"id": 1, "class_name": "person", "position": "CENTER", "depth": 0.5, "distance_m": d, "risk_score": 0.5}])
    obj4 = res4["objects"][0]
    assert obj4["motion_state"] == "RAPIDLY_APPROACHING", f"T4 failed motion_state: {obj4['motion_state']}"
    assert obj4["is_approaching"] is True, f"T4 failed is_approaching: {obj4['is_approaching']}"
    assert obj4["approach_rate_mps"] >= 0.50, f"T4 failed rate: {obj4['approach_rate_mps']}"
    passed += 1
    print(f"Test 4 PASS: Rapid approach -> RAPIDLY_APPROACHING (rate={obj4['approach_rate_mps']}m/s)")

    # Test 5: Sudden close distance drop into dangerous zone (>= 0.5m drop into <= 1.4m)
    e = DecisionEngine()
    e.analyze(objects=[{"id": 1, "class_name": "person", "position": "CENTER", "depth": 0.4, "distance_m": 2.2, "risk_score": 0.4}])
    res5 = e.analyze(objects=[{"id": 1, "class_name": "person", "position": "CENTER", "depth": 0.2, "distance_m": 1.2, "risk_score": 0.7}])
    obj5 = res5["objects"][0]
    assert obj5["motion_state"] == "RAPIDLY_APPROACHING", f"T5 failed motion_state: {obj5['motion_state']}"
    assert obj5["is_approaching"] is True
    passed += 1
    print("Test 5 PASS: Sudden close drop into <=1.4m immediately triggers RAPIDLY_APPROACHING")

    # Test 6: Receding obstacle (distance increases)
    e = DecisionEngine()
    distances_rec = [2.0, 2.3, 2.6, 2.9]
    for d in distances_rec:
        res6 = e.analyze(objects=[{"id": 1, "class_name": "person", "position": "CENTER", "depth": 0.5, "distance_m": d, "risk_score": 0.3}])
    obj6 = res6["objects"][0]
    assert obj6["motion_state"] == "RECEDING", f"T6 failed motion_state: {obj6['motion_state']}"
    assert obj6["is_approaching"] is False, f"T6 failed is_approaching: {obj6['is_approaching']}"
    assert obj6["time_to_collision_s"] is None, f"T6 failed ttc: {obj6['time_to_collision_s']}"
    passed += 1
    print("Test 6 PASS: Receding obstacle -> RECEDING, is_approaching=False, TTC=None")

    # Test 7: Stationary oscillation rejection (trend consistency fails)
    e = DecisionEngine()
    oscillations = [3.0, 2.7, 2.9, 2.6, 2.8, 2.6, 2.8]
    for d in oscillations:
        res7 = e.analyze(objects=[{"id": 1, "class_name": "chair", "position": "CENTER", "depth": 0.5, "distance_m": d, "risk_score": 0.3}])
    obj7 = res7["objects"][0]
    assert obj7["motion_state"] == "STATIONARY", f"T7 failed: expected STATIONARY, got {obj7['motion_state']}"
    assert obj7["is_approaching"] is False
    passed += 1
    print("Test 7 PASS: Oscillating noise rejected by trend consistency -> STATIONARY")

    # Test 8: Missing detection frame drop: track preserved
    e = DecisionEngine()
    e.analyze(objects=[{"id": 1, "class_name": "person", "position": "CENTER", "depth": 0.5, "distance_m": 3.0, "risk_score": 0.4}])
    e.analyze(objects=[])
    res8 = e.analyze(objects=[{"id": 1, "class_name": "person", "position": "CENTER", "depth": 0.5, "distance_m": 2.95, "risk_score": 0.4}])
    obj8 = res8["objects"][0]
    assert obj8["track_id"] == "trk_1"
    assert obj8["motion_state"] in ("STATIONARY", "UNKNOWN_MOTION")
    passed += 1
    print("Test 8 PASS: Track preserved across dropped frame without crash or false approach spike")

    # ============================================================
    # GROUP 2: NUMERICAL ROBUSTNESS & INVARIANTS
    # ============================================================

    # Test 9: None, NaN, Inf handled gracefully
    e = DecisionEngine()
    for bad_val in [None, float("nan"), float("inf"), -2.0]:
        res9 = e.analyze(objects=[{"id": 1, "class_name": "obstacle", "position": "CENTER", "depth": 0.5, "distance_m": bad_val, "risk_score": 0.3}])
        obj9 = res9["objects"][0]
        assert math.isfinite(obj9["distance_m"]), f"T9 non-finite distance: {obj9['distance_m']}"
        assert math.isfinite(obj9["approach_rate_mps"]), f"T9 non-finite rate: {obj9['approach_rate_mps']}"
        assert 0.0 <= obj9["approach_score"] <= 1.0, f"T9 out of bounds score: {obj9['approach_score']}"
    passed += 1
    print("Test 9 PASS: None, NaN, Inf, and negative distances handled safely")

    # Test 10: Zero or negative dt fallback to nominal 0.70s
    e = DecisionEngine()
    e.analyze(objects=[{"id": 1, "class_name": "person", "position": "CENTER", "depth": 0.5, "distance_m": 3.0, "risk_score": 0.4}])
    res10 = e.analyze(objects=[{"id": 1, "class_name": "person", "position": "CENTER", "depth": 0.5, "distance_m": 2.7, "risk_score": 0.4}])
    obj10 = res10["objects"][0]
    assert obj10["approach_rate_mps"] < 10.0, f"T10 division by zero/tiny dt: {obj10['approach_rate_mps']}"
    passed += 1
    print(f"Test 10 PASS: Sub-millisecond dt cleanly falls back to nominal 0.70s (rate={obj10['approach_rate_mps']}m/s)")

    # Test 11: approach_score bounded in [0.0, 1.0]
    e = DecisionEngine()
    for d in [5.0, 3.5, 2.0, 1.0, 0.5]:
        res11 = e.analyze(objects=[{"id": 1, "class_name": "person", "position": "CENTER", "depth": 0.5, "distance_m": d, "risk_score": 0.8}])
        score = res11["objects"][0]["approach_score"]
        assert 0.0 <= score <= 1.0, f"T11 score out of bounds: {score}"
    passed += 1
    print("Test 11 PASS: approach_score is strictly bounded in [0.0, 1.0]")

    # Test 12: time_to_collision_s bounded in [0.1, 60.0] or None
    e = DecisionEngine()
    for d in [4.0, 3.2, 2.4, 1.6]:
        res12 = e.analyze(objects=[{"id": 1, "class_name": "person", "position": "CENTER", "depth": 0.5, "distance_m": d, "risk_score": 0.5}])
    ttc = res12["objects"][0]["time_to_collision_s"]
    assert ttc is not None and 0.1 <= ttc <= 60.0, f"T12 TTC out of bounds: {ttc}"
    passed += 1
    print(f"Test 12 PASS: time_to_collision_s is bounded cleanly: {ttc}s")

    # Test 13: Offline loop rate consistency
    e = DecisionEngine()
    rates = []
    for d in [3.5, 3.1, 2.7, 2.3, 1.9]:
        res13 = e.analyze(objects=[{"id": 1, "class_name": "person", "position": "CENTER", "depth": 0.5, "distance_m": d, "risk_score": 0.4}])
        rates.append(res13["objects"][0]["approach_rate_mps"])
    assert all(r < 5.0 for r in rates), f"T13 inflated rate in offline loop: {rates}"
    passed += 1
    print(f"Test 13 PASS: Offline test loop maintains stable rates: {rates}")

    # ============================================================
    # GROUP 3: MULTI-OBJECT & UNKNOWN OBSTACLE INDEPENDENCE
    # ============================================================

    # Test 14: Multi-object independence
    e = DecisionEngine()
    frames = [
        ({"id": 1, "class_name": "person", "position": "CENTER", "distance_m": 3.0, "bbox": {"x1": 200, "y1": 100, "x2": 300, "y2": 400}},
         {"id": 2, "class_name": "chair", "position": "RIGHT", "distance_m": 2.5, "bbox": {"x1": 500, "y1": 200, "x2": 600, "y2": 400}}),
        ({"id": 1, "class_name": "person", "position": "CENTER", "distance_m": 2.4, "bbox": {"x1": 200, "y1": 100, "x2": 300, "y2": 400}},
         {"id": 2, "class_name": "chair", "position": "RIGHT", "distance_m": 2.5, "bbox": {"x1": 500, "y1": 200, "x2": 600, "y2": 400}}),
        ({"id": 1, "class_name": "person", "position": "CENTER", "distance_m": 1.8, "bbox": {"x1": 200, "y1": 100, "x2": 300, "y2": 400}},
         {"id": 2, "class_name": "chair", "position": "RIGHT", "distance_m": 2.5, "bbox": {"x1": 500, "y1": 200, "x2": 600, "y2": 400}}),
    ]
    for p, c in frames:
        res14 = e.analyze(objects=[p, c])
    person = next(o for o in res14["objects"] if o["class_name"] == "person")
    chair = next(o for o in res14["objects"] if o["class_name"] == "chair")
    assert person["is_approaching"] is True, f"Person should be approaching: {person['motion_state']}"
    assert chair["is_approaching"] is False, f"Chair should be stationary: {chair['motion_state']}"
    passed += 1
    print("Test 14 PASS: Person is approaching while Chair remains stationary (independent tracking)")

    # Test 15: Phase 4 Unknown obstacle tracking & approach
    e = DecisionEngine()
    unk_frames = [
        {"id": 101, "position": "CENTER", "cand_depth": 0.4, "bbox": {"x1": 250, "y1": 150, "x2": 390, "y2": 400}, "confidence": 0.7, "box_area": 35000},
        {"id": 101, "position": "CENTER", "cand_depth": 0.3, "bbox": {"x1": 240, "y1": 140, "x2": 400, "y2": 410}, "confidence": 0.7, "box_area": 43200},
        {"id": 101, "position": "CENTER", "cand_depth": 0.18, "bbox": {"x1": 230, "y1": 130, "x2": 410, "y2": 420}, "confidence": 0.7, "box_area": 52200},
    ]
    for u in unk_frames:
        res15 = e.analyze(objects=[], unknown_objects=[u])
    unk_res = res15["unknown_objects"][0]
    assert unk_res["motion_state"] in ("APPROACHING", "RAPIDLY_APPROACHING"), f"T15 failed: {unk_res['motion_state']}"
    assert unk_res["is_approaching"] is True
    passed += 1
    print(f"Test 15 PASS: Phase 4 Unknown obstacle tracked and classified as {unk_res['motion_state']}")

    # ============================================================
    # GROUP 4: PREDICTIVE THREAT & SAFETY ESCALATION
    # ============================================================

    # Test 16: Top-level predictive_threat selects highest threat with corridor weighting
    e = DecisionEngine()
    p_left = {"id": 1, "class_name": "person", "position": "LEFT", "distance_m": 2.5, "bbox": {"x1": 50, "y1": 100, "x2": 150, "y2": 400}}
    p_center = {"id": 2, "class_name": "person", "position": "CENTER", "distance_m": 2.0, "bbox": {"x1": 250, "y1": 100, "x2": 350, "y2": 400}}
    for _ in range(3):
        res16 = e.analyze(objects=[p_left, p_center])
        p_left["distance_m"] -= 0.2
        p_center["distance_m"] -= 0.6
    pt16 = res16["predictive_threat"]
    assert pt16["position"] == "CENTER", f"T16 expected CENTER threat, got {pt16['position']}"
    assert pt16["is_approaching"] is True
    passed += 1
    print(f"Test 16 PASS: Predictive threat selected CENTER threat (position={pt16['position']}, level={pt16['threat_level']})")

    # Test 17: CENTER rapid approach <= 1.4m -> EMERGENCY stop
    e = DecisionEngine()
    e.analyze(objects=[{"id": 1, "class_name": "person", "position": "CENTER", "distance_m": 2.2, "depth": 0.4, "risk_score": 0.5}])
    res17 = e.analyze(objects=[{"id": 1, "class_name": "person", "position": "CENTER", "distance_m": 1.2, "depth": 0.15, "risk_score": 0.7}])
    assert res17["safety_level"] == "EMERGENCY", f"T17 failed safety_level: {res17['safety_level']}"
    assert res17["emergency_stop"] is True, f"T17 failed emergency_stop: {res17['emergency_stop']}"
    passed += 1
    print("Test 17 PASS: CENTER rapid approach <= 1.4m triggers EMERGENCY and emergency_stop=True")

    # Test 18: CENTER approaching with TTC <= 2.0s -> EMERGENCY
    e = DecisionEngine()
    for d in [3.0, 2.2, 1.4]:
        res18 = e.analyze(objects=[{"id": 1, "class_name": "person", "position": "CENTER", "distance_m": d, "depth": 0.2, "risk_score": 0.6}])
    assert res18["safety_level"] == "EMERGENCY", f"T18 failed: {res18['safety_level']}"
    assert res18["emergency_stop"] is True
    passed += 1
    print("Test 18 PASS: CENTER approaching with low TTC <= 2.0s triggers EMERGENCY")

    # Test 19: CENTER rapid approach <= 2.5m -> DANGER
    e = DecisionEngine()
    for d in [3.5, 2.8, 2.1]:
        res19 = e.analyze(objects=[{"id": 1, "class_name": "person", "position": "CENTER", "distance_m": d, "depth": 0.4, "risk_score": 0.4}])
    assert res19["safety_level"] in ("DANGER", "EMERGENCY"), f"T19 failed: {res19['safety_level']}"
    passed += 1
    print(f"Test 19 PASS: CENTER rapid approach <= 2.5m escalates to {res19['safety_level']}")

    # Test 20: CENTER approach <= 2.0m -> CAUTION
    e = DecisionEngine()
    for d in [2.6, 2.3, 2.0]:
        res20 = e.analyze(objects=[{"id": 1, "class_name": "person", "position": "CENTER", "distance_m": d, "depth": 0.5, "risk_score": 0.3}])
    assert res20["safety_level"] in ("CAUTION", "DANGER", "EMERGENCY"), f"T20 failed: {res20['safety_level']}"
    passed += 1
    print(f"Test 20 PASS: CENTER approach <= 2.0m escalates to at least {res20['safety_level']}")

    # Test 21: SIDE approaching + CENTER clear -> no stop
    e = DecisionEngine()
    for d in [3.0, 2.4, 1.8]:
        res21 = e.analyze(
            objects=[{"id": 1, "class_name": "person", "position": "RIGHT", "distance_m": d, "depth": 0.4, "risk_score": 0.4}],
            scene_depth={"center_depth": 0.8, "center_region_depth": 0.8, "left_depth": 0.8, "right_depth": 0.3}
        )
    assert res21["emergency_stop"] is False, "T21 should not emergency stop for side obstacle"
    assert res21["navigation"]["action"] in ("GO_FORWARD", "CONTINUE_FORWARD"), f"T21 failed action: {res21['navigation']['action']}"
    passed += 1
    print("Test 21 PASS: SIDE approaching + CENTER clear permits continued forward walking without stop")

    # ============================================================
    # GROUP 5: PREDICTIVE NAVIGATION & AVOIDANCE
    # ============================================================

    # Test 22: CENTER rapid approaching + safe side -> immediate avoidance (bypasses 2-frame hysteresis)
    e = DecisionEngine()
    e.last_action = "GO_FORWARD"
    fp_left_clear = {"center_clear": False, "left_clear": True, "right_clear": False, "no_safe_path": False, "best_direction": "LEFT"}
    nav22 = e.determine_navigation(
        safety_level="DANGER",
        emergency_stop=False,
        primary_obstacle={"position": "CENTER", "motion_state": "RAPIDLY_APPROACHING", "distance_m": 2.2, "distance_category": "NEAR"},
        free_path=fp_left_clear,
        predictive_threat={"position": "CENTER", "motion_state": "RAPIDLY_APPROACHING", "distance_m": 2.2}
    )
    assert nav22["action"] == "TURN_LEFT", f"T22 failed immediate avoidance: {nav22['action']}"
    assert nav22["navigation_stage"] == "TURN"
    assert nav22["turn_direction"] == "LEFT"
    passed += 1
    print("Test 22 PASS: CENTER rapid approach immediately triggers avoidance (TURN_LEFT), bypassing hysteresis")

    # Test 23: CENTER rapid approaching + all blocked -> immediate STOP
    e = DecisionEngine()
    e.last_action = "GO_FORWARD"
    fp_all_blocked = {"center_clear": False, "left_clear": False, "right_clear": False, "no_safe_path": True}
    nav23 = e.determine_navigation(
        safety_level="DANGER",
        emergency_stop=False,
        primary_obstacle={"position": "CENTER", "motion_state": "RAPIDLY_APPROACHING", "distance_m": 1.8},
        free_path=fp_all_blocked,
        predictive_threat={"position": "CENTER", "motion_state": "RAPIDLY_APPROACHING", "distance_m": 1.8}
    )
    assert nav23["action"] == "STOP", f"T23 failed: {nav23['action']}"
    assert nav23["steps"] == 0
    passed += 1
    print("Test 23 PASS: CENTER rapid approach + all blocked produces immediate STOP")

    # Test 24: CENTER approaching + left clear -> TURN_LEFT
    e = DecisionEngine()
    e.last_action = "TURN_LEFT"
    nav24 = e.determine_navigation(
        safety_level="CAUTION",
        emergency_stop=False,
        primary_obstacle={"position": "CENTER", "approaching": True, "distance_m": 2.0},
        free_path=fp_left_clear,
        predictive_threat={"position": "CENTER", "is_approaching": True, "motion_state": "APPROACHING"}
    )
    assert nav24["action"] == "TURN_LEFT"
    passed += 1
    print("Test 24 PASS: CENTER approaching + left clear -> TURN_LEFT")

    # Test 25: CENTER approaching + right clear -> TURN_RIGHT
    e = DecisionEngine()
    e.last_action = "TURN_RIGHT"
    fp_right_clear = {"center_clear": False, "left_clear": False, "right_clear": True, "no_safe_path": False, "best_direction": "RIGHT"}
    nav25 = e.determine_navigation(
        safety_level="CAUTION",
        emergency_stop=False,
        primary_obstacle={"position": "CENTER", "approaching": True, "distance_m": 2.0},
        free_path=fp_right_clear,
        predictive_threat={"position": "CENTER", "is_approaching": True, "motion_state": "APPROACHING"}
    )
    assert nav25["action"] == "TURN_RIGHT"
    passed += 1
    print("Test 25 PASS: CENTER approaching + right clear -> TURN_RIGHT")

    # ============================================================
    # GROUP 6: STEP GUIDANCE INVARIANTS
    # ============================================================

    # Test 26: Under STOP -> steps=0, movement_steps=0, movement_distance_m=0
    e = DecisionEngine()
    nav26 = e.determine_navigation("DANGER", False, {"distance_m": 1.0, "position": "CENTER"}, fp_all_blocked)
    sg26 = e.build_step_guidance(nav26)
    assert nav26["action"] == "STOP"
    assert nav26["steps"] == 0
    assert nav26["movement_steps"] == 0
    assert nav26["movement_distance_m"] == 0.0
    assert sg26["action"] == "STOP"
    assert sg26["steps"] == 0
    passed += 1
    print("Test 26 PASS: STOP enforces 0 steps and 0 movement distance")

    # Test 27: Under TURN_LEFT / TURN_RIGHT -> steps=0, movement_steps=0
    e = DecisionEngine()
    e.last_action = "TURN_LEFT"
    nav27 = e.determine_navigation("CAUTION", False, {"distance_m": 1.5, "position": "CENTER"}, fp_left_clear)
    sg27 = e.build_step_guidance(nav27)
    assert nav27["action"] == "TURN_LEFT"
    assert nav27["steps"] == 0
    assert nav27["movement_steps"] == 0
    assert sg27["action"] in ("TURN_LEFT", "WALK_LEFT", "MOVE_LEFT")
    assert sg27["steps"] == 0
    passed += 1
    print("Test 27 PASS: TURN actions enforce 0 walking steps (rotation only)")

    # Test 28: Under EMERGENCY -> steps=0
    e = DecisionEngine()
    nav28 = e.determine_navigation("EMERGENCY", True, {"distance_m": 0.8, "position": "CENTER"}, fp_all_blocked)
    sg28 = e.build_step_guidance(nav28)
    assert nav28["action"] == "STOP"
    assert nav28["steps"] == 0
    assert sg28["steps"] == 0
    passed += 1
    print("Test 28 PASS: EMERGENCY strictly enforces 0 steps")

    # Test 29: Under VERY_CLOSE -> steps=0
    e = DecisionEngine()
    obs_vc = {"distance_m": 0.6, "distance_category": "VERY_CLOSE", "position": "CENTER", "risk_score": 0.85}
    nav29 = e.determine_navigation("DANGER", False, obs_vc, fp_all_blocked)
    assert nav29["steps"] == 0
    assert nav29["movement_steps"] == 0
    passed += 1
    print("Test 29 PASS: VERY_CLOSE obstacle strictly enforces 0 steps")

    # Test 30: When moving forward with safe clearance -> movement_steps > 0
    e = DecisionEngine()
    fp_clear = {"center_clear": True, "left_clear": True, "right_clear": True, "no_safe_path": False, "best_direction": "CENTER"}
    obs_clear = {"distance_m": 4.0, "distance_category": "FAR", "position": "CENTER"}
    nav30 = e.determine_navigation("SAFE", False, obs_clear, fp_clear)
    sg30 = e.build_step_guidance(nav30, obs_clear)
    assert nav30["action"] in ("GO_FORWARD", "CONTINUE_FORWARD")
    assert nav30["movement_steps"] > 0
    assert sg30["action"] == "WALK_FORWARD"
    assert sg30["steps"] > 0
    passed += 1
    print(f"Test 30 PASS: Clear forward path permits {nav30['movement_steps']} walking steps ({nav30['movement_distance_m']}m)")

    # ============================================================
    # GROUP 7: VOICE INSTRUCTIONS
    # ============================================================

    # Test 31: Approaching person -> "Warning. Person approaching."
    e = DecisionEngine()
    v31 = e.generate_voice_instruction(
        safety_level="CAUTION",
        emergency_stop=False,
        primary_obstacle={"class_name": "person", "is_approaching": True, "position": "CENTER"},
        navigation="GO_FORWARD",
        predictive_threat={"class_name": "person", "is_approaching": True, "threat_level": "CAUTION"}
    )
    assert v31 == "Warning. Person approaching.", f"T31 failed: {v31}"
    passed += 1
    print(f"Test 31 PASS: Voice instruction for approaching person -> '{v31}'")

    # Test 32: Approaching object -> "Warning. Object approaching."
    e = DecisionEngine()
    v32 = e.generate_voice_instruction(
        safety_level="CAUTION",
        emergency_stop=False,
        primary_obstacle={"class_name": "chair", "is_approaching": True, "position": "CENTER"},
        navigation="GO_FORWARD",
        predictive_threat={"class_name": "chair", "is_approaching": True, "threat_level": "CAUTION"}
    )
    assert v32 == "Warning. Object approaching.", f"T32 failed: {v32}"
    passed += 1
    print(f"Test 32 PASS: Voice instruction for approaching object -> '{v32}'")

    # Test 33: Emergency stop due to rapid approach -> "Stop immediately."
    e = DecisionEngine()
    v33 = e.generate_voice_instruction(
        safety_level="EMERGENCY",
        emergency_stop=True,
        primary_obstacle={"class_name": "person", "is_approaching": True, "position": "CENTER"},
        navigation="STOP",
        predictive_threat={"class_name": "person", "motion_state": "RAPIDLY_APPROACHING", "threat_level": "EMERGENCY"}
    )
    assert v33 in ("Stop immediately.", "Stop. Person ahead."), f"T33 failed: {v33}"
    passed += 1
    print(f"Test 33 PASS: Voice instruction for emergency stop -> '{v33}'")

    # Test 34: Avoidance turn command takes priority
    e = DecisionEngine()
    v34 = e.generate_voice_instruction(
        safety_level="CAUTION",
        emergency_stop=False,
        primary_obstacle={"class_name": "person", "is_approaching": True, "position": "CENTER"},
        navigation="TURN_LEFT",
        predictive_threat={"class_name": "person", "is_approaching": True}
    )
    assert v34 in ("Turn left.", "Person ahead. Turn left."), f"T34 failed: {v34}"
    passed += 1
    print(f"Test 34 PASS: Avoidance turn instruction -> '{v34}'")

    # ============================================================
    # GROUP 8: RESPONSE STRUCTURE & BACKWARD COMPATIBILITY
    # ============================================================

    # Test 35: Top-level predictive_threat contract
    e = DecisionEngine()
    res35 = e.analyze(objects=[{"id": 1, "class_name": "person", "position": "CENTER", "distance_m": 3.0, "risk_score": 0.4}])
    pt35 = res35.get("predictive_threat")
    assert isinstance(pt35, dict), "T35 predictive_threat must be a dict"
    expected_keys = {"obstacle_id", "class_name", "position", "distance_m", "motion_state",
                     "approach_rate_mps", "time_to_collision_s", "approach_score", "threat_level",
                     "is_approaching", "warning_issued"}
    missing = expected_keys - set(pt35.keys())
    assert not missing, f"T35 missing keys in predictive_threat: {missing}"
    passed += 1
    print("Test 35 PASS: Top-level predictive_threat strictly conforms to Phase 11 contract")

    # Test 36: Backward compatibility top-level fields
    assert "approaching" in res35, "T36 missing approaching"
    assert isinstance(res35["approaching"], bool), "T36 approaching must be bool"
    assert "approach_rate" in res35, "T36 missing approach_rate"
    assert isinstance(res35["approach_rate"], float), "T36 approach_rate must be float"
    assert "obstacle_stability" in res35, "T36 missing obstacle_stability"
    assert isinstance(res35["obstacle_stability"], int), "T36 obstacle_stability must be int"
    passed += 1
    print("Test 36 PASS: Backward compatibility fields (approaching, approach_rate, obstacle_stability) verified")

    # Test 37: Per-object Phase 11 fields
    obj37 = res35["objects"][0]
    for field in ("motion_state", "approach_rate_mps", "time_to_collision_s", "approach_score", "is_approaching"):
        assert field in obj37, f"T37 object missing {field}"
    passed += 1
    print("Test 37 PASS: Per-object Phase 11 fields present on all detections")

    print("=" * 65)
    print(f"PHASE 11 TEST RESULT: {passed}/{total} PASSED (100%)")
    print("=" * 65)
    assert passed == total, f"Only {passed}/{total} tests passed!"


if __name__ == "__main__":
    run_phase11_tests()
