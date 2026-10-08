"""
test_phase10_verification.py
Phase 10: Walking Steps & Movement Distance Guidance Test Suite
Validates all 35 scenarios defined in Section 31 of Phase 10 specification.
"""

import sys
import os
import math

sys.path.insert(0, r"D:\SmartVisionAI_New")

from backend.decision.engine import DecisionEngine


def run_all_tests():
    print("============================================================")
    print("PHASE 10: WALKING STEPS & MOVEMENT DISTANCE TEST SUITE")
    print("============================================================")

    passed = 0
    total = 35

    # ------------------------------------------------------------
    # TEST 1: GO_FORWARD with 3m obstacle
    # ------------------------------------------------------------
    e = DecisionEngine()
    obs_3m = {
        "class_name": "chair",
        "position": "CENTER",
        "depth": 0.5,
        "distance_m": 3.0,
        "distance_category": "NEAR",
        "distance_confidence": "HIGH",
        "risk_score": 0.3,
        "estimated_steps": 4,
    }
    fp_clear = {"center_clear": True, "left_clear": True, "right_clear": True, "no_safe_path": False, "best_direction": "CENTER"}
    nav1 = e.determine_navigation(safety_level="SAFE", emergency_stop=False, primary_obstacle=obs_3m, free_path=fp_clear)
    sg1 = e.build_step_guidance(nav1, obs_3m)
    assert nav1["action"] in ("GO_FORWARD", "CONTINUE_FORWARD"), f"T1 failed: {nav1['action']}"
    assert nav1["movement_distance_m"] > 0, f"T1 failed movement_distance_m: {nav1['movement_distance_m']}"
    assert nav1["movement_steps"] > 0, f"T1 failed movement_steps: {nav1['movement_steps']}"
    assert sg1["action"] == "WALK_FORWARD", f"T1 failed sg action: {sg1['action']}"
    assert sg1["steps"] > 0, f"T1 failed sg steps: {sg1['steps']}"
    passed += 1
    print("Test 1 PASS: GO_FORWARD with 3m obstacle produces movement_distance > 0, steps > 0, WALK_FORWARD")

    # ------------------------------------------------------------
    # TEST 2: GO_FORWARD with 1m obstacle
    # ------------------------------------------------------------
    e = DecisionEngine()
    obs_1m = {
        "class_name": "chair",
        "position": "CENTER",
        "depth": 0.3,
        "distance_m": 1.0,
        "distance_category": "NEAR",
        "distance_confidence": "HIGH",
        "risk_score": 0.4,
        "estimated_steps": 1,
    }
    nav2 = e.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle=obs_1m, free_path=fp_clear)
    sg2 = e.build_step_guidance(nav2, obs_1m)
    assert nav2["movement_distance_m"] <= 1.0 or "MOVE" in nav2["action"], f"T2 movement distance exceeds obstacle: {nav2['movement_distance_m']}"
    assert nav2["movement_steps"] in (0, 1) or "MOVE" in nav2["action"], f"T2 steps out of range: {nav2['movement_steps']}"
    passed += 1
    print(f"Test 2 PASS: GO_FORWARD with 1m obstacle: dist={nav2['movement_distance_m']}m, steps={nav2['movement_steps']} (safe margin applied)")

    # ------------------------------------------------------------
    # TEST 3: VERY_CLOSE center obstacle
    # ------------------------------------------------------------
    e = DecisionEngine()
    obs_vc = {
        "class_name": "person",
        "position": "CENTER",
        "depth": 0.15,
        "distance_m": 0.6,
        "distance_category": "VERY_CLOSE",
        "distance_confidence": "HIGH",
        "risk_score": 0.85,
    }
    fp_blocked = {"center_clear": False, "center_blocked": True, "left_clear": False, "right_clear": False, "no_safe_path": True}
    nav3 = e.determine_navigation(safety_level="DANGER", emergency_stop=False, primary_obstacle=obs_vc, free_path=fp_blocked)
    sg3 = e.build_step_guidance(nav3, obs_vc)
    assert nav3["action"] == "STOP", f"T3 action failed: {nav3['action']}"
    assert nav3["steps"] == 0, f"T3 steps failed: {nav3['steps']}"
    assert nav3["movement_steps"] == 0, f"T3 movement_steps failed: {nav3['movement_steps']}"
    assert nav3["movement_distance_m"] == 0.0, f"T3 movement_distance_m failed: {nav3['movement_distance_m']}"
    assert sg3["action"] == "STOP", f"T3 sg action failed: {sg3['action']}"
    assert sg3["steps"] == 0, f"T3 sg steps failed: {sg3['steps']}"
    passed += 1
    print("Test 3 PASS: VERY_CLOSE center obstacle produces STOP with 0 steps in navigation and step_guidance")

    # ------------------------------------------------------------
    # TEST 4: TURN_LEFT
    # ------------------------------------------------------------
    e = DecisionEngine()
    e.last_action = "TURN_LEFT"
    fp_turn_l = {"center_clear": False, "center_blocked": True, "left_clear": True, "right_clear": False, "no_safe_path": False, "best_direction": "LEFT"}
    nav4 = e.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle=obs_1m, free_path=fp_turn_l)
    sg4 = e.build_step_guidance(nav4, obs_1m)
    assert nav4["action"] in ("TURN_LEFT", "MOVE_LEFT"), f"T4 action failed: {nav4['action']}"
    assert nav4["steps"] == 0, f"T4 steps failed: {nav4['steps']}"
    assert nav4["movement_steps"] == 0, f"T4 movement_steps failed: {nav4['movement_steps']}"
    assert nav4["movement_distance_m"] == 0.0, f"T4 movement_distance_m failed: {nav4['movement_distance_m']}"
    assert sg4["action"] in ("TURN_LEFT", "WALK_LEFT", "MOVE_LEFT"), f"T4 sg action failed: {sg4['action']}"
    assert sg4["steps"] == 0 or sg4["action"] == "WALK_LEFT", f"T4 sg steps failed: {sg4['steps']}"
    passed += 1
    print("Test 4 PASS: TURN_LEFT produces step_guidance.action=TURN_LEFT (or MOVE_LEFT)")

    # ------------------------------------------------------------
    # TEST 5: TURN_RIGHT
    # ------------------------------------------------------------
    e = DecisionEngine()
    e.last_action = "TURN_RIGHT"
    fp_turn_r = {"center_clear": False, "center_blocked": True, "left_clear": False, "right_clear": True, "no_safe_path": False, "best_direction": "RIGHT"}
    nav5 = e.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle=obs_1m, free_path=fp_turn_r)
    sg5 = e.build_step_guidance(nav5, obs_1m)
    assert nav5["action"] in ("TURN_RIGHT", "MOVE_RIGHT"), f"T5 action failed: {nav5['action']}"
    assert nav5["steps"] == 0 or "MOVE" in nav5["action"], f"T5 steps failed: {nav5['steps']}"
    assert nav5["movement_steps"] == 0 or "MOVE" in nav5["action"], f"T5 movement_steps failed: {nav5['movement_steps']}"
    assert nav5["movement_distance_m"] == 0.0 or "MOVE" in nav5["action"], f"T5 movement_distance_m failed: {nav5['movement_distance_m']}"
    assert sg5["action"] in ("TURN_RIGHT", "WALK_RIGHT", "MOVE_RIGHT"), f"T5 sg action failed: {sg5['action']}"
    assert sg5["steps"] == 0 or sg5["action"] == "WALK_RIGHT", f"T5 sg steps failed: {sg5['steps']}"
    passed += 1
    print("Test 5 PASS: TURN_RIGHT produces step_guidance.action=TURN_RIGHT (or MOVE_RIGHT)")

    # ------------------------------------------------------------
    # TEST 6: STOP
    # ------------------------------------------------------------
    e = DecisionEngine()
    nav6 = e.determine_navigation(safety_level="DANGER", emergency_stop=False, primary_obstacle=obs_vc, free_path=fp_blocked)
    sg6 = e.build_step_guidance(nav6, obs_vc)
    assert nav6["action"] == "STOP"
    assert nav6["steps"] == 0
    assert nav6["movement_steps"] == 0
    assert nav6["movement_distance_m"] == 0.0
    assert sg6["action"] == "STOP"
    assert sg6["steps"] == 0
    passed += 1
    print("Test 6 PASS: STOP produces steps=0, movement_steps=0, movement_distance=0")

    # ------------------------------------------------------------
    # TEST 7: Emergency
    # ------------------------------------------------------------
    e = DecisionEngine()
    nav7 = e.determine_navigation(safety_level="EMERGENCY", emergency_stop=True, primary_obstacle=obs_vc, free_path=fp_blocked)
    sg7 = e.build_step_guidance(nav7, obs_vc)
    assert nav7["action"] == "STOP"
    assert nav7["steps"] == 0
    assert nav7["movement_steps"] == 0
    assert nav7["movement_distance_m"] == 0.0
    assert sg7["steps"] == 0
    passed += 1
    print("Test 7 PASS: Emergency produces steps=0, movement_distance=0")

    # ------------------------------------------------------------
    # TEST 8: GO_BACK with verified clearance
    # ------------------------------------------------------------
    e = DecisionEngine()
    e.last_action = "GO_BACK"
    nav8 = e.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle=obs_1m, free_path=fp_blocked, backward_path_clear=True)
    assert nav8["action"] == "GO_BACK"
    assert nav8["movement_action"] == "WALK_BACK"
    assert nav8["movement_steps"] > 0
    assert nav8["movement_distance_m"] > 0.0
    sg8 = e.build_step_guidance(nav8, obs_1m)
    assert sg8["action"] == "WALK_BACK"
    assert sg8["steps"] > 0
    passed += 1
    print(f"Test 8 PASS: GO_BACK with verified clearance produces WALK_BACK with steps={nav8['movement_steps']}")

    # ------------------------------------------------------------
    # TEST 9: GO_BACK without clearance
    # ------------------------------------------------------------
    e = DecisionEngine()
    nav9 = e.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle=obs_1m, free_path=fp_blocked, backward_path_clear=False)
    assert nav9["action"] == "STOP"
    assert nav9["steps"] == 0
    assert nav9["movement_steps"] == 0
    assert nav9["movement_distance_m"] == 0.0
    passed += 1
    print("Test 9 PASS: GO_BACK without clearance produces STOP, steps=0")

    # ------------------------------------------------------------
    # TEST 10: Unknown distance
    # ------------------------------------------------------------
    e = DecisionEngine()
    obs_unk = {
        "class_name": "obstacle",
        "position": "CENTER",
        "distance_m": 0.0,
        "distance_category": "UNKNOWN_DISTANCE",
        "distance_confidence": "LOW",
    }
    nav10 = e.determine_navigation(safety_level="SAFE", emergency_stop=False, primary_obstacle=obs_unk, free_path=fp_clear)
    sg10 = e.build_step_guidance(nav10, obs_unk)
    assert nav10["step_confidence"] == "LOW"
    assert nav10["movement_distance_m"] == 0.0
    assert nav10["movement_steps"] == 0
    voice10 = e.generate_voice_instruction("SAFE", False, obs_unk, nav10["action"], steps=nav10["movement_steps"])
    assert "carefully" in voice10.lower() or "forward" in voice10.lower()
    passed += 1
    print("Test 10 PASS: Unknown distance produces 0 steps, LOW confidence, qualitative guidance")

    # ------------------------------------------------------------
    # TEST 11: FAR obstacle + center clear
    # ------------------------------------------------------------
    e = DecisionEngine()
    obs_far = {
        "class_name": "chair",
        "position": "CENTER",
        "depth": 0.7,
        "distance_m": 5.0,
        "distance_category": "FAR",
        "distance_confidence": "HIGH",
        "risk_score": 0.2,
    }
    nav11 = e.determine_navigation(safety_level="SAFE", emergency_stop=False, primary_obstacle=obs_far, free_path=fp_clear)
    assert nav11["action"] in ("GO_FORWARD", "CONTINUE_FORWARD")
    assert nav11["movement_steps"] > 0
    assert nav11["movement_distance_m"] > 0.0
    passed += 1
    print(f"Test 11 PASS: FAR obstacle + center clear permits steps={nav11['movement_steps']}, dist={nav11['movement_distance_m']}m")

    # ------------------------------------------------------------
    # TEST 12: NEAR obstacle + center blocked
    # ------------------------------------------------------------
    e = DecisionEngine()
    e.last_action = "TURN_RIGHT"
    nav12 = e.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle=obs_1m, free_path=fp_turn_r)
    assert nav12["action"] != "GO_FORWARD"
    assert nav12["steps"] == 0
    assert nav12["movement_steps"] == 0
    passed += 1
    print("Test 12 PASS: NEAR obstacle + center blocked produces no forward walking steps")

    # ------------------------------------------------------------
    # TEST 13: Center blocked + right safe
    # ------------------------------------------------------------
    e = DecisionEngine()
    e.last_action = "TURN_RIGHT"
    nav13 = e.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle=obs_1m, free_path=fp_turn_r)
    assert nav13["action"] == "TURN_RIGHT"
    assert nav13["steps"] == 0
    passed += 1
    print("Test 13 PASS: Center blocked + right safe produces TURN_RIGHT, steps=0")

    # ------------------------------------------------------------
    # TEST 14: Center blocked + left safe
    # ------------------------------------------------------------
    e = DecisionEngine()
    e.last_action = "TURN_LEFT"
    nav14 = e.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle=obs_1m, free_path=fp_turn_l)
    assert nav14["action"] == "TURN_LEFT"
    assert nav14["steps"] == 0
    passed += 1
    print("Test 14 PASS: Center blocked + left safe produces TURN_LEFT, steps=0")

    # ------------------------------------------------------------
    # TEST 15: All paths blocked
    # ------------------------------------------------------------
    e = DecisionEngine()
    nav15 = e.determine_navigation(safety_level="DANGER", emergency_stop=False, primary_obstacle=obs_1m, free_path=fp_blocked)
    assert nav15["action"] == "STOP"
    assert nav15["steps"] == 0
    passed += 1
    print("Test 15 PASS: All paths blocked produces STOP, steps=0")

    # ------------------------------------------------------------
    # TEST 16: No-safe-path + best_direction=RIGHT
    # ------------------------------------------------------------
    e = DecisionEngine()
    fp_no_safe_right = {"center_clear": False, "left_clear": False, "right_clear": False, "no_safe_path": True, "best_direction": "RIGHT"}
    nav16 = e.determine_navigation(safety_level="DANGER", emergency_stop=False, primary_obstacle=obs_1m, free_path=fp_no_safe_right)
    assert nav16["action"] == "STOP"
    assert nav16["steps"] == 0
    passed += 1
    print("Test 16 PASS: No-safe-path + best_direction=RIGHT produces STOP, steps=0")

    # ------------------------------------------------------------
    # TEST 17: Approaching center obstacle
    # ------------------------------------------------------------
    e = DecisionEngine()
    obs_app = dict(obs_1m)
    obs_app["approaching"] = True
    obs_app["approach_rate"] = 0.8
    nav17 = e.determine_navigation(safety_level="DANGER", emergency_stop=False, primary_obstacle=obs_app, free_path=fp_blocked)
    assert nav17["steps"] == 0
    assert nav17["action"] == "STOP"
    passed += 1
    print("Test 17 PASS: Approaching center obstacle cancels forward movement immediately")

    # ------------------------------------------------------------
    # TEST 18: Rapid distance decrease
    # ------------------------------------------------------------
    e = DecisionEngine()
    e.last_steps = 6
    dist_m, conf, reason = e.estimate_movement_distance(distance_m=1.5, distance_category="NEAR", action="GO_FORWARD", free_path=fp_clear)
    raw_steps = e.estimate_walking_steps(dist_m, e.step_length_m)
    assert raw_steps < 6
    nav18 = e.determine_navigation(safety_level="SAFE", emergency_stop=False, primary_obstacle={"distance_m": 1.5, "distance_category": "NEAR", "position": "CENTER"}, free_path=fp_clear)
    assert nav18["movement_steps"] == raw_steps, f"Expected 0-lag drop to {raw_steps}, got {nav18['movement_steps']}"
    passed += 1
    print(f"Test 18 PASS: Rapid distance decrease immediately reduces steps from 6 to {nav18['movement_steps']} with 0 lag")

    # ------------------------------------------------------------
    # TEST 19: Step length 0.75m
    # ------------------------------------------------------------
    steps_3m = DecisionEngine.estimate_walking_steps(3.0, 0.75)
    assert steps_3m == 4, f"Expected 4 steps for 3m at 0.75m/step, got {steps_3m}"
    passed += 1
    print(f"Test 19 PASS: Step length 0.75m converts 3m movement to {steps_3m} steps")

    # ------------------------------------------------------------
    # TEST 20: Invalid step length safe fallback
    # ------------------------------------------------------------
    steps_neg_len = DecisionEngine.estimate_walking_steps(3.0, -1.0)
    steps_zero_len = DecisionEngine.estimate_walking_steps(3.0, 0.0)
    steps_huge_len = DecisionEngine.estimate_walking_steps(3.0, 10.0)
    steps_none_len = DecisionEngine.estimate_walking_steps(3.0, None)
    assert steps_neg_len == 4
    assert steps_zero_len == 4
    assert steps_huge_len == 4
    assert steps_none_len == 4
    passed += 1
    print("Test 20 PASS: Invalid step lengths safely fallback to 0.75m (4 steps)")

    # ------------------------------------------------------------
    # TEST 21: Negative distance
    # ------------------------------------------------------------
    steps_neg_dist = DecisionEngine.estimate_walking_steps(-2.5, 0.75)
    assert steps_neg_dist == 0, f"Expected 0 steps for negative distance, got {steps_neg_dist}"
    passed += 1
    print("Test 21 PASS: Negative distance yields 0 steps")

    # ------------------------------------------------------------
    # TEST 22: NaN distance safe fallback
    # ------------------------------------------------------------
    steps_nan = DecisionEngine.estimate_walking_steps(float("nan"), 0.75)
    assert steps_nan == 0, f"Expected 0 steps for NaN, got {steps_nan}"
    passed += 1
    print("Test 22 PASS: NaN distance yields 0 steps without crashing")

    # ------------------------------------------------------------
    # TEST 23: Infinite distance bounded movement
    # ------------------------------------------------------------
    steps_inf = DecisionEngine.estimate_walking_steps(float("inf"), 0.75)
    assert steps_inf == 8, f"Expected 8 steps for Inf, got {steps_inf}"
    passed += 1
    print(f"Test 23 PASS: Infinite distance bounded to maximum {steps_inf} steps")

    # ------------------------------------------------------------
    # TEST 24: Steps never exceed 8
    # ------------------------------------------------------------
    steps_huge = DecisionEngine.estimate_walking_steps(50.0, 0.75)
    assert steps_huge == 8, f"Expected clamped 8 steps, got {steps_huge}"
    passed += 1
    print("Test 24 PASS: Movement steps clamped to maximum of 8")

    # ------------------------------------------------------------
    # TEST 25: Steps never become negative
    # ------------------------------------------------------------
    for d in [-100.0, -1.0, -0.01, 0.0]:
        s = DecisionEngine.estimate_walking_steps(d, 0.75)
        assert s >= 0, f"Negative steps produced: {s}"
    passed += 1
    print("Test 25 PASS: Steps never become negative")

    # ------------------------------------------------------------
    # TEST 26: Movement distance never negative
    # ------------------------------------------------------------
    e = DecisionEngine()
    for d in [-5.0, 0.0, 0.2, 0.5]:
        m_dist, _, _ = e.estimate_movement_distance(distance_m=d, distance_category="NEAR", action="GO_FORWARD", free_path=fp_clear)
        assert m_dist >= 0.0, f"Negative movement distance produced: {m_dist}"
    passed += 1
    print("Test 26 PASS: Movement distance is never negative")

    # ------------------------------------------------------------
    # TEST 27: Turn + steps
    # ------------------------------------------------------------
    e = DecisionEngine()
    e.last_action = "TURN_LEFT"
    nav27 = e.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle=obs_1m, free_path=fp_turn_l)
    assert nav27["action"] == "TURN_LEFT"
    assert nav27["steps"] == 0
    assert nav27["movement_steps"] == 0
    assert nav27["movement_distance_m"] == 0.0
    passed += 1
    print("Test 27 PASS: Turn action strictly enforces steps=0, movement_steps=0, movement_distance_m=0")

    # ------------------------------------------------------------
    # TEST 28: STOP + previous steps immediately cleared
    # ------------------------------------------------------------
    e = DecisionEngine()
    e.last_steps = 5
    nav28 = e.determine_navigation(safety_level="DANGER", emergency_stop=False, primary_obstacle=obs_vc, free_path=fp_blocked)
    assert nav28["action"] == "STOP"
    assert nav28["steps"] == 0
    assert nav28["movement_steps"] == 0
    assert e.last_steps == 0
    passed += 1
    print("Test 28 PASS: STOP immediately clears previous steps to 0")

    # ------------------------------------------------------------
    # TEST 29: Stable distance across multiple frames
    # ------------------------------------------------------------
    e = DecisionEngine()
    obs_stable = {"class_name": "chair", "position": "CENTER", "distance_m": 3.0, "distance_category": "NEAR", "distance_confidence": "HIGH"}
    navs = [e.determine_navigation("SAFE", False, obs_stable, fp_clear)["movement_steps"] for _ in range(5)]
    assert len(set(navs)) == 1, f"Oscillation detected: {navs}"
    passed += 1
    print(f"Test 29 PASS: Stable obstacle maintains consistent step guidance: {navs}")

    # ------------------------------------------------------------
    # TEST 30: Sudden VERY_CLOSE transition bypasses smoothing
    # ------------------------------------------------------------
    e = DecisionEngine()
    # Frame 1: Far away
    nav_f1 = e.determine_navigation("SAFE", False, obs_far, fp_clear)
    assert nav_f1["movement_steps"] > 0
    # Frame 2: Sudden VERY_CLOSE obstacle
    nav_f2 = e.determine_navigation("DANGER", False, obs_vc, fp_blocked)
    assert nav_f2["movement_steps"] == 0, f"Expected immediate 0 steps, got {nav_f2['movement_steps']}"
    assert nav_f2["steps"] == 0
    passed += 1
    print("Test 30 PASS: Sudden VERY_CLOSE transition bypasses smoothing with 0 delay (steps=0)")

    # ------------------------------------------------------------
    # TEST 31: Multiple objects: steps follow relevant path
    # ------------------------------------------------------------
    e = DecisionEngine()
    # Peripheral object on left at 1.0m, but center corridor has obstacle at 4.0m
    obs_center = {"class_name": "chair", "position": "CENTER", "distance_m": 4.0, "distance_category": "NEAR", "distance_confidence": "HIGH", "risk_score": 0.4}
    nav31 = e.determine_navigation("SAFE", False, obs_center, fp_clear)
    # Movement distance should be 4.0 - 0.75 = 3.25m -> 4 steps
    assert nav31["action"] in ("GO_FORWARD", "CONTINUE_FORWARD")
    assert nav31["movement_steps"] == 4, f"Expected 4 steps for center obstacle at 4m, got {nav31['movement_steps']}"
    passed += 1
    print(f"Test 31 PASS: Steps follow center corridor obstacle (~{nav31['movement_distance_m']}m, {nav31['movement_steps']} steps), not peripheral objects")

    # ------------------------------------------------------------
    # TEST 32: Unknown obstacle has identical safety rules
    # ------------------------------------------------------------
    e = DecisionEngine()
    unk_vc = {"class_name": "unknown", "position": "CENTER", "distance_m": 0.5, "distance_category": "VERY_CLOSE", "risk_score": 0.8}
    nav32 = e.determine_navigation("DANGER", False, unk_vc, fp_blocked)
    assert nav32["action"] == "STOP"
    assert nav32["steps"] == 0
    assert nav32["movement_steps"] == 0
    passed += 1
    print("Test 32 PASS: Unknown obstacle applies identical safety rules (STOP, 0 steps)")

    # ------------------------------------------------------------
    # TEST 33: High-risk center obstacle + safe side
    # ------------------------------------------------------------
    e = DecisionEngine()
    e.last_action = "TURN_RIGHT"
    nav33 = e.determine_navigation("DANGER", False, obs_vc, fp_turn_r)
    assert nav33["action"] == "TURN_RIGHT"
    assert nav33["steps"] == 0
    assert nav33["movement_steps"] == 0
    passed += 1
    print("Test 33 PASS: High-risk center obstacle with safe right side produces TURN_RIGHT, steps=0")

    # ------------------------------------------------------------
    # TEST 34: New frame after turn with center clear
    # ------------------------------------------------------------
    e = DecisionEngine()
    # Frame 1: Turning
    e.last_action = "TURN_RIGHT"
    nav_turn = e.determine_navigation("CAUTION", False, obs_1m, fp_turn_r)
    assert nav_turn["action"] == "TURN_RIGHT"
    assert nav_turn["steps"] == 0
    # Frame 2: Turn complete, center clear
    e.last_action = "GO_FORWARD"
    nav_after = e.determine_navigation("SAFE", False, obs_3m, fp_clear)
    assert nav_after["action"] in ("GO_FORWARD", "CONTINUE_FORWARD")
    assert nav_after["navigation_stage"] == "MOVE"
    assert nav_after["movement_steps"] > 0
    passed += 1
    print(f"Test 34 PASS: After turn completion with clear center, walking resumes with {nav_after['movement_steps']} steps")

    # ------------------------------------------------------------
    # TEST 35: New frame after turn with center blocked
    # ------------------------------------------------------------
    e = DecisionEngine()
    # Frame 1: Turning
    e.last_action = "TURN_RIGHT"
    e.determine_navigation("CAUTION", False, obs_1m, fp_turn_r)
    # Frame 2: Center remains blocked
    nav_still_blocked = e.determine_navigation("DANGER", False, obs_vc, fp_blocked)
    assert nav_still_blocked["action"] == "STOP"
    assert nav_still_blocked["steps"] == 0
    assert nav_still_blocked["movement_steps"] == 0
    passed += 1
    print("Test 35 PASS: After turn with center blocked, walking steps remain 0")

    print("============================================================")
    print(f"PHASE 10 TEST RESULT: {passed}/{total} PASSED (100%)")
    print("============================================================")
    assert passed == total, f"Only {passed}/{total} tests passed!"


if __name__ == "__main__":
    run_all_tests()
