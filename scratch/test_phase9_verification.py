"""
======================================================================
SmartVisionAI - Phase 9 Dedicated Verification Suite
TURN-vs-MOVEMENT SEPARATION & TURN ANGLE ESTIMATION
======================================================================
Tests 1-30:
1. GO_FORWARD -> stage=MOVE, movement_required=True, turn_required=False, movement_action=WALK_FORWARD
2. TURN_LEFT -> stage=TURN, turn_required=True, movement_required=False, turn_direction=LEFT
3. TURN_RIGHT -> stage=TURN, turn_required=True, movement_required=False, turn_direction=RIGHT
4. STOP -> stage=STOP, movement_required=False, turn_required=False
5. GO_BACK with verified clearance -> stage=MOVE, movement_action=WALK_BACK
6. GO_BACK without clearance -> STOP
7. LEFT turn angle -> valid angle, 15 <= angle <= 60, direction=LEFT
8. RIGHT turn angle -> valid angle, 15 <= angle <= 60, direction=RIGHT
9. All blocked -> STOP, angle=0, direction=NONE
10. Emergency -> immediate STOP, angle=0
11. No-safe-path + best_direction=RIGHT -> STOP
12. Center blocked + left safe -> TURN_LEFT
13. Center blocked + right safe -> TURN_RIGHT
14. Center safe -> GO_FORWARD
15. Approaching center obstacle + left safe -> TURN_LEFT
16. Approaching center obstacle + right safe -> TURN_RIGHT
17. Turn does NOT require movement -> turn_required=True, movement_required=False
18. Movement does NOT require turn -> turn_required=False, movement_required=True
19. Turn angle is not negative -> turn_angle_deg >= 0
20. Turn angle bounded -> angle <= 60
21. Turn state persists -> TURN_LEFT followed by stable TURN_LEFT -> TURNING_LEFT
22. Direction changes -> TURN_LEFT followed by stable TURN_RIGHT -> old turn cancelled, new nav evaluated
23. Emergency during turn -> immediate STOP
24. New scene after turn -> old safe corridor becomes blocked -> MOVE is NOT blindly issued
25. Voice mapping -> TURN_LEFT: "Turn left.", TURN_RIGHT: "Turn right.", GO_FORWARD: forward inst, STOP: "Stop."
26. NaN / Inf inputs -> safe fallback, no crash
27. Missing / None free_path -> safe fallback, no crash
28. Unknown obstacle participates identically in turn-vs-movement decisions
29. High center coverage -> angle estimation yields 45.0
30. Low path confidence -> turn confidence is LOW with coarse 30.0
"""

import sys
import os
sys.path.insert(0, r"D:\SmartVisionAI_New")

from backend.decision.engine import DecisionEngine

def run_phase9_verification():
    print("=" * 70)
    print("STARTING PHASE 9 TURN-VS-MOVEMENT VERIFICATION SUITE")
    print("=" * 70)

    # ----------------------------------------------------------------
    # TEST 1: GO_FORWARD
    # ----------------------------------------------------------------
    print("\n--- TEST 1: GO_FORWARD ---")
    e = DecisionEngine()
    fp1 = {"left_clear": True, "center_clear": True, "right_clear": True, "no_safe_path": False}
    res1 = e.determine_navigation(safety_level="SAFE", emergency_stop=False, primary_obstacle=None, free_path=fp1)
    print(f"Result 1: action={res1['action']}, stage={res1['navigation_stage']}, turn_req={res1['turn_required']}, mov_req={res1['movement_required']}, mov_act={res1['movement_action']}")
    assert res1["action"] == "GO_FORWARD"
    assert res1["navigation_stage"] == "MOVE"
    assert res1["turn_required"] is False
    assert res1["movement_required"] is True
    assert res1["movement_action"] == "WALK_FORWARD"
    assert res1["turn_direction"] == "NONE"
    print("TEST 1 Result: PASS (GO_FORWARD -> stage=MOVE, movement_action=WALK_FORWARD)")

    # ----------------------------------------------------------------
    # TEST 2: TURN_LEFT
    # ----------------------------------------------------------------
    print("\n--- TEST 2: TURN_LEFT ---")
    e = DecisionEngine()
    e.last_action = "TURN_LEFT"
    fp2 = {"left_clear": True, "center_clear": False, "right_clear": False, "no_safe_path": False, "center_coverage": 0.5}
    res2 = e.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle={"position": "CENTER"}, free_path=fp2)
    print(f"Result 2: action={res2['action']}, stage={res2['navigation_stage']}, turn_req={res2['turn_required']}, mov_req={res2['movement_required']}, turn_dir={res2['turn_direction']}")
    assert res2["action"] == "TURN_LEFT"
    assert res2["navigation_stage"] == "TURN"
    assert res2["turn_required"] is True
    assert res2["movement_required"] is False
    assert res2["turn_direction"] == "LEFT"
    assert res2["movement_action"] == "NONE"
    print("TEST 2 Result: PASS (TURN_LEFT -> stage=TURN, turn_req=True, mov_req=False, turn_dir=LEFT)")

    # ----------------------------------------------------------------
    # TEST 3: TURN_RIGHT
    # ----------------------------------------------------------------
    print("\n--- TEST 3: TURN_RIGHT ---")
    e = DecisionEngine()
    e.last_action = "TURN_RIGHT"
    fp3 = {"left_clear": False, "center_clear": False, "right_clear": True, "no_safe_path": False, "center_coverage": 0.5}
    res3 = e.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle={"position": "CENTER"}, free_path=fp3)
    print(f"Result 3: action={res3['action']}, stage={res3['navigation_stage']}, turn_req={res3['turn_required']}, mov_req={res3['movement_required']}, turn_dir={res3['turn_direction']}")
    assert res3["action"] == "TURN_RIGHT"
    assert res3["navigation_stage"] == "TURN"
    assert res3["turn_required"] is True
    assert res3["movement_required"] is False
    assert res3["turn_direction"] == "RIGHT"
    assert res3["movement_action"] == "NONE"
    print("TEST 3 Result: PASS (TURN_RIGHT -> stage=TURN, turn_req=True, mov_req=False, turn_dir=RIGHT)")

    # ----------------------------------------------------------------
    # TEST 4: STOP
    # ----------------------------------------------------------------
    print("\n--- TEST 4: STOP ---")
    e = DecisionEngine()
    fp4 = {"left_clear": False, "center_clear": False, "right_clear": False, "no_safe_path": True}
    res4 = e.determine_navigation(safety_level="SAFE", emergency_stop=False, primary_obstacle=None, free_path=fp4)
    print(f"Result 4: action={res4['action']}, stage={res4['navigation_stage']}, turn_req={res4['turn_required']}, mov_req={res4['movement_required']}")
    assert res4["action"] == "STOP"
    assert res4["navigation_stage"] == "STOP"
    assert res4["turn_required"] is False
    assert res4["movement_required"] is False
    assert res4["turn_direction"] == "NONE"
    assert res4["movement_action"] == "NONE"
    assert res4["turn_angle_deg"] == 0.0
    print("TEST 4 Result: PASS (STOP -> stage=STOP, turn_req=False, mov_req=False)")

    # ----------------------------------------------------------------
    # TEST 5: GO_BACK with verified clearance
    # ----------------------------------------------------------------
    print("\n--- TEST 5: GO_BACK with verified clearance ---")
    e = DecisionEngine()
    e.last_action = "GO_BACK"
    res5 = e.determine_navigation(safety_level="SAFE", emergency_stop=False, primary_obstacle=None, free_path=fp4, backward_path_clear=True)
    print(f"Result 5: action={res5['action']}, stage={res5['navigation_stage']}, mov_act={res5['movement_action']}")
    assert res5["action"] == "GO_BACK"
    assert res5["navigation_stage"] == "MOVE"
    assert res5["movement_action"] == "WALK_BACK"
    assert res5["movement_required"] is True
    assert res5["turn_required"] is False
    print("TEST 5 Result: PASS (GO_BACK with clearance -> stage=MOVE, mov_act=WALK_BACK)")

    # ----------------------------------------------------------------
    # TEST 6: GO_BACK without clearance
    # ----------------------------------------------------------------
    print("\n--- TEST 6: GO_BACK without clearance ---")
    e = DecisionEngine()
    res6 = e.determine_navigation(safety_level="SAFE", emergency_stop=False, primary_obstacle=None, free_path=fp4, backward_path_clear=False)
    print(f"Result 6: action={res6['action']}, stage={res6['navigation_stage']}")
    assert res6["action"] == "STOP"
    assert res6["navigation_stage"] == "STOP"
    print("TEST 6 Result: PASS (GO_BACK without clearance -> STOP)")

    # ----------------------------------------------------------------
    # TEST 7: LEFT turn angle
    # ----------------------------------------------------------------
    print("\n--- TEST 7: LEFT turn angle ---")
    e = DecisionEngine()
    angle, t_dir, conf = e.estimate_turn_angle(action="TURN_LEFT", direction="LEFT", free_path={"left_clear": True, "center_coverage": 0.4, "path_confidence": "HIGH"})
    print(f"Result 7: angle={angle}, dir={t_dir}, conf={conf}")
    assert 15.0 <= angle <= 60.0
    assert t_dir == "LEFT"
    assert angle >= 0.0
    print("TEST 7 Result: PASS (Valid LEFT turn angle within [15, 60])")

    # ----------------------------------------------------------------
    # TEST 8: RIGHT turn angle
    # ----------------------------------------------------------------
    print("\n--- TEST 8: RIGHT turn angle ---")
    e = DecisionEngine()
    angle, t_dir, conf = e.estimate_turn_angle(action="TURN_RIGHT", direction="RIGHT", free_path={"right_clear": True, "center_coverage": 0.4, "path_confidence": "HIGH"})
    print(f"Result 8: angle={angle}, dir={t_dir}, conf={conf}")
    assert 15.0 <= angle <= 60.0
    assert t_dir == "RIGHT"
    assert angle >= 0.0
    print("TEST 8 Result: PASS (Valid RIGHT turn angle within [15, 60])")

    # ----------------------------------------------------------------
    # TEST 9: All blocked
    # ----------------------------------------------------------------
    print("\n--- TEST 9: All blocked ---")
    e = DecisionEngine()
    res9 = e.determine_navigation(safety_level="SAFE", emergency_stop=False, primary_obstacle=None, free_path=fp4)
    print(f"Result 9: action={res9['action']}, angle={res9['turn_angle_deg']}, dir={res9['turn_direction']}")
    assert res9["action"] == "STOP"
    assert res9["turn_angle_deg"] == 0.0
    assert res9["turn_direction"] == "NONE"
    print("TEST 9 Result: PASS (All blocked -> STOP, angle=0, dir=NONE)")

    # ----------------------------------------------------------------
    # TEST 10: Emergency
    # ----------------------------------------------------------------
    print("\n--- TEST 10: Emergency ---")
    e = DecisionEngine()
    res10 = e.determine_navigation(safety_level="EMERGENCY", emergency_stop=True, primary_obstacle=None, free_path=fp1)
    print(f"Result 10: action={res10['action']}, stage={res10['navigation_stage']}, angle={res10['turn_angle_deg']}")
    assert res10["action"] == "STOP"
    assert res10["navigation_stage"] == "STOP"
    assert res10["turn_angle_deg"] == 0.0
    print("TEST 10 Result: PASS (Emergency -> immediate STOP, angle=0)")

    # ----------------------------------------------------------------
    # TEST 11: No-safe-path + best_direction=RIGHT
    # ----------------------------------------------------------------
    print("\n--- TEST 11: No-safe-path + best_direction=RIGHT ---")
    e = DecisionEngine()
    fp11 = {"left_clear": False, "center_clear": False, "right_clear": False, "no_safe_path": True, "best_direction": "RIGHT"}
    res11 = e.determine_navigation(safety_level="SAFE", emergency_stop=False, primary_obstacle=None, free_path=fp11)
    print(f"Result 11: action={res11['action']}, stage={res11['navigation_stage']}")
    assert res11["action"] == "STOP"
    assert res11["navigation_stage"] == "STOP"
    print("TEST 11 Result: PASS (No-safe-path with best_direction=RIGHT -> STOP)")

    # ----------------------------------------------------------------
    # TEST 12: Center blocked + left safe
    # ----------------------------------------------------------------
    print("\n--- TEST 12: Center blocked + left safe ---")
    e = DecisionEngine()
    e.last_action = "TURN_LEFT"
    res12 = e.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle={"position": "CENTER"}, free_path=fp2)
    assert res12["action"] == "TURN_LEFT"
    assert res12["navigation_stage"] == "TURN"
    print("TEST 12 Result: PASS (Center blocked + left safe -> TURN_LEFT)")

    # ----------------------------------------------------------------
    # TEST 13: Center blocked + right safe
    # ----------------------------------------------------------------
    print("\n--- TEST 13: Center blocked + right safe ---")
    e = DecisionEngine()
    e.last_action = "TURN_RIGHT"
    res13 = e.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle={"position": "CENTER"}, free_path=fp3)
    assert res13["action"] == "TURN_RIGHT"
    assert res13["navigation_stage"] == "TURN"
    print("TEST 13 Result: PASS (Center blocked + right safe -> TURN_RIGHT)")

    # ----------------------------------------------------------------
    # TEST 14: Center safe
    # ----------------------------------------------------------------
    print("\n--- TEST 14: Center safe ---")
    e = DecisionEngine()
    res14 = e.determine_navigation(safety_level="SAFE", emergency_stop=False, primary_obstacle=None, free_path=fp1)
    assert res14["action"] == "GO_FORWARD"
    assert res14["navigation_stage"] == "MOVE"
    print("TEST 14 Result: PASS (Center safe -> GO_FORWARD, stage=MOVE)")

    # ----------------------------------------------------------------
    # TEST 15: Approaching center obstacle + left safe
    # ----------------------------------------------------------------
    print("\n--- TEST 15: Approaching center obstacle + left safe ---")
    e = DecisionEngine()
    e.last_action = "TURN_LEFT"
    res15 = e.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle={"position": "CENTER", "approaching": True}, free_path=fp2)
    assert res15["action"] == "TURN_LEFT"
    assert res15["navigation_stage"] == "TURN"
    print("TEST 15 Result: PASS (Approaching center obstacle + left safe -> TURN_LEFT)")

    # ----------------------------------------------------------------
    # TEST 16: Approaching center obstacle + right safe
    # ----------------------------------------------------------------
    print("\n--- TEST 16: Approaching center obstacle + right safe ---")
    e = DecisionEngine()
    e.last_action = "TURN_RIGHT"
    res16 = e.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle={"position": "CENTER", "approaching": True}, free_path=fp3)
    assert res16["action"] == "TURN_RIGHT"
    assert res16["navigation_stage"] == "TURN"
    print("TEST 16 Result: PASS (Approaching center obstacle + right safe -> TURN_RIGHT)")

    # ----------------------------------------------------------------
    # TEST 17: Turn does NOT require movement
    # ----------------------------------------------------------------
    print("\n--- TEST 17: Turn does NOT require movement ---")
    e = DecisionEngine()
    e.last_action = "TURN_LEFT"
    res17 = e.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle={"position": "CENTER"}, free_path=fp2)
    assert res17["turn_required"] is True
    assert res17["movement_required"] is False
    assert res17["movement_action"] == "NONE"
    print("TEST 17 Result: PASS (Turn does NOT require movement)")

    # ----------------------------------------------------------------
    # TEST 18: Movement does NOT require turn
    # ----------------------------------------------------------------
    print("\n--- TEST 18: Movement does NOT require turn ---")
    e = DecisionEngine()
    res18 = e.determine_navigation(safety_level="SAFE", emergency_stop=False, primary_obstacle=None, free_path=fp1)
    assert res18["turn_required"] is False
    assert res18["movement_required"] is True
    assert res18["turn_direction"] == "NONE"
    print("TEST 18 Result: PASS (Movement does NOT require turn)")

    # ----------------------------------------------------------------
    # TEST 19: Turn angle is not negative
    # ----------------------------------------------------------------
    print("\n--- TEST 19: Turn angle is not negative ---")
    e = DecisionEngine()
    angle_l, dir_l, _ = e.estimate_turn_angle(action="TURN_LEFT", direction="LEFT", free_path=fp2)
    angle_r, dir_r, _ = e.estimate_turn_angle(action="TURN_RIGHT", direction="RIGHT", free_path=fp3)
    assert angle_l >= 0.0 and dir_l == "LEFT"
    assert angle_r >= 0.0 and dir_r == "RIGHT"
    print("TEST 19 Result: PASS (Turn angle is non-negative magnitude)")

    # ----------------------------------------------------------------
    # TEST 20: Turn angle bounded
    # ----------------------------------------------------------------
    print("\n--- TEST 20: Turn angle bounded ---")
    e = DecisionEngine()
    extreme_fp = {"left_clear": True, "center_coverage": 1.0, "center_risk": 1.0}
    angle_ext, _, _ = e.estimate_turn_angle(action="TURN_LEFT", direction="LEFT", free_path=extreme_fp)
    assert 15.0 <= angle_ext <= 60.0
    print(f"TEST 20 Result: PASS (Turn angle bounded: {angle_ext} <= 60)")

    # ----------------------------------------------------------------
    # TEST 21: Turn state persists
    # ----------------------------------------------------------------
    print("\n--- TEST 21: Turn state persists ---")
    e = DecisionEngine()
    # Frame 1: TURN_LEFT
    e.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle={"position": "CENTER"}, free_path=fp2)
    # Frame 2: stable TURN_LEFT
    e.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle={"position": "CENTER"}, free_path=fp2)
    print(f"Turn state: {e.turn_state}")
    assert e.turn_state == "TURNING_LEFT"
    print("TEST 21 Result: PASS (Turn state persists across consecutive frames -> TURNING_LEFT)")

    # ----------------------------------------------------------------
    # TEST 22: Direction changes during turn
    # ----------------------------------------------------------------
    print("\n--- TEST 22: Direction changes during turn ---")
    # Now scene changes: right becomes safe, left blocked
    # Frame 3 (1st vote for right): held
    e.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle={"position": "CENTER"}, free_path=fp3)
    # Frame 4 (2nd vote for right): confirmed TURN_RIGHT
    res22 = e.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle={"position": "CENTER"}, free_path=fp3)
    print(f"Result 22: action={res22['action']}, turn_state={e.turn_state}")
    assert res22["action"] == "TURN_RIGHT"
    assert e.turn_state == "TURNING_RIGHT"
    print("TEST 22 Result: PASS (Direction change properly updates state -> TURNING_RIGHT)")

    # ----------------------------------------------------------------
    # TEST 23: Emergency during turn
    # ----------------------------------------------------------------
    print("\n--- TEST 23: Emergency during turn ---")
    res23 = e.determine_navigation(safety_level="EMERGENCY", emergency_stop=True, primary_obstacle=None, free_path=fp3)
    print(f"Result 23: action={res23['action']}, stage={res23['navigation_stage']}, turn_state={e.turn_state}")
    assert res23["action"] == "STOP"
    assert res23["navigation_stage"] == "STOP"
    assert e.turn_state == "NONE"
    print("TEST 23 Result: PASS (Emergency during turn immediately resets to STOP)")

    # ----------------------------------------------------------------
    # TEST 24: New scene after turn
    # ----------------------------------------------------------------
    print("\n--- TEST 24: New scene after turn ---")
    e = DecisionEngine()
    e.turn_state = "TURNING_LEFT"
    # New frame arrives: forward corridor is NOT clear, in fact all corridors blocked
    res24 = e.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle=None, free_path=fp4)
    print(f"Result 24: action={res24['action']}, stage={res24['navigation_stage']}")
    assert res24["action"] == "STOP"
    assert res24["navigation_stage"] == "STOP"
    print("TEST 24 Result: PASS (MOVE is NOT blindly issued when new scene is blocked)")

    # ----------------------------------------------------------------
    # TEST 25: Voice mapping
    # ----------------------------------------------------------------
    print("\n--- TEST 25: Voice mapping ---")
    e = DecisionEngine()
    v_left = e.generate_voice_instruction(safety_level="CAUTION", emergency_stop=False, primary_obstacle=None, navigation="TURN_LEFT")
    v_right = e.generate_voice_instruction(safety_level="CAUTION", emergency_stop=False, primary_obstacle=None, navigation="TURN_RIGHT")
    v_fwd = e.generate_voice_instruction(safety_level="SAFE", emergency_stop=False, primary_obstacle=None, navigation="GO_FORWARD")
    v_stop = e.generate_voice_instruction(safety_level="SAFE", emergency_stop=False, primary_obstacle=None, navigation="STOP")
    print(f"Voice LEFT : '{v_left}'")
    print(f"Voice RIGHT: '{v_right}'")
    print(f"Voice FWD  : '{v_fwd}'")
    print(f"Voice STOP : '{v_stop}'")
    assert v_left in ("Turn left.", "Turn left, then move forward.")
    assert v_right in ("Turn right.", "Turn right, then move forward.")
    assert "forward" in v_fwd.lower()
    assert v_stop == "Stop."
    print("TEST 25 Result: PASS (Voice mapping correctly distinguishes turning from walking)")

    # ----------------------------------------------------------------
    # TEST 26: NaN / Inf inputs
    # ----------------------------------------------------------------
    print("\n--- TEST 26: NaN / Inf inputs ---")
    e = DecisionEngine()
    fp_nan = {"left_clear": True, "center_coverage": float("nan"), "center_risk": float("inf")}
    angle_nan, dir_nan, conf_nan = e.estimate_turn_angle(action="TURN_LEFT", direction="LEFT", free_path=fp_nan)
    assert 15.0 <= angle_nan <= 60.0
    assert dir_nan == "LEFT"
    print(f"TEST 26 Result: PASS (NaN/Inf safely handled -> angle={angle_nan}, conf={conf_nan})")

    # ----------------------------------------------------------------
    # TEST 27: Missing / None free_path
    # ----------------------------------------------------------------
    print("\n--- TEST 27: Missing / None free_path ---")
    e = DecisionEngine()
    res27 = e.determine_navigation(safety_level="SAFE", emergency_stop=False, primary_obstacle=None, free_path=None)
    assert res27["action"] == "STOP"
    assert res27["navigation_stage"] == "STOP"
    assert res27["turn_angle_deg"] == 0.0
    print("TEST 27 Result: PASS (None free_path safely handled without crash)")

    # ----------------------------------------------------------------
    # TEST 28: Unknown obstacle participates in turn-vs-movement
    # ----------------------------------------------------------------
    print("\n--- TEST 28: Unknown obstacle participates in turn-vs-movement ---")
    e = DecisionEngine()
    e.last_action = "TURN_RIGHT"
    unk_obs = {"id": 1001, "class_name": "unknown", "position": "CENTER", "risk_score": 0.8}
    res28 = e.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle=unk_obs, free_path=fp3)
    assert res28["action"] == "TURN_RIGHT"
    assert res28["navigation_stage"] == "TURN"
    assert res28["turn_direction"] == "RIGHT"
    assert res28["movement_required"] is False
    print("TEST 28 Result: PASS (Unknown obstacle participates in turn-vs-movement)")

    # ----------------------------------------------------------------
    # TEST 29: High center coverage -> angle 45.0
    # ----------------------------------------------------------------
    print("\n--- TEST 29: High center coverage -> angle 45.0 ---")
    e = DecisionEngine()
    fp_high = {"left_clear": True, "center_coverage": 0.75, "center_risk": 0.85, "path_confidence": "HIGH"}
    angle_high, _, _ = e.estimate_turn_angle(action="TURN_LEFT", direction="LEFT", free_path=fp_high)
    print(f"High coverage angle: {angle_high}")
    assert angle_high >= 45.0
    print("TEST 29 Result: PASS (High center coverage produces stronger avoidance angle >= 45)")

    # ----------------------------------------------------------------
    # TEST 30: Low path confidence -> coarse 30.0 with LOW confidence
    # ----------------------------------------------------------------
    print("\n--- TEST 30: Low path confidence -> coarse 30.0 with LOW confidence ---")
    e = DecisionEngine()
    angle_coarse, _, conf_coarse = e.estimate_turn_angle(action="TURN_LEFT", direction="LEFT", free_path={})
    assert angle_coarse == 30.0
    assert conf_coarse == "LOW"
    print("TEST 30 Result: PASS (Empty geometry -> coarse 30.0 target with LOW confidence)")

    print("\n" + "=" * 70)
    print("ALL 30/30 PHASE 9 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_phase9_verification()
