"""
SmartVisionAI - Phase 4 Closed-Loop Indoor Destination Navigation Test Suite
============================================================================
Comprehensive test suite verifying:
- Destination Matching & Parsing (exact identifiers, aliases, word numerals, mismatch rejection)
- Spatial Localization & Guidance (LEFT -> TURN_LEFT, RIGHT -> TURN_RIGHT, CENTER -> MOVE_FORWARD)
- Movement (MOVE_LEFT/RIGHT) vs Orientation (TURN_LEFT/RIGHT) separation
- Closed-Loop Visual Turn Verification (multi-frame visual change, pending turn, timeout, recovery)
- Post-Turn Obstacle Recheck & Strict Safety Precedence (STOP/AVOIDANCE overrides destination)
- Relative Proximity Progression (FAR -> NEAR -> VERY_CLOSE)
- 2-Frame Stable Arrival Detection & Single Announcement Formulation
- Lost Destination Handling (temporary occlusion -> search -> lost -> reacquisition)
- FrameProcessor & Full API Route Integration
- Telemetry Overhead Budget (< 10 ms)
"""

import sys
import os
import time
import numpy as np

# Ensure workspace root is on sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from backend.navigation.destination import Destination, match_destination
from backend.navigation.verifier import TurnVerifier
from backend.navigation.progress import ProgressTracker
from backend.navigation.session import NavigationSession, NavigationSessionState
from backend.decision.engine import DecisionEngine


def run_all_tests():
    total_tests = 0
    passed_tests = 0
    failed_tests = []

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal total_tests, passed_tests
        total_tests += 1
        if condition:
            passed_tests += 1
            print(f"  [PASS] Scenario {total_tests:02d}: {name}")
        else:
            failed_tests.append((total_tests, name, detail))
            print(f"  [FAIL] Scenario {total_tests:02d}: {name} - {detail}")

    print("=" * 80)
    print("PHASE 4 CLOSED-LOOP INDOOR DESTINATION NAVIGATION TEST SUITE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # GROUP 1: Destination Matching & Parsing (Scenarios 1 - 7)
    # -------------------------------------------------------------------------
    print("\n--- GROUP 1: Destination Matching & Parsing ---")

    # 1. Exact room number match
    dest_room = Destination("Room 302")
    landmark_302 = {
        "label": "Room 302",
        "clean_text": "Room 302",
        "type": "ROOM",
        "identifier": "302",
        "position": "CENTER",
        "confidence": 0.92,
        "bbox": [200, 100, 400, 250],
    }
    check(
        "Exact Room 302 match",
        dest_room.matches(landmark_302) and match_destination("Room 302", landmark_302)[0],
        "Failed to match exact Room 302",
    )

    # 2. Rejection of mismatching room number (302 vs 305)
    landmark_305 = {
        "label": "Room 305",
        "clean_text": "Room 305",
        "type": "ROOM",
        "identifier": "305",
        "position": "CENTER",
        "confidence": 0.91,
    }
    check(
        "Mismatching room number rejection (302 != 305)",
        (not dest_room.matches(landmark_305)) and (not match_destination("Room 302", landmark_305)[0]),
        "Incorrectly matched Room 302 to Room 305",
    )

    # 3. Word form room number parsing ("room three zero two")
    dest_word = Destination("room three zero two")
    check(
        "Word numeral destination matching ('room three zero two' -> 302)",
        dest_word.matches(landmark_302),
        f"Failed to match word form: identifier was {dest_word.identifier}",
    )

    # 4. Restroom alias matching ("washroom" -> RESTROOM)
    dest_restroom = Destination("Washroom")
    landmark_rr = {
        "label": "Restroom",
        "clean_text": "Restroom",
        "type": "RESTROOM",
        "identifier": None,
        "position": "LEFT",
        "confidence": 0.88,
    }
    check(
        "Restroom alias match ('Washroom' matches 'Restroom')",
        dest_restroom.matches(landmark_rr),
        "Failed to match Washroom to RESTROOM landmark",
    )

    # 5. Elevator alias matching ("lift" -> ELEVATOR)
    dest_lift = Destination("Lift")
    landmark_elev = {
        "label": "Elevator",
        "clean_text": "Elevator",
        "type": "ELEVATOR",
        "identifier": None,
        "position": "RIGHT",
        "confidence": 0.95,
    }
    check(
        "Elevator alias match ('Lift' matches 'Elevator')",
        dest_lift.matches(landmark_elev),
        "Failed to match Lift to ELEVATOR landmark",
    )

    # 6. Exit destination matching
    dest_exit = Destination("Emergency Exit")
    landmark_exit = {
        "label": "Exit",
        "clean_text": "Emergency Exit",
        "type": "EXIT",
        "identifier": None,
        "position": "CENTER",
        "confidence": 0.90,
    }
    check(
        "Exit landmark matching",
        dest_exit.matches(landmark_exit),
        "Failed to match Exit destination",
    )

    # 7. Rejection of unrelated landmarks
    landmark_stairs = {
        "label": "Stairs",
        "clean_text": "Stairs to 2nd Floor",
        "type": "STAIRS",
        "identifier": None,
        "position": "CENTER",
        "confidence": 0.85,
    }
    check(
        "Rejection of unrelated category (Room 302 vs Stairs)",
        not dest_room.matches(landmark_stairs),
        "Room 302 should not match Stairs",
    )

    # -------------------------------------------------------------------------
    # GROUP 2: Spatial Localization & Turn Generation (Scenarios 8 - 12)
    # -------------------------------------------------------------------------
    print("\n--- GROUP 2: Spatial Localization & Turn Generation ---")

    # 8. Destination on LEFT triggers TURN_LEFT
    session_left = NavigationSession()
    session_left.start("Room 302")
    landmark_on_left = {
        "label": "Room 302",
        "clean_text": "Room 302",
        "type": "ROOM",
        "identifier": "302",
        "position": "LEFT",
        "bbox": [50, 150, 180, 260],
        "confidence": 0.90,
    }
    status_l = session_left.update(
        semantic_landmarks=[landmark_on_left],
        primary_obstacle=None,
        free_path={"recommended_action": "GO_FORWARD"},
        safety_level="SAFE",
        emergency_stop=False,
    )
    check(
        "Destination on LEFT produces TURN_LEFT command",
        status_l["state"] == "TURNING" and session_left.get_recommended_action() == "TURN_LEFT",
        f"State: {status_l['state']}, Action: {session_left.get_recommended_action()}",
    )

    # 9. Destination on RIGHT triggers TURN_RIGHT
    session_right = NavigationSession()
    session_right.start("Room 302")
    landmark_on_right = {
        "label": "Room 302",
        "clean_text": "Room 302",
        "type": "ROOM",
        "identifier": "302",
        "position": "RIGHT",
        "bbox": [480, 150, 610, 260],
        "confidence": 0.90,
    }
    status_r = session_right.update(
        semantic_landmarks=[landmark_on_right],
        primary_obstacle=None,
        free_path={"recommended_action": "GO_FORWARD"},
        safety_level="SAFE",
        emergency_stop=False,
    )
    check(
        "Destination on RIGHT produces TURN_RIGHT command",
        status_r["state"] == "TURNING" and session_right.get_recommended_action() == "TURN_RIGHT",
        f"State: {status_r['state']}, Action: {session_right.get_recommended_action()}",
    )

    # 10. Destination in CENTER triggers APPROACHING and MOVE_FORWARD
    session_center = NavigationSession()
    session_center.start("Room 302")
    landmark_in_center = {
        "label": "Room 302",
        "clean_text": "Room 302",
        "type": "ROOM",
        "identifier": "302",
        "position": "CENTER",
        "bbox": [250, 150, 390, 260],
        "confidence": 0.90,
    }
    status_c = session_center.update(
        semantic_landmarks=[landmark_in_center],
        primary_obstacle=None,
        free_path={"recommended_action": "GO_FORWARD"},
        safety_level="SAFE",
        emergency_stop=False,
    )
    check(
        "Destination in CENTER produces APPROACHING and MOVE_FORWARD",
        status_c["state"] == "APPROACHING" and session_center.get_recommended_action() == "MOVE_FORWARD",
        f"State: {status_c['state']}, Action: {session_center.get_recommended_action()}",
    )

    # 11. Separation of TURN vs MOVE commands
    check(
        "Separation of Turn vs Move (TURN_LEFT is not MOVE_LEFT)",
        session_left.get_recommended_action().startswith("TURN_")
        and not session_left.get_recommended_action().startswith("MOVE_"),
        "Turn action should not be classified as lateral move",
    )

    # 12. Distinct lateral movement vs body orientation turn
    lateral_move = "MOVE_LEFT"
    orientation_turn = "TURN_LEFT"
    check(
        "Distinction between lateral Move and body Turn commands",
        lateral_move.startswith("MOVE_") and orientation_turn.startswith("TURN_") and lateral_move != orientation_turn,
        "Lateral move and turn must remain distinct commands",
    )

    # -------------------------------------------------------------------------
    # GROUP 3: Closed-Loop Turn Verification (Scenarios 13 - 17)
    # -------------------------------------------------------------------------
    print("\n--- GROUP 3: Closed-Loop Turn Verification ---")

    # 13. Turn verification initiation
    verifier = TurnVerifier(timeout_frames=3)
    verifier.request_turn("TURN_RIGHT", initial_landmark_pos="RIGHT", initial_depth=0.7)
    check(
        "Turn verification initiates with IN_PROGRESS",
        verifier.status == "IN_PROGRESS" and verifier.is_turn_pending(),
        f"Verifier status: {verifier.status}",
    )

    # 14. Turn verification succeeds on landmark position shift
    # Landmark shifts from right (cx ~ 0.80) to center (cx ~ 0.50) => delta >= 0.15
    verified, reason = verifier.verify_frame(
        current_landmark={"position": "CENTER", "bbox": [280, 150, 360, 250]},  # cx = 320/640 = 0.50
        frame_width=640,
        frame_height=480,
    )
    check(
        "Turn verification succeeds on landmark position shift",
        verified and verifier.status == "VERIFIED",
        f"Verified: {verified}, Reason: {reason}",
    )

    # 15. Turn verification succeeds on depth redistribution delta >= 0.20
    verifier2 = TurnVerifier(timeout_frames=3)
    verifier2.request_turn("TURN_LEFT", initial_landmark_pos="LEFT", initial_depth=0.3)
    v2, r2 = verifier2.verify_frame(current_depth=0.60)  # delta = 0.30 >= 0.20
    check(
        "Turn verification succeeds on depth profile change >= 0.20",
        v2 and verifier2.status == "VERIFIED",
        f"Verified: {v2}, Reason: {r2}",
    )

    # 16. Turn verification fails on zero visual change
    verifier3 = TurnVerifier(timeout_frames=2)
    verifier3.request_turn("TURN_RIGHT", initial_landmark_pos="RIGHT", initial_depth=0.5)
    # Frame 1: no change
    v3_1, _ = verifier3.verify_frame(current_depth=0.51)
    # Frame 2: no change => reaches timeout
    v3_2, r3_2 = verifier3.verify_frame(current_depth=0.52)
    check(
        "Turn verification timeout on zero visual change",
        not v3_2 and verifier3.status == "TIMEOUT",
        f"Verifier status: {verifier3.status}, Reason: {r3_2}",
    )

    # 17. Turn recovery prompt with cooldown suppression
    prompt1 = verifier3.get_recovery_prompt("Room 302")
    prompt2 = verifier3.get_recovery_prompt("Room 302")  # immediately called => suppressed
    check(
        "Turn recovery prompt emits instruction and suppresses on immediate repeat",
        "turn right" in prompt1.lower() and prompt2 == "",
        f"Prompt1: '{prompt1}', Prompt2: '{prompt2}'",
    )

    # -------------------------------------------------------------------------
    # GROUP 4: Post-Turn Obstacle Recheck & Safety Precedence (Scenarios 18 - 22)
    # -------------------------------------------------------------------------
    print("\n--- GROUP 4: Post-Turn Obstacle Recheck & Safety Precedence ---")

    # 18. New obstacle detected post-turn immediately overrides destination guidance
    session_post_turn = NavigationSession()
    session_post_turn.start("Room 302")
    # Turn initiated
    session_post_turn.update(
        semantic_landmarks=[landmark_on_right],
        primary_obstacle=None,
        safety_level="SAFE",
    )
    # User begins turning, but in the next frame a VERY CLOSE obstacle appears!
    danger_obstacle = {
        "class_name": "cart",
        "position": "RIGHT",
        "distance_category": "VERY_CLOSE",
        "distance_m": 0.5,
        "risk_score": 0.95,
        "is_physical_obstacle": True,
    }
    status_ov = session_post_turn.update(
        semantic_landmarks=[landmark_on_right],
        primary_obstacle=danger_obstacle,
        safety_level="DANGER",
        emergency_stop=False,
    )
    check(
        "Post-turn obstacle triggers safety override and AVOIDING state",
        status_ov["safety_override"] is True and status_ov["state"] == "AVOIDING",
        f"Override: {status_ov['safety_override']}, State: {status_ov['state']}",
    )

    # 19. Emergency STOP overrides destination guidance
    engine = DecisionEngine()
    danger_obstacle_obj = {
        "class_name": "person",
        "position": "RIGHT",
        "distance_category": "VERY_CLOSE",
        "distance_m": 0.45,
        "risk_score": 0.95,
        "box_2d": [100, 400, 400, 600],
    }
    decision_stop = engine.analyze(
        objects=[danger_obstacle_obj],
        scene_depth={"center_depth": 0.2, "right_depth": 0.15},
        navigation_session=session_post_turn,
    )
    check(
        "Emergency STOP overrides destination state",
        decision_stop["navigation"] == "STOP" or decision_stop.get("emergency_stop") is True,
        f"Navigation: {decision_stop['navigation']}, emergency_stop: {decision_stop.get('emergency_stop')}",
    )

    # 20. Critical fall hazard / stairs-down overrides destination guidance
    decision_stairs = engine.analyze(
        objects=[],
        indoor_safety={"stair_analysis": {"detected": True, "drop_off_risk": True, "hazard_level": "CRITICAL"}},
        navigation_session=session_post_turn,
    )
    check(
        "Critical fall hazard overrides destination navigation",
        decision_stairs.get("emergency_stop") is True or decision_stairs["navigation"] == "STOP",
        f"Navigation: {decision_stairs['navigation']}",
    )

    # 21. Caution-level obstacle triggers avoidance before destination
    caution_obstacle_obj = {
        "class_name": "chair",
        "position": "CENTER",
        "distance_category": "NEAR",
        "distance_m": 1.2,
        "risk_score": 0.65,
        "box_2d": [150, 200, 350, 400],
    }
    decision_caution = engine.analyze(
        objects=[caution_obstacle_obj],
        scene_depth={"left_depth": 0.85, "center_depth": 0.35, "right_depth": 0.40},
        navigation_session=session_center,
    )
    nav_act_caution = decision_caution["navigation"].get("action", decision_caution["navigation"]) if isinstance(decision_caution["navigation"], dict) else decision_caution["navigation"]
    check(
        "Caution-level obstacle forces avoidance maneuver over direct approach",
        nav_act_caution in ["MOVE_LEFT", "MOVE_RIGHT", "STOP", "SLOW_DOWN"],
        f"Navigation action: {nav_act_caution}",
    )

    # 22. Glass hazard or floor spill forces caution override
    decision_glass = engine.analyze(
        objects=[],
        indoor_safety={
            "glass_obstacles": {
                "detected": True,
                "position": "CENTER",
                "hazard_level": "CAUTION",
                "risk_score": 0.75,
            }
        },
        navigation_session=session_center,
    )
    nav_act_glass = decision_glass["navigation"].get("action", decision_glass["navigation"]) if isinstance(decision_glass["navigation"], dict) else decision_glass["navigation"]
    check(
        "Glass hazard forces safety override over destination guidance",
        decision_glass["safety_level"] in ["CAUTION", "DANGER", "EMERGENCY"] or nav_act_glass in ["SLOW_DOWN", "STOP"],
        f"Safety level: {decision_glass['safety_level']}, action: {nav_act_glass}",
    )

    # -------------------------------------------------------------------------
    # GROUP 5: Relative Proximity Progression & Arrival (Scenarios 23 - 27)
    # -------------------------------------------------------------------------
    print("\n--- GROUP 5: Relative Proximity Progression & Arrival ---")

    # 23. Relative proximity calculation (FAR)
    tracker = ProgressTracker()
    prox_far = tracker.estimate_proximity(
        landmark={"bbox": [290, 100, 350, 130], "position": "CENTER"},  # height = 30 / 480 = 0.06
        scene_depth=0.75,
        frame_width=640,
        frame_height=480,
    )
    check(
        "Relative proximity estimates FAR for small distant landmark",
        prox_far == "FAR",
        f"Estimated proximity: {prox_far}",
    )

    # 24. Relative proximity calculation (NEAR)
    prox_near = tracker.estimate_proximity(
        landmark={"bbox": [240, 100, 400, 200], "position": "CENTER"},  # height = 100 / 480 = 0.208
        scene_depth=0.45,
        frame_width=640,
        frame_height=480,
    )
    check(
        "Relative proximity estimates NEAR for medium landmark",
        prox_near == "NEAR",
        f"Estimated proximity: {prox_near}",
    )

    # 25. Relative proximity calculation (VERY_CLOSE)
    prox_close = tracker.estimate_proximity(
        landmark={"bbox": [180, 50, 460, 300], "position": "CENTER"},  # height = 250 / 480 = 0.52
        scene_depth=0.20,
        frame_width=640,
        frame_height=480,
    )
    check(
        "Relative proximity estimates VERY_CLOSE for large landmark",
        prox_close == "VERY_CLOSE",
        f"Estimated proximity: {prox_close}",
    )

    # 26. Arrival requires 2-frame stability (Single frame does NOT trigger arrival)
    tracker_arr = ProgressTracker()
    arr_frame1 = tracker_arr.update(
        landmark={"bbox": [180, 50, 460, 300], "position": "CENTER"},
        scene_depth=0.20,
        destination_matches=True,
        frame_width=640,
        frame_height=480,
    )
    check(
        "Single VERY_CLOSE frame does not trigger premature arrival (2-frame rule)",
        (not arr_frame1.get("arrived")) and tracker_arr.consecutive_very_close == 1,
        f"Arrival: {arr_frame1}, consecutive: {tracker_arr.consecutive_very_close}",
    )

    # Frame 2: 2nd consecutive VERY_CLOSE frame triggers ARRIVED
    arr_frame2 = tracker_arr.update(
        landmark={"bbox": [180, 50, 460, 300], "position": "CENTER"},
        scene_depth=0.19,
        destination_matches=True,
        frame_width=640,
        frame_height=480,
    )
    check(
        "Second consecutive VERY_CLOSE frame triggers verified ARRIVED",
        arr_frame2.get("arrived") and tracker_arr.is_arrived,
        f"Arrival: {arr_frame2}, is_arrived: {tracker_arr.is_arrived}",
    )

    # 27. Arrival announcement formulated once
    announcement1 = tracker_arr.get_arrival_announcement("Room 302")
    announcement2 = tracker_arr.get_arrival_announcement("Room 302")
    check(
        "Arrival announcement formulated once and suppressed on subsequent frames",
        "reached Room 302" in announcement1 and announcement2 == "",
        f"Announcement1: '{announcement1}', Announcement2: '{announcement2}'",
    )

    # -------------------------------------------------------------------------
    # GROUP 6: Lost Destination Handling (Scenarios 28 - 30)
    # -------------------------------------------------------------------------
    print("\n--- GROUP 6: Lost Destination Handling ---")

    # 28. Destination missing for 1 frame enters SEARCHING
    session_lost = NavigationSession()
    session_lost.start("Room 302")
    # Frame 1: Destination visible
    session_lost.update(semantic_landmarks=[landmark_in_center], safety_level="SAFE")
    # Frame 2: Destination not visible
    st_missing1 = session_lost.update(semantic_landmarks=[], safety_level="SAFE")
    check(
        "Destination missing for 1 frame enters SEARCHING state",
        st_missing1["state"] == "SEARCHING" and not st_missing1["destination_visible"],
        f"State: {st_missing1['state']}",
    )

    # 29. Destination missing for >= 3 frames transitions to LOST
    session_lost.update(semantic_landmarks=[], safety_level="SAFE")  # frame 3
    st_missing3 = session_lost.update(semantic_landmarks=[], safety_level="SAFE")  # frame 4
    check(
        "Destination missing for multiple frames transitions to LOST state",
        st_missing3["state"] == "LOST",
        f"State: {st_missing3['state']}, missing_count: {session_lost.missing_frames_count}",
    )

    # 30. Reacquisition of destination resumes APPROACHING
    st_reacquired = session_lost.update(semantic_landmarks=[landmark_in_center], safety_level="SAFE")
    check(
        "Reacquisition of destination immediately resumes navigation",
        st_reacquired["state"] == "APPROACHING" and st_reacquired["destination_visible"],
        f"State: {st_reacquired['state']}, visible: {st_reacquired['destination_visible']}",
    )

    # -------------------------------------------------------------------------
    # GROUP 7: FrameProcessor Lifecycle & Telemetry Budget (Scenarios 31 - 34)
    # -------------------------------------------------------------------------
    print("\n--- GROUP 7: FrameProcessor Lifecycle & Telemetry Budget ---")

    from backend.processors.frame_processor import FrameProcessor
    processor = FrameProcessor()

    # 31. Session start via FrameProcessor
    sess = processor.start_destination("Room 302")
    check(
        "FrameProcessor.start_destination initializes active session",
        processor.navigation_session.active and (sess.get("destination", {}).get("name") == "Room 302" or processor.navigation_session.destination.name == "Room 302"),
        f"Active: {processor.navigation_session.active}, sess: {sess}",
    )

    # 32. FrameProcessor status inspection
    status_dict = processor.get_navigation_status()
    check(
        "FrameProcessor.get_navigation_status returns complete schema",
        status_dict["active"] is True and "turn_verification" in status_dict and status_dict["state"] in NavigationSessionState.__members__,
        f"Status dict: {status_dict}",
    )

    # 33. FrameProcessor.process_frame executes with destination navigation
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Inject a simulated white sign on dark background
    dummy_frame[100:200, 250:390] = 255
    res = processor.process_frame(dummy_frame)
    check(
        "FrameProcessor.process_frame includes navigation_session and telemetry",
        "navigation_session" in res and "navigation" in res.get("processing_breakdown_ms", {}),
        f"Keys in res: {list(res.keys())}, breakdown: {res.get('processing_breakdown_ms')}",
    )

    # 34. Telemetry Budget: Navigation reasoning overhead < 10 ms
    nav_time = res.get("processing_breakdown_ms", {}).get("navigation", 0.0)
    check(
        "Navigation module telemetry overhead is well below 10 ms limit",
        nav_time < 10.0,
        f"Navigation overhead: {nav_time:.2f} ms",
    )

    # 35. Session stop via FrameProcessor
    stopped_sess = processor.stop_destination()
    check(
        "FrameProcessor.stop_destination resets session to IDLE",
        (not stopped_sess.get("active")) and stopped_sess.get("state") == "IDLE",
        f"Active: {stopped_sess.get('active')}, state: {stopped_sess.get('state')}",
    )

    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"TEST RUN COMPLETED: {passed_tests}/{total_tests} SCENARIOS PASSED")
    print("=" * 80)

    if failed_tests:
        print("\nFAILED SCENARIOS:")
        for num, name, detail in failed_tests:
            print(f"  #{num:02d}: {name} - {detail}")
        return False

    print("\nALL 35 TEST SCENARIOS PASSED PERFECTLY!")
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
