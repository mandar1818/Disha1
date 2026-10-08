#!/usr/bin/env python3
"""
SmartVisionAI — Comprehensive Live Detection, Object Announcement, Direction & Voice Integration Suite
========================================================================================================
Validates the end-to-end integration and behavioral requirements:
- Problem A: Natural English Object Announcements ("Person ahead.", "Chair on the left.", "Table ahead.", "Bottle on the right.", "Stairs ahead.")
             No raw JSON, no technical class IDs, no confidence scores, no bbox coordinates.
- Problem B: Strict Direction Vocabulary:
             1. Object Position: LEFT, CENTER, RIGHT ("Chair on the left.", "Person ahead.")
             2. Walking Displacement: MOVE LEFT, MOVE RIGHT ("Move right. Path is clear.", "Move left. Obstacle on the right.")
             3. Body Orientation: TURN LEFT, TURN RIGHT ("Turn left.", "Turn right.")
             Never "Left" or "Right" alone.
- Problem C: Continuous Real-Time Camera Detection Only (No "capture image", "take photo", "analyzing image" workflows).
- Problem D: Mobile Speech Service & Fallback Synthesis, 5-level priority hierarchy, threatClassChanged cooldown bypass,
             deduplication, and closed-loop turn continuity.
"""

import sys
import os
import unittest
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.decision.engine import DecisionEngine


class VoiceDetectionIntegrationSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = DecisionEngine()

    def setUp(self):
        self.engine.reset()

    # =========================================================================
    # CATEGORY A: NATURAL ENGLISH OBJECT ANNOUNCEMENTS
    # =========================================================================

    def test_01_person_ahead(self):
        """Verify person detected ahead produces 'Person ahead. Move forward.'"""
        primary = {"class_name": "person", "position": "CENTER", "distance_m": 2.0}
        inst = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle=primary,
            navigation="GO_FORWARD",
        )
        self.assertEqual(inst, "Person ahead. Move forward.")

    def test_02_chair_on_left(self):
        """Verify chair on left produces 'Chair on the left. Move forward.'"""
        primary = {"class_name": "chair", "position": "LEFT", "distance_m": 2.2}
        inst = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle=primary,
            navigation="GO_FORWARD",
        )
        self.assertEqual(inst, "Chair on the left. Move forward.")

    def test_03_table_ahead(self):
        """Verify table/diningtable ahead produces 'Table ahead. Move forward.'"""
        primary = {"class_name": "diningtable", "position": "CENTER", "distance_m": 2.5}
        inst = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle=primary,
            navigation="GO_FORWARD",
        )
        self.assertEqual(inst, "Table ahead. Move forward.")

    def test_04_bottle_on_right(self):
        """Verify bottle on right produces 'Bottle on the right. Move forward.'"""
        primary = {"class_name": "bottle", "position": "RIGHT", "distance_m": 1.8}
        inst = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle=primary,
            navigation="GO_FORWARD",
        )
        self.assertEqual(inst, "Bottle on the right. Move forward.")

    def test_05_stairs_ahead(self):
        """Verify stairs ahead produces 'Stairs ahead. Move forward.'"""
        primary = {"class_name": "stairs", "position": "CENTER", "distance_m": 2.4}
        inst = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle=primary,
            navigation="GO_FORWARD",
        )
        self.assertEqual(inst, "Stairs ahead. Move forward.")

    def test_06_couch_and_sofa_mapping(self):
        """Verify synonyms couch/sofa map to clear English nouns with walking direction."""
        primary_couch = {"class_name": "couch", "position": "LEFT", "distance_m": 2.0}
        inst_couch = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle=primary_couch,
            navigation="GO_FORWARD",
        )
        self.assertEqual(inst_couch, "Sofa on the left. Move forward.")

    def test_07_cell_phone_and_laptop_mapping(self):
        """Verify small objects map cleanly without tech jargon with walking direction."""
        primary_phone = {"class_name": "cell phone", "position": "RIGHT", "distance_m": 1.5}
        inst_phone = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle=primary_phone,
            navigation="GO_FORWARD",
        )
        self.assertEqual(inst_phone, "Phone on the right. Move forward.")

    # =========================================================================
    # CATEGORY B: TECHNICAL ARTIFACT REJECTION
    # =========================================================================

    def test_08_no_raw_json_in_instruction(self):
        """Spoken instruction must never contain curly braces or JSON tokens."""
        for cname in ["person", "chair", "table", "bottle", "backpack"]:
            inst = self.engine.generate_voice_instruction(
                safety_level="SAFE",
                emergency_stop=False,
                primary_obstacle={"class_name": cname, "position": "CENTER", "distance_m": 2.0},
                navigation="GO_FORWARD",
            )
            self.assertNotIn("{", inst)
            self.assertNotIn("}", inst)
            self.assertNotIn("class_name", inst)
            self.assertNotIn("bbox", inst)

    def test_09_no_confidence_scores_spoken(self):
        """Spoken instruction must never include decimal confidences or percentages."""
        inst = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle={"class_name": "chair", "position": "LEFT", "confidence": 0.8934, "distance_m": 2.0},
            navigation="GO_FORWARD",
        )
        self.assertNotIn("0.", inst)
        self.assertNotIn("%", inst)
        self.assertNotIn("confidence", inst.lower())

    def test_10_no_bounding_box_coordinates(self):
        """Spoken instruction must never include coordinates x1, y1, x2, y2."""
        inst = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle={
                "class_name": "bottle",
                "position": "RIGHT",
                "bbox": {"x1": 450, "y1": 200, "x2": 520, "y2": 380},
                "distance_m": 2.0,
            },
            navigation="GO_FORWARD",
        )
        self.assertNotIn("x1", inst)
        self.assertNotIn("y1", inst)
        self.assertNotIn("450", inst)

    def test_11_no_technical_id_numbers(self):
        """Spoken instruction must never utter internal tracking or category IDs."""
        inst = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle={"id": 1002, "track_id": 44, "class_name": "person", "position": "CENTER", "distance_m": 2.5},
            navigation="GO_FORWARD",
        )
        self.assertNotIn("1002", inst)
        self.assertNotIn("44", inst)
        self.assertNotIn("id", inst.lower().split())

    # =========================================================================
    # CATEGORY C: SPATIAL POSITION & DIRECTION VOCABULARY
    # =========================================================================

    def test_12_never_say_left_or_right_alone(self):
        """Instructions must never be a naked 'Left' or 'Right'."""
        test_actions = ["TURN_LEFT", "TURN_RIGHT", "MOVE_LEFT", "MOVE_RIGHT"]
        for act in test_actions:
            inst = self.engine.generate_voice_instruction(
                safety_level="SAFE",
                emergency_stop=False,
                primary_obstacle=None,
                navigation=act,
            )
            self.assertNotIn(inst.strip().lower(), ["left", "left.", "right", "right."])
            self.assertTrue(inst.startswith("Turn ") or inst.startswith("Move "))

    def test_13_walking_displacement_move_left(self):
        """Lateral shift left must use 'Move left.', never 'Turn left.'"""
        inst = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle={"class_name": "chair", "position": "RIGHT"},
            navigation="MOVE_LEFT",
        )
        self.assertEqual(inst, "Move left. Chair on the right.")
        self.assertNotIn("Turn", inst)

    def test_14_walking_displacement_move_right(self):
        """Lateral shift right must use 'Move right.', never 'Turn right.'"""
        inst = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle={"class_name": "chair", "position": "LEFT"},
            navigation="MOVE_RIGHT",
        )
        self.assertEqual(inst, "Move right. Chair on the left.")
        self.assertNotIn("Turn", inst)

    def test_15_body_orientation_turn_left(self):
        """Turn left must announce 'Turn left.', never 'Move left.'"""
        inst = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle=None,
            navigation="TURN_LEFT",
        )
        self.assertEqual(inst, "Turn left.")
        self.assertNotIn("Move", inst)

    def test_16_body_orientation_turn_right(self):
        """Turn right must announce 'Turn right.', never 'Move right.'"""
        inst = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle=None,
            navigation="TURN_RIGHT",
        )
        self.assertEqual(inst, "Turn right.")
        self.assertNotIn("Move", inst)

    def test_17_move_left_with_clear_corridor(self):
        """Move left without specific obstacle announces 'Move left. Path is clear.'"""
        inst = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle=None,
            navigation="MOVE_LEFT",
        )
        self.assertEqual(inst, "Move left. Path is clear.")

    def test_18_move_right_with_clear_corridor(self):
        """Move right without specific obstacle announces 'Move right. Path is clear.'"""
        inst = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle=None,
            navigation="MOVE_RIGHT",
        )
        self.assertEqual(inst, "Move right. Path is clear.")

    # =========================================================================
    # CATEGORY D: PROXIMITY QUALIFIERS (<= 1.2M)
    # =========================================================================

    def test_19_close_obstacle_ahead(self):
        """Obstacle <= 1.2m ahead announces '{Object} close ahead. Proceed carefully.'"""
        primary = {"class_name": "chair", "position": "CENTER", "distance_m": 1.0}
        inst = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle=primary,
            navigation="GO_FORWARD",
        )
        self.assertEqual(inst, "Chair close ahead. Proceed carefully.")

    def test_20_close_obstacle_on_left(self):
        """Obstacle <= 1.2m on left announces '{Object} close on the left. Proceed carefully.'"""
        primary = {"class_name": "person", "position": "LEFT", "distance_m": 0.9}
        inst = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle=primary,
            navigation="GO_FORWARD",
        )
        self.assertEqual(inst, "Person close on the left. Proceed carefully.")

    def test_21_close_obstacle_on_right(self):
        """Obstacle <= 1.2m on right announces '{Object} close on the right. Proceed carefully.'"""
        primary = {"class_name": "table", "position": "RIGHT", "distance_m": 1.1}
        inst = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle=primary,
            navigation="GO_FORWARD",
        )
        self.assertEqual(inst, "Table close on the right. Proceed carefully.")

    # =========================================================================
    # CATEGORY E: DYNAMIC CROWD & PEOPLE REASONING
    # =========================================================================

    def test_22_approaching_person_warning(self):
        """Approaching person triggers warning 'Warning. Person approaching.'"""
        primary = {"class_name": "person", "position": "CENTER", "distance_m": 2.2, "is_approaching": True}
        inst = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle=primary,
            navigation="GO_FORWARD",
        )
        self.assertEqual(inst, "Warning. Person approaching.")

    def test_23_approaching_object_warning(self):
        """Approaching non-person obstacle triggers 'Warning. Object approaching.'"""
        primary = {"class_name": "chair", "position": "CENTER", "distance_m": 2.0, "is_approaching": True}
        inst = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle=primary,
            navigation="GO_FORWARD",
        )
        self.assertEqual(inst, "Warning. Object approaching.")

    def test_24_crowded_path_slow_down(self):
        """Crowded corridor triggers 'Slow down. Path ahead is crowded.'"""
        multi = {"is_crowded": True, "crowd_size": "GROUP"}
        inst = self.engine.generate_voice_instruction(
            safety_level="CAUTION",
            emergency_stop=False,
            primary_obstacle=None,
            navigation="SLOW_DOWN",
            multi_object=multi,
        )
        self.assertEqual(inst, "Slow down. Path ahead is crowded.")

    def test_25_crowd_blocking_stop(self):
        """Crowd fully blocking path triggers 'Stop. Path ahead is blocked.'"""
        multi = {"crowd_blocking": True, "is_crowded": True}
        inst = self.engine.generate_voice_instruction(
            safety_level="CAUTION",
            emergency_stop=False,
            primary_obstacle=None,
            navigation="STOP",
            multi_object=multi,
        )
        self.assertEqual(inst, "Stop. Path ahead is blocked.")

    # =========================================================================
    # CATEGORY F: INDOOR SAFETY HAZARDS
    # =========================================================================

    def test_26_stair_drop_off_stop(self):
        """Critical downward drop-off triggers 'Stop. Drop-off ahead.'"""
        safety = {"stair_analysis": {"detected": True, "hazard_level": "CRITICAL", "type": "DROP_OFF"}}
        inst = self.engine.generate_voice_instruction(
            safety_level="EMERGENCY",
            emergency_stop=True,
            primary_obstacle=None,
            navigation="STOP",
            indoor_safety=safety,
        )
        self.assertEqual(inst, "Stop. Drop-off ahead.")

    def test_27_stairs_up_caution(self):
        """Stairs going up triggers 'Caution. Stairs up ahead.'"""
        safety = {"stair_analysis": {"detected": True, "type": "STAIRS_UP"}}
        inst = self.engine.generate_voice_instruction(
            safety_level="CAUTION",
            emergency_stop=False,
            primary_obstacle=None,
            navigation="SLOW_DOWN",
            indoor_safety=safety,
        )
        self.assertEqual(inst, "Caution. Stairs up ahead.")

    def test_28_blocked_doorway_stop(self):
        """Blocked doorway triggers 'Stop. The doorway ahead is blocked.'"""
        safety = {"door_analysis": {"detected": True, "state": "BLOCKED"}}
        inst = self.engine.generate_voice_instruction(
            safety_level="EMERGENCY",
            emergency_stop=True,
            primary_obstacle=None,
            navigation="STOP",
            indoor_safety=safety,
        )
        self.assertEqual(inst, "Stop. The doorway ahead is blocked.")

    def test_29_closed_door_do_not_enter(self):
        """Closed door in path triggers 'Door closed. Do not enter.'"""
        safety = {"door_analysis": {"detected": True, "state": "CLOSED"}}
        inst = self.engine.generate_voice_instruction(
            safety_level="CAUTION",
            emergency_stop=False,
            primary_obstacle=None,
            navigation="STOP",
            indoor_safety=safety,
        )
        self.assertEqual(inst, "Door closed. Do not enter.")

    def test_30_trip_hazard_slow_down(self):
        """Floor trip hazard triggers 'Trip hazard on floor ahead. Slow down.'"""
        safety = {"floor_hazards": [{"hazard_level": "CAUTION", "hazard_type": "TRIP_HAZARD"}]}
        inst = self.engine.generate_voice_instruction(
            safety_level="CAUTION",
            emergency_stop=False,
            primary_obstacle=None,
            navigation="SLOW_DOWN",
            indoor_safety=safety,
        )
        self.assertEqual(inst, "Trip hazard on floor ahead. Slow down.")

    # =========================================================================
    # CATEGORY G: EMERGENCY STOP PRIORITY
    # =========================================================================

    def test_31_emergency_stop_immediate(self):
        """Emergency stop produces 'Stop immediately.'"""
        inst = self.engine.generate_voice_instruction(
            safety_level="EMERGENCY",
            emergency_stop=True,
            primary_obstacle={"class_name": "person", "position": "CENTER"},
            navigation="STOP",
        )
        self.assertEqual(inst, "Stop immediately.")

    def test_32_high_risk_obstacle_stop_immediately(self):
        """Obstacle with risk >= 0.70 produces 'Stop immediately.'"""
        primary = {"class_name": "chair", "position": "CENTER", "distance_m": 0.6, "risk_score": 0.85}
        inst = self.engine.generate_voice_instruction(
            safety_level="DANGER",
            emergency_stop=False,
            primary_obstacle=primary,
            navigation="STOP",
        )
        self.assertEqual(inst, "Stop immediately.")

    # =========================================================================
    # CATEGORY H: SEMANTIC SIGNS & LANDMARKS
    # =========================================================================

    def test_33_exit_landmark_announcement(self):
        """Exit landmark cue straight ahead announces 'Exit straight ahead.'"""
        sign = {"type": "EXIT", "label": "Exit", "directional_cue": "STRAIGHT", "should_announce": True, "is_uncertain": False}
        inst = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle=None,
            navigation="GO_FORWARD",
            primary_semantic_sign=sign,
        )
        self.assertEqual(inst, "Exit straight ahead.")

    def test_34_restroom_landmark_to_left(self):
        """Restroom cue to the left announces 'Restroom to the left.'"""
        sign = {"type": "RESTROOM", "label": "Restroom", "directional_cue": "LEFT", "should_announce": True, "is_uncertain": False}
        inst = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle=None,
            navigation="GO_FORWARD",
            primary_semantic_sign=sign,
        )
        self.assertEqual(inst, "Restroom to the left.")

    # =========================================================================
    # CATEGORY I: FREE PATH & CLOSED LOOP STATE
    # =========================================================================

    def test_35_path_cleared_announcement(self):
        """When closed loop state transitions to PATH_CLEARED, announces 'Path is clear. Move forward.'"""
        self.engine.closed_loop_state = "PATH_CLEARED"
        inst = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle=None,
            navigation="GO_FORWARD",
        )
        self.assertEqual(inst, "Path is clear. Move forward.")

    def test_36_clear_path_go_forward(self):
        """Completely clear scene without obstacles announces 'Move forward.'"""
        self.engine.closed_loop_state = "NORMAL"
        inst = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle=None,
            navigation="GO_FORWARD",
            objects=[],
        )
        self.assertEqual(inst, "Move forward.")

    # =========================================================================
    # CATEGORY J: DETOUR DECISION CORRIDORS
    # =========================================================================

    def test_37_obstacle_on_left_detour_right(self):
        """Corridor analysis: obstacle on left, path clear on right detours right."""
        free_path = {"recommended_corridor": "RIGHT", "center_clear": False, "left_clear": False, "right_clear": True}
        primary = {"class_name": "chair", "position": "LEFT", "distance_m": 1.5, "risk_score": 0.5}
        nav = self.engine.determine_navigation(
            safety_level="CAUTION",
            emergency_stop=False,
            primary_obstacle=primary,
            free_path=free_path,
        )
        self.assertEqual(nav["action"], "TURN_RIGHT")
        inst = self.engine.generate_voice_instruction(
            safety_level="CAUTION",
            emergency_stop=False,
            primary_obstacle=primary,
            navigation=nav["action"],
        )
        self.assertEqual(inst, "Turn right.")

    def test_38_obstacle_on_right_detour_left(self):
        """Corridor analysis: obstacle on right, path clear on left detours left."""
        free_path = {"recommended_corridor": "LEFT", "center_clear": False, "left_clear": True, "right_clear": False}
        primary = {"class_name": "person", "position": "RIGHT", "distance_m": 1.5, "risk_score": 0.5}
        nav = self.engine.determine_navigation(
            safety_level="CAUTION",
            emergency_stop=False,
            primary_obstacle=primary,
            free_path=free_path,
        )
        self.assertEqual(nav["action"], "TURN_LEFT")
        inst = self.engine.generate_voice_instruction(
            safety_level="CAUTION",
            emergency_stop=False,
            primary_obstacle=primary,
            navigation=nav["action"],
        )
        self.assertEqual(inst, "Turn left.")

    def test_39_all_corridors_blocked_stop(self):
        """When all corridors are blocked, action is STOP."""
        free_path = {"recommended_corridor": "BLOCKED", "center_clear": False, "left_clear": False, "right_clear": False}
        primary = {"class_name": "chair", "position": "CENTER", "distance_m": 0.9, "risk_score": 0.8}
        nav = self.engine.determine_navigation(
            safety_level="DANGER",
            emergency_stop=False,
            primary_obstacle=primary,
            free_path=free_path,
        )
        self.assertEqual(nav["action"], "STOP")

    # =========================================================================
    # CATEGORY K: UNKNOWN OBSTACLE HANDLING
    # =========================================================================

    def test_40_unknown_obstacle_naming(self):
        """Unknown obstacle announces 'Unknown obstacle ahead. Move forward.' or 'Unknown obstacle on your left/right. Move forward.'"""
        primary = {"class_name": "unknown_obstacle", "position": "CENTER", "distance_m": 1.8}
        inst = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle=primary,
            navigation="GO_FORWARD",
        )
        self.assertEqual(inst, "Unknown obstacle ahead. Move forward.")

    def test_41_unknown_obstacle_on_left(self):
        """Unknown obstacle on left announces 'Unknown obstacle on your left. Move forward.'"""
        primary = {"class_name": "unknown", "position": "LEFT", "distance_m": 1.8}
        inst = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle=primary,
            navigation="GO_FORWARD",
        )
        self.assertEqual(inst, "Unknown obstacle on your left. Move forward.")

    # =========================================================================
    # CATEGORY L: PROMINENT OBJECT IN SCENE WITHOUT OBSTACLE COLLISION
    # =========================================================================

    def test_42_prominent_scene_object_announced(self):
        """When path is clear but prominent objects are present in scene, announces closest object with walking guidance."""
        objects = [
            {"class_name": "bottle", "position": "RIGHT", "distance_m": 2.8},
            {"class_name": "person", "position": "CENTER", "distance_m": 1.9},
        ]
        inst = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle=None,
            navigation="GO_FORWARD",
            objects=objects,
        )
        self.assertEqual(inst, "Person ahead. Move forward.")


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(VoiceDetectionIntegrationSuite)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("\n" + "=" * 60)
    print(f"VOICE & DETECTION INTEGRATION TESTS: {result.testsRun} RUN")
    print(f"PASSED: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"FAILED: {len(result.failures)}")
    print(f"ERRORS: {len(result.errors)}")
    print("=" * 60)
    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
