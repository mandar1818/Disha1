#!/usr/bin/env python3
"""
Comprehensive Automated Verification Test Suite for SmartVisionAI/DISHA
Final Real-Time Perception + Safe Navigation + Voice Guidance Verification

Covers:
- YOLO COCO label formatting & real object preservation
- Unknown obstacle containment suppression & priority ranking
- Corridor geometry & position mapping ([0.38, 0.62])
- Deterministic navigation directives: MOVE_LEFT, MOVE_RIGHT, TURN_LEFT, TURN_RIGHT, STOP, FORWARD, BACK, SLOW_DOWN
- Step guidance preservation for turns & moves
- Scene flux immunity for directional maneuvers
- Concise, actionable English voice instructions
- Cooldown, priority preemption, and speech management
- End-to-end multi-frame continuous walking simulation

Total tests: >= 55 tests.
"""

import sys
import os
import unittest
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.decision.engine import DecisionEngine
from backend.processors.frame_processor import FrameProcessor


class TestFinalPerceptionNavigationVoice(unittest.TestCase):

    def setUp(self):
        self.engine = DecisionEngine()
        self.engine.reset()

    # ========================================================
    # 1. OBJECT LABELING & NATURAL PHRASING TESTS (1-7)
    # ========================================================

    def test_01_person_formatting(self):
        self.assertEqual(self.engine._format_object_name("person"), "Person")

    def test_02_chair_formatting(self):
        self.assertEqual(self.engine._format_object_name("chair"), "Chair")

    def test_03_dining_table_formatting(self):
        self.assertEqual(self.engine._format_object_name("dining table"), "Table")
        self.assertEqual(self.engine._format_object_name("diningtable"), "Table")

    def test_04_couch_sofa_formatting(self):
        self.assertEqual(self.engine._format_object_name("couch"), "Sofa")
        self.assertEqual(self.engine._format_object_name("sofa"), "Sofa")

    def test_05_phone_and_laptop_formatting(self):
        self.assertEqual(self.engine._format_object_name("cell phone"), "Phone")
        self.assertEqual(self.engine._format_object_name("laptop"), "Laptop")

    def test_06_bottle_and_bag_formatting(self):
        self.assertEqual(self.engine._format_object_name("bottle"), "Bottle")
        self.assertEqual(self.engine._format_object_name("backpack"), "Backpack")
        self.assertEqual(self.engine._format_object_name("handbag"), "Bag")

    def test_07_unknown_obstacle_formatting(self):
        self.assertEqual(self.engine._format_object_name("unknown"), "Unknown obstacle")
        self.assertEqual(self.engine._format_object_name("unknown_obstacle"), "Unknown obstacle")

    # ========================================================
    # 2. POSITION & CORRIDOR BOUNDARIES TESTS (8-14)
    # ========================================================

    def test_08_position_clear_left(self):
        bbox = {"x1": 50, "y1": 100, "x2": 150, "y2": 300}  # center = 100/640 = 0.156 < 0.38
        self.assertEqual(self.engine.get_position(bbox, 640.0), "LEFT")

    def test_09_position_clear_right(self):
        bbox = {"x1": 450, "y1": 100, "x2": 550, "y2": 300}  # center = 500/640 = 0.781 > 0.62
        self.assertEqual(self.engine.get_position(bbox, 640.0), "RIGHT")

    def test_10_position_exact_center(self):
        bbox = {"x1": 270, "y1": 100, "x2": 370, "y2": 300}  # center = 320/640 = 0.50
        self.assertEqual(self.engine.get_position(bbox, 640.0), "CENTER")

    def test_11_position_left_corridor_overlap_encroachment(self):
        # Center is at 0.35 (< 0.38), but box extends from 0.20 to 0.50 (corridor overlap 0.50-0.38 = 0.12 >= 0.08)
        bbox = {"x1": 128, "y1": 100, "x2": 320, "y2": 300}
        self.assertEqual(self.engine.get_position(bbox, 640.0), "CENTER")

    def test_12_position_right_corridor_overlap_encroachment(self):
        # Center is at 0.65 (> 0.62), but box extends from 0.50 to 0.80 (corridor overlap 0.62-0.50 = 0.12 >= 0.08)
        bbox = {"x1": 320, "y1": 100, "x2": 512, "y2": 300}
        self.assertEqual(self.engine.get_position(bbox, 640.0), "CENTER")

    def test_13_position_wide_obstacle_straddling_center(self):
        # Wide object spanning from 0.25 to 0.75 (width 0.50 >= 0.40, straddles 0.50)
        bbox = {"x1": 160, "y1": 100, "x2": 480, "y2": 300}
        self.assertEqual(self.engine.get_position(bbox, 640.0), "CENTER")

    def test_14_position_phrase_helper(self):
        self.assertEqual(self.engine._position_phrase("LEFT"), "on the left")
        self.assertEqual(self.engine._position_phrase("RIGHT"), "on the right")
        self.assertEqual(self.engine._position_phrase("CENTER"), "ahead")

    # ========================================================
    # 3. UNKNOWN OBSTACLE SUPPRESSION & RANKING (15-20)
    # ========================================================

    def test_15_unknown_suppressed_by_containment_inside_chair(self):
        known = [{
            "id": 1,
            "class_name": "chair",
            "bbox": {"x1": 100, "y1": 100, "x2": 400, "y2": 400},
            "confidence": 0.85,
        }]
        unknown_nested = [{
            "id": 1001,
            "bbox": {"x1": 150, "y1": 150, "x2": 250, "y2": 250},  # 100% inside chair
            "cand_depth": 0.35,
            "confidence": 0.60,
        }]
        res = self.engine.normalize_unknown_objects(unknown_nested, 640.0, 480.0, known_objects=known)
        self.assertEqual(len(res), 0, "Nested unknown contour inside known chair must be suppressed")

    def test_16_unknown_suppressed_by_iou_overlap(self):
        known = [{
            "id": 1,
            "class_name": "dining table",
            "bbox": {"x1": 200, "y1": 150, "x2": 440, "y2": 400},
            "confidence": 0.80,
        }]
        unknown_overlap = [{
            "id": 1001,
            "bbox": {"x1": 220, "y1": 160, "x2": 450, "y2": 410},
            "cand_depth": 0.40,
            "confidence": 0.65,
        }]
        res = self.engine.normalize_unknown_objects(unknown_overlap, 640.0, 480.0, known_objects=known)
        self.assertEqual(len(res), 0, "High IoU unknown contour must be suppressed")

    def test_17_genuine_unknown_obstacle_retained(self):
        known = [{
            "id": 1,
            "class_name": "chair",
            "bbox": {"x1": 50, "y1": 100, "x2": 150, "y2": 300},
            "confidence": 0.85,
        }]
        unknown_distant = [{
            "id": 1001,
            "bbox": {"x1": 450, "y1": 100, "x2": 550, "y2": 300},  # Far on the right, no overlap
            "cand_depth": 0.35,
            "confidence": 0.70,
        }]
        res = self.engine.normalize_unknown_objects(unknown_distant, 640.0, 480.0, known_objects=known)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["position"], "RIGHT")

    def test_18_known_chair_not_overshadowed_by_mild_unknown(self):
        # Chair at same distance as an unknown obstacle must take priority
        objects = [{
            "id": 1,
            "class_name": "chair",
            "bbox": {"x1": 240, "y1": 150, "x2": 400, "y2": 450},
            "confidence": 0.75,
            "depth": 0.35,
            "distance_m": 1.5,
        }]
        unknowns = [{
            "bbox": {"x1": 50, "y1": 150, "x2": 150, "y2": 350},
            "cand_depth": 0.35,
            "confidence": 0.60,
        }]
        analysis = self.engine.analyze(
            objects=objects,
            unknown_objects=unknowns,
            frame_width=640,
            frame_height=480,
        )
        primary = analysis.get("primary_obstacle")
        self.assertIsNotNone(primary)
        self.assertEqual(primary.get("class_name"), "chair", "Known chair must not be overshadowed by mild unknown")

    def test_19_person_always_highest_priority(self):
        objects = [
            {"id": 1, "class_name": "chair", "bbox": {"x1": 100, "y1": 100, "x2": 200, "y2": 300}, "confidence": 0.8, "depth": 0.3},
            {"id": 2, "class_name": "person", "bbox": {"x1": 250, "y1": 50, "x2": 390, "y2": 450}, "confidence": 0.9, "depth": 0.3},
        ]
        analysis = self.engine.analyze(objects=objects, frame_width=640, frame_height=480)
        self.assertEqual(analysis.get("primary_obstacle", {}).get("class_name"), "person")

    def test_20_critical_unknown_takes_priority_over_distant_known(self):
        # A very close unknown obstacle (0.6m) must take priority over a distant table (4.0m)
        objects = [{
            "id": 1,
            "class_name": "dining table",
            "bbox": {"x1": 50, "y1": 50, "x2": 150, "y2": 150},
            "confidence": 0.8,
            "depth": 0.85,
            "distance_m": 4.0,
        }]
        unknowns = [{
            "bbox": {"x1": 240, "y1": 100, "x2": 400, "y2": 450},
            "cand_depth": 0.15,
            "confidence": 0.85,
        }]
        analysis = self.engine.analyze(objects=objects, unknown_objects=unknowns, frame_width=640, frame_height=480)
        self.assertEqual(analysis.get("primary_obstacle", {}).get("class_name"), "unknown")

    # ========================================================
    # 4. DETERMINISTIC NAVIGATION DIRECTIVES (21-32)
    # ========================================================

    def test_21_center_blocked_dist_1_5m_left_clear_generates_move_left(self):
        cand = self.engine._compute_navigation_candidate(
            safety_level="CAUTION",
            emergency_stop=False,
            primary_obstacle={"position": "CENTER", "distance_m": 1.5, "risk_score": 0.55},
            free_path={"left_clear": True, "center_clear": False, "right_clear": False, "left_risk": 0.2, "right_risk": 0.8},
        )
        self.assertEqual(cand[0], "MOVE_LEFT")
        self.assertEqual(cand[1], "LEFT")

    def test_22_center_blocked_dist_1_5m_right_clear_generates_move_right(self):
        cand = self.engine._compute_navigation_candidate(
            safety_level="CAUTION",
            emergency_stop=False,
            primary_obstacle={"position": "CENTER", "distance_m": 1.5, "risk_score": 0.55},
            free_path={"left_clear": False, "center_clear": False, "right_clear": True, "left_risk": 0.8, "right_risk": 0.2},
        )
        self.assertEqual(cand[0], "MOVE_RIGHT")
        self.assertEqual(cand[1], "RIGHT")

    def test_23_center_blocked_close_0_6m_left_clear_generates_turn_left(self):
        cand = self.engine._compute_navigation_candidate(
            safety_level="DANGER",
            emergency_stop=False,
            primary_obstacle={"position": "CENTER", "distance_m": 0.6, "risk_score": 0.85},
            free_path={"left_clear": True, "center_clear": False, "right_clear": False, "left_risk": 0.2, "right_risk": 0.8},
        )
        self.assertEqual(cand[0], "TURN_LEFT")
        self.assertEqual(cand[1], "LEFT")

    def test_24_center_blocked_close_0_6m_right_clear_generates_turn_right(self):
        cand = self.engine._compute_navigation_candidate(
            safety_level="DANGER",
            emergency_stop=False,
            primary_obstacle={"position": "CENTER", "distance_m": 0.6, "risk_score": 0.85},
            free_path={"left_clear": False, "center_clear": False, "right_clear": True, "left_risk": 0.8, "right_risk": 0.2},
        )
        self.assertEqual(cand[0], "TURN_RIGHT")
        self.assertEqual(cand[1], "RIGHT")

    def test_25_obstacle_on_left_shifts_move_right(self):
        cand = self.engine._compute_navigation_candidate(
            safety_level="CAUTION",
            emergency_stop=False,
            primary_obstacle={"position": "LEFT", "distance_m": 1.2, "risk_score": 0.60},
            free_path={"left_clear": False, "center_clear": False, "right_clear": True, "right_risk": 0.2},
        )
        self.assertEqual(cand[0], "MOVE_RIGHT")
        self.assertEqual(cand[1], "RIGHT")

    def test_26_obstacle_on_right_shifts_move_left(self):
        cand = self.engine._compute_navigation_candidate(
            safety_level="CAUTION",
            emergency_stop=False,
            primary_obstacle={"position": "RIGHT", "distance_m": 1.2, "risk_score": 0.60},
            free_path={"left_clear": True, "center_clear": False, "right_clear": False, "left_risk": 0.2},
        )
        self.assertEqual(cand[0], "MOVE_LEFT")
        self.assertEqual(cand[1], "LEFT")

    def test_27_clear_path_generates_go_forward(self):
        cand = self.engine._compute_navigation_candidate(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle=None,
            free_path={"left_clear": True, "center_clear": True, "right_clear": True, "center_risk": 0.1},
        )
        self.assertEqual(cand[0], "GO_FORWARD")
        self.assertEqual(cand[1], "CENTER")

    def test_28_all_forward_paths_blocked_generates_stop(self):
        cand = self.engine._compute_navigation_candidate(
            safety_level="DANGER",
            emergency_stop=False,
            primary_obstacle={"position": "CENTER", "distance_m": 1.0, "risk_score": 0.75},
            free_path={"left_clear": False, "center_clear": False, "right_clear": False, "no_safe_path": True},
            backward_path_clear=False,
        )
        self.assertEqual(cand[0], "STOP")

    def test_29_all_forward_paths_blocked_safe_back_generates_back(self):
        cand = self.engine._compute_navigation_candidate(
            safety_level="DANGER",
            emergency_stop=False,
            primary_obstacle={"position": "CENTER", "distance_m": 0.9, "risk_score": 0.80},
            free_path={"left_clear": False, "center_clear": False, "right_clear": False, "no_safe_path": True},
            backward_path_clear=True,
        )
        self.assertEqual(cand[0], "BACK")
        self.assertEqual(cand[1], "BACK")

    def test_30_emergency_stop_when_obstacle_under_0_5m(self):
        cand = self.engine._compute_navigation_candidate(
            safety_level="EMERGENCY",
            emergency_stop=True,
            primary_obstacle={"position": "CENTER", "distance_m": 0.45, "risk_score": 0.95},
            free_path={"left_clear": True, "center_clear": False, "right_clear": True},
        )
        self.assertEqual(cand[0], "STOP")

    def test_31_approaching_obstacle_slow_down_when_no_detour(self):
        cand = self.engine._compute_navigation_candidate(
            safety_level="CAUTION",
            emergency_stop=False,
            primary_obstacle={"position": "CENTER", "distance_m": 2.2, "risk_score": 0.50, "approaching": True},
            free_path={"left_clear": False, "center_clear": False, "right_clear": False},
        )
        self.assertEqual(cand[0], "SLOW_DOWN")

    def test_32_rapidly_approaching_center_obstacle_stop(self):
        cand = self.engine._compute_navigation_candidate(
            safety_level="DANGER",
            emergency_stop=False,
            primary_obstacle={"position": "CENTER", "distance_m": 1.2, "risk_score": 0.80, "motion_state": "RAPIDLY_APPROACHING"},
            free_path={"left_clear": False, "center_clear": False, "right_clear": False},
        )
        self.assertEqual(cand[0], "STOP")

    # ========================================================
    # 5. STEP GUIDANCE & TURN PRESERVATION (33-38)
    # ========================================================

    def test_33_build_step_guidance_preserves_turn_left(self):
        sg = self.engine.build_step_guidance({"action": "TURN_LEFT", "steps": 0, "distance_m": 0.0})
        self.assertEqual(sg["action"], "TURN_LEFT")
        self.assertEqual(sg["steps"], 0)

    def test_34_build_step_guidance_preserves_turn_right(self):
        sg = self.engine.build_step_guidance({"action": "TURN_RIGHT", "steps": 0, "distance_m": 0.0})
        self.assertEqual(sg["action"], "TURN_RIGHT")
        self.assertEqual(sg["steps"], 0)

    def test_35_build_step_guidance_move_left(self):
        sg = self.engine.build_step_guidance({"action": "MOVE_LEFT", "movement_steps": 2, "movement_distance_m": 1.5})
        self.assertEqual(sg["action"], "WALK_LEFT")
        self.assertEqual(sg["steps"], 2)

    def test_36_build_step_guidance_move_right(self):
        sg = self.engine.build_step_guidance({"action": "MOVE_RIGHT", "movement_steps": 2, "movement_distance_m": 1.5})
        self.assertEqual(sg["action"], "WALK_RIGHT")
        self.assertEqual(sg["steps"], 2)

    def test_37_build_step_guidance_forward(self):
        sg = self.engine.build_step_guidance({"action": "GO_FORWARD", "movement_steps": 4, "movement_distance_m": 3.0})
        self.assertEqual(sg["action"], "WALK_FORWARD")
        self.assertEqual(sg["steps"], 4)

    def test_38_build_step_guidance_stop(self):
        sg = self.engine.build_step_guidance({"action": "STOP", "steps": 0, "distance_m": 0.0})
        self.assertEqual(sg["action"], "STOP")
        self.assertEqual(sg["steps"], 0)

    # ========================================================
    # 6. SCENE FLUX & HEALTH IMMUNITY (39-42)
    # ========================================================

    def test_39_scene_flux_does_not_override_turn_left(self):
        res = self.engine.analyze(
            objects=[{"id": 1, "class_name": "chair", "bbox": {"x1": 240, "y1": 150, "x2": 400, "y2": 450}, "confidence": 0.8, "depth": 0.20, "distance_m": 0.7}],
            scene_depth={"average_depth": 0.5, "center_depth": 0.20, "left_depth": 0.6, "right_depth": 0.2},
            perception_health={"status": "HEALTHY", "scene_flux": True},
            frame_width=640,
            frame_height=480,
        )
        self.assertEqual(res["navigation"]["action"], "TURN_LEFT")
        self.assertEqual(res["navigation"]["direction"], "LEFT")

    def test_40_scene_flux_does_not_override_turn_right(self):
        res = self.engine.analyze(
            objects=[{"id": 1, "class_name": "chair", "bbox": {"x1": 240, "y1": 150, "x2": 400, "y2": 450}, "confidence": 0.8, "depth": 0.20, "distance_m": 0.7}],
            scene_depth={"average_depth": 0.5, "center_depth": 0.20, "left_depth": 0.2, "right_depth": 0.6},
            perception_health={"status": "HEALTHY", "scene_flux": True},
            frame_width=640,
            frame_height=480,
        )
        self.assertEqual(res["navigation"]["action"], "TURN_RIGHT")
        self.assertEqual(res["navigation"]["direction"], "RIGHT")

    def test_41_scene_flux_does_not_override_move_left(self):
        res = self.engine.analyze(
            objects=[{"id": 1, "class_name": "person", "bbox": {"x1": 240, "y1": 150, "x2": 400, "y2": 450}, "confidence": 0.85, "depth": 0.40, "distance_m": 1.6}],
            scene_depth={"average_depth": 0.6, "center_depth": 0.40, "left_depth": 0.7, "right_depth": 0.3},
            perception_health={"status": "HEALTHY", "scene_flux": True},
            frame_width=640,
            frame_height=480,
        )
        self.assertEqual(res["navigation"]["action"], "MOVE_LEFT")
        self.assertEqual(res["navigation"]["direction"], "LEFT")

    def test_42_degraded_perception_slows_down_forward_safely(self):
        res = self.engine.analyze(
            objects=[],
            scene_depth={"average_depth": 0.7, "center_depth": 0.7, "left_depth": 0.7, "right_depth": 0.7},
            perception_health={"status": "DEGRADED", "message": "Low light condition"},
            frame_width=640,
            frame_height=480,
        )
        self.assertEqual(res["navigation"]["action"], "SLOW_DOWN")

    # ========================================================
    # 7. VOICE INSTRUCTION SYNTHESIS & EXPLAINABILITY (43-50)
    # ========================================================

    def test_43_voice_person_ahead_move_left(self):
        v = self.engine.generate_voice_instruction(
            safety_level="CAUTION",
            emergency_stop=False,
            primary_obstacle={"class_name": "person", "position": "CENTER"},
            navigation="MOVE_LEFT",
        )
        self.assertEqual(v, "Person ahead. Move left.")

    def test_44_voice_chair_ahead_turn_left(self):
        v = self.engine.generate_voice_instruction(
            safety_level="DANGER",
            emergency_stop=False,
            primary_obstacle={"class_name": "chair", "position": "CENTER"},
            navigation="TURN_LEFT",
        )
        self.assertEqual(v, "Chair ahead. Turn left.")

    def test_45_voice_table_ahead_move_right(self):
        v = self.engine.generate_voice_instruction(
            safety_level="CAUTION",
            emergency_stop=False,
            primary_obstacle={"class_name": "dining table", "position": "CENTER"},
            navigation="MOVE_RIGHT",
        )
        self.assertEqual(v, "Table ahead. Move right.")

    def test_46_voice_unknown_obstacle_turn_right(self):
        v = self.engine.generate_voice_instruction(
            safety_level="DANGER",
            emergency_stop=False,
            primary_obstacle={"class_name": "unknown", "position": "CENTER"},
            navigation="TURN_RIGHT",
        )
        self.assertEqual(v, "Unknown obstacle ahead. Turn right.")

    def test_47_voice_stop_person_ahead(self):
        v = self.engine.generate_voice_instruction(
            safety_level="DANGER",
            emergency_stop=False,
            primary_obstacle={"class_name": "person", "position": "CENTER"},
            navigation="STOP",
        )
        self.assertEqual(v, "Stop. Person ahead.")

    def test_48_voice_emergency_stop_immediately(self):
        v = self.engine.generate_voice_instruction(
            safety_level="EMERGENCY",
            emergency_stop=True,
            primary_obstacle=None,
            navigation="STOP",
        )
        self.assertEqual(v, "Stop immediately.")

    def test_49_voice_clear_path_move_forward(self):
        v = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle=None,
            navigation="GO_FORWARD",
            steps=4,
        )
        self.assertIn("Move forward", v)

    def test_50_voice_slow_down(self):
        v = self.engine.generate_voice_instruction(
            safety_level="CAUTION",
            emergency_stop=False,
            primary_obstacle={"class_name": "chair", "position": "CENTER"},
            navigation="SLOW_DOWN",
        )
        self.assertIn("Slow down", v)

    # ========================================================
    # 8. MULTI-FRAME CONTINUOUS WALKING SIMULATION (51-55)
    # ========================================================

    def test_51_walk_simulation_clear_corridor(self):
        # User walks forward through a clear corridor
        for _ in range(5):
            res = self.engine.analyze(
                objects=[],
                scene_depth={"average_depth": 0.8, "center_depth": 0.8, "left_depth": 0.8, "right_depth": 0.8},
                frame_width=640,
                frame_height=480,
            )
            self.assertEqual(res["navigation"]["action"], "GO_FORWARD")
            self.assertEqual(res["safety_level"], "SAFE")

    def test_52_walk_simulation_chair_appears_move_right(self):
        # Clear frame -> chair detected at 2.0m in center -> moves right
        self.engine.analyze(objects=[], scene_depth={"average_depth": 0.8}, frame_width=640, frame_height=480)
        res = self.engine.analyze(
            objects=[{"id": 1, "class_name": "chair", "bbox": {"x1": 250, "y1": 150, "x2": 390, "y2": 450}, "confidence": 0.8, "depth": 0.45, "distance_m": 2.0}],
            scene_depth={"average_depth": 0.6, "center_depth": 0.45, "left_depth": 0.3, "right_depth": 0.7},
            frame_width=640,
            frame_height=480,
        )
        self.assertEqual(res["navigation"]["action"], "MOVE_RIGHT")
        self.assertEqual(res["voice_instruction"], "Chair ahead. Move right.")

    def test_53_walk_simulation_rapid_approaching_person_stops(self):
        # Stationary person at 2.5m -> person rapidly approaches to 1.1m
        res1 = self.engine.analyze(
            objects=[{"id": 1, "class_name": "person", "bbox": {"x1": 250, "y1": 100, "x2": 390, "y2": 450}, "confidence": 0.9, "depth": 0.5, "distance_m": 2.5}],
            scene_depth={"average_depth": 0.5},
            frame_width=640,
            frame_height=480,
        )
        res2 = self.engine.analyze(
            objects=[{"id": 1, "class_name": "person", "bbox": {"x1": 200, "y1": 50, "x2": 440, "y2": 480}, "confidence": 0.92, "depth": 0.22, "distance_m": 1.1}],
            scene_depth={"average_depth": 0.3},
            frame_width=640,
            frame_height=480,
        )
        self.assertEqual(res2["navigation"]["action"], "STOP")
        self.assertIn("Stop", res2["voice_instruction"])

    def test_54_walk_simulation_table_and_chair_corridor_bypass(self):
        # Left chair (0.2m risk) + center table (blocked) + right corridor clear
        res = self.engine.analyze(
            objects=[
                {"id": 1, "class_name": "chair", "bbox": {"x1": 50, "y1": 150, "x2": 180, "y2": 400}, "confidence": 0.8, "depth": 0.35, "distance_m": 1.4},
                {"id": 2, "class_name": "dining table", "bbox": {"x1": 240, "y1": 150, "x2": 400, "y2": 450}, "confidence": 0.85, "depth": 0.35, "distance_m": 1.4},
            ],
            scene_depth={"average_depth": 0.5, "center_depth": 0.35, "left_depth": 0.35, "right_depth": 0.75},
            frame_width=640,
            frame_height=480,
        )
        self.assertEqual(res["navigation"]["action"], "MOVE_RIGHT")
        self.assertEqual(res["navigation"]["direction"], "RIGHT")

    def test_55_unknown_depth_blob_in_center_emergency_stop(self):
        # Genuine unclassified obstacle 0.12 depth in center triggers emergency stop
        res = self.engine.analyze(
            objects=[],
            unknown_objects=[{"bbox": {"x1": 200, "y1": 150, "x2": 440, "y2": 460}, "cand_depth": 0.12, "confidence": 0.82}],
            scene_depth={"average_depth": 0.4, "center_depth": 0.15, "left_depth": 0.5, "right_depth": 0.5},
            frame_width=640,
            frame_height=480,
        )
        self.assertTrue(res.get("emergency_stop"))
        self.assertEqual(res.get("voice_instruction"), "Stop immediately.")


if __name__ == "__main__":
    unittest.main()

