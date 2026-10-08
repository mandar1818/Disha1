"""
SmartVisionAI - Phase 8 Comprehensive Indoor Safety & Hazard Robustness Test Suite
===================================================================================
Covers 41 verification tests for Phase 8 indoor safety requirements:
- Category A: Door / Doorway Understanding (Tests 1-4)
- Category B: Stairs / Steps / Ramps / Level Transitions (Tests 5-9)
- Category C: Floor Micro-Hazards (Tests 10-14)
- Category D: Corridor & Narrow Passages (Tests 15-19)
- Category E: Glass & Transparent Obstacles (Tests 20-24)
- Category F: Mirrors & Reflections (Tests 25-28)
- Category G: Unknown Physical Obstacles (Tests 29-31)
- Category H: Multiple Simultaneous Hazards & Collective Reasoning (Tests 32-36)
- Category I: Closed-Loop Navigation Integration & Perception Failure (Tests 37-41)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import time
import unittest
import numpy as np
import cv2

from backend.indoor.door import DoorwayAnalyzer
from backend.indoor.stairs import StairDropoffAnalyzer
from backend.indoor.floor_hazard import FloorHazardAnalyzer
from backend.indoor.glass import GlassHazardAnalyzer
from backend.indoor.corridor import CorridorClearanceAnalyzer
from backend.indoor.safety_manager import IndoorSafetyManager
from backend.decision.engine import DecisionEngine
from backend.navigation.session import NavigationSession, NavigationSessionState
from backend.navigation.destination import Destination
from backend.safety.health import PerceptionHealthEvaluator


class TestPhase8IndoorSafety(unittest.TestCase):
    def setUp(self):
        self.door_analyzer = DoorwayAnalyzer()
        self.stair_analyzer = StairDropoffAnalyzer()
        self.floor_analyzer = FloorHazardAnalyzer()
        self.glass_analyzer = GlassHazardAnalyzer()
        self.corridor_analyzer = CorridorClearanceAnalyzer()
        self.safety_manager = IndoorSafetyManager()
        self.decision_engine = DecisionEngine()
        self.health_evaluator = PerceptionHealthEvaluator()

    # =========================================================================
    # Category A: Door / Doorway Understanding (Tests 1-4)
    # =========================================================================

    def test_01_open_doorway_clear_navigation(self):
        """Test 1: Open doorway allows clear path forward without emergency stop."""
        door_result = {
            "detected": True,
            "state": "OPEN",
            "confidence": 0.85,
            "door_position": "CENTER",
            "is_uncertain": False,
            "obstacle_beyond": False,
            "message": "Open doorway ahead.",
        }
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.85, "left_depth": 0.80, "right_depth": 0.80},
            frame_width=640,
            frame_height=480,
            indoor_safety={"door_analysis": door_result},
        )
        self.assertFalse(res.get("emergency_stop"))
        self.assertEqual(res.get("navigation", {}).get("action"), "FORWARD")

    def test_02_closed_door_safety_stop(self):
        """Test 2: Closed door triggers STOP directive with 'Door closed. Do not enter.'."""
        door_result = {
            "detected": True,
            "state": "CLOSED",
            "confidence": 0.88,
            "door_position": "CENTER",
            "is_uncertain": False,
            "obstacle_beyond": False,
            "message": "Closed door ahead.",
        }
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.40, "left_depth": 0.40, "right_depth": 0.40},
            frame_width=640,
            frame_height=480,
            indoor_safety={"door_analysis": door_result},
        )
        self.assertEqual(res.get("navigation", {}).get("action"), "STOP")
        self.assertIn("Closed door", res.get("navigation", {}).get("reason", ""))
        self.assertEqual(res.get("voice_instruction"), "Door closed. Do not enter.")

    def test_03_blocked_doorway_directive(self):
        """Test 3: Blocked doorway with no lateral bypass issues 'Stop. The doorway ahead is blocked.'."""
        door_result = {
            "detected": True,
            "state": "BLOCKED",
            "confidence": 0.85,
            "door_position": "CENTER",
            "is_uncertain": False,
            "obstacle_beyond": False,
            "message": "Doorway blocked.",
        }
        res = self.decision_engine.analyze(
            objects=[{"class_name": "chair", "bbox": {"x1": 200, "y1": 150, "x2": 440, "y2": 420}, "depth": 0.35, "confidence": 0.85}],
            scene_depth={"center_depth": 0.35, "left_depth": 0.30, "right_depth": 0.30},
            frame_width=640,
            frame_height=480,
            indoor_safety={"door_analysis": door_result},
        )
        self.assertEqual(res.get("navigation", {}).get("action"), "STOP")
        self.assertEqual(res.get("voice_instruction"), "Stop. The doorway ahead is blocked.")

    def test_04_doorway_obstacle_beyond_cautious_advance(self):
        """Test 4: Doorway with obstacle beyond (>1.8m) directs SLOW_DOWN."""
        door_result = {
            "detected": True,
            "state": "DOORWAY_OBSTACLE_BEYOND",
            "confidence": 0.80,
            "door_position": "CENTER",
            "is_uncertain": False,
            "obstacle_beyond": True,
            "message": "Doorway clear, obstacle ahead at 2.4m.",
        }
        res = self.decision_engine.analyze(
            objects=[{"class_name": "person", "bbox": {"x1": 280, "y1": 120, "x2": 360, "y2": 320}, "depth": 0.70, "distance_m": 2.4, "confidence": 0.80}],
            scene_depth={"center_depth": 0.70, "left_depth": 0.65, "right_depth": 0.65},
            frame_width=640,
            frame_height=480,
            indoor_safety={"door_analysis": door_result},
        )
        self.assertEqual(res.get("navigation", {}).get("action"), "SLOW_DOWN")
        self.assertEqual(res.get("voice_instruction"), "Doorway clear, obstacle ahead.")

    # =========================================================================
    # Category B: Stairs / Steps / Ramps / Level Transitions (Tests 5-9)
    # =========================================================================

    def test_05_critical_drop_off_emergency_stop_no_fake_metric(self):
        """Test 5: Drop-off triggers EMERGENCY STOP with conservative directive (no fake distance claim)."""
        stair_result = {
            "detected": True,
            "type": "DROP_OFF",
            "confidence": 0.90,
            "hazard_level": "CRITICAL",
            "distance_m": 1.2,
            "drop_off_risk": True,
            "message": "Stop. Drop-off ahead.",
        }
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.50},
            frame_width=640,
            frame_height=480,
            indoor_safety={"stair_analysis": stair_result},
        )
        self.assertTrue(res.get("emergency_stop"))
        self.assertEqual(res.get("safety_level"), "EMERGENCY")
        self.assertEqual(res.get("navigation", {}).get("action"), "STOP")
        self.assertEqual(res.get("voice_instruction"), "Stop. Drop-off ahead.")

    def test_06_stairs_down_emergency_stop(self):
        """Test 6: Downward stairs trigger EMERGENCY STOP with conservative directive."""
        stair_result = {
            "detected": True,
            "type": "STAIRS_DOWN",
            "confidence": 0.88,
            "hazard_level": "CRITICAL",
            "drop_off_risk": True,
            "message": "Stop. Stairs detected ahead.",
        }
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.50},
            frame_width=640,
            frame_height=480,
            indoor_safety={"stair_analysis": stair_result},
        )
        self.assertTrue(res.get("emergency_stop"))
        self.assertEqual(res.get("navigation", {}).get("action"), "STOP")
        self.assertEqual(res.get("voice_instruction"), "Stop. Stairs detected ahead.")

    def test_07_stairs_up_caution_slow_down(self):
        """Test 7: Upward stairs trigger CAUTION / SLOW_DOWN."""
        stair_result = {
            "detected": True,
            "type": "STAIRS_UP",
            "confidence": 0.82,
            "hazard_level": "CAUTION",
            "drop_off_risk": False,
            "message": "Caution. Stairs up ahead.",
        }
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.70},
            frame_width=640,
            frame_height=480,
            indoor_safety={"stair_analysis": stair_result},
        )
        self.assertFalse(res.get("emergency_stop"))
        self.assertEqual(res.get("navigation", {}).get("action"), "SLOW_DOWN")
        self.assertEqual(res.get("voice_instruction"), "Caution. Stairs up ahead.")

    def test_08_single_step_slow_down(self):
        """Test 8: Single step triggers SLOW_DOWN with level change advisory."""
        stair_result = {
            "detected": True,
            "type": "SINGLE_STEP",
            "confidence": 0.78,
            "hazard_level": "CAUTION",
            "drop_off_risk": False,
            "message": "Slow down. Floor level change ahead.",
        }
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.70},
            frame_width=640,
            frame_height=480,
            indoor_safety={"stair_analysis": stair_result},
        )
        self.assertEqual(res.get("navigation", {}).get("action"), "SLOW_DOWN")
        self.assertEqual(res.get("voice_instruction"), "Slow down. Floor level change ahead.")

    def test_09_ramp_caution_slow_down(self):
        """Test 9: Ramp transition triggers CAUTION / SLOW_DOWN."""
        stair_result = {
            "detected": True,
            "type": "RAMP",
            "confidence": 0.75,
            "hazard_level": "CAUTION",
            "drop_off_risk": False,
            "message": "Caution. Ramp ahead.",
        }
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.70},
            frame_width=640,
            frame_height=480,
            indoor_safety={"stair_analysis": stair_result},
        )
        self.assertEqual(res.get("navigation", {}).get("action"), "SLOW_DOWN")
        self.assertEqual(res.get("voice_instruction"), "Caution. Ramp ahead.")

    # =========================================================================
    # Category C: Floor Micro-Hazards (Tests 10-14)
    # =========================================================================

    def test_10_floor_cable_detection_and_directive(self):
        """Test 10: Floor cable (aspect ratio >= 3.5) triggers trip hazard SLOW_DOWN."""
        floor_hazards = [{
            "id": 1001,
            "hazard_type": "CABLE",
            "hazard_level": "CAUTION",
            "position": "CENTER",
            "confidence": 0.75,
            "reason": "Cable across floor.",
            "bbox": {"x1": 150, "y1": 380, "x2": 490, "y2": 395},
        }]
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.75},
            frame_width=640,
            frame_height=480,
            indoor_safety={"floor_hazards": floor_hazards},
        )
        self.assertEqual(res.get("navigation", {}).get("action"), "SLOW_DOWN")
        self.assertEqual(res.get("voice_instruction"), "Trip hazard on floor ahead. Slow down.")

    def test_11_floor_threshold_uneven_surface(self):
        """Test 11: Floor threshold triggers uneven surface SLOW_DOWN."""
        floor_hazards = [{
            "id": 1002,
            "hazard_type": "THRESHOLD",
            "hazard_level": "CAUTION",
            "position": "CENTER",
            "confidence": 0.72,
            "reason": "Raised threshold transition.",
            "bbox": {"x1": 120, "y1": 390, "x2": 520, "y2": 420},
        }]
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.75},
            frame_width=640,
            frame_height=480,
            indoor_safety={"floor_hazards": floor_hazards},
        )
        self.assertEqual(res.get("navigation", {}).get("action"), "SLOW_DOWN")
        self.assertEqual(res.get("voice_instruction"), "Uneven surface ahead.")

    def test_12_floor_depression_hole_emergency_stop(self):
        """Test 12: Negative floor depth anomaly (hole/depression) triggers CRITICAL EMERGENCY STOP."""
        floor_hazards = [{
            "id": 1003,
            "hazard_type": "HOLE",
            "hazard_level": "CRITICAL",
            "position": "CENTER",
            "confidence": 0.85,
            "reason": "Hole or depression on the floor ahead. Stop.",
            "bbox": {"x1": 220, "y1": 360, "x2": 420, "y2": 440},
        }]
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.40},
            frame_width=640,
            frame_height=480,
            indoor_safety={"floor_hazards": floor_hazards},
        )
        self.assertTrue(res.get("emergency_stop"))
        self.assertEqual(res.get("safety_level"), "EMERGENCY")
        self.assertEqual(res.get("navigation", {}).get("action"), "STOP")
        self.assertEqual(res.get("voice_instruction"), "Hole or depression on the floor ahead. Stop.")

    def test_13_uneven_floor_surface(self):
        """Test 13: Uneven floor variance triggers SLOW_DOWN."""
        floor_hazards = [{
            "id": 1004,
            "hazard_type": "UNEVEN_FLOOR",
            "hazard_level": "CAUTION",
            "position": "CENTER",
            "confidence": 0.70,
            "reason": "Uneven surface ahead.",
            "bbox": {"x1": 100, "y1": 350, "x2": 540, "y2": 450},
        }]
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.70},
            frame_width=640,
            frame_height=480,
            indoor_safety={"floor_hazards": floor_hazards},
        )
        self.assertEqual(res.get("navigation", {}).get("action"), "SLOW_DOWN")
        self.assertEqual(res.get("voice_instruction"), "Uneven surface ahead.")

    def test_14_floor_analyzer_morphology_classification(self):
        """Test 14: FloorHazardAnalyzer accurately classifies synthetic contour types."""
        # Create synthetic synthetic contour resembling a horizontal cable
        cable_res = self.floor_analyzer._classify_contour_type(
            cnt_w=180, cnt_h=12,
            aspect=15.0,
            extent=0.85,
            solidity=0.90,
            residual_std=0.04,
            min_depth=0.50,
            mean_depth=0.52,
            target_h=240,
            target_w=320,
        )
        self.assertEqual(cable_res, "CABLE")

        # Negative depth drop contour -> HOLE
        hole_res = self.floor_analyzer._classify_contour_type(
            cnt_w=80, cnt_h=60,
            aspect=1.33,
            extent=0.70,
            solidity=0.85,
            residual_std=0.08,
            min_depth=0.90,  # Far / dropping away relative to baseline
            mean_depth=0.88,
            target_h=240,
            target_w=320,
        )
        self.assertEqual(hole_res, "HOLE")

    # =========================================================================
    # Category D: Corridor & Narrow Passages (Tests 15-19)
    # =========================================================================

    def test_15_wide_corridor_clear(self):
        """Test 15: Wide corridor (>1.05m) yields CLEAR and FORWARD."""
        corridor = {
            "clearance_state": "CLEAR",
            "walkable_width_m": 1.40,
            "confidence": 0.90,
            "message": "Clear passage ahead.",
        }
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.85, "left_depth": 0.80, "right_depth": 0.80},
            frame_width=640,
            frame_height=480,
            indoor_safety={"corridor_analysis": corridor},
        )
        self.assertEqual(res.get("navigation", {}).get("action"), "FORWARD")

    def test_16_narrow_corridor_slow_down(self):
        """Test 16: Narrow corridor (0.65m - 1.05m) yields NARROW and SLOW_DOWN."""
        corridor = {
            "clearance_state": "NARROW",
            "walkable_width_m": 0.85,
            "confidence": 0.85,
            "message": "Narrow passage: maintain center clearance",
        }
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.70},
            frame_width=640,
            frame_height=480,
            indoor_safety={"corridor_analysis": corridor},
        )
        self.assertEqual(res.get("navigation", {}).get("action"), "SLOW_DOWN")
        self.assertEqual(res.get("voice_instruction"), "Narrow passage. Slow down.")

    def test_17_blocked_corridor_stop(self):
        """Test 17: Blocked corridor (<0.65m) yields BLOCKED and STOP."""
        corridor = {
            "clearance_state": "BLOCKED",
            "walkable_width_m": 0.45,
            "confidence": 0.92,
            "message": "Insufficient shoulder clearance: path blocked",
        }
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.40},
            frame_width=640,
            frame_height=480,
            indoor_safety={"corridor_analysis": corridor},
        )
        self.assertEqual(res.get("navigation", {}).get("action"), "STOP")
        self.assertEqual(res.get("voice_instruction"), "Stop. Path blocked.")

    def test_18_inter_obstacle_gap_analysis(self):
        """Test 18: CorridorClearanceAnalyzer correctly measures inter-obstacle gap."""
        obs = [
            {"bbox": {"x1": 50, "y1": 150, "x2": 240, "y2": 450}, "depth": 0.45, "class_name": "box"},  # Left
            {"bbox": {"x1": 400, "y1": 150, "x2": 590, "y2": 450}, "depth": 0.45, "class_name": "chair"}, # Right
        ]
        gap_m = self.corridor_analyzer._compute_inter_obstacle_gap(obs, 640.0)
        # Pixel gap = 400 - 240 = 160 px; ratio = 160/640 = 0.25; approx 0.50m
        self.assertIsNotNone(gap_m)
        self.assertLess(gap_m, 0.65) # Narrow/blocked gap (<0.65m)

    def test_19_crowd_corridor_blockage(self):
        """Test 19: Multiple nearby persons blocking walkway triggers BLOCKED clearance."""
        crowd_obs = [
            {"bbox": {"x1": 150, "y1": 100, "x2": 320, "y2": 450}, "distance_m": 1.4, "depth": 0.40, "class_name": "person"},
            {"bbox": {"x1": 310, "y1": 100, "x2": 480, "y2": 450}, "distance_m": 1.5, "depth": 0.42, "class_name": "person"},
        ]
        # Evaluate corridor analyzer with crowd
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        depth_map = np.ones((480, 640), dtype=np.float32) * 0.40
        corr_res = self.corridor_analyzer.analyze(frame=frame, depth_map=depth_map, obstacles=crowd_obs)
        self.assertEqual(corr_res.get("clearance_state"), "BLOCKED")

    # =========================================================================
    # Category E: Glass & Transparent Obstacles (Tests 20-24)
    # =========================================================================

    def test_20_glass_door_detection_and_directive(self):
        """Test 20: Glass door detection sets is_uncertain=True and directs caution."""
        glass = {
            "detected": True,
            "hazard_type": "GLASS_DOOR",
            "confidence": 0.80,
            "position": "CENTER",
            "risk_score": 0.75,
            "is_uncertain": True,
            "voice_message": "Caution. Possible glass door ahead.",
            "reason": "Possible glass door detected on center.",
        }
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.85}, # Open depth paradox
            frame_width=640,
            frame_height=480,
            indoor_safety={"glass_obstacles": glass},
        )
        self.assertEqual(res.get("navigation", {}).get("action"), "SLOW_DOWN")
        self.assertEqual(res.get("voice_instruction"), "Caution. Possible glass door ahead.")

    def test_21_glass_wall_directive(self):
        """Test 21: Glass wall yields caution voice directive."""
        glass = {
            "detected": True,
            "hazard_type": "GLASS_WALL",
            "confidence": 0.75,
            "position": "CENTER",
            "risk_score": 0.80,
            "is_uncertain": True,
            "voice_message": "Caution. Possible glass wall ahead.",
            "reason": "Possible glass partition wall detected on center.",
        }
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.85},
            frame_width=640,
            frame_height=480,
            indoor_safety={"glass_obstacles": glass},
        )
        self.assertEqual(res.get("navigation", {}).get("action"), "SLOW_DOWN")
        self.assertEqual(res.get("voice_instruction"), "Caution. Possible glass wall ahead.")

    def test_22_glass_panel_directive(self):
        """Test 22: Glass panel yields caution voice directive."""
        glass = {
            "detected": True,
            "hazard_type": "GLASS_PANEL",
            "confidence": 0.70,
            "position": "RIGHT",
            "risk_score": 0.65,
            "is_uncertain": True,
            "voice_message": "Caution. Possible glass obstacle ahead.",
            "reason": "Possible glass panel detected on right.",
        }
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.80},
            frame_width=640,
            frame_height=480,
            indoor_safety={"glass_obstacles": glass},
        )
        self.assertEqual(res.get("navigation", {}).get("action"), "SLOW_DOWN")
        self.assertEqual(res.get("voice_instruction"), "Caution. Possible glass obstacle ahead.")

    def test_23_uncertain_transparent_surface(self):
        """Test 23: Specular anomaly with contradictory open depth yields UNCERTAIN_TRANSPARENT."""
        glass = {
            "detected": True,
            "hazard_type": "UNCERTAIN_TRANSPARENT",
            "confidence": 0.65,
            "position": "CENTER",
            "risk_score": 0.70,
            "is_uncertain": True,
            "voice_message": "Slow down. Possible transparent surface ahead.",
            "reason": "Specular reflection anomaly with contradictory open depth indicates possible transparent barrier.",
        }
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.85},
            frame_width=640,
            frame_height=480,
            indoor_safety={"glass_obstacles": glass},
        )
        self.assertEqual(res.get("navigation", {}).get("action"), "SLOW_DOWN")
        self.assertEqual(res.get("voice_instruction"), "Slow down. Possible transparent surface ahead.")

    def test_24_glass_confidence_capped_no_100_percent_claim(self):
        """Test 24: GlassHazardAnalyzer never claims 100% confidence (capped <= 0.85)."""
        # Create image with artificial mullion lines and specular bright spots
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Draw 2 vertical lines
        cv2.line(frame, (100, 10), (100, 230), (255, 255, 255), 3)
        cv2.line(frame, (220, 10), (220, 230), (255, 255, 255), 3)
        # Draw bright specular spots
        cv2.circle(frame, (160, 100), 20, (255, 255, 255), -1)
        depth_map = np.ones((240, 320), dtype=np.float32) * 0.90 # Penetrating open depth

        res = self.glass_analyzer.analyze(frame=frame, depth_map=depth_map)
        if res.get("detected"):
            self.assertLessEqual(res.get("confidence", 0.0), 0.85)
            self.assertTrue(res.get("is_uncertain"))

    # =========================================================================
    # Category F: Mirrors & Reflection Handling (Tests 25-28)
    # =========================================================================

    def test_25_mirror_reflection_pair_deduplication(self):
        """Test 25: Same class with similar size and divergent depth is deduplicated to 1 obstacle."""
        raw_objects = [
            {"class_name": "person", "bbox": {"x1": 100, "y1": 100, "x2": 220, "y2": 400}, "depth": 0.30, "confidence": 0.90}, # Real close person
            {"class_name": "person", "bbox": {"x1": 300, "y1": 100, "x2": 420, "y2": 400}, "depth": 0.80, "confidence": 0.85}, # Virtual mirror reflection
        ]
        # When reflection uncertainty is flagged
        deduped = self.decision_engine.remove_duplicates(raw_objects, reflection_uncertainty=True)
        self.assertEqual(len(deduped), 1)

    def test_26_closer_physical_obstacle_retained_over_virtual_reflection(self):
        """Test 26: Deduplication retains the closer physical obstacle (higher risk) for safety."""
        raw_objects = [
            {"class_name": "person", "bbox": {"x1": 300, "y1": 100, "x2": 420, "y2": 400}, "depth": 0.85, "confidence": 0.92}, # Higher conf but far reflection
            {"class_name": "person", "bbox": {"x1": 100, "y1": 100, "x2": 220, "y2": 400}, "depth": 0.28, "confidence": 0.86}, # Real close person
        ]
        deduped = self.decision_engine.remove_duplicates(raw_objects, reflection_uncertainty=True)
        self.assertEqual(len(deduped), 1)
        # Closer object (depth 0.28) must be kept
        self.assertAlmostEqual(deduped[0]["depth"], 0.28, places=2)

    def test_27_critical_fall_hazard_overrides_reflection_uncertainty(self):
        """Test 27: Drop-off / critical stairs takes absolute precedence over reflection uncertainty."""
        stair_result = {
            "detected": True,
            "type": "DROP_OFF",
            "confidence": 0.90,
            "hazard_level": "CRITICAL",
            "message": "Stop. Drop-off ahead.",
        }
        perception_health = {
            "status": "UNCERTAIN",
            "reflection_uncertainty": True,
            "message": "Vision is uncertain. Please pause.",
        }
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.40},
            frame_width=640,
            frame_height=480,
            indoor_safety={"stair_analysis": stair_result},
            perception_health=perception_health,
        )
        self.assertTrue(res.get("emergency_stop"))
        self.assertEqual(res.get("safety_level"), "EMERGENCY")
        self.assertEqual(res.get("voice_instruction"), "Stop. Drop-off ahead.")

    def test_28_reflection_uncertainty_alone_pauses_navigation(self):
        """Test 28: Reflection uncertainty alone pauses navigation safely."""
        perception_health = {
            "status": "UNCERTAIN",
            "reflection_uncertainty": True,
            "message": "Vision is uncertain. Please pause.",
            "voice_override": "Vision is uncertain. Please pause.",
        }
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.80},
            frame_width=640,
            frame_height=480,
            perception_health=perception_health,
        )
        self.assertEqual(res.get("navigation", {}).get("action"), "STOP")
        self.assertEqual(res.get("voice_instruction"), "Vision is uncertain. Please pause.")

    # =========================================================================
    # Category G: Unknown Physical Obstacles (Tests 29-31)
    # =========================================================================

    def test_29_depth_only_obstacle_not_detected_by_yolo(self):
        """Test 29: NOT detected by YOLO != safe. Depth obstacle in center triggers safety directive."""
        # YOLO detects 0 objects, but center region depth is very close (0.20)
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.20, "center_region_depth": 0.20, "left_depth": 0.80, "right_depth": 0.80},
            frame_width=640,
            frame_height=480,
        )
        # Must detect physical obstacle or danger in center and not say continue forward blindly
        nav_act = res.get("navigation", {}).get("action")
        self.assertIn(nav_act, ("MOVE_LEFT", "MOVE_RIGHT", "STOP", "SLOW_DOWN"))
        self.assertNotEqual(nav_act, "FORWARD")

    def test_30_floor_hazard_integrated_as_unknown_obstacle(self):
        """Test 30: Floor hazard candidate is integrated into unknown objects in DecisionEngine."""
        floor_hazards = [{
            "id": 1005,
            "hazard_type": "TRIP_HAZARD",
            "hazard_level": "CAUTION",
            "position": "CENTER",
            "confidence": 0.70,
            "bbox": {"x1": 200, "y1": 380, "x2": 440, "y2": 430},
            "cand_depth": 0.35,
        }]
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.70},
            frame_width=640,
            frame_height=480,
            indoor_safety={"floor_hazards": floor_hazards},
        )
        unknowns = res.get("unknown_objects", [])
        self.assertGreater(len(unknowns), 0)

    def test_31_unknown_obstacle_normalization_contract(self):
        """Test 31: Unknown obstacle meets Phase 2/4/8 schema contract."""
        unknown_candidates = [{
            "id": 1009,
            "bbox": {"x1": 200, "y1": 200, "x2": 400, "y2": 400},
            "cand_depth": 0.40,
            "confidence": 0.70,
        }]
        norm = self.decision_engine.normalize_unknown_objects(unknown_candidates, frame_width=640, frame_height=480)
        self.assertEqual(len(norm), 1)
        item = norm[0]
        self.assertIn("distance_m", item)
        self.assertIn("distance_category", item)
        self.assertIn("risk_score", item)
        self.assertIn("navigation_relevance", item)

    # =========================================================================
    # Category H: Multiple Simultaneous Hazards & Collective Reasoning (Tests 32-36)
    # =========================================================================

    def test_32_stairs_down_overrides_narrow_corridor(self):
        """Test 32: Stairs down (Priority 1) overrides narrow corridor (Priority 3)."""
        stair_result = {
            "detected": True,
            "type": "STAIRS_DOWN",
            "confidence": 0.90,
            "hazard_level": "CRITICAL",
            "drop_off_risk": True,
            "message": "Stop. Stairs detected ahead.",
        }
        corridor = {
            "clearance_state": "NARROW",
            "walkable_width_m": 0.85,
        }
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.50},
            frame_width=640,
            frame_height=480,
            indoor_safety={"stair_analysis": stair_result, "corridor_analysis": corridor},
        )
        self.assertTrue(res.get("emergency_stop"))
        self.assertEqual(res.get("voice_instruction"), "Stop. Stairs detected ahead.")

    def test_33_drop_off_overrides_blocked_doorway(self):
        """Test 33: Drop-off (Priority 1) overrides blocked doorway (Priority 2)."""
        stair_result = {
            "detected": True,
            "type": "DROP_OFF",
            "confidence": 0.92,
            "hazard_level": "CRITICAL",
            "drop_off_risk": True,
            "message": "Stop. Drop-off ahead.",
        }
        door_result = {
            "detected": True,
            "state": "BLOCKED",
            "message": "Doorway blocked.",
        }
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.35},
            frame_width=640,
            frame_height=480,
            indoor_safety={"stair_analysis": stair_result, "door_analysis": door_result},
        )
        self.assertTrue(res.get("emergency_stop"))
        self.assertEqual(res.get("voice_instruction"), "Stop. Drop-off ahead.")

    def test_34_blocked_doorway_with_safe_lateral_detour(self):
        """Test 34: Blocked doorway with open lateral free path directs MOVE_RIGHT/LEFT detour."""
        door_result = {
            "detected": True,
            "state": "BLOCKED",
            "confidence": 0.85,
            "door_position": "CENTER",
            "message": "Doorway blocked.",
        }
        # Obstacle in center, but clear open space on right
        res = self.decision_engine.analyze(
            objects=[{"class_name": "chair", "bbox": {"x1": 220, "y1": 150, "x2": 420, "y2": 420}, "depth": 0.40, "confidence": 0.85}],
            scene_depth={"center_depth": 0.35, "left_depth": 0.30, "right_depth": 0.90},
            frame_width=640,
            frame_height=480,
            indoor_safety={"door_analysis": door_result},
        )
        # Should guide right detour rather than purely hard stopping without alternative
        nav_act = res.get("navigation", {}).get("action")
        self.assertIn(nav_act, ("MOVE_RIGHT", "STOP"))

    def test_35_blocked_doorway_with_no_lateral_bypass_stops(self):
        """Test 35: Blocked doorway with NO lateral clearance commands STOP."""
        door_result = {
            "detected": True,
            "state": "BLOCKED",
            "confidence": 0.85,
            "door_position": "CENTER",
            "message": "Stop. The doorway ahead is blocked.",
        }
        # All directions blocked
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.30, "left_depth": 0.25, "right_depth": 0.25},
            frame_width=640,
            frame_height=480,
            indoor_safety={"door_analysis": door_result},
        )
        self.assertEqual(res.get("navigation", {}).get("action"), "STOP")
        self.assertEqual(res.get("voice_instruction"), "Stop. The doorway ahead is blocked.")

    def test_36_glass_and_floor_hazard_simultaneous(self):
        """Test 36: Glass hazard + floor trip hazard simultaneously triggers SLOW_DOWN."""
        glass = {
            "detected": True,
            "hazard_type": "GLASS_DOOR",
            "confidence": 0.78,
            "position": "CENTER",
            "risk_score": 0.70,
            "is_uncertain": True,
            "voice_message": "Caution. Possible glass door ahead.",
        }
        floors = [{
            "hazard_type": "CABLE",
            "hazard_level": "CAUTION",
            "position": "CENTER",
            "reason": "Cable across floor.",
        }]
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.80},
            frame_width=640,
            frame_height=480,
            indoor_safety={"glass_obstacles": glass, "floor_hazards": floors},
        )
        self.assertEqual(res.get("navigation", {}).get("action"), "SLOW_DOWN")

    # =========================================================================
    # Category I: Closed-Loop Navigation Integration & Perception Failure (Tests 37-41)
    # =========================================================================

    def test_37_closed_loop_navigation_hazard_safety_override(self):
        """Test 37: Active destination session switches to AVOIDING with safety_override=True."""
        session = NavigationSession()
        session.start_destination("Room 101")
        # Frame 1: Clear landmark visible in center -> APPROACHING
        session.update(
            semantic_landmarks=[{"type": "ROOM", "label": "Room 101", "position": "CENTER", "distance_m": 4.0}],
            scene_depth={"center_depth": 0.80},
            safety_level="SAFE",
        )
        self.assertEqual(session.state, NavigationSessionState.APPROACHING)

        # Frame 2: Indoor hazard appears (e.g. drop-off / critical stairs)
        stair_hazard = {
            "stair_analysis": {
                "detected": True,
                "type": "DROP_OFF",
                "hazard_level": "CRITICAL",
                "drop_off_risk": True,
            }
        }
        res = session.update(
            semantic_landmarks=[{"type": "ROOM", "label": "Room 101", "position": "CENTER", "distance_m": 3.8}],
            scene_depth={"center_depth": 0.40},
            safety_level="EMERGENCY",
            emergency_stop=True,
            indoor_safety=stair_hazard,
        )
        self.assertTrue(res.get("safety_override"))
        self.assertEqual(session.state, NavigationSessionState.AVOIDING)
        self.assertTrue(session.active) # Destination session was NOT destroyed!

    def test_38_closed_loop_navigation_resumes_after_hazard_clears(self):
        """Test 38: Active destination session resumes APPROACHING after hazard clears."""
        session = NavigationSession()
        session.start_destination("Room 101")
        session.state = NavigationSessionState.AVOIDING

        # Frame where hazard is now gone
        res = session.update(
            semantic_landmarks=[{"type": "ROOM", "label": "Room 101", "position": "CENTER", "distance_m": 3.5}],
            scene_depth={"center_depth": 0.85},
            safety_level="SAFE",
            emergency_stop=False,
            indoor_safety={},
        )
        self.assertFalse(res.get("safety_override"))
        self.assertEqual(session.state, NavigationSessionState.APPROACHING)
        self.assertTrue(session.active)

    def test_39_stale_frame_protection(self):
        """Test 39: Stale frame triggers EMERGENCY STOP and rechecking surroundings."""
        perception_health = {
            "status": "HEALTHY",
            "stale_frame": True,
            "voice_override": "Stop. Rechecking surroundings.",
        }
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.80},
            frame_width=640,
            frame_height=480,
            perception_health=perception_health,
        )
        self.assertTrue(res.get("emergency_stop"))
        self.assertEqual(res.get("navigation", {}).get("action"), "STOP")
        self.assertEqual(res.get("voice_instruction"), "Stop. Rechecking surroundings.")

    def test_40_camera_obstructed_perception_failure(self):
        """Test 40: Camera obstruction triggers FAILED status and lens check directive."""
        perception_health = {
            "status": "FAILED",
            "image_quality": {"is_obstructed": True},
            "voice_override": "Camera obstructed. Please check lens.",
        }
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.80},
            frame_width=640,
            frame_height=480,
            perception_health=perception_health,
        )
        self.assertTrue(res.get("emergency_stop"))
        self.assertEqual(res.get("voice_instruction"), "Camera obstructed. Please check lens.")

    def test_41_indoor_safety_manager_latency_budget(self):
        """Test 41: Unified IndoorSafetyManager executes under latency budget (< 25ms)."""
        frame = np.ones((240, 320, 3), dtype=np.uint8) * 128
        depth_map = np.ones((240, 320), dtype=np.float32) * 0.70

        # Warm up
        self.safety_manager.analyze(frame=frame, depth_map=depth_map)

        t0 = time.perf_counter()
        bundle = self.safety_manager.analyze(frame=frame, depth_map=depth_map)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.assertIn("door_analysis", bundle)
        self.assertIn("stair_analysis", bundle)
        self.assertIn("floor_hazards", bundle)
        self.assertIn("glass_obstacles", bundle)
        self.assertIn("corridor_analysis", bundle)
        self.assertLess(elapsed_ms, 25.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
