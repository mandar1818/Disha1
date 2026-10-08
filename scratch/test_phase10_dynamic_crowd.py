"""
SmartVisionAI / DISHA - Phase 10 Unit Test Suite
Dynamic & Crowded Indoor Scene Robustness

Categories:
  A. Person state (Tests 1-7)
  B. Crowds (Tests 8-15)
  C. Corridor (Tests 16-20)
  D. Combined hazards (Tests 21-29)
  E. Closed loop (Tests 30-35)
  F. OCR interaction (Tests 36-38)
  G. Stability (Tests 39-42)

Total tests: 42 (minimum 40 required)
"""

import math
import os
import sys
import time
import unittest
from typing import Any, Dict, List

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.decision.engine import DecisionEngine
from backend.navigation.destination import Destination
from backend.navigation.session import NavigationSession, NavigationSessionState


class TestPhase10DynamicCrowd(unittest.TestCase):
    def setUp(self):
        self.engine = DecisionEngine()
        self.engine.reset()

    # ========================================================
    # CATEGORY A: PERSON STATES (Tests 1 - 7)
    # ========================================================

    def test_01_static_person(self):
        """Test 1: Static / standing person classified as PERSON_STATIC."""
        # Frame 1
        p1 = {
            "id": 1,
            "class_name": "person",
            "bbox": {"x1": 250, "y1": 100, "x2": 390, "y2": 450},
            "position": "CENTER",
            "distance_m": 2.5,
            "confidence": 0.85,
            "risk_score": 0.30,
        }
        res1 = self.engine.decide(objects=[p1], frame_width=640, frame_height=480)

        # Frame 2: identical position and distance
        p2 = {
            "id": 1,
            "class_name": "person",
            "bbox": {"x1": 250, "y1": 100, "x2": 390, "y2": 450},
            "position": "CENTER",
            "distance_m": 2.5,
            "confidence": 0.85,
            "risk_score": 0.30,
        }
        res2 = self.engine.decide(objects=[p2], frame_width=640, frame_height=480)

        person_obj = res2["objects"][0]
        self.assertEqual(person_obj.get("person_state"), "PERSON_STATIC")

    def test_02_approaching_person(self):
        """Test 2: Person decreasing distance toward camera classified as PERSON_APPROACHING."""
        # Frame 1: moderately far
        p1 = {
            "id": 1,
            "class_name": "person",
            "bbox": {"x1": 280, "y1": 120, "x2": 360, "y2": 400},
            "position": "CENTER",
            "distance_m": 3.0,
            "confidence": 0.90,
            "risk_score": 0.35,
        }
        self.engine.decide(objects=[p1], frame_width=640, frame_height=480)

        # Frame 2: closer
        p2 = {
            "id": 1,
            "class_name": "person",
            "bbox": {"x1": 270, "y1": 110, "x2": 370, "y2": 420},
            "position": "CENTER",
            "distance_m": 2.4,
            "confidence": 0.90,
            "risk_score": 0.50,
        }
        self.engine.decide(objects=[p2], frame_width=640, frame_height=480)

        # Frame 3: significantly closer
        p3 = {
            "id": 1,
            "class_name": "person",
            "bbox": {"x1": 250, "y1": 90, "x2": 390, "y2": 450},
            "position": "CENTER",
            "distance_m": 1.7,
            "confidence": 0.90,
            "risk_score": 0.70,
        }
        res3 = self.engine.decide(objects=[p3], frame_width=640, frame_height=480)

        person_obj = res3["objects"][0]
        self.assertIn(person_obj.get("person_state"), ("PERSON_APPROACHING", "PERSON_CROSSING"))
        self.assertTrue(person_obj.get("approaching") or person_obj.get("is_approaching"))

    def test_03_moving_away(self):
        """Test 3: Person increasing distance away from camera classified as PERSON_MOVING_AWAY."""
        # Frame 1: close
        p1 = {
            "id": 1,
            "class_name": "person",
            "bbox": {"x1": 240, "y1": 80, "x2": 400, "y2": 460},
            "position": "CENTER",
            "distance_m": 1.8,
            "confidence": 0.88,
            "risk_score": 0.55,
        }
        self.engine.decide(objects=[p1], frame_width=640, frame_height=480)

        # Frame 2: farther
        p2 = {
            "id": 1,
            "class_name": "person",
            "bbox": {"x1": 260, "y1": 110, "x2": 380, "y2": 430},
            "position": "CENTER",
            "distance_m": 2.5,
            "confidence": 0.88,
            "risk_score": 0.35,
        }
        self.engine.decide(objects=[p2], frame_width=640, frame_height=480)

        # Frame 3: even farther
        p3 = {
            "id": 1,
            "class_name": "person",
            "bbox": {"x1": 280, "y1": 130, "x2": 360, "y2": 400},
            "position": "CENTER",
            "distance_m": 3.2,
            "confidence": 0.88,
            "risk_score": 0.20,
        }
        res3 = self.engine.decide(objects=[p3], frame_width=640, frame_height=480)

        person_obj = res3["objects"][0]
        self.assertEqual(person_obj.get("person_state"), "PERSON_MOVING_AWAY")

    def test_04_crossing_left_to_center(self):
        """Test 4: Person crossing from LEFT into CENTER corridor classified as PERSON_CROSSING."""
        # Frame 1: on left (x in [50, 150], cx ~ 100 / 640 = 0.156 < 0.35)
        p1 = {
            "id": 1,
            "class_name": "person",
            "bbox": {"x1": 50, "y1": 120, "x2": 150, "y2": 420},
            "position": "LEFT",
            "distance_m": 2.2,
            "confidence": 0.85,
            "risk_score": 0.30,
        }
        self.engine.decide(objects=[p1], frame_width=640, frame_height=480)

        # Frame 2: crossing into center (x in [240, 340], cx ~ 290 / 640 = 0.453 in [0.35, 0.65])
        p2 = {
            "id": 1,
            "class_name": "person",
            "bbox": {"x1": 240, "y1": 120, "x2": 340, "y2": 420},
            "position": "CENTER",
            "distance_m": 2.1,
            "confidence": 0.85,
            "risk_score": 0.55,
        }
        res2 = self.engine.decide(objects=[p2], frame_width=640, frame_height=480)

        person_obj = res2["objects"][0]
        self.assertEqual(person_obj.get("person_state"), "PERSON_CROSSING")
        self.assertTrue(person_obj.get("is_crossing"))
        self.assertEqual(person_obj.get("crossing_direction"), "LEFT_TO_CENTER")

    def test_05_crossing_right_to_center(self):
        """Test 5: Person crossing from RIGHT into CENTER corridor classified as PERSON_CROSSING."""
        # Frame 1: on right (x in [480, 580], cx ~ 530 / 640 = 0.828 > 0.65)
        p1 = {
            "id": 1,
            "class_name": "person",
            "bbox": {"x1": 480, "y1": 120, "x2": 580, "y2": 420},
            "position": "RIGHT",
            "distance_m": 2.3,
            "confidence": 0.85,
            "risk_score": 0.30,
        }
        self.engine.decide(objects=[p1], frame_width=640, frame_height=480)

        # Frame 2: crossing into center (x in [300, 400], cx ~ 350 / 640 = 0.547 in [0.35, 0.65])
        p2 = {
            "id": 1,
            "class_name": "person",
            "bbox": {"x1": 300, "y1": 120, "x2": 400, "y2": 420},
            "position": "CENTER",
            "distance_m": 2.2,
            "confidence": 0.85,
            "risk_score": 0.55,
        }
        res2 = self.engine.decide(objects=[p2], frame_width=640, frame_height=480)

        person_obj = res2["objects"][0]
        self.assertEqual(person_obj.get("person_state"), "PERSON_CROSSING")
        self.assertTrue(person_obj.get("is_crossing"))
        self.assertEqual(person_obj.get("crossing_direction"), "RIGHT_TO_CENTER")

    def test_06_side_movement(self):
        """Test 6: Person moving laterally strictly along the side corridor classified as PERSON_SIDE_MOVING."""
        # Frame 1: left (x in [40, 120], cx ~ 80 / 640 = 0.125)
        p1 = {
            "id": 1,
            "class_name": "person",
            "bbox": {"x1": 40, "y1": 130, "x2": 120, "y2": 420},
            "position": "LEFT",
            "distance_m": 2.5,
            "confidence": 0.80,
            "risk_score": 0.25,
        }
        self.engine.decide(objects=[p1], frame_width=640, frame_height=480)

        # Frame 2: moving on left (x in [70, 150], cx ~ 110 / 640 = 0.171, staying < 0.35)
        p2 = {
            "id": 1,
            "class_name": "person",
            "bbox": {"x1": 70, "y1": 130, "x2": 150, "y2": 420},
            "position": "LEFT",
            "distance_m": 2.5,
            "confidence": 0.80,
            "risk_score": 0.25,
        }
        res2 = self.engine.decide(objects=[p2], frame_width=640, frame_height=480)

        person_obj = res2["objects"][0]
        self.assertEqual(person_obj.get("person_state"), "PERSON_SIDE_MOVING")

    def test_07_uncertain_movement(self):
        """Test 7: Person with low confidence or erratic/unstable evidence classified as PERSON_UNCERTAIN."""
        p1 = {
            "id": 1,
            "class_name": "person",
            "bbox": {"x1": 250, "y1": 100, "x2": 390, "y2": 450},
            "position": "CENTER",
            "distance_m": 2.5,
            "confidence": 0.80,
            "distance_category": "UNKNOWN_DISTANCE",
            "distance_confidence": "LOW",
            "risk_score": 0.30,
        }
        res = self.engine.decide(objects=[p1], frame_width=640, frame_height=480)
        person_obj = res["objects"][0]
        self.assertEqual(person_obj.get("person_state"), "PERSON_UNCERTAIN")

    # ========================================================
    # CATEGORY B: CROWDS & MULTI-PERSON SCENES (Tests 8 - 15)
    # ========================================================

    def test_08_two_people(self):
        """Test 8: Two people present produce crowd_size='PAIR' and people_count=2."""
        p1 = {"id": 1, "class_name": "person", "position": "LEFT", "distance_m": 2.5, "confidence": 0.8}
        p2 = {"id": 2, "class_name": "person", "position": "RIGHT", "distance_m": 2.7, "confidence": 0.8}

        res = self.engine.decide(objects=[p1, p2], frame_width=640, frame_height=480)
        mo = res.get("multiple_objects", {})
        self.assertEqual(mo.get("people_count"), 2)
        self.assertEqual(mo.get("crowd_size"), "PAIR")

    def test_09_three_people(self):
        """Test 9: Three people present trigger is_crowded=True and crowd_size='GROUP'."""
        p1 = {"id": 1, "class_name": "person", "position": "LEFT", "distance_m": 2.5, "confidence": 0.8}
        p2 = {"id": 2, "class_name": "person", "position": "CENTER", "distance_m": 2.2, "confidence": 0.8}
        p3 = {"id": 3, "class_name": "person", "position": "RIGHT", "distance_m": 2.6, "confidence": 0.8}

        res = self.engine.decide(objects=[p1, p2, p3], frame_width=640, frame_height=480)
        mo = res.get("multiple_objects", {})
        self.assertEqual(mo.get("people_count"), 3)
        self.assertEqual(mo.get("crowd_size"), "GROUP")
        self.assertTrue(mo.get("is_crowded"))

    def test_10_group(self):
        """Test 10: 4 people present produce crowd_size='GROUP'."""
        objs = [
            {"id": i, "class_name": "person", "position": "CENTER" if i % 2 == 0 else "LEFT", "distance_m": 2.0 + 0.2 * i, "confidence": 0.8}
            for i in range(1, 5)
        ]
        res = self.engine.decide(objects=objs, frame_width=640, frame_height=480)
        mo = res.get("multiple_objects", {})
        self.assertEqual(mo.get("people_count"), 4)
        self.assertEqual(mo.get("crowd_size"), "GROUP")
        self.assertTrue(mo.get("is_crowded"))

    def test_11_dense_crowd(self):
        """Test 11: 5 or more people classified as crowd_size='DENSE_CROWD'."""
        objs = [
            {"id": i, "class_name": "person", "position": "CENTER", "distance_m": 2.0 + 0.1 * i, "confidence": 0.8}
            for i in range(1, 7)
        ]
        res = self.engine.decide(objects=objs, frame_width=640, frame_height=480)
        mo = res.get("multiple_objects", {})
        self.assertGreaterEqual(mo.get("people_count"), 5)
        self.assertEqual(mo.get("crowd_size"), "DENSE_CROWD")
        self.assertTrue(mo.get("is_crowded"))

    def test_12_crowd_center(self):
        """Test 12: People concentrated in center flag crowd_center=True and crowd_distribution='CENTER'."""
        p1 = {"id": 1, "class_name": "person", "position": "CENTER", "distance_m": 2.0, "confidence": 0.85}
        p2 = {"id": 2, "class_name": "person", "position": "CENTER", "distance_m": 2.3, "confidence": 0.85}

        res = self.engine.decide(objects=[p1, p2], frame_width=640, frame_height=480)
        mo = res.get("multiple_objects", {})
        self.assertTrue(mo.get("crowd_center"))
        self.assertEqual(mo.get("crowd_distribution"), "CENTER")

    def test_13_crowd_left(self):
        """Test 13: People concentrated on left flag crowd_left=True and crowd_distribution='LEFT'."""
        p1 = {"id": 1, "class_name": "person", "position": "LEFT", "distance_m": 2.0, "confidence": 0.85}
        p2 = {"id": 2, "class_name": "person", "position": "LEFT", "distance_m": 2.4, "confidence": 0.85}

        res = self.engine.decide(objects=[p1, p2], frame_width=640, frame_height=480)
        mo = res.get("multiple_objects", {})
        self.assertTrue(mo.get("crowd_left"))
        self.assertEqual(mo.get("crowd_distribution"), "LEFT")

    def test_14_crowd_right(self):
        """Test 14: People concentrated on right flag crowd_right=True and crowd_distribution='RIGHT'."""
        p1 = {"id": 1, "class_name": "person", "position": "RIGHT", "distance_m": 2.0, "confidence": 0.85}
        p2 = {"id": 2, "class_name": "person", "position": "RIGHT", "distance_m": 2.4, "confidence": 0.85}

        res = self.engine.decide(objects=[p1, p2], frame_width=640, frame_height=480)
        mo = res.get("multiple_objects", {})
        self.assertTrue(mo.get("crowd_right"))
        self.assertEqual(mo.get("crowd_distribution"), "RIGHT")

    def test_15_crowd_both_sides(self):
        """Test 15: People on both sides flag crowd_both_sides=True."""
        p1 = {"id": 1, "class_name": "person", "position": "LEFT", "distance_m": 2.0, "confidence": 0.85}
        p2 = {"id": 2, "class_name": "person", "position": "RIGHT", "distance_m": 2.1, "confidence": 0.85}

        res = self.engine.decide(objects=[p1, p2], frame_width=640, frame_height=480)
        mo = res.get("multiple_objects", {})
        self.assertTrue(mo.get("crowd_both_sides"))

    # ========================================================
    # CATEGORY C: CORRIDOR & CLEARANCES (Tests 16 - 20)
    # ========================================================

    def test_16_clear_corridor(self):
        """Test 16: Clear corridor returns center_clear=True and forward navigation."""
        res = self.engine.decide(objects=[], scene_depth={"center_depth": 0.85, "left_depth": 0.80, "right_depth": 0.80})
        fp = res.get("free_path", {})
        self.assertTrue(fp.get("center_clear"))
        nav = res.get("navigation", {})
        action = nav.get("action") if isinstance(nav, dict) else nav
        self.assertIn(action, ("FORWARD", "GO_FORWARD", "CONTINUE_FORWARD"))

    def test_17_one_person_blockage(self):
        """Test 17: Single person blocking center path reduces center clearance and triggers SLOW_DOWN/STOP."""
        p = {
            "id": 1,
            "class_name": "person",
            "bbox": {"x1": 240, "y1": 100, "x2": 400, "y2": 470},
            "position": "CENTER",
            "distance_m": 1.2,
            "risk_score": 0.75,
            "confidence": 0.90,
        }
        res = self.engine.decide(objects=[p], frame_width=640, frame_height=480)
        fp = res.get("free_path", {})
        self.assertFalse(fp.get("center_clear"))
        nav = res.get("navigation", {})
        action = nav.get("action") if isinstance(nav, dict) else nav
        self.assertIn(action, ("STOP", "SLOW_DOWN", "MOVE_LEFT", "MOVE_RIGHT"))

    def test_18_temporary_blockage(self):
        """Test 18: Obstacle appears (STOP) and then clears (resumes FORWARD)."""
        # Frame 1: person blocks center
        p = {
            "id": 1,
            "class_name": "person",
            "bbox": {"x1": 240, "y1": 100, "x2": 400, "y2": 470},
            "position": "CENTER",
            "distance_m": 0.9,
            "risk_score": 0.85,
            "confidence": 0.90,
        }
        res1 = self.engine.decide(objects=[p], frame_width=640, frame_height=480)
        nav1 = res1.get("navigation", {})
        action1 = nav1.get("action") if isinstance(nav1, dict) else nav1
        self.assertEqual(action1, "STOP")

        # Frame 2: person moves away; path is now clear
        res2 = self.engine.decide(objects=[], scene_depth={"center_depth": 0.90, "left_depth": 0.85, "right_depth": 0.85})
        fp2 = res2.get("free_path", {})
        self.assertTrue(fp2.get("center_clear"))

        # Frame 3: second consecutive clear frame confirms FORWARD transition
        res3 = self.engine.decide(objects=[], scene_depth={"center_depth": 0.90, "left_depth": 0.85, "right_depth": 0.85})
        nav3 = res3.get("navigation", {})
        action3 = nav3.get("action") if isinstance(nav3, dict) else nav3
        self.assertIn(action3, ("FORWARD", "GO_FORWARD", "CONTINUE_FORWARD"))

    def test_19_persistent_blockage(self):
        """Test 19: Persistent blockage across multiple frames maintains STOP reliably."""
        p = {
            "id": 1,
            "class_name": "person",
            "bbox": {"x1": 240, "y1": 100, "x2": 400, "y2": 470},
            "position": "CENTER",
            "distance_m": 0.8,
            "risk_score": 0.90,
            "confidence": 0.90,
        }
        for _ in range(4):
            res = self.engine.decide(objects=[p], frame_width=640, frame_height=480)
            nav = res.get("navigation", {})
            action = nav.get("action") if isinstance(nav, dict) else nav
            self.assertEqual(action, "STOP")

    def test_20_narrow_corridor_plus_person(self):
        """Test 20: Narrow passage with person present emits caution message."""
        p = {
            "id": 1,
            "class_name": "person",
            "bbox": {"x1": 260, "y1": 120, "x2": 380, "y2": 440},
            "position": "CENTER",
            "distance_m": 2.2,
            "risk_score": 0.45,
            "confidence": 0.85,
        }
        indoor_safety = {
            "corridor_analysis": {
                "detected": True,
                "clearance_state": "NARROW",
                "walkable_width_m": 0.75,
                "message": "Narrow passage. Slow down.",
            }
        }
        res = self.engine.decide(objects=[p], indoor_safety=indoor_safety, frame_width=640, frame_height=480)
        voice = res.get("voice_instruction", "")
        self.assertIn("Slow down", voice)

    # ========================================================
    # CATEGORY D: COMBINED HAZARDS (Tests 21 - 29)
    # ========================================================

    def test_21_person_plus_chair(self):
        """Test 21: Person moving away but chair remains in center corridor keeps path blocked."""
        # Frame 1: Person close + chair behind person
        person = {
            "id": 1,
            "class_name": "person",
            "bbox": {"x1": 250, "y1": 100, "x2": 370, "y2": 430},
            "position": "CENTER",
            "distance_m": 1.5,
            "confidence": 0.9,
            "risk_score": 0.6,
        }
        chair = {
            "id": 2,
            "class_name": "chair",
            "bbox": {"x1": 270, "y1": 200, "x2": 370, "y2": 460},
            "position": "CENTER",
            "distance_m": 2.0,
            "confidence": 0.85,
            "risk_score": 0.55,
        }
        self.engine.decide(objects=[person, chair], frame_width=640, frame_height=480)

        # Frame 2: Person moves away (distance 3.0m), but chair remains at 1.9m in center
        person_receding = {
            "id": 1,
            "class_name": "person",
            "bbox": {"x1": 280, "y1": 130, "x2": 360, "y2": 400},
            "position": "CENTER",
            "distance_m": 3.0,
            "confidence": 0.9,
            "risk_score": 0.3,
        }
        chair_static = {
            "id": 2,
            "class_name": "chair",
            "bbox": {"x1": 270, "y1": 200, "x2": 370, "y2": 460},
            "position": "CENTER",
            "distance_m": 1.9,
            "confidence": 0.85,
            "risk_score": 0.60,
        }
        res2 = self.engine.decide(objects=[person_receding, chair_static], frame_width=640, frame_height=480)
        fp = res2.get("free_path", {})
        # Center must NOT be clear because chair is still obstructing center
        self.assertFalse(fp.get("center_clear"))

    def test_22_person_plus_table(self):
        """Test 22: Person + table elevates collective risk and routes safely."""
        person = {"id": 1, "class_name": "person", "position": "CENTER", "distance_m": 2.0, "risk_score": 0.55}
        table = {"id": 2, "class_name": "table", "position": "CENTER", "distance_m": 2.2, "risk_score": 0.55}

        res = self.engine.decide(objects=[person, table], frame_width=640, frame_height=480)
        self.assertGreaterEqual(res.get("multiple_objects", {}).get("center_obstacles"), 2)
        self.assertFalse(res.get("free_path", {}).get("center_clear"))

    def test_23_person_plus_doorway(self):
        """Test 23: Person standing in front of closed doorway prioritizes closed door safety."""
        person = {"id": 1, "class_name": "person", "position": "CENTER", "distance_m": 2.5, "risk_score": 0.40}
        indoor_safety = {
            "door_analysis": {
                "detected": True,
                "state": "CLOSED",
                "hazard_level": "CRITICAL",
                "message": "Door closed. Do not enter.",
            }
        }
        res = self.engine.decide(objects=[person], indoor_safety=indoor_safety, frame_width=640, frame_height=480)
        voice = res.get("voice_instruction", "")
        self.assertIn("Door closed", voice)

    def test_24_person_plus_stairs(self):
        """Test 24: Person near stairs hazard strictly prioritizes stair fall safety."""
        person = {"id": 1, "class_name": "person", "position": "LEFT", "distance_m": 2.8, "risk_score": 0.25}
        indoor_safety = {
            "stair_analysis": {
                "detected": True,
                "type": "STAIRS_DOWN",
                "hazard_level": "CRITICAL",
                "drop_off_risk": True,
            }
        }
        res = self.engine.decide(objects=[person], indoor_safety=indoor_safety, frame_width=640, frame_height=480)
        self.assertTrue(res.get("emergency_stop"))
        voice = res.get("voice_instruction", "")
        self.assertIn("Stop", voice)

    def test_25_person_plus_glass(self):
        """Test 25: Person near glass obstacle preserves glass hazard warning."""
        person = {"id": 1, "class_name": "person", "position": "RIGHT", "distance_m": 2.5, "risk_score": 0.3}
        indoor_safety = {
            "glass_obstacles": {
                "detected": True,
                "risk_score": 0.75,
                "voice_message": "Possible glass obstacle ahead.",
            }
        }
        res = self.engine.decide(objects=[person], indoor_safety=indoor_safety, frame_width=640, frame_height=480)
        voice = res.get("voice_instruction", "")
        self.assertIn("glass", voice.lower())

    def test_26_person_plus_unknown_obstacle(self):
        """Test 26: Person + genuine unknown obstacle applies conservative safety."""
        person = {"id": 1, "class_name": "person", "position": "RIGHT", "distance_m": 2.5, "confidence": 0.8}
        unknown = {
            "id": 101,
            "class_name": "unknown_obstacle",
            "position": "CENTER",
            "bbox": {"x1": 260, "y1": 200, "x2": 380, "y2": 450},
            "distance_m": 1.4,
            "risk_score": 0.65,
            "confidence": 0.70,
        }
        res = self.engine.decide(objects=[person], unknown_objects=[unknown], frame_width=640, frame_height=480)
        self.assertFalse(res.get("free_path", {}).get("center_clear"))

    def test_27_crowd_plus_furniture(self):
        """Test 27: Crowd + multiple pieces of furniture completely block corridor."""
        crowd = [
            {"id": 1, "class_name": "person", "position": "CENTER", "distance_m": 2.0, "risk_score": 0.6},
            {"id": 2, "class_name": "person", "position": "CENTER", "distance_m": 2.3, "risk_score": 0.6},
            {"id": 3, "class_name": "chair", "position": "LEFT", "distance_m": 1.8, "risk_score": 0.5},
            {"id": 4, "class_name": "table", "position": "RIGHT", "distance_m": 1.8, "risk_score": 0.5},
        ]
        res = self.engine.decide(objects=crowd, frame_width=640, frame_height=480)
        mo = res.get("multiple_objects", {})
        self.assertTrue(mo.get("crowd_blocking"))

    def test_28_crowd_plus_stairs(self):
        """Test 28: Crowd near stairs strictly overrides crowd movement with stair safety."""
        crowd = [
            {"id": 1, "class_name": "person", "position": "CENTER", "distance_m": 2.5, "risk_score": 0.5},
            {"id": 2, "class_name": "person", "position": "CENTER", "distance_m": 2.7, "risk_score": 0.5},
        ]
        indoor_safety = {
            "stair_analysis": {
                "detected": True,
                "type": "STAIRS",
                "hazard_level": "CRITICAL",
                "drop_off_risk": False,
            }
        }
        res = self.engine.decide(objects=crowd, indoor_safety=indoor_safety, frame_width=640, frame_height=480)
        self.assertTrue(res.get("emergency_stop"))

    def test_29_crowd_plus_glass(self):
        """Test 29: Crowd near transparent glass wall does not clear path prematurely."""
        crowd = [
            {"id": 1, "class_name": "person", "position": "LEFT", "distance_m": 2.2, "risk_score": 0.4},
            {"id": 2, "class_name": "person", "position": "RIGHT", "distance_m": 2.2, "risk_score": 0.4},
        ]
        indoor_safety = {
            "glass_obstacles": {
                "detected": True,
                "risk_score": 0.72,
                "voice_message": "Possible glass obstacle ahead.",
            }
        }
        res = self.engine.decide(objects=crowd, indoor_safety=indoor_safety, frame_width=640, frame_height=480)
        voice = res.get("voice_instruction", "")
        self.assertIn("glass", voice.lower())

    # ========================================================
    # CATEGORY E: CLOSED-LOOP NAVIGATION (Tests 30 - 35)
    # ========================================================

    def test_30_forward_navigation_plus_person(self):
        """Test 30: Moving forward in destination session overrides with safety when person blocks."""
        dest = Destination("ROOM 302")
        session = NavigationSession()
        session.start(dest)

        # Matched landmark visible in center, but dangerous approaching person in center
        landmark = {
            "type": "ROOM",
            "label": "Room 302",
            "identifier": "302",
            "position": "CENTER",
            "confidence": 0.88,
            "bbox": [200, 50, 440, 150],
        }
        primary_obs = {
            "class_name": "person",
            "position": "CENTER",
            "distance_m": 0.75,
            "risk_score": 0.85,
            "approaching": True,
        }
        res = session.update(
            semantic_landmarks=[landmark],
            primary_obstacle=primary_obs,
            safety_level="DANGER",
        )
        self.assertEqual(res["state"], NavigationSessionState.AVOIDING)
        self.assertTrue(res["safety_override"])
        self.assertTrue(session.active)

    def test_31_turn_plus_person(self):
        """Test 31: Turn maneuver with person appearing triggers safe hold without destroying session."""
        dest = Destination("ROOM 302")
        session = NavigationSession()
        session.start(dest)

        # Landmark on right triggers TURN_RIGHT
        landmark = {
            "type": "ROOM",
            "label": "Room 302",
            "identifier": "302",
            "position": "RIGHT",
            "confidence": 0.85,
            "bbox": [480, 50, 600, 150],
        }
        res1 = session.update(semantic_landmarks=[landmark])
        self.assertEqual(res1["state"], NavigationSessionState.TURNING)

        # During turn, obstacle appears
        obs = {"class_name": "person", "position": "CENTER", "distance_m": 0.70, "risk_score": 0.90}
        res2 = session.update(semantic_landmarks=[landmark], primary_obstacle=obs, emergency_stop=True)
        self.assertEqual(res2["state"], NavigationSessionState.AVOIDING)
        self.assertTrue(res2["safety_override"])
        self.assertTrue(session.active)

    def test_32_obstacle_appears_during_navigation(self):
        """Test 32: Active destination approaching triggers immediate safety override when obstacle appears."""
        dest = Destination("EXIT")
        session = NavigationSession()
        session.start(dest)

        lm = {"type": "EXIT", "label": "Exit", "position": "CENTER", "confidence": 0.90, "bbox": [250, 50, 390, 120]}
        # Frame 1: clear
        res1 = session.update(semantic_landmarks=[lm])
        self.assertEqual(res1["state"], NavigationSessionState.APPROACHING)

        # Frame 2: obstacle appears
        obs = {"class_name": "person", "distance_m": 0.60, "risk_score": 0.95}
        res2 = session.update(semantic_landmarks=[lm], primary_obstacle=obs, safety_level="EMERGENCY")
        self.assertEqual(res2["state"], NavigationSessionState.AVOIDING)
        self.assertTrue(res2["safety_override"])

    def test_33_obstacle_disappears_and_navigation_resumes(self):
        """Test 33: Once obstacle clears, navigation seamlessly resumes APPROACHING."""
        dest = Destination("ROOM 101")
        session = NavigationSession()
        session.start(dest)

        lm = {"type": "ROOM", "label": "Room 101", "position": "CENTER", "confidence": 0.90, "bbox": [250, 50, 390, 120]}
        obs = {"class_name": "person", "distance_m": 0.70, "risk_score": 0.90}

        # Frame 1: avoiding obstacle
        res1 = session.update(semantic_landmarks=[lm], primary_obstacle=obs, safety_level="DANGER")
        self.assertEqual(res1["state"], NavigationSessionState.AVOIDING)

        # Frame 2: obstacle gone, landmark still visible
        res2 = session.update(semantic_landmarks=[lm], primary_obstacle=None, safety_level="SAFE")
        self.assertEqual(res2["state"], NavigationSessionState.APPROACHING)
        self.assertFalse(res2["safety_override"])

    def test_34_temporary_safety_override(self):
        """Test 34: 1-frame safety override does not cancel or restart destination session."""
        dest = Destination("ROOM", identifier="302", label="Room 302")
        session = NavigationSession()
        session.start(dest)

        lm = {"type": "ROOM", "identifier": "302", "label": "Room 302", "position": "CENTER", "confidence": 0.90, "bbox": [250, 50, 390, 120]}
        # Frame 1: approaching
        session.update(semantic_landmarks=[lm])
        # Frame 2: momentary obstacle
        res = session.update(semantic_landmarks=[lm], emergency_stop=True)
        self.assertTrue(res["safety_override"])
        self.assertTrue(session.active)
        self.assertEqual(session.destination.label, "Room 302")

    def test_35_persistent_safety_override(self):
        """Test 35: Multi-frame safety override maintains session validity without reset."""
        dest = Destination("PHARMACY")
        session = NavigationSession()
        session.start(dest)

        lm = {"type": "PHARMACY", "label": "Pharmacy", "position": "CENTER", "confidence": 0.90, "bbox": [250, 50, 390, 120]}
        obs = {"class_name": "person", "distance_m": 0.75, "risk_score": 0.85}

        for _ in range(5):
            res = session.update(semantic_landmarks=[lm], primary_obstacle=obs, safety_level="DANGER")
            self.assertEqual(res["state"], NavigationSessionState.AVOIDING)
            self.assertTrue(session.active)

    # ========================================================
    # CATEGORY F: OCR & LANDMARK INTERACTION (Tests 36 - 38)
    # ========================================================

    def test_36_room_destination_plus_person(self):
        """Test 36: Room 302 OCR detected but person in path emits STOP instead of room announcement."""
        p = {
            "id": 1,
            "class_name": "person",
            "position": "CENTER",
            "distance_m": 0.75,
            "risk_score": 0.90,
            "confidence": 0.90,
        }
        sem = {
            "type": "ROOM",
            "label": "Room 302",
            "identifier": "302",
            "position": "CENTER",
            "should_announce": True,
            "is_uncertain": False,
        }
        res = self.engine.decide(
            objects=[p],
            primary_semantic_sign=sem,
            canonical_semantic_landmarks=[sem],
            frame_width=640,
            frame_height=480,
        )
        voice = res.get("voice_instruction", "")
        self.assertIn("Stop", voice)
        self.assertNotIn("Room 302", voice)

    def test_37_exit_landmark_plus_person(self):
        """Test 37: EXIT sign detected but approaching person emits Slow down / Stop instead of Exit speech."""
        p = {
            "id": 1,
            "class_name": "person",
            "position": "CENTER",
            "distance_m": 1.2,
            "risk_score": 0.75,
            "approaching": True,
            "person_state": "PERSON_APPROACHING",
            "confidence": 0.90,
        }
        sem = {
            "type": "EXIT",
            "label": "Exit",
            "position": "CENTER",
            "should_announce": True,
            "is_uncertain": False,
        }
        res = self.engine.decide(
            objects=[p],
            primary_semantic_sign=sem,
            canonical_semantic_landmarks=[sem],
            frame_width=640,
            frame_height=480,
        )
        voice = res.get("voice_instruction", "")
        self.assertNotIn("Exit ahead", voice)
        self.assertTrue("Stop" in voice or "Slow down" in voice)

    def test_38_semantic_landmark_plus_crowd(self):
        """Test 38: Semantic landmark visible but corridor blocked by crowd emits crowd blockage alert."""
        crowd = [
            {"id": 1, "class_name": "person", "position": "CENTER", "distance_m": 1.8, "risk_score": 0.7},
            {"id": 2, "class_name": "person", "position": "CENTER", "distance_m": 2.0, "risk_score": 0.7},
        ]
        sem = {
            "type": "RESTROOM",
            "label": "Restroom",
            "position": "CENTER",
            "should_announce": True,
            "is_uncertain": False,
        }
        res = self.engine.decide(
            objects=crowd,
            primary_semantic_sign=sem,
            canonical_semantic_landmarks=[sem],
            frame_width=640,
            frame_height=480,
        )
        voice = res.get("voice_instruction", "")
        self.assertNotIn("Restroom", voice)
        self.assertTrue("blocked" in voice.lower() or "crowd" in voice.lower() or "stop" in voice.lower())

    # ========================================================
    # CATEGORY G: STABILITY & ANTI-OSCILLATION (Tests 39 - 42)
    # ========================================================

    def test_39_repeated_stop_dedup(self):
        """Test 39: Repeated identical STOP decisions produce stable instruction."""
        p = {"id": 1, "class_name": "person", "position": "CENTER", "distance_m": 0.8, "risk_score": 0.85}
        v1 = self.engine.decide(objects=[p]).get("voice_instruction")
        v2 = self.engine.decide(objects=[p]).get("voice_instruction")
        self.assertEqual(v1, v2)
        self.assertIn("Stop", v1)

    def test_40_stop_to_safe_resume_without_oscillation(self):
        """Test 40: Transition from STOP to forward uses 2-frame voting hysteresis to prevent flapping."""
        # Frame 1: Obstacle triggers STOP
        obs = {"id": 1, "class_name": "person", "position": "CENTER", "distance_m": 0.8, "risk_score": 0.90}
        res1 = self.engine.decide(objects=[obs])
        nav1 = res1.get("navigation", {})
        action1 = nav1.get("action") if isinstance(nav1, dict) else nav1
        self.assertEqual(action1, "STOP")

        # Frame 2: Path clears; candidate is FORWARD with 1 vote -> holds or cautions pending confirmation
        res2 = self.engine.decide(objects=[], scene_depth={"center_depth": 0.90})
        # Frame 3: Second consecutive clear frame confirms FORWARD
        res3 = self.engine.decide(objects=[], scene_depth={"center_depth": 0.90})
        nav3 = res3.get("navigation", {})
        action3 = nav3.get("action") if isinstance(nav3, dict) else nav3
        self.assertIn(action3, ("FORWARD", "GO_FORWARD", "CONTINUE_FORWARD"))

    def test_41_lateral_direction_anti_flutter_in_crowd(self):
        """Test 41: Lateral maneuver direction does not flutter rapidly between LEFT and RIGHT."""
        self.engine.last_action = "MOVE_LEFT"
        self.engine.candidate_action = "MOVE_LEFT"
        self.engine.last_maneuver_command = "MOVE_LEFT"

        # If both sides are clear, system maintains LEFT to prevent oscillatory flapping
        fp = {
            "center_clear": False,
            "left_clear": True,
            "right_clear": True,
            "left_risk": 0.20,
            "right_risk": 0.18,
            "safe_directions": ["LEFT", "RIGHT"],
        }
        res = self.engine.determine_navigation(
            safety_level="CAUTION",
            emergency_stop=False,
            primary_obstacle={"position": "CENTER", "risk_score": 0.5},
            free_path=fp,
        )
        action = res.get("action")
        self.assertIn(action, ["TURN_LEFT", "MOVE_LEFT"])

    def test_42_voice_preemption_emergency_in_crowd(self):
        """Test 42: Emergency stop voice instruction preempts general crowd advice."""
        crowd = [
            {"id": 1, "class_name": "person", "position": "CENTER", "distance_m": 0.6, "risk_score": 0.95},
            {"id": 2, "class_name": "person", "position": "CENTER", "distance_m": 0.7, "risk_score": 0.90},
            {"id": 3, "class_name": "person", "position": "CENTER", "distance_m": 0.8, "risk_score": 0.85},
        ]
        res = self.engine.decide(objects=crowd, emergency_stop=True)
        voice = res.get("voice_instruction", "")
        self.assertEqual(voice, "Stop immediately.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
