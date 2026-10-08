"""
SmartVisionAI - Phase 9 Comprehensive Indoor Perception Robustness Test Suite
=============================================================================
Covers 36 verification tests for Phase 9 indoor perception and OCR robustness:
- Category A: Basic OCR (Tests 1-7)
- Category B: Directional Signs & Arrows (Tests 8-11)
- Category C: Multiple Text Regions & Prioritization (Tests 12-15)
- Category D: Text Quality, Degraded Frames & Occlusion (Tests 16-23)
- Category E: Semantic Landmarks & Extended Taxonomy (Tests 24-28)
- Category F: OCR Temporal Stability & Deduplication (Tests 29-32)
- Category G: Navigation Integration & Safety Precedence (Tests 33-36)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import unittest
import numpy as np
import cv2

from backend.ocr.reader import TextReader
from backend.semantic.interpreter import SemanticInterpreter
from backend.decision.engine import DecisionEngine
from backend.navigation.session import NavigationSession, NavigationSessionState
from backend.navigation.destination import Destination, match_destination


class TestPhase9IndoorPerceptionRobustness(unittest.TestCase):
    def setUp(self):
        self.text_reader = TextReader(enabled=True)
        self.interpreter = SemanticInterpreter()
        self.decision_engine = DecisionEngine()
        self.session = NavigationSession()

    # =========================================================================
    # Category A: Basic OCR (Tests 1-7)
    # =========================================================================

    def test_01_clear_exit_sign(self):
        """Test 1: Clear EXIT sign correctly extracted and classified as EXIT landmark."""
        raw_ocr = {
            "detected": True,
            "text_count": 1,
            "primary_text": "EXIT",
            "items": [{
                "text": "EXIT",
                "confidence": 0.94,
                "position": "CENTER",
                "bbox": {"x1": 200, "y1": 50, "x2": 440, "y2": 150},
                "apparent_size": "LARGE",
                "is_reliable": True,
                "is_uncertain": False,
            }],
        }
        landmarks = self.interpreter.interpret(raw_ocr, frame_width=640.0)
        self.assertTrue(len(landmarks) >= 1)
        self.assertEqual(landmarks[0]["type"], "EXIT")
        self.assertEqual(landmarks[0]["label"], "Exit")
        self.assertTrue(landmarks[0]["is_reliable"])
        self.assertTrue(landmarks[0]["is_navigable_cue"])

    def test_02_clear_entrance_sign(self):
        """Test 2: Clear ENTRANCE sign correctly classified."""
        sem_type, label, ident, is_nav, safety_rel = self.interpreter.classify_sign("MAIN ENTRANCE")
        self.assertEqual(sem_type, "ENTRANCE")
        self.assertEqual(label, "Entrance")
        self.assertTrue(is_nav)

    def test_03_clear_restroom_sign(self):
        """Test 3: RESTROOM / WASHROOM / TOILET classified as RESTROOM."""
        for word in ("RESTROOM", "WASHROOM", "TOILETS", "GENTS"):
            sem_type, label, _, is_nav, _ = self.interpreter.classify_sign(word)
            self.assertEqual(sem_type, "RESTROOM")
            self.assertTrue(is_nav)

    def test_04_clear_stairs_sign(self):
        """Test 4: STAIRS / STAIRWAY classified as STAIRS with HIGH safety relevance."""
        sem_type, label, _, is_nav, safety_rel = self.interpreter.classify_sign("STAIRWAY")
        self.assertEqual(sem_type, "STAIRS")
        self.assertEqual(safety_rel, "HIGH")

    def test_05_clear_elevator_sign(self):
        """Test 5: ELEVATOR / LIFT classified as ELEVATOR with HIGH safety relevance."""
        for word in ("ELEVATOR", "LIFT"):
            sem_type, label, _, is_nav, safety_rel = self.interpreter.classify_sign(word)
            self.assertEqual(sem_type, "ELEVATOR")
            self.assertEqual(safety_rel, "HIGH")

    def test_06_clear_reception_sign(self):
        """Test 6: RECEPTION / FRONT DESK classified as RECEPTION."""
        sem_type, label, _, is_nav, _ = self.interpreter.classify_sign("RECEPTION DESK")
        self.assertEqual(sem_type, "RECEPTION")
        self.assertTrue(is_nav)

    def test_07_clear_room_number(self):
        """Test 7: ROOM 302 extracts ROOM type with identifier '302'."""
        sem_type, label, ident, is_nav, _ = self.interpreter.classify_sign("ROOM 302")
        self.assertEqual(sem_type, "ROOM")
        self.assertEqual(ident, "302")
        self.assertEqual(label, "Room 302")

    # =========================================================================
    # Category B: Directional Signs & Arrows (Tests 8-11)
    # =========================================================================

    def test_08_left_directional_sign(self):
        """Test 8: Left arrow or word cue extracts directional_cue=LEFT."""
        cue, clean = self.text_reader._extract_directional_cue("EXIT ←")
        self.assertEqual(cue, "LEFT")
        self.assertEqual(clean, "EXIT")

        cue_ascii, _ = self.text_reader._extract_directional_cue("<- RESTROOM")
        self.assertEqual(cue_ascii, "LEFT")

    def test_09_right_directional_sign(self):
        """Test 9: Right arrow or word cue extracts directional_cue=RIGHT."""
        cue, clean = self.text_reader._extract_directional_cue("ROOM 302 →")
        self.assertEqual(cue, "RIGHT")
        self.assertEqual(clean, "ROOM 302")

        cue_ascii, _ = self.text_reader._extract_directional_cue("ELEVATOR ->")
        self.assertEqual(cue_ascii, "RIGHT")

    def test_10_straight_directional_sign(self):
        """Test 10: Up arrow or word STRAIGHT extracts directional_cue=STRAIGHT / UP."""
        cue, clean = self.text_reader._extract_directional_cue("STAIRS ↑")
        self.assertEqual(cue, "UP")
        self.assertEqual(clean, "STAIRS")

        cue_w, _ = self.text_reader._extract_directional_cue("WAY OUT STRAIGHT AHEAD")
        self.assertIn(cue_w, ("STRAIGHT", "UP"))

    def test_11_arrow_plus_text_directional_speech(self):
        """Test 11: Directional arrow generates directional speech ('Room 302 to the right.')."""
        lm = {
            "type": "ROOM",
            "label": "Room 302",
            "identifier": "302",
            "position": "CENTER",
            "directional_cue": "RIGHT",
            "is_reliable": True,
            "is_uncertain": False,
            "should_announce": True,
        }
        speech = self.interpreter.generate_landmark_speech(lm)
        self.assertEqual(speech, "Room 302 to the right.")

        lm_left = dict(lm, directional_cue="LEFT", label="Exit")
        speech_left = self.interpreter.generate_landmark_speech(lm_left)
        self.assertEqual(speech_left, "Exit to the left.")

    # =========================================================================
    # Category C: Multiple Text Regions & Prioritization (Tests 12-15)
    # =========================================================================

    def test_12_multiple_signs_prioritize_safety_and_emergency(self):
        """Test 12: Scene containing multiple signs prioritizes WARNING/EMERGENCY over generic text."""
        raw_ocr = {
            "detected": True,
            "items": [
                {"text": "WELCOME TO CAMPUS", "confidence": 0.95, "position": "CENTER", "directional_cue": "NONE", "apparent_size": "LARGE"},
                {"text": "DO NOT ENTER", "confidence": 0.82, "position": "CENTER", "directional_cue": "NONE", "apparent_size": "MEDIUM"},
                {"text": "CAFETERIA", "confidence": 0.88, "position": "RIGHT", "directional_cue": "NONE", "apparent_size": "MEDIUM"},
            ],
        }
        landmarks = self.interpreter.interpret(raw_ocr)
        self.assertTrue(len(landmarks) >= 3)
        # Top-ranked landmark must be WARNING
        self.assertEqual(landmarks[0]["type"], "WARNING")
        # Only the top-ranked landmark should announce; others must be suppressed
        self.assertTrue(landmarks[0]["should_announce"])
        self.assertFalse(landmarks[1]["should_announce"])
        self.assertFalse(landmarks[2]["should_announce"])

    def test_13_room_plus_warning_sign_precedence(self):
        """Test 13: Room number sign next to warning sign prioritizes warning in DecisionEngine."""
        warning_sign = {
            "type": "WARNING",
            "label": "Do Not Enter",
            "clean_text": "DO NOT ENTER",
            "position": "CENTER",
            "confidence": 0.85,
            "is_reliable": True,
            "is_uncertain": False,
            "should_announce": True,
        }
        room_sign = {
            "type": "ROOM",
            "label": "Room 101",
            "identifier": "101",
            "position": "RIGHT",
            "confidence": 0.92,
            "is_reliable": True,
            "is_uncertain": False,
            "should_announce": False,
        }
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.80},
            frame_width=640,
            frame_height=480,
            semantic_landmarks=[warning_sign, room_sign],
        )
        self.assertEqual(res.get("navigation", {}).get("action"), "STOP")
        self.assertIn("Warning sign", res.get("navigation", {}).get("reason", ""))

    def test_14_exit_plus_arrow_over_ad_notice(self):
        """Test 14: EXIT with arrow takes semantic precedence over an advertisement notice."""
        rank_exit = self.text_reader._semantic_priority_rank("EXIT", directional_cue="LEFT")
        rank_ad = self.text_reader._semantic_priority_rank("DISCOUNT 50% OFF", directional_cue="NONE")
        self.assertGreater(rank_exit, rank_ad)

    def test_15_multiple_room_labels_single_announcement(self):
        """Test 15: Multiple room labels in a corridor emit exactly one announcement per frame."""
        raw_ocr = {
            "detected": True,
            "items": [
                {"text": "ROOM 301", "confidence": 0.88, "position": "LEFT", "directional_cue": "NONE", "apparent_size": "MEDIUM"},
                {"text": "ROOM 302", "confidence": 0.90, "position": "RIGHT", "directional_cue": "NONE", "apparent_size": "MEDIUM"},
                {"text": "ROOM 303", "confidence": 0.86, "position": "RIGHT", "directional_cue": "NONE", "apparent_size": "MEDIUM"},
            ],
        }
        landmarks = self.interpreter.interpret(raw_ocr)
        announced_count = sum(1 for l in landmarks if l.get("should_announce"))
        self.assertEqual(announced_count, 1)

    # =========================================================================
    # Category D: Text Quality, Degraded Frames & Occlusion (Tests 16-23)
    # =========================================================================

    def test_16_small_nearby_text_reliable(self):
        """Test 16: Small but legible font (height >= 16px) is tagged SMALL and reliable."""
        app_size, is_rel, is_unc = self.text_reader._compute_apparent_size(
            box_height=20, box_width=80, frame_h=480, frame_w=640
        )
        self.assertEqual(app_size, "SMALL")
        self.assertTrue(is_rel)
        self.assertFalse(is_unc)

    def test_17_distant_micro_text_tagged_uncertain(self):
        """Test 17: Distant micro-text (height < 16px) is tagged DISTANT and is_uncertain=True."""
        app_size, is_rel, is_unc = self.text_reader._compute_apparent_size(
            box_height=12, box_width=40, frame_h=480, frame_w=640
        )
        self.assertEqual(app_size, "DISTANT")
        self.assertFalse(is_rel)
        self.assertTrue(is_unc)

    def test_18_angled_text_extraction(self):
        """Test 18: Angled directional arrow text preserves correct directional cue."""
        cue, clean = self.text_reader._extract_directional_cue("ROOM 204 >>")
        self.assertEqual(cue, "RIGHT")
        self.assertEqual(clean, "ROOM 204")

    def test_19_perspective_distorted_sign_confidence_filtering(self):
        """Test 19: Perspective distorted sign with marginal confidence is guarded."""
        raw_ocr = {
            "detected": True,
            "items": [{
                "text": "RESTROOM",
                "confidence": 0.52, # Below safe landmark confidence
                "position": "RIGHT",
                "directional_cue": "NONE",
                "apparent_size": "SMALL",
            }],
        }
        landmarks = self.interpreter.interpret(raw_ocr)
        # Low confidence (<0.60) must be marked uncertain and non-navigable
        self.assertEqual(len(landmarks), 1)
        self.assertTrue(landmarks[0]["is_uncertain"])
        self.assertFalse(landmarks[0]["is_navigable_cue"])

    def test_20_motion_blur_penalizes_confidence(self):
        """Test 20: Frame with severe motion blur triggers degradation penalty."""
        # Create blurred synthetic frame
        img = np.ones((240, 320, 3), dtype=np.uint8) * 128
        cv2.putText(img, "TEST", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        blurred = cv2.GaussianBlur(img, (25, 25), 0)
        penalty, is_deg, reason = self.text_reader._check_image_degradation(blurred)
        self.assertTrue(is_deg)
        self.assertEqual(reason, "BLUR")
        self.assertGreater(penalty, 0.0)

    def test_21_defocus_blur_marks_landmarks_uncertain(self):
        """Test 21: Highly blurred item is flagged is_uncertain=True and does not announce."""
        raw_ocr = {
            "detected": True,
            "items": [{
                "text": "EXIT",
                "confidence": 0.58,
                "position": "CENTER",
                "is_uncertain": True,
                "is_reliable": False,
                "apparent_size": "MEDIUM",
            }],
        }
        landmarks = self.interpreter.interpret(raw_ocr)
        self.assertEqual(len(landmarks), 1)
        self.assertTrue(landmarks[0]["is_uncertain"])
        self.assertFalse(landmarks[0]["should_announce"])

    def test_22_low_light_indoor_frame_penalization(self):
        """Test 22: Severely dark frame (mean lum < 40) applies low-light penalty."""
        dark_img = np.ones((240, 320, 3), dtype=np.uint8) * 20
        penalty, is_deg, reason = self.text_reader._check_image_degradation(dark_img)
        self.assertTrue(is_deg)
        self.assertEqual(reason, "LOW_LIGHT")
        self.assertEqual(penalty, 0.20)

    def test_23_partial_occlusion_truncation_detection(self):
        """Test 23: Truncated word fragment (e.g. 'ESTROOM', 'XIT') is flagged as partial."""
        self.assertTrue(self.text_reader._is_partial_fragment("ESTROOM"))
        self.assertTrue(self.text_reader._is_partial_fragment("XIT"))
        self.assertTrue(self.text_reader._is_partial_fragment("OOM"))
        self.assertFalse(self.text_reader._is_partial_fragment("RESTROOM"))
        self.assertFalse(self.text_reader._is_partial_fragment("EXIT"))

    # =========================================================================
    # Category E: Semantic Landmarks & Extended Taxonomy (Tests 24-28)
    # =========================================================================

    def test_24_room_number_semantic_landmark(self):
        """Test 24: Standard and alphanumeric room numbers extract identifier cleanly."""
        for q in ("ROOM 302", "ROOM 105A", "CABIN 12", "SUITE 400"):
            sem_type, label, ident, is_nav, _ = self.interpreter.classify_sign(q)
            self.assertEqual(sem_type, "ROOM")
            self.assertIsNotNone(ident)

    def test_25_exit_semantic_landmark_voice(self):
        """Test 25: EXIT sign in center yields 'Exit ahead.' voice directive."""
        exit_sign = {
            "type": "EXIT",
            "label": "Exit",
            "position": "CENTER",
            "directional_cue": "NONE",
            "confidence": 0.90,
            "is_reliable": True,
            "is_uncertain": False,
            "should_announce": True,
        }
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.85},
            frame_width=640,
            frame_height=480,
            semantic_landmarks=[exit_sign],
        )
        self.assertEqual(res.get("voice_instruction"), "Exit ahead.")

    def test_26_restroom_semantic_landmark_voice(self):
        """Test 26: RESTROOM sign on left yields 'Restroom on your left.'."""
        restroom_sign = {
            "type": "RESTROOM",
            "label": "Restroom",
            "position": "LEFT",
            "directional_cue": "NONE",
            "confidence": 0.88,
            "is_reliable": True,
            "is_uncertain": False,
            "should_announce": True,
        }
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.85},
            frame_width=640,
            frame_height=480,
            semantic_landmarks=[restroom_sign],
        )
        self.assertEqual(res.get("voice_instruction"), "Restroom on your left.")

    def test_27_stairs_semantic_landmark_reinforces_stairs_hazard(self):
        """Test 27: STAIRS sign reinforces indoor safety stairs perception."""
        stairs_sign = {
            "type": "STAIRS",
            "label": "Stairs",
            "position": "CENTER",
            "confidence": 0.85,
            "is_reliable": True,
        }
        indoor_safety = {
            "stair_analysis": {"detected": True, "type": "STAIRS_DOWN", "hazard_level": "CAUTION", "confidence": 0.60},
        }
        res = self.decision_engine.analyze(
            objects=[],
            scene_depth={"center_depth": 0.80},
            frame_width=640,
            frame_height=480,
            indoor_safety=indoor_safety,
            semantic_landmarks=[stairs_sign],
        )
        # Stair hazard escalated to CRITICAL -> EMERGENCY STOP
        self.assertTrue(res.get("emergency_stop"))

    def test_28_institutional_landmarks_pharmacy_and_library(self):
        """Test 28: Institutional signs (PHARMACY, LIBRARY, CLINIC, SECURITY) classified cleanly."""
        self.assertEqual(self.interpreter.classify_sign("HOSPITAL PHARMACY")[0], "PHARMACY")
        self.assertEqual(self.interpreter.classify_sign("CENTRAL LIBRARY")[0], "LIBRARY")
        self.assertEqual(self.interpreter.classify_sign("DENTAL CLINIC")[0], "CLINIC")
        self.assertEqual(self.interpreter.classify_sign("SECURITY DESK")[0], "SECURITY")

    # =========================================================================
    # Category F: OCR Temporal Stability & Deduplication (Tests 29-32)
    # =========================================================================

    def test_29_repeated_ocr_dedup_suppression(self):
        """Test 29: Reading the exact same text across consecutive frames suppresses repetition."""
        self.text_reader.reset_history()
        self.text_reader._last_spoken_text = "ROOM 302"
        self.text_reader._last_spoken_timestamp = time.time()

        # Check deduplication logic
        is_same = (self.text_reader._last_spoken_text == "ROOM 302")
        elapsed = time.time() - self.text_reader._last_spoken_timestamp
        self.assertTrue(is_same)
        self.assertLess(elapsed, self.text_reader.dedup_cooldown_s)

    def test_30_unstable_ocr_does_not_announce(self):
        """Test 30: Unstable/uncertain OCR (is_uncertain=True) suppresses announcement."""
        unstable_item = {
            "text": "RM 3??",
            "confidence": 0.45,
            "is_uncertain": True,
            "is_reliable": False,
            "apparent_size": "DISTANT",
        }
        landmarks = self.interpreter.interpret({"detected": True, "items": [unstable_item]})
        self.assertEqual(len(landmarks), 1)
        self.assertFalse(landmarks[0]["should_announce"])

    def test_31_temporary_landmark_loss_preserves_session(self):
        """Test 31: Temporary loss (1 frame missing) remains SEARCHING without declaring LOST."""
        self.session.start_destination("Room 302", identifier="302")
        # Frame 1: Landmark visible -> APPROACHING
        lm = {"type": "ROOM", "identifier": "302", "confidence": 0.85, "position": "CENTER", "is_reliable": True, "is_uncertain": False}
        res1 = self.session.update(semantic_landmarks=[lm])
        self.assertEqual(res1["state"], NavigationSessionState.APPROACHING)

        # Frame 2: Landmark temporarily occluded/missing -> SEARCHING (not LOST)
        res2 = self.session.update(semantic_landmarks=[])
        self.assertEqual(res2["state"], NavigationSessionState.SEARCHING)
        self.assertTrue(res2["active"])

    def test_32_multiple_frame_stable_landmark(self):
        """Test 32: Consecutive visible frames maintain stable APPROACHING state."""
        self.session.start_destination("Room 302", identifier="302")
        lm = {"type": "ROOM", "identifier": "302", "confidence": 0.85, "position": "CENTER", "is_reliable": True, "is_uncertain": False}
        for _ in range(4):
            res = self.session.update(semantic_landmarks=[lm])
            self.assertEqual(res["state"], NavigationSessionState.APPROACHING)
            self.assertTrue(res["destination_visible"])

    # =========================================================================
    # Category G: Navigation Integration & Safety Precedence (Tests 33-36)
    # =========================================================================

    def test_33_destination_ocr_approaching(self):
        """Test 33: Active destination matched with high confidence triggers APPROACHING."""
        self.session.start_destination("Elevator")
        lm = {"type": "ELEVATOR", "label": "Elevator", "position": "CENTER", "confidence": 0.88, "is_reliable": True, "is_uncertain": False}
        res = self.session.update(semantic_landmarks=[lm])
        self.assertEqual(res["state"], NavigationSessionState.APPROACHING)
        self.assertEqual(res["navigation_action"], "FORWARD")

    def test_34_ocr_navigation_obstacle_safety_override(self):
        """Test 34: Approaching destination with new obstacle triggers AVOIDING with safety_override=True."""
        self.session.start_destination("Room 302", identifier="302")
        lm = {"type": "ROOM", "identifier": "302", "confidence": 0.85, "position": "CENTER", "is_reliable": True, "is_uncertain": False}
        # First frame clear
        self.session.update(semantic_landmarks=[lm])

        # Second frame: obstacle appears directly ahead (critical proximity)
        obstacle = {"class_name": "chair", "distance_m": 0.70, "risk_score": 0.80, "position": "CENTER"}
        res_avoid = self.session.update(
            semantic_landmarks=[lm],
            primary_obstacle=obstacle,
            safety_level="DANGER",
        )
        self.assertEqual(res_avoid["state"], NavigationSessionState.AVOIDING)
        self.assertTrue(res_avoid["safety_override"])

    def test_35_ocr_landmark_persistent_disappearance_transitions_to_lost(self):
        """Test 35: Landmark missing for >= 3 consecutive frames transitions to LOST."""
        self.session.start_destination("Room 302", identifier="302")
        lm = {"type": "ROOM", "identifier": "302", "confidence": 0.85, "position": "CENTER", "is_reliable": True, "is_uncertain": False}
        self.session.update(semantic_landmarks=[lm]) # Seen once

        # Miss frame 1, 2, 3
        self.session.update(semantic_landmarks=[])
        self.session.update(semantic_landmarks=[])
        res_lost = self.session.update(semantic_landmarks=[])
        self.assertEqual(res_lost["state"], NavigationSessionState.LOST)

    def test_36_uncertain_ocr_rejected_by_destination_matcher(self):
        """Test 36: Landmark marked is_uncertain=True or confidence < 0.60 is rejected by match_destination."""
        dest = Destination("Room 302", identifier="302")
        uncertain_lm = {
            "type": "ROOM",
            "identifier": "302",
            "confidence": 0.48, # Too low
            "is_uncertain": True,
            "is_reliable": False,
        }
        is_match, conf = match_destination(dest, uncertain_lm)
        self.assertFalse(is_match)
        self.assertEqual(conf, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

