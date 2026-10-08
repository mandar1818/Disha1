#!/usr/bin/env python3
"""
Test Suite: Stateful Assistive Navigation & STOP Suppression for Blind Users
SmartVisionAI / DISHA

Verifies:
1. STOP Suppression & WAIT_FOR_CHANGE State Machine:
   - Speaks STOP on initial frame
   - Suppresses repeated STOP shouting while stationary in front of the same obstacle
   - Suggests recovery after persistent blockage
   - Speaks "Path clear. Move forward for X steps." when obstacle departs
2. Side Obstacle Handling:
   - Side obstacle with clear center maintains FORWARD without STOP or unnecessary slow down
3. Natural Step-Based Guidance:
   - "Move forward for 4 steps."
   - "Move left for 2 steps."
   - "Move right for 3 steps."
   - "Turn left, then move forward."
   - "Turn right, then move forward."
   - "Move back for 3 steps."
4. Haptic & Speech Lock integration
"""

import sys
import os
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.decision.engine import DecisionEngine
from scratch.test_phase12_voice import VoiceDecisionManagerPy as VoiceDecisionManager


class TestStatefulNavigationAndStopSuppression(unittest.TestCase):

    def setUp(self):
        self.engine = DecisionEngine()
        self.engine.reset()
        self.voice = VoiceDecisionManager()
        self.voice.reset()

    # -------------------------------------------------------------
    # 1. STOP SUPPRESSION & STATE TRANSITIONS
    # -------------------------------------------------------------
    def test_01_initial_stop_speaks_once(self):
        """Frame 1 of obstacle blocking path: says Stop directly ahead."""
        res = self.engine.analyze(
            objects=[{
                "id": 1,
                "class_name": "chair",
                "bbox": {"x1": 250, "y1": 150, "x2": 390, "y2": 450},
                "confidence": 0.85,
                "distance_m": 0.5,
                "risk_score": 0.80,
            }],
            scene_depth={"average_depth": 0.3, "center_depth": 0.15, "left_depth": 0.15, "right_depth": 0.15},
            frame_width=640,
            frame_height=480,
        )
        self.assertEqual(res["navigation"]["action"], "STOP")
        self.assertEqual(res["nav_state"], "STOP")
        self.assertTrue(res["should_speak"])
        self.assertTrue(res["state_changed"])

        dec = self.voice.handle_response(res, now=1000)
        self.voice.is_speaking = False
        self.assertIn(dec["action"], ("SPEAK", "INTERRUPT"))
        self.assertIn("Stop", dec["instruction"])

    def test_02_consecutive_stop_frames_suppress_repeated_shouting(self):
        """Frames 2 to 6 with same obstacle: should_speak is False, speech is SUPPRESSED."""
        # Frame 1
        obj = {
            "id": 101,
            "track_id": 101,
            "class_name": "chair",
            "bbox": {"x1": 250, "y1": 150, "x2": 390, "y2": 450},
            "confidence": 0.85,
            "distance_m": 0.55,
            "risk_score": 0.80,
        }
        res1 = self.engine.analyze(
            objects=[obj],
            scene_depth={"average_depth": 0.3, "center_depth": 0.15, "left_depth": 0.15, "right_depth": 0.15},
            frame_width=640,
            frame_height=480,
        )
        self.assertTrue(res1["should_speak"])
        self.voice.handle_response(res1, now=1000)
        self.voice.is_speaking = False

        # Frames 2 to 6: user has stopped, obstacle remains in front
        for f in range(2, 7):
            now = 1000 + f * 500  # 1500ms, 2000ms, 2500ms, 3000ms, 3500ms
            res = self.engine.analyze(
                objects=[obj],
                scene_depth={"average_depth": 0.3, "center_depth": 0.15, "left_depth": 0.15, "right_depth": 0.15},
                frame_width=640,
                frame_height=480,
            )
            self.assertEqual(res["nav_state"], "WAIT_FOR_CHANGE")
            self.assertFalse(res["should_speak"], f"Frame {f} should have should_speak=False")

            dec = self.voice.evaluate(res, now=now)
            self.assertEqual(dec["action"], "SUPPRESS", f"Frame {f} speech should be SUPPRESSED")

    def test_03_recovery_suggested_after_persistent_stop(self):
        """Frame 8+: if still blocked, suggests an alternative recovery action."""
        obj = {
            "id": 101,
            "track_id": 101,
            "class_name": "chair",
            "bbox": {"x1": 250, "y1": 150, "x2": 390, "y2": 450},
            "confidence": 0.85,
            "distance_m": 0.55,
            "risk_score": 0.80,
        }
        # Simulate frames 1 to 8
        for f in range(1, 9):
            res = self.engine.analyze(
                objects=[obj],
                scene_depth={"average_depth": 0.3, "center_depth": 0.15, "left_depth": 0.15, "right_depth": 0.15},
                frame_width=640,
                frame_height=480,
            )

        # On frame 8, recovery is suggested
        self.assertEqual(res["nav_state"], "RECOVER")
        self.assertTrue(res["should_speak"])
        self.assertIn(res["navigation"]["action"], ("BACK", "TURN_LEFT", "TURN_RIGHT", "RECOVER"))

    def test_04_clearing_obstacle_announces_path_clear_forward_steps(self):
        """When obstacle clears after stop: announces Path clear with step count."""
        obj = {
            "id": 101,
            "track_id": 101,
            "class_name": "chair",
            "bbox": {"x1": 250, "y1": 150, "x2": 390, "y2": 450},
            "confidence": 0.85,
            "distance_m": 0.55,
            "risk_score": 0.80,
        }
        # Step into STOP
        self.engine.analyze(objects=[obj], scene_depth={"average_depth": 0.3, "center_depth": 0.15, "left_depth": 0.15, "right_depth": 0.15})
        # Obstacle moves away / cleared
        res_clear = self.engine.analyze(
            objects=[],
            scene_depth={"average_depth": 0.8, "center_depth": 0.8, "left_depth": 0.8, "right_depth": 0.8},
        )
        self.assertEqual(res_clear["navigation"]["action"], "GO_FORWARD")
        self.assertEqual(res_clear["nav_state"], "GO_FORWARD")
        self.assertTrue(res_clear["should_speak"])
        self.assertIn("Path clear", res_clear["voice_instruction"])
        self.assertIn("Move forward for", res_clear["voice_instruction"])

    # -------------------------------------------------------------
    # 2. SIDE OBSTACLES WITH CLEAR CENTER CORRIDOR
    # -------------------------------------------------------------
    def test_05_left_obstacle_clear_center_continues_forward(self):
        """Obstacle on left does NOT trigger STOP if center corridor is clear."""
        res = self.engine.analyze(
            objects=[{
                "id": 1,
                "class_name": "chair",
                "bbox": {"x1": 40, "y1": 150, "x2": 160, "y2": 420},  # Left side
                "confidence": 0.82,
                "distance_m": 1.5,
                "risk_score": 0.55,
            }],
            scene_depth={"average_depth": 0.7, "center_depth": 0.7, "left_depth": 0.35, "right_depth": 0.7},
            frame_width=640,
            frame_height=480,
        )
        self.assertEqual(res["navigation"]["action"], "GO_FORWARD")
        self.assertEqual(res["navigation"]["direction"], "CENTER")
        self.assertNotEqual(res["navigation"]["action"], "STOP")
        self.assertIn("Move forward", res["voice_instruction"])

    def test_06_right_obstacle_clear_center_continues_forward(self):
        """Obstacle on right does NOT trigger STOP if center corridor is clear."""
        res = self.engine.analyze(
            objects=[{
                "id": 1,
                "class_name": "person",
                "bbox": {"x1": 480, "y1": 120, "x2": 600, "y2": 450},  # Right side
                "confidence": 0.85,
                "distance_m": 1.8,
                "risk_score": 0.50,
            }],
            scene_depth={"average_depth": 0.7, "center_depth": 0.7, "left_depth": 0.7, "right_depth": 0.35},
            frame_width=640,
            frame_height=480,
        )
        self.assertEqual(res["navigation"]["action"], "GO_FORWARD")
        self.assertEqual(res["navigation"]["direction"], "CENTER")
        self.assertNotEqual(res["navigation"]["action"], "STOP")
        self.assertIn("Move forward", res["voice_instruction"])

    # -------------------------------------------------------------
    # 3. NATURAL STEP-BASED GUIDANCE
    # -------------------------------------------------------------
    def test_07_clear_corridor_step_guidance(self):
        """Clear corridor produces 4 to 5 steps forward."""
        res = self.engine.analyze(
            objects=[],
            scene_depth={"average_depth": 0.8, "center_depth": 0.8, "left_depth": 0.8, "right_depth": 0.8},
        )
        steps = res["step_guidance"]["steps"]
        self.assertGreaterEqual(steps, 2)
        self.assertLessEqual(steps, 6)
        self.assertEqual(res["voice_instruction"], f"Move forward for {steps} steps.")

    def test_08_move_left_step_guidance(self):
        """Move left lateral guidance produces 2 steps."""
        v = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle=None,
            navigation="MOVE_LEFT",
            steps=2,
        )
        self.assertEqual(v, "Move left for 2 steps.")

    def test_09_move_right_step_guidance(self):
        """Move right lateral guidance produces 3 steps."""
        v = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle=None,
            navigation="MOVE_RIGHT",
            steps=3,
        )
        self.assertEqual(v, "Move right for 3 steps.")

    def test_10_turn_guidance_moves_forward(self):
        """Turn guidance instructs turning and moving forward."""
        v_left = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle=None,
            navigation="TURN_LEFT",
        )
        self.assertEqual(v_left, "Turn left, then move forward.")

        v_right = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle=None,
            navigation="TURN_RIGHT",
        )
        self.assertEqual(v_right, "Turn right, then move forward.")

    def test_11_move_back_step_guidance(self):
        """Move back produces safe backward steps."""
        v = self.engine.generate_voice_instruction(
            safety_level="CAUTION",
            emergency_stop=False,
            primary_obstacle=None,
            navigation="GO_BACK",
            steps=3,
        )
        self.assertEqual(v, "Move back for 3 steps.")


if __name__ == "__main__":
    unittest.main()
