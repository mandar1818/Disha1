#!/usr/bin/env python3
"""
Test Suite: Object Detection Voice, Spatial Position (Center/Left/Right), and Next-Step Action Guidance
Verifies that when an object is detected:
1. Object name is voiced (e.g., Chair, Person, Table, Bottle)
2. Spatial position is identified and voiced (center/ahead, on the left, on the right)
3. Actionable next step is voiced (Stop, Slow down, Move left, Move right, Turn left, Turn right, Move forward)
"""

import sys
import os
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.decision.engine import DecisionEngine


class TestObjectVoicePositionActions(unittest.TestCase):

    def setUp(self):
        self.engine = DecisionEngine()
        self.engine.reset()

    # -----------------------------------------------------------------
    # 1. POSITION CALCULATION (CENTER, LEFT, RIGHT)
    # -----------------------------------------------------------------
    def test_01_position_classification_center(self):
        """Bounding box in horizontal middle is classified as CENTER."""
        bbox = {"x1": 250, "y1": 150, "x2": 390, "y2": 450}
        pos = self.engine.get_position(bbox, 640.0)
        self.assertEqual(pos, "CENTER")

    def test_02_position_classification_left(self):
        """Bounding box in left third is classified as LEFT."""
        bbox = {"x1": 20, "y1": 150, "x2": 150, "y2": 450}
        pos = self.engine.get_position(bbox, 640.0)
        self.assertEqual(pos, "LEFT")

    def test_03_position_classification_right(self):
        """Bounding box in right third is classified as RIGHT."""
        bbox = {"x1": 480, "y1": 150, "x2": 620, "y2": 450}
        pos = self.engine.get_position(bbox, 640.0)
        self.assertEqual(pos, "RIGHT")

    # -----------------------------------------------------------------
    # 2. ACTION: STOP WITH OBJECT & POSITION
    # -----------------------------------------------------------------
    def test_04_stop_center_object(self):
        """Center obstacle in path announces Stop + object + ahead."""
        voice = self.engine.generate_voice_instruction(
            safety_level="EMERGENCY",
            emergency_stop=True,
            primary_obstacle={"class_name": "chair", "position": "CENTER", "distance_m": 0.5},
            navigation="STOP",
        )
        self.assertEqual(voice, "Stop. Chair ahead.")

    def test_05_stop_left_object(self):
        """Left obstacle in path announces Stop + object + on your left."""
        voice = self.engine.generate_voice_instruction(
            safety_level="EMERGENCY",
            emergency_stop=True,
            primary_obstacle={"class_name": "person", "position": "LEFT", "distance_m": 0.5},
            navigation="STOP",
        )
        self.assertEqual(voice, "Stop. Person on your left.")

    def test_06_stop_right_object(self):
        """Right obstacle in path announces Stop + object + on your right."""
        voice = self.engine.generate_voice_instruction(
            safety_level="EMERGENCY",
            emergency_stop=True,
            primary_obstacle={"class_name": "table", "position": "RIGHT", "distance_m": 0.5},
            navigation="STOP",
        )
        self.assertEqual(voice, "Stop. Table on your right.")

    # -----------------------------------------------------------------
    # 3. ACTION: SLOW DOWN WITH OBJECT & POSITION
    # -----------------------------------------------------------------
    def test_07_slow_down_center_object(self):
        """Approaching center obstacle announces Slow down + object + ahead."""
        voice = self.engine.generate_voice_instruction(
            safety_level="CAUTION",
            emergency_stop=False,
            primary_obstacle={"class_name": "chair", "position": "CENTER", "distance_m": 1.5},
            navigation="SLOW_DOWN",
        )
        self.assertEqual(voice, "Slow down. Chair ahead.")

    def test_08_slow_down_left_object(self):
        """Caution obstacle on left announces Slow down + object + on the left."""
        voice = self.engine.generate_voice_instruction(
            safety_level="CAUTION",
            emergency_stop=False,
            primary_obstacle={"class_name": "person", "position": "LEFT", "distance_m": 1.5},
            navigation="SLOW_DOWN",
        )
        self.assertEqual(voice, "Slow down. Person on the left.")

    def test_09_slow_down_right_object(self):
        """Caution obstacle on right announces Slow down + object + on the right."""
        voice = self.engine.generate_voice_instruction(
            safety_level="CAUTION",
            emergency_stop=False,
            primary_obstacle={"class_name": "bottle", "position": "RIGHT", "distance_m": 1.2},
            navigation="SLOW_DOWN",
        )
        self.assertEqual(voice, "Slow down. Bottle on the right.")

    # -----------------------------------------------------------------
    # 4. ACTION: LATERAL ADJUSTMENT (MOVE LEFT / MOVE RIGHT)
    # -----------------------------------------------------------------
    def test_10_move_left_object_on_right(self):
        """Obstacle on right with left detour announces Move left + object + on the right."""
        voice = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle={"class_name": "chair", "position": "RIGHT"},
            navigation="MOVE_LEFT",
            steps=2,
        )
        self.assertEqual(voice, "Move left. Chair on the right.")

    def test_11_move_right_object_on_left(self):
        """Obstacle on left with right detour announces Move right + object + on the left."""
        voice = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle={"class_name": "person", "position": "LEFT"},
            navigation="MOVE_RIGHT",
            steps=3,
        )
        self.assertEqual(voice, "Move right. Person on the left.")

    def test_12_move_left_center_obstacle(self):
        """Center obstacle with left detour announces object ahead + Move left."""
        voice = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle={"class_name": "table", "position": "CENTER"},
            navigation="MOVE_LEFT",
            steps=2,
        )
        self.assertEqual(voice, "Table ahead. Move left.")

    def test_13_move_right_center_obstacle(self):
        """Center obstacle with right detour announces object ahead + Move right."""
        voice = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle={"class_name": "chair", "position": "CENTER"},
            navigation="MOVE_RIGHT",
            steps=3,
        )
        self.assertEqual(voice, "Chair ahead. Move right.")

    # -----------------------------------------------------------------
    # 5. ACTION: BODY ROTATION (TURN LEFT / TURN RIGHT)
    # -----------------------------------------------------------------
    def test_14_turn_left_center_obstacle(self):
        """Close center obstacle requiring turn announces object ahead + Turn left."""
        voice = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle={"class_name": "person", "position": "CENTER"},
            navigation="TURN_LEFT",
        )
        self.assertEqual(voice, "Person ahead. Turn left.")

    def test_15_turn_right_center_obstacle(self):
        """Close center obstacle requiring turn announces object ahead + Turn right."""
        voice = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle={"class_name": "chair", "position": "CENTER"},
            navigation="TURN_RIGHT",
        )
        self.assertEqual(voice, "Chair ahead. Turn right.")

    def test_16_turn_left_right_obstacle(self):
        """Obstacle on right requiring left turn announces object on the right + Turn left."""
        voice = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle={"class_name": "table", "position": "RIGHT"},
            navigation="TURN_LEFT",
        )
        self.assertEqual(voice, "Table on the right. Turn left.")

    def test_17_turn_right_left_obstacle(self):
        """Obstacle on left requiring right turn announces object on the left + Turn right."""
        voice = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle={"class_name": "person", "position": "LEFT"},
            navigation="TURN_RIGHT",
        )
        self.assertEqual(voice, "Person on the left. Turn right.")

    # -----------------------------------------------------------------
    # 6. ACTION: FORWARD WALKING WITH DETECTED OBJECTS IN SCENE
    # -----------------------------------------------------------------
    def test_18_forward_with_left_object(self):
        """Safe walking with object on left announces object + on the left + Move forward."""
        voice = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle={"class_name": "chair", "position": "LEFT", "distance_m": 2.5},
            navigation="FORWARD",
            steps=4,
        )
        self.assertEqual(voice, "Chair on the left. Move forward.")

    def test_19_forward_with_right_object(self):
        """Safe walking with object on right announces object + on the right + Move forward."""
        voice = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle={"class_name": "bottle", "position": "RIGHT", "distance_m": 2.8},
            navigation="FORWARD",
            steps=4,
        )
        self.assertEqual(voice, "Bottle on the right. Move forward.")

    def test_20_forward_with_center_distant_object(self):
        """Safe walking with distant center object announces object ahead + Move forward."""
        voice = self.engine.generate_voice_instruction(
            safety_level="SAFE",
            emergency_stop=False,
            primary_obstacle={"class_name": "person", "position": "CENTER", "distance_m": 3.5},
            navigation="FORWARD",
            steps=4,
        )
        self.assertEqual(voice, "Person ahead. Move forward.")

    # -----------------------------------------------------------------
    # 7. END-TO-END ANALYZE INTEGRATION
    # -----------------------------------------------------------------
    def test_21_end_to_end_analyze_populates_position_and_voice(self):
        """Engine.analyze populates position (CENTER, LEFT, RIGHT) and structured voice instruction."""
        res = self.engine.analyze(
            objects=[
                {
                    "id": 1,
                    "class_name": "chair",
                    "bbox": {"x1": 50, "y1": 150, "x2": 180, "y2": 450},  # LEFT
                    "distance_m": 2.2,
                    "confidence": 0.88,
                },
                {
                    "id": 2,
                    "class_name": "person",
                    "bbox": {"x1": 490, "y1": 120, "x2": 610, "y2": 460},  # RIGHT
                    "distance_m": 2.5,
                    "confidence": 0.90,
                }
            ],
            scene_depth={"average_depth": 0.7, "center_depth": 0.7, "left_depth": 0.4, "right_depth": 0.4},
            frame_width=640,
            frame_height=480,
        )
        # Verify detected objects have position
        objs = res.get("objects", [])
        self.assertEqual(len(objs), 2)
        pos_by_class = {o["class_name"]: o["position"] for o in objs}
        self.assertEqual(pos_by_class["chair"], "LEFT")
        self.assertEqual(pos_by_class["person"], "RIGHT")

        # Verify voice instruction includes object, position, and next action
        v_inst = res.get("voice_instruction", "")
        self.assertTrue(
            "on the left" in v_inst or "on the right" in v_inst or "ahead" in v_inst,
            f"Voice instruction '{v_inst}' must contain spatial position phrase"
        )
        self.assertTrue(
            "Move forward" in v_inst or "Turn" in v_inst or "Stop" in v_inst,
            f"Voice instruction '{v_inst}' must contain action directive"
        )


if __name__ == "__main__":
    unittest.main()

