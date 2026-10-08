import sys, os
sys.path.insert(0, os.path.abspath("."))
import cv2
import numpy as np
from backend.processors.frame_processor import FrameProcessor
from backend.decision.engine import DecisionEngine

fp = FrameProcessor()
engine = DecisionEngine()

print("=" * 70)
print("SCENARIO 1: Real Indoor Scene with Person Ahead (backend/test.jpg)")
print("=" * 70)
img = cv2.imread("backend/test.jpg")
res = fp.process(img)
print("Safety level:       ", res.get("safety_level"))
print("Emergency stop:     ", res.get("emergency_stop"))
nav = res.get("navigation")
action = nav.get("action") if isinstance(nav, dict) else nav
print("Navigation action:  ", action)
print("Voice instruction:  ", repr(res.get("voice_instruction")))
po = res.get("primary_obstacle")
if po:
    dist = po.get("distance_m", 0)
    print(f"Primary obstacle:    {po.get('class_name')} at {dist:.2f}m ({po.get('position')})")
fp_info = res.get("free_path", {})
print(f"Corridor Clearance:  Left={fp_info.get('left_clear')} | Center={fp_info.get('center_clear')} | Right={fp_info.get('right_clear')}")

print("\n" + "=" * 70)
print("SCENARIO 2: Distant Person at 3.0m Ahead (Center)")
print("=" * 70)
engine.reset()
distant_person = {
    "class_name": "person",
    "distance_m": 3.0,
    "position": "CENTER",
    "risk_score": 0.15,
    "confidence": 0.90,
    "bbox": {"x1": 250, "y1": 100, "x2": 390, "y2": 400},
}
res_distant = engine.analyze(
    objects=[distant_person],
    scene_depth={"average_depth": 0.8, "center_depth": 0.8, "left_depth": 0.85, "right_depth": 0.85},
    frame_width=640,
    frame_height=480,
)
print("Safety level:       ", res_distant.get("safety_level"))
print("Emergency stop:     ", res_distant.get("emergency_stop"))
nav_dist = res_distant.get("navigation")
act_dist = nav_dist.get("action") if isinstance(nav_dist, dict) else nav_dist
print("Navigation action:  ", act_dist)
print("Voice instruction:  ", repr(res_distant.get("voice_instruction")))

print("\n" + "=" * 70)
print("SCENARIO 3: Obstacle on the Right at 1.0m (Center Path Clear)")
print("=" * 70)
engine.reset()
side_obstacle = {
    "class_name": "person",
    "distance_m": 1.0,
    "position": "RIGHT",
    "risk_score": 0.35,
    "confidence": 0.88,
    "bbox": {"x1": 460, "y1": 120, "x2": 580, "y2": 450},
}
res_side = engine.analyze(
    objects=[side_obstacle],
    scene_depth={"average_depth": 0.7, "center_depth": 0.75, "left_depth": 0.75, "right_depth": 0.40},
    frame_width=640,
    frame_height=480,
)
print("Safety level:       ", res_side.get("safety_level"))
print("Emergency stop:     ", res_side.get("emergency_stop"))
nav_side = res_side.get("navigation")
act_side = nav_side.get("action") if isinstance(nav_side, dict) else nav_side
print("Navigation action:  ", act_side)
print("Voice instruction:  ", repr(res_side.get("voice_instruction")))

print("\n" + "=" * 70)
print("SCENARIO 4: Imminent Collision Hazard at 0.35m in Center")
print("=" * 70)
engine.reset()
critical_obstacle = {
    "class_name": "person",
    "distance_m": 0.35,
    "position": "CENTER",
    "risk_score": 0.88,
    "confidence": 0.95,
    "bbox": {"x1": 150, "y1": 50, "x2": 490, "y2": 470},
}
res_crit = engine.analyze(
    objects=[critical_obstacle],
    scene_depth={"average_depth": 0.15, "center_depth": 0.10, "left_depth": 0.15, "right_depth": 0.15},
    frame_width=640,
    frame_height=480,
)
print("Safety level:       ", res_crit.get("safety_level"))
print("Emergency stop:     ", res_crit.get("emergency_stop"))
nav_crit = res_crit.get("navigation")
act_crit = nav_crit.get("action") if isinstance(nav_crit, dict) else nav_crit
print("Navigation action:  ", act_crit)
print("Voice instruction:  ", repr(res_crit.get("voice_instruction")))

print("\n" + "=" * 70)
print("SCENARIO 6: Unknown Obstacle Ahead at 1.2m (Center Path)")
print("=" * 70)
engine.reset()
unknown_obs = {
    "class_name": "unknown_floor_obstacle",
    "distance_m": 1.2,
    "position": "CENTER",
    "risk_score": 0.65,
    "confidence": 0.85,
    "bbox": {"x1": 220, "y1": 200, "x2": 420, "y2": 450},
}
res_unk = engine.analyze(
    objects=[],
    unknown_objects=[unknown_obs],
    scene_depth={"average_depth": 0.6, "center_depth": 0.5, "left_depth": 0.75, "right_depth": 0.3},
    frame_width=640,
    frame_height=480,
)
print("Safety level:       ", res_unk.get("safety_level"))
print("Emergency stop:     ", res_unk.get("emergency_stop"))
nav_unk = res_unk.get("navigation")
act_unk = nav_unk.get("action") if isinstance(nav_unk, dict) else nav_unk
print("Navigation action:  ", act_unk)
print("Voice instruction:  ", repr(res_unk.get("voice_instruction")))
po_unk = res_unk.get("primary_obstacle")
if po_unk:
    print(f"Primary obstacle:    {po_unk.get('class_name')} at {po_unk.get('distance_m')}m")

print("\n" + "=" * 70)
print("SCENARIO 7: Multi-Object Priority (Person at 1.2m vs Cup at 0.9m)")
print("=" * 70)
engine.reset()
cup_obj = {
    "class_name": "cup",
    "distance_m": 0.9,
    "position": "CENTER",
    "risk_score": 0.55,
    "confidence": 0.92,
    "bbox": {"x1": 280, "y1": 280, "x2": 330, "y2": 340},
}
person_obj = {
    "class_name": "person",
    "distance_m": 1.2,
    "position": "CENTER",
    "risk_score": 0.70,
    "confidence": 0.95,
    "bbox": {"x1": 250, "y1": 100, "x2": 390, "y2": 450},
}
res_multi = engine.analyze(
    objects=[cup_obj, person_obj],
    scene_depth={"average_depth": 0.5, "center_depth": 0.45, "left_depth": 0.75, "right_depth": 0.75},
    frame_width=640,
    frame_height=480,
)
po_multi = res_multi.get("primary_obstacle")
print(f"Primary obstacle:    {po_multi.get('class_name')} at {po_multi.get('distance_m')}m (Expected: person)")
nav_multi = res_multi.get("navigation")
act_multi = nav_multi.get("action") if isinstance(nav_multi, dict) else nav_multi
print("Navigation action:  ", act_multi)
print("Voice instruction:  ", repr(res_multi.get("voice_instruction")))
