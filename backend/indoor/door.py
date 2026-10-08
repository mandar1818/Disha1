"""
SmartVisionAI - Door & Doorway Analyzer
=======================================
Analyzes indoor doorways and doors:
- Open doorway (walkable, deeper interior depth, clear opening)
- Closed door (planar close depth, non-walkable barrier)
- Blocked doorway (doorframe detected with obstacle inside)

Uses lightweight vision geometry (Canny + Hough) and relative depth analysis.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


class DoorwayAnalyzer:
    """
    Detects and analyzes doorways and doors in indoor environments.
    """

    def __init__(self) -> None:
        self.min_door_width_ratio = 0.15
        self.max_door_width_ratio = 0.65
        self.min_door_height_ratio = 0.35

    def analyze(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray],
        obstacles: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze the current frame and depth map for door/doorway state.

        Returns:
            Dict containing:
                detected: bool
                state: "OPEN" | "CLOSED" | "BLOCKED" | "POSSIBLE" | "NONE"
                confidence: float (0.0 to 1.0)
                walkable: bool
                bbox: Dict[str, float] with x1, y1, x2, y2 (normalized or absolute)
                interior_depth: float
                message: str
        """
        default_result: Dict[str, Any] = {
            "detected": False,
            "state": "NONE",
            "confidence": 0.0,
            "walkable": False,
            "is_uncertain": False,
            "bbox": None,
            "door_position": "CENTER",
            "interior_depth": 1.0,
            "obstacle_beyond": False,
            "message": "No door detected.",
        }

        if frame is None or frame.size == 0:
            return default_result

        h, w = frame.shape[:2]
        if h < 32 or w < 32:
            return default_result

        # Downscale to 320x240 for rapid analysis (< 4ms)
        target_w, target_h = 320, 240
        scale_x = w / float(target_w)
        scale_y = h / float(target_h)

        small_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY) if small_frame.ndim == 3 else small_frame

        # Resize depth map to matching dimensions if provided
        small_depth = None
        if depth_map is not None and depth_map.size > 0:
            small_depth = cv2.resize(depth_map.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Detect candidate vertical door frame jambs
        candidates = self._find_door_candidates(gray, target_w, target_h)

        if not candidates:
            return default_result

        # Reject candidates that are simply the contour/edges of a person
        if obstacles:
            def _is_person_contour(cand: Dict[str, Any]) -> bool:
                cx1 = cand["x1"] * scale_x
                cx2 = cand["x2"] * scale_x
                cw = cx2 - cx1
                for obs in obstacles:
                    if obs.get("class_name") == "person":
                        ob = obs.get("bbox", {})
                        ox1 = float(ob.get("x1", 0))
                        ox2 = float(ob.get("x2", 0))
                        overlap = max(0.0, min(cx2, ox2) - max(cx1, ox1))
                        if cw > 0 and (overlap / cw) > 0.60:
                            return True
                return False

            candidates = [c for c in candidates if not _is_person_contour(c)]

        if not candidates:
            return default_result

        # Pick the best candidate (most central and prominent)
        best_cand = None
        best_score = -1.0
        for cand in candidates:
            cx = (cand["x1"] + cand["x2"]) / 2.0
            center_dist = abs(cx - target_w / 2.0) / (target_w / 2.0)
            score = cand["edge_score"] * (1.0 - 0.4 * center_dist)
            if score > best_score:
                best_score = score
                best_cand = cand

        if best_cand is None or best_score < 0.50 or not best_cand.get("has_lintel"):
            return default_result

        bx1, by1, bx2, by2 = int(best_cand["x1"]), int(best_cand["y1"]), int(best_cand["x2"]), int(best_cand["y2"])

        # Depth reasoning
        interior_depth = 0.5
        wall_depth = 0.5
        depth_diff = 0.0
        interior_std = 0.0

        if small_depth is not None:
            # Interior region (center 60% of width, middle 60% of height)
            dx1 = int(bx1 + 0.20 * (bx2 - bx1))
            dx2 = int(bx2 - 0.20 * (bx2 - bx1))
            dy1 = int(by1 + 0.20 * (by2 - by1))
            dy2 = int(by2 - 0.20 * (by2 - by1))

            if dx2 > dx1 and dy2 > dy1:
                interior_roi = small_depth[dy1:dy2, dx1:dx2]
                if interior_roi.size > 0:
                    interior_depth = float(np.median(interior_roi))
                    interior_std = float(np.std(interior_roi))

            # Wall regions adjacent to left and right jambs
            wall_samples = []
            left_wall_x1 = max(0, int(bx1 - 0.15 * target_w))
            left_wall_x2 = max(0, int(bx1 - 2))
            if left_wall_x2 > left_wall_x1 and by2 > by1:
                lw = small_depth[by1:by2, left_wall_x1:left_wall_x2]
                if lw.size > 0:
                    wall_samples.append(float(np.median(lw)))

            right_wall_x1 = min(target_w, int(bx2 + 2))
            right_wall_x2 = min(target_w, int(bx2 + 0.15 * target_w))
            if right_wall_x2 > right_wall_x1 and by2 > by1:
                rw = small_depth[by1:by2, right_wall_x1:right_wall_x2]
                if rw.size > 0:
                    wall_samples.append(float(np.median(rw)))

            if wall_samples:
                wall_depth = float(np.mean(wall_samples))
                # Note: relative depth in backend is normalized (0 = closest, 1 = farthest)
                depth_diff = interior_depth - wall_depth

        # Scale bbox back to original image dimensions
        orig_bbox = {
            "x1": round(bx1 * scale_x, 1),
            "y1": round(by1 * scale_y, 1),
            "x2": round(bx2 * scale_x, 1),
            "y2": round(by2 * scale_y, 1),
        }

        # Check for obstacles inside doorway
        # Door lateral position relative to frame
        door_center_x = (orig_bbox["x1"] + orig_bbox["x2"]) / 2.0
        if door_center_x < w * 0.35:
            door_pos = "LEFT"
        elif door_center_x > w * 0.65:
            door_pos = "RIGHT"
        else:
            door_pos = "CENTER"

        # Check for obstacles inside doorway or immediately beyond it
        is_blocked = False
        obstacle_beyond = False
        blocking_obs = None
        beyond_obs = None

        door_w = orig_bbox["x2"] - orig_bbox["x1"]
        if obstacles and door_w > 0:
            for obs in obstacles:
                ob = obs.get("bbox", {})
                ox1 = float(ob.get("x1", 0))
                oy1 = float(ob.get("y1", 0))
                ox2 = float(ob.get("x2", 0))
                oy2 = float(ob.get("y2", 0))
                # Check horizontal overlap with doorway
                overlap_x = max(0.0, min(orig_bbox["x2"], ox2) - max(orig_bbox["x1"], ox1))
                if door_w > 0 and (overlap_x / door_w) > 0.20:
                    obs_dist = float(obs.get("distance_m", obs.get("estimated_distance_m", 99.0)))
                    obs_depth = float(obs.get("depth", 1.0))
                    if obs_dist <= 1.5 or obs_depth <= 0.45:
                        is_blocked = True
                        blocking_obs = obs
                        break
                    elif obs_dist <= 2.8 or obs_depth <= 0.65:
                        obstacle_beyond = True
                        beyond_obs = obs

        # State classification with Phase 8 taxonomy
        is_uncertain = False
        door_width_ratio = (bx2 - bx1) / float(target_w)

        if is_blocked:
            state = "BLOCKED"
            walkable = False
            confidence = 0.90
            obs_name = blocking_obs.get("class_name", "obstacle") if blocking_obs else "obstacle"
            message = f"Doorway blocked by {obs_name}."
        elif obstacle_beyond and small_depth is not None and (depth_diff >= 0.10 or interior_depth >= 0.55):
            state = "DOORWAY_OBSTACLE_BEYOND"
            walkable = False
            confidence = 0.85
            obs_name = beyond_obs.get("class_name", "obstacle") if beyond_obs else "obstacle"
            message = f"Doorway open, obstacle {obs_name} beyond."
        elif small_depth is not None and (depth_diff >= 0.15 or (interior_depth >= 0.65 and wall_depth <= 0.45)):
            # Deep opening through doorway
            state = "OPEN"
            walkable = True
            confidence = round(min(0.95, 0.70 + float(depth_diff)), 2)
            message = "Open doorway ahead. Walkable."
            # Check for partially open door (door leaf intruding or high depth standard deviation across opening)
            if interior_std > 0.12 or door_width_ratio < 0.20:
                state = "PARTIALLY_OPEN"
                walkable = False
                confidence = 0.80
                message = "Partially open door ahead. Caution."
            else:
                state = "OPEN"
                walkable = True
                confidence = round(min(0.95, 0.70 + float(depth_diff)), 2)
                message = "Open doorway ahead. Walkable."
        elif small_depth is not None and interior_depth <= 0.42 and abs(depth_diff) < 0.12:
            # Planar surface blocking doorway at close/medium distance
            state = "CLOSED"
            walkable = False
            confidence = 0.80
            confidence = 0.85
            message = "Closed door ahead. Do not enter."
        elif best_cand.get("edge_score", 0.0) < 0.40 or small_depth is None:
            state = "UNCERTAIN"
            is_uncertain = True
            walkable = False
            confidence = 0.45
            message = "Uncertain doorway ahead. Proceed with caution."
        else:
            # Ambiguous or moderate depth
            state = "POSSIBLE"
            walkable = False
            confidence = 0.60
            message = "Doorway detected ahead."

        return {
            "detected": True,
            "state": state,
            "confidence": float(confidence),
            "walkable": bool(walkable),
            "is_uncertain": bool(is_uncertain),
            "door_position": door_pos,
            "bbox": orig_bbox,
            "interior_depth": round(interior_depth, 3),
            "obstacle_beyond": bool(obstacle_beyond),
            "message": message,
        }

    def _find_door_candidates(
        self,
        gray: np.ndarray,
        width: int,
        height: int,
    ) -> List[Dict[str, Any]]:
        """
        Find candidate rectangular door frames using vertical edges and lintel lines.
        """
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=int(height * self.min_door_height_ratio),
            maxLineGap=20,
        )

        if lines is None:
            return []

        lines = lines.reshape(-1, 4)
        vertical_lines: List[Tuple[float, float, float, float]] = []
        horizontal_lines: List[Tuple[float, float, float, float]] = []

        for x1, y1, x2, y2 in lines:
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            length = math.hypot(dx, dy)

            # Near vertical: dy significantly larger than dx
            if dy > 2.5 * dx and length >= height * self.min_door_height_ratio:
                vx = (x1 + x2) / 2.0
                vy_min = min(y1, y2)
                vy_max = max(y1, y2)
                vertical_lines.append((vx, vy_min, vy_max, length))
            # Near horizontal in upper half (candidate lintel)
            elif dx > 2.5 * dy and max(y1, y2) < height * 0.55:
                horizontal_lines.append((min(x1, x2), max(x1, x2), (y1 + y2) / 2.0, length))

        if len(vertical_lines) < 2:
            return []

        # Sort vertical lines by x-coordinate
        vertical_lines.sort(key=lambda item: item[0])

        candidates = []
        min_w = width * self.min_door_width_ratio
        max_w = width * self.max_door_width_ratio

        for i in range(len(vertical_lines)):
            x_left, y1_l, y2_l, len_l = vertical_lines[i]
            for j in range(i + 1, len(vertical_lines)):
                x_right, y1_r, y2_r, len_r = vertical_lines[j]
                span_w = x_right - x_left
                if span_w < min_w:
                    continue
                if span_w > max_w:
                    break  # Sorted by x, further lines will be even wider

                # Check vertical overlap / alignment
                top_y = min(y1_l, y1_r)
                bottom_y = max(y2_l, y2_r)
                span_h = bottom_y - top_y

                if span_h < height * self.min_door_height_ratio:
                    continue

                # Aspect ratio check: doorways are taller than wide (h / w usually 1.4 to 3.2)
                aspect = span_h / max(1.0, span_w)
                if not (1.2 <= aspect <= 3.8):
                    continue

                # Check for top lintel support
                has_lintel = False
                for hx1, hx2, hy, _ in horizontal_lines:
                    if abs(hy - top_y) < height * 0.15:
                        # Check horizontal overlap with door span
                        overlap = max(0.0, min(x_right, hx2) - max(x_left, hx1))
                        if overlap / max(1.0, span_w) > 0.40:
                            has_lintel = True
                            break

                edge_score = (len_l + len_r) / (2.0 * height)
                if has_lintel:
                    edge_score += 0.35

                candidates.append({
                    "x1": float(x_left),
                    "y1": float(top_y),
                    "x2": float(x_right),
                    "y2": float(bottom_y),
                    "edge_score": float(edge_score),
                    "has_lintel": has_lintel,
                })

        return candidates
