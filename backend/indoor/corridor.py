"""
SmartVisionAI - Corridor & Narrow-Passage Clearance Analyzer
============================================================
Indoor corridor structural perception and human shoulder clearance reasoning:
- Corridor wall convergence & vanishing point
- Left and right wall boundaries
- Physical walkable width estimation (meters)
- Clearance states:
    - CLEAR (>= 1.1m): Normal walking clearance
    - NARROW (0.65m - 1.05m): Constrained clearance -> SLOW_DOWN
    - BLOCKED (< 0.65m): Insufficient human shoulder clearance -> STOP
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


class CorridorClearanceAnalyzer:
    """
    Analyzes indoor corridors, wall boundaries, and narrow passage clearances.
    """

    def __init__(self) -> None:
        self.shoulder_clearance_min_m = 0.65
        self.comfortable_width_min_m = 1.10

    @staticmethod
    def _compute_inter_obstacle_gap(
        obstacles: List[Dict[str, Any]],
        frame_width: float = 640.0,
        full_span_m: float = 2.3,
    ) -> Optional[float]:
        """
        Compute minimum clearance gap between left and right obstacles in meters.
        """
        if not obstacles or len(obstacles) < 2:
            return None
        has_left = False
        has_right = False
        left_edge_x = 0.0
        right_edge_x = float(frame_width)
        for obs in obstacles:
            b = obs.get("bbox", {})
            x1 = float(b.get("x1", 0))
            x2 = float(b.get("x2", 0))
            cx = (x1 + x2) / 2.0
            if cx < frame_width * 0.50:
                has_left = True
                left_edge_x = max(left_edge_x, x2)
            else:
                has_right = True
                right_edge_x = min(right_edge_x, x1)
        if has_left and has_right:
            gap_px = max(0.0, right_edge_x - left_edge_x)
            return round((gap_px / float(frame_width)) * full_span_m, 2)
        return None

    def analyze(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray],
        obstacles: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze corridor structure and compute walkable clearance.

        Returns:
            Dict containing:
                corridor_detected: bool
                walkable_width_m: float
                clearance_state: "CLEAR" | "NARROW" | "BLOCKED"
                vanishing_point: Dict[str, float] or None
                left_boundary_x: float
                right_boundary_x: float
                message: str
        """
        default_result: Dict[str, Any] = {
            "corridor_detected": False,
            "walkable_width_m": 2.0,
            "clearance_state": "CLEAR",
            "vanishing_point": None,
            "left_boundary_x": 0.0,
            "right_boundary_x": 640.0,
            "message": "Path clear.",
        }

        if frame is None or frame.size == 0:
            return default_result

        h, w = frame.shape[:2]
        if h < 32 or w < 32:
            return default_result

        # Downscale for performance (< 3ms)
        target_w, target_h = 320, 240
        small_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY) if small_frame.ndim == 3 else small_frame

        # 1. Perspective Wall Line Detection (Hough lines)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)

        # Focus on mid-to-upper region for corridor walls
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=int(target_h * 0.30),
            maxLineGap=20,
        )

        left_wall_lines: List[Tuple[float, float, float, float]] = []
        right_wall_lines: List[Tuple[float, float, float, float]] = []

        if lines is not None:
            lines = lines.reshape(-1, 4)
            for x1, y1, x2, y2 in lines:
                dx = x2 - x1
                dy = y2 - y1
                length = math.hypot(dx, dy)
                if length < target_h * 0.25:
                    continue

                angle = math.degrees(math.atan2(dy, dx))
                # Normalize angle to -90 to +90
                if angle < -90:
                    angle += 180
                elif angle > 90:
                    angle -= 180

                # Left wall lines typically slant down-left to up-center (angle ~ 25 to 70 deg)
                # Right wall lines typically slant down-right to up-center (angle ~ -25 to -70 deg)
                mid_x = (x1 + x2) / 2.0
                if 15 <= abs(angle) <= 86:
                    if mid_x < target_w * 0.50 and ((x1 < x2 and y1 > y2) or (x2 < x1 and y2 > y1)):
                        left_wall_lines.append((x1, y1, x2, y2))
                    elif mid_x > target_w * 0.50 and ((x1 > x2 and y1 > y2) or (x2 > x1 and y2 > y1)):
                        right_wall_lines.append((x1, y1, x2, y2))

        corridor_detected = len(left_wall_lines) >= 1 or len(right_wall_lines) >= 1

        # Estimate vanishing point if both walls have perspective lines
        vp = None
        if left_wall_lines and right_wall_lines:
            # Intersection of representative left line and right line
            l1 = left_wall_lines[0]
            r1 = right_wall_lines[0]
            vp_pt = self._line_intersection(l1, r1)
            if vp_pt is not None:
                vpx, vpy = vp_pt
                if 0 <= vpx <= target_w and 0 <= vpy <= target_h * 0.75:
                    vp = {
                        "x": round(float(vpx * (w / target_w)), 1),
                        "y": round(float(vpy * (h / target_h)), 1),
                    }

        # 2. Estimate Walkable Corridor Width at Foot Level (y = 0.70 * h)
        eval_y = int(target_h * 0.70)
        left_bound_x = 0.0
        right_bound_x = float(target_w)

        # From left wall lines, estimate x at eval_y
        for x1, y1, x2, y2 in left_wall_lines:
            if abs(y2 - y1) > 1e-3:
                slope = (x2 - x1) / (y2 - y1)
                x_at_y = x1 + slope * (eval_y - y1)
                if 0 <= x_at_y <= target_w * 0.48:
                    left_bound_x = max(left_bound_x, x_at_y)

        # From right wall lines, estimate x at eval_y
        for x1, y1, x2, y2 in right_wall_lines:
            if abs(y2 - y1) > 1e-3:
                slope = (x2 - x1) / (y2 - y1)
                x_at_y = x1 + slope * (eval_y - y1)
                if target_w * 0.52 <= x_at_y <= target_w:
                    right_bound_x = min(right_bound_x, x_at_y)

        # 3. Factor in Lateral Obstacles Protruding into Corridor
        # 3. Factor in Lateral Obstacles Protruding into Corridor and Inter-Obstacle Gap
        scale_x = float(w) / float(target_w)
        left_bound_orig = left_bound_x * scale_x
        right_bound_orig = right_bound_x * scale_x

        full_span_m = 2.3
        inter_obstacle_gap_m = None
        has_left_obstacle = False
        has_right_obstacle = False
        left_obstacle_edge_x = 0.0
        right_obstacle_edge_x = float(w)

        if obstacles:
            for obs in obstacles:
                ob = obs.get("bbox", {})
                ox1 = float(ob.get("x1", 0))
                oy1 = float(ob.get("y1", 0))
                ox2 = float(ob.get("x2", 0))
                oy2 = float(ob.get("y2", 0))
                dist = float(obs.get("distance_m", obs.get("estimated_distance_m", 99.0)))
                # If obstacle is within walking distance ahead (<= 3.0m) and in lower/mid frame
                if dist <= 3.0 and oy2 >= h * 0.45:
                    # If obstacle protrudes from left
                    if ox1 < w * 0.40 and ox2 > left_bound_orig:
                        left_bound_orig = max(left_bound_orig, min(ox2, w * 0.48))
                    # If obstacle protrudes from right
                    if ox2 > w * 0.60 and ox1 < right_bound_orig:
                        right_bound_orig = min(right_bound_orig, max(ox1, w * 0.52))
                if dist <= 3.0 and oy2 >= h * 0.40:
                    cx = (ox1 + ox2) / 2.0
                    if cx < w * 0.50:
                        has_left_obstacle = True
                        left_obstacle_edge_x = max(left_obstacle_edge_x, ox2)
                        left_bound_orig = max(left_bound_orig, min(ox2, w * 0.45))
                    else:
                        has_right_obstacle = True
                        right_obstacle_edge_x = min(right_obstacle_edge_x, ox1)
                        right_bound_orig = min(right_bound_orig, max(ox1, w * 0.55))

            if has_left_obstacle and has_right_obstacle:
                gap_px = max(0.0, right_obstacle_edge_x - left_obstacle_edge_x)
                inter_obstacle_gap_m = round((gap_px / float(w)) * full_span_m, 2)

        # 4. Convert Pixel Clearance to Physical Meters
        pixel_width = max(0.0, right_bound_orig - left_bound_orig)
        # Standard camera FOV (~60 deg) at 2m forward gives ~2.3m full horizontal span
        full_span_m = 2.3
        walkable_width_m = round((pixel_width / float(w)) * full_span_m, 2)

        # Check crowd corridor blockage
        crowd_blocking = False
        if obstacles:
            nearby_persons = sum(
                1 for obs in obstacles
                if str(obs.get("class_name", "")).lower() == "person"
                and float(obs.get("distance_m", obs.get("estimated_distance_m", 99.0))) <= 2.5
            )
            if nearby_persons >= 2 and walkable_width_m < 0.90:
                crowd_blocking = True

        effective_gap = min(walkable_width_m, inter_obstacle_gap_m) if inter_obstacle_gap_m is not None else walkable_width_m
        is_safe_gap = bool(effective_gap >= self.shoulder_clearance_min_m)

        # Clearance classification (only apply corridor constriction if a corridor was detected or crowd blocks)
        if not corridor_detected and not crowd_blocking:
            clearance_state = "CLEAR"
            message = "Path clear."
        elif effective_gap < self.shoulder_clearance_min_m or crowd_blocking:
            clearance_state = "BLOCKED"
            message = "Stop. Path blocked."
        elif effective_gap < self.comfortable_width_min_m:
            clearance_state = "NARROW"
            message = "Narrow passage. Slow down."
        else:
            clearance_state = "CLEAR"
            message = "Corridor clear."

        return {
            "corridor_detected": bool(corridor_detected),
            "walkable_width_m": float(walkable_width_m),
            "inter_obstacle_gap_m": inter_obstacle_gap_m,
            "is_safe_gap": is_safe_gap,
            "clearance_state": clearance_state,
            "vanishing_point": vp,
            "left_boundary_x": round(float(left_bound_orig), 1),
            "right_boundary_x": round(float(right_bound_orig), 1),
            "message": message,
        }

    @staticmethod
    def _line_intersection(
        line1: Tuple[float, float, float, float],
        line2: Tuple[float, float, float, float],
    ) -> Optional[Tuple[float, float]]:
        """
        Calculate intersection of two 2D lines.
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-4:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        return (px, py)
