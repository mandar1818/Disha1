"""
SmartVisionAI - Stairs, Steps & Drop-off Analyzer
=================================================
High-priority indoor fall-hazard detector for blind and visually impaired users:
- Downward drop-offs / floor cliffs (Highest danger -> CRITICAL / STOP)
- Stairs going down (CRITICAL / STOP)
- Stairs going up (WARNING / Caution)

Combines vertical depth gradient analysis and horizontal step edge harmonics.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


class StairDropoffAnalyzer:
    """
    Detects stairs and sudden floor drop-offs using depth gradients and edge harmonics.
    """

    def __init__(self) -> None:
        self.min_step_lines = 3
        self.dropoff_gradient_thresh = 0.15

    def analyze(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """
        Analyze current frame and depth map for stairs or drop-offs.

        Returns:
            Dict containing:
                detected: bool
                type: "STAIRS_DOWN" | "STAIRS_UP" | "DROP_OFF" | "NONE"
                confidence: float (0.0 to 1.0)
                hazard_level: "CRITICAL" | "WARNING" | "SAFE"
                distance_m: float
                step_count_estimate: int
                message: str
        """
        default_result: Dict[str, Any] = {
            "detected": False,
            "type": "NONE",
            "confidence": 0.0,
            "hazard_level": "SAFE",
            "is_uncertain": False,
            "distance_m": 99.0,
            "step_count_estimate": 0,
            "message": "Floor clear.",
        }

        if frame is None or frame.size == 0:
            return default_result

        h, w = frame.shape[:2]
        if h < 32 or w < 32:
            return default_result

        # Downscale for rapid processing (< 4ms)
        target_w, target_h = 320, 240
        small_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY) if small_frame.ndim == 3 else small_frame

        small_depth = None
        if depth_map is not None and depth_map.size > 0:
            small_depth = cv2.resize(depth_map.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # 1. Depth Discontinuity & Drop-off Analysis (Lower half of frame)
        dropoff_detected = False
        dropoff_confidence = 0.0
        dropoff_y = 0.0
        dropoff_distance_m = 99.0
        ramp_detected = False
        ramp_confidence = 0.0

        if small_depth is not None:
            # Walking corridor region in depth map
            y_start = int(target_h * 0.45)
            x_start = int(target_w * 0.25)
            x_end = int(target_w * 0.75)

            corridor_depth = small_depth[y_start:target_h, x_start:x_end]
            if corridor_depth.size > 0:
                # Vertical profile (median depth per row)
                row_medians = np.median(corridor_depth, axis=1)

                # Check for sudden positive jump in depth (going down into void / lower floor)
                # In normalized depth (0 = close, 1 = far): normal flat floor decreases as y increases.
                # Check for sudden substantial positive jump in depth (going down into void / lower floor)
                # In normalized depth (0 = close, 1 = far): normal flat floor decreases as y increases.
                # A jump where ground genuinely falls away requires a substantial depth delta (> 0.20)
                # or consistent step lines. Small 0.03-0.05 variations are floor reflections/textures/shoes.
                diffs = np.diff(row_medians)
                # Find significant positive jumps indicating real drop-off
                positive_jumps = np.where(diffs > 0.18)[0]

                if len(positive_jumps) >= 3:
                    # Consecutive upward jumps in depth as row increases
                    peak_jump_idx = int(positive_jumps[0])
                    jump_magnitude = float(np.max(diffs))
                    if jump_magnitude > 0.22:
                        dropoff_detected = True
                        dropoff_y = y_start + peak_jump_idx
                        rel_pos = (dropoff_y - y_start) / float(target_h - y_start)
                        dropoff_distance_m = round(max(0.6, 2.5 - 1.8 * rel_pos), 1)
                        dropoff_confidence = round(min(0.95, 0.65 + jump_magnitude * 2.0), 2)
                elif len(positive_jumps) >= 1 and float(diffs[positive_jumps[0]]) > 0.28:
                    dropoff_detected = True
                    dropoff_y = y_start + int(positive_jumps[0])
                    rel_pos = (dropoff_y - y_start) / float(target_h - y_start)
                    dropoff_distance_m = round(max(0.6, 2.5 - 1.8 * rel_pos), 1)
                    dropoff_confidence = round(min(0.85, 0.55 + float(diffs[positive_jumps[0]]) * 1.5), 2)

                # Ramp slope analysis (smooth gradient along corridor)
                if not dropoff_detected and len(row_medians) > 20:
                    y_axis = np.arange(len(row_medians), dtype=np.float32)
                    poly, residuals, _, _, _ = np.polyfit(y_axis, row_medians, 1, full=True)
                    slope = float(poly[0])
                    if slope > 0.005:
                        ramp_detected = True
                        ramp_confidence = 0.72
                    elif slope < -0.015:
                        ramp_detected = True
                        ramp_confidence = 0.70

        # 2. Horizontal Step Edge Harmonics (Periodic horizontal lines)
        step_edges = self._find_horizontal_step_lines(gray, target_w, target_h)
        step_count = len(step_edges)

        # 3. Decision Logic & Hazard Classification
        if dropoff_detected and step_count >= 2:
            # Step edges + depth jump = stairs going down
            return {
                "detected": True,
                "type": "STAIRS_DOWN",
                "confidence": max(dropoff_confidence, 0.85),
                "hazard_level": "CRITICAL",
                "is_uncertain": False,
                "distance_m": dropoff_distance_m,
                "step_count_estimate": step_count,
                "message": "Stop. Stairs detected ahead.",
            }

        if dropoff_detected and dropoff_distance_m <= 1.4 and jump_magnitude > 0.25:
            # Severe drop-off / platform edge
            return {
                "detected": True,
                "type": "DROP_OFF",
                "confidence": dropoff_confidence,
                "hazard_level": "CRITICAL",
                "is_uncertain": False,
                "distance_m": dropoff_distance_m,
                "step_count_estimate": 1,
                "message": "Stop. Drop-off ahead.",
            }
        elif dropoff_detected and dropoff_distance_m <= 2.0:
            # Moderate depth shift without step lines: warn with caution, do not freeze in emergency
            return {
                "detected": True,
                "type": "DROP_OFF",
                "confidence": dropoff_confidence,
                "hazard_level": "WARNING",
                "is_uncertain": False,
                "distance_m": dropoff_distance_m,
                "step_count_estimate": 1,
                "message": "Caution. Floor level change ahead.",
            }

        if step_count >= self.min_step_lines:
            # Determine if stairs are going UP or DOWN using depth gradient
            stairs_type = "STAIRS_UP"
            hazard = "WARNING"
            confidence = 0.75 + min(0.15, step_count * 0.03)
            first_step_y = step_edges[0]
            rel_pos = float(first_step_y) / float(target_h)
            dist_m = round(max(0.8, 3.0 - 2.0 * rel_pos), 1)

            if small_depth is not None and len(step_edges) >= 2:
                # Compare depth at first step vs last step
                y_top = int(min(step_edges))
                y_bot = int(max(step_edges))
                if 0 <= y_top < target_h and 0 <= y_bot < target_h:
                    d_top = float(np.median(small_depth[y_top, int(target_w * 0.35):int(target_w * 0.65)]))
                    d_bot = float(np.median(small_depth[y_bot, int(target_w * 0.35):int(target_w * 0.65)]))
                    # If bottom step is farther than top step -> descending stairs!
                    if d_bot > d_top + 0.04:
                        stairs_type = "STAIRS_DOWN"
                        hazard = "CRITICAL"
                        confidence = 0.85

            msg = "Stop. Stairs detected ahead." if stairs_type == "STAIRS_DOWN" else "Caution. Stairs up ahead."

            return {
                "detected": True,
                "type": stairs_type,
                "confidence": round(confidence, 2),
                "hazard_level": hazard,
                "is_uncertain": False,
                "distance_m": dist_m,
                "step_count_estimate": step_count,
                "message": msg,
            }

        # Single step or raised level transition
        if step_count in (1, 2) and small_depth is not None:
            return {
                "detected": True,
                "type": "SINGLE_STEP",
                "confidence": 0.72,
                "hazard_level": "CAUTION",
                "is_uncertain": False,
                "distance_m": 1.5,
                "step_count_estimate": 1,
                "message": "Slow down. Floor level change ahead.",
            }

        # Continuous smooth ramp
        if ramp_detected:
            return {
                "detected": True,
                "type": "RAMP",
                "confidence": round(ramp_confidence, 2),
                "hazard_level": "CAUTION",
                "is_uncertain": False,
                "distance_m": 2.0,
                "step_count_estimate": 0,
                "message": "Caution. Ramp ahead.",
            }

        # Uncertain level transition if weak dropoff or irregular edges
        if dropoff_detected:
            return {
                "detected": True,
                "type": "UNCERTAIN_LEVEL_TRANSITION",
                "confidence": 0.50,
                "hazard_level": "CAUTION",
                "is_uncertain": True,
                "distance_m": dropoff_distance_m,
                "step_count_estimate": 1,
                "message": "Slow down. Floor level change ahead.",
            }

        return default_result

    def _find_horizontal_step_lines(
        self,
        gray: np.ndarray,
        width: int,
        height: int,
    ) -> List[float]:
        """
        Find periodic horizontal step edges in the lower walking corridor.
        """
        y_min = int(height * 0.40)
        roi = gray[y_min:height, :]
        blurred = cv2.GaussianBlur(roi, (3, 3), 0)

        # Sobel horizontal gradient emphasizes horizontal lines (steps)
        sobel_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=3)
        abs_sobel = cv2.convertScaleAbs(sobel_y)
        _, thresh = cv2.threshold(abs_sobel, 35, 255, cv2.THRESH_BINARY)

        lines = cv2.HoughLinesP(
            thresh,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=int(width * 0.25),
            maxLineGap=25,
        )

        if lines is None:
            return []

        lines = lines.reshape(-1, 4)
        horizontal_y: List[float] = []
        for x1, y1, x2, y2 in lines:
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            # Horizontal lines: dx much greater than dy
            if dx > 3.0 * dy:
                y_coord = y_min + (y1 + y2) / 2.0
                horizontal_y.append(float(y_coord))

        if len(horizontal_y) < 2:
            return []

        horizontal_y.sort()

        # Cluster nearby lines (within 8 pixels)
        clustered: List[float] = []
        curr_cluster = [horizontal_y[0]]
        for y in horizontal_y[1:]:
            if y - curr_cluster[-1] <= 8.0:
                curr_cluster.append(y)
            else:
                clustered.append(float(np.mean(curr_cluster)))
                curr_cluster = [y]
        if curr_cluster:
            clustered.append(float(np.mean(curr_cluster)))

        # Verify periodic spacing between clustered lines (stairs typically have 12-45px spacing)
        if len(clustered) >= 3:
            spacings = np.diff(clustered)
            valid_spacings = [s for s in spacings if 10.0 <= s <= 50.0]
            if len(valid_spacings) >= 2:
                return clustered

        return clustered if len(clustered) >= 4 else []
