"""
SmartVisionAI - Floor-Level Micro-Hazard Analyzer
=================================================
Detects low-lying tripping hazards in the immediate walking path:
- Cables, cords, wires
- Low thresholds, door sills
- Small dropped boxes, shoes, items on the floor

Fits a ground-plane depth baseline in the lower walking corridor and segments
positive height anomalies protruding from the floor.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import cv2
import numpy as np


class FloorHazardAnalyzer:
    """
    Detects floor-level micro-hazards in the lower walking corridor.
    """

    def __init__(self) -> None:
        self.min_area = 120
        self.max_area = 15000

    @staticmethod
    def _classify_contour_type(
        cnt_w: float,
        cnt_h: float,
        aspect: float,
        extent: float = 0.8,
        solidity: float = 0.85,
        residual_std: float = 0.05,
        min_depth: float = 0.5,
        mean_depth: float = 0.5,
        target_h: int = 240,
        target_w: int = 320,
    ) -> str:
        """
        Classify morphological anomaly type on floor surface.
        """
        if mean_depth > 0.80 or min_depth > 0.80:
            return "HOLE"
        if aspect >= 3.5 and cnt_h <= 15:
            return "CABLE"
        elif aspect >= 2.5:
            return "THRESHOLD"
        elif residual_std > 0.11:
            return "UNEVEN_FLOOR"
        return "TRIP_HAZARD"

    def analyze(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray],
        known_objects: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Analyze the floor region for low-lying obstacles.

        Returns:
            List of detected floor hazards as dicts compatible with unknown obstacles:
            [
                {
                    "class_name": "unknown_floor_obstacle",
                    "bbox": {"x1": float, "y1": float, "x2": float, "y2": float},
                    "hazard_type": "TRIP_HAZARD",
                    "distance_m": float,
                    "risk_score": float,
                    "confidence": float,
                    "position": "LEFT" | "CENTER" | "RIGHT",
                    "description": str,
                }
            ]
        """
        if frame is None or depth_map is None:
            return []

        h, w = frame.shape[:2]
        if h < 32 or w < 32:
            return []

        # Downscale for performance (< 4ms)
        target_w, target_h = 320, 240
        scale_x = w / float(target_w)
        scale_y = h / float(target_h)

        small_depth = cv2.resize(depth_map.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Immediate floor region (bottom 40% of frame, center 60% of width)
        y_start = int(target_h * 0.60)
        x_start = int(target_w * 0.20)
        x_end = int(target_w * 0.80)

        roi_depth = small_depth[y_start:target_h, x_start:x_end]
        roi_h, roi_w = roi_depth.shape

        if roi_h < 10 or roi_w < 10:
            return []

        # Estimate expected ground-plane depth for each row y
        # On a clean flat floor, median depth decreases smoothly toward the bottom
        row_medians = np.median(roi_depth, axis=1)
        y_indices = np.arange(roi_h, dtype=np.float32)

        # Fit line: expected_depth = slope * y + intercept
        try:
            poly = np.polyfit(y_indices, row_medians, deg=1)
            expected_floor = np.polyval(poly, y_indices)[:, np.newaxis]
        except Exception:
            expected_floor = row_medians[:, np.newaxis]

        # Anomaly mask: pixels significantly closer (lower depth value) than expected floor
        # Positive anomaly mask (rising above floor: cables, objects, raised thresholds)
        # relative depth: smaller = closer
        depth_anomaly = (expected_floor - roi_depth) > 0.09
        anomaly_uint8 = (depth_anomaly.astype(np.uint8)) * 255
        pos_anomaly = (expected_floor - roi_depth) > 0.08
        pos_uint8 = (pos_anomaly.astype(np.uint8)) * 255

        # Negative anomaly mask (dropping below floor: holes, depressions)
        # relative depth: larger = farther / dropping away
        neg_anomaly = (roi_depth - expected_floor) > 0.08
        neg_uint8 = (neg_anomaly.astype(np.uint8)) * 255

        # Morphological clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(anomaly_uint8, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        pos_cleaned = cv2.morphologyEx(pos_uint8, cv2.MORPH_OPEN, kernel)
        pos_cleaned = cv2.morphologyEx(pos_cleaned, cv2.MORPH_CLOSE, kernel)

        # Connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        neg_cleaned = cv2.morphologyEx(neg_uint8, cv2.MORPH_OPEN, kernel)
        neg_cleaned = cv2.morphologyEx(neg_cleaned, cv2.MORPH_CLOSE, kernel)

        hazards: List[Dict[str, Any]] = []

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
        # Floor surface roughness / unevenness check
        depth_residuals = np.abs(expected_floor - roi_depth)
        residual_std = float(np.std(depth_residuals))
        if residual_std > 0.11:
            hazards.append({
                "class_name": "unknown_floor_obstacle",
                "bbox": {
                    "x1": float(round(x_start * scale_x, 1)),
                    "y1": float(round(y_start * scale_y, 1)),
                    "x2": float(round(x_end * scale_x, 1)),
                    "y2": float(round(target_h * scale_y, 1)),
                },
                "hazard_type": "UNEVEN_FLOOR",
                "distance_m": 1.2,
                "risk_score": 0.65,
                "confidence": 0.78,
                "position": "CENTER",
                "cand_depth": float(np.median(roi_depth)),
                "description": "Uneven surface ahead.",
            })

        # Process positive anomalies (objects, thresholds, cables)
        num_pos, _, stats_pos, _ = cv2.connectedComponentsWithStats(pos_cleaned, connectivity=8)
        for i in range(1, num_pos):
            area = stats_pos[i, cv2.CC_STAT_AREA]
            if not (self.min_area <= area <= self.max_area):
                continue

            bx = stats[i, cv2.CC_STAT_LEFT]
            by = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            bx = stats_pos[i, cv2.CC_STAT_LEFT]
            by = stats_pos[i, cv2.CC_STAT_TOP]
            bw = stats_pos[i, cv2.CC_STAT_WIDTH]
            bh = stats_pos[i, cv2.CC_STAT_HEIGHT]

            # Global coordinates in small frame
            gx1 = x_start + bx
            gy1 = y_start + by
            gx2 = gx1 + bw
            gy2 = gy1 + bh
            gx1, gy1 = x_start + bx, y_start + by
            gx2, gy2 = gx1 + bw, gy1 + bh

            # Scale to original resolution
            orig_x1 = float(round(gx1 * scale_x, 1))
            orig_y1 = float(round(gy1 * scale_y, 1))
            orig_x2 = float(round(gx2 * scale_x, 1))
            orig_y2 = float(round(gy2 * scale_y, 1))

            # Check overlap with any already-detected known YOLO object
            # Check overlap with known YOLO objects
            already_covered = False
            if known_objects:
                for k in known_objects:
                    kb = k.get("bbox", {})
                    kx1 = float(kb.get("x1", 0))
                    ky1 = float(kb.get("y1", 0))
                    kx2 = float(kb.get("x2", 0))
                    ky2 = float(kb.get("y2", 0))
                    ix1 = max(orig_x1, kx1)
                    iy1 = max(orig_y1, ky1)
                    ix2 = min(orig_x2, kx2)
                    iy2 = min(orig_y2, ky2)
                    iw = max(0.0, ix2 - ix1)
                    ih = max(0.0, iy2 - iy1)
                    if (iw * ih) / max(1.0, (orig_x2 - orig_x1) * (orig_y2 - orig_y1)) > 0.35:
                        already_covered = True
                        break

            if already_covered:
                continue

            # Compute hazard characteristics
            obj_depth_roi = small_depth[gy1:gy2, gx1:gx2]
            obj_depth = float(np.median(obj_depth_roi)) if obj_depth_roi.size > 0 else 0.3
            # Estimate physical distance based on y-position in frame (closer to bottom = closer to user)
            rel_y = (gy2 - y_start) / float(target_h - y_start)
            distance_m = round(max(0.4, 2.0 - 1.5 * rel_y), 1)

            # Lateral position
            cx_orig = (orig_x1 + orig_x2) / 2.0
            pos = "LEFT" if cx_orig < w * 0.35 else ("RIGHT" if cx_orig > w * 0.65 else "CENTER")

            # Classify specific morphology
            aspect_ratio = float(bw) / max(1.0, float(bh))
            if aspect_ratio >= 3.5 and bh <= 10:
                h_type = "CABLE"
                h_desc = "Cable on the floor."
                risk = 0.60
            elif aspect_ratio >= 2.5:
                h_type = "THRESHOLD"
                h_desc = "Raised floor threshold ahead."
                risk = 0.55
            else:
                h_type = "TRIP_HAZARD"
                h_desc = "Obstacle on the floor."
                risk = 0.65

            if pos == "CENTER":
                risk += 0.15
            if distance_m <= 1.0:
                risk += 0.10

            hazards.append({
                "class_name": "unknown_floor_obstacle",
                "bbox": {"x1": orig_x1, "y1": orig_y1, "x2": orig_x2, "y2": orig_y2},
                "hazard_type": h_type,
                "distance_m": distance_m,
                "risk_score": round(min(0.90, risk), 2),
                "confidence": round(min(0.88, 0.60 + 0.25 * (1.0 - obj_depth)), 2),
                "position": pos,
                "cand_depth": obj_depth,
                "description": h_desc,
            })

        # Process negative anomalies (depressions, holes)
        num_neg, _, stats_neg, _ = cv2.connectedComponentsWithStats(neg_cleaned, connectivity=8)
        for i in range(1, num_neg):
            area = stats_neg[i, cv2.CC_STAT_AREA]
            if not (self.min_area <= area <= self.max_area):
                continue

            bx = stats_neg[i, cv2.CC_STAT_LEFT]
            by = stats_neg[i, cv2.CC_STAT_TOP]
            bw = stats_neg[i, cv2.CC_STAT_WIDTH]
            bh = stats_neg[i, cv2.CC_STAT_HEIGHT]

            gx1, gy1 = x_start + bx, y_start + by
            gx2, gy2 = gx1 + bw, gy1 + bh

            orig_x1 = float(round(gx1 * scale_x, 1))
            orig_y1 = float(round(gy1 * scale_y, 1))
            orig_x2 = float(round(gx2 * scale_x, 1))
            orig_y2 = float(round(gy2 * scale_y, 1))

            obj_depth_roi = small_depth[gy1:gy2, gx1:gx2]
            obj_depth = float(np.median(obj_depth_roi)) if obj_depth_roi.size > 0 else 0.8
            rel_y = (gy2 - y_start) / float(target_h - y_start)
            distance_m = round(max(0.4, 2.0 - 1.5 * rel_y), 1)

            cx_orig = (orig_x1 + orig_x2) / 2.0
            pos = "LEFT" if cx_orig < w * 0.35 else ("RIGHT" if cx_orig > w * 0.65 else "CENTER")

            hazards.append({
                "class_name": "unknown_floor_obstacle",
                "bbox": {"x1": orig_x1, "y1": orig_y1, "x2": orig_x2, "y2": orig_y2},
                "hazard_type": "DEPRESSION",
                "distance_m": distance_m,
                "risk_score": 0.88 if pos == "CENTER" else 0.75,
                "confidence": 0.82,
                "position": pos,
                "cand_depth": obj_depth,
                "description": "Depression on the floor ahead. Stop.",
            })

        return hazards[:4]

