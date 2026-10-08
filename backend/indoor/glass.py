"""
SmartVisionAI - Glass & Transparent Obstacle Analyzer
=====================================================
Specialized safety reasoning for transparent surfaces:
- Glass doors and glass partition walls
- Glass panels with structural mullions/frames
- Specular reflection anomalies and depth-penetration discrepancies

Prevents blind users from colliding with transparent barriers that MiDaS depth
might otherwise perceive as open/deep space.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import cv2
import numpy as np


class GlassHazardAnalyzer:
    """
    Detects glass doors and transparent partition obstacles.
    """

    def __init__(self) -> None:
        self.min_specular_threshold = 240
        self.min_reflection_area = 150

    def analyze(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """
        Analyze frame for transparent/glass obstacles.

        Returns:
            Dict containing:
                detected: bool
                confidence: float (0.0 to 1.0)
                position: "LEFT" | "CENTER" | "RIGHT"
                risk_score: float
                reason: str
        """
        default_result: Dict[str, Any] = {
            "detected": False,
            "hazard_type": "NONE",
            "confidence": 0.0,
            "position": "CENTER",
            "risk_score": 0.0,
            "is_uncertain": False,
            "reason": "No glass obstacle detected.",
            "voice_message": "",
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

        # 1. Specular Highlights / Reflections Detection
        # Glass surfaces reflect lights and environment, creating high-luminance low-saturation patches
        _, bright_thresh = cv2.threshold(gray, self.min_specular_threshold, 255, cv2.THRESH_BINARY)
        bright_count = cv2.countNonZero(bright_thresh)

        # 2. Vertical Mullion / Framing Line Detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=int(target_h * 0.40),
            maxLineGap=20,
        )

        vertical_frames: List[float] = []
        if lines is not None:
            lines = lines.reshape(-1, 4)
            for x1, y1, x2, y2 in lines:
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                if dy > 3.0 * dx and math.hypot(dx, dy) >= target_h * 0.40:
                    vertical_frames.append(float((x1 + x2) / 2.0))

        # Sort vertical lines
        vertical_frames.sort()

        # 3. Low Texture / High Specularity Region Analysis
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_var = float(laplacian.var())

        # 4. Depth Anomaly Analysis (if depth map available)
        # If visual frame lines exist or low texture + specular highlights, but depth in the center shows deep penetration (> 0.50),
        # this is characteristic of transparent glass.
        depth_anomaly = False
        center_depth = 0.0
        if depth_map is not None and depth_map.size > 0:
            small_depth = cv2.resize(depth_map.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            center_depth = float(np.median(small_depth[int(target_h * 0.3):int(target_h * 0.7), int(target_w * 0.3):int(target_w * 0.7)]))
            # If visual lines present but depth thinks center is far (> 0.60), potential glass pane!
            if center_depth > 0.60 and (bright_count > 70 or (len(vertical_frames) >= 2 and texture_var < 150.0)):
                depth_anomaly = True

        detected = False
        hazard_type = "NONE"
        conf = 0.0
        risk = 0.0
        pos = "CENTER"
        voice_msg = ""
        reason = ""

        if len(vertical_frames) >= 2 and (bright_count > 100 or depth_anomaly):
            detected = True
            gap_px = vertical_frames[-1] - vertical_frames[0]
            gap_ratio = gap_px / float(target_w)
            avg_x = float(np.mean(vertical_frames))

            if avg_x < target_w * 0.35:
                pos = "LEFT"
            elif avg_x > target_w * 0.65:
                pos = "RIGHT"
            else:
                pos = "CENTER"

            if 0.20 <= gap_ratio <= 0.65:
                hazard_type = "GLASS_DOOR"
                voice_msg = "Caution. Possible glass door ahead."
                reason = f"Possible glass door detected on {pos.lower()}."
                conf = 0.80
                risk = 0.75 if pos == "CENTER" else 0.65
            else:
                hazard_type = "GLASS_WALL"
                voice_msg = "Caution. Possible glass wall ahead."
                reason = f"Possible glass partition wall detected on {pos.lower()}."
                conf = 0.75
                risk = 0.80 if pos == "CENTER" else 0.70

            if depth_anomaly:
                conf = min(0.85, conf + 0.08)

        elif len(vertical_frames) >= 1 and bright_count > 250:
            detected = True
            hazard_type = "GLASS_PANEL"
            avg_x = vertical_frames[0]
            if avg_x < target_w * 0.35:
                pos = "LEFT"
            elif avg_x > target_w * 0.65:
                pos = "RIGHT"
            else:
                pos = "CENTER"
            conf = 0.70
            risk = 0.70 if pos == "CENTER" else 0.60
            voice_msg = "Caution. Possible glass obstacle ahead."
            reason = f"Possible glass panel detected on {pos.lower()}."

        elif depth_anomaly and bright_count > 150:
            detected = True
            hazard_type = "UNCERTAIN_TRANSPARENT"
            pos = "CENTER"
            conf = 0.65
            risk = 0.70
            voice_msg = "Slow down. Possible transparent surface ahead."
            reason = "Specular reflection anomaly with contradictory open depth indicates possible transparent barrier."

        if detected:
            return {
                "detected": True,
                "hazard_type": hazard_type,
                "confidence": round(float(min(0.85, conf)), 2),
                "position": pos,
                "risk_score": round(float(risk), 2),
                "is_uncertain": True,
                "voice_message": voice_msg,
                "reason": reason,
            }

        return default_result
