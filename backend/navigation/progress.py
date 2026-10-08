"""
SmartVisionAI - Destination Progress and Arrival Tracking
=========================================================
Tracks relative proximity progression (FAR -> NEAR -> VERY_CLOSE)
and evaluates arrival conditions without claiming non-existent metric distance.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union


class ProgressTracker:
    """
    Monitors destination proximity progression and verifies arrival.
    """

    def __init__(self, min_stable_frames: int = 2) -> None:
        self.min_stable_frames = int(min_stable_frames)
        self.current_proximity: str = "UNKNOWN"  # "FAR" | "NEAR" | "VERY_CLOSE" | "UNKNOWN"
        self.consecutive_close_frames: int = 0
        self.stable_match_count: int = 0
        self.missing_frames: int = 0
        self.has_arrived: bool = False
        self.arrival_announced: bool = False

    @property
    def consecutive_very_close(self) -> int:
        """Alias for consecutive_close_frames."""
        return self.consecutive_close_frames

    @property
    def is_arrived(self) -> bool:
        """Alias for has_arrived."""
        return self.has_arrived

    def _extract_bbox(self, bbox: Any) -> Tuple[float, float, float, float]:
        """Safely extract (x1, y1, x2, y2) from list/tuple or dict."""
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        elif isinstance(bbox, dict):
            return (
                float(bbox.get("x1", 0)),
                float(bbox.get("y1", 0)),
                float(bbox.get("x2", 0)),
                float(bbox.get("y2", 0)),
            )
        return 0.0, 0.0, 0.0, 0.0

    def estimate_proximity(
        self,
        landmark: Dict[str, Any],
        scene_depth: Optional[Any] = None,
        frame_width: float = 640.0,
        frame_height: float = 480.0,
    ) -> str:
        """
        Estimate relative proximity (FAR, NEAR, VERY_CLOSE) based on
        optical bounding-box coverage and relative depth.
        """
        bbox = landmark.get("bbox")
        fw = float(frame_width) if frame_width > 0 else 640.0
        fh = float(frame_height) if frame_height > 0 else 480.0
        frame_area = fw * fh

        bx1, by1, bx2, by2 = self._extract_bbox(bbox)
        bw = max(0.0, bx2 - bx1)
        bh = max(0.0, by2 - by1)
        box_area = bw * bh
        h_ratio = bh / fh if fh > 0 else 0.0
        area_ratio = box_area / frame_area if frame_area > 0 else 0.0

        # Relative depth
        depth = landmark.get("depth")
        if depth is None and scene_depth is not None:
            if isinstance(scene_depth, (int, float)):
                depth = float(scene_depth)
            elif isinstance(scene_depth, dict):
                depth = scene_depth.get("center_depth", scene_depth.get("average_depth"))

        depth_val = float(depth) if depth is not None else 1.0

        # Proximity classification thresholds
        if h_ratio >= 0.30 or area_ratio >= 0.08 or (depth is not None and depth_val <= 0.25):
            return "VERY_CLOSE"
        elif h_ratio >= 0.12 or area_ratio >= 0.02 or (depth is not None and depth_val <= 0.60):
            return "NEAR"
        else:
            return "FAR"

    def update(
        self,
        matched_landmark: Optional[Dict[str, Any]] = None,
        landmark: Optional[Dict[str, Any]] = None,
        scene_depth: Optional[Any] = None,
        destination_matches: bool = True,
        safety_level: str = "SAFE",
        emergency_stop: bool = False,
        frame_width: float = 640.0,
        frame_height: float = 480.0,
    ) -> Dict[str, Any]:
        """
        Update progress state with current frame evidence.

        Returns:
            Dictionary containing progress status and arrival flag.
        """
        active_landmark = matched_landmark or landmark

        if active_landmark and destination_matches:
            self.missing_frames = 0
            self.stable_match_count += 1
            prox = self.estimate_proximity(
                landmark=active_landmark,
                scene_depth=scene_depth,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            self.current_proximity = prox

            if prox == "VERY_CLOSE":
                self.consecutive_close_frames += 1
            else:
                self.consecutive_close_frames = 0
        else:
            self.missing_frames += 1
            self.stable_match_count = 0
            self.consecutive_close_frames = 0
            if self.missing_frames >= 4:
                self.current_proximity = "UNKNOWN"

        # Check arrival criteria
        is_arrived_now = False
        if not self.has_arrived:
            if (
                self.consecutive_close_frames >= self.min_stable_frames
                and self.stable_match_count >= self.min_stable_frames
                and self.current_proximity == "VERY_CLOSE"
                and not emergency_stop
                and safety_level not in ("EMERGENCY", "DANGER")
            ):
                pos = str(active_landmark.get("position", "CENTER")).upper() if active_landmark else "CENTER"
                if pos in ("CENTER", "NONE"):
                    self.has_arrived = True
                    is_arrived_now = True

        return {
            "proximity": self.current_proximity,
            "stable_match_count": self.stable_match_count,
            "missing_frames": self.missing_frames,
            "arrived": self.has_arrived,
            "just_arrived": is_arrived_now,
        }

    def get_arrival_announcement(self, destination_label: str) -> str:
        """Formulate arrival announcement once; suppress on subsequent calls."""
        if not self.has_arrived:
            return ""
        if self.arrival_announced:
            return ""
        self.arrival_announced = True
        return self.format_arrival_speech(destination_label)

    def format_arrival_speech(self, destination_label: str) -> str:
        """Formulate concise arrival voice announcement."""
        lbl = str(destination_label).strip() if destination_label else "destination"
        return f"You have reached {lbl}."

    def reset(self) -> None:
        """Reset progress tracker state."""
        self.current_proximity = "UNKNOWN"
        self.consecutive_close_frames = 0
        self.stable_match_count = 0
        self.missing_frames = 0
        self.has_arrived = False
        self.arrival_announced = False
