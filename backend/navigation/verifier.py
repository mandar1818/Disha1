"""
SmartVisionAI - Closed-Loop Turn Verification Engine
====================================================
Verifies whether a user has executed a commanded rotation (TURN_LEFT / TURN_RIGHT)
using visual scene feedback (destination displacement, scene depth redistribution,
free-path corridor changes) across consecutive frames.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple, Union


class TurnVerifier:
    """
    Verifies rotation commands against incoming visual evidence.
    """

    def __init__(self, timeout_frames: int = 5, retry_cooldown_s: float = 6.0) -> None:
        self.max_verification_frames = int(timeout_frames)
        self.retry_cooldown_s = float(retry_cooldown_s)

        self.status: str = "NOT_REQUIRED"  # "NOT_REQUIRED" | "IN_PROGRESS" | "VERIFIED" | "FAILED" | "TIMEOUT"
        self.pending_turn: str = "NONE"     # "TURN_LEFT" | "TURN_RIGHT" | "NONE"
        self.frames_waited: int = 0
        self.last_prompt_time: float = 0.0

        # Snapshot of visual state when turn was commanded
        self.initial_landmark_pos: Optional[str] = None
        self.initial_landmark_cx: Optional[float] = None
        self.initial_depth_profile: Dict[str, float] = {}
        self.initial_depth: Optional[float] = None
        self.initial_best_direction: Optional[str] = None

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

    def request_turn(
        self,
        turn_direction: str,
        initial_landmark_pos: Optional[str] = None,
        initial_depth: Optional[float] = None,
        landmark: Optional[Dict[str, Any]] = None,
        scene_depth: Optional[Any] = None,
        frame_width: float = 640.0,
    ) -> None:
        """Alias for start_verification."""
        self.start_verification(
            turn_direction=turn_direction,
            landmark=landmark,
            scene_depth=scene_depth,
            initial_landmark_pos=initial_landmark_pos,
            initial_depth=initial_depth,
            frame_width=frame_width,
        )

    def is_turn_pending(self) -> bool:
        """Check if a turn verification is currently active."""
        return self.status in ("IN_PROGRESS", "VERIFYING")

    def start_verification(
        self,
        turn_direction: str,
        landmark: Optional[Dict[str, Any]] = None,
        scene_depth: Optional[Any] = None,
        free_path: Optional[Dict[str, Any]] = None,
        initial_landmark_pos: Optional[str] = None,
        initial_depth: Optional[float] = None,
        frame_width: float = 640.0,
    ) -> None:
        """Initiate turn verification window."""
        turn_dir = str(turn_direction).upper()
        if turn_dir not in ("TURN_LEFT", "TURN_RIGHT"):
            self.reset()
            return

        self.pending_turn = turn_dir
        self.status = "IN_PROGRESS"
        self.frames_waited = 0

        # Landmark position & cx
        if landmark and isinstance(landmark, dict):
            self.initial_landmark_pos = str(landmark.get("position", "CENTER")).upper()
            bbox = landmark.get("bbox")
            if bbox and frame_width > 0:
                bx1, _, bx2, _ = self._extract_bbox(bbox)
                self.initial_landmark_cx = ((bx1 + bx2) / 2.0) / float(frame_width)
            else:
                self.initial_landmark_cx = 0.5
        else:
            self.initial_landmark_pos = str(initial_landmark_pos).upper() if initial_landmark_pos else None
            self.initial_landmark_cx = None

        # Depth
        if initial_depth is not None:
            self.initial_depth = float(initial_depth)
        elif scene_depth is not None:
            if isinstance(scene_depth, (int, float)):
                self.initial_depth = float(scene_depth)
            elif isinstance(scene_depth, dict):
                self.initial_depth = float(scene_depth.get("center_depth", scene_depth.get("average_depth", 1.0)))
                self.initial_depth_profile = {
                    "left": float(scene_depth.get("left_depth", 1.0)),
                    "center": float(scene_depth.get("center_depth", 1.0)),
                    "right": float(scene_depth.get("right_depth", 1.0)),
                }

        # Free-path
        fp = free_path or {}
        self.initial_best_direction = fp.get("best_direction")

    def verify_frame(
        self,
        current_landmark: Optional[Dict[str, Any]] = None,
        current_depth: Optional[float] = None,
        scene_depth: Optional[Any] = None,
        free_path: Optional[Dict[str, Any]] = None,
        frame_width: float = 640.0,
        frame_height: float = 480.0,
    ) -> Tuple[bool, str]:
        """Convenience wrapper for verify returning (is_verified, reason)."""
        depth_param: Any = current_depth if current_depth is not None else scene_depth
        status, verified, reason = self._evaluate(
            current_landmark=current_landmark,
            scene_depth=depth_param,
            free_path=free_path,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        return verified, reason

    def verify(
        self,
        current_landmark: Optional[Dict[str, Any]] = None,
        scene_depth: Optional[Any] = None,
        free_path: Optional[Dict[str, Any]] = None,
        frame_width: float = 640.0,
        frame_height: float = 480.0,
    ) -> Tuple[str, bool]:
        """Inspect incoming frame for visual change evidencing rotation."""
        status, verified, _ = self._evaluate(
            current_landmark=current_landmark,
            scene_depth=scene_depth,
            free_path=free_path,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        return status, verified

    def _evaluate(
        self,
        current_landmark: Optional[Dict[str, Any]] = None,
        scene_depth: Optional[Any] = None,
        free_path: Optional[Dict[str, Any]] = None,
        frame_width: float = 640.0,
        frame_height: float = 480.0,
    ) -> Tuple[str, bool, str]:
        if self.status not in ("IN_PROGRESS", "VERIFYING"):
            return self.status, (self.status == "VERIFIED"), "Not in verification window"

        self.frames_waited += 1
        verified = False
        reason = ""

        # 1. Landmark position shift
        if current_landmark and isinstance(current_landmark, dict):
            curr_pos = str(current_landmark.get("position", "")).upper()
            bbox = current_landmark.get("bbox")
            curr_cx = None
            if bbox and frame_width > 0:
                bx1, _, bx2, _ = self._extract_bbox(bbox)
                curr_cx = ((bx1 + bx2) / 2.0) / float(frame_width)

            if self.pending_turn == "TURN_RIGHT":
                if self.initial_landmark_pos in ("RIGHT", "CENTER") and curr_pos in ("CENTER", "LEFT"):
                    verified = True
                    reason = f"Landmark shifted from {self.initial_landmark_pos} to {curr_pos}"
                elif self.initial_landmark_cx is not None and curr_cx is not None:
                    if (self.initial_landmark_cx - curr_cx) >= 0.15:
                        verified = True
                        reason = f"Landmark cx shifted leftward by {self.initial_landmark_cx - curr_cx:.2f}"

            elif self.pending_turn == "TURN_LEFT":
                if self.initial_landmark_pos in ("LEFT", "CENTER") and curr_pos in ("CENTER", "RIGHT"):
                    verified = True
                    reason = f"Landmark shifted from {self.initial_landmark_pos} to {curr_pos}"
                elif self.initial_landmark_cx is not None and curr_cx is not None:
                    if (curr_cx - self.initial_landmark_cx) >= 0.15:
                        verified = True
                        reason = f"Landmark cx shifted rightward by {curr_cx - self.initial_landmark_cx:.2f}"

        # 2. Scene depth redistribution
        if not verified:
            if isinstance(scene_depth, (int, float)) and self.initial_depth is not None:
                delta = abs(float(scene_depth) - self.initial_depth)
                if delta >= 0.20:
                    verified = True
                    reason = f"Depth delta {delta:.2f} >= 0.20"
            elif isinstance(scene_depth, dict) and self.initial_depth_profile:
                curr_left = float(scene_depth.get("left_depth", 1.0))
                curr_right = float(scene_depth.get("right_depth", 1.0))
                l_delta = abs(curr_left - self.initial_depth_profile.get("left", 1.0))
                r_delta = abs(curr_right - self.initial_depth_profile.get("right", 1.0))
                if l_delta >= 0.20 or r_delta >= 0.20:
                    verified = True
                    reason = f"Depth redistribution l_delta={l_delta:.2f}, r_delta={r_delta:.2f}"

        # 3. Free-path orientation shift
        if not verified and free_path and isinstance(free_path, dict):
            curr_best = free_path.get("best_direction")
            if self.initial_best_direction and curr_best and curr_best != self.initial_best_direction:
                verified = True
                reason = f"Free path shifted from {self.initial_best_direction} to {curr_best}"

        if verified:
            self.status = "VERIFIED"
            return "VERIFIED", True, reason

        if self.frames_waited >= self.max_verification_frames:
            self.status = "TIMEOUT"
            return "TIMEOUT", False, f"Timeout after {self.frames_waited} frames without visual change"

        return "IN_PROGRESS", False, f"Waiting for visual change ({self.frames_waited}/{self.max_verification_frames})"

    def get_recovery_prompt(self, destination_label: Optional[str] = None) -> str:
        """Return recovery prompt with cooldown suppression."""
        res = self.get_recovery_instruction(destination_label)
        return res or ""

    def get_recovery_instruction(self, destination_label: Optional[str] = None) -> Optional[str]:
        """Formulate controlled turn recovery guidance if turn verification timed out."""
        if self.status not in ("TIMEOUT", "TIMED_OUT"):
            return None

        now = time.time()
        if (now - self.last_prompt_time) < self.retry_cooldown_s:
            return None

        self.last_prompt_time = now
        self.frames_waited = 0
        self.status = "IN_PROGRESS"

        turn_desc = "turn right" if self.pending_turn == "TURN_RIGHT" else "turn left"
        if destination_label:
            return f"Please {turn_desc} toward {destination_label}."
        return f"Please {turn_desc} and continue."

    def reset(self) -> None:
        """Reset verifier to idle state."""
        self.status = "NOT_REQUIRED"
        self.pending_turn = "NONE"
        self.frames_waited = 0
        self.initial_landmark_pos = None
        self.initial_landmark_cx = None
        self.initial_depth_profile.clear()
        self.initial_depth = None
        self.initial_best_direction = None
