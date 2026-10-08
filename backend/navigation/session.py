"""
SmartVisionAI - Closed-Loop Navigation Session Coordinator
=========================================================
Manages indoor navigation session state across consecutive camera frames:
- Destination lifecycle: IDLE -> SEARCHING -> TURNING -> VERIFYING -> APPROACHING -> ARRIVED / LOST
- Integrates TurnVerifier and ProgressTracker
- Enforces strict safety precedence (safety hazards strictly override destination guidance)
- Prevents oscillation and manages voice guidance cooldown
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from backend.navigation.destination import Destination, match_destination
from backend.navigation.verifier import TurnVerifier
from backend.navigation.progress import ProgressTracker


class NavigationSessionState(str, Enum):
    IDLE = "IDLE"
    SEARCHING = "SEARCHING"
    APPROACHING = "APPROACHING"
    TURNING = "TURNING"
    VERIFYING = "VERIFYING"
    AVOIDING = "AVOIDING"
    ARRIVED = "ARRIVED"
    LOST = "LOST"
    CANCELLED = "CANCELLED"


class NavigationSession:
    """
    Stateful coordinator for destination navigation across continuous frames.
    """

    def __init__(
        self,
        searching_prompt_cooldown_s: float = 6.0,
        arrival_cooldown_s: float = 10.0,
    ) -> None:
        self.destination: Optional[Destination] = None
        self.state: str = NavigationSessionState.IDLE
        self.turn_verifier = TurnVerifier()
        self.progress_tracker = ProgressTracker()

        self.last_seen_landmark: Optional[Dict[str, Any]] = None
        self.last_search_prompt_time: float = 0.0
        self.searching_prompt_cooldown_s = float(searching_prompt_cooldown_s)
        self.arrival_announced: bool = False
        self.arrival_time: float = 0.0
        self.arrival_cooldown_s = float(arrival_cooldown_s)

    @property
    def active(self) -> bool:
        """Check if destination navigation is actively engaged."""
        return bool(
            self.destination
            and self.destination.active
            and self.state not in (NavigationSessionState.IDLE, NavigationSessionState.CANCELLED)
        )

    @property
    def missing_frames_count(self) -> int:
        """Expose count of frames missing the target landmark."""
        return self.progress_tracker.missing_frames

    def start(
        self,
        destination: Any,
        identifier: Optional[str] = None,
        label: Optional[str] = None,
    ) -> Destination:
        """Convenience alias for start_destination."""
        return self.start_destination(destination, identifier=identifier, label=label)

    def stop(self) -> None:
        """Convenience alias for stop_destination."""
        self.stop_destination()

    def start_destination(
        self,
        destination: Any = None,
        dest_type: Any = None,
        identifier: Optional[str] = None,
        label: Optional[str] = None,
    ) -> Destination:
        """
        Start a new navigation session toward the requested destination.
        """
        target = destination if destination is not None else dest_type
        if isinstance(target, Destination):
            self.destination = target
            self.destination.active = True
        elif isinstance(target, dict):
            self.destination = Destination.from_dict(target) or Destination(str(target.get("name", "Destination")))
        else:
            self.destination = Destination(
                dest_type=str(target),
                identifier=identifier,
                label=label,
                active=True,
            )

        self.state = NavigationSessionState.SEARCHING
        self.turn_verifier.reset()
        self.progress_tracker.reset()
        self.last_seen_landmark = None
        self.last_search_prompt_time = 0.0
        self.arrival_announced = False
        self.arrival_time = 0.0
        return self.destination

    def stop_destination(self) -> NavigationSession:
        """
        Cancel active destination navigation and return self.
        """
        self.state = NavigationSessionState.IDLE
        if self.destination:
            self.destination.active = False
        self.turn_verifier.reset()
        self.progress_tracker.reset()
        return self

    def get_status(self) -> Dict[str, Any]:
        """Return structured session status."""
        return {
            "active": self.active,
            "state": self.state,
            "destination": self.destination.to_dict() if self.destination else None,
            "destination_visible": bool(self.last_seen_landmark is not None and self.progress_tracker.missing_frames == 0),
            "destination_position": self.last_seen_landmark.get("position", "UNKNOWN") if self.last_seen_landmark else "UNKNOWN",
            "progress": self.progress_tracker.current_proximity,
            "turn_verification": {
                "status": self.turn_verifier.status,
                "pending_turn": self.turn_verifier.pending_turn,
                "frames_waited": self.turn_verifier.frames_waited,
            },
            "arrival": self.progress_tracker.has_arrived,
        }

    def get_recommended_action(self) -> str:
        """Return the current high-level recommended action for destination progress."""
        if self.state == NavigationSessionState.TURNING:
            return self.turn_verifier.pending_turn or "TURN"
        elif self.state == NavigationSessionState.APPROACHING:
            return "MOVE_FORWARD"
        elif self.state == NavigationSessionState.ARRIVED:
            return "STOP"
        elif self.state == NavigationSessionState.AVOIDING:
            return "AVOID"
        return "NONE"

    def update(
        self,
        semantic_landmarks: Optional[List[Dict[str, Any]]] = None,
        scene_depth: Optional[Any] = None,
        primary_obstacle: Optional[Dict[str, Any]] = None,
        free_path: Optional[Dict[str, Any]] = None,
        safety_level: str = "SAFE",
        emergency_stop: bool = False,
        indoor_safety: Optional[Dict[str, Any]] = None,
        frame_width: float = 640.0,
        frame_height: float = 480.0,
    ) -> Dict[str, Any]:
        """
        Evaluate incoming frame and update navigation state machine.

        Returns:
            Dictionary containing session state, recommended destination command,
            and safety override indicators.
        """
        if not self.destination or not self.destination.active or self.state in (NavigationSessionState.IDLE, NavigationSessionState.CANCELLED):
            return {
                "active": False,
                "state": self.state,
                "destination": None,
                "destination_visible": False,
                "safety_override": False,
                "voice_instruction": None,
                "navigation_action": None,
                "arrival": False,
            }

        landmarks = list(semantic_landmarks or [])
        depth_info = scene_depth or {}
        fp = free_path or {}

        # ----------------------------------------------------
        # 1. Match destination against semantic landmarks
        # ----------------------------------------------------
        matched_landmark: Optional[Dict[str, Any]] = None
        for lm in landmarks:
            is_match, conf = match_destination(self.destination, lm)
            if is_match:
                matched_landmark = lm
                self.last_seen_landmark = lm
                break

        # ----------------------------------------------------
        # 2. Update Progress & Arrival
        # ----------------------------------------------------
        prog_res = self.progress_tracker.update(
            matched_landmark=matched_landmark,
            scene_depth=depth_info,
            safety_level=safety_level,
            emergency_stop=emergency_stop,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        if prog_res.get("arrived"):
            self.state = NavigationSessionState.ARRIVED
            voice = None
            if not self.arrival_announced:
                self.arrival_announced = True
                self.arrival_time = time.time()
                voice = self.progress_tracker.format_arrival_speech(self.destination.label)

            return {
                "active": True,
                "state": NavigationSessionState.ARRIVED,
                "destination": self.destination.to_dict(),
                "destination_visible": (matched_landmark is not None),
                "destination_position": matched_landmark.get("position", "CENTER") if matched_landmark else "CENTER",
                "progress": self.progress_tracker.current_proximity,
                "safety_override": False,
                "voice_instruction": voice,
                "navigation_action": "STOP",
                "arrival": True,
                "turn_verification": {
                    "status": "NOT_REQUIRED",
                    "pending_turn": "NONE",
                },
            }

        # ----------------------------------------------------
        # 3. STRICT SAFETY OVERRIDE
        # ----------------------------------------------------
        is_hazard = False
        obs_dist = float(primary_obstacle.get("distance", 999.0)) if primary_obstacle else 999.0
        obs_risk = float(primary_obstacle.get("risk_score", 0.0) or primary_obstacle.get("risk", 0.0)) if primary_obstacle else 0.0
        obs_pos = str(primary_obstacle.get("position", "")).upper() if primary_obstacle else ""
        obs_app = bool(primary_obstacle.get("approaching") or primary_obstacle.get("is_approaching") or primary_obstacle.get("rapidly_approaching")) if primary_obstacle else False
        obs_cross = bool(primary_obstacle.get("is_crossing") or primary_obstacle.get("person_state") == "PERSON_CROSSING") if primary_obstacle else False

        if (
            emergency_stop
            or safety_level in ("EMERGENCY", "DANGER")
            or obs_risk >= 0.70
            or obs_dist <= 0.80
            or (obs_pos == "CENTER" and obs_dist <= 1.8 and (obs_app or obs_cross or obs_risk >= 0.50))
            or fp.get("no_safe_path")
        ):
            is_hazard = True

        if indoor_safety and isinstance(indoor_safety, dict):
            stair = indoor_safety.get("stair_analysis", {})
            if stair.get("detected") and (stair.get("drop_off_risk") or stair.get("hazard_level") in ("CRITICAL", "CAUTION")):
                is_hazard = True
            corridor = indoor_safety.get("corridor_analysis", {})
            if corridor.get("clearance_state") == "BLOCKED":
                is_hazard = True
            door = indoor_safety.get("door_analysis", {})
            if door.get("detected") and door.get("state") in ("CLOSED", "BLOCKED"):
                is_hazard = True
            floors = indoor_safety.get("floor_hazards", [])
            if any(f.get("hazard_level") in ("CRITICAL", "CAUTION") for f in floors if isinstance(f, dict)):
                is_hazard = True
            glass = indoor_safety.get("glass_obstacles", {})
            if glass.get("detected") and glass.get("risk_score", 0.0) >= 0.70:
                is_hazard = True

        if is_hazard:
            self.state = NavigationSessionState.AVOIDING
            return {
                "active": True,
                "state": NavigationSessionState.AVOIDING,
                "destination": self.destination.to_dict(),
                "destination_visible": (matched_landmark is not None),
                "destination_position": matched_landmark.get("position", "UNKNOWN") if matched_landmark else "UNKNOWN",
                "progress": self.progress_tracker.current_proximity,
                "safety_override": True,
                "voice_instruction": None,
                "navigation_action": None,
                "arrival": False,
                "turn_verification": {
                    "status": self.turn_verifier.status,
                    "pending_turn": self.turn_verifier.pending_turn,
                },
            }

        # ----------------------------------------------------
        # 4. TURN VERIFICATION
        # ----------------------------------------------------
        if self.turn_verifier.status in ("IN_PROGRESS", "VERIFYING"):
            status_str, is_verified = self.turn_verifier.verify(
                current_landmark=matched_landmark,
                scene_depth=depth_info,
                free_path=fp,
                frame_width=frame_width,
            )

            if is_verified:
                target_pos = str(matched_landmark.get("position", "CENTER")).upper() if matched_landmark else "CENTER"
                if target_pos == "CENTER":
                    self.state = NavigationSessionState.APPROACHING
                else:
                    self.state = NavigationSessionState.APPROACHING if matched_landmark else NavigationSessionState.SEARCHING
            elif status_str in ("TIMEOUT", "TIMED_OUT"):
                recovery_cue = self.turn_verifier.get_recovery_instruction(self.destination.label)
                if recovery_cue:
                    return {
                        "active": True,
                        "state": NavigationSessionState.VERIFYING,
                        "destination": self.destination.to_dict(),
                        "destination_visible": (matched_landmark is not None),
                        "destination_position": matched_landmark.get("position", "UNKNOWN") if matched_landmark else "UNKNOWN",
                        "progress": self.progress_tracker.current_proximity,
                        "safety_override": False,
                        "voice_instruction": recovery_cue,
                        "navigation_action": self.turn_verifier.pending_turn,
                        "arrival": False,
                        "turn_verification": {
                            "status": "TIMEOUT",
                            "pending_turn": self.turn_verifier.pending_turn,
                        },
                    }

        # ----------------------------------------------------
        # 5. DESTINATION GUIDANCE
        # ----------------------------------------------------
        if matched_landmark:
            target_pos = str(matched_landmark.get("position", "CENTER")).upper()

            if target_pos == "RIGHT":
                self.state = NavigationSessionState.TURNING
                self.turn_verifier.start_verification(
                    turn_direction="TURN_RIGHT",
                    landmark=matched_landmark,
                    scene_depth=depth_info,
                    free_path=fp,
                    frame_width=frame_width,
                )
                return {
                    "active": True,
                    "state": NavigationSessionState.TURNING,
                    "destination": self.destination.to_dict(),
                    "destination_visible": True,
                    "destination_position": "RIGHT",
                    "progress": self.progress_tracker.current_proximity,
                    "safety_override": False,
                    "voice_instruction": "Turn right.",
                    "navigation_action": "TURN_RIGHT",
                    "arrival": False,
                    "turn_verification": {
                        "status": "IN_PROGRESS",
                        "pending_turn": "TURN_RIGHT",
                    },
                }

            elif target_pos == "LEFT":
                self.state = NavigationSessionState.TURNING
                self.turn_verifier.start_verification(
                    turn_direction="TURN_LEFT",
                    landmark=matched_landmark,
                    scene_depth=depth_info,
                    free_path=fp,
                    frame_width=frame_width,
                )
                return {
                    "active": True,
                    "state": NavigationSessionState.TURNING,
                    "destination": self.destination.to_dict(),
                    "destination_visible": True,
                    "destination_position": "LEFT",
                    "progress": self.progress_tracker.current_proximity,
                    "safety_override": False,
                    "voice_instruction": "Turn left.",
                    "navigation_action": "TURN_LEFT",
                    "arrival": False,
                    "turn_verification": {
                        "status": "IN_PROGRESS",
                        "pending_turn": "TURN_LEFT",
                    },
                }

            else:  # CENTER or NONE
                self.state = NavigationSessionState.APPROACHING
                return {
                    "active": True,
                    "state": NavigationSessionState.APPROACHING,
                    "destination": self.destination.to_dict(),
                    "destination_visible": True,
                    "destination_position": "CENTER",
                    "progress": self.progress_tracker.current_proximity,
                    "safety_override": False,
                    "voice_instruction": "Continue forward.",
                    "navigation_action": "FORWARD",
                    "arrival": False,
                    "turn_verification": {
                        "status": "NOT_REQUIRED",
                        "pending_turn": "NONE",
                    },
                }

        # ----------------------------------------------------
        # 6. DESTINATION NOT VISIBLE (SEARCHING / LOST)
        # ----------------------------------------------------
        missing_count = self.progress_tracker.missing_frames
        if missing_count >= 3:
            self.state = NavigationSessionState.LOST
            now = time.time()
            voice = None
            if (now - self.last_search_prompt_time) >= self.searching_prompt_cooldown_s:
                self.last_search_prompt_time = now
                voice = "Destination not visible. Please continue slowly."

            return {
                "active": True,
                "state": NavigationSessionState.LOST,
                "destination": self.destination.to_dict(),
                "destination_visible": False,
                "destination_position": "UNKNOWN",
                "progress": self.progress_tracker.current_proximity,
                "safety_override": False,
                "voice_instruction": voice,
                "navigation_action": "FORWARD",
                "arrival": False,
                "turn_verification": {
                    "status": "NOT_REQUIRED",
                    "pending_turn": "NONE",
                },
            }
        else:
            self.state = NavigationSessionState.SEARCHING
            return {
                "active": True,
                "state": NavigationSessionState.SEARCHING,
                "destination": self.destination.to_dict(),
                "destination_visible": False,
                "destination_position": "UNKNOWN",
                "progress": self.progress_tracker.current_proximity,
                "safety_override": False,
                "voice_instruction": None,
                "navigation_action": "FORWARD",
                "arrival": False,
                "turn_verification": {
                    "status": "NOT_REQUIRED",
                    "pending_turn": "NONE",
                },
            }
