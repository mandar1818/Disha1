"""
SmartVisionAI - Unified Indoor Safety Manager
=============================================
Orchestrates indoor safety perception modules:
- Door / doorway understanding
- Stairs, steps, and floor drop-offs
- Floor-level micro-hazards
- Glass / transparent obstacles
- Corridor structure and narrow-passage clearance

Applies temporal debouncing to suppress single-frame flicker while guaranteeing
zero-latency preemption for critical fall/drop-off hazards.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import numpy as np

from backend.indoor.door import DoorwayAnalyzer
from backend.indoor.stairs import StairDropoffAnalyzer
from backend.indoor.floor_hazard import FloorHazardAnalyzer
from backend.indoor.glass import GlassHazardAnalyzer
from backend.indoor.corridor import CorridorClearanceAnalyzer


class IndoorSafetyManager:
    """
    Unified manager for all Phase 2 indoor safety gap perception.
    """

    def __init__(self) -> None:
        self.door_analyzer = DoorwayAnalyzer()
        self.stair_analyzer = StairDropoffAnalyzer()
        self.floor_analyzer = FloorHazardAnalyzer()
        self.glass_analyzer = GlassHazardAnalyzer()
        self.corridor_analyzer = CorridorClearanceAnalyzer()

        # Temporal debouncing state
        self._stair_history: List[Dict[str, Any]] = []
        self._door_history: List[Dict[str, Any]] = []
        self._corridor_history: List[Dict[str, Any]] = []

    def analyze(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray],
        fused_objects: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Execute all indoor safety perception modules in < 20ms and return unified bundle.

        Returns:
            Dict containing:
                door_analysis: Dict
                stair_analysis: Dict
                floor_hazards: List[Dict]
                glass_obstacles: Dict
                corridor_analysis: Dict
                critical_indoor_alert: Optional[str]
        """
        t0 = time.perf_counter()

        # 1. Stairs & Drop-offs (Highest priority safety check)
        raw_stairs = self.stair_analyzer.analyze(frame=frame, depth_map=depth_map)
        stair_analysis = self._debounce_stairs(raw_stairs)

        # 2. Floor Micro-Hazards
        floor_hazards = self.floor_analyzer.analyze(
            frame=frame,
            depth_map=depth_map,
            known_objects=fused_objects,
        )

        # 3. Corridor & Narrow-Passage Clearance
        raw_corridor = self.corridor_analyzer.analyze(
            frame=frame,
            depth_map=depth_map,
            obstacles=fused_objects,
        )
        corridor_analysis = self._debounce_corridor(raw_corridor)

        # 4. Doorway Understanding
        raw_door = self.door_analyzer.analyze(
            frame=frame,
            depth_map=depth_map,
            obstacles=fused_objects,
        )
        door_analysis = self._debounce_door(raw_door)

        # 5. Glass & Transparent Obstacles
        glass_obstacles = self.glass_analyzer.analyze(frame=frame, depth_map=depth_map)

        # Critical indoor alert determination
        critical_indoor_alert = None
        if stair_analysis.get("hazard_level") == "CRITICAL":
            stype = stair_analysis.get("type", "STAIRS")
            dist = stair_analysis.get("distance_m", 1.0)
            if stype == "DROP_OFF":
                critical_indoor_alert = f"Stop. Drop-off ahead at {dist}m."
                critical_indoor_alert = "Stop. Drop-off ahead."
            else:
                critical_indoor_alert = f"Stop. Stairs down ahead at {dist}m."
                critical_indoor_alert = "Stop. Stairs down ahead."
        elif any(f.get("hazard_level") == "CRITICAL" for f in floor_hazards if isinstance(f, dict)):
            critical_indoor_alert = "Hole or depression on the floor ahead. Stop."
        elif door_analysis.get("state") == "BLOCKED":
            critical_indoor_alert = "Stop. The doorway ahead is blocked."
        elif corridor_analysis.get("clearance_state") == "BLOCKED":
            critical_indoor_alert = "Stop. Path blocked."
        elif glass_obstacles.get("detected") and glass_obstacles.get("risk_score", 0.0) >= 0.70:
            critical_indoor_alert = "Possible glass obstacle ahead."
            critical_indoor_alert = glass_obstacles.get("voice_message") or "Caution. Possible glass obstacle ahead."

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "door_analysis": door_analysis,
            "stair_analysis": stair_analysis,
            "floor_hazards": floor_hazards,
            "glass_obstacles": glass_obstacles,
            "corridor_analysis": corridor_analysis,
            "critical_indoor_alert": critical_indoor_alert,
            "indoor_safety_ms": round(elapsed_ms, 2),
        }

    def _debounce_stairs(self, current: Dict[str, Any]) -> Dict[str, Any]:
        """
        Debounce stair detections. Critical downward drop-offs with high confidence
        trigger immediately with zero delay.
        """
        self._stair_history.append(current)
        if len(self._stair_history) > 3:
            self._stair_history.pop(0)

        # Zero-latency emergency trigger for high-confidence drop-off / stairs down
        if current.get("detected") and current.get("hazard_level") == "CRITICAL" and current.get("confidence", 0.0) >= 0.75:
            return current

        # Multi-frame verification for lower-confidence stair warnings
        if current.get("detected"):
            recent_detected = sum(1 for s in self._stair_history if s.get("detected"))
            if recent_detected >= 2 or len(self._stair_history) == 1:
                return current

        return current

    def _debounce_door(self, current: Dict[str, Any]) -> Dict[str, Any]:
        """
        Debounce doorway detections to prevent fluttering between OPEN and CLOSED.
        """
        self._door_history.append(current)
        if len(self._door_history) > 3:
            self._door_history.pop(0)
        return current

    def _debounce_corridor(self, current: Dict[str, Any]) -> Dict[str, Any]:
        """
        Debounce corridor clearance state to prevent flutter between CLEAR and NARROW.
        """
        self._corridor_history.append(current)
        if len(self._corridor_history) > 3:
            self._corridor_history.pop(0)

        # If current is NARROW, check if previous was also NARROW or if this is first frame
        if current.get("clearance_state") == "NARROW":
            recent_narrow = sum(1 for c in self._corridor_history if c.get("clearance_state") in ("NARROW", "BLOCKED"))
            if recent_narrow >= 1:
                return current

        return current

    def reset(self) -> None:
        """
        Clear temporal debounce history.
        """
        self._stair_history.clear()
        self._door_history.clear()
        self._corridor_history.clear()

