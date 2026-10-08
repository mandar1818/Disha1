"""
============================================================
SmartVisionAI
============================================================

File:
    backend/decision/engine.py

Purpose:
    Decision Engine for SmartVisionAI.

Responsibilities:
    - Filter invalid detections
    - Remove duplicate detections using IoU
    - Assign object priorities
    - Determine object position
    - Combine confidence + depth + priority
    - Calculate risk score
    - Identify primary obstacle
    - Analyze free paths
    - Determine safety level
    - Emergency-stop logic
    - Navigation decision
    - Multi-object reasoning
    - Generate voice instruction

Important:
    MiDaS provides RELATIVE depth, not guaranteed metric distance.

============================================================
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

from backend.config import DEFAULT_YOLO_CONFIDENCE


# ============================================================
# DECISION ENGINE
# ============================================================

class DecisionEngine:
    """
    Rule-based decision engine for SmartVisionAI.

    The engine converts YOLO + MiDaS information into:

        risk
        safety level
        emergency stop
        navigation
        voice instruction
    """

    # --------------------------------------------------------
    # Thresholds
    # --------------------------------------------------------

    EMERGENCY_THRESHOLD = 0.18
    AWARE_THRESHOLD = 0.32
    CAUTION_THRESHOLD = 0.50
    DANGER_THRESHOLD = 0.68

    # --------------------------------------------------------
    # General configuration
    # --------------------------------------------------------

    DEFAULT_CONFIDENCE = DEFAULT_YOLO_CONFIDENCE
    IOU_THRESHOLD = 0.60

    # Smaller MiDaS value = closer object.
    # These values are relative-depth thresholds.

    VERY_CLOSE_DEPTH = 0.20
    CLOSE_DEPTH = 0.35
    MEDIUM_DEPTH = 0.55

    # --------------------------------------------------------
    # COCO object priorities
    # --------------------------------------------------------

    OBJECT_PRIORITIES: Dict[str, float] = {
        # Humans
        "person": 1.00,

        # Vehicles
        "car": 0.95,
        "bus": 0.95,
        "truck": 0.95,
        "motorcycle": 0.90,
        "bicycle": 0.85,
        "train": 0.90,

        # Large / important obstacles
        "dining table": 0.75,
        "table": 0.75,
        "chair": 0.55,
        "bench": 0.65,
        "couch": 0.65,
        "bed": 0.60,

        # Common objects
        "backpack": 0.45,
        "suitcase": 0.55,
        "handbag": 0.40,

        "bottle": 0.15,
        "cup": 0.15,
        "book": 0.10,
        "cell phone": 0.15,
        "laptop": 0.15,

        # Animals
        "dog": 0.65,
        "cat": 0.55,
        "horse": 0.70,
        "cow": 0.70,
        "sheep": 0.60,

        # Outdoor obstacles
        "traffic light": 0.45,
        "stop sign": 0.60,
        "fire hydrant": 0.60,

        # Sports / miscellaneous
        "sports ball": 0.20,
        "skateboard": 0.40,
        "surfboard": 0.30,

        # Kitchen / household
        "refrigerator": 0.70,
        "microwave": 0.30,
        "oven": 0.50,
        "sink": 0.45,
        "toilet": 0.35,

        # Default environmental objects
        "potted plant": 0.25,
        "tv": 0.20,
        "keyboard": 0.10,
        "mouse": 0.10,
    }

    # --------------------------------------------------------
    # Human / vehicle overrides
    # --------------------------------------------------------

    HUMAN_CLASSES = {
        "person",
    }

    VEHICLE_CLASSES = {
        "car",
        "bus",
        "truck",
        "motorcycle",
        "bicycle",
        "train",
    }

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        emergency_threshold: float = EMERGENCY_THRESHOLD,
        danger_threshold: float = DANGER_THRESHOLD,
        caution_threshold: float = CAUTION_THRESHOLD,
        aware_threshold: float = AWARE_THRESHOLD,
    ) -> None:

        self.emergency_threshold = float(
            emergency_threshold
        )

        self.danger_threshold = float(
            danger_threshold
        )

        self.caution_threshold = float(
            caution_threshold
        )

        self.aware_threshold = float(
            aware_threshold
        )

        self.loaded = True
        self.track_history: List[Dict[str, Any]] = []
        self.object_tracks: Dict[str, Dict[str, Any]] = {}
        self._next_track_num: int = 1
        self.frame_index: int = 0
        self.last_action: str = "FORWARD"
        self.candidate_action: str = "FORWARD"
        self.action_vote_count: int = 0
        self.last_direction: str = "CENTER"
        self.direction_vote_count: int = 0
        self.turn_state: str = "NONE"
        self.step_length_m: float = 0.75
        self.safety_margin_m: float = 0.75
        self.max_guidance_distance_m: float = 6.0
        self.last_steps: int = 0
        # Phase 3: Closed-Loop Navigation state tracking
        self.closed_loop_state: str = "CLEAR_PATH"
        self.last_maneuver_command: str = "NONE"
        # Navigation state machine and STOP suppression tracking
        self.nav_state: str = "FORWARD"
        self.last_nav_state: str = "FORWARD"
        self.stop_frame_count: int = 0
        self.stop_threat_id: Any = None
        self.stop_threat_class: str = ""
        self.stop_timestamp: float = 0.0
        self.state_changed: bool = False
        self.should_speak: bool = True
        self.recovery_suggested: bool = False
        self.path_cleared_from_stop: bool = False

    def reset(self) -> None:
        """Reset all internal tracking, history, and hysteresis state."""
        self.track_history = []
        self.object_tracks = {}
        self._next_track_num = 1
        self.frame_index = 0
        self.last_action = "FORWARD"
        self.candidate_action = "FORWARD"
        self.action_vote_count = 0
        self.last_direction = "CENTER"
        self.direction_vote_count = 0
        self.turn_state = "NONE"
        self.last_steps = 0
        self.closed_loop_state = "CLEAR_PATH"
        self.last_maneuver_command = "NONE"
        self.nav_state = "FORWARD"
        self.last_nav_state = "FORWARD"
        self.stop_frame_count = 0
        self.stop_threat_id = None
        self.stop_threat_class = ""
        self.stop_timestamp = 0.0
        self.state_changed = False
        self.should_speak = True
        self.recovery_suggested = False
        self.path_cleared_from_stop = False

    # ========================================================
    # STATUS
    # ========================================================

    def status(self) -> Dict[str, Any]:
        """
        Return engine status.
        """

        return {
            "loaded": True,
            "engine": "DecisionEngine",
            "emergency_threshold":
                self.emergency_threshold,
            "danger_threshold":
                self.danger_threshold,
            "caution_threshold":
                self.caution_threshold,
            "aware_threshold":
                self.aware_threshold,
            "closed_loop_state":
                self.closed_loop_state,
            "last_maneuver_command":
                self.last_maneuver_command,
        }

    def decide(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Convenience alias for analyze().
        """
        return self.analyze(*args, **kwargs)

    # ========================================================
    # NUMERIC HELPERS
    # ========================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:
        """
        Safely convert a value to float.
        """

        try:
            result = float(value)

            if result != result:
                return default

            return result

        except Exception:
            return default

    @staticmethod
    def _clamp(
        value: float,
        minimum: float = 0.0,
        maximum: float = 1.0,
    ) -> float:
        """
        Clamp a number between minimum and maximum.
        """

        return max(
            minimum,
            min(
                maximum,
                value,
            ),
        )

    # ========================================================
    # WALKING STEPS & MOVEMENT DISTANCE (PHASE 10 & 11)
    # ========================================================

    @classmethod
    def estimate_walking_steps(
        cls,
        distance_m: Any,
        step_length_m: Any = 0.75,
    ) -> int:
        """
        Convert distance in meters to approximate walking steps.
        Bounded: [1, 8] for positive finite distance, 8 for +inf, 0 for <= 0 or NaN/invalid.
        """
        try:
            d = float(distance_m)
            if math.isnan(d) or d <= 0.0:
                return 0
            if math.isinf(d):
                return 8 if d > 0 else 0

            sl = 0.75
            if step_length_m is not None:
                try:
                    cand_sl = float(step_length_m)
                    if math.isfinite(cand_sl) and 0.35 <= cand_sl <= 2.5:
                        sl = cand_sl
                except Exception:
                    sl = 0.75

            steps = int(round(d / sl))
            return max(1, min(8, steps))
        except Exception:
            return 0

    def estimate_movement_distance(
        self,
        distance_m: Any,
        distance_category: str = "NEAR",
        action: str = "FORWARD",
        free_path: Optional[Dict[str, Any]] = None,
        safety_margin_m: Optional[float] = None,
        max_guidance_distance_m: Optional[float] = None,
    ) -> Tuple[float, str, str]:
        """
        Estimate safe guidance movement distance with safety margin.
        Returns: (movement_distance_m, step_confidence, step_reason)
        """
        try:
            d = float(distance_m)
        except Exception:
            d = 0.0

        cat = str(distance_category).upper() if distance_category else ""
        act = str(action).upper() if action else ""

        if not math.isfinite(d) or d <= 0.0 or cat == "UNKNOWN_DISTANCE":
            if act in ("FORWARD", "GO_FORWARD", "CONTINUE_FORWARD") and cat != "UNKNOWN_DISTANCE":
                return 3.0, "HIGH", "Forward corridor clear ahead"
            return 0.0, "LOW", "Unknown obstacle distance; exercise caution"

        if act in ("STOP", "EMERGENCY_STOP"):
            return 0.0, "HIGH", "No movement during stop"

        if act in ("TURN_LEFT", "TURN_RIGHT"):
            return 0.0, "HIGH", "Body turn orientation"

        if act == "MOVE_LEFT":
            return 1.5, "HIGH", "Lateral step left"

        if act == "MOVE_RIGHT":
            return 2.25, "HIGH", "Lateral step right"

        if act in ("BACK", "GO_BACK"):
            return 2.25, "HIGH", "Safe backward movement"

        sm = float(safety_margin_m) if safety_margin_m is not None and safety_margin_m >= 0 else self.safety_margin_m
        max_d = float(max_guidance_distance_m) if max_guidance_distance_m is not None and max_guidance_distance_m > 0 else self.max_guidance_distance_m

        if d <= sm:
            return 0.0, "HIGH", "Obstacle within safety margin; stop forward movement"

        eff = max(0.0, d - sm)
        if act == "SLOW_DOWN":
            eff = max(0.5, eff * 0.5)
            return round(min(max_d, eff), 2), "MEDIUM", "Cautionary forward movement at reduced speed"

        return round(min(max_d, eff), 2), "HIGH", "Forward movement safe"

    # ========================================================
    # BOUNDING BOX
    # ========================================================

    @classmethod
    def _bbox(
        cls,
        obj: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Normalize bounding box.
        """

        bbox = obj.get(
            "bbox",
            {},
        )

        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return {
                "x1": cls._safe_float(bbox[0]),
                "y1": cls._safe_float(bbox[1]),
                "x2": cls._safe_float(bbox[2]),
                "y2": cls._safe_float(bbox[3]),
            }

        if not isinstance(
            bbox,
            dict,
        ):
            bbox = {}

        return {
            "x1": cls._safe_float(
                bbox.get("x1", 0)
            ),
            "y1": cls._safe_float(
                bbox.get("y1", 0)
            ),
            "x2": cls._safe_float(
                bbox.get("x2", 0)
            ),
            "y2": cls._safe_float(
                bbox.get("y2", 0)
            ),
        }

    # ========================================================
    # IOU
    # ========================================================

    @classmethod
    def calculate_iou(
        cls,
        box_a: Dict[str, Any],
        box_b: Dict[str, Any],
    ) -> float:
        """
        Calculate Intersection over Union.

        IoU is used to identify duplicate detections.
        """

        try:

            ax1 = cls._safe_float(
                box_a.get("x1", 0)
            )
            ay1 = cls._safe_float(
                box_a.get("y1", 0)
            )
            ax2 = cls._safe_float(
                box_a.get("x2", 0)
            )
            ay2 = cls._safe_float(
                box_a.get("y2", 0)
            )

            bx1 = cls._safe_float(
                box_b.get("x1", 0)
            )
            by1 = cls._safe_float(
                box_b.get("y1", 0)
            )
            bx2 = cls._safe_float(
                box_b.get("x2", 0)
            )
            by2 = cls._safe_float(
                box_b.get("y2", 0)
            )

            intersection_x1 = max(
                ax1,
                bx1,
            )

            intersection_y1 = max(
                ay1,
                by1,
            )

            intersection_x2 = min(
                ax2,
                bx2,
            )

            intersection_y2 = min(
                ay2,
                by2,
            )

            intersection_width = max(
                0.0,
                intersection_x2
                - intersection_x1,
            )

            intersection_height = max(
                0.0,
                intersection_y2
                - intersection_y1,
            )

            intersection_area = (
                intersection_width
                * intersection_height
            )

            area_a = max(
                0.0,
                ax2 - ax1,
            ) * max(
                0.0,
                ay2 - ay1,
            )

            area_b = max(
                0.0,
                bx2 - bx1,
            ) * max(
                0.0,
                by2 - by1,
            )

            union_area = (
                area_a
                + area_b
                - intersection_area
            )

            if union_area <= 0:
                return 0.0

            return cls._clamp(
                intersection_area
                / union_area
            )

        except Exception:
            return 0.0

    # ========================================================
    # DUPLICATE FILTER
    # ========================================================

    def remove_duplicates(
        self,
        objects: List[Dict[str, Any]],
        reflection_uncertainty: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate detections and mirror reflection ghost duplicates.

        Two detections of the same class with high IoU or matching mirror reflection
        characteristics (similar size, divergent depth) are deduplicated, keeping
        the physical closer obstacle for safety.
        """
        if not objects:
            return []

        sorted_objects = sorted(
            objects,
            key=lambda item: (
                -self._safe_float(item.get("confidence", 0.0)),
                self._safe_float(item.get("depth", 1.0)),
            ),
        )

        kept: List[Dict[str, Any]] = []

        for current in sorted_objects:
            current_class = str(
                current.get("class_name", current.get("label", "unknown"))
            ).lower().strip()
            current_box = self._bbox(current)
            d_curr = self._safe_float(current.get("depth", 1.0))
            duplicate = False

            for idx, existing in enumerate(kept):
                existing_class = str(
                    existing.get("class_name", existing.get("label", "unknown"))
                ).lower().strip()

                if current_class != existing_class:
                    continue

                existing_box = self._bbox(existing)
                iou = self.calculate_iou(current_box, existing_box)

                if iou >= self.IOU_THRESHOLD:
                    duplicate = True
                    break

                # Mirror reflection duplicate check
                if reflection_uncertainty:
                    w1 = abs(self._safe_float(current_box.get("x2", 0)) - self._safe_float(current_box.get("x1", 0)))
                    w2 = abs(self._safe_float(existing_box.get("x2", 0)) - self._safe_float(existing_box.get("x1", 0)))

                    if w1 > 0 and w2 > 0 and abs(w1 - w2) / max(w1, w2) < 0.35:
                        d_exist = self._safe_float(existing.get("depth", 1.0))
                        if abs(d_curr - d_exist) >= 0.35:
                            # Candidate mirror reflection pair!
                            # Keep the closer one (smaller depth = higher physical risk)
                            if d_curr < d_exist:
                                kept[idx] = current  # Replace further virtual reflection with closer physical object
                            duplicate = True
                            break

            if not duplicate:
                kept.append(current)

        return kept

    # ========================================================
    # OBJECT PRIORITY
    # ========================================================

    def get_priority(
        self,
        class_name: str,
    ) -> float:
        """
        Return object priority.

        Humans and vehicles receive explicit priority
        overrides because they are especially important
        for an assistive navigation system.
        """

        name = str(
            class_name
            or "unknown"
        ).lower().strip()

        if name in self.HUMAN_CLASSES:
            return 1.0

        if name in self.VEHICLE_CLASSES:
            return 0.95

        return self._clamp(
            self.OBJECT_PRIORITIES.get(
                name,
                0.30,
            )
        )

    # ========================================================
    # POSITION
    # ========================================================

    def get_position(
        self,
        bbox: Dict[str, Any],
        frame_width: float,
    ) -> str:
        """
        Determine whether an object is:

            LEFT
            CENTER
            RIGHT
        """

        width = self._safe_float(
            frame_width,
            1.0,
        )

        if width <= 0:
            width = 1.0

        x1 = self._safe_float(
            bbox.get("x1", 0)
        )

        x2 = self._safe_float(
            bbox.get("x2", 0)
        )

        center_x = (
            x1 + x2
        ) / 2.0

        normalized_x = (
            center_x
            / width
        )

        norm_x1 = max(0.0, min(1.0, min(x1, x2) / width))
        norm_x2 = max(0.0, min(1.0, max(x1, x2) / width))
        corridor_overlap = max(0.0, min(norm_x2, 0.62) - max(norm_x1, 0.38))

        # Center collision corridor overlap (if wide object encroaches significantly into walking path)
        if corridor_overlap >= 0.08 or ((norm_x2 - norm_x1) >= 0.40 and norm_x1 <= 0.50 <= norm_x2):
            return "CENTER"

        if normalized_x < 0.38:
            return "LEFT"

        if normalized_x > 0.62:
            return "RIGHT"

        return "CENTER"

    # ========================================================
    # DEPTH NORMALIZATION
    # ========================================================

    def normalize_depth(
        self,
        depth: Any,
    ) -> float:
        """
        Normalize relative depth.

        MiDaS output may differ depending on preprocessing.

        The returned value is interpreted as:

            0.0 -> very close
            1.0 -> relatively far
        """

        value = self._safe_float(
            depth,
            1.0,
        )

        return self._clamp(
            value
        )

    # ========================================================
    # DEPTH RISK
    # ========================================================

    def depth_risk(
        self,
        depth: float,
    ) -> float:
        """
        Convert relative depth into risk.

        Smaller depth means greater risk.
        """

        depth = self.normalize_depth(
            depth
        )

        risk = 1.0 - depth

        return self._clamp(
            risk
        )

    # ========================================================
    # OBJECT RISK
    # ========================================================

    def calculate_risk(
        self,
        confidence: float,
        depth: float,
        priority: float,
        position: str,
        bbox: Optional[Dict[str, Any]] = None,
        frame_width: Optional[float] = None,
        frame_height: Optional[float] = None,
        distance_m: Optional[float] = None,
    ) -> float:
        """
        Calculate object risk.

        Risk considers:
            distance (metric proximity)
            relative depth (MiDaS)
            confidence
            object priority
            position (CENTER elevated)
            object size / bbox
        """

        confidence = self._clamp(
            self._safe_float(
                confidence
            )
        )

        depth_risk = self.depth_risk(
            depth
        )

        priority = self._clamp(
            self._safe_float(
                priority
            )
        )

        # ----------------------------------------------------
        # Proximity risk: blend relative depth with distance
        # ----------------------------------------------------
        if distance_m is not None:
            dist_val = self._safe_float(distance_m)
            if dist_val > 0.0 and math.isfinite(dist_val):
                # 0.4m or less = 1.0 (imminent collision), 5.0m or more = 0.0
                dist_risk = self._clamp(1.0 - (dist_val - 0.4) / 4.6)
                prox_risk = 0.50 * depth_risk + 0.50 * dist_risk
            else:
                prox_risk = depth_risk
        else:
            prox_risk = depth_risk

        # ----------------------------------------------------
        # Base risk
        # ----------------------------------------------------
        risk = (
            0.40 * prox_risk
            + 0.30 * priority
            + 0.15 * confidence
        )

        # ----------------------------------------------------
        # Position danger
        # ----------------------------------------------------
        pos_upper = str(position).upper().strip()
        if pos_upper == "CENTER":
            risk += 0.12
        elif pos_upper in {
            "LEFT",
            "RIGHT",
        }:
            risk += 0.03

        # ----------------------------------------------------
        # Close distance bonus (< 1.2m)
        # ----------------------------------------------------
        if distance_m is not None:
            dist_val = self._safe_float(distance_m)
            if 0.0 < dist_val <= 0.8:
                risk += 0.15
            elif 0.8 < dist_val <= 1.2:
                risk += 0.08

        # ----------------------------------------------------
        # Bounding-box size
        # ----------------------------------------------------
        if (
            bbox is not None
            and frame_width
            and frame_height
        ):
            width = max(
                0.0,
                self._safe_float(
                    bbox.get("x2", 0)
                )
                - self._safe_float(
                    bbox.get("x1", 0)
                ),
            )

            height = max(
                0.0,
                self._safe_float(
                    bbox.get("y2", 0)
                )
                - self._safe_float(
                    bbox.get("y1", 0)
                ),
            )

            frame_area = (
                self._safe_float(
                    frame_width
                )
                * self._safe_float(
                    frame_height
                )
            )

            object_area = (
                width * height
            )

            if frame_area > 0:
                area_ratio = (
                    object_area
                    / frame_area
                )

                # Large objects are more likely
                # to represent immediate obstacles.
                size_bonus = self._clamp(
                    area_ratio * 0.30
                )

                risk += size_bonus

        return round(
            self._clamp(risk),
            4,
        )

    # ========================================================
    # OBJECT NORMALIZATION
    # ========================================================

    def normalize_objects(
        self,
        objects: Any,
        frame_width: float,
        frame_height: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Normalize detector output into the common format.
        """

        if not isinstance(
            objects,
            list,
        ):
            return []

        normalized: List[
            Dict[str, Any]
        ] = []

        for idx, item in enumerate(objects):

            if not isinstance(
                item,
                dict,
            ):
                continue

            class_name = str(
                item.get(
                    "class_name",
                    item.get(
                        "label",
                        "unknown",
                    ),
                )
            ).lower().strip()

            raw_conf = item.get("confidence", None)
            if raw_conf is None:
                confidence = 0.85
            else:
                confidence = self._clamp(
                    self._safe_float(
                        raw_conf,
                        0.0,
                    )
                )

            if confidence < self.DEFAULT_CONFIDENCE:
                continue

            bbox = self._bbox(
                item
            )

            depth = self.normalize_depth(
                item.get(
                    "depth",
                    1.0,
                )
            )

            priority = self.get_priority(
                class_name
            )

            if "position" in item and item["position"]:
                position = str(item["position"]).upper().strip()
            else:
                position = self.get_position(
                    bbox,
                    frame_width,
                )

            dist_info = self.estimate_distance_and_steps(
                depth=depth,
                bbox=bbox,
                class_name=class_name,
                confidence=confidence,
                frame_width=frame_width,
                frame_height=frame_height,
            )

            if "distance_m" in item and item["distance_m"] is not None:
                cand_d = self._safe_float(item["distance_m"])
                if math.isfinite(cand_d) and cand_d > 0:
                    dist_m = round(cand_d, 2)
                    dist_cat = self.get_distance_category(dist_m)
                    dist_conf = str(item.get("distance_confidence", dist_info.get("distance_confidence", "HIGH")))
                else:
                    dist_m = float(dist_info.get("estimated_distance_m", 2.5))
                    dist_cat = str(dist_info.get("distance_category", "NEAR"))
                    dist_conf = str(dist_info.get("distance_confidence", "MEDIUM"))
            else:
                dist_m = float(dist_info.get("estimated_distance_m", 2.5))
                dist_cat = str(dist_info.get("distance_category", "NEAR"))
                dist_conf = str(dist_info.get("distance_confidence", "MEDIUM"))

            if "risk_score" in item and item["risk_score"] is not None:
                risk = self._clamp(self._safe_float(item["risk_score"]))
            else:
                risk = self.calculate_risk(
                    confidence=confidence,
                    depth=depth,
                    priority=priority,
                    position=position,
                    bbox=bbox,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    distance_m=dist_m,
                )

            if risk >= 0.70 or (position == "CENTER" and dist_m < 2.5):
                nav_relevance = "HIGH"
            elif risk >= 0.40 or dist_m <= 3.5:
                nav_relevance = "MEDIUM"
            elif risk >= 0.15:
                nav_relevance = "LOW"
            else:
                nav_relevance = "NONE"

            raw_id = item.get("id", None)
            if raw_id is not None:
                try:
                    obj_id = int(raw_id)
                except Exception:
                    obj_id = idx + 1
            else:
                obj_id = idx + 1

            bx1 = self._safe_float(bbox.get("x1", 0.0))
            by1 = self._safe_float(bbox.get("y1", 0.0))
            bx2 = self._safe_float(bbox.get("x2", 0.0))
            by2 = self._safe_float(bbox.get("y2", 0.0))
            obj_cx = round((bx1 + bx2) / 2.0, 2)
            obj_cy = round((by1 + by2) / 2.0, 2)
            obj_w = round(max(0.0, bx2 - bx1), 2)
            obj_h = round(max(0.0, by2 - by1), 2)

            normalized.append(
                {
                    "id":
                        obj_id,

                    "class_name":
                        class_name,

                    "confidence":
                        round(
                            confidence,
                            4,
                        ),

                    "bbox":
                        bbox,

                    "center_x":
                        obj_cx,

                    "center_y":
                        obj_cy,

                    "width":
                        obj_w,

                    "height":
                        obj_h,

                    "position":
                        position,

                    "distance_m":
                        dist_m,

                    "distance_category":
                        dist_cat,

                    "distance_confidence":
                        dist_conf,

                    "priority":
                        round(
                            priority,
                            4,
                        ),

                    "risk_score":
                        risk,

                    "approaching":
                        False,

                    "is_approaching":
                        False,

                    "rapidly_approaching":
                        False,

                    "is_rapidly_approaching":
                        False,

                    "approach_rate":
                        0.0,

                    "approach_rate_mps":
                        0.0,

                    "approach_score":
                        0.0,

                    "motion_state":
                        "UNKNOWN_MOTION",

                    "time_to_collision_s":
                        None,

                    "navigation_relevance":
                        nav_relevance,

                    # Backward compatibility fields
                    "depth":
                        round(
                            depth,
                            4,
                        ),

                    "estimated_distance_m":
                        dist_m,

                    "estimated_steps":
                        dist_info.get("estimated_steps", 0),
                }
            )

        return normalized

    # ========================================================
    # UNKNOWN OBJECT NORMALIZATION
    # ========================================================

    def normalize_unknown_objects(
        self,
        unknown_candidates: Any,
        frame_width: float,
        frame_height: Optional[float] = None,
        known_objects: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Normalize unknown obstacle candidates into the canonical Phase 2/4 contract.

        Each entry satisfies:
            id: int (1001+)
            position: "LEFT" | "CENTER" | "RIGHT"
            bbox: {x1, y1, x2, y2}
            distance_m: float
            distance_category: "VERY_CLOSE" | "NEAR" | "FAR" | "UNKNOWN_DISTANCE"
            distance_confidence: "HIGH" | "MEDIUM" | "LOW"
            risk_score: float
            navigation_relevance: "HIGH" | "MEDIUM" | "LOW" | "NONE"
            confidence: float (candidate physical obstacle confidence)
        """
        if not isinstance(unknown_candidates, list):
            return []

        normalized: List[Dict[str, Any]] = []

        for idx, item in enumerate(unknown_candidates):
            if not isinstance(item, dict):
                continue

            bbox = self._bbox(item)

            # Suppress candidate if it overlaps significantly with an existing known object
            if known_objects:
                suppressed = False
                cand_area = max(1.0, (bbox["x2"] - bbox["x1"]) * (bbox["y2"] - bbox["y1"]))
                for k_obj in known_objects:
                    k_box = self._bbox(k_obj)
                    inter_x1 = max(bbox["x1"], k_box["x1"])
                    inter_y1 = max(bbox["y1"], k_box["y1"])
                    inter_x2 = min(bbox["x2"], k_box["x2"])
                    inter_y2 = min(bbox["y2"], k_box["y2"])
                    inter_w = max(0.0, inter_x2 - inter_x1)
                    inter_h = max(0.0, inter_y2 - inter_y1)
                    inter_area = inter_w * inter_h
                    containment = inter_area / cand_area
                    iou = self.calculate_iou(bbox, k_box)
                    if iou >= 0.20 or containment >= 0.25:
                        suppressed = True
                        break
                if suppressed:
                    continue

            depth = self.normalize_depth(
                item.get("cand_depth", item.get("depth", 0.5))
            )
            confidence = self._clamp(
                self._safe_float(item.get("confidence", 0.50)),
                0.40,
                0.85,
            )

            cand_id = int(item.get("id", 1001 + idx))
            position = self.get_position(bbox, frame_width)

            # Conservative priority for unknown physical obstacle
            priority = 0.60

            dist_info = self.estimate_distance_and_steps(
                depth=depth,
                bbox=bbox,
                class_name="unknown",
                confidence=confidence,
                frame_width=frame_width,
                frame_height=frame_height,
            )

            dist_m = float(dist_info.get("estimated_distance_m", 2.5))
            dist_cat = str(dist_info.get("distance_category", "NEAR"))
            dist_conf = str(dist_info.get("distance_confidence", "MEDIUM"))
            if not math.isfinite(dist_m) or dist_m <= 0:
                dist_cat = "UNKNOWN_DISTANCE"
                dist_conf = "LOW"

            risk = self.calculate_risk(
                confidence=confidence,
                depth=depth,
                priority=priority,
                position=position,
                bbox=bbox,
                frame_width=frame_width,
                frame_height=frame_height,
                distance_m=dist_m,
            )

            if risk >= 0.70 or (position == "CENTER" and dist_m < 2.5):
                nav_relevance = "HIGH"
            elif risk >= 0.40 or dist_m <= 3.5:
                nav_relevance = "MEDIUM"
            elif risk >= 0.15:
                nav_relevance = "LOW"
            else:
                nav_relevance = "NONE"

            bx1 = self._safe_float(bbox.get("x1", 0.0))
            by1 = self._safe_float(bbox.get("y1", 0.0))
            bx2 = self._safe_float(bbox.get("x2", 0.0))
            by2 = self._safe_float(bbox.get("y2", 0.0))
            cand_cx = round((bx1 + bx2) / 2.0, 2)
            cand_cy = round((by1 + by2) / 2.0, 2)
            cand_w = round(max(0.0, bx2 - bx1), 2)
            cand_h = round(max(0.0, by2 - by1), 2)

            normalized.append(
                {
                    "id": cand_id,
                    "class_name": str(item.get("class_name") or "unknown"),
                    "position": position,
                    "bbox": bbox,
                    "center_x": cand_cx,
                    "center_y": cand_cy,
                    "width": cand_w,
                    "height": cand_h,
                    "distance_m": round(dist_m, 2),
                    "distance_category": dist_cat,
                    "distance_confidence": dist_conf,
                    "risk_score": risk,
                    "navigation_relevance": nav_relevance,
                    "confidence": round(confidence, 4),
                    # Internal helper fields for DecisionEngine integration
                    "depth": round(depth, 4),
                    "priority": priority,
                    "estimated_distance_m": round(dist_m, 2),
                    "estimated_steps": int(dist_info.get("estimated_steps", 0)),
                }
            )

        return normalized

    # ========================================================
    # DISTANCE & STEP ESTIMATION
    # ========================================================

    @classmethod
    def estimate_distance_and_steps(
        cls,
        depth: float,
        bbox: Any,
        class_name: str,
        confidence: float,
        frame_width: Optional[float] = 640.0,
        frame_height: Optional[float] = 480.0,
        step_length_m: float = 0.75,
    ) -> Dict[str, Any]:
        """
        Multi-cue approximate distance estimator for SmartVisionAI.
        
        Combines:
        1. MiDaS normalized relative depth (0.0=close, 1.0=far)
        2. Bounding box relative height & area geometry
        3. Object class typical physical scale
        4. Vertical ground plane proximity
        """
        # Validate depth input
        is_valid_depth = True
        try:
            depth_val = float(depth)
            if math.isnan(depth_val) or math.isinf(depth_val) or depth_val < 0.0:
                is_valid_depth = False
        except (TypeError, ValueError):
            is_valid_depth = False

        if not is_valid_depth:
            return {
                "estimated_distance_m": 2.5,
                "distance_m": 2.5,
                "distance_category": "UNKNOWN_DISTANCE",
                "distance_confidence": "LOW",
                "estimated_steps": 3,
            }

        depth_val = cls._clamp(depth_val)
        # Base relative distance curve (approx 0.5m to 6.0m)
        dist_base = 0.5 + depth_val * 5.5

        # Bounding box geometry
        if isinstance(bbox, dict):
            x1 = cls._safe_float(bbox.get("x1", 0.0))
            y1 = cls._safe_float(bbox.get("y1", 0.0))
            x2 = cls._safe_float(bbox.get("x2", 0.0))
            y2 = cls._safe_float(bbox.get("y2", 0.0))
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x1 = cls._safe_float(bbox[0])
            y1 = cls._safe_float(bbox[1])
            x2 = cls._safe_float(bbox[2])
            y2 = cls._safe_float(bbox[3])
        else:
            x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0

        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        frame_w = max(1.0, float(frame_width or 640.0))
        frame_h = max(1.0, float(frame_height or 480.0))

        box_h_ratio = cls._clamp(box_h / frame_h)
        box_area_ratio = cls._clamp((box_w * box_h) / (frame_w * frame_h))
        bottom_ratio = cls._clamp(y2 / frame_h)

        # Scale by class
        cname = str(class_name or "").lower().strip()
        if cname in cls.HUMAN_CLASSES:
            geometry_dist = max(0.5, 3.5 * (1.0 - box_h_ratio * 0.85))
            dist = 0.65 * dist_base + 0.35 * geometry_dist
        elif cname in cls.VEHICLE_CLASSES:
            geometry_dist = max(0.8, 5.0 * (1.0 - box_area_ratio * 0.90))
            dist = 0.60 * dist_base + 0.40 * geometry_dist
        elif cname in {"dining table", "table", "chair", "couch", "bed", "bench"}:
            geometry_dist = max(0.5, 4.0 * (1.0 - box_h_ratio * 0.75))
            dist = 0.70 * dist_base + 0.30 * geometry_dist
        else:
            dist = dist_base

        if bottom_ratio > 0.85 and box_h_ratio > 0.25:
            dist = min(dist, dist * 0.85)

        dist = max(0.5, min(7.0, dist))
        estimated_distance_m = round(dist, 2)

        conf = cls._safe_float(confidence, 0.5)
        if conf >= 0.65 and box_area_ratio >= 0.04 and 0.15 <= depth_val <= 0.85:
            confidence_level = "HIGH"
        elif conf >= 0.45:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

        if estimated_distance_m < 1.4:
            distance_category = "VERY_CLOSE"
        elif estimated_distance_m <= 3.5:
            distance_category = "NEAR"
        else:
            distance_category = "FAR"

        raw_steps = round(estimated_distance_m / max(0.1, step_length_m))
        estimated_steps = max(1, min(8, raw_steps))

        return {
            "estimated_distance_m": estimated_distance_m,
            "distance_m": estimated_distance_m,
            "distance_category": distance_category,
            "distance_confidence": confidence_level,
            "estimated_steps": estimated_steps,
        }

    @classmethod
    def get_distance_category(cls, dist_m: float) -> str:
        """
        Map a metric distance to canonical category: VERY_CLOSE, NEAR, FAR, UNKNOWN_DISTANCE.
        """
        try:
            d = float(dist_m)
            if not math.isfinite(d) or d <= 0:
                return "UNKNOWN_DISTANCE"
            if d < 1.4:
                return "VERY_CLOSE"
            elif d <= 3.5:
                return "NEAR"
            else:
                return "FAR"
        except Exception:
            return "UNKNOWN_DISTANCE"

    # ========================================================
    # PERSON STATE CLASSIFICATION (PHASE 10)
    # ========================================================

    def classify_person_state(
        self,
        det_item: Dict[str, Any],
        motion_state: str,
        approach_rate: float,
        is_crossing: bool,
        crossing_direction: str,
        dx_norm: float,
        total_dx_norm: float,
        norm_cx: float,
        stability: int,
        all_persons: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Classify human dynamic state conservatively based on frame-to-frame temporal evidence.
        States:
          - PERSON_STATIC: standing or motionless
          - PERSON_APPROACHING: decreasing distance toward camera
          - PERSON_MOVING_AWAY: increasing distance away from camera
          - PERSON_CROSSING: lateral translation across camera field intersecting corridor
          - PERSON_SIDE_MOVING: moving laterally outside center corridor
          - PERSON_GROUP: two or more people clustered together
          - PERSON_UNCERTAIN: low confidence, erratic movement, or contradictory evidence
        """
        explicit = det_item.get("person_state")
        if explicit:
            return str(explicit).upper()

        cname = str(det_item.get("class_name", "")).lower().strip()
        if cname not in ("person", "pedestrian", "human", "man", "woman", "child") and "person" not in cname:
            return "NONE"

        det_conf = self._safe_float(det_item.get("confidence", 0.70))
        dist_conf = str(det_item.get("distance_confidence", "HIGH")).upper()
        dist_cat = str(det_item.get("distance_category", "")).upper()

        # 1. PERSON_UNCERTAIN: low detection confidence or erratic/contradictory motion
        if (
            det_conf < 0.55
            or dist_conf == "LOW"
            or dist_cat in ("UNKNOWN_DISTANCE", "UNKNOWN")
            or (stability < 2 and motion_state == "UNKNOWN_MOTION" and abs(approach_rate) > 0.40)
        ):
            return "PERSON_UNCERTAIN"

        # 2. PERSON_GROUP: proximity to other person boxes (depth <= 1.2m and close horizontally)
        if all_persons and len(all_persons) >= 2:
            this_dist = self._safe_float(det_item.get("distance_m", det_item.get("estimated_distance_m", 99.0)))
            this_cx = self._safe_float(det_item.get("center_x", det_item.get("cx", 0.0)))
            for other in all_persons:
                if other is det_item:
                    continue
                other_dist = self._safe_float(other.get("distance_m", other.get("estimated_distance_m", 99.0)))
                other_cx = self._safe_float(other.get("center_x", other.get("cx", 0.0)))
                if abs(other_dist - this_dist) <= 1.2 and (abs(other_cx - this_cx) <= 150.0 or abs(other_dist - this_dist) <= 0.50):
                    return "PERSON_GROUP"

        # 3. PERSON_CROSSING: moving across corridor boundaries into or through path
        if is_crossing or crossing_direction in ("LEFT_TO_CENTER", "RIGHT_TO_CENTER", "CENTER_TO_LEFT", "CENTER_TO_RIGHT", "CROSSING_LEFT", "CROSSING_RIGHT"):
            return "PERSON_CROSSING"

        # 4. PERSON_SIDE_MOVING: moving laterally outside center corridor
        if (norm_cx < 0.35 or norm_cx > 0.65) and (abs(dx_norm) >= 0.03 or abs(total_dx_norm) >= 0.06) and motion_state not in ("APPROACHING", "RAPIDLY_APPROACHING"):
            return "PERSON_SIDE_MOVING"

        # 5. PERSON_APPROACHING: moving towards camera
        if motion_state in ("APPROACHING", "RAPIDLY_APPROACHING") or approach_rate >= 0.15:
            return "PERSON_APPROACHING"

        # 6. PERSON_MOVING_AWAY: moving away from camera
        if motion_state == "RECEDING" or approach_rate <= -0.15:
            return "PERSON_MOVING_AWAY"

        # 7. PERSON_STATIC: stationary position and distance
        if motion_state == "STATIONARY" or (abs(approach_rate) < 0.15 and abs(dx_norm) < 0.03):
            return "PERSON_STATIC"

        return "PERSON_UNCERTAIN"

    # ========================================================
    # MULTI-OBJECT TRACKING & TEMPORAL DISTANCE SMOOTHING
    # ========================================================

    def track_and_smooth_distances(
        self,
        objects: List[Dict[str, Any]],
        unknown_objects: List[Dict[str, Any]],
        frame_width: float = 640.0,
        frame_height: float = 480.0,
    ) -> None:
        """
        Multi-object temporal tracking and adaptive distance smoothing.

        Performs:
        1. Multi-object track association per detection via IoU + centroid proximity.
        2. Adaptive EMA distance smoothing per track to stabilize noise without lag.
        3. SAFETY-CRITICAL BYPASS: alpha = 1.0 (immediate update with 0 lag) whenever
           obstacle is dangerously close (<= 1.2m) or makes a sudden rapid approach.
        4. Independent per-object tracks so distance histories never cross-pollinate.
        5. Track retention across intermittent frame drops (up to 3 frames) without crashing.
        """
        self.frame_index += 1
        frame_w = max(1.0, float(frame_width or 640.0))
        frame_h = max(1.0, float(frame_height or 480.0))

        # Flatten list of item references with origin tag
        all_items: List[tuple[Dict[str, Any], bool]] = []
        for obj in objects:
            all_items.append((obj, False))
        for unk in unknown_objects:
            all_items.append((unk, True))

        all_persons = [
            item for item, _ in all_items
            if str(item.get("class_name", "")).lower().strip() in self.HUMAN_CLASSES
            or "person" in str(item.get("class_name", "")).lower()
        ]

        # Extract features for association
        detections_info = []
        for idx, (item, is_unknown) in enumerate(all_items):
            box = item.get("bbox", {})
            if isinstance(box, dict):
                bx1 = self._safe_float(box.get("x1", 0.0))
                by1 = self._safe_float(box.get("y1", 0.0))
                bx2 = self._safe_float(box.get("x2", 0.0))
                by2 = self._safe_float(box.get("y2", 0.0))
            elif isinstance(box, (list, tuple)) and len(box) >= 4:
                bx1 = self._safe_float(box[0])
                by1 = self._safe_float(box[1])
                bx2 = self._safe_float(box[2])
                by2 = self._safe_float(box[3])
            else:
                bx1, by1, bx2, by2 = 0.0, 0.0, 0.0, 0.0

            cx = (bx1 + bx2) / 2.0
            cy = (by1 + by2) / 2.0
            cname = str(item.get("class_name", "unknown" if is_unknown else "")).lower().strip()
            raw_dist = self._safe_float(
                item.get("distance_m", item.get("estimated_distance_m", 2.5)),
                2.5,
            )

            detections_info.append({
                "index": idx,
                "item": item,
                "is_unknown": is_unknown,
                "class_name": cname,
                "box": {"x1": bx1, "y1": by1, "x2": bx2, "y2": by2},
                "cx": cx,
                "cy": cy,
                "raw_dist": raw_dist,
                "item_id": item.get("id"),
            })

        # Pairwise association scoring
        match_candidates = []
        for det in detections_info:
            for track_id, trk in self.object_tracks.items():
                if det["class_name"] != trk["class_name"]:
                    continue

                det_item_id = det.get("item_id")
                trk_det_id = trk.get("detection_id")
                if det_item_id is not None and trk_det_id is not None:
                    if det_item_id == trk_det_id:
                        match_candidates.append((2.0, det["index"], track_id))
                        continue
                    else:
                        continue

                b1, b2 = det["box"], trk["box"]
                ix1 = max(b1["x1"], b2["x1"])
                iy1 = max(b1["y1"], b2["y1"])
                ix2 = min(b1["x2"], b2["x2"])
                iy2 = min(b1["y2"], b2["y2"])
                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                inter = iw * ih
                a1 = max(1.0, (b1["x2"] - b1["x1"]) * (b1["y2"] - b1["y1"]))
                a2 = max(1.0, (b2["x2"] - b2["x1"]) * (b2["y2"] - b2["y1"]))
                union = a1 + a2 - inter
                iou = inter / union if union > 0 else 0.0

                norm_cdist = math.hypot((det["cx"] - trk["cx"]) / frame_w, (det["cy"] - trk["cy"]) / frame_h)
                px_cdist = math.hypot(det["cx"] - trk["cx"], det["cy"] - trk["cy"])

                if iou >= 0.18 or norm_cdist <= 0.22 or px_cdist <= 140.0:
                    score = iou * 0.70 + (1.0 - min(1.0, norm_cdist)) * 0.30
                    match_candidates.append((score, det["index"], track_id))

        # Greedy match assignment
        match_candidates.sort(key=lambda x: x[0], reverse=True)
        assigned_dets = set()
        assigned_tracks = set()
        matched_pairs = []

        for score, d_idx, trk_id in match_candidates:
            if d_idx in assigned_dets or trk_id in assigned_tracks:
                continue
            assigned_dets.add(d_idx)
            assigned_tracks.add(trk_id)
            matched_pairs.append((d_idx, trk_id))

        # Process matched tracks
        for d_idx, trk_id in matched_pairs:
            det = detections_info[d_idx]
            item = det["item"]
            trk = self.object_tracks[trk_id]
            raw_dist = det["raw_dist"]
            prev_smoothed = trk["smoothed_distance_m"]
            prev_raw = trk["raw_distance_m"]

            # Adaptive smoothing factor
            # Safety bypass: immediate update for close dangerous obstacles or rapid approaches
            if raw_dist <= 1.2:
                alpha = 1.0
            elif raw_dist <= 1.4 and prev_smoothed > 1.8:
                alpha = 1.0
            elif raw_dist < prev_smoothed - 0.50:
                alpha = 0.90
            elif raw_dist < prev_raw - 0.12:
                alpha = 0.65
            else:
                alpha = 0.35

            smoothed_dist = round(alpha * raw_dist + (1.0 - alpha) * prev_smoothed, 2)
            smoothed_dist = max(0.5, min(7.0, smoothed_dist))

            stability = min(15, trk["stability"] + 1)

            trk["raw_history"].append(raw_dist)
            if len(trk["raw_history"]) > 6:
                trk["raw_history"].pop(0)

            trk["smoothed_history"].append(smoothed_dist)
            if len(trk["smoothed_history"]) > 6:
                trk["smoothed_history"].pop(0)

            # Timestamp & delta_time calculation
            curr_ts = time.time()
            prev_ts = trk.get("current_timestamp", None)
            trk["previous_timestamp"] = prev_ts
            trk["current_timestamp"] = curr_ts
            if prev_ts is not None:
                dt = curr_ts - prev_ts
            else:
                dt = 0.70
            if dt <= 0.05 or dt > 2.0 or not math.isfinite(dt):
                dt = 0.70

            # Distance history tracking
            trk["previous_distance_m"] = prev_smoothed
            trk["current_distance_m"] = smoothed_dist
            trk["distance_history"] = list(trk["smoothed_history"])

            # Centroid tracking & horizontal displacement (Phase 10)
            prev_cx = trk.get("cx", det["cx"])
            trk.setdefault("cx_history", [prev_cx]).append(det["cx"])
            if len(trk["cx_history"]) > 6:
                trk["cx_history"].pop(0)

            prev_cy = trk.get("cy", det["cy"])
            trk.setdefault("cy_history", [prev_cy]).append(det["cy"])
            if len(trk["cy_history"]) > 6:
                trk["cy_history"].pop(0)

            norm_prev_x = prev_cx / frame_w
            norm_curr_x = det["cx"] / frame_w
            dx_norm = (det["cx"] - prev_cx) / frame_w
            total_dx_norm = (trk["cx_history"][-1] - trk["cx_history"][0]) / frame_w if len(trk["cx_history"]) >= 2 else 0.0

            is_crossing = False
            crossing_direction = "NONE"
            if len(trk["cx_history"]) >= 2:
                if norm_prev_x < 0.35 and norm_curr_x >= 0.35:
                    is_crossing = True
                    crossing_direction = "LEFT_TO_CENTER"
                elif norm_prev_x > 0.65 and norm_curr_x <= 0.65:
                    is_crossing = True
                    crossing_direction = "RIGHT_TO_CENTER"
                elif norm_prev_x >= 0.35 and norm_curr_x < 0.35:
                    is_crossing = True
                    crossing_direction = "CENTER_TO_LEFT"
                elif norm_prev_x <= 0.65 and norm_curr_x > 0.65:
                    is_crossing = True
                    crossing_direction = "CENTER_TO_RIGHT"
                elif abs(total_dx_norm) >= 0.10 and (0.25 <= norm_curr_x <= 0.75 or 0.25 <= norm_prev_x <= 0.75):
                    is_crossing = True
                    crossing_direction = "CROSSING_LEFT" if total_dx_norm < 0 else "CROSSING_RIGHT"

            # Compute approach rate: delta distance / delta time
            rate = (prev_smoothed - smoothed_dist) / dt
            sm_hist = trk["smoothed_history"]
            raw_h = trk["raw_history"]

            if len(sm_hist) >= 3:
                sm_total = sm_hist[-3] - sm_hist[-1]
                sm_window_rate = sm_total / (2.0 * dt)
                if sm_total > 0.15:
                    rate = max(rate, sm_window_rate)
            elif len(raw_h) >= 3:
                raw_total = raw_h[-3] - raw_h[-1]
                raw_window_rate = raw_total / (2.0 * dt)
                if raw_total > 0.20:
                    rate = max(rate, raw_window_rate)

            if not math.isfinite(rate):
                rate = 0.0
            approach_rate_mps = round(rate, 2)

            # Trend consistency & noise filtering
            if len(sm_hist) >= 2:
                pairwise_diffs = [sm_hist[i - 1] - sm_hist[i] for i in range(1, len(sm_hist))]
                descent_count = sum(1 for d in pairwise_diffs if d > 0.02)
                trend_consistency = descent_count / float(len(pairwise_diffs))
                total_descent = sm_hist[0] - sm_hist[-1]
            else:
                trend_consistency = 0.0
                total_descent = 0.0

            consecutive_app = trk.get("consecutive_approaching_frames", 0)
            consecutive_stat = trk.get("consecutive_stationary_frames", 0)
            consecutive_rec = trk.get("consecutive_receding_frames", 0)

            approaching = False
            rapidly_approaching = False

            # Motion state classification with consecutive-frame confirmation
            # 1. Safety bypass: sudden close drop into danger zone
            if raw_dist <= 1.4 and (prev_smoothed - raw_dist) >= 0.50:
                approaching = True
                rapidly_approaching = True
                motion_state = "RAPIDLY_APPROACHING"
                consecutive_app += 1
                consecutive_stat = 0
                consecutive_rec = 0
                approach_rate_mps = max(0.50, round((prev_smoothed - raw_dist) / dt, 2))
            elif raw_dist <= 1.2 and (prev_smoothed - raw_dist) >= 0.25:
                approaching = True
                consecutive_app += 1
                consecutive_stat = 0
                consecutive_rec = 0
                if approach_rate_mps >= 0.50:
                    rapidly_approaching = True
                    motion_state = "RAPIDLY_APPROACHING"
                else:
                    motion_state = "APPROACHING"
            # 2. Rapid approach: rate >= 0.50 m/s
            elif approach_rate_mps >= 0.50:
                consecutive_app += 1
                consecutive_stat = 0
                consecutive_rec = 0
                if consecutive_app >= 2 or total_descent >= 0.45 or (prev_smoothed - smoothed_dist) >= 0.35:
                    rapidly_approaching = True
                    approaching = True
                    motion_state = "RAPIDLY_APPROACHING"
                else:
                    approaching = False
                    rapidly_approaching = False
                    motion_state = "UNKNOWN_MOTION"
            # 3. Moderate approach: 0.15 <= rate < 0.50 m/s
            elif approach_rate_mps >= 0.15:
                consecutive_stat = 0
                consecutive_rec = 0
                if (prev_smoothed - smoothed_dist) >= 0.04 or (prev_raw - raw_dist) >= 0.04:
                    consecutive_app += 1
                else:
                    consecutive_app = 1

                if consecutive_app >= 2 and (trend_consistency >= 0.60 or total_descent >= 0.25):
                    approaching = True
                    rapidly_approaching = False
                    motion_state = "APPROACHING"
                else:
                    approaching = False
                    rapidly_approaching = False
                    motion_state = "UNKNOWN_MOTION"
            # 4. Receding: rate <= -0.15 m/s
            elif approach_rate_mps <= -0.15:
                consecutive_rec += 1
                consecutive_app = 0
                consecutive_stat = 0
                approaching = False
                rapidly_approaching = False
                motion_state = "RECEDING"
            # 5. Stationary: -0.15 < rate < 0.15 m/s
            else:
                consecutive_stat += 1
                consecutive_app = 0
                consecutive_rec = 0
                approaching = False
                rapidly_approaching = False
                motion_state = "STATIONARY"

            # Bounded approach score: [0.0, 1.0]
            if approaching or rapidly_approaching:
                rate_norm = min(1.0, max(0.0, approach_rate_mps / 1.0))
                consec_norm = min(1.0, consecutive_app / 3.0)
                trend_norm = max(0.0, min(1.0, trend_consistency))
                stab_norm = min(1.0, stability / 4.0)
                det_conf = float(det.get("confidence", item.get("confidence", 0.70)))
                conf_norm = max(0.0, min(1.0, det_conf))
                raw_score = (
                    0.35 * rate_norm +
                    0.25 * consec_norm +
                    0.20 * trend_norm +
                    0.10 * stab_norm +
                    0.10 * conf_norm
                )
                if rapidly_approaching:
                    raw_score = max(0.65, raw_score * 1.25)
                approach_score = round(max(0.0, min(1.0, raw_score)), 2)
            else:
                approach_score = 0.0

            # Time to collision (TTC) in seconds
            if smoothed_dist > 0.0 and approach_rate_mps > 0.05 and math.isfinite(smoothed_dist) and math.isfinite(approach_rate_mps):
                ttc = smoothed_dist / approach_rate_mps
                time_to_collision_s = round(min(60.0, max(0.1, ttc)), 1)
            else:
                time_to_collision_s = None

            # Person state classification (Phase 10)
            person_state = self.classify_person_state(
                det_item=item,
                motion_state=motion_state,
                approach_rate=approach_rate_mps,
                is_crossing=is_crossing,
                crossing_direction=crossing_direction,
                dx_norm=dx_norm,
                total_dx_norm=total_dx_norm,
                norm_cx=norm_curr_x,
                stability=stability,
                all_persons=all_persons,
            )

            trk["box"] = det["box"]
            trk["cx"] = det["cx"]
            trk["cy"] = det["cy"]
            trk["raw_distance_m"] = raw_dist
            trk["smoothed_distance_m"] = smoothed_dist
            trk["previous_distance_m"] = prev_smoothed
            trk["current_distance_m"] = smoothed_dist
            trk["distance_history"] = list(trk["smoothed_history"])
            trk["stability"] = stability
            trk["missed_frames"] = 0
            trk["last_seen_frame"] = self.frame_index
            trk["approaching"] = approaching
            trk["rapidly_approaching"] = rapidly_approaching
            trk["approach_rate"] = approach_rate_mps
            trk["approach_rate_mps"] = approach_rate_mps
            trk["approach_score"] = approach_score
            trk["motion_state"] = motion_state
            trk["person_state"] = person_state
            trk["is_crossing"] = is_crossing
            trk["crossing_direction"] = crossing_direction
            trk["dx_norm"] = round(dx_norm, 4)
            trk["total_dx_norm"] = round(total_dx_norm, 4)
            trk["consecutive_approaching_frames"] = consecutive_app
            trk["consecutive_stationary_frames"] = consecutive_stat
            trk["consecutive_receding_frames"] = consecutive_rec
            trk["time_to_collision_s"] = time_to_collision_s

            # Canonical categorization and confidence from smoothed distance
            if item.get("distance_category") == "UNKNOWN_DISTANCE":
                cat = "UNKNOWN_DISTANCE"
                conf = "LOW"
            elif smoothed_dist < 1.4:
                cat = "VERY_CLOSE"
                conf = "HIGH" if stability >= 2 and item.get("distance_confidence") != "LOW" else item.get("distance_confidence", "MEDIUM")
            elif smoothed_dist <= 3.5:
                cat = "NEAR"
                conf = "HIGH" if stability >= 2 and item.get("distance_confidence") != "LOW" else item.get("distance_confidence", "MEDIUM")
            else:
                cat = "FAR"
                conf = item.get("distance_confidence", "LOW")

            # Update item in-place
            item["distance_m"] = smoothed_dist
            item["estimated_distance_m"] = smoothed_dist
            item["distance_category"] = cat
            item["distance_confidence"] = conf
            item["track_id"] = trk_id
            item["approaching"] = approaching
            item["is_approaching"] = approaching
            item["rapidly_approaching"] = rapidly_approaching
            item["is_rapidly_approaching"] = rapidly_approaching
            item["approach_rate"] = approach_rate_mps
            item["approach_rate_mps"] = approach_rate_mps
            item["approach_score"] = approach_score
            item["motion_state"] = motion_state
            item["person_state"] = person_state
            item["is_crossing"] = is_crossing
            item["crossing_direction"] = crossing_direction
            item["dx_norm"] = round(dx_norm, 4)
            item["total_dx_norm"] = round(total_dx_norm, 4)
            item["time_to_collision_s"] = time_to_collision_s
            item["obstacle_stability"] = stability
            item["consecutive_approaching_frames"] = consecutive_app
            item["consecutive_stationary_frames"] = consecutive_stat
            item["consecutive_receding_frames"] = consecutive_rec
            item["estimated_steps"] = max(1, min(8, round(smoothed_dist / 0.75)))

        # Process unmatched detections (initialize new tracks)
        for det in detections_info:
            if det["index"] in assigned_dets:
                continue
            item = det["item"]
            raw_dist = det["raw_dist"]
            trk_id = f"trk_{self._next_track_num}"
            self._next_track_num += 1
            curr_ts = time.time()

            norm_cx = det["cx"] / frame_w
            init_ps = item.get("person_state")
            if not init_ps:
                if det["class_name"] in ("person", "pedestrian", "human") or "person" in det["class_name"]:
                    init_ps = self.classify_person_state(
                        det_item=item,
                        motion_state="UNKNOWN_MOTION",
                        approach_rate=0.0,
                        is_crossing=False,
                        crossing_direction="NONE",
                        dx_norm=0.0,
                        total_dx_norm=0.0,
                        norm_cx=norm_cx,
                        stability=1,
                        all_persons=all_persons,
                    )
                else:
                    init_ps = "NONE"

            self.object_tracks[trk_id] = {
                "track_id": trk_id,
                "detection_id": det.get("item_id"),
                "class_name": det["class_name"],
                "box": det["box"],
                "cx": det["cx"],
                "cy": det["cy"],
                "cx_history": [det["cx"]],
                "cy_history": [det["cy"]],
                "raw_distance_m": raw_dist,
                "smoothed_distance_m": raw_dist,
                "previous_distance_m": raw_dist,
                "current_distance_m": raw_dist,
                "raw_history": [raw_dist],
                "smoothed_history": [raw_dist],
                "distance_history": [raw_dist],
                "previous_timestamp": None,
                "current_timestamp": curr_ts,
                "stability": 1,
                "missed_frames": 0,
                "last_seen_frame": self.frame_index,
                "approaching": False,
                "rapidly_approaching": False,
                "approach_rate": 0.0,
                "approach_rate_mps": 0.0,
                "approach_score": 0.0,
                "motion_state": "UNKNOWN_MOTION",
                "person_state": init_ps,
                "is_crossing": False,
                "crossing_direction": "NONE",
                "dx_norm": 0.0,
                "total_dx_norm": 0.0,
                "consecutive_approaching_frames": 0,
                "consecutive_stationary_frames": 0,
                "consecutive_receding_frames": 0,
                "time_to_collision_s": None,
            }

            item["track_id"] = trk_id
            item["obstacle_stability"] = 1
            item["approaching"] = False
            item["is_approaching"] = False
            item["rapidly_approaching"] = False
            item["is_rapidly_approaching"] = False
            item["approach_rate"] = 0.0
            item["approach_rate_mps"] = 0.0
            item["approach_score"] = 0.0
            item["motion_state"] = "UNKNOWN_MOTION"
            item["person_state"] = init_ps
            item["is_crossing"] = False
            item["crossing_direction"] = "NONE"
            item["dx_norm"] = 0.0
            item["total_dx_norm"] = 0.0
            item["time_to_collision_s"] = None
            item["consecutive_approaching_frames"] = 0
            item["consecutive_stationary_frames"] = 0
            item["consecutive_receding_frames"] = 0
            item["estimated_distance_m"] = raw_dist

        # Age out unmatched tracks (retain across up to 3 intermittent misses)
        stale_track_ids = []
        for trk_id, trk in self.object_tracks.items():
            if trk_id not in assigned_tracks:
                trk["missed_frames"] += 1
                if trk["missed_frames"] > 3:
                    stale_track_ids.append(trk_id)

        for trk_id in stale_track_ids:
            del self.object_tracks[trk_id]

    # ========================================================
    # PREDICTIVE THREAT SELECTION (PHASE 11)
    # ========================================================

    def compute_predictive_threat(
        self,
        objects: List[Dict[str, Any]],
        unknown_objects: Optional[List[Dict[str, Any]]] = None,
        primary_obstacle: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Identify the highest predictive threat across known and unknown obstacles.
        Combines corridor priority (CENTER > sides), approach rate, approach score,
        proximity distance, and risk score.
        """
        all_candidates: List[Dict[str, Any]] = []
        if isinstance(objects, list):
            all_candidates.extend(objects)
        if isinstance(unknown_objects, list):
            all_candidates.extend(unknown_objects)

        approaching_candidates: List[Dict[str, Any]] = []
        for cand in all_candidates:
            if not isinstance(cand, dict):
                continue
            is_app = bool(cand.get("approaching", False) or cand.get("is_approaching", False) or cand.get("rapidly_approaching", False))
            motion = str(cand.get("motion_state", "")).upper()
            if is_app or motion in ("APPROACHING", "RAPIDLY_APPROACHING"):
                approaching_candidates.append(cand)

        if not approaching_candidates:
            p_id = primary_obstacle.get("track_id", primary_obstacle.get("id")) if primary_obstacle else None
            p_cls = str(primary_obstacle.get("class_name", "none")) if primary_obstacle else "none"
            p_pos = str(primary_obstacle.get("position", "NONE")).upper() if primary_obstacle else "NONE"
            p_dist = round(self._safe_float(primary_obstacle.get("distance_m", 0.0)), 2) if primary_obstacle else 0.0
            p_mot = str(primary_obstacle.get("motion_state", "STATIONARY")).upper() if primary_obstacle else "STATIONARY"
            return {
                "obstacle_id": p_id,
                "track_id": str(primary_obstacle.get("track_id", "")) if primary_obstacle else "",
                "class_name": p_cls,
                "position": p_pos,
                "distance_m": p_dist,
                "approach_rate_mps": 0.0,
                "approach_score": 0.0,
                "time_to_collision_s": None,
                "motion_state": p_mot,
                "threat_level": "NONE",
                "is_approaching": False,
                "warning_issued": False,
            }

        def threat_scorer(c: Dict[str, Any]) -> float:
            pos = str(c.get("position", "CENTER")).upper()
            corridor_weight = 1.35 if pos == "CENTER" else 0.80
            rate = self._safe_float(c.get("approach_rate", c.get("approach_rate_mps", 0.0)))
            dist = self._safe_float(c.get("distance_m", c.get("estimated_distance_m", 3.0)), 3.0)
            score = self._safe_float(c.get("approach_score", 0.0))
            risk = self._safe_float(c.get("risk_score", 0.0))
            is_rapid = bool(c.get("rapidly_approaching", False) or str(c.get("motion_state", "")).upper() == "RAPIDLY_APPROACHING")

            rapid_bonus = 0.35 if is_rapid else 0.0
            dist_factor = max(0.0, 1.0 - min(1.0, dist / 6.0))
            rate_factor = min(1.0, max(0.0, rate / 1.5))

            return corridor_weight * (0.35 * score + 0.25 * rate_factor + 0.25 * dist_factor + 0.15 * risk + rapid_bonus)

        approaching_candidates.sort(key=threat_scorer, reverse=True)
        best = approaching_candidates[0]

        best_dist = round(self._safe_float(best.get("distance_m", best.get("estimated_distance_m", 0.0))), 2)
        raw_dist = round(self._safe_float(best.get("raw_distance_m", best_dist)), 2)
        eval_dist = min(best_dist, raw_dist) if raw_dist > 0 else best_dist
        best_rate = round(self._safe_float(best.get("approach_rate", best.get("approach_rate_mps", 0.0))), 2)
        best_score = round(self._safe_float(best.get("approach_score", 0.0)), 2)
        best_motion = str(best.get("motion_state", "APPROACHING")).upper()
        if best_motion not in ("STATIONARY", "APPROACHING", "RAPIDLY_APPROACHING", "RECEDING", "UNKNOWN_MOTION"):
            best_motion = "APPROACHING"

        best_pos = str(best.get("position", "CENTER")).upper()
        best_ttc = best.get("time_to_collision_s", None)
        ttc_val = float(best_ttc) if best_ttc is not None and math.isfinite(float(best_ttc)) else None

        # Determine threat level
        if best_pos == "CENTER" and (
            (best_motion == "RAPIDLY_APPROACHING" and (eval_dist <= 1.4 or (ttc_val is not None and ttc_val <= 2.0)))
            or (ttc_val is not None and ttc_val <= 2.0 and eval_dist <= 2.0)
        ):
            threat_lvl = "EMERGENCY"
        elif best_pos == "CENTER" and (
            (best_motion == "RAPIDLY_APPROACHING" and eval_dist <= 2.5)
            or (best_motion in ("APPROACHING", "RAPIDLY_APPROACHING") and (ttc_val is not None and ttc_val <= 3.0))
        ):
            threat_lvl = "DANGER"
        elif (best_pos == "CENTER" and eval_dist <= 2.2) or (ttc_val is not None and ttc_val <= 6.0) or (best_motion == "RAPIDLY_APPROACHING"):
            threat_lvl = "CAUTION"
        elif best_motion in ("APPROACHING", "RAPIDLY_APPROACHING"):
            threat_lvl = "AWARE"
        else:
            threat_lvl = "NONE"

        warning_issued = bool(threat_lvl in ("CAUTION", "DANGER", "EMERGENCY"))

        return {
            "obstacle_id": best.get("track_id", best.get("id")),
            "track_id": str(best.get("track_id", "trk_1")),
            "class_name": str(best.get("class_name", "obstacle")),
            "position": best_pos,
            "distance_m": best_dist,
            "approach_rate_mps": best_rate,
            "approach_score": best_score,
            "time_to_collision_s": best_ttc,
            "motion_state": best_motion,
            "threat_level": threat_lvl,
            "is_approaching": True,
            "warning_issued": warning_issued,
        }

    # ========================================================
    # TEMPORAL TRACKING & APPROACH DETECTION
    # ========================================================

    def update_tracking_and_approach(
        self,
        primary_obstacle: Optional[Dict[str, Any]],
        predictive_threat: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract primary obstacle tracking and approach status.
        Maintains backward compatibility for tracking_info and self.track_history.
        """
        if not primary_obstacle and not predictive_threat:
            self.track_history = []
            return {
                "approaching": False,
                "approach_rate": 0.0,
                "approach_score": 0.0,
                "rapidly_approaching": False,
                "motion_state": "UNKNOWN_MOTION",
                "time_to_collision_s": None,
                "obstacle_stability": 0,
            }

        ref = predictive_threat if predictive_threat else primary_obstacle
        approaching = bool(ref.get("approaching", False) or ref.get("motion_state") in ("APPROACHING", "RAPIDLY_APPROACHING"))
        rapidly_approaching = bool(ref.get("rapidly_approaching", False) or ref.get("motion_state") == "RAPIDLY_APPROACHING")
        approach_rate = float(ref.get("approach_rate_mps", ref.get("approach_rate", 0.0)))
        approach_score = float(ref.get("approach_score", 0.0))
        motion_state = str(ref.get("motion_state", "UNKNOWN_MOTION"))
        ttc = ref.get("time_to_collision_s", None)
        stability = int(primary_obstacle.get("obstacle_stability", 1) if primary_obstacle else 1)

        # Backward compatibility track history
        if primary_obstacle:
            record = {
                "class_name": str(primary_obstacle.get("class_name", "")),
                "estimated_distance_m": float(primary_obstacle.get("distance_m", primary_obstacle.get("estimated_distance_m", 2.0))),
                "stability": stability,
                "approaching": approaching,
                "approach_rate": approach_rate,
            }
            self.track_history.append(record)
            if len(self.track_history) > 5:
                self.track_history.pop(0)

        return {
            "approaching": approaching,
            "approach_rate": approach_rate,
            "approach_score": approach_score,
            "rapidly_approaching": rapidly_approaching,
            "motion_state": motion_state,
            "time_to_collision_s": ttc,
            "obstacle_stability": stability,
        }

    # ========================================================
    # FREE PATH & SAFE WALKING CORRIDOR (PHASE 7)
    # ========================================================

    @staticmethod
    def _compute_interval_coverage(
        intervals: List[Tuple[float, float]],
        span_start: float,
        span_end: float,
    ) -> float:
        """
        Compute normalized union coverage of 1D intervals within [span_start, span_end].
        """
        if not intervals or span_end <= span_start:
            return 0.0

        clamped: List[Tuple[float, float]] = []
        for s, e in intervals:
            cs = max(span_start, s)
            ce = min(span_end, e)
            if ce > cs:
                clamped.append((cs, ce))

        if not clamped:
            return 0.0

        clamped.sort(key=lambda x: x[0])
        merged = [clamped[0]]
        for cur in clamped[1:]:
            prev_s, prev_e = merged[-1]
            if cur[0] <= prev_e:
                merged[-1] = (prev_s, max(prev_e, cur[1]))
            else:
                merged.append(cur)

        total_len = sum(e - s for s, e in merged)
        return min(1.0, max(0.0, total_len / (span_end - span_start)))

    def analyze_free_path(
        self,
        scene_depth: Optional[Dict[str, Any]] = None,
        objects: Optional[List[Dict[str, Any]]] = None,
        unknown_objects: Optional[List[Dict[str, Any]]] = None,
        frame_width: int = 640,
        frame_height: int = 480,
    ) -> Dict[str, Any]:
        """
        Phase 7 Free-Path Analysis and Safe Walking Corridor Detection.

        Combines:
        1. Monocular relative depth map regional baselines (LEFT, CENTER, RIGHT)
        2. Detected object bounding boxes (known YOLO + Phase 4 genuine unknowns)
        3. Phase 5 smoothed approximate distances and distance categories
        4. Spatial corridor coverage and horizontal overlap
        5. Walking-relevant vertical ground proximity (suppresses top-of-frame objects)
        6. Independent side corridor clearances and least-dangerous best-direction selection
        """
        scene = scene_depth if isinstance(scene_depth, dict) else {}

        left_depth = self.normalize_depth(
            scene.get("left_depth", scene.get("left", 1.0))
        )
        center_depth = self.normalize_depth(
            scene.get("center_region_depth", scene.get("center_depth", 1.0))
        )
        right_depth = self.normalize_depth(
            scene.get("right_depth", scene.get("right", 1.0))
        )

        scene_left_risk = max(0.0, (0.50 - left_depth) / 0.50) if left_depth < 0.50 else 0.0
        scene_center_risk = max(0.0, (0.50 - center_depth) / 0.50) if center_depth < 0.50 else 0.0
        scene_right_risk = max(0.0, (0.50 - right_depth) / 0.50) if right_depth < 0.50 else 0.0

        width = float(frame_width) if frame_width and frame_width > 0 else 640.0
        height = float(frame_height) if frame_height and frame_height > 0 else 480.0

        all_obstacles: List[Dict[str, Any]] = []
        if isinstance(objects, list):
            all_obstacles.extend(objects)
        if isinstance(unknown_objects, list):
            all_obstacles.extend(unknown_objects)

        center_intervals: List[Tuple[float, float]] = []
        left_intervals: List[Tuple[float, float]] = []
        right_intervals: List[Tuple[float, float]] = []

        center_contribs: List[float] = []
        left_contribs: List[float] = []
        right_contribs: List[float] = []

        center_blocking_flags: List[bool] = []
        left_blocking_flags: List[bool] = []
        right_blocking_flags: List[bool] = []

        max_center_dist_weight = 0.0
        max_left_dist_weight = 0.0
        max_right_dist_weight = 0.0

        has_low_conf = False
        has_med_conf = False

        for obs in all_obstacles:
            bbox = obs.get("bbox") if isinstance(obs.get("bbox"), dict) else {}
            x1 = max(0.0, min(1.0, self._safe_float(bbox.get("x1", 0)) / width))
            y1 = max(0.0, min(1.0, self._safe_float(bbox.get("y1", 0)) / height))
            x2 = max(0.0, min(1.0, self._safe_float(bbox.get("x2", 0)) / width))
            y2 = max(0.0, min(1.0, self._safe_float(bbox.get("y2", 0)) / height))
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            obj_width = x2 - x1

            # Vertical ground-proximity filter:
            # Physical walking hazards are located on or near the ground plane.
            # Bottom edge y2 < 0.35 is in top background/sky/ceiling -> not a walking hazard.
            if y2 < 0.35:
                vertical_factor = 0.0
            else:
                vertical_factor = min(1.0, max(0.0, (y2 - 0.30) / 0.30))

            # Horizontal overlaps with normalized corridors:
            # Left: [0.0, 0.38], span = 0.38
            # Center: [0.38, 0.62], span = 0.24
            # Right: [0.62, 1.0], span = 0.38
            cov_c = max(0.0, min(x2, 0.62) - max(x1, 0.38)) / 0.24
            cov_l = max(0.0, min(x2, 0.38) - max(x1, 0.0)) / 0.38
            cov_r = max(0.0, min(x2, 1.0) - max(x1, 0.62)) / 0.38

            dist_m = self._safe_float(obs.get("distance_m", obs.get("estimated_distance_m", 2.0)))
            dist_cat = str(obs.get("distance_category", "")).upper()
            if not dist_cat or dist_cat == "NONE":
                if dist_m < 1.4:
                    dist_cat = "VERY_CLOSE"
                elif dist_m < 3.5:
                    dist_cat = "NEAR"
                else:
                    dist_cat = "FAR"

            risk = self._safe_float(obs.get("risk_score", 0.0))

            if dist_cat == "VERY_CLOSE":
                dist_weight = 1.0
            elif dist_cat == "NEAR":
                dist_weight = 0.85
            elif dist_cat == "FAR":
                dist_weight = 0.50 if obj_width > 0.60 else 0.25
            else:
                dist_weight = 0.60

            dist_conf = str(obs.get("distance_confidence", "HIGH")).upper()
            det_conf = self._safe_float(obs.get("confidence", 0.8))
            is_unknown = bool(obs.get("class_name") in ("unknown_obstacle", "obstacle", None, ""))
            if dist_conf == "LOW" or det_conf < 0.40:
                has_low_conf = True
            elif dist_conf == "MEDIUM" or det_conf < 0.65 or is_unknown:
                has_med_conf = True

            # Phase 10 Crossing Person Anticipation
            is_crossing = bool(obs.get("is_crossing", False) or obs.get("person_state") == "PERSON_CROSSING")
            crossing_dir = str(obs.get("crossing_direction", "NONE")).upper()
            if is_crossing and crossing_dir in ("LEFT_TO_CENTER", "RIGHT_TO_CENTER", "CROSSING_LEFT", "CROSSING_RIGHT"):
                has_med_conf = True
                if cov_c > 0.04 or abs((x1 + x2) / 2.0 - 0.50) < 0.25:
                    center_blocking_flags.append(True)
                    center_contribs.append(0.65)

            if vertical_factor > 0.0:
                # Center corridor
                if cov_c > 0.02:
                    center_intervals.append((x1, x2))
                    c_contrib = risk * dist_weight * vertical_factor * min(1.0, cov_c * 1.5)
                    center_contribs.append(c_contrib)
                    max_center_dist_weight = max(max_center_dist_weight, dist_weight)
                    if vertical_factor >= 0.35:
                        if (dist_cat == "VERY_CLOSE" and cov_c >= 0.15 and risk >= 0.40) or \
                           (dist_cat == "NEAR" and cov_c >= 0.20 and risk >= 0.45) or \
                           (risk >= 0.65 and cov_c >= 0.25) or \
                           (cov_c >= 0.50 and dist_cat in ("VERY_CLOSE", "NEAR")):
                            center_blocking_flags.append(True)

                # Left corridor
                if cov_l > 0.02:
                    left_intervals.append((x1, x2))
                    l_contrib = risk * dist_weight * vertical_factor * min(1.0, cov_l * 1.5)
                    left_contribs.append(l_contrib)
                    max_left_dist_weight = max(max_left_dist_weight, dist_weight)
                    if vertical_factor >= 0.35:
                        if (dist_cat == "VERY_CLOSE" and cov_l >= 0.15 and risk >= 0.40) or \
                           (dist_cat == "NEAR" and cov_l >= 0.20 and risk >= 0.45) or \
                           (risk >= 0.65 and cov_l >= 0.25) or \
                           (cov_l >= 0.50 and dist_cat in ("VERY_CLOSE", "NEAR")):
                            left_blocking_flags.append(True)

                # Right corridor
                if cov_r > 0.02:
                    right_intervals.append((x1, x2))
                    r_contrib = risk * dist_weight * vertical_factor * min(1.0, cov_r * 1.5)
                    right_contribs.append(r_contrib)
                    max_right_dist_weight = max(max_right_dist_weight, dist_weight)
                    if vertical_factor >= 0.35:
                        if (dist_cat == "VERY_CLOSE" and cov_r >= 0.15 and risk >= 0.40) or \
                           (dist_cat == "NEAR" and cov_r >= 0.20 and risk >= 0.45) or \
                           (risk >= 0.65 and cov_r >= 0.25) or \
                           (cov_r >= 0.50 and dist_cat in ("VERY_CLOSE", "NEAR")):
                            right_blocking_flags.append(True)

        center_coverage = self._compute_interval_coverage(center_intervals, 0.38, 0.62)
        left_coverage = self._compute_interval_coverage(left_intervals, 0.0, 0.38)
        right_coverage = self._compute_interval_coverage(right_intervals, 0.62, 1.0)

        max_c_risk = max(center_contribs) if center_contribs else 0.0
        max_l_risk = max(left_contribs) if left_contribs else 0.0
        max_r_risk = max(right_contribs) if right_contribs else 0.0

        center_risk = min(1.0, max(scene_center_risk, max_c_risk, center_coverage * max_center_dist_weight))
        left_risk = min(1.0, max(scene_left_risk, max_l_risk, left_coverage * max_left_dist_weight))
        right_risk = min(1.0, max(scene_right_risk, max_r_risk, right_coverage * max_right_dist_weight))

        center_blocked = bool(
            center_depth <= self.VERY_CLOSE_DEPTH
            or any(center_blocking_flags)
            or (center_depth <= self.CLOSE_DEPTH and max_c_risk >= 0.35)
            or center_risk >= 0.55
            or (center_coverage >= 0.45 and max_center_dist_weight >= 0.70)
        )
        center_clear = bool(not center_blocked)

        left_blocked = bool(
            left_depth <= self.VERY_CLOSE_DEPTH
            or any(left_blocking_flags)
            or (left_depth <= self.CLOSE_DEPTH and max_l_risk >= 0.35)
            or left_risk >= 0.60
            or (left_coverage >= 0.45 and max_left_dist_weight >= 0.70)
        )
        left_clear = bool(not left_blocked)

        right_blocked = bool(
            right_depth <= self.VERY_CLOSE_DEPTH
            or any(right_blocking_flags)
            or (right_depth <= self.CLOSE_DEPTH and max_r_risk >= 0.35)
            or right_risk >= 0.60
            or (right_coverage >= 0.45 and max_right_dist_weight >= 0.70)
        )
        right_clear = bool(not right_blocked)

        safe_directions: List[str] = []
        if left_clear:
            safe_directions.append("LEFT")
        if center_clear:
            safe_directions.append("CENTER")
        if right_clear:
            safe_directions.append("RIGHT")

        no_safe_path = bool(len(safe_directions) == 0)

        # Best direction selection:
        # Priority:
        # 1. CENTER forward corridor if clear and risk is balanced
        # 2. Safest clear direction among remaining safe directions
        # 3. Least dangerous direction when all corridors are blocked
        if center_clear and (center_risk <= min(left_risk, right_risk) + 0.15):
            best_direction = "CENTER"
        elif safe_directions:
            risk_map = {"LEFT": left_risk, "CENTER": center_risk, "RIGHT": right_risk}
            best_direction = min(safe_directions, key=lambda d: risk_map[d])
        else:
            risk_map = {"LEFT": left_risk, "CENTER": center_risk, "RIGHT": right_risk}
            best_direction = min(risk_map, key=risk_map.get)

        if best_direction == "CENTER":
            corridor_width = round(0.24 * max(0.0, 1.0 - center_coverage), 4)
        elif best_direction == "LEFT":
            corridor_width = round(0.38 * max(0.0, 1.0 - left_coverage), 4)
        else:
            corridor_width = round(0.38 * max(0.0, 1.0 - right_coverage), 4)

        if has_low_conf:
            path_confidence = "LOW"
        elif has_med_conf:
            path_confidence = "MEDIUM"
        else:
            path_confidence = "HIGH"

        return {
            "left_clear": left_clear,
            "center_clear": center_clear,
            "right_clear": right_clear,
            "left_depth": round(left_depth, 4),
            "center_depth": round(center_depth, 4),
            "right_depth": round(right_depth, 4),
            "left_risk": round(left_risk, 4),
            "center_risk": round(center_risk, 4),
            "right_risk": round(right_risk, 4),
            "best_direction": best_direction,

            # Phase 7 Safe Walking Corridor fields
            "center_blocked": center_blocked,
            "left_blocked": left_blocked,
            "right_blocked": right_blocked,
            "center_coverage": round(center_coverage, 4),
            "left_coverage": round(left_coverage, 4),
            "right_coverage": round(right_coverage, 4),
            "corridor_width": corridor_width,
            "safe_directions": safe_directions,
            "no_safe_path": no_safe_path,
            "path_confidence": path_confidence,

            # Backward compatibility
            "left": round(left_depth, 4),
            "center": round(center_depth, 4),
            "right": round(right_depth, 4),
            "is_center_blocked": center_blocked,
        }

    # ========================================================
    # SAFETY LEVEL
    # ========================================================

    def get_safety_level(
        self,
        primary_risk: float,
        emergency_stop: bool,
    ) -> str:
        """
        Convert risk into safety level.
        """

        risk = self._clamp(
            self._safe_float(
                primary_risk
            )
        )

        if emergency_stop:
            return "EMERGENCY"

        # Risk increases from SAFE -> AWARE -> CAUTION -> DANGER.
        # The thresholds are evaluated from highest to lowest so
        # the safety levels remain mutually reachable.
        if risk >= self.danger_threshold:
            return "DANGER"

        if risk >= self.caution_threshold:
            return "CAUTION"

        if risk >= self.aware_threshold:
            return "AWARE"

        return "SAFE"

    # ========================================================
    # EMERGENCY LOGIC
    # ========================================================

    def should_emergency_stop(
        self,
        primary_obstacle: Optional[
            Dict[str, Any]
        ],
        scene: Dict[str, Any],
        predictive_threat: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Determine whether the system should command an
        emergency stop.

        Emergency stop requires a genuinely dangerous
        situation rather than merely detecting an object.

        Strong emergency conditions:

            - predictive threat EMERGENCY
            - rapidly approaching center obstacle <= 1.4m or TTC <= 2.0s
            - very close center obstacle
            - very high risk center obstacle
            - very close human/vehicle in center
            - center path severely blocked
        """

        # ----------------------------------------------------
        # Predictive threat evaluation (Phase 11)
        # ----------------------------------------------------
        if predictive_threat and isinstance(predictive_threat, dict):
            pt_pos = str(predictive_threat.get("position", "")).upper()
            pt_mot = str(predictive_threat.get("motion_state", "")).upper()
            pt_dist = self._safe_float(predictive_threat.get("distance_m", 99.0), 99.0)
            pt_ttc = predictive_threat.get("time_to_collision_s")
            pt_ttc_val = float(pt_ttc) if pt_ttc is not None and math.isfinite(float(pt_ttc)) else None
            pt_threat = str(predictive_threat.get("threat_level", "NONE")).upper()

            if pt_threat == "EMERGENCY":
                return True
            if pt_pos == "CENTER":
                if pt_mot == "RAPIDLY_APPROACHING" and (pt_dist <= 1.4 or (pt_ttc_val is not None and pt_ttc_val <= 2.0)):
                    return True
        # ----------------------------------------------------
        # Scene-level depth emergency check (Requirement 8: Emergency STOP works without YOLO detection)
        # ----------------------------------------------------
        if isinstance(scene, dict):
            scene_center_depth = self.normalize_depth(
                scene.get(
                    "center_region_depth",
                    scene.get("center_depth", 1.0),
                )
            )
            scene_avg_depth = self.normalize_depth(
                scene.get("average_depth", 1.0)
            )
            if scene_center_depth <= 0.08 or (scene_avg_depth <= 0.08 and scene_center_depth <= 0.15):
                return True

        if not primary_obstacle:
            return False

        risk = self._safe_float(
            primary_obstacle.get(
                "risk_score",
                0.0,
            )
        )

        depth = self.normalize_depth(
            primary_obstacle.get(
                "depth",
                1.0,
            )
        )

        position = str(
            primary_obstacle.get(
                "position",
                "",
            )
        ).upper()

        class_name = str(
            primary_obstacle.get(
                "class_name",
                "",
            )
        ).lower()

        center_depth = self.normalize_depth(
            scene.get(
                "center_region_depth",
                scene.get(
                    "center_depth",
                    1.0,
                ),
            )
        )

        curr_dist_m = self._safe_float(
            primary_obstacle.get(
                "distance_m",
                primary_obstacle.get(
                    "estimated_distance_m",
                    99.0,
                ),
            ),
            99.0,
        )

        # ----------------------------------------------------
        # Rapidly approaching center obstacle (Phase 11 / Requirement 8)
        # ----------------------------------------------------
        obs_mot = str(primary_obstacle.get("motion_state", "")).upper()
        obs_ttc = primary_obstacle.get("time_to_collision_s")
        obs_ttc_val = float(obs_ttc) if obs_ttc is not None and math.isfinite(float(obs_ttc)) else None

        if obs_mot == "RAPIDLY_APPROACHING":
            if position == "CENTER" and (curr_dist_m <= 2.5 or (obs_ttc_val is not None and obs_ttc_val <= 2.5)):
                return True

        if position == "CENTER":
            if obs_ttc_val is not None and obs_ttc_val <= 2.0 and curr_dist_m <= 2.0 and obs_mot in ("APPROACHING", "RAPIDLY_APPROACHING"):
                return True

        # ----------------------------------------------------
        # Extremely close physical obstacle <= 0.55m in center
        # ----------------------------------------------------
        if curr_dist_m <= 0.55 and position == "CENTER":
            return True

        if position == "CENTER" and (depth <= self.emergency_threshold or (risk >= 0.90 and curr_dist_m <= 1.0)):
            return True

        # ----------------------------------------------------
        # Center path severely blocked by deep zero-depth wall
        # ----------------------------------------------------
        if (
            position == "CENTER"
            and center_depth <= 0.08
            and risk >= 0.70
        ):
            return True

        return False

    # ========================================================
    # NAVIGATION DECISION ENGINE (PHASE 8)
    # ========================================================

    def _compute_navigation_candidate(
        self,
        safety_level: str,
        emergency_stop: bool,
        primary_obstacle: Optional[Dict[str, Any]],
        free_path: Optional[Dict[str, Any]],
        backward_path_clear: bool = False,
        predictive_threat: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, str]:
        """
        Compute deterministic navigation candidate (action, direction, reason) without hysteresis.
        """
        fp = free_path if isinstance(free_path, dict) else {}
        left_clear = bool(fp.get("left_clear", False))
        center_clear = bool(fp.get("center_clear", False))
        right_clear = bool(fp.get("right_clear", False))
        no_safe_path = bool(fp.get("no_safe_path", False))
        left_risk = self._safe_float(fp.get("left_risk", 1.0))
        right_risk = self._safe_float(fp.get("right_risk", 1.0))
        center_risk = self._safe_float(fp.get("center_risk", 1.0))
        left_cov = self._safe_float(fp.get("left_coverage", 0.0))
        right_cov = self._safe_float(fp.get("right_coverage", 0.0))

        # Extract primary obstacle info
        obs_pos = str(primary_obstacle.get("position", "")).upper() if primary_obstacle else ""
        if not obs_pos and primary_obstacle and "bbox" in primary_obstacle:
            obs_pos = self.get_position(primary_obstacle["bbox"], 640.0)
        if not obs_pos and primary_obstacle:
            obs_pos = "CENTER"
        obs_risk = self._safe_float(primary_obstacle.get("risk_score", 0.0)) if primary_obstacle else 0.0
        obs_dist = self._safe_float(
            primary_obstacle.get("distance_m", primary_obstacle.get("estimated_distance_m", 99.0)),
            99.0,
        ) if primary_obstacle else 99.0
        obs_depth = self.normalize_depth(primary_obstacle.get("depth", 1.0)) if primary_obstacle else 1.0
        obs_cat = str(primary_obstacle.get("distance_category", "")).upper() if primary_obstacle else ""

        # 1. Emergency stop override (stop only if no lateral corridor is clear or collision < 0.5m)
        if emergency_stop or safety_level == "EMERGENCY":
            if not primary_obstacle or center_clear or (not left_clear and not right_clear) or (primary_obstacle and obs_dist <= 0.50):
                return (
                    "STOP",
                    "CENTER",
                    "Emergency stop: immediate obstacle in path",
                )
            prefix = "TURN" if (0.0 < obs_dist <= 0.8 or obs_cat == "VERY_CLOSE") else "MOVE"
            if left_clear and not right_clear:
                return (
                    f"{prefix}_LEFT",
                    "LEFT",
                    f"Obstacle ahead. {prefix.title()} left into clear path.",
                )
            elif right_clear and not left_clear:
                return (
                    f"{prefix}_RIGHT",
                    "RIGHT",
                    f"Obstacle ahead. {prefix.title()} right into clear path.",
                )
            elif left_clear and right_clear:
                chosen = "LEFT" if left_risk <= right_risk else "RIGHT"
                return (
                    f"{prefix}_{chosen}",
                    chosen,
                    f"Obstacle ahead. {prefix.title()} {chosen.lower()} into clear path.",
                )

        # Extract predictive threat / tracking info
        threat_mot = str(predictive_threat.get("motion_state", "")).upper() if predictive_threat else ""
        threat_rapid = bool(threat_mot == "RAPIDLY_APPROACHING")
        threat_app = bool(predictive_threat.get("is_approaching", False)) if predictive_threat else False
        obs_mot = str(primary_obstacle.get("motion_state", "")).upper() if primary_obstacle else ""
        obs_rapid = bool(obs_mot == "RAPIDLY_APPROACHING" or (primary_obstacle and primary_obstacle.get("rapidly_approaching")))
        obs_app = bool(obs_rapid or (primary_obstacle and (primary_obstacle.get("approaching") or primary_obstacle.get("is_approaching"))))

        is_rapid = threat_rapid or obs_rapid
        is_app = threat_app or obs_app

        # 2. Very close or high-risk obstacle -> immediate STOP (Requirement 7)
        if primary_obstacle:
            if (0.0 < obs_dist <= 0.8 and obs_cat != "UNKNOWN_DISTANCE") or (obs_depth <= self.VERY_CLOSE_DEPTH and obs_cat != "UNKNOWN_DISTANCE"):
                if not right_clear and not left_clear:
                    if backward_path_clear:
                        return (
                            "BACK",
                            "BACK",
                            "Obstacle very close and forward corridors blocked; safe backward path clear",
                        )
                    return (
                        "STOP",
                        "CENTER",
                        "Obstacle very close: immediate stop",
                    )
                elif right_clear and not left_clear:
                    return (
                        "TURN_RIGHT",
                        "RIGHT",
                        "Center obstacle very close; turn right into clear corridor",
                    )
                elif left_clear and not right_clear:
                    return (
                        "TURN_LEFT",
                        "LEFT",
                        "Center obstacle very close; turn left into clear corridor",
                    )
                elif right_clear and left_clear:
                    if left_risk < right_risk:
                        return (
                            "TURN_LEFT",
                            "LEFT",
                            "Center obstacle very close; turn left into safer corridor",
                        )
                    return (
                        "TURN_RIGHT",
                        "RIGHT",
                        "Center obstacle very close; turn right into safer corridor",
                    )
            if (obs_risk >= 0.85 or (obs_risk >= 0.75 and obs_dist <= 1.0)) and obs_pos == "CENTER":
                prefix = "TURN" if (0.0 < obs_dist <= 0.8 or obs_cat == "VERY_CLOSE") else "MOVE"
                if not right_clear and not left_clear:
                    if backward_path_clear:
                        return (
                            "BACK",
                            "BACK",
                            "All forward corridors are blocked; safe backward path clear",
                        )
                    return (
                        "STOP",
                        "CENTER",
                        "High-risk obstacle ahead: immediate stop",
                    )
                elif right_clear and not left_clear:
                    return (
                        f"{prefix}_RIGHT",
                        "RIGHT",
                        "Center obstacle blocks forward path; right corridor is clear",
                    )
                elif left_clear and not right_clear:
                    return (
                        f"{prefix}_LEFT",
                        "LEFT",
                        "Center obstacle blocks forward path; left corridor is clear",
                    )

        # 3. Rapidly approaching obstacle -> STOP or SLOW_DOWN (Requirement 8)
        if is_rapid:
            if obs_dist <= 0.8 or (obs_pos == "CENTER" and obs_dist <= 1.4) or (obs_pos == "CENTER" and not left_clear and not right_clear):
                return (
                    "STOP",
                    "CENTER",
                    "Rapidly approaching obstacle in path",
                )
            elif center_clear and obs_pos != "CENTER":
                return (
                    "GO_FORWARD",
                    "CENTER",
                    f"Obstacle on {obs_pos.lower()}; center path is clear",
                )
            elif left_clear and not right_clear:
                return (
                    "TURN_LEFT",
                    "LEFT",
                    "Rapidly approaching obstacle; detour left",
                )
            elif right_clear and not left_clear:
                return (
                    "TURN_RIGHT",
                    "RIGHT",
                    "Rapidly approaching obstacle; detour right",
                )
            else:
                return (
                    "STOP",
                    "CENTER",
                    "Rapidly approaching obstacle in path",
                )

        # 4. Moderately approaching center obstacle -> detour or SLOW_DOWN (Requirement 8)
        if is_app and obs_pos == "CENTER" and obs_dist <= 2.5:
            if left_clear and not right_clear:
                return (
                    "TURN_LEFT",
                    "LEFT",
                    "Approaching center obstacle; left corridor is clear",
                )
            elif right_clear and not left_clear:
                return (
                    "TURN_RIGHT",
                    "RIGHT",
                    "Approaching center obstacle; right corridor is clear",
                )
            elif left_clear and right_clear:
                chosen_side = "RIGHT" if right_risk <= left_risk else "LEFT"
                # Anti-oscillation hysteresis: maintain active turn direction unless other side is substantially safer
                if "LEFT" in self.last_action and left_risk <= right_risk + 0.08:
                    chosen_side = "LEFT"
                elif "RIGHT" in self.last_action and right_risk <= left_risk + 0.08:
                    chosen_side = "RIGHT"
                else:
                    chosen_side = "RIGHT" if right_risk <= left_risk else "LEFT"
                return (
                    f"TURN_{chosen_side}",
                    chosen_side,
                    f"Approaching center obstacle; {chosen_side.lower()} corridor is safer",
                )
            else:
                if obs_dist <= 1.8:
                    return (
                        "STOP",
                        "CENTER",
                        "Approaching center obstacle with no safe detour",
                    )
                return (
                    "SLOW_DOWN",
                    "CENTER",
                    "Approaching center obstacle; slow down",
                )

        # 5. All forward corridors blocked or no safe path
        if no_safe_path or (not left_clear and not center_clear and not right_clear):
            if backward_path_clear:
                return (
                    "BACK",
                    "BACK",
                    "All forward corridors are blocked; safe backward path clear",
                )
            return (
                "STOP",
                "NONE",
                "All forward corridors are blocked",
            )

        # 6. Center corridor blocked or unsafe (Requirement 5)
        center_unsafe = bool(
            not center_clear
            or (obs_pos == "CENTER" and (obs_risk >= 0.45 or obs_dist <= 2.5 or safety_level in ("CAUTION", "DANGER", "EMERGENCY")))
        )

        if center_unsafe:
            # CENTER obstacle: choose a genuinely safe left/right path when available. If no safe path -> STOP.
            prefix = "TURN" if (0.0 < obs_dist <= 0.8 or obs_cat == "VERY_CLOSE" or (primary_obstacle and "distance_m" not in primary_obstacle and "estimated_distance_m" not in primary_obstacle and obs_cat != "FAR")) else "MOVE"
            if left_clear and not right_clear:
                return (
                    f"{prefix}_LEFT",
                    "LEFT",
                    "Center obstacle blocks forward path; left corridor is clear",
                )
            elif right_clear and not left_clear:
                return (
                    f"{prefix}_RIGHT",
                    "RIGHT",
                    "Center obstacle blocks forward path; right corridor is clear",
                )
            elif left_clear and right_clear:
                # Anti-oscillation hysteresis: favor existing active turn direction
                if self.last_action in ("TURN_LEFT", "MOVE_LEFT") and left_risk <= right_risk + 0.08:
                    return (
                        f"{prefix}_LEFT",
                        "LEFT",
                        "Center obstacle blocks forward path; continuing safe left detour",
                    )
                elif self.last_action in ("TURN_RIGHT", "MOVE_RIGHT") and right_risk <= left_risk + 0.08:
                    return (
                        f"{prefix}_RIGHT",
                        "RIGHT",
                        "Center obstacle blocks forward path; continuing safe right detour",
                    )
                elif right_risk < left_risk - 0.04:
                    return (
                        f"{prefix}_RIGHT",
                        "RIGHT",
                        "Center obstacle blocks forward path; right corridor is safer",
                    )
                elif left_risk < right_risk - 0.04:
                    return (
                        f"{prefix}_LEFT",
                        "LEFT",
                        "Center obstacle blocks forward path; left corridor is safer",
                    )
                elif right_cov < left_cov - 0.05:
                    return (
                        f"{prefix}_RIGHT",
                        "RIGHT",
                        "Center obstacle blocks forward path; right corridor has more clearance",
                    )
                elif left_cov < right_cov - 0.05:
                    return (
                        f"{prefix}_LEFT",
                        "LEFT",
                        "Center obstacle blocks forward path; left corridor has more clearance",
                    )
                else:
                    return (
                        f"{prefix}_RIGHT",
                        "RIGHT",
                        "Center obstacle blocks forward path; right corridor is clear",
                    )
            else:
                if backward_path_clear:
                    return (
                        "BACK",
                        "BACK",
                        "Center blocked and side corridors unsafe; safe backward path clear",
                    )
                return (
                    "STOP",
                    "NONE",
                    "Center blocked and no safe side corridor available",
                )

        # 7. LEFT obstacle -> prefer RIGHT when safe (Requirement 6)
        if obs_pos == "LEFT":
            if center_clear and obs_dist > 0.8 and obs_risk < 0.65:
                return (
                    "GO_FORWARD",
                    "CENTER",
                    "Obstacle on left; center path is clear",
                )
            elif right_clear:
                if obs_risk >= 0.40 or obs_dist <= 2.5:
                    return (
                        "MOVE_RIGHT",
                        "RIGHT",
                        "Obstacle on left; shift right into clear corridor",
                    )
                elif center_clear:
                    return (
                        "GO_FORWARD",
                        "CENTER",
                        "Obstacle on left; path is clear to proceed",
                    )
                else:
                    return (
                        "TURN_RIGHT",
                        "RIGHT",
                        "Obstacle on left; right corridor is clear",
                    )
            elif center_clear:
                return (
                    "GO_FORWARD",
                    "CENTER",
                    "Obstacle on left; center path is clear",
                )
            elif left_clear:
                return (
                    "SLOW_DOWN",
                    "CENTER",
                    "Obstacle on left and right blocked; proceed with caution",
                )
            else:
                return (
                    "STOP",
                    "NONE",
                    "Obstacle on left and other paths blocked",
                )

        # 8. RIGHT obstacle -> prefer LEFT when safe (Requirement 6)
        if obs_pos == "RIGHT":
            if center_clear and obs_dist > 0.8 and obs_risk < 0.65:
                return (
                    "GO_FORWARD",
                    "CENTER",
                    "Obstacle on right; center path is clear",
                )
            elif left_clear:
                if obs_risk >= 0.40 or obs_dist <= 2.5:
                    return (
                        "MOVE_LEFT",
                        "LEFT",
                        "Obstacle on right; shift left into clear corridor",
                    )
                elif center_clear:
                    return (
                        "GO_FORWARD",
                        "CENTER",
                        "Obstacle on right; path is clear to proceed",
                    )
                else:
                    return (
                        "TURN_LEFT",
                        "LEFT",
                        "Obstacle on right; left corridor is clear",
                    )
            elif center_clear:
                return (
                    "GO_FORWARD",
                    "CENTER",
                    "Obstacle on right; center path is clear",
                )
            elif right_clear:
                return (
                    "SLOW_DOWN",
                    "CENTER",
                    "Obstacle on right and left blocked; proceed with caution",
                )
            else:
                return (
                    "STOP",
                    "NONE",
                    "Obstacle on right and other paths blocked",
                )

        # 9. Center corridor is clear and safe
        if center_clear:
            if primary_obstacle and obs_cat == "FAR":
                reason = "Distant obstacle ahead; center corridor is safe to proceed"
            else:
                reason = "Center path is clear"
            return ("GO_FORWARD", "CENTER", reason)

        # 10. Fallback if neither center nor sides are clear
        return ("STOP", "NONE", "All forward corridors are blocked")

    def estimate_turn_angle(
        self,
        action: str,
        direction: str,
        free_path: Optional[Dict[str, Any]] = None,
        primary_obstacle: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, str, str]:
        """
        Estimate approximate navigation turn angle target and direction.
        Returns: (turn_angle_deg, turn_direction, turn_confidence)

        IMPORTANT: A single RGB forward-facing camera cannot measure physical user rotation.
        turn_angle_deg is a recommended navigation target, NOT a measured physical angle.
        """
        act = str(action).upper() if action else ""
        d = str(direction).upper() if direction else ""

        if act in ("STOP", "BACK", "GO_BACK", "FORWARD", "GO_FORWARD", "CONTINUE_FORWARD", "SLOW_DOWN") or d in ("NONE", "BACK", "CENTER"):
            return 0.0, "NONE", "HIGH"

        if act in ("TURN_LEFT", "MOVE_LEFT") or d == "LEFT":
            turn_dir = "LEFT"
        elif act in ("TURN_RIGHT", "MOVE_RIGHT") or d == "RIGHT":
            turn_dir = "RIGHT"
        else:
            return 0.0, "NONE", "HIGH"

        fp = free_path if isinstance(free_path, dict) else {}
        center_cov = self._safe_float(fp.get("center_coverage", 0.0))
        center_risk = self._safe_float(fp.get("center_risk", 0.0))
        path_conf = str(fp.get("path_confidence", "")).upper()

        side_clear = bool(fp.get("left_clear", False) if turn_dir == "LEFT" else fp.get("right_clear", False))
        side_cov = self._safe_float(fp.get("left_coverage", 0.0) if turn_dir == "LEFT" else fp.get("right_coverage", 0.0))
        side_risk = self._safe_float(fp.get("left_risk", 1.0) if turn_dir == "LEFT" else fp.get("right_risk", 1.0))

        has_geometry = bool(fp and ("center_coverage" in fp or "center_risk" in fp or "left_clear" in fp or "right_clear" in fp))

        if not has_geometry:
            return 30.0, turn_dir, "LOW"

        # Lateral shift vs full turn angle
        if act in ("MOVE_LEFT", "MOVE_RIGHT"):
            angle = 15.0
        elif center_cov >= 0.70 or center_risk >= 0.80 or (center_cov >= 0.50 and side_cov <= 0.15):
            angle = 45.0
        elif center_cov >= 0.35 or center_risk >= 0.50:
            angle = 30.0
        else:
            angle = 15.0

        angle = max(15.0, min(60.0, float(angle)))

        if path_conf == "HIGH" and side_clear and side_risk < 0.40:
            conf = "HIGH"
        elif path_conf in ("MEDIUM", "HIGH") or side_clear:
            conf = "MEDIUM"
        else:
            conf = "LOW"

        return angle, turn_dir, conf

    def determine_navigation(
        self,
        safety_level: str,
        emergency_stop: bool,
        primary_obstacle: Optional[Dict[str, Any]],
        free_path: Optional[Dict[str, Any]],
        backward_path_clear: bool = False,
        predictive_threat: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Determine navigation command with temporal hysteresis to prevent oscillation.
        Returns canonical navigation dictionary.
        """
        cand_action, cand_direction, cand_reason = self._compute_navigation_candidate(
            safety_level=safety_level,
            emergency_stop=emergency_stop,
            primary_obstacle=primary_obstacle,
            free_path=free_path,
            backward_path_clear=backward_path_clear,
            predictive_threat=predictive_threat,
        )

        threat_is_rapid_center = bool(
            predictive_threat
            and str(predictive_threat.get("position", "")).upper() == "CENTER"
            and str(predictive_threat.get("motion_state", "")).upper() == "RAPIDLY_APPROACHING"
        )
        if not threat_is_rapid_center and primary_obstacle:
            if (
                str(primary_obstacle.get("position", "")).upper() == "CENTER"
                and str(primary_obstacle.get("motion_state", "")).upper() == "RAPIDLY_APPROACHING"
            ):
                threat_is_rapid_center = True

        # Extract obstacle position and risk for hysteresis decisions
        obs_pos = str(primary_obstacle.get("position", "")).upper() if primary_obstacle else ""
        obs_risk = self._safe_float(primary_obstacle.get("risk_score", 0.0)) if primary_obstacle else 0.0
        obs_dist = self._safe_float(
            primary_obstacle.get("distance_m", primary_obstacle.get("estimated_distance_m", 99.0)),
            99.0,
        ) if primary_obstacle else 99.0
        fp = free_path if isinstance(free_path, dict) else {}
        center_is_clear = bool(fp.get("center_clear", True))
        left_clear = bool(fp.get("left_clear", False))
        right_clear = bool(fp.get("right_clear", False))
        left_risk = self._safe_float(fp.get("left_risk", 1.0))
        right_risk = self._safe_float(fp.get("right_risk", 1.0))
        center_risk = self._safe_float(fp.get("center_risk", 1.0))

        # Navigation state machine handling & STOP suppression
        now_ts = time.time()
        is_cand_stop = cand_action in ("STOP", "EMERGENCY_STOP")

        curr_threat_id = None
        curr_threat_cls = ""
        if primary_obstacle:
            curr_threat_id = primary_obstacle.get("track_id", primary_obstacle.get("id"))
            curr_threat_cls = str(primary_obstacle.get("class_name", ""))

        is_same_stop_threat = bool(
            self.stop_threat_id is not None
            and (curr_threat_id == self.stop_threat_id or (curr_threat_cls and curr_threat_cls == self.stop_threat_class))
        )

        if is_cand_stop:
            if self.nav_state in ("STOP", "WAIT_FOR_CHANGE"):
                self.stop_frame_count += 1
                if is_same_stop_threat or (primary_obstacle and obs_dist >= 0.40):
                    self.nav_state = "WAIT_FOR_CHANGE"
                    # After 8 frames (~2.5s) in stop, assess recovery
                    if self.stop_frame_count >= 8 and not self.recovery_suggested:
                        self.recovery_suggested = True
                        if backward_path_clear:
                            cand_action = "BACK"
                            cand_direction = "BACK"
                            cand_reason = "Move back for 3 steps to clear blockage"
                            self.nav_state = "RECOVER"
                        elif left_clear:
                            cand_action = "TURN_LEFT"
                            cand_direction = "LEFT"
                            cand_reason = "Turn left, then move forward to bypass blockage"
                            self.nav_state = "RECOVER"
                        elif right_clear:
                            cand_action = "TURN_RIGHT"
                            cand_direction = "RIGHT"
                            cand_reason = "Turn right, then move forward to bypass blockage"
                            self.nav_state = "RECOVER"
                        else:
                            cand_action = "RECOVER"
                            cand_direction = "NONE"
                            cand_reason = "Turn around, then move forward"
                            self.nav_state = "RECOVER"
                        self.should_speak = True
                    else:
                        self.should_speak = False
                else:
                    self.stop_threat_id = curr_threat_id
                    self.stop_threat_class = curr_threat_cls
                    self.should_speak = True
            else:
                self.nav_state = "STOP"
                self.stop_frame_count = 1
                self.stop_threat_id = curr_threat_id
                self.stop_threat_class = curr_threat_cls
                self.stop_timestamp = now_ts
                self.should_speak = True
                self.recovery_suggested = False
            self.path_cleared_from_stop = False
        else:
            if self.nav_state in ("STOP", "WAIT_FOR_CHANGE", "RECOVER"):
                self.stop_frame_count = 0
                self.stop_threat_id = None
                self.stop_threat_class = ""
                self.recovery_suggested = False
                self.should_speak = True
                self.path_cleared_from_stop = True
            else:
                self.path_cleared_from_stop = False
            self.nav_state = cand_action

        self.state_changed = bool(self.last_nav_state != self.nav_state)
        self.last_nav_state = self.nav_state

        # Closed-loop verification & hysteresis filtering
        # 1. Emergency stop override (Requirement 7: Emergency STOP overrides all commands immediately)
        if cand_action in ("STOP", "EMERGENCY_STOP") and self.nav_state != "RECOVER":
            self.last_action = "STOP"
            self.candidate_action = "STOP"
            self.action_vote_count = 2
            self.closed_loop_state = "EMERGENCY_OVERRIDE" if emergency_stop else "STOP"
            self.last_maneuver_command = "NONE"
            action = "STOP"
            direction = cand_direction
            reason = cand_reason

        # 2. Immediate safety response for SLOW_DOWN
        elif cand_action == "SLOW_DOWN":
            self.last_action = "SLOW_DOWN"
            self.candidate_action = "SLOW_DOWN"
            self.action_vote_count = 2
            self.closed_loop_state = "SLOW_DOWN"
            self.last_maneuver_command = "NONE"
            action = "SLOW_DOWN"
            direction = cand_direction
            reason = cand_reason

        # 3. Closed-Loop verification for in-flight maneuvers (MOVE_LEFT, MOVE_RIGHT, TURN_LEFT, TURN_RIGHT)
        elif self.last_maneuver_command in ("TURN_LEFT", "TURN_RIGHT", "MOVE_LEFT", "MOVE_RIGHT"):
            # 3A. Verification: Path cleared after maneuver (Maneuver Succeeded)
            if cand_action in ("FORWARD", "GO_FORWARD", "CONTINUE_FORWARD") and center_is_clear and (not primary_obstacle or obs_pos != "CENTER" or obs_risk < 0.40):
                self.last_action = "FORWARD"
                self.candidate_action = "FORWARD"
                self.action_vote_count = 2
                self.closed_loop_state = "PATH_CLEARED"
                self.last_maneuver_command = "NONE"
                action = "FORWARD"
                direction = "CENTER"
                reason = "Maneuver verified: forward path is clear"

            # 3B. Verification: Maneuver path obstructed or new obstacle appears (Requirements 4, 5, 6)
            # Never blindly continue old maneuver if corridor is blocked or candidate changed
            elif (
                (self.last_maneuver_command in ("TURN_LEFT", "MOVE_LEFT") and (not left_clear or obs_pos == "LEFT"))
                or (self.last_maneuver_command in ("TURN_RIGHT", "MOVE_RIGHT") and (not right_clear or obs_pos == "RIGHT"))
                or (cand_action != self.last_maneuver_command and cand_action in ("TURN_LEFT", "TURN_RIGHT", "MOVE_LEFT", "MOVE_RIGHT", "STOP", "BACK", "GO_BACK"))
            ):
                self.last_action = cand_action
                self.candidate_action = cand_action
                self.action_vote_count = 2
                self.closed_loop_state = "OBSTACLE_AFTER_MANEUVER" if primary_obstacle else "PATH_BLOCKED_RECALCULATED"
                self.last_maneuver_command = cand_action if cand_action in ("TURN_LEFT", "TURN_RIGHT", "MOVE_LEFT", "MOVE_RIGHT") else "NONE"
                action = cand_action
                direction = cand_direction
                reason = f"Maneuver path blocked; recalculated to {cand_action} ({cand_reason})"

            # 3C. Verification: Maneuver still active, safe, and needed
            elif cand_action == self.last_maneuver_command:
                self.last_action = cand_action
                self.candidate_action = cand_action
                self.action_vote_count = 2
                self.closed_loop_state = "MANEUVER_IN_PROGRESS"
                action = cand_action
                direction = cand_direction
                reason = cand_reason

            else:
                # Fallback to candidate action
                self.last_action = cand_action
                self.candidate_action = cand_action
                self.action_vote_count = 2
                self.closed_loop_state = "MANEUVER_UPDATED"
                self.last_maneuver_command = cand_action if cand_action in ("TURN_LEFT", "TURN_RIGHT", "MOVE_LEFT", "MOVE_RIGHT") else "NONE"
                action = cand_action
                direction = cand_direction
                reason = cand_reason

        # 4. Immediate avoidance when path is obstructed and no maneuver was previously active
        elif (threat_is_rapid_center or not center_is_clear or (primary_obstacle and obs_risk >= 0.40)) and cand_action in ("TURN_LEFT", "TURN_RIGHT", "MOVE_LEFT", "MOVE_RIGHT") and self.last_action not in ("TURN_LEFT", "TURN_RIGHT", "MOVE_LEFT", "MOVE_RIGHT"):
            self.last_action = cand_action
            self.candidate_action = cand_action
            self.action_vote_count = 2
            self.closed_loop_state = "MANEUVER_INITIATED"
            self.last_maneuver_command = cand_action
            action = cand_action
            direction = cand_direction
            reason = cand_reason

        # 5. Immediate escape when all forward corridors are blocked
        elif cand_action in ("BACK", "GO_BACK") and backward_path_clear:
            self.last_action = cand_action
            self.candidate_action = cand_action
            self.action_vote_count = 2
            self.closed_loop_state = "ESCAPE_MANEUVER"
            self.last_maneuver_command = cand_action
            action = cand_action
            direction = cand_direction
            reason = cand_reason

        # 5B. Recovery action when stationary stop has persisted
        elif self.nav_state == "RECOVER" or cand_action == "RECOVER":
            self.last_action = cand_action
            self.candidate_action = cand_action
            self.action_vote_count = 2
            self.closed_loop_state = "RECOVERY_MANEUVER"
            self.last_maneuver_command = cand_action if cand_action in ("TURN_LEFT", "TURN_RIGHT", "MOVE_LEFT", "MOVE_RIGHT") else "NONE"
            action = cand_action
            direction = cand_direction
            reason = cand_reason

        # 5C. Path cleared after STOP: immediately transition to FORWARD without holding STOP
        elif self.last_action in ("STOP", "EMERGENCY_STOP") and cand_action in ("FORWARD", "GO_FORWARD", "CONTINUE_FORWARD") and center_is_clear and not emergency_stop and (not primary_obstacle or obs_pos != "CENTER" or obs_risk < 0.35):
            self.last_action = cand_action
            self.candidate_action = cand_action
            self.action_vote_count = 2
            self.closed_loop_state = "PATH_CLEARED"
            self.last_maneuver_command = "NONE"
            action = cand_action
            direction = cand_direction
            reason = "Path cleared after stop: forward path is clear"

        # 6. Default / Clear Path / Anti-Oscillation Hysteresis (Requirement 8)
        else:
            # Anti-oscillation guard: prevent rapid flipping between LEFT and RIGHT if both are clear
            if (
                self.last_action in ("TURN_LEFT", "MOVE_LEFT")
                and cand_action in ("TURN_RIGHT", "MOVE_RIGHT")
                and left_clear
                and left_risk <= right_risk + 0.10
            ):
                action = self.last_action
                direction = "LEFT"
                reason = "Continuing left maneuver to prevent oscillation"
                self.closed_loop_state = "ANTI_OSCILLATION_HOLD"
            elif (
                self.last_action in ("TURN_RIGHT", "MOVE_RIGHT")
                and cand_action in ("TURN_LEFT", "MOVE_LEFT")
                and right_clear
                and right_risk <= left_risk + 0.10
            ):
                action = self.last_action
                direction = "RIGHT"
                reason = "Continuing right maneuver to prevent oscillation"
                self.closed_loop_state = "ANTI_OSCILLATION_HOLD"
            else:
                is_same = (cand_action == self.last_action) or (
                    cand_action in ("FORWARD", "GO_FORWARD") and self.last_action in ("FORWARD", "GO_FORWARD")
                ) or (
                    cand_action in ("BACK", "GO_BACK") and self.last_action in ("BACK", "GO_BACK")
                )
                if is_same:
                    self.candidate_action = cand_action
                    self.action_vote_count = 2
                    action = cand_action
                    direction = cand_direction
                    reason = cand_reason
                    self.closed_loop_state = "CLEAR_PATH" if cand_action in ("FORWARD", "GO_FORWARD") else "MAINTAINED"
                    if cand_action in ("FORWARD", "GO_FORWARD"):
                        self.last_maneuver_command = "NONE"
                else:
                    if cand_action == self.candidate_action:
                        self.action_vote_count += 1
                        if self.action_vote_count >= 2:
                            self.last_action = cand_action
                            action = cand_action
                            direction = cand_direction
                            reason = cand_reason
                            self.closed_loop_state = "CONFIRMED_ACTION"
                            if cand_action in ("FORWARD", "GO_FORWARD"):
                                self.last_maneuver_command = "NONE"
                        else:
                            if not center_is_clear and self.last_action in ("FORWARD", "GO_FORWARD"):
                                action = "SLOW_DOWN"
                                direction = "CENTER"
                                reason = "Center path obstructed: slowing down pending confirmation"
                                self.closed_loop_state = "CAUTION_PENDING"
                            else:
                                action = self.last_action
                                direction = (
                                    "CENTER"
                                    if action in ("FORWARD", "GO_FORWARD", "CONTINUE_FORWARD", "SLOW_DOWN")
                                    else ("LEFT" if action in ("TURN_LEFT", "MOVE_LEFT") else ("RIGHT" if action in ("TURN_RIGHT", "MOVE_RIGHT") else "NONE"))
                                )
                                reason = f"Holding {action.lower()} pending confirmation"
                                self.closed_loop_state = "HOLDING_PENDING"
                    else:
                        self.candidate_action = cand_action
                        self.action_vote_count = 1
                        if not center_is_clear and self.last_action in ("FORWARD", "GO_FORWARD"):
                            action = "SLOW_DOWN"
                            direction = "CENTER"
                            reason = "Center path obstructed: slowing down pending confirmation"
                            self.closed_loop_state = "CAUTION_PENDING"
                        else:
                            action = self.last_action
                            direction = (
                                "CENTER"
                                if action in ("FORWARD", "GO_FORWARD", "CONTINUE_FORWARD", "SLOW_DOWN")
                                else ("LEFT" if action in ("TURN_LEFT", "MOVE_LEFT") else ("RIGHT" if action in ("TURN_RIGHT", "MOVE_RIGHT") else "NONE"))
                            )
                            reason = f"Holding {action.lower()} pending confirmation"
                            self.closed_loop_state = "HOLDING_PENDING"

        self.last_direction = direction
        self.direction_vote_count = self.action_vote_count

        # Determine turn angle and direction (Phase 9)
        turn_angle_deg, turn_direction, turn_confidence = self.estimate_turn_angle(
            action=action,
            direction=direction,
            free_path=free_path,
            primary_obstacle=primary_obstacle,
        )

        # Stage and movement action mapping (Phase 9 & Requirement 4)
        if (emergency_stop and action != "RECOVER") or action in ("STOP", "EMERGENCY_STOP"):
            navigation_stage = "STOP"
            turn_required = False
            movement_required = False
            turn_direction = "NONE"
            movement_action = "NONE"
            turn_angle_deg = 0.0
            self.turn_state = "NONE"
        elif action == "RECOVER":
            navigation_stage = "TURN"
            turn_required = True
            movement_required = False
            turn_direction = "AROUND"
            movement_action = "NONE"
            turn_angle_deg = 180.0
            self.turn_state = "NONE"
        elif action == "TURN_LEFT":
            navigation_stage = "TURN"
            turn_required = True
            movement_required = False
            turn_direction = "LEFT"
            movement_action = "NONE"
            self.turn_state = "TURNING_LEFT"
        elif action == "TURN_RIGHT":
            navigation_stage = "TURN"
            turn_required = True
            movement_required = False
            turn_direction = "RIGHT"
            movement_action = "NONE"
            self.turn_state = "TURNING_RIGHT"
        elif action == "MOVE_LEFT":
            navigation_stage = "MOVE"
            turn_required = False
            movement_required = True
            turn_direction = "LEFT"
            movement_action = "WALK_LEFT"
            self.turn_state = "NONE"
        elif action == "MOVE_RIGHT":
            navigation_stage = "MOVE"
            turn_required = False
            movement_required = True
            turn_direction = "RIGHT"
            movement_action = "WALK_RIGHT"
            self.turn_state = "NONE"
        elif action in ("FORWARD", "GO_FORWARD", "CONTINUE_FORWARD"):
            navigation_stage = "MOVE"
            turn_required = False
            movement_required = True
            turn_direction = "NONE"
            movement_action = "WALK_FORWARD"
            if self.turn_state in ("TURNING_LEFT", "TURNING_RIGHT"):
                self.turn_state = "READY_TO_MOVE"
            else:
                self.turn_state = "NONE"
        elif action == "SLOW_DOWN":
            navigation_stage = "MOVE"
            turn_required = False
            movement_required = True
            turn_direction = "NONE"
            movement_action = "WALK_FORWARD"
            self.turn_state = "NONE"
        elif action in ("BACK", "GO_BACK"):
            if backward_path_clear:
                navigation_stage = "MOVE"
                turn_required = False
                movement_required = True
                turn_direction = "NONE"
                movement_action = "WALK_BACK"
                self.turn_state = "NONE"
            else:
                navigation_stage = "STOP"
                turn_required = False
                movement_required = False
                turn_direction = "NONE"
                movement_action = "NONE"
                turn_angle_deg = 0.0
                self.turn_state = "NONE"
        else:
            navigation_stage = "IDLE"
            turn_required = False
            movement_required = False
            turn_direction = "NONE"
            movement_action = "NONE"
            turn_angle_deg = 0.0
            self.turn_state = "NONE"

        primary_dist = 0.0
        conf = "HIGH"

        if primary_obstacle:
            primary_dist = round(self._safe_float(primary_obstacle.get("distance_m", 0.0)), 2)
            conf = str(primary_obstacle.get("distance_confidence", "HIGH")).upper()
            if conf not in ("HIGH", "MEDIUM", "LOW"):
                conf = "HIGH"

        # Estimate movement distance and steps (Phase 10 & 11)
        movement_dist_m = 0.0
        step_conf = "HIGH"
        step_reason = "No forward movement during stop/turn"
        movement_steps = 0

        if action in ("FORWARD", "GO_FORWARD", "CONTINUE_FORWARD"):
            movement_dist_m, step_conf, step_reason = self.estimate_movement_distance(
                distance_m=primary_dist,
                distance_category=primary_obstacle.get("distance_category", "NEAR") if primary_obstacle else "FAR",
                action=action,
                free_path=free_path,
            )
            movement_steps = self.estimate_walking_steps(movement_dist_m, self.step_length_m)
            if not primary_obstacle and movement_steps < 1:
                movement_steps = 4
                movement_dist_m = 3.0
            self.last_steps = movement_steps
        elif action == "SLOW_DOWN":
            movement_dist_m, step_conf, step_reason = self.estimate_movement_distance(
                distance_m=primary_dist,
                distance_category=primary_obstacle.get("distance_category", "NEAR") if primary_obstacle else "FAR",
                action="SLOW_DOWN",
                free_path=free_path,
            )
            movement_steps = max(1, self.estimate_walking_steps(movement_dist_m, self.step_length_m))
            self.last_steps = movement_steps
        elif action == "MOVE_LEFT":
            movement_dist_m = 1.5
            movement_steps = 2
            step_conf = "HIGH"
            step_reason = "Move left for 2 steps"
            self.last_steps = movement_steps
        elif action == "MOVE_RIGHT":
            movement_dist_m = 2.25
            movement_steps = 3
            step_conf = "HIGH"
            step_reason = "Move right for 3 steps"
            self.last_steps = movement_steps
        elif action in ("BACK", "GO_BACK") and backward_path_clear:
            movement_dist_m = 2.25
            movement_steps = 3
            step_conf = "HIGH"
            step_reason = "Move back for 3 steps"
            self.last_steps = movement_steps
        else:
            self.last_steps = 0

        steps = movement_steps
        norm_action = action
        if norm_action in ("FORWARD", "GO_FORWARD", "CONTINUE_FORWARD"):
            norm_action = "GO_FORWARD"
        elif norm_action in ("BACK", "GO_BACK"):
            norm_action = "GO_BACK"

        return {
            "action": norm_action,
            "direction": direction,
            "distance_m": primary_dist,
            "steps": steps,
            "movement_steps": movement_steps,
            "movement_distance_m": movement_dist_m,
            "step_confidence": step_conf,
            "step_reason": step_reason,
            "confidence": conf,
            "turn_required": turn_required,
            "movement_required": movement_required,
            "reason": reason,
            "turn_angle_deg": turn_angle_deg,
            "turn_direction": turn_direction,
            "movement_action": movement_action,
            "navigation_stage": navigation_stage,
            "turn_confidence": turn_confidence,
            "angle": turn_angle_deg,
            "closed_loop_state": self.closed_loop_state,
            "environment_verified": True,
            "nav_state": self.nav_state,
            "should_speak": self.should_speak,
            "state_changed": self.state_changed,
        }

    # Backward compatibility alias
    def _compute_navigation(
        self,
        safety_level: str,
        emergency_stop: bool,
        primary_obstacle: Optional[Dict[str, Any]],
        free_path: Dict[str, Any],
    ) -> str:
        res = self._compute_navigation_candidate(
            safety_level=safety_level,
            emergency_stop=emergency_stop,
            primary_obstacle=primary_obstacle,
            free_path=free_path,
        )
        return res[0]

    # ========================================================
    # VOICE INSTRUCTION
    # ========================================================

    def generate_voice_instruction(
        self,
        safety_level: str,
        emergency_stop: bool,
        primary_obstacle: Optional[
            Dict[str, Any]
        ],
        navigation: str,
        objects: Optional[
            List[Dict[str, Any]]
        ] = None,
        steps: Optional[int] = None,
        predictive_threat: Optional[Dict[str, Any]] = None,
        indoor_safety: Optional[Dict[str, Any]] = None,
        primary_semantic_sign: Optional[Dict[str, Any]] = None,
        multi_object: Optional[Dict[str, Any]] = None,
        turn_angle_deg: Optional[float] = None,
    ) -> str:
        """
        Generate short, actionable English voice guidance.

        Strictly directive and concise:
            - Stop immediately.
            - Stop. Path is blocked.
            - Move left.
            - Move right.
            - Move forward.
            - Move forward about five steps.
            - Turn left.
            - Turn right.
            - Turn left about 45 degrees.
            - Turn right about 45 degrees.
            - Go back.
            - Slow down.
        """
        def _step_word(n: int) -> str:
            if n == 1:
                return "for 1 step"
            return f"for {n} steps"

        nav = str(navigation).upper() if navigation else ""

        # Rapidly approaching check
        is_rapid = False
        if predictive_threat and isinstance(predictive_threat, dict):
            if str(predictive_threat.get("motion_state", "")).upper() == "RAPIDLY_APPROACHING":
                is_rapid = True
        if primary_obstacle and isinstance(primary_obstacle, dict):
            if str(primary_obstacle.get("motion_state", "")).upper() == "RAPIDLY_APPROACHING" or primary_obstacle.get("rapidly_approaching"):
                is_rapid = True

        # If a safe lateral detour is active, ALWAYS speak the directional command
        if nav in ("MOVE_LEFT", "MOVE_RIGHT", "TURN_LEFT", "TURN_RIGHT"):
            pass  # Fall through to detailed directional handlers below
        elif (emergency_stop or safety_level == "EMERGENCY" or is_rapid) and nav == "STOP":
            if indoor_safety and isinstance(indoor_safety, dict):
                stair = indoor_safety.get("stair_analysis", {})
                if stair.get("detected") and stair.get("hazard_level") == "CRITICAL":
                    if stair.get("type") == "DROP_OFF":
                        return "Stop. Drop-off ahead."
                    return "Stop. Stairs detected ahead."
                floors = indoor_safety.get("floor_hazards", [])
                if any(f.get("hazard_level") == "CRITICAL" for f in floors if isinstance(f, dict)):
                    return "Hole or depression on the floor ahead. Stop."
                corridor = indoor_safety.get("corridor_analysis", {})
                if corridor.get("clearance_state") == "BLOCKED":
                    return "Stop. Path blocked."
            if primary_obstacle and primary_obstacle.get("class_name") and "unknown" not in str(primary_obstacle.get("class_name", "")).lower():
                cname = self._format_object_name(primary_obstacle.get("class_name", ""))
                obs_pos = str(primary_obstacle.get("position", "CENTER")).upper()
                if obs_pos == "LEFT":
                    return f"Stop. {cname} on your left."
                elif obs_pos == "RIGHT":
                    return f"Stop. {cname} on your right."
                return f"Stop. {cname} ahead."
            return "Stop immediately."

        # Navigation action-based directives
        if nav == "STOP":
            if indoor_safety and isinstance(indoor_safety, dict):
                stair = indoor_safety.get("stair_analysis", {})
                if stair.get("detected") and stair.get("hazard_level") == "CRITICAL":
                    if stair.get("type") == "DROP_OFF":
                        return "Stop. Drop-off ahead."
                    return "Stop. Stairs detected ahead."
                floors = indoor_safety.get("floor_hazards", [])
                if any(f.get("hazard_level") == "CRITICAL" for f in floors if isinstance(f, dict)):
                    return "Hole or depression on the floor ahead. Stop."
                door = indoor_safety.get("door_analysis", {})
                if door.get("detected") and door.get("state") == "CLOSED":
                    return "Door closed. Do not enter."
                if door.get("detected") and door.get("state") == "BLOCKED":
                    return "Stop. The doorway ahead is blocked."
                corridor = indoor_safety.get("corridor_analysis", {})
                if corridor.get("clearance_state") == "BLOCKED":
                    return "Stop. Path blocked."
                glass = indoor_safety.get("glass_obstacles", {})
                if glass.get("detected") and glass.get("risk_score", 0.0) >= 0.70:
                    return glass.get("voice_message") or "Possible glass obstacle ahead."

            # Crowd path blockage check (Phase 10)
            if multi_object and isinstance(multi_object, dict):
                if multi_object.get("crowd_blocking") or (multi_object.get("is_crowded") and multi_object.get("center_obstacles", 0) >= 2):
                    return "Stop. Path ahead is blocked."

            # Semantic warning sign in center
            if primary_semantic_sign and primary_semantic_sign.get("type") == "WARNING":
                clean = str(primary_semantic_sign.get("clean_text", "")).upper()
                if "DO NOT ENTER" in clean or "NO ENTRY" in clean:
                    return "Warning. Do not enter."
                return "Warning. Danger ahead."

            if primary_obstacle and primary_obstacle.get("class_name"):
                obs_cls = str(primary_obstacle.get("class_name", "")).lower()
                obs_name = "Unknown obstacle" if "unknown" in obs_cls else self._format_object_name(primary_obstacle.get("class_name", ""))
                obs_pos = str(primary_obstacle.get("position", "CENTER")).upper()
                if obs_pos == "LEFT":
                    return f"Stop. {obs_name} on your left."
                elif obs_pos == "RIGHT":
                    return f"Stop. {obs_name} on your right."
                return f"Stop. {obs_name} ahead."
            obs_dist = self._safe_float(primary_obstacle.get("distance_m", 99.0)) if primary_obstacle else 99.0
            obs_risk = self._safe_float(primary_obstacle.get("risk_score", 0.0)) if primary_obstacle else 0.0
            if obs_dist <= 0.8 or obs_risk >= 0.70 or safety_level in ("DANGER", "EMERGENCY") or emergency_stop:
                return "Stop immediately."
            if primary_obstacle is None and safety_level == "SAFE" and not emergency_stop:
                return "Stop."
            return "Stop. Path is blocked."

        if nav == "RECOVER":
            return "Turn around, then move forward."

        if nav == "TURN_LEFT":
            if getattr(self, "nav_state", "") == "RECOVER":
                return "Turn left, then move forward."
            if primary_obstacle and isinstance(primary_obstacle, dict):
                obs_cls = str(primary_obstacle.get("class_name", "")).lower()
                obs_name = "Unknown obstacle" if "unknown" in obs_cls else self._format_object_name(primary_obstacle.get("class_name", ""))
                obs_pos = str(primary_obstacle.get("position", "CENTER")).upper()
                pos_phr = "on the right" if obs_pos == "RIGHT" else ("on the left" if obs_pos == "LEFT" else "ahead")
                return f"{obs_name} {pos_phr}. Turn left."
            if turn_angle_deg and 30 <= turn_angle_deg <= 75:
                return f"Turn left about {round(turn_angle_deg)} degrees."
            return "Turn left, then move forward."

        if nav == "TURN_RIGHT":
            if getattr(self, "nav_state", "") == "RECOVER":
                return "Turn right, then move forward."
            if primary_obstacle and isinstance(primary_obstacle, dict):
                obs_cls = str(primary_obstacle.get("class_name", "")).lower()
                obs_name = "Unknown obstacle" if "unknown" in obs_cls else self._format_object_name(primary_obstacle.get("class_name", ""))
                obs_pos = str(primary_obstacle.get("position", "CENTER")).upper()
                pos_phr = "on the left" if obs_pos == "LEFT" else ("on the right" if obs_pos == "RIGHT" else "ahead")
                return f"{obs_name} {pos_phr}. Turn right."
            if turn_angle_deg and 30 <= turn_angle_deg <= 75:
                return f"Turn right about {round(turn_angle_deg)} degrees."
            return "Turn right, then move forward."

        if nav == "MOVE_LEFT":
            if primary_obstacle and isinstance(primary_obstacle, dict):
                obs_cls = str(primary_obstacle.get("class_name", "")).lower()
                obs_name = self._format_object_name(primary_obstacle.get("class_name", ""))
                obs_pos = str(primary_obstacle.get("position", "RIGHT")).upper()
                if "unknown" in obs_cls:
                    if obs_pos == "RIGHT":
                        return "Unknown obstacle on your right. Move left."
                    return "Unknown obstacle ahead. Move left."
                if obs_pos == "RIGHT":
                    return f"Move left. {obs_name} on the right."
                elif obs_pos in ("CENTER", "AHEAD", "NONE", ""):
                    return f"{obs_name} ahead. Move left."
            if steps and steps >= 1:
                return f"Move left for {steps} steps."
            return "Move left for 2 steps."

        if nav == "MOVE_RIGHT":
            if primary_obstacle and isinstance(primary_obstacle, dict):
                obs_cls = str(primary_obstacle.get("class_name", "")).lower()
                obs_pos = str(primary_obstacle.get("position", "LEFT")).upper()
                if "unknown" in obs_cls:
                    if obs_pos == "LEFT":
                        return "Unknown obstacle on your left. Move right."
                    return "Unknown obstacle ahead. Move right."
                obs_name = self._format_object_name(obs_cls)
                if obs_pos == "LEFT":
                    return f"Move right. {obs_name} on the left."
                elif obs_pos in ("CENTER", "AHEAD", "NONE", ""):
                    return f"{obs_name} ahead. Move right."
            if steps and steps >= 1:
                return f"Move right for {steps} steps."
            return "Move right for 3 steps."

        if nav == "SLOW_DOWN":
            if indoor_safety and isinstance(indoor_safety, dict):
                stair = indoor_safety.get("stair_analysis", {})
                if stair.get("detected"):
                    stype = stair.get("type")
                    if stype == "STAIRS_UP":
                        return "Caution. Stairs up ahead."
                    elif stype == "RAMP":
                        return "Caution. Ramp ahead."
                    elif stype in ("SINGLE_STEP", "UNCERTAIN_LEVEL_TRANSITION"):
                        return "Slow down. Floor level change ahead."
                floors = indoor_safety.get("floor_hazards", [])
                caution_floors = [f for f in floors if isinstance(f, dict) and f.get("hazard_level") == "CAUTION"]
                if caution_floors:
                    cf_type = caution_floors[0].get("hazard_type", "")
                    if cf_type in ("CABLE", "TRIP_HAZARD"):
                        return "Trip hazard on floor ahead. Slow down."
                    elif cf_type in ("THRESHOLD", "UNEVEN_FLOOR"):
                        return "Uneven surface ahead."
                    return "Caution. Floor hazard ahead."
                door = indoor_safety.get("door_analysis", {})
                if door.get("detected") and door.get("state") == "DOORWAY_OBSTACLE_BEYOND":
                    return "Doorway clear, obstacle ahead."
                if door.get("detected") and door.get("state") == "PARTIALLY_OPEN":
                    return "Door partially open. Proceed with caution."
                corridor = indoor_safety.get("corridor_analysis", {})
                if corridor.get("clearance_state") == "NARROW":
                    return "Narrow passage. Slow down."
                glass = indoor_safety.get("glass_obstacles", {})
                if glass.get("detected"):
                    return glass.get("voice_message") or "Caution. Possible glass obstacle ahead."

            # Phase 10 Dynamic Scene checks:
            # 1. Crowded corridor
            if multi_object and isinstance(multi_object, dict):
                if multi_object.get("is_crowded") or multi_object.get("crowd_size") in ("GROUP", "DENSE_CROWD"):
                    return "Slow down. Path ahead is crowded."

            # 2. Approaching or crossing person in center
            if primary_obstacle and isinstance(primary_obstacle, dict):
                p_cls = str(primary_obstacle.get("class_name", "")).lower()
                p_pos = str(primary_obstacle.get("position", "")).upper()
                p_state = str(primary_obstacle.get("person_state", "")).upper()
                if "person" in p_cls or p_state in ("PERSON_APPROACHING", "PERSON_CROSSING", "PERSON_STATIC"):
                    if p_pos == "CENTER" or p_state in ("PERSON_APPROACHING", "PERSON_CROSSING"):
                        return "Slow down. Person ahead."

            if predictive_threat and isinstance(predictive_threat, dict):
                pt_cls = str(predictive_threat.get("class_name", "")).lower()
                if "person" in pt_cls or predictive_threat.get("is_approaching"):
                    return "Slow down. Person ahead."

            if primary_obstacle and isinstance(primary_obstacle, dict):
                obs_name = self._format_object_name(primary_obstacle.get("class_name", ""))
                pos_phr = self._position_phrase(primary_obstacle.get("position", "CENTER"))
                return f"Slow down. {obs_name} {pos_phr}."

            return "Slow down."

        if nav in ("BACK", "GO_BACK"):
            if steps and steps >= 1:
                return f"Move back for {steps} steps."
            return "Move back for 3 steps."

        if nav in ("FORWARD", "GO_FORWARD", "CONTINUE_FORWARD"):
            # Check if path recently cleared after being blocked
            if getattr(self, "path_cleared_from_stop", False) or getattr(self, "closed_loop_state", "") == "PATH_CLEARED" or getattr(self, "last_nav_state", "") in ("STOP", "WAIT_FOR_CHANGE"):
                if steps and steps >= 1:
                    return f"Path clear. Move forward for {steps} steps."
                return "Path clear. Move forward for 4 steps."

            # Check for semantic landmark announcement (Phase 3 & 9)
            if primary_semantic_sign and primary_semantic_sign.get("should_announce") and not primary_semantic_sign.get("is_uncertain"):
                sem_type = str(primary_semantic_sign.get("type", "")).upper()
                pos = str(primary_semantic_sign.get("position", "CENTER")).upper()
                pos_phrase = "on your left" if pos == "LEFT" else ("on your right" if pos == "RIGHT" else "ahead")
                cue = str(primary_semantic_sign.get("directional_cue", "NONE")).upper()
                label = primary_semantic_sign.get("label", sem_type.title())

                if sem_type in ("EXIT", "ENTRANCE", "RESTROOM", "STAIRS", "ELEVATOR",
                                "RECEPTION", "ROOM", "OFFICE", "CAFETERIA", "LOBBY",
                                "INFORMATION", "EMERGENCY", "PHARMACY", "CLINIC", "LIBRARY", "SECURITY"):
                    if cue == "RIGHT":
                        return f"{label} to the right."
                    elif cue == "LEFT":
                        return f"{label} to the left."
                    elif cue == "STRAIGHT":
                        return f"{label} straight ahead."
                    elif cue == "UP":
                        return f"{label} upstairs."
                    elif cue == "DOWN":
                        return f"{label} downstairs."

                    pos_phrase = "on your left" if pos == "LEFT" else ("on your right" if pos == "RIGHT" else "ahead")
                    return f"{label} {pos_phrase}."

            # Predictive approaching obstacle warning (Phase 11 / Requirement 8)
            is_app = False
            app_cls = "obstacle"
            if predictive_threat and isinstance(predictive_threat, dict):
                if predictive_threat.get("is_approaching") or str(predictive_threat.get("motion_state", "")).upper() in ("APPROACHING", "RAPIDLY_APPROACHING"):
                    is_app = True
                    app_cls = str(predictive_threat.get("class_name", "")).lower()
            if not is_app and primary_obstacle and isinstance(primary_obstacle, dict):
                if primary_obstacle.get("is_approaching") or str(primary_obstacle.get("motion_state", "")).upper() in ("APPROACHING", "RAPIDLY_APPROACHING"):
                    is_app = True
                    app_cls = str(primary_obstacle.get("class_name", "")).lower()

            if is_app:
                if "person" in app_cls:
                    return "Warning. Person approaching."
                return "Warning. Object approaching."

            # Natural English object and position announcements with walking guidance (Phases 11-12)
            if primary_obstacle and isinstance(primary_obstacle, dict):
                obs_cat = str(primary_obstacle.get("distance_category", "")).upper()
                obs_dist = self._safe_float(primary_obstacle.get("distance_m", 99.0))
                if obs_cat == "UNKNOWN_DISTANCE" or (steps == 0 and obs_dist <= 0):
                    return "Continue forward carefully."
                obs_cls = str(primary_obstacle.get("class_name", "")).strip().lower()
                obs_pos = str(primary_obstacle.get("position", "CENTER")).upper()
                if "unknown" in obs_cls:
                    return f"Unknown obstacle on your {obs_pos.lower()}. Move forward." if obs_pos in ("LEFT", "RIGHT") else "Unknown obstacle ahead. Move forward."
                obs_name = self._format_object_name(obs_cls)
                pos_phr = "on the left" if obs_pos == "LEFT" else ("on the right" if obs_pos == "RIGHT" else "ahead")
                obs_dist = self._safe_float(primary_obstacle.get("distance_m", 99.0))
                if obs_dist <= 1.2:
                    return f"{obs_name} close {pos_phr}. Proceed carefully."
                return f"{obs_name} {pos_phr}. Move forward."

            if objects and isinstance(objects, list) and len(objects) > 0:
                valid_objs = [o for o in objects if isinstance(o, dict) and o.get("class_name")]
                if valid_objs:
                    sorted_objs = sorted(valid_objs, key=lambda o: self._safe_float(o.get("distance_m", o.get("depth", 99.0))))
                    closest_obj = sorted_objs[0]
                    cls_name = self._format_object_name(closest_obj.get("class_name", ""))
                    obj_pos = str(closest_obj.get("position", "CENTER")).upper()
                    pos_phr = "on the left" if obj_pos == "LEFT" else ("on the right" if obj_pos == "RIGHT" else "ahead")
                    return f"{cls_name} {pos_phr}. Move forward."

            if steps and steps >= 1:
                return f"Move forward {_step_word(steps)}."
            return "Move forward."

        # Safety-level fallback
        if safety_level == "DANGER":
            return "Slow down."

        return "Move forward."

    # ========================================================
    # OBJECT & POSITION PHRASING
    # ========================================================

    def _format_object_name(
        self,
        class_name: Optional[str],
    ) -> str:
        """
        Format technical class labels into clear, natural English object names.
        """
        if not class_name:
            return "Obstacle"
        cls = str(class_name).strip().lower()
        if "unknown" in cls:
            return "Unknown obstacle"
        mapping = {
            "person": "Person",
            "chair": "Chair",
            "diningtable": "Table",
            "dining table": "Table",
            "table": "Table",
            "bottle": "Bottle",
            "door": "Door",
            "stair": "Stairs",
            "stairs": "Stairs",
            "couch": "Sofa",
            "sofa": "Sofa",
            "bed": "Bed",
            "tv": "Screen",
            "tvmonitor": "Screen",
            "laptop": "Laptop",
            "cell phone": "Phone",
            "backpack": "Backpack",
            "handbag": "Bag",
            "suitcase": "Suitcase",
            "cup": "Cup",
            "fork": "Fork",
            "knife": "Knife",
            "spoon": "Spoon",
            "bowl": "Bowl",
            "potted plant": "Plant",
            "plant": "Plant",
            "obstacle": "Obstacle",
        }
        if cls in mapping:
            return mapping[cls]
        return cls.replace("_", " ").title()

    @staticmethod
    def _position_phrase(
        position: str,
    ) -> str:
        """
        Convert position to natural voice wording.
        """
        pos = str(position).upper()
        if pos == "LEFT":
            return "on the left"
        if pos == "RIGHT":
            return "on the right"
        return "ahead"

    # ========================================================
    # MULTI-OBJECT REASONING
    # ========================================================

    def multi_object_reasoning(
        self,
        objects: List[
            Dict[str, Any]
        ],
        unknown_objects: Optional[
            List[Dict[str, Any]]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Analyze multiple objects.

        Produces:

            object_count
            important_objects
            center_obstacles
            human_detected
            vehicle_detected
        """

        all_items: List[Dict[str, Any]] = list(objects or [])
        if unknown_objects:
            all_items.extend(unknown_objects)

        if not all_items:

            return {
                "object_count": 0,
                "left_obstacles": 0,
                "center_obstacles": 0,
                "right_obstacles": 0,
                "human_detected": False,
                "vehicle_detected": False,
                "highest_risk_object_id": None,
                "closest_object_id": None,
                "people_count": 0,
                "crowd_size": "NONE",
                "crowd_distribution": "NONE",
                "crowd_center": False,
                "crowd_left": False,
                "crowd_right": False,
                "crowd_both_sides": False,
                "is_crowded": False,
                "crowd_blocking": False,
                "person_states": [],
            }

        sorted_objects = sorted(
            all_items,
            key=lambda item:
                self._safe_float(
                    item.get(
                        "risk_score",
                        0.0,
                    )
                ),
            reverse=True,
        )

        left_obstacles = sum(
            1 for obj in all_items
            if str(obj.get("position", "")).upper() == "LEFT"
        )

        center_obstacles = sum(
            1 for obj in all_items
            if str(obj.get("position", "")).upper() == "CENTER"
        )

        right_obstacles = sum(
            1 for obj in all_items
            if str(obj.get("position", "")).upper() == "RIGHT"
        )

        highest_risk_id = None
        if sorted_objects:
            highest_risk_id = sorted_objects[0].get("id")

        closest_id = None
        # Select closest obstacle using canonical distance_m (filter for valid positive distances)
        valid_dist_items = [
            item for item in all_items
            if self._safe_float(item.get("distance_m", 0.0)) > 0.1
            and item.get("distance_category") != "UNKNOWN_DISTANCE"
        ]
        if valid_dist_items:
            closest_obj = min(
                valid_dist_items,
                key=lambda item: self._safe_float(item.get("distance_m", 999.0))
            )
            closest_id = closest_obj.get("id")
        elif all_items:
            closest_id = all_items[0].get("id")

        class_names = {
            str(
                obj.get(
                    "class_name",
                    "",
                )
            ).lower()
            for obj in (objects or [])
        }

        # Phase 10 Crowd & Dynamic Scene Reasoning
        persons = [
            obj for obj in all_items
            if str(obj.get("class_name", "")).lower().strip() in self.HUMAN_CLASSES
            or "person" in str(obj.get("class_name", "")).lower()
        ]
        people_count = len(persons)
        left_people = sum(1 for p in persons if str(p.get("position", "")).upper() == "LEFT")
        center_people = sum(1 for p in persons if str(p.get("position", "")).upper() == "CENTER")
        right_people = sum(1 for p in persons if str(p.get("position", "")).upper() == "RIGHT")

        if people_count == 0:
            crowd_size = "NONE"
        elif people_count == 1:
            crowd_size = "SINGLE"
        elif people_count == 2:
            crowd_size = "PAIR"
        elif 3 <= people_count <= 4:
            crowd_size = "GROUP"
        else:
            crowd_size = "DENSE_CROWD"

        if people_count == 0:
            crowd_distribution = "NONE"
        elif center_people > 0 and left_people == 0 and right_people == 0:
            crowd_distribution = "CENTER"
        elif left_people > 0 and center_people == 0 and right_people == 0:
            crowd_distribution = "LEFT"
        elif right_people > 0 and center_people == 0 and left_people == 0:
            crowd_distribution = "RIGHT"
        elif left_people > 0 and right_people > 0 and center_people == 0:
            crowd_distribution = "BOTH_SIDES"
        elif left_people > 0 and right_people > 0 and center_people > 0:
            crowd_distribution = "BOTH_SIDES" if people_count <= 3 else "DISPERSED"
        elif center_people >= 2 and (left_people == 0 or right_people == 0):
            crowd_distribution = "CENTER"
        else:
            crowd_distribution = "DISPERSED"

        crowd_center = bool(center_people > 0)
        crowd_left = bool(left_people > 0)
        crowd_right = bool(right_people > 0)
        crowd_both_sides = bool((left_people > 0 and right_people > 0) or (left_obstacles > 0 and right_obstacles > 0 and people_count >= 2))
        is_crowded = bool(people_count >= 3 or (people_count >= 2 and center_people >= 2))

        # Check if corridor ahead is blocked by crowd or group of people
        crowd_blocking = bool(
            center_people >= 2
            or (people_count >= 2 and center_people >= 1 and any(
                self._safe_float(p.get("distance_m", p.get("estimated_distance_m", 99.0))) <= 2.2
                for p in persons if str(p.get("position", "")).upper() == "CENTER"
            ))
            or (people_count >= 3 and is_crowded)
        )

        # Ensure person_state is populated for all persons
        person_states = []
        for p in persons:
            p_state = p.get("person_state")
            if not p_state:
                mot = str(p.get("motion_state", "")).upper()
                rate = self._safe_float(p.get("approach_rate", p.get("approach_rate_mps", 0.0)))
                if mot in ("APPROACHING", "RAPIDLY_APPROACHING") or rate >= 0.15 or p.get("approaching") or p.get("is_approaching"):
                    p_state = "PERSON_APPROACHING"
                elif mot == "RECEDING" or rate <= -0.15:
                    p_state = "PERSON_MOVING_AWAY"
                elif p.get("is_crossing") or p.get("crossing_direction") in ("LEFT_TO_CENTER", "RIGHT_TO_CENTER", "CROSSING_LEFT", "CROSSING_RIGHT"):
                    p_state = "PERSON_CROSSING"
                elif mot == "STATIONARY":
                    p_state = "PERSON_STATIC"
                elif p.get("confidence", 0.7) < 0.45 or mot == "UNKNOWN_MOTION":
                    p_state = "PERSON_UNCERTAIN"
                else:
                    p_state = "PERSON_STATIC"
                p["person_state"] = p_state
            person_states.append(str(p_state))

        return {
            "object_count":
                len(all_items),

            "left_obstacles":
                left_obstacles,

            "center_obstacles":
                center_obstacles,

            "right_obstacles":
                right_obstacles,

            "human_detected":
                bool(
                    class_names
                    & self.HUMAN_CLASSES
                ),

            "vehicle_detected":
                bool(
                    class_names
                    & self.VEHICLE_CLASSES
                ),

            "highest_risk_object_id":
                highest_risk_id,

            "closest_object_id":
                closest_id,

            # Phase 10 Crowd & Dynamic Scene fields
            "people_count": people_count,
            "crowd_size": crowd_size,
            "crowd_distribution": crowd_distribution,
            "crowd_center": crowd_center,
            "crowd_left": crowd_left,
            "crowd_right": crowd_right,
            "crowd_both_sides": crowd_both_sides,
            "is_crowded": is_crowded,
            "crowd_blocking": crowd_blocking,
            "person_states": person_states,
        }

    # ========================================================
    # STRUCTURED NAVIGATION & STEP BUILDERS
    # ========================================================

    def build_navigation_result(
        self,
        raw_direction: Any,
        safety_level: str = "SAFE",
        emergency_stop: bool = False,
        primary_obstacle: Optional[Dict[str, Any]] = None,
        free_path: Optional[Dict[str, Any]] = None,
        backward_path_clear: bool = False,
        predictive_threat: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Map current navigation decision into the canonical structured navigation contract.
        """
        if isinstance(raw_direction, dict) and "action" in raw_direction:
            return raw_direction

        return self.determine_navigation(
            safety_level=safety_level,
            emergency_stop=emergency_stop,
            primary_obstacle=primary_obstacle,
            free_path=free_path or {},
            backward_path_clear=backward_path_clear,
            predictive_threat=predictive_threat,
        )

    def build_step_guidance(
        self,
        navigation: Dict[str, Any],
        primary_obstacle: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build step guidance contract.
        """
        act = str(navigation.get("action", "")).upper()
        nav_steps = int(navigation.get("movement_steps", navigation.get("steps", 0)))
        dist_m = float(navigation.get("movement_distance_m", navigation.get("distance_m", 0.0)))
        step_conf = str(navigation.get("step_confidence", navigation.get("confidence", "HIGH")))
        step_reason = str(navigation.get("step_reason", navigation.get("reason", "")))

        if act in ("STOP", "EMERGENCY_STOP"):
            sg_action = "STOP"
            sg_steps = 0
            sg_dist = 0.0
        elif act in ("TURN_LEFT", "TURN_RIGHT"):
            sg_action = act
            sg_steps = 0
            sg_dist = 0.0
        elif act == "MOVE_LEFT":
            sg_action = "WALK_LEFT"
            sg_steps = max(1, nav_steps)
            sg_dist = dist_m if dist_m > 0 else 0.5
        elif act == "MOVE_RIGHT":
            sg_action = "WALK_RIGHT"
            sg_steps = max(1, nav_steps)
            sg_dist = dist_m if dist_m > 0 else 0.5
        elif act in ("FORWARD", "GO_FORWARD", "CONTINUE_FORWARD"):
            sg_action = "WALK_FORWARD"
            sg_steps = nav_steps
            sg_dist = dist_m
        elif act == "SLOW_DOWN":
            sg_action = "WALK_FORWARD"
            sg_steps = max(1, nav_steps)
            sg_dist = dist_m
        elif act in ("BACK", "GO_BACK"):
            sg_action = "WALK_BACK"
            sg_steps = max(1, nav_steps)
            sg_dist = dist_m if dist_m > 0 else 0.75
        else:
            sg_action = "NONE"
            sg_steps = 0
            sg_dist = 0.0

        if sg_steps == 0:
            step_cat = "0 steps"
        elif sg_steps <= 1:
            step_cat = "0–1 steps"
        elif sg_steps <= 3:
            step_cat = "2–3 steps"
        elif sg_steps <= 6:
            step_cat = "4–6 steps"
        else:
            step_cat = "6+ steps"

        return {
            "action": sg_action,
            "steps": sg_steps,
            "step_category": step_cat,
            "distance_m": round(sg_dist, 2),
            "step_length_m": self.step_length_m,
            "safety_margin_m": self.safety_margin_m,
            "step_confidence": step_conf,
            "confidence": step_conf,
            "reason": step_reason,
        }

    # ========================================================
    # MAIN ANALYZE
    # ========================================================

    def analyze(
        self,
        objects: Optional[
            List[Dict[str, Any]]
        ] = None,
        scene_depth: Optional[
            Dict[str, Any]
        ] = None,
        frame_width: float = 640,
        frame_height: float = 480,
        unknown_objects: Optional[
            List[Dict[str, Any]]
        ] = None,
        indoor_safety: Optional[
            Dict[str, Any]
        ] = None,
        semantic_landmarks: Optional[
            List[Dict[str, Any]]
        ] = None,
        navigation_session: Optional[
            Any
        ] = None,
        perception_health: Optional[
            Dict[str, Any]
        ] = None,
        primary_semantic_sign: Optional[
            Dict[str, Any]
        ] = None,
        emergency_stop: Optional[
            bool
        ] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Main decision-engine function.

        Input:

            objects:
                YOLO detections with depth.

            scene_depth:
                MiDaS scene depth information.

            frame_width:
                Camera frame width.

            frame_height:
                Camera frame height.

            unknown_objects:
                Candidate physical obstacles not classified by YOLO.

            indoor_safety:
                Phase 2 indoor safety analysis bundle (doors, stairs, floor, glass, corridor).

            semantic_landmarks:
                Phase 3 semantic indoor landmarks extracted from OCR.

        Output:

            safety_level
            emergency_stop
            navigation
            step_guidance
            voice_instruction
            primary_obstacle
            objects
            unknown_objects
            free_path
            scene
            multiple_objects
            predictive_threat
            approaching
            approach_rate
            obstacle_stability
            emergency_alert
            navigation_confidence
        """

        if objects is None:
            objects = []

        if scene_depth is None:
            scene_depth = {}

        canonical_semantic_landmarks = list(semantic_landmarks or [])
        if primary_semantic_sign and not canonical_semantic_landmarks:
            canonical_semantic_landmarks = [primary_semantic_sign]
        elif canonical_semantic_landmarks and not primary_semantic_sign:
            primary_semantic_sign = canonical_semantic_landmarks[0]

        # ----------------------------------------------------
        # Normalize detections
        # ----------------------------------------------------

        normalized = self.normalize_objects(
            objects,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        # ----------------------------------------------------
        # Remove duplicates
        # ----------------------------------------------------

        is_ref_uncertain = False
        if perception_health and isinstance(perception_health, dict):
            is_ref_uncertain = bool(perception_health.get("reflection_uncertainty"))

        filtered_objects = (
            self.remove_duplicates(
                normalized,
                reflection_uncertainty=is_ref_uncertain,
            )
        )

        # ----------------------------------------------------
        # Sort by risk
        # ----------------------------------------------------

        filtered_objects.sort(
            key=lambda item:
                self._safe_float(
                    item.get(
                        "risk_score",
                        0.0,
                    )
                ),
            reverse=True,
        )

        # ----------------------------------------------------
        # Unknown obstacles
        # ----------------------------------------------------

        all_unknown_candidates = list(unknown_objects or [])
        if indoor_safety and isinstance(indoor_safety, dict):
            floor_hazards = indoor_safety.get("floor_hazards", [])
            if floor_hazards and isinstance(floor_hazards, list):
                all_unknown_candidates.extend(floor_hazards)

        normalized_unknowns = self.normalize_unknown_objects(
            all_unknown_candidates,
            frame_width=frame_width,
            frame_height=frame_height,
            known_objects=filtered_objects,
        )

        # Depth-based regional obstacle extraction when YOLO detects no objects and no contour candidates were passed (Requirements 2, 3, 4)
        if not filtered_objects and not normalized_unknowns and isinstance(scene_depth, dict):
            fallback_unknowns: List[Dict[str, Any]] = []
            fw = float(frame_width) if frame_width else 640.0
            fh = float(frame_height) if frame_height else 480.0
            c_d = self.normalize_depth(scene_depth.get("center_region_depth", scene_depth.get("center_depth", 1.0)))
            l_d = self.normalize_depth(scene_depth.get("left_depth", 1.0))
            r_d = self.normalize_depth(scene_depth.get("right_depth", 1.0))

            if c_d < 0.20:
                conf = round(min(0.85, max(0.45, 0.50 + 0.40 * (0.50 - c_d))), 4)
                fallback_unknowns.append({
                    "id": 1001,
                    "bbox": {"x1": 0.35 * fw, "y1": 0.25 * fh, "x2": 0.65 * fw, "y2": fh},
                    "cand_depth": c_d,
                    "confidence": conf,
                    "class_name": "unknown",
                })
            if l_d < 0.20:
                conf = round(min(0.85, max(0.45, 0.50 + 0.40 * (0.50 - l_d))), 4)
                fallback_unknowns.append({
                    "id": 1002,
                    "bbox": {"x1": 0.0, "y1": 0.25 * fh, "x2": 0.35 * fw, "y2": fh},
                    "cand_depth": l_d,
                    "confidence": conf,
                    "class_name": "unknown",
                })
            if r_d < 0.20:
                conf = round(min(0.85, max(0.45, 0.50 + 0.40 * (0.50 - r_d))), 4)
                fallback_unknowns.append({
                    "id": 1003,
                    "bbox": {"x1": 0.65 * fw, "y1": 0.25 * fh, "x2": fw, "y2": fh},
                    "cand_depth": r_d,
                    "confidence": conf,
                    "class_name": "unknown",
                })

            if fallback_unknowns:
                normalized_unknowns = self.normalize_unknown_objects(
                    fallback_unknowns,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    known_objects=filtered_objects,
                )

        # ----------------------------------------------------
        # Multi-object temporal tracking and distance smoothing (Phase 5 & 11)
        # ----------------------------------------------------

        self.track_and_smooth_distances(
            objects=filtered_objects,
            unknown_objects=normalized_unknowns,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        # ----------------------------------------------------
        # Sort by risk
        # ----------------------------------------------------

        filtered_objects.sort(
            key=lambda item:
                self._safe_float(
                    item.get(
                        "risk_score",
                        0.0,
                    )
                ),
            reverse=True,
        )

        normalized_unknowns.sort(
            key=lambda item:
                self._safe_float(
                    item.get(
                        "risk_score",
                        0.0,
                    )
                ),
            reverse=True,
        )

        # Canonical contract list for response with Phase 11 fields
        canonical_unknowns = [
            {
                "id": u["id"],
                "class_name": u.get("class_name", "unknown"),
                "position": u["position"],
                "bbox": u["bbox"],
                "center_x": u.get("center_x", 0.0),
                "center_y": u.get("center_y", 0.0),
                "width": u.get("width", 0.0),
                "height": u.get("height", 0.0),
                "distance_m": u["distance_m"],
                "distance_category": u["distance_category"],
                "distance_confidence": u["distance_confidence"],
                "risk_score": u["risk_score"],
                "navigation_relevance": u["navigation_relevance"],
                "confidence": u["confidence"],
                "motion_state": u.get("motion_state", "UNKNOWN_MOTION"),
                "approach_rate_mps": u.get("approach_rate_mps", 0.0),
                "time_to_collision_s": u.get("time_to_collision_s"),
                "approach_score": u.get("approach_score", 0.0),
                "is_approaching": u.get("is_approaching", False),
                "approaching": u.get("approaching", False),
                "approach_rate": u.get("approach_rate", 0.0),
                "person_state": u.get("person_state", "NONE"),
                "is_crossing": u.get("is_crossing", False),
                "crossing_direction": u.get("crossing_direction", "NONE"),
            }
            for u in normalized_unknowns
        ]

        # ----------------------------------------------------
        # Free path (Phase 7 Safe Walking Corridor Detection)
        # ----------------------------------------------------

        free_path = (
            self.analyze_free_path(
                scene_depth=scene_depth,
                objects=filtered_objects,
                unknown_objects=normalized_unknowns,
                frame_width=frame_width,
                frame_height=frame_height,
            )
        )

        # ----------------------------------------------------
        # Primary obstacle
        # ----------------------------------------------------

        all_candidates = list(filtered_objects)
        all_candidates.extend(normalized_unknowns)

        def _candidate_rank(item: Dict[str, Any]) -> float:
            cls = str(item.get("class_name", "")).lower()
            prio = self.get_priority(cls)
            if any(h in cls for h in ("person", "human")):
                prio = 1.0
            elif "unknown" in cls:
                prio = 0.45
            risk = self._safe_float(item.get("risk_score", 0.0))
            dist = self._safe_float(item.get("distance_m", item.get("depth", 99.0)))
            prox = max(0.0, 1.0 - (dist / 5.0)) if dist > 0 else 0.0
            return prio * 0.40 + risk * 0.40 + prox * 0.20

        all_candidates.sort(key=_candidate_rank, reverse=True)
        filtered_objects.sort(key=_candidate_rank, reverse=True)
        normalized_unknowns.sort(key=_candidate_rank, reverse=True)

        primary_obstacle: Optional[
            Dict[str, Any]
        ] = None

        if all_candidates:
            primary_obstacle = (
                all_candidates[0]
            )

        tracking_info = self.update_tracking_and_approach(
            primary_obstacle
        )

        # ----------------------------------------------------
        # Multi-object reasoning
        # ----------------------------------------------------

        multi_object = (
            self.multi_object_reasoning(
                filtered_objects,
                unknown_objects=normalized_unknowns,
            )
        )

        # ----------------------------------------------------
        # Predictive threat assessment (Phase 11)
        # ----------------------------------------------------

        predictive_threat = self.compute_predictive_threat(
            objects=filtered_objects,
            unknown_objects=normalized_unknowns,
            primary_obstacle=primary_obstacle,
        )

        # ----------------------------------------------------
        # Emergency logic
        # ----------------------------------------------------

        calc_emergency_stop = (
            self.should_emergency_stop(
                primary_obstacle,
                scene_depth,
                predictive_threat=predictive_threat,
            )
        )
        emergency_stop = bool(emergency_stop or calc_emergency_stop)

        if predictive_threat.get("threat_level") == "EMERGENCY":
            emergency_stop = True

        # Critical indoor fall hazard override
        if indoor_safety and isinstance(indoor_safety, dict):
            stair_info = indoor_safety.get("stair_analysis", {})
            if stair_info.get("detected") and stair_info.get("hazard_level") == "CRITICAL":
                emergency_stop = True

        # Stairs hazard reinforcement with semantic STAIRS sign (Phase 3)
        if canonical_semantic_landmarks and indoor_safety and isinstance(indoor_safety, dict):
            has_stairs_sign = any(s.get("type") == "STAIRS" for s in canonical_semantic_landmarks)
            if has_stairs_sign:
                stair_info = indoor_safety.get("stair_analysis", {})
                if stair_info.get("detected") or stair_info.get("hazard_level") in ("CRITICAL", "CAUTION"):
                    stair_info["confidence"] = min(1.0, float(stair_info.get("confidence", 0.5)) + 0.35)
                    stair_info["hazard_level"] = "CRITICAL"
                    emergency_stop = True

        # Perception Health Emergency Override (Phase 5)
        if perception_health and isinstance(perception_health, dict):
            if perception_health.get("status") == "FAILED" or perception_health.get("image_quality", {}).get("is_obstructed"):
                emergency_stop = True

        # ----------------------------------------------------
        # Primary risk
        # ----------------------------------------------------

        if primary_obstacle:

            primary_risk = (
                self._safe_float(
                    primary_obstacle.get(
                        "risk_score",
                        0.0,
                    )
                )
            )

        else:

            primary_risk = 0.0

        # Multi-object collective center path reasoning:
        # If multiple obstacles occupy the center path, elevate risk so the system
        # does not conclude "center is safe" simply because each obstacle alone has moderate risk.
        center_count = int(multi_object.get("center_obstacles", 0))
        left_count = int(multi_object.get("left_obstacles", 0))
        right_count = int(multi_object.get("right_obstacles", 0))

        if center_count >= 2:
            collective_bonus = min(0.20, (center_count - 1) * 0.10)
            primary_risk = min(0.95, max(primary_risk + collective_bonus, self.aware_threshold + 0.05))
            if primary_risk >= 0.50 and isinstance(free_path, dict):
                free_path["is_center_blocked"] = True
                free_path["center_clear"] = False
                free_path["center_blocked"] = True
                if "CENTER" in free_path.get("safe_directions", []):
                    free_path["safe_directions"].remove("CENTER")
                free_path["no_safe_path"] = bool(len(free_path.get("safe_directions", [])) == 0)
                if free_path.get("best_direction") == "CENTER":
                    rem = free_path.get("safe_directions", [])
                    if "RIGHT" in rem and "LEFT" in rem:
                        free_path["best_direction"] = "RIGHT" if free_path.get("right_risk", 1.0) <= free_path.get("left_risk", 1.0) else "LEFT"
                    elif rem:
                        free_path["best_direction"] = rem[0]
                    else:
                        free_path["best_direction"] = "RIGHT" if free_path.get("right_risk", 1.0) <= free_path.get("left_risk", 1.0) else "LEFT"

        # Phase 10 Crowd Density & Dynamic Obstacle Risk Escalation
        is_crowd = bool(multi_object.get("is_crowded", False) or multi_object.get("crowd_size") in ("GROUP", "DENSE_CROWD"))
        crowd_block = bool(multi_object.get("crowd_blocking", False))
        if crowd_block or (is_crowd and center_count >= 1):
            collective_bonus = 0.25 if crowd_block else 0.15
            primary_risk = min(0.95, max(primary_risk + collective_bonus, 0.55 if not crowd_block else 0.70))
            if isinstance(free_path, dict):
                free_path["is_center_blocked"] = True
                free_path["center_clear"] = False
                free_path["center_blocked"] = True
                if "CENTER" in free_path.get("safe_directions", []):
                    free_path["safe_directions"].remove("CENTER")
                free_path["no_safe_path"] = bool(len(free_path.get("safe_directions", [])) == 0)
                if free_path.get("best_direction") == "CENTER":
                    rem = free_path.get("safe_directions", [])
                    if "RIGHT" in rem and "LEFT" in rem:
                        free_path["best_direction"] = "RIGHT" if free_path.get("right_risk", 1.0) <= free_path.get("left_risk", 1.0) else "LEFT"
                    elif rem:
                        free_path["best_direction"] = rem[0]
                    else:
                        free_path["best_direction"] = "RIGHT" if free_path.get("right_risk", 1.0) <= free_path.get("left_risk", 1.0) else "LEFT"

        # Check crossing persons intersecting center corridor
        has_crossing_to_center = any(
            bool(o.get("is_crossing")) and str(o.get("crossing_direction", "")).upper() in ("LEFT_TO_CENTER", "RIGHT_TO_CENTER", "CROSSING_LEFT", "CROSSING_RIGHT")
            for o in (filtered_objects + normalized_unknowns)
        )
        if has_crossing_to_center:
            primary_risk = max(primary_risk, 0.55)
            if isinstance(free_path, dict):
                free_path["center_risk"] = max(float(free_path.get("center_risk", 0.0)), 0.55)
                free_path["path_confidence"] = "MEDIUM"

        # Multi-obstacle all-corridor reasoning:
        # Only declare all paths blocked if BOTH left and right are genuinely high risk (>= 0.65)
        if center_count >= 1 and left_count >= 1 and right_count >= 1:
            if isinstance(free_path, dict):
                l_risk = float(free_path.get("left_risk", 0.0))
                r_risk = float(free_path.get("right_risk", 0.0))
                if l_risk >= 0.65 and r_risk >= 0.65:
                    free_path["left_clear"] = False
                    free_path["right_clear"] = False
                    free_path["center_clear"] = False
                    free_path["safe_directions"] = []
                    free_path["no_safe_path"] = True
                elif l_risk < r_risk:
                    free_path["left_clear"] = True
                    free_path["best_direction"] = "LEFT"
                    free_path["safe_directions"] = ["LEFT"]
                    free_path["no_safe_path"] = False
                else:
                    free_path["right_clear"] = True
                    free_path["best_direction"] = "RIGHT"
                    free_path["safe_directions"] = ["RIGHT"]
                    free_path["no_safe_path"] = False

        # ----------------------------------------------------
        # Safety
        # ----------------------------------------------------

        safety_level = (
            self.get_safety_level(
                primary_risk,
                emergency_stop,
            )
        )

        # Predictive safety escalation (Phase 11)
        pt_pos = str(predictive_threat.get("position", "")).upper()
        pt_mot = str(predictive_threat.get("motion_state", "")).upper()
        pt_dist = self._safe_float(predictive_threat.get("distance_m", 99.0), 99.0)
        pt_threat = str(predictive_threat.get("threat_level", "NONE")).upper()

        if not emergency_stop:
            if pt_threat == "EMERGENCY":
                emergency_stop = True
                safety_level = "EMERGENCY"
            elif pt_threat == "DANGER" or (pt_pos == "CENTER" and pt_mot == "RAPIDLY_APPROACHING" and pt_dist <= 2.5):
                if safety_level in ("SAFE", "AWARE", "CAUTION"):
                    safety_level = "DANGER"
            elif pt_threat == "CAUTION" or (pt_pos == "CENTER" and pt_mot in ("APPROACHING", "RAPIDLY_APPROACHING") and pt_dist <= 2.2):
                if safety_level in ("SAFE", "AWARE"):
                    safety_level = "CAUTION"

        # ----------------------------------------------------
        # Navigation
        # ----------------------------------------------------

        navigation = (
            self.determine_navigation(
                safety_level=safety_level,
                emergency_stop=emergency_stop,
                primary_obstacle=primary_obstacle,
                free_path=free_path,
                predictive_threat=predictive_threat,
            )
        )

        # ----------------------------------------------------
        # Indoor Safety Navigation & Clearance Adjustments
        # Indoor Safety Navigation & Clearance Adjustments (Phase 8)
        # ----------------------------------------------------
        if indoor_safety and isinstance(indoor_safety, dict):
            stair = indoor_safety.get("stair_analysis", {})
            corridor = indoor_safety.get("corridor_analysis", {})
            glass = indoor_safety.get("glass_obstacles", {})
            door = indoor_safety.get("door_analysis", {})
            floors = indoor_safety.get("floor_hazards", [])

            critical_floors = [f for f in floors if isinstance(f, dict) and f.get("hazard_level") == "CRITICAL"]
            caution_floors = [f for f in floors if isinstance(f, dict) and f.get("hazard_level") == "CAUTION"]

            # Priority 1: Critical drop-offs and floor depressions (Fall hazard)
            if stair.get("detected") and stair.get("hazard_level") == "CRITICAL":
                emergency_stop = True
                safety_level = "EMERGENCY"
                stype = stair.get("type", "STAIRS")
                msg = "Stop. Drop-off ahead." if stype == "DROP_OFF" else "Stop. Stairs detected ahead."
                navigation = {
                    "action": "STOP",
                    "direction": "CENTER",
                    "reason": stair.get("message", "Critical drop-off or stairs down ahead"),
                    "reason": stair.get("message", msg),
                    "movement_steps": 0,
                    "confidence": "HIGH",
                }
            elif critical_floors:
                emergency_stop = True
                safety_level = "EMERGENCY"
                navigation = {
                    "action": "STOP",
                    "direction": "CENTER",
                    "reason": "Hole or depression on the floor ahead. Stop.",
                    "movement_steps": 0,
                    "confidence": "HIGH",
                }
            # Priority 2: Blocked corridor path
            elif corridor.get("clearance_state") == "BLOCKED":
                if not emergency_stop:
                    free_dir = None
                    if free_path and isinstance(free_path, dict):
                        d = str(free_path.get("best_direction", "")).upper()
                        if d in ("LEFT", "RIGHT") and (free_path.get("left_clear") or free_path.get("right_clear")):
                            free_dir = d
                        elif free_path.get("left_clear"):
                            free_dir = "LEFT"
                        elif free_path.get("right_clear"):
                            free_dir = "RIGHT"

                    if free_dir:
                        navigation = {
                            "action": f"MOVE_{free_dir}",
                            "direction": free_dir,
                            "reason": f"Center path blocked. Move {free_dir.lower()} to bypass.",
                            "movement_steps": 1,
                            "confidence": "MEDIUM",
                        }
                    else:
                        safety_level = "DANGER" if safety_level != "EMERGENCY" else safety_level
                        navigation = {
                            "action": "STOP",
                            "direction": "CENTER",
                            "reason": "Insufficient shoulder clearance: path blocked",
                            "movement_steps": 0,
                            "confidence": "HIGH",
                        }
            # Priority 3: Doorway blocked or closed
            elif door.get("detected") and door.get("state") == "CLOSED":
                if not emergency_stop:
                    safety_level = "CAUTION" if safety_level == "SAFE" else safety_level
                    navigation = {
                        "action": "STOP",
                        "direction": "CENTER",
                        "reason": "Closed door ahead: do not enter",
                        "movement_steps": 0,
                        "confidence": "HIGH",
                    }
            elif door.get("detected") and door.get("state") == "BLOCKED":
                if not emergency_stop:
                    # Check if safe lateral detour is available
                    free_dir = None
                    if free_path and isinstance(free_path, dict):
                        d = str(free_path.get("best_direction", free_path.get("direction", ""))).upper()
                        if d == "LEFT" and free_path.get("left_clear"):
                            free_dir = "LEFT"
                        elif d == "RIGHT" and free_path.get("right_clear"):
                            free_dir = "RIGHT"
                        elif free_path.get("left_clear"):
                            free_dir = "LEFT"
                        elif free_path.get("right_clear"):
                            free_dir = "RIGHT"

                    if free_dir:
                        navigation = {
                            "action": f"MOVE_{free_dir}",
                            "direction": free_dir,
                            "reason": f"Doorway blocked. Move {free_dir.lower()} to bypass.",
                            "movement_steps": 1,
                            "confidence": "MEDIUM",
                        }
                    else:
                        safety_level = "DANGER" if safety_level != "EMERGENCY" else safety_level
                        navigation = {
                            "action": "STOP",
                            "direction": "CENTER",
                            "reason": door.get("message", "Stop. The doorway ahead is blocked."),
                            "movement_steps": 0,
                            "confidence": "HIGH",
                        }
            # Priority 4: Caution hazards (slow down)
            elif stair.get("detected") and stair.get("hazard_level") == "CAUTION":
                if not emergency_stop:
                    safety_level = "CAUTION" if safety_level == "SAFE" else safety_level
                    navigation = {
                        "action": "SLOW_DOWN",
                        "direction": "CENTER",
                        "reason": stair.get("message", "Slow down. Floor level change ahead."),
                        "movement_steps": 1,
                        "confidence": "HIGH",
                    }
            elif door.get("detected") and door.get("state") == "DOORWAY_OBSTACLE_BEYOND":
                if not emergency_stop:
                    safety_level = "CAUTION" if safety_level == "SAFE" else safety_level
                    navigation = {
                        "action": "SLOW_DOWN",
                        "direction": "CENTER",
                        "reason": "Doorway clear, obstacle ahead.",
                        "movement_steps": 1,
                        "confidence": "MEDIUM",
                    }
            elif door.get("detected") and door.get("state") == "PARTIALLY_OPEN":
                if not emergency_stop:
                    safety_level = "CAUTION" if safety_level == "SAFE" else safety_level
                    navigation = {
                        "action": "SLOW_DOWN",
                        "direction": "CENTER",
                        "reason": "Door partially open. Proceed with caution.",
                        "movement_steps": 1,
                        "confidence": "MEDIUM",
                    }
            elif corridor.get("clearance_state") == "NARROW":
                if not emergency_stop:
                    safety_level = "CAUTION" if safety_level == "SAFE" else safety_level
                    navigation = {
                        "action": "SLOW_DOWN",
                        "direction": "CENTER",
                        "reason": "Narrow passage: maintain center clearance",
                        "movement_steps": 1,
                        "confidence": "HIGH",
                    }
            elif glass.get("detected") and (glass.get("risk_score", 0.0) >= 0.60 or glass.get("is_uncertain")):
                if not emergency_stop:
                    safety_level = "CAUTION" if safety_level == "SAFE" else safety_level
                    navigation = {
                        "action": "SLOW_DOWN",
                        "direction": glass.get("position", "CENTER"),
                        "reason": glass.get("voice_message") or glass.get("reason", "Possible glass obstacle ahead"),
                        "movement_steps": 1,
                        "confidence": "HIGH",
                    }
            elif caution_floors:
                if not emergency_stop:
                    safety_level = "CAUTION" if safety_level == "SAFE" else safety_level
                    cf = caution_floors[0]
                    navigation = {
                        "action": "SLOW_DOWN",
                        "direction": cf.get("position", "CENTER"),
                        "reason": cf.get("reason", "Uneven surface ahead."),
                        "movement_steps": 1,
                        "confidence": "HIGH",
                    }

        # Check for WARNING signs in center (e.g. DO NOT ENTER / NO ENTRY)
        if canonical_semantic_landmarks:
            for s in canonical_semantic_landmarks:
                if s.get("type") == "WARNING" and s.get("position") in ("CENTER", "NONE"):
                    curr_act = navigation.get("action", navigation) if isinstance(navigation, dict) else navigation
                    if curr_act in ("FORWARD", "GO_FORWARD", "CONTINUE_FORWARD", "SLOW_DOWN"):
                        safety_level = "CAUTION" if safety_level == "SAFE" else safety_level
                        navigation = {
                            "action": "STOP",
                            "direction": "CENTER",
                            "reason": f"Warning sign: {s.get('clean_text', 'Do not enter')}",
                            "movement_steps": 0,
                            "confidence": "HIGH",
                        }
                    break

        # Crowd corridor blockage adjustment (Phase 10)
        if multi_object and isinstance(multi_object, dict):
            if multi_object.get("crowd_blocking"):
                curr_act = navigation.get("action", navigation) if isinstance(navigation, dict) else navigation
                if curr_act != "STOP":
                    safety_level = "DANGER" if safety_level != "EMERGENCY" else safety_level
                    navigation = {
                        "action": "STOP",
                        "direction": "CENTER",
                        "reason": "Corridor blocked by crowd ahead.",
                        "movement_steps": 0,
                        "confidence": "HIGH",
                    }

        # ----------------------------------------------------
        # Phase 5 Real-World Robustness & Perception Health Navigation Adjustments
        # ----------------------------------------------------
        if perception_health and isinstance(perception_health, dict):
            p_stat = perception_health.get("status", "HEALTHY")
            p_img = perception_health.get("image_quality", {})
            if p_stat == "FAILED" or p_img.get("is_obstructed"):
                emergency_stop = True
                safety_level = "EMERGENCY"
                navigation = {
                    "action": "STOP",
                    "direction": "CENTER",
                    "reason": perception_health.get("message", "Camera obstructed. Please check lens."),
                    "movement_steps": 0,
                    "confidence": "HIGH",
                }
            elif perception_health.get("stale_frame"):
                if safety_level == "SAFE":
                    safety_level = "CAUTION"
            elif p_stat == "UNCERTAIN" or perception_health.get("reflection_uncertainty"):
                if safety_level == "SAFE":
                    safety_level = "CAUTION"
            elif perception_health.get("scene_flux"):
                pass  # Intentional navigation turns/moves preserved during camera motion
            elif p_stat == "DEGRADED":
                curr_act = navigation.get("action", navigation) if isinstance(navigation, dict) else navigation
                if curr_act in ("FORWARD", "GO_FORWARD", "CONTINUE_FORWARD"):
                    safety_level = "CAUTION" if safety_level == "SAFE" else safety_level
                    navigation = {
                        "action": "SLOW_DOWN",
                        "direction": "CENTER",
                        "reason": perception_health.get("message", "Degraded perception: proceed with caution"),
                        "movement_steps": 1,
                        "confidence": "HIGH",
                    }

        # ----------------------------------------------------
        # Voice
        # ----------------------------------------------------

        voice_instruction = (
            self.generate_voice_instruction(
                safety_level=safety_level,
                emergency_stop=emergency_stop,
                primary_obstacle=primary_obstacle,
                navigation=navigation.get("action", navigation) if isinstance(navigation, dict) else navigation,
                objects=filtered_objects,
                steps=navigation.get("movement_steps", navigation.get("steps", 0)) if isinstance(navigation, dict) else 0,
                predictive_threat=predictive_threat,
                indoor_safety=indoor_safety,
                primary_semantic_sign=primary_semantic_sign,
                multi_object=multi_object,
                turn_angle_deg=navigation.get("turn_angle_deg") if isinstance(navigation, dict) else None,
            )
        )

        voice_source = "NAVIGATION"
        if emergency_stop or safety_level == "EMERGENCY":
            voice_source = "SAFETY"
        elif primary_semantic_sign and primary_semantic_sign.get("should_announce"):
            label = primary_semantic_sign.get("label", "")
            if label and label in voice_instruction:
                voice_source = "SEMANTIC"

        # Perception Health Voice Guidance Overrides (Phase 5)
        if perception_health and isinstance(perception_health, dict):
            p_stat = perception_health.get("status", "HEALTHY")
            p_voice = perception_health.get("voice_override")
            if p_stat == "FAILED" or perception_health.get("image_quality", {}).get("is_obstructed"):
                voice_instruction = p_voice or "Camera obstructed. Please check lens."
                voice_source = "SAFETY"
            elif perception_health.get("stale_frame"):
                pass
            elif p_stat == "UNCERTAIN" or perception_health.get("reflection_uncertainty"):
                # Critical indoor drop-off / stairs hazard takes precedence over general reflection uncertainty
                # Critical indoor drop-off / stairs / floor hazard takes precedence over general reflection uncertainty
                has_critical_stair = False
                if indoor_safety and isinstance(indoor_safety, dict):
                    stair_data = indoor_safety.get("stair_analysis", {})
                    if stair_data.get("detected") and stair_data.get("hazard_level") == "CRITICAL":
                        has_critical_stair = True
                    if not has_critical_stair:
                        floors_data = indoor_safety.get("floor_hazards", [])
                        if any(f.get("hazard_level") == "CRITICAL" for f in floors_data if isinstance(f, dict)):
                            has_critical_stair = True
                if not has_critical_stair and not emergency_stop:
                    voice_instruction = p_voice or "Vision is uncertain. Please pause."
                    voice_source = "SAFETY"
            elif p_voice:
                curr_act = navigation.get("action", navigation) if isinstance(navigation, dict) else navigation
                if curr_act in ("FORWARD", "GO_FORWARD", "CONTINUE_FORWARD", "SLOW_DOWN") and safety_level not in ("EMERGENCY", "DANGER") and not emergency_stop:
                    voice_instruction = p_voice
                    voice_source = "HEALTH"

        # ----------------------------------------------------
        # Phase 4 Closed-Loop Indoor Destination Navigation
        # ----------------------------------------------------
        session_info: Dict[str, Any] = {"active": False, "state": "IDLE"}
        if navigation_session is not None:
            if isinstance(navigation_session, dict):
                session_info = dict(navigation_session)
            elif hasattr(navigation_session, "update"):
                session_info = navigation_session.update(
                    semantic_landmarks=canonical_semantic_landmarks,
                    scene_depth=scene_depth,
                    primary_obstacle=primary_obstacle,
                    free_path=free_path,
                    safety_level=safety_level,
                    emergency_stop=emergency_stop,
                    indoor_safety=indoor_safety,
                    frame_width=frame_width,
                    frame_height=frame_height,
                ) or {"active": False, "state": "IDLE"}

            if session_info.get("active") and not session_info.get("safety_override"):
                curr_state = session_info.get("state")
                dest_voice = session_info.get("voice_instruction")

                # Safety check: if an emergency stop or critical hazard was flagged, safety overrides!
                is_safe_to_guide = (
                    not emergency_stop
                    and safety_level not in ("EMERGENCY", "DANGER")
                )

                if is_safe_to_guide:
                    if curr_state == "ARRIVED":
                        navigation = {
                            "action": "STOP",
                            "direction": "CENTER",
                            "reason": "Destination reached",
                            "movement_steps": 0,
                            "confidence": "HIGH",
                        }
                        if dest_voice:
                            voice_instruction = dest_voice
                            voice_source = "DESTINATION"

                    elif curr_state == "TURNING":
                        t_act = str(session_info.get("navigation_action", "TURN_RIGHT")).upper()
                        t_dir = "RIGHT" if t_act == "TURN_RIGHT" else "LEFT"
                        fp_dict = free_path if isinstance(free_path, dict) else {}
                        side_blocked = (t_dir == "RIGHT" and fp_dict.get("right_risk", 0.0) >= 0.70) or \
                                       (t_dir == "LEFT" and fp_dict.get("left_risk", 0.0) >= 0.70)
                        if not side_blocked:
                            navigation = {
                                "action": t_act,
                                "direction": t_dir,
                                "reason": f"Turn towards {session_info.get('destination', {}).get('label', 'destination')}",
                                "movement_steps": 0,
                                "confidence": "HIGH",
                            }
                            if dest_voice:
                                voice_instruction = dest_voice
                                voice_source = "DESTINATION"

                    elif curr_state == "VERIFYING":
                        if dest_voice:
                            voice_instruction = dest_voice
                            voice_source = "DESTINATION"

                    elif curr_state in ("SEARCHING", "LOST"):
                        if dest_voice:
                            voice_instruction = dest_voice
                            voice_source = "DESTINATION"

                    elif curr_state == "APPROACHING":
                        curr_nav_action = navigation.get("action", navigation) if isinstance(navigation, dict) else navigation
                        if curr_nav_action in ("FORWARD", "GO_FORWARD", "CONTINUE_FORWARD"):
                            if dest_voice:
                                voice_instruction = dest_voice
                                voice_source = "DESTINATION"

        # ----------------------------------------------------
        # Scene normalization
        # ----------------------------------------------------

        scene = {
            "average_depth":
                round(
                    self.normalize_depth(
                        scene_depth.get(
                            "average_depth",
                            1.0,
                        )
                    ),
                    4,
                ),

            "center_depth":
                round(
                    self.normalize_depth(
                        scene_depth.get(
                            "center_depth",
                            1.0,
                        )
                    ),
                    4,
                ),

            "left_depth":
                round(
                    self.normalize_depth(
                        scene_depth.get(
                            "left_depth",
                            1.0,
                        )
                    ),
                    4,
                ),

            "center_region_depth":
                round(
                    self.normalize_depth(
                        scene_depth.get(
                            "center_region_depth",
                            scene_depth.get(
                                "center_depth",
                                1.0,
                            ),
                        )
                    ),
                    4,
                ),

            "right_depth":
                round(
                    self.normalize_depth(
                        scene_depth.get(
                            "right_depth",
                            1.0,
                        )
                    ),
                    4,
                ),
        }

        # ----------------------------------------------------
        # Final response
        # ----------------------------------------------------

        structured_navigation = self.build_navigation_result(
            raw_direction=navigation,
            safety_level=safety_level,
            emergency_stop=emergency_stop,
            primary_obstacle=primary_obstacle,
            predictive_threat=predictive_threat,
        )

        step_guidance = self.build_step_guidance(
            navigation=structured_navigation,
            primary_obstacle=primary_obstacle,
        )

        return {
            "success": True,
            "safety_level":
                safety_level,

            "emergency_stop":
                bool(
                    emergency_stop
                ),

            "navigation":
                structured_navigation,

            "step_guidance":
                step_guidance,

            "voice_instruction":
                voice_instruction,

            "primary_obstacle":
                primary_obstacle,

            "objects":
                filtered_objects,

            "unknown_objects":
                canonical_unknowns,

            "free_path":
                free_path,

            "scene":
                scene,

            "multiple_objects":
                multi_object,

            "predictive_threat":
                predictive_threat,

            # Compatibility alias
            "multi_object":
                multi_object,

            "approaching":
                bool(
                    predictive_threat.get(
                        "is_approaching",
                        tracking_info.get(
                            "approaching",
                            False,
                        ),
                    )
                ),

            "approach_rate":
                float(
                    predictive_threat.get(
                        "approach_rate_mps",
                        tracking_info.get(
                            "approach_rate",
                            0.0,
                        ),
                    )
                ),

            "obstacle_stability":
                int(
                    tracking_info.get(
                        "obstacle_stability",
                        0,
                    )
                ),

            "emergency_alert":
                bool(
                    emergency_stop
                    or safety_level == "EMERGENCY"
                ),

            "navigation_confidence":
                0.92 if safety_level != "UNKNOWN" else 0.50,

            "nav_state": self.nav_state,
            "state_changed": self.state_changed,
            "should_speak": self.should_speak,

            # Indoor Safety Perception Contracts
            "door_analysis": indoor_safety.get("door_analysis", {}) if indoor_safety else {},
            "stair_analysis": indoor_safety.get("stair_analysis", {}) if indoor_safety else {},
            "floor_hazards": indoor_safety.get("floor_hazards", []) if indoor_safety else [],
            "glass_obstacles": indoor_safety.get("glass_obstacles", {}) if indoor_safety else {},
            "corridor_analysis": indoor_safety.get("corridor_analysis", {}) if indoor_safety else {},

            # Phase 3 Semantic Indoor Understanding Contracts
            "semantic_landmarks": canonical_semantic_landmarks,
            "primary_semantic_sign": primary_semantic_sign,
            "voice_instruction_source": voice_source,

            # Phase 4 Closed-Loop Indoor Destination Navigation Contracts
            "navigation_session": session_info,

            # Phase 5 Real-World Robustness & Perception Health Contracts
            "perception_health": perception_health or {
                "status": "HEALTHY",
                "score": 0.95,
                "message": "Vision system healthy.",
                "image_quality": {"status": "GOOD", "is_obstructed": False},
                "model_health": {"status": "HEALTHY", "yolo_valid": True, "depth_valid": True},
                "reflection_uncertainty": False,
                "scene_flux": False,
                "is_crowded": False,
                "stale_frame": False,
            },
        }


# ============================================================
# EXPORT
# ============================================================

__all__ = [
    "DecisionEngine",
]