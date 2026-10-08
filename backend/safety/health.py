"""
============================================================
SmartVisionAI - Perception Health & Robustness Layer
============================================================
Implements real-world fail-safe operation:
- Fast image quality evaluation (brightness, blur, glare, obstruction)
- Model health and exception containment (YOLO, MiDaS)
- Mirror, glass, and reflection contradiction uncertainty
- Stale frame detection and safety timeouts
- Sudden scene flux shock absorber
- Crowd density & anti-oscillation lateral stabilization
============================================================
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class PerceptionHealthStatus:
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNCERTAIN = "UNCERTAIN"
    FAILED = "FAILED"


# ============================================================
# IMAGE QUALITY EVALUATOR
# ============================================================

class ImageQualityEvaluator:
    """
    Evaluates frame visual fidelity in < 1ms using downscaled OpenCV operations.
    Detects low light, excessive glare, severe motion blur, and lens obstruction.
    """

    def __init__(
        self,
        dark_threshold: float = 28.0,
        severe_dark_threshold: float = 12.0,
        glare_threshold: float = 235.0,
        low_contrast_threshold: float = 12.0,
        blur_threshold: float = 18.0,
    ) -> None:
        self.dark_threshold = dark_threshold
        self.severe_dark_threshold = severe_dark_threshold
        self.glare_threshold = glare_threshold
        self.low_contrast_threshold = low_contrast_threshold
        self.blur_threshold = blur_threshold

    def evaluate(self, frame: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Evaluate frame image quality metrics.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return {
                "status": "POOR",
                "brightness": 0.0,
                "contrast": 0.0,
                "blur_score": 0.0,
                "is_low_light": True,
                "is_severely_dark": True,
                "is_overexposed": False,
                "is_low_contrast": True,
                "is_blurred": True,
                "is_obstructed": True,
                "message": "Empty or missing frame.",
            }

        h, w = frame.shape[:2]
        if h < 16 or w < 16:
            return {
                "status": "POOR",
                "brightness": 0.0,
                "contrast": 0.0,
                "blur_score": 0.0,
                "is_low_light": True,
                "is_severely_dark": True,
                "is_overexposed": False,
                "is_low_contrast": True,
                "is_blurred": True,
                "is_obstructed": True,
                "message": "Invalid frame resolution.",
            }

        # Downscale to 320x240 for ultra-fast diagnostics (< 1ms)
        small = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small

        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        # Variance of Laplacian measures high-frequency edge sharpness
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = float(laplacian.var())

        # Boolean diagnostic flags
        is_severely_dark = bool(brightness < self.severe_dark_threshold)
        is_low_light = bool(brightness < self.dark_threshold)
        is_overexposed = bool(brightness > self.glare_threshold)
        is_low_contrast = bool(contrast < self.low_contrast_threshold)
        is_blurred = bool(blur_score < self.blur_threshold)

        # Lens obstruction detection:
        # 1. Pitch black (lens covered by hand, pocket, or cloth): low mean + low contrast
        # 2. Uniform monochrome occlusion (e.g. wall/surface flush against lens): near-zero contrast + near-zero edges
        is_obstructed = bool(
            (brightness < 10.0 and contrast < 6.0)
            or (contrast < 4.0 and blur_score < 5.0)
        )

        # Quality classification
        if is_obstructed:
            status = "OBSTRUCTED"
            msg = "Camera obstructed. Please check lens."
        elif is_severely_dark:
            status = "POOR"
            msg = "Severe low light. Vision severely impaired."
        elif is_overexposed:
            status = "POOR"
            msg = "Excessive glare detected."
        elif is_blurred and is_low_contrast:
            status = "POOR"
            msg = "Image blurred and low contrast."
        elif is_low_light:
            status = "DEGRADED"
            msg = "Low light detected. Please proceed with caution."
        elif is_blurred:
            status = "DEGRADED"
            msg = "Image blurred. Please hold steady."
        elif is_low_contrast:
            status = "DEGRADED"
            msg = "Low image contrast."
        else:
            status = "GOOD"
            msg = "Lighting and image clarity are good."

        return {
            "status": status,
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "blur_score": round(blur_score, 2),
            "is_low_light": is_low_light,
            "is_severely_dark": is_severely_dark,
            "is_overexposed": is_overexposed,
            "is_low_contrast": is_low_contrast,
            "is_blurred": is_blurred,
            "is_obstructed": is_obstructed,
            "message": msg,
        }


# ============================================================
# MODEL HEALTH EVALUATOR
# ============================================================

class ModelHealthEvaluator:
    """
    Monitors YOLO and MiDaS inference status, sanitizes depth outputs,
    and prevents unhandled runtime exceptions from crashing the pipeline.
    """

    @staticmethod
    def evaluate(
        yolo_valid: bool = True,
        yolo_error: Optional[str] = None,
        depth_valid: bool = True,
        depth_error: Optional[str] = None,
        depth_map: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate ML models operational health.
        """
        # Validate depth map contents if depth was reported valid
        sanitized_depth_valid = bool(depth_valid)
        sanitized_depth_error = depth_error

        if sanitized_depth_valid:
            if depth_map is None or not isinstance(depth_map, np.ndarray) or depth_map.size == 0:
                sanitized_depth_valid = False
                sanitized_depth_error = "Depth map is None or empty"
            elif not np.isfinite(depth_map).all():
                sanitized_depth_valid = False
                sanitized_depth_error = "Depth map contains non-finite values (NaN/Inf)"
            elif float(np.var(depth_map)) < 1e-5:
                sanitized_depth_valid = False
                sanitized_depth_error = "Depth map has zero variance (degenerate)"

        if yolo_valid and sanitized_depth_valid:
            status = "HEALTHY"
            msg = "YOLO and MiDaS models operational."
        elif yolo_valid or sanitized_depth_valid:
            status = "DEGRADED"
            if not yolo_valid:
                msg = f"YOLO detection unavailable: {yolo_error or 'Unknown error'}. Depth-only navigation active."
            else:
                msg = f"MiDaS depth estimation unavailable: {sanitized_depth_error or 'Unknown error'}. Bounding-box safety active."
        else:
            status = "FAILED"
            msg = "Both YOLO and MiDaS inference failed."

        return {
            "status": status,
            "yolo_valid": bool(yolo_valid),
            "yolo_error": yolo_error,
            "depth_valid": sanitized_depth_valid,
            "depth_error": sanitized_depth_error,
            "message": msg,
        }


# ============================================================
# SCENE STABILITY & TEMPORAL EVALUATOR
# ============================================================

class SceneStabilityEvaluator:
    """
    Evaluates temporal and physical scene stability:
    - Mirror / reflection contradictions
    - Stale inference / frame delays
    - Sudden scene flux shock absorption
    - Crowd density and anti-oscillation lateral damping
    """

    def __init__(
        self,
        stale_threshold_s: float = 10.0,
        scene_flux_diff_threshold: float = 45.0,
    ) -> None:
        self.stale_threshold_s = stale_threshold_s
        self.scene_flux_diff_threshold = scene_flux_diff_threshold

        self.last_frame_time: Optional[float] = None
        self.previous_gray: Optional[np.ndarray] = None
        self.last_lateral_dir: Optional[str] = None
        self.pending_lateral_dir: Optional[str] = None
        self.pending_dir_count: int = 0

    def check_stale_frame(
        self,
        frame_timestamp: Optional[float] = None,
        now: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """
        Check if the incoming frame or processing interval exceeds staleness threshold.
        Measures server-side inter-frame interval to avoid client-server clock skew.
        """
        current_time = now if now is not None else time.time()

        if self.last_frame_time is not None:
            interval = current_time - self.last_frame_time
            self.last_frame_time = current_time
            if interval > self.stale_threshold_s:
                return True, round(interval, 3)
            return False, round(interval, 3)

        self.last_frame_time = current_time
        return False, 0.0

    def check_scene_flux(self, frame: Optional[np.ndarray]) -> Tuple[bool, float]:
        """
        Detect sudden massive visual flux (e.g. abrupt camera whipping or rapid 180 turn).
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return False, 0.0

        small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small

        if self.previous_gray is None or self.previous_gray.shape != gray.shape:
            self.previous_gray = gray
            return False, 0.0

        diff = float(np.mean(cv2.absdiff(gray, self.previous_gray)))
        self.previous_gray = gray

        is_flux = diff > self.scene_flux_diff_threshold
        return is_flux, round(diff, 2)

    def check_reflection_uncertainty(
        self,
        glass_obstacles: Optional[Dict[str, Any]],
        scene_depth: Optional[Dict[str, Any]],
        objects: Optional[List[Dict[str, Any]]],
    ) -> Tuple[bool, str]:
        """
        Detect contradictory visual signals from mirrors and glass reflections:
        1. High-confidence glass hazard combined with deep open depth.
        2. Duplicate YOLO objects with high IoU but wildly diverging depth.
        """
        # 1. Glass reflection conflict: glass hazard says transparent barrier ahead,
        # but scene depth mistakenly reports open path (> 0.70 depth).
        if glass_obstacles and isinstance(glass_obstacles, dict):
            if glass_obstacles.get("detected") and glass_obstacles.get("risk_score", 0.0) >= 0.70:
                if scene_depth and isinstance(scene_depth, dict):
                    center_d = float(scene_depth.get("center_region_depth", scene_depth.get("center_depth", 1.0)))
                    if center_d > 0.65:
                        return True, "Glass partition reflection contradicts apparent open depth."

        # 2. Duplicate object reflection anomaly in mirrors:
        # Real person + reflection in mirror creates two bounding boxes overlapping horizontally
        # or with high IoU, but one has significantly different depth.
        if objects and len(objects) >= 2:
            for i in range(len(objects)):
                for j in range(i + 1, len(objects)):
                    o1 = objects[i]
                    o2 = objects[j]
                    if o1.get("class_name") == o2.get("class_name"):
                        b1 = o1.get("bbox", {})
                        b2 = o2.get("bbox", {})
                        d1 = o1.get("depth", 1.0)
                        d2 = o2.get("depth", 1.0)
                        if isinstance(b1, dict) and isinstance(b2, dict) and d1 is not None and d2 is not None:
                            # Check horizontal overlap and size similarity
                            w1 = abs(b1.get("x2", 0) - b1.get("x1", 0))
                            w2 = abs(b2.get("x2", 0) - b2.get("x1", 0))
                            if w1 > 0 and w2 > 0 and abs(w1 - w2) / max(w1, w2) < 0.35:
                                # Divergent depth: one appears 2x further (virtual mirror reflection)
                                if abs(float(d1) - float(d2)) >= 0.35:
                                    return True, "Duplicate visual reflection detected (mirror reflection uncertainty)."

        return False, ""

    @staticmethod
    def check_crowd(objects: Optional[List[Dict[str, Any]]]) -> Tuple[bool, int]:
        """
        Evaluate whether the indoor environment is crowded with multiple dynamic humans.
        """
        if not objects:
            return False, 0

        person_count = sum(
            1 for o in objects
            if str(o.get("class_name", "")).lower() == "person"
        )
        return person_count >= 3, person_count

    def stabilize_lateral_direction(
        self,
        candidate_dir: str,
        is_crowded: bool,
    ) -> str:
        """
        Enforce a 2-frame hysteresis on lateral avoidance directions (MOVE_LEFT vs MOVE_RIGHT)
        in crowded indoor scenes to prevent rapid oscillatory fluttering.
        """
        cand = str(candidate_dir).upper()
        if cand not in ("LEFT", "RIGHT", "MOVE_LEFT", "MOVE_RIGHT"):
            self.last_lateral_dir = cand
            self.pending_lateral_dir = None
            self.pending_dir_count = 0
            return candidate_dir

        if not is_crowded:
            self.last_lateral_dir = cand
            self.pending_lateral_dir = None
            self.pending_dir_count = 0
            return candidate_dir

        if self.last_lateral_dir is None:
            self.last_lateral_dir = cand
            self.pending_lateral_dir = None
            self.pending_dir_count = 0
            return candidate_dir

        if cand == self.last_lateral_dir:
            self.pending_lateral_dir = None
            self.pending_dir_count = 0
            return candidate_dir
        else:
            # Direction change requested. Require 2 consecutive frames of the new direction.
            if self.pending_lateral_dir == cand:
                self.pending_dir_count += 1
            else:
                self.pending_lateral_dir = cand
                self.pending_dir_count = 1

            if self.pending_dir_count >= 2:
                self.last_lateral_dir = cand
                self.pending_lateral_dir = None
                self.pending_dir_count = 0
                return candidate_dir
            else:
                # Suppress flutter, return previous stabilized direction
                return self.last_lateral_dir


# ============================================================
# UNIFIED PERCEPTION HEALTH EVALUATOR
# ============================================================

class PerceptionHealthEvaluator:
    """
    Unified manager synthesizing frame quality, model health, reflection uncertainty,
    stale frame protection, and crowd dynamics into actionable health contracts.
    """

    def __init__(self) -> None:
        self.image_evaluator = ImageQualityEvaluator()
        self.model_evaluator = ModelHealthEvaluator()
        self.stability_evaluator = SceneStabilityEvaluator()

        # Cooldown management to prevent speech spam for non-emergency advisories
        self._last_voice_times: Dict[str, float] = {}
        self._voice_cooldown_s = 4.0

    def should_announce(self, key: str, now: Optional[float] = None) -> bool:
        """
        Check if an advisory announcement key can be spoken without spamming.
        """
        current_time = now if now is not None else time.time()
        last = self._last_voice_times.get(key, 0.0)
        if current_time - last >= self._voice_cooldown_s:
            self._last_voice_times[key] = current_time
            return True
        return False

    def evaluate(
        self,
        frame: Optional[np.ndarray],
        yolo_valid: bool = True,
        yolo_error: Optional[str] = None,
        depth_valid: bool = True,
        depth_error: Optional[str] = None,
        depth_map: Optional[np.ndarray] = None,
        objects: Optional[List[Dict[str, Any]]] = None,
        scene_depth: Optional[Dict[str, Any]] = None,
        glass_obstacles: Optional[Dict[str, Any]] = None,
        frame_timestamp: Optional[float] = None,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run complete robustness assessment and generate unified perception health bundle.
        """
        current_time = now if now is not None else time.time()

        # 1. Image quality
        img_health = self.image_evaluator.evaluate(frame)

        # 2. Model health
        model_health = self.model_evaluator.evaluate(
            yolo_valid=yolo_valid,
            yolo_error=yolo_error,
            depth_valid=depth_valid,
            depth_error=depth_error,
            depth_map=depth_map,
        )

        # 3. Scene stability
        stale_frame, stale_age_s = self.stability_evaluator.check_stale_frame(
            frame_timestamp=frame_timestamp,
            now=current_time,
        )

        scene_flux, flux_diff = self.stability_evaluator.check_scene_flux(frame)

        refl_uncertainty, refl_reason = self.stability_evaluator.check_reflection_uncertainty(
            glass_obstacles=glass_obstacles,
            scene_depth=scene_depth,
            objects=objects,
        )

        is_crowded, person_count = self.stability_evaluator.check_crowd(objects)

        # 4. Overall status determination
        safety_override: Optional[str] = None
        voice_override: Optional[str] = None

        if img_health["is_obstructed"]:
            status = PerceptionHealthStatus.FAILED
            score = 0.05
            message = "Camera obstructed. Please check lens."
            safety_override = "STOP"
            voice_override = "Camera obstructed. Please check lens."

        elif model_health["status"] == "FAILED":
            status = PerceptionHealthStatus.FAILED
            score = 0.05
            message = "Perception models failed."
            safety_override = "STOP"
            voice_override = "Vision system unavailable. Please stop."

        elif stale_frame:
            status = PerceptionHealthStatus.DEGRADED
            score = 0.50
            message = f"Inter-frame interval ({stale_age_s:.1f}s)."
            safety_override = None
            voice_override = None

        elif refl_uncertainty:
            status = PerceptionHealthStatus.UNCERTAIN
            score = 0.45
            message = f"Uncertain reflection geometry: {refl_reason}"
            safety_override = "STOP"
            voice_override = "Vision is uncertain. Please pause."

        elif scene_flux:
            status = PerceptionHealthStatus.UNCERTAIN
            score = 0.50
            message = "Sudden scene flux detected."
            safety_override = "WAIT"

        elif img_health["status"] in ("POOR", "DEGRADED") or model_health["status"] == "DEGRADED":
            status = PerceptionHealthStatus.DEGRADED
            score = 0.65
            if img_health["is_severely_dark"]:
                message = "Severe low light detected."
                if self.should_announce("low_light", current_time):
                    voice_override = "Low light detected. Please proceed with caution."
            elif img_health["is_overexposed"]:
                message = "Excessive glare detected."
                if self.should_announce("glare", current_time):
                    voice_override = "Excessive glare detected."
            elif img_health["is_blurred"]:
                message = "Image blurred. Please hold steady."
                if self.should_announce("blurred", current_time):
                    voice_override = "Image blurred. Please hold steady."
            elif img_health["is_low_light"]:
                message = "Low light detected."
                if self.should_announce("low_light", current_time):
                    voice_override = "Low light detected. Please proceed with caution."
            else:
                message = f"Degraded perception: {model_health['message']}"

        else:
            status = PerceptionHealthStatus.HEALTHY
            score = 0.95
            message = "Vision system healthy."
            if is_crowded and self.should_announce("crowd", current_time):
                voice_override = "Crowded area. Moving cautiously."

        return {
            "status": status,
            "score": score,
            "message": message,
            "safety_override": safety_override,
            "voice_override": voice_override,
            "image_quality": img_health,
            "model_health": model_health,
            "reflection_uncertainty": refl_uncertainty,
            "scene_flux": scene_flux,
            "scene_flux_diff": flux_diff,
            "is_crowded": is_crowded,
            "person_count": person_count,
            "stale_frame": stale_frame,
            "stale_age_s": stale_age_s,
        }
