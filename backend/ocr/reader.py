"""
============================================================
SmartVisionAI - Pretrained OCR Module (Phase 2)
============================================================
File:
    backend/ocr/reader.py

Purpose:
    Detect and read visible text, signs, labels, and documents from
    live camera frames using lightweight, pretrained ONNX OCR models
    (RapidOCR / PaddleOCR ONNX conversion).

Responsibilities:
    - Load pretrained ONNX OCR detector and recognizer once
    - Fast CPU execution without large model re-training
    - Filter out unreadable noise, low-confidence snippets, and micro-text
    - Spatially associate detected text with LEFT / CENTER / RIGHT sectors
    - Return standardized, non-breaking JSON output for /detect
============================================================
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class TextReader:
    """
    Pretrained ONNX-based Optical Character Recognition engine.
    Detects text regions, recognizes words/phrases, and maps
    spatial coordinates to navigation sectors (LEFT / CENTER / RIGHT).
    """

    def __init__(
        self,
        min_confidence: float = 0.55,
        min_box_size_px: int = 12,
        enabled: bool = True,
        dedup_cooldown_s: float = 10.0,
    ) -> None:
        self.min_confidence = float(min_confidence)
        self.min_box_size_px = int(min_box_size_px)
        self.enabled = bool(enabled)
        self.dedup_cooldown_s = float(dedup_cooldown_s)
        self._engine: Optional[Any] = None
        self._loaded: bool = False
        self._error: Optional[str] = None
        self._last_spoken_text: Optional[str] = None
        self._last_spoken_timestamp: float = 0.0

        if self.enabled:
            self._init_engine()

    def _init_engine(self) -> None:
        """
        Lazily initialize RapidOCR engine.
        """
        try:
            import rapidocr_onnxruntime
            # Instantiate with default pretrained ONNX models (ch_PP-OCRv4_det/rec or mobile)
            self._engine = rapidocr_onnxruntime.RapidOCR()
            self._loaded = True
            self._error = None
        except Exception as exc:
            self._loaded = False
            self._engine = None
            self._error = str(exc)
            print(f"[TextReader] Warning: Failed to initialize RapidOCR: {exc}")

    def status(self) -> Dict[str, Any]:
        """
        Return current OCR service status.
        """
        return {
            "enabled": self.enabled,
            "loaded": self._loaded,
            "engine": "rapidocr-onnxruntime",
            "min_confidence": self.min_confidence,
            "min_box_size_px": self.min_box_size_px,
            "error": self._error,
        }

    @staticmethod
    def _extract_directional_cue(text: str) -> Tuple[str, str]:
        """
        Extract directional arrow symbols or words from text before stripping.
        Returns (directional_cue, cleaned_text).
        directional_cue is one of: "LEFT", "RIGHT", "STRAIGHT", "UP", "DOWN", "NONE".
        """
        if not text:
            return "NONE", ""

        raw = text.strip()
        cue = "NONE"

        # Check for unicode and ascii arrows
        # Right arrows: →, ->, =>, >>, ►, ▶, ➔, ➜, ➤
        if re.search(r"(?:[\u2192\u2794-\u27BF\u25BA\u25B6]|->|=>|>>|>\|)", raw):
            cue = "RIGHT"
        # Left arrows: ←, <-, <=, <<, ◄, ◀
        elif re.search(r"(?:[\u2190\u25C4\u25C0]|<-|<=|<<|\|<)", raw):
            cue = "LEFT"
        # Up arrows: ↑, ▲
        elif re.search(r"(?:[\u2191\u25B2]|\^)", raw):
            cue = "UP"
        # Down arrows: ↓, ▼
        elif re.search(r"[\u2193\u25BC]", raw):
            cue = "DOWN"
        # Straight / Ahead word cues
        elif re.search(r"\b(?:STRAIGHT|AHEAD|FORWARD)\b", raw, re.IGNORECASE):
            cue = "STRAIGHT"
        elif re.search(r"\b(?:TO THE RIGHT|ON THE RIGHT|KEEP RIGHT|TURN RIGHT)\b", raw, re.IGNORECASE):
            cue = "RIGHT"
        elif re.search(r"\b(?:TO THE LEFT|ON THE LEFT|KEEP LEFT|TURN LEFT)\b", raw, re.IGNORECASE):
            cue = "LEFT"

        # Clean arrows out of text
        clean = re.sub(r"[\u2190-\u2199\u2794-\u27BF\u25B2\u25BC\u25C4\u25BA\u25B6\u25C0]", " ", raw)
        clean = re.sub(r"(?:->|<-|=>|<=|>>|<<|>\||\|<|-->|<--)", " ", clean)
        clean = re.sub(r"\s+", " ", clean).strip()

        return cue, clean

    @staticmethod
    def _compute_apparent_size(
        box_height: int,
        box_width: int,
        frame_h: int,
        frame_w: int,
    ) -> Tuple[str, bool, bool]:
        """
        Classify apparent text size and determine reliability vs uncertainty.
        Returns (apparent_size, is_reliable, is_uncertain).
        """
        if frame_h <= 0 or frame_w <= 0:
            return "MEDIUM", True, False

        area_ratio = float(box_height * box_width) / float(frame_h * frame_w)

        if box_height >= 45 or area_ratio >= 0.035:
            return "LARGE", True, False
        elif box_height >= 24 or area_ratio >= 0.015:
            return "MEDIUM", True, False
        elif box_height >= 16 or area_ratio >= 0.005:
            return "SMALL", True, False
        else:
            # Distant / Micro-text (< 16px height) has high character ambiguity
            return "DISTANT", False, True

    @staticmethod
    def _check_image_degradation(image: np.ndarray) -> Tuple[float, bool, str]:
        """
        Assess frame for low-light or motion blur to adjust OCR confidence.
        Returns (confidence_penalty, is_degraded, degradation_reason).
        """
        if image is None or image.size == 0:
            return 0.0, False, "NONE"

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        # 1. Low light check
        mean_lum = float(np.mean(gray))
        if mean_lum < 40.0:
            return 0.20, True, "LOW_LIGHT"

        # 2. Blur check via Laplacian variance on subsampled patch
        small_gray = gray[::2, ::2]
        lap_var = float(cv2.Laplacian(small_gray, cv2.CV_64F).var())
        if lap_var < 55.0:
            return 0.20, True, "BLUR"

        return 0.0, False, "NONE"

    @staticmethod
    def _semantic_priority_rank(text: str, directional_cue: str = "NONE") -> int:
        """
        Compute semantic priority tier for multi-region sorting:
        EMERGENCY/WARNING (100) > EXIT (90) > STAIRS/ELEVATOR (80) >
        ROOM/OFFICE/DEST (70) > RESTROOM (65) > RECEPTION/LOBBY (50) >
        DIRECTIONAL (40) > OTHER (10).
        """
        t = text.upper()
        if re.search(r"\b(?:DANGER|WARNING|CAUTION|DO NOT ENTER|NO ENTRY|EMERGENCY|FIRE ALARM)\b", t):
            return 100
        if re.search(r"\b(?:EXIT|FIRE EXIT|EMERGENCY EXIT|WAY OUT)\b", t):
            return 90
        if re.search(r"\b(?:STAIRS|STAIRWAY|STEPS|ELEVATOR|LIFT|FIRE ESCAPE)\b", t):
            return 80
        if re.search(r"\b(?:ROOM|RM|WARD|LAB|LABORATORY|OFFICE|SUITE|UNIT|CABIN)\s*[A-Z0-9\-]+\b", t):
            return 70
        if re.search(r"\b(?:RESTROOM|WASHROOM|TOILET|BATHROOM|LADIES|GENTS|WC)\b", t):
            return 65
        if re.search(r"\b(?:RECEPTION|ENTRANCE|CAFETERIA|LOBBY|INFORMATION|HELP DESK)\b", t):
            return 50
        if directional_cue != "NONE":
            return 40
        return 10

    @staticmethod
    def _is_partial_fragment(text: str) -> bool:
        """
        Detect whether a word is an un-navigable partial snippet
        (e.g., partial word missing initial or terminal characters).
        """
        t = text.strip().upper()
        # Common truncation patterns from occluded signs
        fragments = {"ESTROOM", "XIT", "LEVATOR", "OOM", "TAIRS", "ASHROOM", "ATHROOM"}
        if t in fragments:
            return True
        return False

    @staticmethod
    def _is_meaningful_text(text: str) -> bool:
        """
        Filter out garbage characters, single-letter artifacts, and unreadable noise.
        Keep meaningful alphanumeric words, numbers, and sign abbreviations.
        """
        stripped = text.strip()
        if len(stripped) < 2:
            # Allow common single-character signage e.g. "P" (parking), "M" (metro)
            return stripped in {"P", "M", "E", "W", "N", "S"}

        # Require at least one alphanumeric character
        if not re.search(r"[a-zA-Z0-9]", stripped):
            return False

        # Exclude common noise sequences of purely repeated symbols
        if re.fullmatch(r"[\W_]+", stripped):
            return False

        return True

    @staticmethod
    def _get_position(
        x1: int,
        x2: int,
        frame_width: int,
    ) -> str:
        """
        Determine whether the text is predominantly on the LEFT, CENTER, or RIGHT.
        """
        if frame_width <= 0:
            return "CENTER"

        center_x = (x1 + x2) / 2.0
        norm_center = center_x / float(frame_width)

        if norm_center < 0.35:
            return "LEFT"
        if norm_center > 0.65:
            return "RIGHT"
        return "CENTER"

    def extract_text(
        self,
        image: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """
        Run OCR on the given camera frame.

        Returns structured dictionary:
        {
            "detected": bool,
            "text_count": int,
            "primary_text": str,
            "items": [
                {
                    "text": str,
                    "confidence": float,
                    "position": "LEFT" | "CENTER" | "RIGHT",
                    "bbox": {"x1": int, "y1": int, "x2": int, "y2": int}
                },
                ...
            ]
        }
        """
        empty_result: Dict[str, Any] = {
            "detected": False,
            "text_count": 0,
            "primary_text": "",
            "items": [],
            "is_repeat": False,
            "should_speak": False,
        }

        if not self.enabled or not self._loaded or self._engine is None:
            return empty_result

        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return empty_result

        h, w = image.shape[:2]
        if h < 32 or w < 32:
            return empty_result

        # Downscale large frames to max dimension 640px for real-time CPU efficiency
        max_dim = max(h, w)
        if max_dim > 640:
            scale = 640.0 / float(max_dim)
            new_w = max(32, int(round(w * scale)))
            new_h = max(32, int(round(h * scale)))
            ocr_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            scale_x = float(w) / float(new_w)
            scale_y = float(h) / float(new_h)
        else:
            ocr_img = image
            scale_x = 1.0
            scale_y = 1.0

        try:
            ocr_results, _ = self._engine(ocr_img)
        except Exception as exc:
            # Fail safely without crashing the real-time detection pipeline
            return empty_result
        finally:
            if max_dim > 640:
                del ocr_img

        if not ocr_results:
            return empty_result

        # Check frame degradation (low-light, motion blur)
        penalty, is_degraded, deg_reason = self._check_image_degradation(image)

        items: List[Dict[str, Any]] = []

        for entry in ocr_results:
            # Entry format: [box, text, score]
            if not entry or len(entry) < 3:
                continue

            box, text, score = entry[0], entry[1], entry[2]
            try:
                confidence = float(score)
            except (ValueError, TypeError):
                continue

            if confidence < self.min_confidence:
                continue
            raw_text = str(text).strip()
            # Extract directional cues (arrows, directional words) before stripping
            directional_cue, clean_text = self._extract_directional_cue(raw_text)

            clean_text = str(text).strip()
            if not self._is_meaningful_text(clean_text):
                continue

            # Compute bounding box from the polygon points (scaled back to original dimensions)
            try:
                xs = [float(pt[0]) * scale_x for pt in box]
                ys = [float(pt[1]) * scale_y for pt in box]
                x1 = max(0, int(round(min(xs))))
                y1 = max(0, int(round(min(ys))))
                x2 = min(w, int(round(max(xs))))
                y2 = min(h, int(round(max(ys))))
            except Exception:
                continue

            box_width = x2 - x1
            box_height = y2 - y1

            # Ignore extremely small or narrow unreadable regions
            if box_width < self.min_box_size_px or box_height < self.min_box_size_px:
                continue

            # Apparent size & distance categorization
            app_size, is_reliable, is_uncertain = self._compute_apparent_size(
                box_height, box_width, h, w
            )

            # Check for partial word fragment / occluded text
            is_partial = self._is_partial_fragment(clean_text)
            if is_partial:
                is_uncertain = True
                is_reliable = False
                confidence = min(confidence, 0.50)

            # Apply degradation penalty
            effective_conf = max(0.10, confidence - penalty)
            if is_degraded:
                is_uncertain = True
                is_reliable = False

            if effective_conf < self.min_confidence:
                continue

            position = self._get_position(x1, x2, w)
            sem_rank = self._semantic_priority_rank(clean_text, directional_cue)

            items.append({
                "text": clean_text,
                "confidence": round(confidence, 3),
                "raw_text": raw_text,
                "confidence": round(effective_conf, 3),
                "raw_confidence": round(confidence, 3),
                "position": position,
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
                "directional_cue": directional_cue,
                "apparent_size": app_size,
                "is_reliable": is_reliable,
                "is_uncertain": is_uncertain,
                "is_partial": is_partial,
                "semantic_rank": sem_rank,
            })

        if not items:
            return empty_result

        # Sort items: highest confidence first, or top-to-bottom
        items.sort(key=lambda x: x["confidence"], reverse=True)
        # Sort items: highest semantic priority tier first, then highest confidence
        items.sort(key=lambda x: (x["semantic_rank"], x["confidence"]), reverse=True)

        # Primary text: most confident / prominent readable snippet
        primary_text = items[0]["text"]
        # Primary text: most safety/navigational relevant readable snippet
        primary_item = items[0]
        primary_text = primary_item["text"]
        primary_directional_cue = primary_item.get("directional_cue", "NONE")
        is_primary_uncertain = primary_item.get("is_uncertain", False)

        return {
            "detected": True,
            "text_count": len(items),
            "primary_text": primary_text,
            "primary_directional_cue": primary_directional_cue,
            "is_uncertain": is_primary_uncertain,
            "items": items,
            "is_repeat": False,
            "should_speak": True,
        }

    def read_text(
        self,
        image: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """
        Extract text from frame with intelligent de-duplication tracking.
        Determines 'is_repeat' and 'should_speak' to prevent voice spamming.
        """
        result = self.extract_text(image)
        if not result["detected"] or not result["primary_text"]:
            result["is_repeat"] = False
            result["should_speak"] = False
            return result

        current_time = time.time()
        norm_current = re.sub(r"\s+", " ", result["primary_text"].strip().lower())
        norm_last = re.sub(r"\s+", " ", (self._last_spoken_text or "").strip().lower())

        is_same_text = (norm_current == norm_last)
        elapsed = current_time - self._last_spoken_timestamp

        if is_same_text and elapsed < self.dedup_cooldown_s:
            result["is_repeat"] = True
            result["should_speak"] = False
        else:
            result["is_repeat"] = False
            result["should_speak"] = True
            self._last_spoken_text = result["primary_text"]
            self._last_spoken_timestamp = current_time

        return result

    def reset_history(self) -> None:
        """
        Reset deduplication cache.
        """
        self._last_spoken_text = None
        self._last_spoken_timestamp = 0.0

    def __call__(self, image: Optional[np.ndarray]) -> Dict[str, Any]:
        return self.read_text(image)

