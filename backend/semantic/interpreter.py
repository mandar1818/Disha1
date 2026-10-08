"""
SmartVisionAI - Semantic Indoor Interpreter & Landmark Layer
============================================================
Translates raw OCR text detections into semantic indoor understanding:
- Signage classification: EXIT, ENTRANCE, RESTROOM, STAIRS, ELEVATOR,
  RECEPTION, ROOM, OFFICE, CAFETERIA, LOBBY, WARNING, EMERGENCY, INFORMATION
- Room / Area identifier extraction (e.g. "ROOM 302" -> ROOM, id="302", "WARD 5" -> id="5")
- Spatial localization: LEFT, CENTER, RIGHT
- Text de-noising: Unicode/ASCII arrows removal, case/whitespace normalization
- Temporal stability and voice deduplication to suppress repetitive announcements
- Preparation for indoor destination navigation
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple


SemanticLandmarkType = str  # "EXIT" | "ENTRANCE" | "RESTROOM" | "STAIRS" | ...


class SemanticLandmark:
    """
    Standardized semantic indoor landmark representation.
    """

    def __init__(
        self,
        landmark_type: str,
        label: str,
        raw_text: str,
        clean_text: str,
        position: str,
        confidence: float,
        bbox: Dict[str, Any],
        identifier: Optional[str] = None,
        is_navigable_cue: bool = True,
        safety_relevance: str = "MEDIUM",
        should_announce: bool = True,
        directional_cue: str = "NONE",
        apparent_size: str = "MEDIUM",
        is_reliable: bool = True,
        is_uncertain: bool = False,
    ) -> None:
        self.type = landmark_type
        self.label = label
        self.raw_text = raw_text
        self.clean_text = clean_text
        self.position = position.upper()
        self.confidence = float(confidence)
        self.bbox = bbox
        self.identifier = identifier
        self.is_navigable_cue = bool(is_navigable_cue)
        self.safety_relevance = safety_relevance
        self.should_announce = bool(should_announce)
        self.directional_cue = str(directional_cue).upper()
        self.apparent_size = str(apparent_size).upper()
        self.is_reliable = bool(is_reliable)
        self.is_uncertain = bool(is_uncertain)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "type": self.type,
            "label": self.label,
            "text": self.raw_text,
            "clean_text": self.clean_text,
            "identifier": self.identifier,
            "position": self.position,
            "directional_cue": self.directional_cue,
            "apparent_size": self.apparent_size,
            "is_reliable": self.is_reliable,
            "is_uncertain": self.is_uncertain,
            "confidence": round(self.confidence, 3),
            "bbox": self.bbox,
            "is_navigable_cue": self.is_navigable_cue,
            "safety_relevance": self.safety_relevance,
            "should_announce": self.should_announce,
        }


class SemanticInterpreter:
    """
    Interprets raw OCR text detections into semantic indoor landmarks.
    """

    def __init__(self, announcement_cooldown_s: float = 8.0) -> None:
        self.announcement_cooldown_s = float(announcement_cooldown_s)
        # History format: {landmark_key: last_announced_timestamp}
        self._history: Dict[str, float] = {}
        self._last_announced_type: Optional[str] = None
        self._last_announced_time: float = 0.0

    # ------------------------------------------------------------------
    # Text Normalization & Cleaning
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_text(raw_text: str) -> str:
        """
        Clean text of directional arrows, extraneous punctuation, and optical noise.
        """
        if not raw_text:
            return ""

        text = raw_text.strip()

        # Remove directional arrows (Unicode and ASCII)
        # Unicode arrows: →, ←, ↑, ↓, ↔, ↕, ➔, ➜, ➤, ▲, ▼, ◄, ►
        text = re.sub(r"[\u2190-\u2199\u2794-\u27BF\u25B2\u25BC\u25C4\u25BA]", " ", text)
        # ASCII arrows: ->, <-, =>, <=, >>, <<, >|, |<
        text = re.sub(r"(?:->|<-|=>|<=|>>|<<|>\||\|<|-->|<--)", " ", text)
        # Remove standalone boundary bars and slashes
        text = re.sub(r"(?:^[|/\\\-_:\.]+|[|/\\\-_:\.]+$)", " ", text)

        # Collapse whitespace and convert to uppercase
        clean = re.sub(r"\s+", " ", text).strip().upper()

        # OCR character corrections where safe (e.g. EX1T -> EXIT, R00M -> ROOM, 5TAIRS -> STAIRS)
        clean = re.sub(r"\bEX1T\b", "EXIT", clean)
        clean = re.sub(r"\bR00M\b", "ROOM", clean)
        clean = re.sub(r"\b5TAIRS\b", "STAIRS", clean)
        clean = re.sub(r"\bT0ILET\b", "TOILET", clean)

        return clean

    # ------------------------------------------------------------------
    # Semantic Classification
    # ------------------------------------------------------------------

    def classify_sign(self, clean_text: str) -> Tuple[str, str, Optional[str], bool, str]:
        """
        Classify normalized text into a semantic category.

        Returns:
            Tuple of:
                semantic_type: str
                display_label: str
                identifier: Optional[str] (e.g. room number "302")
                is_navigable_cue: bool
                safety_relevance: "HIGH" | "MEDIUM" | "LOW"
        """
        text = clean_text.upper()

        # 1. EMERGENCY & WARNING (Highest relevance safety signs)
        if re.search(r"\b(?:DO NOT ENTER|NO ENTRY|KEEP OUT|RESTRICTED|AUTHORIZED PERSONNEL ONLY)\b", text):
            return "WARNING", "Do Not Enter", None, False, "HIGH"
        if re.search(r"\b(?:DANGER|WARNING|CAUTION|HIGH VOLTAGE|SLIPPERY FLOOR|WET FLOOR)\b", text):
            return "WARNING", "Warning", None, False, "HIGH"
        if re.search(r"\b(?:FIRE ALARM|FIRE EXTINGUISHER|EXTINGUISHER|FIRST AID|DEFIBRILLATOR|AED)\b", text):
            return "EMERGENCY", "Emergency Equipment", None, False, "HIGH"

        # 2. EXIT
        if re.search(r"\b(?:EMERGENCY EXIT|FIRE EXIT)\b", text):
            return "EXIT", "Emergency Exit", None, True, "HIGH"
        if re.search(r"\b(?:EXIT|WAY OUT|AUSGANG|SORTIE)\b", text):
            return "EXIT", "Exit", None, True, "HIGH"

        # 3. ENTRANCE
        if re.search(r"\b(?:MAIN ENTRANCE|ENTRANCE|ENTRY|INGANG|ENTREE)\b", text):
            return "ENTRANCE", "Entrance", None, True, "MEDIUM"

        # 4. RESTROOM
        if re.search(r"\b(?:RESTROOM|RESTROOMS|REST ROOM|WASHROOM|WASHROOMS|TOILET|TOILETS|BATHROOM|BATHROOMS|LADIES|GENTS|WOMEN|MEN|WC|LAVATORY)\b", text):
            label = "Restroom"
            if re.search(r"\b(?:WOMEN|LADIES)\b", text):
                label = "Women's Restroom"
            elif re.search(r"\b(?:MEN|GENTS)\b", text):
                label = "Men's Restroom"
            return "RESTROOM", label, None, True, "HIGH"

        # 5. STAIRS
        if re.search(r"\b(?:STAIRS|STAIRWAY|STAIRCASE|STEPS|STAIRWELL|FIRE ESCAPE)\b", text):
            return "STAIRS", "Stairs", None, True, "HIGH"

        # 6. ELEVATOR
        if re.search(r"\b(?:ELEVATOR|ELEVATORS|LIFT|LIFTS)\b", text):
            return "ELEVATOR", "Elevator", None, True, "HIGH"

        # 7. ROOM / WARD / LAB (Extract identifier)
        m_room = re.search(r"\b(?:ROOM|RM|WARD|LAB|LABORATORY|HALL|CABIN|UNIT|SUITE)\s*([A-Za-z0-9\-]+)\b", text)
        if m_room:
            identifier = m_room.group(1).strip()
            # Determine prefix type
            prefix = "Room"
            if "WARD" in text:
                prefix = "Ward"
            elif "LAB" in text:
                prefix = "Lab"
            elif "HALL" in text:
                prefix = "Hall"
            return "ROOM", f"{prefix} {identifier}", identifier, True, "MEDIUM"

        if re.search(r"\b(?:CONFERENCE ROOM|MEETING ROOM|CLASSROOM|AUDITORIUM)\b", text):
            return "ROOM", text.title(), None, True, "MEDIUM"

        # 8. OFFICE (Extract optional identifier)
        m_office = re.search(r"\b(?:OFFICE)\s*([A-Za-z0-9\-]+)\b", text)
        if m_office:
            identifier = m_office.group(1).strip()
            return "OFFICE", f"Office {identifier}", identifier, True, "MEDIUM"
        if re.search(r"\b(?:OFFICE|ADMINISTRATION|DIRECTOR|DEAN|FACULTY|STAFF|PRINCIPAL)\b", text):
            return "OFFICE", text.title(), None, True, "MEDIUM"

        # 9. RECEPTION
        if re.search(r"\b(?:RECEPTION|FRONT DESK|INFORMATION DESK|ENQUIRY|CONCIERGE|REGISTRATION|CHECK IN)\b", text):
            return "RECEPTION", "Reception", None, True, "HIGH"

        # 10. CAFETERIA
        if re.search(r"\b(?:CAFETERIA|CANTEEN|DINING|FOOD COURT|CAFE|MESS|RESTAURANT|PANTRY)\b", text):
            return "CAFETERIA", "Cafeteria", None, True, "MEDIUM"

        # 11. LOBBY
        if re.search(r"\b(?:LOBBY|FOYER|ATRIUM|WAITING AREA|WAITING ROOM|LOUNGE)\b", text):
            return "LOBBY", "Lobby", None, True, "MEDIUM"

        # 12. INFORMATION
        if re.search(r"\b(?:INFORMATION|DIRECTORY|HELP DESK|NOTICE|GUIDE|MAP)\b", text):
            return "INFORMATION", "Information", None, True, "LOW"

        # 13. Default: readable sign without recognized landmark keywords
        # 13. PHARMACY & HEALTHCARE
        if re.search(r"\b(?:PHARMACY|DISPENSARY|CHEMIST|DRUG STORE)\b", text):
            return "PHARMACY", "Pharmacy", None, True, "MEDIUM"
        if re.search(r"\b(?:CLINIC|TRIAGE|ICU|OPD|RADIOLOGY|PATHOLOGY|PHYSIOTHERAPY)\b", text):
            return "CLINIC", text.title(), None, True, "MEDIUM"

        # 14. LIBRARY & CAMPUS
        if re.search(r"\b(?:LIBRARY|READING ROOM)\b", text):
            return "LIBRARY", "Library", None, True, "MEDIUM"
        if re.search(r"\b(?:SECURITY|GUARD|SECURITY DESK)\b", text):
            return "SECURITY", "Security Desk", None, True, "HIGH"

        # 15. Default: readable sign without recognized landmark keywords
        return "UNKNOWN_SIGN", text.title(), None, False, "LOW"

    # ------------------------------------------------------------------
    # Full OCR Items Interpretation
    # ------------------------------------------------------------------

    def interpret(
        self,
        ocr_result: Dict[str, Any],
        frame_width: float = 640.0,
    ) -> List[Dict[str, Any]]:
        """
        Process OCR result dictionary and produce list of SemanticLandmarks.

        Returns:
            List of semantic landmark dictionaries.
        """
        if not ocr_result or not isinstance(ocr_result, dict):
            return []

        raw_items = ocr_result.get("items", [])
        if not raw_items:
            # Check primary_text fallback
            primary = ocr_result.get("primary_text", "").strip()
            if primary:
                raw_items = [{
                    "text": primary,
                    "confidence": 0.70,
                    "position": "CENTER",
                    "bbox": {"x1": 0, "y1": 0, "x2": int(frame_width), "y2": 50},
                }]
            else:
                return []

        landmarks: List[Dict[str, Any]] = []
        current_time = time.time()

        for item in raw_items:
            raw_text = str(item.get("text", "")).strip()
            if not raw_text:
                continue

            clean_text = self.normalize_text(raw_text)
            if len(clean_text) < 2 and clean_text not in ("P", "M", "E", "W", "N", "S"):
                continue

            sem_type, label, ident, is_nav, safety_rel = self.classify_sign(clean_text)

            # Spatial positioning: resolve from item position or bbox center
            pos = str(item.get("position", "")).upper()
            bbox = item.get("bbox", {})
            if pos not in ("LEFT", "CENTER", "RIGHT") and bbox and frame_width > 0:
                cx = (float(bbox.get("x1", 0)) + float(bbox.get("x2", 0))) / 2.0
                rel_x = cx / float(frame_width)
                if rel_x < 0.35:
                    pos = "LEFT"
                elif rel_x > 0.65:
                    pos = "RIGHT"
                else:
                    pos = "CENTER"
            if pos not in ("LEFT", "CENTER", "RIGHT"):
                pos = "CENTER"

            conf = float(item.get("confidence", 0.75))

            # Directional cue extraction
            directional_cue = str(item.get("directional_cue", "NONE")).upper()
            if directional_cue == "NONE":
                # Fallback to inspecting raw_text for arrows
                orig_raw = str(item.get("raw_text", raw_text))
                if re.search(r"(?:[\u2192\u2794-\u27BF\u25BA\u25B6]|->|=>|>>|>\|)", orig_raw):
                    directional_cue = "RIGHT"
                elif re.search(r"(?:[\u2190\u25C4\u25C0]|<-|<=|<<|\|<)", orig_raw):
                    directional_cue = "LEFT"
                elif re.search(r"(?:[\u2191\u25B2]|\^)", orig_raw):
                    directional_cue = "UP"
                elif re.search(r"[\u2193\u25BC]", orig_raw):
                    directional_cue = "DOWN"

            # Apparent size & reliability
            apparent_size = str(item.get("apparent_size", "MEDIUM")).upper()
            is_reliable = bool(item.get("is_reliable", True))
            is_uncertain = bool(item.get("is_uncertain", False))

            # If confidence is poor, text is distant, or partial fragment, mark uncertain & unnavigable
            if conf < 0.60 or apparent_size == "DISTANT" or item.get("is_partial"):
                is_reliable = False
                is_uncertain = True
                is_nav = False

            # Deduplication key for announcement stability
            dedup_key = f"{sem_type}:{ident or ''}:{pos}"
            dedup_key = f"{sem_type}:{ident or ''}:{pos}:{directional_cue}"
            last_spoken = self._history.get(dedup_key, 0.0)
            elapsed = current_time - last_spoken

            # An announcement is warranted if cooldown has expired
            should_announce = (elapsed >= self.announcement_cooldown_s)
            # An announcement is warranted if cooldown has expired and text is reliable
            should_announce = (elapsed >= self.announcement_cooldown_s) and is_reliable and not is_uncertain

            landmark = SemanticLandmark(
                landmark_type=sem_type,
                label=label,
                raw_text=raw_text,
                clean_text=clean_text,
                position=pos,
                confidence=conf,
                bbox=bbox,
                identifier=ident,
                is_navigable_cue=is_nav,
                safety_relevance=safety_rel,
                should_announce=should_announce,
                directional_cue=directional_cue,
                apparent_size=apparent_size,
                is_reliable=is_reliable,
                is_uncertain=is_uncertain,
            )

            landmarks.append(landmark.to_dict())

        # Sort landmarks: high safety relevance first, then high confidence
        # Sort landmarks: high safety relevance first, then reliability, then confidence
        relevance_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        landmarks.sort(
            key=lambda l: (
                relevance_rank.get(l.get("safety_relevance", "LOW"), 0),
                1 if l.get("is_reliable", True) else 0,
                l.get("confidence", 0.0),
            ),
            reverse=True,
        )

        # Multi-region voice safety: at most ONE landmark may speak per frame
        announced_in_frame = False
        for l in landmarks:
            if l.get("should_announce") and not announced_in_frame:
                announced_in_frame = True
            else:
                l["should_announce"] = False

        return landmarks

    def mark_announced(self, landmark: Dict[str, Any]) -> None:
        """
        Record that a semantic landmark was announced to prevent repetition.
        """
        sem_type = landmark.get("type", "UNKNOWN_SIGN")
        ident = landmark.get("identifier") or ""
        pos = landmark.get("position", "CENTER")
        key = f"{sem_type}:{ident}:{pos}"
        cue = landmark.get("directional_cue", "NONE")
        key = f"{sem_type}:{ident}:{pos}:{cue}"
        now = time.time()
        self._history[key] = now
        self._last_announced_type = sem_type
        self._last_announced_time = now

    def generate_landmark_speech(
        self,
        landmark: Dict[str, Any],
    ) -> str:
        """
        Formulate a short, natural English voice cue for an indoor landmark.
        Handles directional arrows (LEFT, RIGHT, STRAIGHT, UP, DOWN) as well as spatial positions.
        """
        label = landmark.get("label", landmark.get("type", "sign")).strip()
        pos = str(landmark.get("position", "CENTER")).upper()
        cue = str(landmark.get("directional_cue", "NONE")).upper()

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

        if pos == "LEFT":
            return f"{label} on your left."
        elif pos == "RIGHT":
            return f"{label} on your right."
        else:
            return f"{label} ahead."

    def reset(self) -> None:
        """Clear temporal history."""
        self._history.clear()
        self._last_announced_type = None
        self._last_announced_time = 0.0

