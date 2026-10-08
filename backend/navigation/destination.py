"""
SmartVisionAI - Destination Representation and Landmark Matching
================================================================
Handles indoor destination specification and matching against detected
Phase 3 semantic indoor landmarks.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple, Union


SUPPORTED_DESTINATION_TYPES = {
    "ROOM",
    "OFFICE",
    "RESTROOM",
    "ELEVATOR",
    "STAIRS",
    "RECEPTION",
    "CAFETERIA",
    "LOBBY",
    "EXIT",
    "ENTRANCE",
    "INFORMATION",
}

WORD_TO_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}

ALIAS_TO_TYPE = {
    "restroom": "RESTROOM",
    "washroom": "RESTROOM",
    "toilet": "RESTROOM",
    "bathroom": "RESTROOM",
    "wc": "RESTROOM",
    "elevator": "ELEVATOR",
    "lift": "ELEVATOR",
    "stair": "STAIRS",
    "stairs": "STAIRS",
    "staircase": "STAIRS",
    "exit": "EXIT",
    "emergency exit": "EXIT",
    "fire exit": "EXIT",
    "entrance": "ENTRANCE",
    "entry": "ENTRANCE",
    "reception": "RECEPTION",
    "front desk": "RECEPTION",
    "cafeteria": "CAFETERIA",
    "canteen": "CAFETERIA",
    "cafe": "CAFETERIA",
    "lobby": "LOBBY",
    "information": "INFORMATION",
    "helpdesk": "INFORMATION",
    "help desk": "INFORMATION",
    "office": "OFFICE",
    "room": "ROOM",
}


def parse_destination_query(query: str) -> Tuple[str, Optional[str], str]:
    """
    Parse natural language query into (dest_type, identifier, label).
    Handles word numerals ("room three zero two" -> "ROOM", "302", "Room 302")
    and synonyms ("washroom" -> "RESTROOM", None, "Restroom").
    """
    raw = str(query).strip()
    s = raw.lower()

    # Convert word digits: "three zero two" -> "3 0 2"
    for word, digit in WORD_TO_DIGIT.items():
        s = re.sub(rf"\b{word}\b", digit, s)
    # Collapse digits with spaces: "3 0 2" -> "302"
    s = re.sub(r'(?<=\d)\s+(?=\d)', '', s)

    # Detect category / type
    dest_type = "ROOM"
    matched_type = False
    for alias, cat in sorted(ALIAS_TO_TYPE.items(), key=lambda x: -len(x[0])):
        if re.search(rf"\b{re.escape(alias)}\b", s):
            dest_type = cat
            matched_type = True
            break

    # Detect numeric or alphanumeric identifier (e.g. 302, 105A)
    id_match = re.search(r'\b(\d+[a-zA-Z]?)\b', s)
    identifier = id_match.group(1).upper() if id_match else None

    # If digits are found without explicit category, default to ROOM
    if not matched_type and identifier:
        dest_type = "ROOM"

    # Natural label
    if identifier:
        label = f"{dest_type.title()} {identifier}"
    else:
        label = dest_type.title()

    return dest_type, identifier, label


class Destination:
    """
    Represents an indoor destination requested by the user.
    """

    def __init__(
        self,
        dest_type: str,
        identifier: Optional[str] = None,
        label: Optional[str] = None,
        active: bool = True,
    ) -> None:
        raw_input = str(dest_type).strip()

        # If only one string was passed, parse it flexibly
        if identifier is None and label is None:
            parsed_type, parsed_id, parsed_label = parse_destination_query(raw_input)
            self.type = parsed_type
            self.identifier = parsed_id
            self.label = parsed_label
        else:
            self.type = raw_input.upper()
            self.identifier = str(identifier).strip().upper() if identifier is not None and str(identifier).strip() else None
            parsed_type, parsed_id, _ = parse_destination_query(raw_input)
            if raw_input.upper() in SUPPORTED_DESTINATION_TYPES:
                self.type = raw_input.upper()
            elif parsed_type:
                self.type = parsed_type
            else:
                self.type = raw_input.upper()

            self.identifier = str(identifier).strip().upper() if identifier is not None and str(identifier).strip() else parsed_id
            if label and str(label).strip():
                self.label = str(label).strip()
            else:
                if self.identifier:
                    self.label = f"{self.type.title()} {self.identifier}"
                else:
                    self.label = self.type.title()

        self.active = bool(active)

    @property
    def name(self) -> str:
        """Alias for label."""
        return self.label

    def to_dict(self) -> Dict[str, Any]:
        """Serialize destination to dictionary."""
        return {
            "type": self.type,
            "identifier": self.identifier,
            "label": self.label,
            "name": self.label,
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional[Destination]:
        """Instantiate Destination from dictionary or return None."""
        if not data or not isinstance(data, dict):
            return None
        dest_type = data.get("type") or data.get("name") or data.get("destination")
        if not dest_type:
            return None
        return cls(
            dest_type=str(dest_type),
            identifier=data.get("identifier"),
            label=data.get("label") or data.get("name"),
            active=bool(data.get("active", True)),
        )

    def matches(self, landmark: Dict[str, Any]) -> bool:
        """Check if a semantic landmark matches this destination."""
        is_match, _ = match_destination(self, landmark)
        return is_match


def match_destination(
    destination: Union[Destination, str, None],
    landmark: Dict[str, Any],
) -> Tuple[bool, float]:
    """
    Match a detected Phase 3 semantic landmark against the active destination.

    Returns:
        (is_match: bool, confidence: float)
    """
    if isinstance(destination, str):
        destination = Destination(destination)

    if not destination or not destination.active or not landmark:
        return False, 0.0

    l_type = str(landmark.get("type", "")).strip().upper()
    l_ident = landmark.get("identifier")
    l_ident_str = str(l_ident).strip().upper() if l_ident is not None and str(l_ident).strip() else None
    l_text = str(landmark.get("clean_text", landmark.get("text", landmark.get("label", "")))).strip().upper()
    l_conf = float(landmark.get("confidence", 0.75))

    # Phase 9: Reject uncertain, unreliable, or low-confidence landmarks for active destination navigation
    if (
        landmark.get("is_uncertain") is True
        or landmark.get("is_reliable") is False
        or landmark.get("is_navigable_cue") is False
        or l_conf < 0.60
    ):
        return False, 0.0

    target_type = destination.type
    target_id = destination.identifier

    # Check if landmark type matches or is an alias
    type_matched = (l_type == target_type)
    if not type_matched:
        # Check text keywords for category synonyms
        if target_type == "RESTROOM" and any(w in l_text for w in ("RESTROOM", "WASHROOM", "TOILET", "BATHROOM", "WC")):
            type_matched = True
        elif target_type == "ELEVATOR" and any(w in l_text for w in ("ELEVATOR", "LIFT")):
            type_matched = True
        elif target_type == "STAIRS" and any(w in l_text for w in ("STAIR", "STAIRS", "STAIRCASE")):
            type_matched = True
        elif target_type == "EXIT" and ("EXIT" in l_text or l_type in ("EXIT", "EMERGENCY")):
            type_matched = True

    if not type_matched:
        return False, 0.0

    # If destination specifies a room/office identifier, strict identifier matching
    if target_id is not None:
        if l_ident_str:
            if l_ident_str == target_id:
                return True, l_conf
            else:
                # Explicit mismatch in room/office number (e.g. 302 vs 305)
                return False, 0.0
        else:
            # Check if target identifier exists in text
            if re.search(rf"\b{re.escape(target_id)}\b", l_text):
                return True, l_conf
            return False, 0.0

    # Destination without identifier matches category
    return True, l_conf
