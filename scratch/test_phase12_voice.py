#!/usr/bin/env python3
"""
SmartVisionAI - Phase 12 Comprehensive Voice Guidance Test Suite
Tests:
- 5-Level Priority Hierarchy (Emergency, Rapid Danger, Turn, Movement, General Awareness)
- Cooldowns (Normal 2500ms, Approaching 2500ms, Safety 3000ms, Emergency 1000ms, Repeat 7000ms)
- Significant State Change Bypass (Action, Safety, Motion, Distance, Track ID, Step Delta >= 2)
- Text Normalization & English-Only Formatting
- Step Word Formatting (1-10 words, singular/plural, >10 digits, null/0)
- Turn vs Movement Guidance (Turns strictly directional - NEVER step counts)
- Speech Lock & Queue Management (Max 1 queue slot, priority preemption, stale queue eviction)
- End-to-End Walk Scenario Simulation
"""

import sys
import os
import re
import math
import unittest
from typing import Optional, Dict, Any, List

# =====================================================================
# Phase 12 Voice Decision Manager Reference Implementation in Python
# (Mirrors mobile/src/services/speech.ts exactly for high-fidelity verification)
# =====================================================================

COOLDOWNS = {
    "NORMAL": 2500,
    "APPROACHING": 2500,
    "SAFETY": 3000,
    "EMERGENCY": 1000,
    "REPEAT": 7000,
}

STEP_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}

def normalize_instruction(text: str) -> str:
    if not text:
        return ""
    cleaned = text.lower()
    cleaned = re.sub(r"[.,!?:;\-_]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()

def format_step_words(steps: int) -> str:
    word = STEP_WORDS.get(steps, str(steps))
    return f"{word} step" if steps == 1 else f"{word} steps"

def get_safety_rank(level_or_alert: str) -> int:
    clean = (level_or_alert or "").upper().strip()
    if "EMERGENCY" in clean:
        return 4
    if "DANGER" in clean or "RAPID" in clean:
        return 3
    if "CAUTION" in clean:
        return 2
    if "AWARE" in clean:
        return 1
    return 0

def get_motion_rank(motion: str) -> int:
    clean = (motion or "").upper().strip()
    if clean in ("RAPIDLY_APPROACHING", "RAPID"):
        return 3
    if clean == "APPROACHING":
        return 2
    if clean in ("STATIONARY", "STABLE"):
        return 1
    return 0

def get_distance_rank(category_or_meters: Any) -> int:
    if isinstance(category_or_meters, (int, float)):
        if category_or_meters < 0.8:
            return 3
        if category_or_meters < 1.5:
            return 2
        if category_or_meters < 3.0:
            return 1
        return 0
    clean = str(category_or_meters or "").upper().strip()
    if "VERY_CLOSE" in clean or "IMMEDIATE" in clean:
        return 3
    if "CLOSE" in clean or "NEAR" in clean:
        return 2
    if "MEDIUM" in clean:
        return 1
    return 0

class VoiceDecisionManagerPy:
    def __init__(self):
        self.reset()

    def reset(self):
        self.is_speaking = False
        self.last_spoken_text = ""
        self.last_spoken_at = 0
        self.last_priority = 0
        self.last_safety_alert = "SAFE"
        self.last_motion_trend = "STATIONARY"
        self.last_distance_category = "FAR"
        self.last_threat_track_id = None
        self.last_action = "NONE"
        self.last_steps = None
        self.queued_instruction = None
        self.haptic_triggered = False

    def evaluate(self, response: Dict[str, Any], now: int = 0) -> Dict[str, Any]:
        if not response or not response.get("success", False):
            return {
                "action": "IGNORE",
                "instruction": "",
                "priority": 1,
                "reason": "Invalid response",
                "bypass_cooldown": False,
            }

        # 1. EXTRACT DATA FIELDS
        emergency_stop = (
            response.get("emergency_stop") is True
            or response.get("emergency_alert") is True
            or str(response.get("safety_level", "")).upper() == "EMERGENCY"
        )
        safety_alert = (
            response.get("safety_alert")
            or ("EMERGENCY_STOP" if emergency_stop else response.get("safety_level", "SAFE"))
        )

        obstacle = response.get("primary_obstacle") or {}
        threat = response.get("predictive_threat") or {}
        primary_threat = threat or obstacle or {}

        nav_raw = response.get("navigation")
        step_guidance = response.get("step_guidance") or {}

        raw_action = "NONE"
        if step_guidance.get("action"):
            raw_action = str(step_guidance["action"]).upper()
        elif isinstance(nav_raw, dict) and nav_raw.get("action"):
            raw_action = str(nav_raw["action"]).upper()
        elif isinstance(nav_raw, dict) and nav_raw.get("direction"):
            raw_action = str(nav_raw["direction"]).upper()
        elif isinstance(nav_raw, str) and nav_raw:
            raw_action = nav_raw.upper()

        action = raw_action
        if raw_action in ("WALK_FORWARD", "GO_FORWARD", "FORWARD", "CONTINUE_FORWARD"):
            action = "MOVE_FORWARD"
        elif raw_action in ("WALK_BACK", "GO_BACK", "BACK"):
            action = "MOVE_BACK"
        elif raw_action in ("MOVE_LEFT",):
            action = "TURN_LEFT"
        elif raw_action in ("MOVE_RIGHT",):
            action = "TURN_RIGHT"

        steps = None
        if "steps" in step_guidance and isinstance(step_guidance["steps"], (int, float)):
            steps = int(step_guidance["steps"])
        elif isinstance(nav_raw, dict) and "movement_steps" in nav_raw and isinstance(nav_raw["movement_steps"], (int, float)):
            steps = int(nav_raw["movement_steps"])
        elif isinstance(nav_raw, dict) and "steps" in nav_raw and isinstance(nav_raw["steps"], (int, float)):
            steps = int(nav_raw["steps"])

        threat_class = str(
            primary_threat.get("class_name")
            or ("person" if (response.get("multiple_objects") or {}).get("human_detected") else "obstacle")
        ).lower()
        is_person = threat_class == "person"

        distance_m = None
        if "distance_m" in primary_threat and isinstance(primary_threat["distance_m"], (int, float)):
            distance_m = float(primary_threat["distance_m"])
        elif "estimated_distance_m" in primary_threat and isinstance(primary_threat["estimated_distance_m"], (int, float)):
            distance_m = float(primary_threat["estimated_distance_m"])
        elif isinstance(nav_raw, dict) and "distance_m" in nav_raw and isinstance(nav_raw["distance_m"], (int, float)):
            distance_m = float(nav_raw["distance_m"])

        threat_motion = str(
            threat.get("motion_state")
            or obstacle.get("motion_state")
            or obstacle.get("motion_trend")
            or ("APPROACHING" if response.get("approaching") else "STATIONARY")
        ).upper()

        is_rapid_approach = (
            threat_motion in ("RAPIDLY_APPROACHING", "RAPID")
            or safety_alert == "WARNING_RAPID_APPROACH"
            or str(threat.get("threat_level", "")).upper() == "EMERGENCY"
        )

        is_approaching = (
            is_rapid_approach
            or threat_motion == "APPROACHING"
            or response.get("approaching") is True
            or obstacle.get("approaching") is True
            or threat.get("is_approaching") is True
        )

        distance_category = primary_threat.get("distance_category")
        if not distance_category:
            if distance_m is not None:
                if distance_m < 0.8:
                    distance_category = "VERY_CLOSE"
                elif distance_m < 1.5:
                    distance_category = "CLOSE"
                elif distance_m < 3.0:
                    distance_category = "MEDIUM"
                else:
                    distance_category = "FAR"
            else:
                distance_category = "UNKNOWN_DISTANCE"

        threat_track_id = primary_threat.get("track_id") or primary_threat.get("obstacle_id")

        # 2. DETERMINE PRIORITY & CANDIDATE INSTRUCTION (5-Level Strict Hierarchy)
        priority = 1
        instruction = ""

        # Level 5: EMERGENCY
        if (
            emergency_stop
            or safety_alert == "EMERGENCY_STOP"
            or str(response.get("safety_level", "")).upper() == "EMERGENCY"
            or (action == "STOP" and distance_m is not None and distance_m < 0.6)
            or (is_rapid_approach and distance_m is not None and distance_m < 0.8)
        ):
            priority = 5
            instruction = "Stop immediately."
        # Level 4: RAPID DANGER / RAPIDLY APPROACHING
        elif (
            is_rapid_approach
            or safety_alert == "WARNING_RAPID_APPROACH"
            or (str(response.get("safety_level", "")).upper() == "DANGER" and (is_approaching or action == "STOP"))
        ):
            priority = 4
            if is_approaching:
                instruction = "Warning. Person approaching." if is_person else "Warning. Object approaching."
            else:
                instruction = "Stop."
        # Level 3: TURN (strictly directional - NEVER step counts)
        elif action in ("TURN_LEFT", "TURN_RIGHT"):
            priority = 3
            instruction = "Turn left." if action == "TURN_LEFT" else "Turn right."
        # Level 2: MOVEMENT / WALKING STEPS
        elif action in ("MOVE_FORWARD", "MOVE_BACK"):
            priority = 2
            dir_word = "forward" if action == "MOVE_FORWARD" else "back"
            if steps is not None and steps > 0:
                instruction = f"Move {dir_word} {format_step_words(steps)}."
            else:
                instruction = f"Move {dir_word}."
        # Level 1: GENERAL AWARENESS / NORMAL APPROACHING
        else:
            priority = 1
            if is_approaching:
                instruction = "Warning. Person approaching." if is_person else "Warning. Object approaching."
            elif action == "STOP":
                instruction = "Stop."
            elif response.get("voice_instruction") and str(response["voice_instruction"]).strip():
                instruction = str(response["voice_instruction"]).strip()
            else:
                obj_count = len(response.get("objects", []))
                if obj_count == 0 and not obstacle and not threat:
                    instruction = "Path clear."
                else:
                    instruction = "Continue carefully."

        if not instruction.strip():
            return {
                "action": "IGNORE",
                "instruction": "",
                "priority": 1,
                "reason": "Empty instruction",
                "bypass_cooldown": False,
            }

        if response.get("should_speak") is False:
            return {
                "action": "SUPPRESS",
                "instruction": instruction,
                "priority": priority,
                "reason": "Backend indicated should_speak = False (stationary stop / unchanged situation)",
                "bypass_cooldown": False,
            }

        # 3. SIGNIFICANT STATE CHANGE DETECTION
        action_changed = (
            action != "NONE"
            and action != self.last_action
        )

        safety_escalated = get_safety_rank(safety_alert) > get_safety_rank(self.last_safety_alert)
        motion_escalated = get_motion_rank(threat_motion) > get_motion_rank(self.last_motion_trend)
        dist_escalated = get_distance_rank(distance_category) > get_distance_rank(self.last_distance_category)

        threat_track_changed = (
            threat_track_id is not None
            and self.last_threat_track_id is not None
            and threat_track_id != self.last_threat_track_id
            and (action != "STOP" or (distance_m is not None and distance_m < 0.40))
        )

        step_count_changed = (
            steps is not None
            and self.last_steps is not None
            and action != "STOP"
            and abs(steps - self.last_steps) >= 2
        )

        bypass_cooldown = (
            action_changed
            or safety_escalated
            or motion_escalated
            or dist_escalated
            or threat_track_changed
            or step_count_changed
        )

        # 4. NORMALIZED COMPARISON & COOLDOWN / REPEAT TIMING
        norm_candidate = normalize_instruction(instruction)
        norm_last_spoken = normalize_instruction(self.last_spoken_text)
        is_same_text = norm_candidate == norm_last_spoken and len(norm_candidate) > 0
        elapsed = now - self.last_spoken_at

        is_stop_instruction = action == "STOP" or norm_candidate.startswith("stop")
        was_already_stopped = self.last_action == "STOP" or norm_last_spoken.startswith("stop")
        if is_stop_instruction and was_already_stopped and not safety_escalated and not motion_escalated and (distance_m is None or distance_m >= 0.40):
            if elapsed < COOLDOWNS["REPEAT"]:
                return {
                    "action": "SUPPRESS",
                    "instruction": instruction,
                    "priority": priority,
                    "reason": "User already stopped; continuous STOP suppressed",
                    "bypass_cooldown": False,
                }

        # Determine required cooldown window for candidate
        if priority == 5:
            required_cooldown = COOLDOWNS["EMERGENCY"]
        elif priority == 4 or is_approaching:
            required_cooldown = COOLDOWNS["APPROACHING"]
        elif "CAUTION" in str(safety_alert).upper() or "DANGER" in str(safety_alert).upper():
            required_cooldown = COOLDOWNS["SAFETY"]
        else:
            required_cooldown = COOLDOWNS["NORMAL"]

        # REPEAT check for persistent condition
        if is_same_text and not bypass_cooldown:
            if elapsed >= COOLDOWNS["REPEAT"]:
                if self.is_speaking:
                    if priority > self.last_priority:
                        return {
                            "action": "INTERRUPT",
                            "instruction": instruction,
                            "priority": priority,
                            "reason": f"Priority {priority} interrupts Priority {self.last_priority} speech",
                            "bypass_cooldown": False,
                        }
                    return {
                        "action": "SUPPRESS",
                        "instruction": instruction,
                        "priority": priority,
                        "reason": "Speech active; repeat suppressed",
                        "bypass_cooldown": False,
                    }
                return {
                    "action": "REPEAT",
                    "instruction": instruction,
                    "priority": priority,
                    "reason": f"Persistent condition repeated after {elapsed}ms",
                    "bypass_cooldown": False,
                }
            return {
                "action": "SUPPRESS",
                "instruction": instruction,
                "priority": priority,
                "reason": f"Duplicate instruction within repeat window ({elapsed}ms < {COOLDOWNS['REPEAT']}ms)",
                "bypass_cooldown": False,
            }

        # COOLDOWN check
        if not bypass_cooldown and self.last_spoken_at > 0 and elapsed < required_cooldown:
            return {
                "action": "SUPPRESS",
                "instruction": instruction,
                "priority": priority,
                "reason": f"Cooldown active ({elapsed}ms < {required_cooldown}ms)",
                "bypass_cooldown": False,
            }

        # 5. SPEECH LOCK & QUEUE (when speech is currently in progress)
        if self.is_speaking:
            if priority > self.last_priority:
                return {
                    "action": "INTERRUPT",
                    "instruction": instruction,
                    "priority": priority,
                    "reason": f"Higher priority ({priority} > {self.last_priority}) interrupts active speech",
                    "bypass_cooldown": bypass_cooldown,
                }

            if priority >= 2:
                if self.queued_instruction is None:
                    return {
                        "action": "QUEUE",
                        "instruction": instruction,
                        "priority": priority,
                        "reason": f"Queued during active speech (Priority {priority})",
                        "bypass_cooldown": bypass_cooldown,
                        "queued": True,
                    }
                elif priority > self.queued_instruction.get("priority", 0):
                    return {
                        "action": "QUEUE",
                        "instruction": instruction,
                        "priority": priority,
                        "reason": "Replaces lower priority queued instruction",
                        "bypass_cooldown": bypass_cooldown,
                        "queued": True,
                    }
                else:
                    return {
                        "action": "SUPPRESS",
                        "instruction": instruction,
                        "priority": priority,
                        "reason": "Queue full with equal or higher priority",
                        "bypass_cooldown": bypass_cooldown,
                        "queued": False,
                    }

            return {
                "action": "SUPPRESS",
                "instruction": instruction,
                "priority": priority,
                "reason": "Priority 1 instruction suppressed while speaking",
                "bypass_cooldown": bypass_cooldown,
            }

        # 6. SPEAK IMMEDIATELY
        return {
            "action": "SPEAK",
            "instruction": instruction,
            "priority": priority,
            "reason": "State change bypass allows immediate speech" if bypass_cooldown else "Standard speech trigger",
            "bypass_cooldown": bypass_cooldown,
        }

    def handle_response(self, response: Dict[str, Any], now: int = 0) -> Dict[str, Any]:
        decision = self.evaluate(response, now)

        obstacle = response.get("primary_obstacle") or {}
        threat = response.get("predictive_threat") or {}
        primary_threat = threat or obstacle or {}

        if decision["action"] in ("SPEAK", "INTERRUPT", "REPEAT"):
            if decision["priority"] == 5:
                self.haptic_triggered = True
                self.queued_instruction = None

            self.last_spoken_text = decision["instruction"]
            self.last_spoken_at = now
            self.last_priority = decision["priority"]
            self.is_speaking = True
        elif decision["action"] == "QUEUE":
            step_guidance = response.get("step_guidance") or {}
            nav_raw = response.get("navigation")
            act = step_guidance.get("action") or (nav_raw.get("action") if isinstance(nav_raw, dict) else (nav_raw.get("direction") if isinstance(nav_raw, dict) else nav_raw)) or "NONE"
            norm_act = str(act).upper()
            if norm_act in ("WALK_FORWARD", "GO_FORWARD", "FORWARD", "CONTINUE_FORWARD"):
                norm_act = "MOVE_FORWARD"
            elif norm_act in ("WALK_BACK", "GO_BACK", "BACK"):
                norm_act = "MOVE_BACK"
            elif norm_act in ("MOVE_LEFT",):
                norm_act = "TURN_LEFT"
            elif norm_act in ("MOVE_RIGHT",):
                norm_act = "TURN_RIGHT"
            self.queued_instruction = {
                "text": decision["instruction"],
                "priority": decision["priority"],
                "action": norm_act,
                "timestamp": now,
            }

        # Update telemetry
        prev_safety = self.last_safety_alert
        emergency_stop = response.get("emergency_stop") is True or response.get("emergency_alert") is True
        safety_alert = response.get("safety_alert") or ("EMERGENCY_STOP" if emergency_stop else response.get("safety_level", "SAFE"))
        self.last_safety_alert = str(safety_alert)
        safety_escalated = get_safety_rank(self.last_safety_alert) > get_safety_rank(prev_safety)

        threat_motion = threat.get("motion_state") or obstacle.get("motion_state") or obstacle.get("motion_trend") or ("APPROACHING" if response.get("approaching") else "STATIONARY")
        self.last_motion_trend = str(threat_motion)

        if primary_threat:
            dist = primary_threat.get("distance_m", primary_threat.get("estimated_distance_m"))
            self.last_distance_category = primary_threat.get("distance_category") or ("VERY_CLOSE" if isinstance(dist, (int, float)) and dist < 0.8 else "FAR")
            self.last_threat_track_id = primary_threat.get("track_id") or primary_threat.get("obstacle_id")

        step_guidance = response.get("step_guidance") or {}
        nav_raw = response.get("navigation")
        act = step_guidance.get("action") or (nav_raw.get("action") if isinstance(nav_raw, dict) else (nav_raw.get("direction") if isinstance(nav_raw, dict) else nav_raw)) or "NONE"
        norm_act = str(act).upper()
        if norm_act in ("WALK_FORWARD", "GO_FORWARD", "FORWARD", "CONTINUE_FORWARD"):
            norm_act = "MOVE_FORWARD"
        elif norm_act in ("WALK_BACK", "GO_BACK", "BACK"):
            norm_act = "MOVE_BACK"
        elif norm_act in ("MOVE_LEFT",):
            norm_act = "TURN_LEFT"
        elif norm_act in ("MOVE_RIGHT",):
            norm_act = "TURN_RIGHT"
        self.last_action = norm_act

        if "steps" in step_guidance and isinstance(step_guidance["steps"], (int, float)):
            self.last_steps = int(step_guidance["steps"])
        elif isinstance(nav_raw, dict) and "movement_steps" in nav_raw and isinstance(nav_raw["movement_steps"], (int, float)):
            self.last_steps = int(nav_raw["movement_steps"])

        # Discard stale queued movement instructions on state change
        if decision.get("bypass_cooldown") and self.queued_instruction:
            is_queued_movement = self.queued_instruction.get("action") in ("MOVE_FORWARD", "MOVE_BACK")
            if is_queued_movement and (norm_act != self.queued_instruction.get("action") or decision.get("priority", 0) >= 3 or safety_escalated):
                self.queued_instruction = None

        return decision

    def on_speech_done(self, now: int = 0) -> Optional[Dict[str, Any]]:
        self.is_speaking = False
        if not self.queued_instruction:
            return None
        queued = self.queued_instruction
        if now - queued["timestamp"] > 3500:
            self.queued_instruction = None
            return None
        self.queued_instruction = None
        self.last_spoken_text = queued["text"]
        self.last_spoken_at = now
        self.last_priority = queued["priority"]
        self.is_speaking = True
        return queued


# =====================================================================
# UNIT TEST SUITE (54 Comprehensive Scenarios)
# =====================================================================

class TestPhase12VoiceGuidance(unittest.TestCase):
    def setUp(self):
        self.manager = VoiceDecisionManagerPy()

    # -------------------------------------------------------------
    # 1. Text Normalization (Scenarios 1-4)
    # -------------------------------------------------------------
    def test_01_normalization_case(self):
        self.assertEqual(normalize_instruction("Turn left."), "turn left")
        self.assertEqual(normalize_instruction("TURN LEFT"), "turn left")
        self.assertEqual(normalize_instruction("Turn Left"), "turn left")
        self.assertEqual(normalize_instruction("turn left"), "turn left")

    def test_02_normalization_punctuation(self):
        self.assertEqual(normalize_instruction("Turn left!"), "turn left")
        self.assertEqual(normalize_instruction("Turn left?"), "turn left")
        self.assertEqual(normalize_instruction("Turn left, please"), "turn left please")
        self.assertEqual(normalize_instruction("Stop immediately."), "stop immediately")

    def test_03_normalization_whitespace(self):
        self.assertEqual(normalize_instruction("   Move    forward.   "), "move forward")
        self.assertEqual(normalize_instruction("\tMove forward\n"), "move forward")

    def test_04_normalization_step_instruction(self):
        self.assertEqual(
            normalize_instruction("Move forward three steps."),
            normalize_instruction("move forward three steps")
        )

    # -------------------------------------------------------------
    # 2. Step Formatting (Scenarios 5-7)
    # -------------------------------------------------------------
    def test_05_format_step_words_1_to_10(self):
        expected = {
            1: "one step",
            2: "two steps",
            3: "three steps",
            4: "four steps",
            5: "five steps",
            6: "six steps",
            7: "seven steps",
            8: "eight steps",
            9: "nine steps",
            10: "ten steps",
        }
        for n, exp in expected.items():
            self.assertEqual(format_step_words(n), exp)

    def test_06_format_step_words_above_10(self):
        self.assertEqual(format_step_words(12), "12 steps")
        self.assertEqual(format_step_words(15), "15 steps")

    def test_07_format_step_words_zero_and_none(self):
        resp_none = {"success": True, "navigation": {"action": "MOVE_FORWARD"}}
        dec_none = self.manager.evaluate(resp_none, now=1000)
        self.assertEqual(dec_none["instruction"], "Move forward.")

        resp_zero = {"success": True, "navigation": {"action": "MOVE_BACK", "movement_steps": 0}}
        dec_zero = self.manager.evaluate(resp_zero, now=1000)
        self.assertEqual(dec_zero["instruction"], "Move back.")

    # -------------------------------------------------------------
    # 3. Priority 5: EMERGENCY (Scenarios 8-12)
    # -------------------------------------------------------------
    def test_08_priority_5_emergency_stop_flag(self):
        resp = {"success": True, "emergency_stop": True}
        dec = self.manager.evaluate(resp, now=1000)
        self.assertEqual(dec["priority"], 5)
        self.assertEqual(dec["instruction"], "Stop immediately.")

    def test_09_priority_5_safety_alert_emergency_stop(self):
        resp = {"success": True, "safety_alert": "EMERGENCY_STOP"}
        dec = self.manager.evaluate(resp, now=1000)
        self.assertEqual(dec["priority"], 5)
        self.assertEqual(dec["instruction"], "Stop immediately.")

    def test_10_priority_5_safety_level_emergency(self):
        resp = {"success": True, "safety_level": "EMERGENCY"}
        dec = self.manager.evaluate(resp, now=1000)
        self.assertEqual(dec["priority"], 5)
        self.assertEqual(dec["instruction"], "Stop immediately.")

    def test_11_priority_5_action_stop_close_distance(self):
        resp = {
            "success": True,
            "navigation": {"action": "STOP", "distance_m": 0.45},
            "primary_obstacle": {"distance_m": 0.45, "class_name": "wall"}
        }
        dec = self.manager.evaluate(resp, now=1000)
        self.assertEqual(dec["priority"], 5)
        self.assertEqual(dec["instruction"], "Stop immediately.")

    def test_12_priority_5_rapid_approach_critical_distance(self):
        resp = {
            "success": True,
            "predictive_threat": {
                "motion_state": "RAPIDLY_APPROACHING",
                "distance_m": 0.65,
                "class_name": "person",
                "is_approaching": True,
            }
        }
        dec = self.manager.evaluate(resp, now=1000)
        self.assertEqual(dec["priority"], 5)
        self.assertEqual(dec["instruction"], "Stop immediately.")

    # -------------------------------------------------------------
    # 4. Priority 4: RAPID DANGER / RAPIDLY APPROACHING (Scenarios 13-16)
    # -------------------------------------------------------------
    def test_13_priority_4_rapid_approaching_person(self):
        resp = {
            "success": True,
            "predictive_threat": {
                "motion_state": "RAPIDLY_APPROACHING",
                "distance_m": 1.4,
                "class_name": "person",
                "is_approaching": True,
            }
        }
        dec = self.manager.evaluate(resp, now=1000)
        self.assertEqual(dec["priority"], 4)
        self.assertEqual(dec["instruction"], "Warning. Person approaching.")

    def test_14_priority_4_rapid_approaching_object(self):
        resp = {
            "success": True,
            "predictive_threat": {
                "motion_state": "RAPIDLY_APPROACHING",
                "distance_m": 1.8,
                "class_name": "bicycle",
                "is_approaching": True,
            }
        }
        dec = self.manager.evaluate(resp, now=1000)
        self.assertEqual(dec["priority"], 4)
        self.assertEqual(dec["instruction"], "Warning. Object approaching.")

    def test_15_priority_4_warning_rapid_approach_alert(self):
        resp = {
            "success": True,
            "safety_alert": "WARNING_RAPID_APPROACH",
            "primary_obstacle": {"class_name": "person", "is_approaching": True}
        }
        dec = self.manager.evaluate(resp, now=1000)
        self.assertEqual(dec["priority"], 4)
        self.assertEqual(dec["instruction"], "Warning. Person approaching.")

    def test_16_priority_4_danger_stop_action(self):
        resp = {
            "success": True,
            "safety_level": "DANGER",
            "navigation": {"action": "STOP"}
        }
        dec = self.manager.evaluate(resp, now=1000)
        self.assertEqual(dec["priority"], 4)
        self.assertEqual(dec["instruction"], "Stop.")

    # -------------------------------------------------------------
    # 5. Priority 3: TURN (Scenarios 17-20)
    # -------------------------------------------------------------
    def test_17_priority_3_turn_left(self):
        resp = {"success": True, "navigation": {"action": "TURN_LEFT"}}
        dec = self.manager.evaluate(resp, now=1000)
        self.assertEqual(dec["priority"], 3)
        self.assertEqual(dec["instruction"], "Turn left.")

    def test_18_priority_3_turn_right(self):
        resp = {"success": True, "navigation": {"action": "TURN_RIGHT"}}
        dec = self.manager.evaluate(resp, now=1000)
        self.assertEqual(dec["priority"], 3)
        self.assertEqual(dec["instruction"], "Turn right.")

    def test_19_priority_3_turn_strictly_no_steps(self):
        resp = {
            "success": True,
            "navigation": {"action": "TURN_LEFT", "movement_steps": 4},
            "step_guidance": {"action": "TURN_LEFT", "steps": 4}
        }
        dec = self.manager.evaluate(resp, now=1000)
        self.assertEqual(dec["priority"], 3)
        self.assertEqual(dec["instruction"], "Turn left.")
        self.assertNotIn("step", dec["instruction"].lower())

    def test_20_priority_3_turn_right_strictly_no_steps(self):
        resp = {
            "success": True,
            "navigation": {"action": "TURN_RIGHT", "movement_steps": 2},
            "step_guidance": {"action": "TURN_RIGHT", "steps": 2}
        }
        dec = self.manager.evaluate(resp, now=1000)
        self.assertEqual(dec["priority"], 3)
        self.assertEqual(dec["instruction"], "Turn right.")
        self.assertNotIn("step", dec["instruction"].lower())

    # -------------------------------------------------------------
    # 6. Priority 2: MOVEMENT / WALKING STEPS (Scenarios 21-25)
    # -------------------------------------------------------------
    def test_21_priority_2_move_forward_with_steps(self):
        resp = {
            "success": True,
            "step_guidance": {"action": "WALK_FORWARD", "steps": 3}
        }
        dec = self.manager.evaluate(resp, now=1000)
        self.assertEqual(dec["priority"], 2)
        self.assertEqual(dec["instruction"], "Move forward three steps.")

    def test_22_priority_2_move_back_with_steps(self):
        resp = {
            "success": True,
            "step_guidance": {"action": "WALK_BACK", "steps": 2}
        }
        dec = self.manager.evaluate(resp, now=1000)
        self.assertEqual(dec["priority"], 2)
        self.assertEqual(dec["instruction"], "Move back two steps.")

    def test_23_priority_2_move_forward_one_step_singular(self):
        resp = {
            "success": True,
            "navigation": {"action": "MOVE_FORWARD", "movement_steps": 1}
        }
        dec = self.manager.evaluate(resp, now=1000)
        self.assertEqual(dec["priority"], 2)
        self.assertEqual(dec["instruction"], "Move forward one step.")

    def test_24_priority_2_move_forward_no_steps(self):
        resp = {
            "success": True,
            "navigation": {"action": "MOVE_FORWARD"}
        }
        dec = self.manager.evaluate(resp, now=1000)
        self.assertEqual(dec["priority"], 2)
        self.assertEqual(dec["instruction"], "Move forward.")

    def test_25_priority_2_move_back_no_steps(self):
        resp = {
            "success": True,
            "navigation": {"action": "MOVE_BACK"}
        }
        dec = self.manager.evaluate(resp, now=1000)
        self.assertEqual(dec["priority"], 2)
        self.assertEqual(dec["instruction"], "Move back.")

    # -------------------------------------------------------------
    # 7. Priority 1: GENERAL AWARENESS / NORMAL APPROACHING (Scenarios 26-28)
    # -------------------------------------------------------------
    def test_26_priority_1_normal_approaching_person(self):
        resp = {
            "success": True,
            "predictive_threat": {
                "motion_state": "APPROACHING",
                "distance_m": 2.5,
                "class_name": "person",
                "is_approaching": True,
            }
        }
        dec = self.manager.evaluate(resp, now=1000)
        self.assertEqual(dec["priority"], 1)
        self.assertEqual(dec["instruction"], "Warning. Person approaching.")

    def test_27_priority_1_normal_approaching_object(self):
        resp = {
            "success": True,
            "predictive_threat": {
                "motion_state": "APPROACHING",
                "distance_m": 2.8,
                "class_name": "chair",
                "is_approaching": True,
            }
        }
        dec = self.manager.evaluate(resp, now=1000)
        self.assertEqual(dec["priority"], 1)
        self.assertEqual(dec["instruction"], "Warning. Object approaching.")

    def test_28_priority_1_path_clear(self):
        resp = {"success": True, "objects": []}
        dec = self.manager.evaluate(resp, now=1000)
        self.assertEqual(dec["priority"], 1)
        self.assertEqual(dec["instruction"], "Path clear.")

    # -------------------------------------------------------------
    # 8. Cooldowns & Repeat (Scenarios 29-38)
    # -------------------------------------------------------------
    def test_29_cooldown_normal_suppress(self):
        resp1 = {"success": True, "step_guidance": {"action": "WALK_FORWARD", "steps": 3}}
        self.manager.handle_response(resp1, now=1000)
        self.manager.is_speaking = False

        # Different instruction arrived at 2500ms (elapsed 1500ms < 2500ms, step delta 1 < 2)
        resp2 = {"success": True, "step_guidance": {"action": "WALK_FORWARD", "steps": 2}}
        dec = self.manager.evaluate(resp2, now=2500)
        self.assertEqual(dec["action"], "SUPPRESS")

    def test_30_cooldown_normal_allow(self):
        resp1 = {"success": True, "step_guidance": {"action": "WALK_FORWARD", "steps": 3}}
        self.manager.handle_response(resp1, now=1000)
        self.manager.is_speaking = False

        # Arrived at 3600ms (elapsed 2600ms >= 2500ms)
        resp2 = {"success": True, "step_guidance": {"action": "WALK_FORWARD", "steps": 2}}
        dec = self.manager.evaluate(resp2, now=3600)
        self.assertEqual(dec["action"], "SPEAK")

    def test_31_cooldown_approaching_suppress(self):
        resp1 = {
            "success": True,
            "predictive_threat": {"motion_state": "APPROACHING", "class_name": "person", "is_approaching": True}
        }
        self.manager.handle_response(resp1, now=1000)

        # Re-eval at 3000ms (elapsed 2000ms < 2500ms)
        dec = self.manager.evaluate(resp1, now=3000)
        self.assertEqual(dec["action"], "SUPPRESS")

    def test_32_cooldown_approaching_allow(self):
        resp1 = {
            "success": True,
            "predictive_threat": {"motion_state": "APPROACHING", "class_name": "person", "is_approaching": True}
        }
        self.manager.handle_response(resp1, now=1000)
        self.manager.is_speaking = False

        # Re-eval at 8100ms (elapsed 7100ms >= 7000ms repeat window)
        dec = self.manager.evaluate(resp1, now=8100)
        self.assertEqual(dec["action"], "REPEAT")

    def test_33_cooldown_safety_suppress(self):
        resp1 = {"success": True, "safety_level": "CAUTION", "voice_instruction": "Caution ahead."}
        self.manager.handle_response(resp1, now=1000)

        # Different instruction arrived at 3000ms (elapsed 2000ms < 3000ms)
        resp2 = {"success": True, "safety_level": "CAUTION", "voice_instruction": "Obstacle near."}
        dec = self.manager.evaluate(resp2, now=3000)
        self.assertEqual(dec["action"], "SUPPRESS")

    def test_34_cooldown_safety_allow(self):
        resp1 = {"success": True, "safety_level": "CAUTION", "voice_instruction": "Caution ahead."}
        self.manager.handle_response(resp1, now=1000)
        self.manager.is_speaking = False

        # Arrived at 4100ms (elapsed 3100ms >= 3000ms)
        resp2 = {"success": True, "safety_level": "CAUTION", "voice_instruction": "Obstacle near."}
        dec = self.manager.evaluate(resp2, now=4100)
        self.assertEqual(dec["action"], "SPEAK")

    def test_35_cooldown_repeat_persistent_condition(self):
        resp = {"success": True, "navigation": {"action": "TURN_LEFT"}}
        self.manager.handle_response(resp, now=1000)
        self.manager.is_speaking = False

        # 7000ms later -> REPEAT
        dec = self.manager.evaluate(resp, now=8000)
        self.assertEqual(dec["action"], "REPEAT")
        self.assertEqual(dec["instruction"], "Turn left.")

    def test_36_cooldown_repeat_suppressed_before_window(self):
        resp = {"success": True, "navigation": {"action": "TURN_LEFT"}}
        self.manager.handle_response(resp, now=1000)

        # 6500ms later (< 7000ms) -> SUPPRESS
        dec = self.manager.evaluate(resp, now=7500)
        self.assertEqual(dec["action"], "SUPPRESS")

    def test_37_emergency_first_trigger_zero_delay(self):
        resp1 = {"success": True, "navigation": {"action": "MOVE_FORWARD"}}
        self.manager.handle_response(resp1, now=1000)

        # Emergency 100ms later -> immediate preemption with ZERO cooldown
        resp2 = {"success": True, "emergency_stop": True}
        dec = self.manager.evaluate(resp2, now=1100)
        self.assertEqual(dec["action"], "INTERRUPT")
        self.assertEqual(dec["priority"], 5)
        self.assertEqual(dec["instruction"], "Stop immediately.")

    def test_38_emergency_retrigger_cooldown(self):
        resp = {"success": True, "emergency_stop": True}
        self.manager.handle_response(resp, now=1000)
        self.manager.is_speaking = False

        # Retrigger at 500ms (< 1000ms) -> SUPPRESS
        dec1 = self.manager.evaluate(resp, now=1500)
        self.assertEqual(dec1["action"], "SUPPRESS")

        # Retrigger at 8100ms (>= 7000ms repeat) -> REPEAT
        dec2 = self.manager.evaluate(resp, now=9100)
        self.assertEqual(dec2["action"], "REPEAT")

    # -------------------------------------------------------------
    # 9. Significant State Change Bypass (Scenarios 39-46)
    # -------------------------------------------------------------
    def test_39_bypass_action_change(self):
        resp1 = {"success": True, "navigation": {"action": "MOVE_FORWARD"}}
        self.manager.handle_response(resp1, now=1000)
        self.manager.is_speaking = False

        # Action changes to TURN_LEFT at 1500ms (< 2500ms) -> BYPASS
        resp2 = {"success": True, "navigation": {"action": "TURN_LEFT"}}
        dec = self.manager.evaluate(resp2, now=1500)
        self.assertTrue(dec["bypass_cooldown"])
        self.assertEqual(dec["action"], "SPEAK")
        self.assertEqual(dec["instruction"], "Turn left.")

    def test_40_bypass_safety_escalation(self):
        resp1 = {"success": True, "safety_level": "SAFE", "navigation": {"action": "MOVE_FORWARD"}}
        self.manager.handle_response(resp1, now=1000)
        self.manager.is_speaking = False

        # Safety escalates to WARNING_RAPID_APPROACH at 1300ms -> BYPASS
        resp2 = {
            "success": True,
            "safety_alert": "WARNING_RAPID_APPROACH",
            "primary_obstacle": {"class_name": "person", "is_approaching": True}
        }
        dec = self.manager.evaluate(resp2, now=1300)
        self.assertTrue(dec["bypass_cooldown"])
        self.assertEqual(dec["action"], "SPEAK")

    def test_41_bypass_motion_escalation_stationary_to_approaching(self):
        resp1 = {
            "success": True,
            "primary_obstacle": {"motion_state": "STATIONARY", "class_name": "person"}
        }
        self.manager.handle_response(resp1, now=1000)
        self.manager.is_speaking = False

        # Escalates to APPROACHING at 1400ms -> BYPASS
        resp2 = {
            "success": True,
            "predictive_threat": {"motion_state": "APPROACHING", "class_name": "person", "is_approaching": True}
        }
        dec = self.manager.evaluate(resp2, now=1400)
        self.assertTrue(dec["bypass_cooldown"])
        self.assertEqual(dec["action"], "SPEAK")

    def test_42_bypass_motion_escalation_approaching_to_rapid(self):
        resp1 = {
            "success": True,
            "predictive_threat": {"motion_state": "APPROACHING", "class_name": "person", "is_approaching": True}
        }
        self.manager.handle_response(resp1, now=1000)
        self.manager.is_speaking = False

        # Escalates to RAPIDLY_APPROACHING at 1400ms -> BYPASS
        resp2 = {
            "success": True,
            "predictive_threat": {"motion_state": "RAPIDLY_APPROACHING", "class_name": "person", "is_approaching": True}
        }
        dec = self.manager.evaluate(resp2, now=1400)
        self.assertTrue(dec["bypass_cooldown"])
        self.assertEqual(dec["action"], "SPEAK")
        self.assertEqual(dec["priority"], 4)

    def test_43_bypass_distance_category_escalation(self):
        resp1 = {
            "success": True,
            "primary_obstacle": {"distance_category": "FAR", "distance_m": 3.5, "class_name": "chair"}
        }
        self.manager.handle_response(resp1, now=1000)
        self.manager.is_speaking = False

        # Escalates to CLOSE at 1400ms -> BYPASS
        resp2 = {
            "success": True,
            "primary_obstacle": {"distance_category": "CLOSE", "distance_m": 1.2, "class_name": "chair"},
            "navigation": {"action": "TURN_RIGHT"}
        }
        dec = self.manager.evaluate(resp2, now=1400)
        self.assertTrue(dec["bypass_cooldown"])
        self.assertEqual(dec["action"], "SPEAK")

    def test_44_bypass_threat_track_change(self):
        resp1 = {
            "success": True,
            "primary_obstacle": {"track_id": "track_101", "class_name": "person", "distance_m": 2.0},
            "navigation": {"action": "MOVE_FORWARD"}
        }
        self.manager.handle_response(resp1, now=1000)
        self.manager.is_speaking = False

        # Different threat track becomes primary at 1400ms -> BYPASS
        resp2 = {
            "success": True,
            "primary_obstacle": {"track_id": "track_202", "class_name": "person", "distance_m": 1.5},
            "navigation": {"action": "MOVE_FORWARD"}
        }
        dec = self.manager.evaluate(resp2, now=1400)
        self.assertTrue(dec["bypass_cooldown"])
        self.assertEqual(dec["action"], "SPEAK")

    def test_45_bypass_step_count_delta_ge_2(self):
        resp1 = {
            "success": True,
            "step_guidance": {"action": "WALK_FORWARD", "steps": 5}
        }
        self.manager.handle_response(resp1, now=1000)
        self.manager.is_speaking = False

        # Steps drop from 5 to 3 (delta = 2 >= 2) at 1500ms (< 2500ms) -> BYPASS
        resp2 = {
            "success": True,
            "step_guidance": {"action": "WALK_FORWARD", "steps": 3}
        }
        dec = self.manager.evaluate(resp2, now=1500)
        self.assertTrue(dec["bypass_cooldown"])
        self.assertEqual(dec["action"], "SPEAK")
        self.assertEqual(dec["instruction"], "Move forward three steps.")

    def test_46_no_bypass_step_count_delta_lt_2(self):
        resp1 = {
            "success": True,
            "step_guidance": {"action": "WALK_FORWARD", "steps": 5}
        }
        self.manager.handle_response(resp1, now=1000)
        self.manager.is_speaking = False

        # Steps drop from 5 to 4 (delta = 1 < 2) at 1500ms (< 2500ms) -> NO BYPASS (SUPPRESS)
        resp2 = {
            "success": True,
            "step_guidance": {"action": "WALK_FORWARD", "steps": 4}
        }
        dec = self.manager.evaluate(resp2, now=1500)
        self.assertFalse(dec["bypass_cooldown"])
        self.assertEqual(dec["action"], "SUPPRESS")

    # -------------------------------------------------------------
    # 10. Speech Lock & Queue Management (Scenarios 47-53)
    # -------------------------------------------------------------
    def test_47_speech_lock_higher_priority_interrupts(self):
        # Priority 2 speaking
        resp1 = {"success": True, "step_guidance": {"action": "WALK_FORWARD", "steps": 3}}
        self.manager.handle_response(resp1, now=1000)
        self.assertTrue(self.manager.is_speaking)

        # Priority 4 arrives while speaking -> INTERRUPTS
        resp2 = {
            "success": True,
            "predictive_threat": {"motion_state": "RAPIDLY_APPROACHING", "class_name": "person", "is_approaching": True}
        }
        dec = self.manager.evaluate(resp2, now=1200)
        self.assertEqual(dec["action"], "INTERRUPT")
        self.assertEqual(dec["priority"], 4)

    def test_48_speech_lock_emergency_interrupts_all(self):
        # Priority 4 speaking
        resp1 = {
            "success": True,
            "predictive_threat": {"motion_state": "RAPIDLY_APPROACHING", "class_name": "person", "is_approaching": True}
        }
        self.manager.handle_response(resp1, now=1000)
        self.assertTrue(self.manager.is_speaking)

        # Priority 5 arrives -> INTERRUPTS and clears queue
        resp2 = {"success": True, "emergency_stop": True}
        dec = self.manager.handle_response(resp2, now=1200)
        self.assertEqual(dec["action"], "INTERRUPT")
        self.assertEqual(dec["priority"], 5)
        self.assertIsNone(self.manager.queued_instruction)
        self.assertTrue(self.manager.haptic_triggered)

    def test_49_queue_management_priority_ge_2(self):
        # Priority 3 speaking
        resp1 = {"success": True, "navigation": {"action": "TURN_LEFT"}}
        self.manager.handle_response(resp1, now=1000)
        self.assertTrue(self.manager.is_speaking)

        # Priority 2 arrives -> cannot interrupt Priority 3, but eligible for QUEUE
        resp2 = {"success": True, "step_guidance": {"action": "WALK_FORWARD", "steps": 3}}
        dec = self.manager.handle_response(resp2, now=1300)
        self.assertEqual(dec["action"], "QUEUE")
        self.assertIsNotNone(self.manager.queued_instruction)
        self.assertEqual(self.manager.queued_instruction["text"], "Move forward three steps.")

    def test_50_queue_management_priority_1_dropped(self):
        # Priority 2 speaking
        resp1 = {"success": True, "step_guidance": {"action": "WALK_FORWARD", "steps": 3}}
        self.manager.handle_response(resp1, now=1000)
        self.assertTrue(self.manager.is_speaking)

        # Priority 1 arrives -> dropped/suppressed (no queue for priority 1)
        resp2 = {"success": True, "objects": []}
        dec = self.manager.evaluate(resp2, now=1300)
        self.assertEqual(dec["action"], "SUPPRESS")
        self.assertIsNone(self.manager.queued_instruction)

    def test_51_queue_replacement_higher_priority(self):
        # Priority 3 speaking
        resp1 = {"success": True, "navigation": {"action": "TURN_LEFT"}}
        self.manager.handle_response(resp1, now=1000)

        # Priority 2 enters queue
        resp2 = {"success": True, "step_guidance": {"action": "WALK_FORWARD", "steps": 2}}
        self.manager.handle_response(resp2, now=1200)
        self.assertEqual(self.manager.queued_instruction["priority"], 2)

        # Priority 3 arrives while still speaking -> replaces Priority 2 in queue
        resp3 = {"success": True, "navigation": {"action": "TURN_RIGHT"}}
        dec = self.manager.handle_response(resp3, now=1400)
        self.assertEqual(dec["action"], "QUEUE")
        self.assertEqual(self.manager.queued_instruction["priority"], 3)
        self.assertEqual(self.manager.queued_instruction["text"], "Turn right.")

    def test_52_queue_reject_lower_priority(self):
        # Priority 4 speaking
        resp1 = {
            "success": True,
            "predictive_threat": {"motion_state": "RAPIDLY_APPROACHING", "class_name": "obstacle", "is_approaching": True}
        }
        self.manager.handle_response(resp1, now=1000)

        # Queue gets Priority 3
        resp2 = {"success": True, "navigation": {"action": "TURN_LEFT"}}
        self.manager.handle_response(resp2, now=1200)
        self.assertEqual(self.manager.queued_instruction["priority"], 3)

        # Priority 2 arrives -> cannot replace Priority 3 in queue
        resp3 = {"success": True, "step_guidance": {"action": "WALK_FORWARD", "steps": 2}}
        dec = self.manager.handle_response(resp3, now=1400)
        self.assertEqual(dec["action"], "SUPPRESS")
        self.assertEqual(self.manager.queued_instruction["priority"], 3)

    def test_53_stale_queue_eviction_on_state_change(self):
        # Priority 3 speaking
        resp1 = {"success": True, "navigation": {"action": "TURN_LEFT"}}
        self.manager.handle_response(resp1, now=1000)

        # Movement instruction queued
        resp2 = {"success": True, "step_guidance": {"action": "WALK_FORWARD", "steps": 3}}
        self.manager.handle_response(resp2, now=1200)
        self.assertIsNotNone(self.manager.queued_instruction)

        # Significant state change occurs (safety escalates to DANGER)
        resp3 = {
            "success": True,
            "safety_level": "DANGER",
            "predictive_threat": {"motion_state": "RAPIDLY_APPROACHING", "class_name": "person", "is_approaching": True}
        }
        dec = self.manager.handle_response(resp3, now=1400)
        self.assertEqual(dec["action"], "INTERRUPT")
        # Queued movement instruction must be evicted
        self.assertIsNone(self.manager.queued_instruction)

    # -------------------------------------------------------------
    # 11. End-to-End Walk Scenario (Scenario 54)
    # -------------------------------------------------------------
    def test_54_end_to_end_simulated_walk(self):
        """
        Simulate a continuous 10-frame assistive navigation session:
        Frame 1: Clear corridor -> "Path clear." (Priority 1, SPEAK)
        Frame 2: 100ms later, clear corridor -> SUPPRESS (duplicate/cooldown)
        Frame 3: Person approaching -> "Warning. Person approaching." (Priority 1, BYPASS motion, SPEAK)
        Frame 4: Person accelerates -> "Warning. Person approaching." (Priority 4, BYPASS rapid, INTERRUPT)
        Frame 5: Person very close (< 0.8m) -> "Stop immediately." (Priority 5, INTERRUPT, haptics)
        Frame 6: Path clear to right -> "Turn right." (Priority 3, BYPASS action change, SPEAK)
        Frame 7: Forward corridor, 4 steps -> "Move forward four steps." (Priority 2, BYPASS action change, SPEAK)
        Frame 8: Step count drops to 2 steps (delta = 2) -> "Move forward two steps." (Priority 2, BYPASS step delta, SPEAK)
        Frame 9: Step count drops to 1 step (delta = 1) at t=500ms -> SUPPRESS (cooldown active, delta < 2)
        Frame 10: Sudden obstacle at 0.4m -> "Stop immediately." (Priority 5, BYPASS emergency, INTERRUPT)
        """
        # Frame 1 (t=1000): Clear scene
        f1 = {"success": True, "objects": []}
        d1 = self.manager.handle_response(f1, now=1000)
        self.assertEqual(d1["action"], "SPEAK")
        self.assertEqual(d1["instruction"], "Path clear.")
        self.assertEqual(d1["priority"], 1)

        # Frame 2 (t=1100): Clear scene again -> duplicate suppressed
        f2 = {"success": True, "objects": []}
        d2 = self.manager.handle_response(f2, now=1100)
        self.assertEqual(d2["action"], "SUPPRESS")

        self.manager.is_speaking = False

        # Frame 3 (t=2000): Person approaching -> motion escalation bypasses cooldown
        f3 = {
            "success": True,
            "predictive_threat": {"motion_state": "APPROACHING", "class_name": "person", "distance_m": 2.5, "is_approaching": True}
        }
        d3 = self.manager.handle_response(f3, now=2000)
        self.assertEqual(d3["action"], "SPEAK")
        self.assertEqual(d3["instruction"], "Warning. Person approaching.")
        self.assertEqual(d3["priority"], 1)

        # Frame 4 (t=2500): Person accelerates to rapid approach -> interrupts speaking
        f4 = {
            "success": True,
            "predictive_threat": {"motion_state": "RAPIDLY_APPROACHING", "class_name": "person", "distance_m": 1.4, "is_approaching": True}
        }
        d4 = self.manager.handle_response(f4, now=2500)
        self.assertEqual(d4["action"], "INTERRUPT")
        self.assertEqual(d4["instruction"], "Warning. Person approaching.")
        self.assertEqual(d4["priority"], 4)

        # Frame 5 (t=3000): Person reaches critical distance (< 0.8m) -> EMERGENCY STOP
        f5 = {
            "success": True,
            "predictive_threat": {"motion_state": "RAPIDLY_APPROACHING", "class_name": "person", "distance_m": 0.65, "is_approaching": True}
        }
        d5 = self.manager.handle_response(f5, now=3000)
        self.assertEqual(d5["action"], "INTERRUPT")
        self.assertEqual(d5["instruction"], "Stop immediately.")
        self.assertEqual(d5["priority"], 5)
        self.assertTrue(self.manager.haptic_triggered)

        self.manager.is_speaking = False

        # Frame 6 (t=4500): Turn instruction -> action change bypass
        f6 = {
            "success": True,
            "navigation": {"action": "TURN_RIGHT"}
        }
        d6 = self.manager.handle_response(f6, now=4500)
        self.assertEqual(d6["action"], "SPEAK")
        self.assertEqual(d6["instruction"], "Turn right.")
        self.assertEqual(d6["priority"], 3)

        self.manager.is_speaking = False

        # Frame 7 (t=5500): Walk forward 4 steps -> action change bypass
        f7 = {
            "success": True,
            "step_guidance": {"action": "WALK_FORWARD", "steps": 4}
        }
        d7 = self.manager.handle_response(f7, now=5500)
        self.assertEqual(d7["action"], "SPEAK")
        self.assertEqual(d7["instruction"], "Move forward four steps.")
        self.assertEqual(d7["priority"], 2)

        self.manager.is_speaking = False

        # Frame 8 (t=6200): Step count drops from 4 to 2 (delta = 2 >= 2) -> step delta bypass
        f8 = {
            "success": True,
            "step_guidance": {"action": "WALK_FORWARD", "steps": 2}
        }
        d8 = self.manager.handle_response(f8, now=6200)
        self.assertEqual(d8["action"], "SPEAK")
        self.assertEqual(d8["instruction"], "Move forward two steps.")
        self.assertEqual(d8["priority"], 2)

        self.manager.is_speaking = False

        # Frame 9 (t=6700): Step count drops from 2 to 1 (delta = 1 < 2, elapsed 500ms < 2500ms) -> SUPPRESS
        f9 = {
            "success": True,
            "step_guidance": {"action": "WALK_FORWARD", "steps": 1}
        }
        d9 = self.manager.handle_response(f9, now=6700)
        self.assertEqual(d9["action"], "SUPPRESS")

        # Frame 10 (t=7000): Sudden obstacle at 0.4m -> Priority 5 EMERGENCY STOP
        f10 = {
            "success": True,
            "primary_obstacle": {"distance_m": 0.4, "class_name": "barrier"},
            "navigation": {"action": "STOP", "distance_m": 0.4}
        }
        d10 = self.manager.handle_response(f10, now=7000)
        self.assertEqual(d10["action"], "SPEAK")
        self.assertEqual(d10["instruction"], "Stop immediately.")
        self.assertEqual(d10["priority"], 5)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhase12VoiceGuidance)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print(f"\nPhase 12 Voice Guidance Test Summary: Ran {result.testsRun} tests, Errors: {len(result.errors)}, Failures: {len(result.failures)}")
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
