/**
 * SmartVisionAI - Voice Decision Manager & Intelligent Alert Management (Phase 12)
 *
 * Responsibilities:
 * - Centralized mobile-side voice decision layer (SPEAK, SUPPRESS, REPEAT, INTERRUPT, QUEUE, IGNORE)
 * - Strict 5-level priority hierarchy:
 *     5: EMERGENCY ("Stop immediately.")
 *     4: RAPID DANGER / RAPIDLY APPROACHING ("Warning. Person approaching." / "Warning. Object approaching." / "Stop.")
 *     3: TURN ("Turn left." / "Turn right." - NO steps)
 *     2: MOVEMENT / WALKING STEPS ("Move forward three steps." / "Move back two steps." / "Move forward." / "Move back.")
 *     1: GENERAL AWARENESS / NORMAL APPROACHING ("Warning. Person approaching." / "Warning. Object approaching.")
 * - English-only speech output with standard en-US TTS
 * - Emergency speech preemption with zero cooldown on first trigger
 * - Audio & haptic vibration alarm preemption
 * - Normalized instruction comparison (stripping case, punctuation, whitespace)
 * - Configurable cooldown windows and repeat intervals
 * - Significant state change bypass (action change, safety escalation, motion escalation, distance escalation, track ID change, step delta >= 2)
 * - Speech lock preventing overlapping speech and max 1 pending queue with stale movement eviction
 */

import { Vibration } from "react-native";
import * as Speech from "expo-speech";
import {
  SPEECH_COOLDOWN_MS,
  VoiceLanguage,
  getVoiceLanguage,
  depthToEstimatedDistance,
  distanceToSteps,
} from "src/constants/config";
import type {
  SmartVisionResponse,
  DetectionObject,
  PredictiveThreat,
} from "src/types/smartvision";

/* =========================================================
   TYPES & ENUMS
   ========================================================= */

export type VoicePriority = 1 | 2 | 3 | 4 | 5;

export type VoiceSafetyEvent =
  | "CONNECTION_LOST"
  | "CONNECTION_RESTORED"
  | "CAMERA_OCCLUDED";

export type VoiceDecisionAction =
  | "SPEAK"
  | "SUPPRESS"
  | "REPEAT"
  | "INTERRUPT"
  | "QUEUE"
  | "IGNORE";

export interface VoiceDecision {
  action: VoiceDecisionAction;
  instruction: string;
  priority: VoicePriority;
  reason: string;
  bypass_cooldown?: boolean;
  queued?: boolean;
}

export interface QueuedInstruction {
  text: string;
  priority: VoicePriority;
  action: string;
  timestamp: number;
}

export interface VoiceManagerState {
  isSpeaking: boolean;
  lastSpokenText: string;
  lastSpokenAt: number;
  lastPriority: VoicePriority | 0;
  lastSafetyAlert: string;
  lastMotionTrend: string;
  lastDistanceCategory: string;
  lastThreatTrackId: string | number | null;
  lastThreatClass?: string;
  lastAction: string;
  lastSteps: number | null;
  lastOcrText: string;
  lastOcrSpokenAt: number;
  queuedInstruction: QueuedInstruction | null;
}

export interface VoiceOptions {
  priority?: VoicePriority;
  bypassCooldown?: boolean;
  action?: string;
}

export interface AssistiveVoiceInstruction {
  text: string;
  isEmergency: boolean;
  distanceM?: number;
  steps?: number;
  confidence?: "HIGH" | "MEDIUM" | "LOW";
  approaching?: boolean;
  language?: VoiceLanguage;
  priority?: VoicePriority;
}

/* =========================================================
   CONSTANTS & CONFIGURATION
   ========================================================= */

export const COOLDOWNS = {
  NORMAL: 2500,        // 2500ms between standard navigation instructions
  APPROACHING: 2500,   // 2500ms between approaching obstacle warnings
  SAFETY: 3000,        // 3000ms between general safety warnings
  EMERGENCY: 1000,     // 1000ms re-trigger cooldown for emergency
  REPEAT: 7000,        // 7000ms repeat persistent condition
  CONNECTION: 5000,    // 5000ms between repeated connection lost notifications
  OCR: 10000,          // 10000ms de-duplication window for identical OCR text
};

export const EMERGENCY_COOLDOWN_MS = 1000;
export const DEDUPLICATION_WINDOW_MS = 2500;

export const SAFETY_MESSAGES = {
  EMERGENCY_STOP: "Stop immediately.",
  CONNECTION_LOST: "Connection lost. Stop walking.",
  CONNECTION_RESTORED: "Connection restored. Resuming guidance.",
  CAMERA_OCCLUDED: "Camera view obstructed or scene too dark. Please check camera view.",
  PATH_CLEAR: "Path clear.",
  CONTINUE_CAREFULLY: "Continue carefully.",
} as const;

export const LANGUAGE_VOICE_CODES: Record<VoiceLanguage, string> = {
  en: "en-US",
  hi: "en-US", // Fallback to English
  mr: "en-US", // Fallback to English
};

const STEP_WORDS: Record<number, string> = {
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
};

/* =========================================================
   TEXT NORMALIZATION & FORMATTING HELPERS
   ========================================================= */

/**
 * Normalizes instruction text by lowercasing, stripping punctuation,
 * and collapsing whitespace for accurate semantic deduplication.
 */
export function normalizeInstruction(text: string): string {
  if (!text) return "";
  return text
    .toLowerCase()
    .replace(/[.,!?:;\-_]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * Format step numbers 1-10 as lowercase English words with singular/plural.
 */
export function formatStepWords(steps: number): string {
  const word = STEP_WORDS[steps] || String(steps);
  return steps === 1 ? `${word} step` : `${word} steps`;
}

/**
 * Format raw class names into natural English object nouns for spoken announcements.
 */
export function formatObjectName(clsName: string): string {
  const c = (clsName || "").toLowerCase().trim();
  if (c.includes("unknown")) {
    return "Unknown obstacle";
  }
  const map: Record<string, string> = {
    person: "Person",
    chair: "Chair",
    diningtable: "Table",
    table: "Table",
    bottle: "Bottle",
    cup: "Cup",
    door: "Door",
    stairs: "Stairs",
    stair: "Stairs",
    couch: "Couch",
    sofa: "Sofa",
    bed: "Bed",
    laptop: "Laptop",
    tv: "TV",
    backpack: "Backpack",
    cell_phone: "Phone",
    cellphone: "Phone",
    book: "Book",
    car: "Car",
    bicycle: "Bicycle",
    motorcycle: "Motorcycle",
    dog: "Dog",
    cat: "Cat",
  };
  return map[c] || c.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());
}

/**
 * Format spatial positions into clear English location phrases.
 */
export function positionPhrase(pos: string): string {
  const p = (pos || "CENTER").toUpperCase().trim();
  if (p === "LEFT") return "on the left";
  if (p === "RIGHT") return "on the right";
  return "ahead";
}

/* =========================================================
   AUDIO & HAPTIC EMERGENCY ALARM
   ========================================================= */

let lastEmergencyHapticAt = 0;

export function playEmergencyAlert(): void {
  const now = Date.now();
  if (now - lastEmergencyHapticAt < EMERGENCY_COOLDOWN_MS) {
    return;
  }
  lastEmergencyHapticAt = now;

  try {
    // High-urgency alert vibration pattern: buzz, pause, buzz, pause, long buzz
    Vibration.vibrate([0, 250, 100, 250, 100, 350]);
  } catch (err) {
    console.warn("[Speech] Vibration error:", err);
  }
}

/* =========================================================
   RANKING HELPERS FOR ESCALATION DETECTION
   ========================================================= */

function getSafetyRank(levelOrAlert: string): number {
  const clean = (levelOrAlert || "").toUpperCase().trim();
  if (clean.includes("EMERGENCY")) return 4;
  if (clean.includes("DANGER") || clean.includes("RAPID")) return 3;
  if (clean.includes("CAUTION")) return 2;
  if (clean.includes("AWARE")) return 1;
  return 0; // SAFE, NORMAL, NONE
}

function getMotionRank(motion: string): number {
  const clean = (motion || "").toUpperCase().trim();
  if (clean === "RAPIDLY_APPROACHING" || clean === "RAPID") return 3;
  if (clean === "APPROACHING") return 2;
  if (clean === "STATIONARY" || clean === "STABLE") return 1;
  return 0; // RECEDING, NONE, UNKNOWN
}

function getDistanceRank(categoryOrMeters: string | number): number {
  if (typeof categoryOrMeters === "number") {
    if (categoryOrMeters < 0.8) return 3; // very close
    if (categoryOrMeters < 1.5) return 2; // close
    if (categoryOrMeters < 3.0) return 1; // medium
    return 0; // far
  }
  const clean = (categoryOrMeters || "").toUpperCase().trim();
  if (clean.includes("VERY_CLOSE") || clean.includes("IMMEDIATE")) return 3;
  if (clean.includes("CLOSE") || clean.includes("NEAR")) return 2;
  if (clean.includes("MEDIUM")) return 1;
  return 0; // FAR
}

/* =========================================================
   VOICE DECISION MANAGER CLASS
   ========================================================= */

export class VoiceDecisionManager {
  private speechTimeoutId: ReturnType<typeof setTimeout> | null = null;

  private state: VoiceManagerState = {
    isSpeaking: false,
    lastSpokenText: "",
    lastSpokenAt: 0,
    lastPriority: 0,
    lastSafetyAlert: "SAFE",
    lastMotionTrend: "STATIONARY",
    lastDistanceCategory: "FAR",
    lastThreatTrackId: null,
    lastThreatClass: "",
    lastAction: "NONE",
    lastSteps: null,
    queuedInstruction: null,
    lastOcrText: "",
    lastOcrSpokenAt: 0,
  };

  /**
   * Reset all internal state and timers.
   */
  public reset(): void {
    if (this.speechTimeoutId) {
      clearTimeout(this.speechTimeoutId);
      this.speechTimeoutId = null;
    }
    this.state = {
      isSpeaking: false,
      lastSpokenText: "",
      lastSpokenAt: 0,
      lastPriority: 0,
      lastSafetyAlert: "SAFE",
      lastMotionTrend: "STATIONARY",
      lastDistanceCategory: "FAR",
      lastThreatTrackId: null,
      lastThreatClass: "",
      lastAction: "NONE",
      lastSteps: null,
      lastOcrText: "",
      lastOcrSpokenAt: 0,
      queuedInstruction: null,
    };
  }

  /**
   * Get current state snapshot (immutable copy).
   */
  public getState(): Readonly<VoiceManagerState> {
    return { ...this.state };
  }

  /**
   * Set speaking flag directly (used by speech callbacks and test harnesses).
   */
  public setSpeaking(speaking: boolean): void {
    this.state.isSpeaking = speaking;
    if (!speaking && this.state.queuedInstruction) {
      // Speech finished; process queued instruction if still valid
      void this.processQueuedInstruction();
    }
  }

  /**
   * Evaluate a SmartVision backend response and determine the voice guidance decision.
   * Pure evaluation given the current state and evaluation timestamp.
   */
  public evaluate(
    response: SmartVisionResponse,
    now: number = Date.now()
  ): VoiceDecision {
    if (!response || response.success === false) {
      return {
        action: "IGNORE",
        instruction: "",
        priority: 1,
        reason: "Invalid or unsuccessful response",
      };
    }

    // 1. EXTRACT DATA FIELDS
    const emergencyStop =
      response.emergency_stop === true ||
      response.emergency_alert === true ||
      (response.safety_level || "").toUpperCase() === "EMERGENCY";

    const safetyAlert =
      (response as any).safety_alert ||
      (emergencyStop ? "EMERGENCY_STOP" : response.safety_level || "SAFE");

    const obstacle = response.primary_obstacle;
    const threat = response.predictive_threat;

    // Navigation & step guidance
    const navRaw = response.navigation;
    const stepGuidance = response.step_guidance;

    let rawAction = "NONE";
    if (typeof navRaw === "object" && navRaw?.action && String(navRaw.action).toUpperCase() !== "NONE") {
      rawAction = String(navRaw.action).toUpperCase();
    } else if (stepGuidance && stepGuidance.action && String(stepGuidance.action).toUpperCase() !== "NONE") {
      rawAction = String(stepGuidance.action).toUpperCase();
    } else if (typeof navRaw === "object" && navRaw?.direction) {
      rawAction = String(navRaw.direction).toUpperCase();
    } else if (typeof navRaw === "string" && navRaw) {
      rawAction = navRaw.toUpperCase();
    }

    // Normalize action name (Requirement 8: MOVE = walking displacement, TURN = body orientation)
    let action = rawAction;
    if (["WALK_FORWARD", "GO_FORWARD", "FORWARD", "CONTINUE_FORWARD"].includes(rawAction)) {
      action = "MOVE_FORWARD";
    } else if (["WALK_BACK", "GO_BACK", "BACK"].includes(rawAction)) {
      action = "MOVE_BACK";
    } else if (["MOVE_LEFT", "WALK_LEFT"].includes(rawAction)) {
      action = "MOVE_LEFT";
    } else if (["MOVE_RIGHT", "WALK_RIGHT"].includes(rawAction)) {
      action = "MOVE_RIGHT";
    } else if (["TURN_LEFT"].includes(rawAction)) {
      action = "TURN_LEFT";
    } else if (["TURN_RIGHT"].includes(rawAction)) {
      action = "TURN_RIGHT";
    }

    // Steps
    let steps: number | null = null;
    if (stepGuidance && typeof stepGuidance.steps === "number") {
      steps = stepGuidance.steps;
    } else if (typeof navRaw === "object" && typeof (navRaw as any).movement_steps === "number") {
      steps = (navRaw as any).movement_steps;
    } else if (typeof navRaw === "object" && typeof (navRaw as any).steps === "number") {
      steps = (navRaw as any).steps;
    }

    // Threat / obstacle details
    const primaryThreat = threat || obstacle || null;
    const threatClass = (
      primaryThreat?.class_name || (response.multiple_objects?.human_detected ? "person" : "obstacle")
    ).toLowerCase();
    const isPerson = threatClass === "person";

    // Distance
    let distanceM: number | null = null;
    if (primaryThreat && typeof (primaryThreat as any).distance_m === "number") {
      distanceM = (primaryThreat as any).distance_m;
    } else if (primaryThreat && typeof (primaryThreat as any).estimated_distance_m === "number") {
      distanceM = (primaryThreat as any).estimated_distance_m;
    } else if (typeof (navRaw as any)?.distance_m === "number") {
      distanceM = (navRaw as any).distance_m;
    }

    // Motion state
    const threatMotion = (
      threat?.motion_state ||
      obstacle?.motion_state ||
      (obstacle as any)?.motion_trend ||
      (response.approaching ? "APPROACHING" : "STATIONARY")
    ).toUpperCase();

    const isRapidApproach =
      threatMotion === "RAPIDLY_APPROACHING" ||
      safetyAlert === "WARNING_RAPID_APPROACH" ||
      (threat && (threat as any).threat_level === "EMERGENCY");

    const isApproaching =
      isRapidApproach ||
      threatMotion === "APPROACHING" ||
      response.approaching === true ||
      obstacle?.approaching === true ||
      threat?.is_approaching === true;

    // Distance category
    const distanceCategory =
      (primaryThreat as any)?.distance_category ||
      (distanceM !== null
        ? distanceM < 0.8
          ? "VERY_CLOSE"
          : distanceM < 1.5
          ? "CLOSE"
          : distanceM < 3.0
          ? "MEDIUM"
          : "FAR"
        : "UNKNOWN_DISTANCE");

    const threatTrackId =
      (primaryThreat as any)?.track_id || (primaryThreat as any)?.obstacle_id || null;

    // 2. DETERMINE PRIORITY & CANDIDATE INSTRUCTION (5-Level Hierarchy)
    let priority: VoicePriority = 1;
    let instruction = "";

    // Check Phase 13 flags if passed in response
    const hasConnectionLost =
      "connection_lost" in response &&
      Boolean((response as { connection_lost?: boolean }).connection_lost);

    const hasCameraOccluded =
      ("camera_occluded" in response &&
        Boolean((response as { camera_occluded?: boolean }).camera_occluded)) ||
      ("camera_obstructed" in response &&
        Boolean((response as { camera_obstructed?: boolean }).camera_obstructed)) ||
      ("sensor_degraded" in response &&
        Boolean((response as { sensor_degraded?: boolean }).sensor_degraded));

    // Prefer backend's authoritative voice_instruction if present
    const backendVoice = response.voice_instruction?.trim();

    // DIRECTIONAL DETOUR PRECEDENCE:
    // If navigation commands a directional maneuver (MOVE_LEFT, MOVE_RIGHT, TURN_LEFT, TURN_RIGHT),
    // ALWAYS prioritize guiding the user around the obstacle!
    const isDirectionalDetour =
      action === "TURN_LEFT" ||
      action === "TURN_RIGHT" ||
      action === "MOVE_LEFT" ||
      action === "MOVE_RIGHT";

    // PRIORITY 5: IMMINENT EMERGENCY STOP (only when no detour is possible)
    if (
      !isDirectionalDetour &&
      (emergencyStop ||
        safetyAlert === "EMERGENCY_STOP" ||
        (response.safety_level || "").toUpperCase() === "EMERGENCY" ||
        (isRapidApproach && distanceM !== null && distanceM < 0.6))
    ) {
      priority = 5;
      instruction =
        backendVoice ||
        (obstacle?.class_name
          ? `Stop. ${formatObjectName(obstacle.class_name)} ${positionPhrase(obstacle.position || "CENTER")}.`
          : SAFETY_MESSAGES.EMERGENCY_STOP);
    }
    // PRIORITY 5 (Safety Event): CONNECTION LOST
    else if (hasConnectionLost) {
      priority = 5;
      instruction = SAFETY_MESSAGES.CONNECTION_LOST;
    }
    // PRIORITY 4 (Safety Event): CAMERA OCCLUDED / INVALID SENSOR INPUT
    else if (hasCameraOccluded) {
      priority = 4;
      instruction = SAFETY_MESSAGES.CAMERA_OCCLUDED;
    }
    // PRIORITY 4: RAPID DANGER OR STOP ACTION WITHOUT LATERAL DETOUR
    else if (
      !isDirectionalDetour &&
      (isRapidApproach ||
        safetyAlert === "WARNING_RAPID_APPROACH" ||
        action === "STOP" ||
        ((response.safety_level || "").toUpperCase() === "DANGER" && isApproaching))
    ) {
      priority = 4;
      if (backendVoice) {
        instruction = backendVoice;
      } else if (action === "STOP") {
        instruction = obstacle?.class_name
          ? `Stop. ${formatObjectName(obstacle.class_name)} ${positionPhrase(obstacle.position || "CENTER")}.`
          : "Stop. Path is blocked.";
      } else if (isApproaching) {
        instruction = isPerson ? "Warning. Person approaching." : "Warning. Object approaching.";
      } else {
        instruction = "Stop.";
      }
    }
    // PRIORITY 3: TURN / LATERAL MOVEMENT (strictly directional)
    else if (isDirectionalDetour) {
      priority = 3;
      if (backendVoice) {
        const lower = backendVoice.toLowerCase();
        if (
          lower.includes("turn") ||
          lower.includes("move") ||
          lower.includes("shift")
        ) {
          instruction = backendVoice;
        } else if (action === "TURN_LEFT") {
          instruction = `Turn left. ${backendVoice}`;
        } else if (action === "TURN_RIGHT") {
          instruction = `Turn right. ${backendVoice}`;
        } else if (action === "MOVE_LEFT") {
          instruction = `Move left. ${backendVoice}`;
        } else {
          instruction = `Move right. ${backendVoice}`;
        }
      } else if (action === "TURN_LEFT") {
        const pos = positionPhrase(obstacle?.position || "CENTER");
        instruction = obstacle?.class_name
          ? `${formatObjectName(obstacle.class_name)} ${pos}. Turn left.`
          : "Turn left, then move forward.";
      } else if (action === "TURN_RIGHT") {
        const pos = positionPhrase(obstacle?.position || "CENTER");
        instruction = obstacle?.class_name
          ? `${formatObjectName(obstacle.class_name)} ${pos}. Turn right.`
          : "Turn right, then move forward.";
      } else if (action === "MOVE_LEFT") {
        const pos = positionPhrase(obstacle?.position || "RIGHT");
        instruction = obstacle?.class_name
          ? `Move left. ${formatObjectName(obstacle.class_name)} ${pos}.`
          : (steps && steps > 0 ? `Move left for ${steps} steps.` : "Move left for 2 steps.");
      } else {
        const pos = positionPhrase(obstacle?.position || "LEFT");
        instruction = obstacle?.class_name
          ? `Move right. ${formatObjectName(obstacle.class_name)} ${pos}.`
          : (steps && steps > 0 ? `Move right for ${steps} steps.` : "Move right for 3 steps.");
      }
    }
    // PRIORITY 2: MOVEMENT / SPEED GUIDANCE
    else if (
      action === "MOVE_FORWARD" ||
      action === "MOVE_BACK" ||
      action === "SLOW_DOWN"
    ) {
      priority = 2;
      const dirWord = action === "MOVE_FORWARD" ? "forward" : "back";
      const moveClause =
        steps !== null && steps > 0
          ? `Move ${dirWord} ${formatStepWords(steps)}.`
          : `Move ${dirWord}.`;

      if (action === "SLOW_DOWN") {
        const pos = positionPhrase(obstacle?.position || "CENTER");
        instruction =
          backendVoice ||
          (obstacle?.class_name
            ? `Slow down. ${formatObjectName(obstacle.class_name)} ${pos}.`
            : "Slow down.");
      } else if (backendVoice) {
        const lower = backendVoice.toLowerCase();
        if (
          lower.includes("move") ||
          lower.includes("walk") ||
          lower.includes("go") ||
          lower.includes("turn") ||
          lower.includes("stop") ||
          lower.includes("carefully")
        ) {
          instruction = backendVoice;
        } else {
          instruction = `${backendVoice} ${moveClause}`;
        }
      } else {
        instruction = moveClause;
      }
    }
    // PRIORITY 1: GENERAL AWARENESS / NORMAL APPROACHING / CLEAR
    else {
      priority = 1;
      if (backendVoice) {
        instruction = backendVoice;
      } else if (isApproaching) {
        instruction = isPerson ? "Warning. Person approaching." : "Warning. Object approaching.";
      } else if (action === "STOP") {
        instruction = "Stop.";
      } else {
        // Clear scene or no action needed
        const targetObj = obstacle || (response.objects && response.objects.length > 0 ? response.objects[0] : null);
        if (targetObj && targetObj.class_name) {
          const name = formatObjectName(targetObj.class_name);
          const pos = positionPhrase(targetObj.position || "CENTER");
          instruction = `${name} ${pos}. Move forward.`;
        } else if ((response.objects?.length ?? 0) === 0 && !obstacle && !threat) {
          instruction = "Path clear. Move forward.";
        } else {
          instruction = "Move forward.";
        }
      }

      // OCR Signage Reading (Requirement 9: Non-spamming text guidance in safe conditions)
      const ocr = response.ocr;
      if (ocr && ocr.detected && ocr.primary_text && !threat && !obstacle) {
        const ocrClean = ocr.primary_text.trim();
        const normOcr = normalizeInstruction(ocrClean);
        const normLastOcr = normalizeInstruction(this.state.lastOcrText);
        const ocrElapsed = now - this.state.lastOcrSpokenAt;

        if (ocrClean.length >= 2 && (normOcr !== normLastOcr || ocrElapsed >= COOLDOWNS.OCR)) {
          if (
            action === "NONE" ||
            action === "MOVE_FORWARD" ||
            !instruction ||
            instruction === "Path clear." ||
            instruction === "Continue carefully." ||
            instruction === "Go forward."
          ) {
            priority = 1;
            instruction = `Sign reads: ${ocrClean}`;
            this.state.lastOcrText = ocrClean;
            this.state.lastOcrSpokenAt = now;
          }
        }
      }
    }

    if (!instruction.trim()) {
      return {
        action: "IGNORE",
        instruction: "",
        priority: 1,
        reason: "Empty instruction generated",
      };
    }

    if (response.should_speak === false) {
      return {
        action: "SUPPRESS",
        instruction,
        priority,
        reason: "Backend indicated should_speak = false (stationary stop / unchanged situation)",
        bypass_cooldown: false,
      };
    }

    // 3. SIGNIFICANT STATE CHANGE DETECTION
    const actionChanged =
      action !== "NONE" &&
      action !== this.state.lastAction;

    const safetyEscalated =
      getSafetyRank(safetyAlert) > getSafetyRank(this.state.lastSafetyAlert);

    const motionEscalated =
      getMotionRank(threatMotion) > getMotionRank(this.state.lastMotionTrend);

    const distEscalated =
      getDistanceRank(distanceCategory) >
      getDistanceRank(this.state.lastDistanceCategory);

    const threatTrackChanged =
      threatTrackId !== null &&
      this.state.lastThreatTrackId !== null &&
      threatTrackId !== this.state.lastThreatTrackId &&
      (action !== "STOP" || (distanceM !== null && distanceM < 0.40));

    const threatClassChanged =
      threatClass !== "" &&
      Boolean(this.state.lastThreatClass) &&
      threatClass !== this.state.lastThreatClass;

    const stepCountChanged =
      steps !== null &&
      this.state.lastSteps !== null &&
      action !== "STOP" &&
      Math.abs(steps - this.state.lastSteps) >= 2;

    const bypassCooldown =
      actionChanged ||
      safetyEscalated ||
      motionEscalated ||
      distEscalated ||
      threatTrackChanged ||
      threatClassChanged ||
      stepCountChanged;

    // Discard stale queued movement instructions on state change
    if (bypassCooldown && this.state.queuedInstruction) {
      const isQueuedMovement = ["MOVE_FORWARD", "MOVE_BACK", "WALK_FORWARD", "WALK_BACK"].includes(this.state.queuedInstruction.action);
      if (isQueuedMovement && (this.state.queuedInstruction.action !== action || safetyEscalated)) {
        this.state.queuedInstruction = null;
      }
    }

    // 4. NORMALIZED COMPARISON & COOLDOWN / REPEAT TIMING
    const normCandidate = normalizeInstruction(instruction);
    const normLastSpoken = normalizeInstruction(this.state.lastSpokenText);
    const isSameText = normCandidate === normLastSpoken && normCandidate.length > 0;
    const elapsed = now - this.state.lastSpokenAt;

    const isStopInstruction =
      action === "STOP" ||
      normCandidate.startsWith("stop");

    const wasAlreadyStopped =
      this.state.lastAction === "STOP" ||
      normLastSpoken.startsWith("stop");

    // Suppress continuous STOP shouting if situation has not escalated
    if (
      isStopInstruction &&
      wasAlreadyStopped &&
      !safetyEscalated &&
      !motionEscalated &&
      (distanceM === null || distanceM >= 0.40)
    ) {
      if (elapsed < COOLDOWNS.REPEAT) {
        return {
          action: "SUPPRESS",
          instruction,
          priority,
          reason: "User already stopped; continuous STOP suppressed",
          bypass_cooldown: false,
        };
      }
    }

    // Determine required cooldown window for candidate
    let requiredCooldown: number;
    if (priority === 5) {
      requiredCooldown = COOLDOWNS.EMERGENCY; // 1000ms
    } else if (priority === 4 || isApproaching) {
      requiredCooldown = COOLDOWNS.APPROACHING; // 2500ms
    } else if (
      safetyAlert.toUpperCase().includes("CAUTION") ||
      safetyAlert.toUpperCase().includes("DANGER")
    ) {
      requiredCooldown = COOLDOWNS.SAFETY; // 3000ms
    } else if (action === "MOVE_FORWARD" && steps && steps > 1) {
      requiredCooldown = Math.max(COOLDOWNS.NORMAL, steps * 750);
    } else {
      requiredCooldown = COOLDOWNS.NORMAL; // 2500ms
    }

    // REPEAT: Persistent identical instruction after REPEAT cooldown
    if (isSameText && !bypassCooldown) {
      if (elapsed >= COOLDOWNS.REPEAT) {
        // If speaking right now, apply speech lock logic
        if (this.state.isSpeaking) {
          if (priority > this.state.lastPriority) {
            return {
              action: "INTERRUPT",
              instruction,
              priority,
              reason: `Priority ${priority} interrupts Priority ${this.state.lastPriority} speech`,
              bypass_cooldown: false,
            };
          }
          return {
            action: "SUPPRESS",
            instruction,
            priority,
            reason: "Speech active; repeat suppressed",
            bypass_cooldown: false,
          };
        }
        return {
          action: "REPEAT",
          instruction,
          priority,
          reason: `Persistent condition repeated after ${elapsed}ms`,
          bypass_cooldown: false,
        };
      }
      return {
        action: "SUPPRESS",
        instruction,
        priority,
        reason: `Duplicate instruction within repeat window (${elapsed}ms < ${COOLDOWNS.REPEAT}ms)`,
        bypass_cooldown: false,
      };
    }

    // COOLDOWN CHECK (when not a repeat and not bypassed)
    if (!bypassCooldown && this.state.lastSpokenAt > 0 && elapsed < requiredCooldown) {
      return {
        action: "SUPPRESS",
        instruction,
        priority,
        reason: `Cooldown active (${elapsed}ms < ${requiredCooldown}ms)`,
        bypass_cooldown: false,
      };
    }

    // 5. SPEECH LOCK & QUEUE (when speech is currently in progress)
    if (this.state.isSpeaking) {
      // Higher priority interrupts active speech
      if (priority > this.state.lastPriority) {
        return {
          action: "INTERRUPT",
          instruction,
          priority,
          reason: `Higher priority (${priority} > ${this.state.lastPriority}) interrupts active speech`,
          bypass_cooldown: bypassCooldown,
        };
      }

      // Equal or lower priority: check queue eligibility
      if (priority >= 2) {
        if (!this.state.queuedInstruction) {
          return {
            action: "QUEUE",
            instruction,
            priority,
            reason: `Queued during active speech (Priority ${priority})`,
            bypass_cooldown: bypassCooldown,
            queued: true,
          };
        } else if (priority > this.state.queuedInstruction.priority) {
          return {
            action: "QUEUE",
            instruction,
            priority,
            reason: `Replaces lower priority queued instruction (${priority} > ${this.state.queuedInstruction.priority})`,
            bypass_cooldown: bypassCooldown,
            queued: true,
          };
        } else {
          return {
            action: "SUPPRESS",
            instruction,
            priority,
            reason: "Queue full with equal or higher priority instruction",
            bypass_cooldown: bypassCooldown,
            queued: false,
          };
        }
      }

      // Priority 1 is dropped when speech is active
      return {
        action: "SUPPRESS",
        instruction,
        priority,
        reason: "Priority 1 instruction suppressed while speaking",
        bypass_cooldown: bypassCooldown,
      };
    }

    // 6. SPEAK IMMEDIATELY
    return {
      action: "SPEAK",
      instruction,
      priority,
      reason: bypassCooldown
        ? "State change bypass allows immediate speech"
        : "Standard speech trigger",
      bypass_cooldown: bypassCooldown,
    };
  }

  /**
   * Main entry point for processing a live backend detection frame.
   * Evaluates decision, applies priority preemption, queueing, and speech lock.
   */
  public async handleResponse(
    response: SmartVisionResponse
  ): Promise<VoiceDecision> {
    const now = Date.now();
    const decision = this.evaluate(response, now);

    if (__DEV__) {
      console.log(
        `[VOICE_PIPELINE] decision=${decision.action} prio=${decision.priority} bypass=${decision.bypass_cooldown} spoken="${decision.instruction}" reason="${decision.reason}"`
      );
    }

    // Update state tracking fields for escalation detection
    const obstacle = response.primary_obstacle;
    const threat = response.predictive_threat;
    const primaryThreat = threat || obstacle;

    if (decision.action === "SPEAK" || decision.action === "INTERRUPT" || decision.action === "REPEAT") {
      // Priority 5 Emergency triggers audio & haptic alarm and interrupts speech
      if (decision.priority === 5) {
        playEmergencyAlert();
        this.state.queuedInstruction = null; // Clear queue on emergency
      }

      if (decision.action === "INTERRUPT") {
        try {
          await Speech.stop();
        } catch (e) {
          console.warn("[VoiceManager] Speech.stop error:", e);
        }
      }

      // Commit speech state
      this.state.lastSpokenText = decision.instruction;
      this.state.lastSpokenAt = now;
      this.state.lastPriority = decision.priority;
      this.state.isSpeaking = true;

      // Speak text using Expo Speech
      await this.executeSpeech(decision.instruction, decision.priority);
    } else if (decision.action === "QUEUE") {
      // Step action name
      const stepGuidance = response.step_guidance;
      const navRaw = response.navigation;
      const act = stepGuidance?.action || (typeof navRaw === "object" ? (navRaw as any)?.action || (navRaw as any)?.direction : navRaw) || "NONE";
      let normAct = String(act).toUpperCase();
      if (["WALK_FORWARD", "GO_FORWARD", "FORWARD", "CONTINUE_FORWARD"].includes(normAct)) {
        normAct = "MOVE_FORWARD";
      } else if (["WALK_BACK", "GO_BACK", "BACK"].includes(normAct)) {
        normAct = "MOVE_BACK";
      } else if (["MOVE_LEFT", "WALK_LEFT"].includes(normAct)) {
        normAct = "MOVE_LEFT";
      } else if (["MOVE_RIGHT", "WALK_RIGHT"].includes(normAct)) {
        normAct = "MOVE_RIGHT";
      } else if (["TURN_LEFT"].includes(normAct)) {
        normAct = "TURN_LEFT";
      } else if (["TURN_RIGHT"].includes(normAct)) {
        normAct = "TURN_RIGHT";
      }

      this.state.queuedInstruction = {
        text: decision.instruction,
        priority: decision.priority,
        action: normAct,
        timestamp: now,
      };
    }

    // Always update telemetry for baseline state comparisons
    const prevSafety = this.state.lastSafetyAlert;
    const safetyAlert =
      (response as any).safety_alert ||
      (response.emergency_stop ? "EMERGENCY_STOP" : response.safety_level || "SAFE");
    this.state.lastSafetyAlert = String(safetyAlert);

    const threatMotion = (
      threat?.motion_state ||
      obstacle?.motion_state ||
      (obstacle as any)?.motion_trend ||
      (response.approaching ? "APPROACHING" : "STATIONARY")
    );
    this.state.lastMotionTrend = String(threatMotion);

    if (primaryThreat) {
      const dist = (primaryThreat as any).distance_m ?? (primaryThreat as any).estimated_distance_m;
      this.state.lastDistanceCategory = (primaryThreat as any).distance_category || (typeof dist === "number" && dist < 0.8 ? "VERY_CLOSE" : "FAR");
      this.state.lastThreatTrackId = (primaryThreat as any).track_id || (primaryThreat as any).obstacle_id || null;
      this.state.lastThreatClass = (primaryThreat.class_name || "").toLowerCase();
    } else {
      this.state.lastThreatClass = "";
    }

    const stepGuidance = response.step_guidance;
    const navRaw = response.navigation;
    const act = stepGuidance?.action || (typeof navRaw === "object" ? (navRaw as any)?.action || (navRaw as any)?.direction : navRaw) || "NONE";
    let normAct = String(act).toUpperCase();
    if (["WALK_FORWARD", "GO_FORWARD", "FORWARD", "CONTINUE_FORWARD"].includes(normAct)) {
      normAct = "MOVE_FORWARD";
    } else if (["WALK_BACK", "GO_BACK", "BACK"].includes(normAct)) {
      normAct = "MOVE_BACK";
    } else if (["WALK_LEFT", "STEP_LEFT"].includes(normAct)) {
      normAct = "MOVE_LEFT";
    } else if (["WALK_RIGHT", "STEP_RIGHT"].includes(normAct)) {
      normAct = "MOVE_RIGHT";
    }
    this.state.lastAction = normAct;

    if (stepGuidance && typeof stepGuidance.steps === "number") {
      this.state.lastSteps = stepGuidance.steps;
    } else if (typeof navRaw === "object" && typeof (navRaw as any).movement_steps === "number") {
      this.state.lastSteps = (navRaw as any).movement_steps;
    }

    // Discard stale queued movement instructions on state change
    if (decision.bypass_cooldown && this.state.queuedInstruction) {
      const isQueuedMovement = ["MOVE_FORWARD", "MOVE_BACK", "WALK_FORWARD", "WALK_BACK"].includes(this.state.queuedInstruction.action);
      const safetyEscalated = getSafetyRank(this.state.lastSafetyAlert) > getSafetyRank(prevSafety);
      if (isQueuedMovement && (normAct !== this.state.queuedInstruction.action || decision.priority >= 3 || safetyEscalated)) {
        this.state.queuedInstruction = null;
      }
    }

    return decision;
  }

  /**
   * Internal execution of speech with Expo Speech with completion callbacks.
   */
  private async executeSpeech(text: string, priority: VoicePriority): Promise<void> {
    const isEmergency = priority === 5;
    const wordCount = text.trim().split(/\s+/).length;
    const estimatedDurationMs = Math.max(2000, wordCount * 450 + 800);

    if (this.speechTimeoutId) {
      clearTimeout(this.speechTimeoutId);
      this.speechTimeoutId = null;
    }

    // Safety fallback: ensure isSpeaking is reset even if Android drops completion callbacks
    this.speechTimeoutId = setTimeout(() => {
      if (this.state.isSpeaking) {
        this.state.isSpeaking = false;
        void this.processQueuedInstruction();
      }
    }, estimatedDurationMs);

    try {
      Speech.speak(text, {
        language: "en-US",
        pitch: isEmergency ? 1.05 : 1.0,
        rate: isEmergency ? 0.95 : 0.92,
        volume: 1.0,
        onStart: () => {
          this.state.isSpeaking = true;
        },
        onDone: () => {
          if (this.speechTimeoutId) {
            clearTimeout(this.speechTimeoutId);
            this.speechTimeoutId = null;
          }
          this.state.isSpeaking = false;
          void this.processQueuedInstruction();
        },
        onStopped: () => {
          if (this.speechTimeoutId) {
            clearTimeout(this.speechTimeoutId);
            this.speechTimeoutId = null;
          }
          this.state.isSpeaking = false;
        },
        onError: () => {
          if (this.speechTimeoutId) {
            clearTimeout(this.speechTimeoutId);
            this.speechTimeoutId = null;
          }
          this.state.isSpeaking = false;
          void this.processQueuedInstruction();
        },
      });
    } catch (err) {
      if (this.speechTimeoutId) {
        clearTimeout(this.speechTimeoutId);
        this.speechTimeoutId = null;
      }
      console.warn("[VoiceManager] executeSpeech error:", err);
      this.state.isSpeaking = false;
    }
  }

  /**
   * Process any pending queued instruction after speech finishes.
   */
  private async processQueuedInstruction(): Promise<void> {
    const queued = this.state.queuedInstruction;
    if (!queued) return;

    const now = Date.now();
    // Drop stale movement instructions (> 3500ms)
    if (now - queued.timestamp > 3500) {
      this.state.queuedInstruction = null;
      return;
    }

    this.state.queuedInstruction = null;
    this.state.lastSpokenText = queued.text;
    this.state.lastSpokenAt = now;
    this.state.lastPriority = queued.priority;
    this.state.isSpeaking = true;

    await this.executeSpeech(queued.text, queued.priority);
  }

  /**
   * Direct speak method bypassing backend response evaluation.
   */
  public async speak(text: string, options?: VoiceOptions): Promise<boolean> {
    const msg = text?.trim();
    if (!msg) return false;

    const priority = options?.priority ?? 1;
    const now = Date.now();
    const elapsed = now - this.state.lastSpokenAt;

    if (!options?.bypassCooldown && elapsed < COOLDOWNS.NORMAL && normalizeInstruction(msg) === normalizeInstruction(this.state.lastSpokenText)) {
      return false;
    }

    this.state.lastSpokenText = msg;
    this.state.lastSpokenAt = now;
    this.state.lastPriority = priority;
    this.state.isSpeaking = true;

    await this.executeSpeech(msg, priority);
    return true;
  }

  /**
   * Immediate emergency speech with preemption and haptics.
   */
  public async speakEmergency(text: string = "Stop immediately."): Promise<boolean> {
    playEmergencyAlert();
    this.state.queuedInstruction = null;

    try {
      await Speech.stop();
    } catch (e) {
      // ignore
    }

    const now = Date.now();
    this.state.lastSpokenText = text;
    this.state.lastSpokenAt = now;
    this.state.lastPriority = 5;
    this.state.isSpeaking = true;

    await this.executeSpeech(text, 5);
    return true;
  }

  /**
   * Evaluate a discrete Phase 13 safety event (Connection Lost, Connection Restored, Camera Occluded).
   */
  public evaluateSafetyEvent(
    event: VoiceSafetyEvent,
    now: number = Date.now()
  ): VoiceDecision {
    let instruction: string = "";
    let priority: VoicePriority = 4;
    let bypassCooldown = false;

    switch (event) {
      case "CONNECTION_LOST":
        instruction = SAFETY_MESSAGES.CONNECTION_LOST;
        priority = 5;
        break;

      case "CONNECTION_RESTORED":
        instruction = SAFETY_MESSAGES.CONNECTION_RESTORED;
        priority = 4;
        bypassCooldown = true;
        break;

      case "CAMERA_OCCLUDED":
        instruction = SAFETY_MESSAGES.CAMERA_OCCLUDED;
        priority = 4;
        break;
    }

    const normCandidate = normalizeInstruction(instruction);
    const normLastSpoken = normalizeInstruction(this.state.lastSpokenText);
    const isSameText = normCandidate === normLastSpoken && normCandidate.length > 0;
    const elapsed = now - this.state.lastSpokenAt;

    // Suppress repeated identical safety instructions within the repeat window
    if (isSameText && !bypassCooldown) {
      const repeatWindow = event === "CONNECTION_LOST" ? COOLDOWNS.CONNECTION : COOLDOWNS.REPEAT;
      if (elapsed < repeatWindow) {
        return {
          action: "SUPPRESS",
          instruction,
          priority,
          reason: `Duplicate safety event within repeat window (${elapsed}ms < ${repeatWindow}ms)`,
          bypass_cooldown: false,
        };
      }
    }

    // Speech lock & preemption
    if (this.state.isSpeaking) {
      // Preempt lower-priority speech, or equal priority if new safety instruction
      if (
        priority > this.state.lastPriority ||
        (priority === 5 && !isSameText)
      ) {
        return {
          action: "INTERRUPT",
          instruction,
          priority,
          reason: `Safety event ${event} (Priority ${priority}) interrupts active Priority ${this.state.lastPriority} speech`,
          bypass_cooldown: true,
        };
      }

      return {
        action: "SUPPRESS",
        instruction,
        priority,
        reason: `Active speech prevents safety event ${event}`,
        bypass_cooldown: false,
      };
    }

    return {
      action: "SPEAK",
      instruction,
      priority,
      reason: `Safety event ${event} immediate trigger`,
      bypass_cooldown: bypassCooldown,
    };
  }

  /**
   * Process a discrete Phase 13 safety event through the centralized speech manager.
   */
  public async handleSafetyEvent(
    event: VoiceSafetyEvent
  ): Promise<VoiceDecision> {
    const now = Date.now();
    const decision = this.evaluateSafetyEvent(event, now);

    if (
      decision.action === "SPEAK" ||
      decision.action === "INTERRUPT" ||
      decision.action === "REPEAT"
    ) {
      // Discard queued movement commands when entering safe stop/degraded state
      if (event === "CONNECTION_LOST" || event === "CAMERA_OCCLUDED") {
        this.state.queuedInstruction = null;
      }

      if (decision.action === "INTERRUPT") {
        try {
          await Speech.stop();
        } catch (e) {
          console.warn("[VoiceManager] Speech.stop error:", e);
        }
      }

      this.state.lastSpokenText = decision.instruction;
      this.state.lastSpokenAt = now;
      this.state.lastPriority = decision.priority;
      this.state.isSpeaking = true;

      await this.executeSpeech(decision.instruction, decision.priority);
    }

    return decision;
  }

  /**
   * Spoken alert for connection loss: "Connection lost. Stop walking."
   */
  public async speakConnectionLost(): Promise<VoiceDecision> {
    return this.handleSafetyEvent("CONNECTION_LOST");
  }

  /**
   * Spoken alert for connection recovery: "Connection restored. Resuming guidance."
   */
  public async speakConnectionRestored(): Promise<VoiceDecision> {
    return this.handleSafetyEvent("CONNECTION_RESTORED");
  }

  /**
   * Spoken alert for camera obstruction: "Camera view obstructed or scene too dark. Please check camera view."
   */
  public async speakCameraOccluded(): Promise<VoiceDecision> {
    return this.handleSafetyEvent("CAMERA_OCCLUDED");
  }

  /**
   * Stop speech immediately and clear pending queue.
   */
  public async stop(): Promise<void> {
    if (this.speechTimeoutId) {
      clearTimeout(this.speechTimeoutId);
      this.speechTimeoutId = null;
    }
    this.state.isSpeaking = false;
    this.state.queuedInstruction = null;
    try {
      await Speech.stop();
    } catch (e) {
      console.warn("[VoiceManager] stop error:", e);
    }
  }
}

/* =========================================================
   SINGLETON INSTANCE & EXPORTED HELPERS
   ========================================================= */

export const voiceManager = new VoiceDecisionManager();

export async function stopSpeech(): Promise<void> {
  await voiceManager.stop();
}

export async function speak(
  text: string,
  targetLang?: VoiceLanguage,
  options?: Speech.SpeechOptions | VoiceOptions
): Promise<void> {
  const msg = text?.trim();
  if (!msg) return;
  await voiceManager.speak(msg);
}

export async function speakEmergency(
  text: string = "Stop immediately.",
  _targetLang?: VoiceLanguage
): Promise<void> {
  await voiceManager.speakEmergency(text);
}

export function resetSpeechState(): void {
  voiceManager.reset();
}

/**
 * Generate assistive instruction for backward compatibility.
 */
export function generateAssistiveInstruction(
  response: SmartVisionResponse,
  _language?: VoiceLanguage
): AssistiveVoiceInstruction {
  const decision = voiceManager.evaluate(response);
  const isEmergency = decision.priority === 5;

  const obstacle = response.primary_obstacle;
  const dist = (obstacle as any)?.distance_m ?? (obstacle as any)?.estimated_distance_m;
  const steps = (obstacle as any)?.estimated_steps ?? (response.step_guidance?.steps);

  return {
    text: decision.instruction,
    isEmergency,
    distanceM: typeof dist === "number" ? dist : undefined,
    steps: typeof steps === "number" ? steps : undefined,
    confidence: obstacle?.distance_confidence || "MEDIUM",
    approaching: decision.priority === 4 || decision.priority === 1,
    language: "en",
    priority: decision.priority,
  };
}

export async function speakNavigation(
  direction: string,
  reason?: string
): Promise<void> {
  const cleanDir = direction?.trim();
  if (!cleanDir) return;
  const cleanReason = reason?.trim();
  const message = cleanReason ? `${cleanDir}. ${cleanReason}` : cleanDir;
  await voiceManager.speak(message, { priority: 3 });
}

export async function speakSafety(
  level: string,
  message?: string
): Promise<void> {
  const cleanLevel = (level || "").trim().toUpperCase();
  if (cleanLevel === "EMERGENCY") {
    await voiceManager.speakEmergency(message?.trim() || "Stop immediately.");
    return;
  }
  if (cleanLevel === "DANGER") {
    await voiceManager.speak(message?.trim() || "Stop.", { priority: 4 });
    return;
  }
  if (message?.trim()) {
    await voiceManager.speak(message.trim(), { priority: 1 });
  }
}

export function navigationToSpeech(
  navigation: string | { direction?: string; reason?: string } | undefined
): string {
  if (!navigation) return "";
  if (typeof navigation === "string") {
    return navigation.replace(/_/g, " ").toLowerCase();
  }
  const direction = navigation.direction?.replace(/_/g, " ").toLowerCase();
  const reason = navigation.reason?.trim();
  if (!direction) return reason || "";
  return reason ? `${direction}. ${reason}` : direction;
}

/* =========================================================
   PHASE 13 SAFETY EXPORTS
   ========================================================= */

export async function speakConnectionLost(): Promise<VoiceDecision> {
  return voiceManager.speakConnectionLost();
}

export async function speakConnectionRestored(): Promise<VoiceDecision> {
  return voiceManager.speakConnectionRestored();
}

export async function speakCameraOccluded(): Promise<VoiceDecision> {
  return voiceManager.speakCameraOccluded();
}

export async function handleSafetyEvent(
  event: VoiceSafetyEvent
): Promise<VoiceDecision> {
  return voiceManager.handleSafetyEvent(event);
}

export default {
  VoiceDecisionManager,
  voiceManager,
  speak,
  speakEmergency,
  stopSpeech,
  playEmergencyAlert,
  generateAssistiveInstruction,
  resetSpeechState,
  speakNavigation,
  speakSafety,
  speakConnectionLost,
  speakConnectionRestored,
  speakCameraOccluded,
  handleSafetyEvent,
  navigationToSpeech,
  normalizeInstruction,
  formatStepWords,
  COOLDOWNS,
  LANGUAGE_VOICE_CODES,
  SAFETY_MESSAGES,
};