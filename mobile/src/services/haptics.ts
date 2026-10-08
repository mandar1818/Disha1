/**
 * SmartVisionAI - Centralized Haptic Feedback Service (Phase 13)
 *
 * Responsibilities:
 * - Deterministic, non-blocking tactile vibration feedback for visually impaired users.
 * - Standardized patterns for critical navigation, safety, and operational states:
 *     1. EMERGENCY          : Strong, elongated triple pulse ([0, 400, 150, 400, 150, 600])
 *     2. CONNECTION_LOST    : Distinct rhythmic alert pulses ([0, 250, 100, 250, 100, 250])
 *     3. DANGER             : Rapid double warning pulse ([0, 150, 100, 150])
 *     4. TURN_LEFT          : Two snappy, short taps ([0, 80, 80, 80])
 *     5. TURN_RIGHT         : Elongated single directional rumble ([0, 240])
 *     6. CAMERA_OCCLUDED    : Rapid triple-tick warning ([0, 60, 60, 60, 60, 60])
 *     7. PATH_CLEAR         : Subtle single confirmation tick ([0, 50])
 * - Immediate preemption: Higher-priority alerts cancel any active in-flight vibration.
 * - Non-repetitive safety: No infinite vibration loops or timer leaks.
 * - Pure tactile translation: Does NOT make navigation decisions, speak, or call backend.
 */

import { Vibration } from "react-native";

/* =========================================================
   TYPES & ENUMS
   ========================================================= */

export type HapticEvent =
  | "EMERGENCY"
  | "CONNECTION_LOST"
  | "DANGER"
  | "SLOW_DOWN"
  | "TURN_LEFT"
  | "TURN_RIGHT"
  | "MOVE_LEFT"
  | "MOVE_RIGHT"
  | "CAMERA_OCCLUDED"
  | "PATH_CLEAR";

/* =========================================================
   HAPTIC PATTERNS SPECIFICATION (Android / iOS)
   Format: [waitMs, vibrateMs, waitMs, vibrateMs, ...]
   ========================================================= */

export const HAPTIC_PATTERNS: Record<HapticEvent, readonly number[]> = {
  EMERGENCY: [0, 400, 150, 400, 150, 600],
  CONNECTION_LOST: [0, 250, 100, 250, 100, 250],
  DANGER: [0, 150, 100, 150],
  SLOW_DOWN: [0, 120, 80, 120],
  TURN_LEFT: [0, 80, 80, 80],
  TURN_RIGHT: [0, 240],
  MOVE_LEFT: [0, 60, 60, 60],
  MOVE_RIGHT: [0, 160, 60, 160],
  CAMERA_OCCLUDED: [0, 60, 60, 60, 60, 60],
  PATH_CLEAR: [0, 50],
} as const;

/* =========================================================
   INTERNAL STATE
   ========================================================= */

let hapticsEnabled: boolean = true;
let lastTriggeredEvent: HapticEvent | null = null;
let lastTriggeredAt: number = 0;

/* =========================================================
   PUBLIC API
   ========================================================= */

/**
 * Return whether the host platform supports the React Native Vibration API.
 */
export function isHapticSupported(): boolean {
  return (
    typeof Vibration !== "undefined" &&
    typeof Vibration.vibrate === "function" &&
    typeof Vibration.cancel === "function"
  );
}

/**
 * Return current user preference enable status for haptics.
 */
export function isHapticsEnabled(): boolean {
  return hapticsEnabled;
}

/**
 * Enable or disable haptic feedback globally.
 * If disabled while a vibration is active, it is immediately cancelled.
 */
export function setHapticsEnabled(enabled: boolean): void {
  hapticsEnabled = Boolean(enabled);
  if (!hapticsEnabled) {
    stopHaptic();
  }
}

/**
 * Return the vibration pattern array for a given haptic event.
 */
export function getHapticPattern(event: HapticEvent): readonly number[] {
  return HAPTIC_PATTERNS[event] ?? [0, 50];
}

/**
 * Cancel any currently active vibration pattern immediately.
 */
export function stopHaptic(): void {
  try {
    if (isHapticSupported()) {
      Vibration.cancel();
    }
  } catch (error) {
    // Fail silently in environments where vibration fails
    console.warn("[Haptics] stopHaptic error:", error);
  }
}

/**
 * Trigger a haptic feedback pattern for the specified event.
 *
 * Safety & Preemption:
 * - Cancels any currently active vibration before starting the new pattern.
 * - Non-repeating execution (repeat = false) to prevent runaway vibration.
 * - Graceful failure if vibration hardware is absent.
 */
export function triggerHaptic(event: HapticEvent): void {
  if (!hapticsEnabled) {
    return;
  }

  const pattern = HAPTIC_PATTERNS[event];
  if (!pattern || pattern.length === 0) {
    return;
  }

  try {
    // 1. Immediately cancel active pattern to ensure instant preemption
    stopHaptic();

    // 2. Play the distinct pattern once (repeat = false)
    if (isHapticSupported()) {
      Vibration.vibrate([...pattern], false);
    }

    lastTriggeredEvent = event;
    lastTriggeredAt = Date.now();
  } catch (error) {
    // Fail silently rather than crashing camera detection loop or navigation
    console.warn("[Haptics] triggerHaptic error:", error);
  }
}

/**
 * Return diagnostic details about the last triggered haptic event.
 */
export function getLastHapticEvent(): {
  event: HapticEvent | null;
  timestamp: number;
} {
  return {
    event: lastTriggeredEvent,
    timestamp: lastTriggeredAt,
  };
}

/* =========================================================
   CONVENIENCE HELPERS
   ========================================================= */

export function triggerEmergencyHaptic(): void {
  triggerHaptic("EMERGENCY");
}

export function triggerConnectionLostHaptic(): void {
  triggerHaptic("CONNECTION_LOST");
}

export function triggerDangerHaptic(): void {
  triggerHaptic("DANGER");
}

export function triggerSlowDownHaptic(): void {
  triggerHaptic("SLOW_DOWN");
}

export function triggerTurnLeftHaptic(): void {
  triggerHaptic("TURN_LEFT");
}

export function triggerTurnRightHaptic(): void {
  triggerHaptic("TURN_RIGHT");
}

export function triggerMoveLeftHaptic(): void {
  triggerHaptic("MOVE_LEFT");
}

export function triggerMoveRightHaptic(): void {
  triggerHaptic("MOVE_RIGHT");
}

export function triggerCameraOccludedHaptic(): void {
  triggerHaptic("CAMERA_OCCLUDED");
}

export function triggerPathClearHaptic(): void {
  triggerHaptic("PATH_CLEAR");
}

/* =========================================================
   DEFAULT EXPORT
   ========================================================= */

const hapticsService = {
  triggerHaptic,
  stopHaptic,
  getHapticPattern,
  isHapticSupported,
  isHapticsEnabled,
  setHapticsEnabled,
  getLastHapticEvent,
  triggerEmergencyHaptic,
  triggerConnectionLostHaptic,
  triggerDangerHaptic,
  triggerSlowDownHaptic,
  triggerTurnLeftHaptic,
  triggerTurnRightHaptic,
  triggerMoveLeftHaptic,
  triggerMoveRightHaptic,
  triggerCameraOccludedHaptic,
  triggerPathClearHaptic,
};

export default hapticsService;

