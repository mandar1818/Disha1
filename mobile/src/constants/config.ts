/**
 * SmartVisionAI
 * Global application configuration
 *
 * IMPORTANT:
 * This file is shared by:
 * - Camera
 * - Finder
 * - Settings
 * - Safety utilities
 * - UI components
 * - API service
 *
 * Keep all common constants here so every file
 * uses the same configuration.
 */

import Constants from "expo-constants";

/* =========================================================
   APP INFORMATION
   ========================================================= */

export const APP_NAME = "Disha";
export const APP_VERSION = "1.0.0";

/* =========================================================
   BACKEND CONFIGURATION
   ========================================================= */

/**
 * .env example:
 *
 * EXPO_PUBLIC_BACKEND_URL=http://192.168.1.100:8000
 *
 * IMPORTANT:
 * Android physical device CANNOT use localhost
 * to access the backend running on your PC.
 */
const ENV_BACKEND_URL =
  process.env.EXPO_PUBLIC_BACKEND_URL?.trim() || "";

const EXTRA_BACKEND_URL =
  (Constants.expoConfig?.extra as { backendUrl?: string } | undefined)
    ?.backendUrl
    ?.trim() || "";

const DEFAULT_BACKEND_URL =
  ENV_BACKEND_URL ||
  EXTRA_BACKEND_URL ||
  "http://192.168.1.100:8000";

export const BACKEND_URL = DEFAULT_BACKEND_URL.replace(/\/+$/, "");

export const DETECT_URL = `${BACKEND_URL}/detect`;
export const HEALTH_URL = `${BACKEND_URL}/health`;
export const SYSTEM_STATUS_URL = `${BACKEND_URL}/system/status`;

/* =========================================================
   NETWORK
   ========================================================= */

export const REQUEST_TIMEOUT_MS = 30000;

/* =========================================================
   CAMERA / DETECTION
   ========================================================= */

export const IMAGE_QUALITY = 0.25;

/**
 * Minimum YOLO confidence accepted by the mobile app.
 */
export const MIN_CONFIDENCE = 0.25;

/**
 * Authoritative live detection interval.
 *
 * 500 ms = approximately 2 detection requests/second.
 */
export const DETECTION_INTERVAL_MS = 500;

/**
 * Standardized across all detection loops.
 */
export const FINDER_INTERVAL_MS = DETECTION_INTERVAL_MS;

/* =========================================================
   SPEECH
   ========================================================= */

/**
 * Prevents the application from speaking the same
 * navigation message repeatedly.
 */
export const SPEECH_COOLDOWN_MS = 2500;

/* =========================================================
   SAFETY THRESHOLDS
   =========================================================
   
   Standardized risk score expected from backend:
   0.0 - 1.0
   ========================================================= */

export const AWARE_RISK_THRESHOLD = 0.20;
export const CAUTION_RISK_THRESHOLD = 0.40;
export const DANGER_RISK_THRESHOLD = 0.60;
export const EMERGENCY_RISK_THRESHOLD = 0.80;

/* =========================================================
   COLORS
   =========================================================
   
   Keep aliases such as:
   bg
   panel
   panel2
   accent
   muted
   caution
   
   because different UI components use these names.
   ========================================================= */

export const COLORS = {
  /* Main backgrounds */
  background: "#07101C",
  bg: "#07101C",

  /* Surfaces */
  surface: "#0D1928",
  surfaceLight: "#132136",

  /* Panels */
  panel: "#0D1928",
  panel2: "#132136",

  /* Primary */
  primary: "#1677FF",
  primaryDark: "#0D5ED7",

  /* General text */
  white: "#FFFFFF",
  text: "#FFFFFF",
  muted: "#8195AA",
  secondaryText: "#9FB0C2",

  /* Accent */
  accent: "#63D8FF",

  /* Safety colors */
  safe: "#22C55E",      // Green
  aware: "#EAB308",     // Yellow
  caution: "#F59E0B",   // Orange
  danger: "#EF4444",    // Red
  emergency: "#B91C1C", // Strong Red

  /* Navigation */
  forward: "#22C55E",
  left: "#1677FF",
  right: "#1677FF",
  stop: "#EF4444",

  /* Status */
  success: "#22C55E",
  warning: "#F59E0B",
  error: "#EF4444",
  info: "#63D8FF",

  /* Borders */
  border: "#24364D",
  borderLight: "#31465F",

  /* Overlay */
  overlay: "rgba(5, 12, 22, 0.90)",
  overlayLight: "rgba(5, 12, 22, 0.70)",

  transparent: "transparent",
} as const;

/* =========================================================
   RUNTIME BACKEND URL
   ========================================================= */

/**
 * Allows Settings screen to read the currently configured URL.
 */
export function getBackendUrl(): string {
  return BACKEND_URL;
}

/**
 * Runtime setter.
 *
 * NOTE:
 * This changes the URL for the current JavaScript runtime.
 * For permanent Android configuration, use .env.
 */
let runtimeBackendUrl = BACKEND_URL;

export function setBackendUrl(url: string): string {
  const cleaned = url.trim().replace(/\/+$/, "");

  if (!cleaned) {
    runtimeBackendUrl = BACKEND_URL;
  } else {
    runtimeBackendUrl = cleaned;
  }

  return runtimeBackendUrl;
}

/**
 * Returns runtime URL.
 */
export function getRuntimeBackendUrl(): string {
  return runtimeBackendUrl;
}

/* =========================================================
   BACKEND URL HELPERS
   ========================================================= */

export function getDetectUrl(): string {
  return `${runtimeBackendUrl}/detect`;
}

export function getHealthUrl(): string {
  return `${runtimeBackendUrl}/health`;
}

export function getSystemStatusUrl(): string {
  return `${runtimeBackendUrl}/system/status`;
}

/* =========================================================
   BACKEND HEALTH CHECK
   ========================================================= */

export async function checkBackendHealth(): Promise<boolean> {
  const controller = new AbortController();

  const timeout = setTimeout(() => {
    controller.abort();
  }, 5000);

  try {
    const response = await fetch(getHealthUrl(), {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
      signal: controller.signal,
    });

    if (!response.ok) {
      return false;
    }

    const data = await response.json();

    return (
      data?.success === true &&
      data?.status === "healthy"
    );
  } catch {
    return false;
  } finally {
    clearTimeout(timeout);
  }
}

/* =========================================================
   SAFETY HELPERS
   ========================================================= */

export function getSafetyColor(level?: string): string {
  switch ((level ?? "").toUpperCase()) {
    case "SAFE":
      return COLORS.safe;

    case "AWARE":
      return COLORS.aware;

    case "CAUTION":
      return COLORS.caution;

    case "DANGER":
      return COLORS.danger;

    case "EMERGENCY":
      return COLORS.emergency;

    default:
      return COLORS.muted;
  }
}

/* =========================================================
   ASSISTIVE DISTANCE & STEP CALIBRATION
   ========================================================= */

/** Average human walking step length in meters */
export const STEP_LENGTH_M = 0.75;

/** Minimum clamped step count for distance guidance */
export const MIN_STEP_COUNT = 1;

/** Maximum clamped step count for distance guidance */
export const MAX_STEP_COUNT = 8;

/**
 * Convert relative normalized MiDaS depth [0.0, 1.0] to approximate meters.
 *
 * Contract:
 * - 0.0 = very close (~0.5 m)
 * - 1.0 = far background (~6.0 m)
 */
export function depthToEstimatedDistance(relativeDepth: number): number {
  const clamped = Math.max(0.0, Math.min(1.0, Number(relativeDepth) || 0.5));
  // Approximate linear calibration: 0.5m + (clamped * 5.5m)
  const meters = 0.5 + clamped * 5.5;
  return Math.round(meters * 10) / 10;
}

/**
 * Calculate approximate walking steps from estimated distance in meters.
 * Clamped between 1 and 8 steps.
 */
export function distanceToSteps(
  distanceMeters: number,
  stepLengthM: number = STEP_LENGTH_M
): number {
  const steps = Math.round(distanceMeters / stepLengthM);
  return Math.max(MIN_STEP_COUNT, Math.min(MAX_STEP_COUNT, steps));
}

/* =========================================================
   MULTILINGUAL VOICE CONFIGURATION
   ========================================================= */

export type VoiceLanguage = "en" | "hi" | "mr";

let activeVoiceLanguage: VoiceLanguage = "en";

export function getVoiceLanguage(): VoiceLanguage {
  return activeVoiceLanguage;
}

export function setVoiceLanguage(language: VoiceLanguage): void {
  if (language === "en" || language === "hi" || language === "mr") {
    activeVoiceLanguage = language;
  }
}

export const LANGUAGE_LABELS: Record<
  VoiceLanguage,
  { name: string; nativeName: string }
> = {
  en: { name: "English", nativeName: "English" },
  hi: { name: "Hindi", nativeName: "हिन्दी" },
  mr: { name: "Marathi", nativeName: "मराठी" },
};