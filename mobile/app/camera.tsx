import React, {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import {
  CameraView,
  useCameraPermissions,
} from "expo-camera";

import { router } from "expo-router";
import { File, Paths, Directory } from "expo-file-system";

import { detectImage } from "src/services/api";
import {
  speak,
  speakEmergency,
  speakSafety,
  navigationToSpeech,
  generateAssistiveInstruction,
  stopSpeech,
  voiceManager,
  speakConnectionLost,
  speakConnectionRestored,
  speakCameraOccluded,
} from "src/services/speech";
import { connectionWatchdog } from "src/services/connectionWatchdog";
import { triggerHaptic } from "src/services/haptics";
import DetectionOverlay from "src/components/DetectionOverlay";

import type {
  DetectionObject,
  FreePath,
  NavigationResult,
  SmartVisionResponse,
  ConnectionState,
  CameraIntegrityState,
  CameraIntegrityDiagnostics,
} from "src/types/smartvision";

import {
  COLORS,
  DETECTION_INTERVAL_MS,
  IMAGE_QUALITY,
  MIN_CONFIDENCE,
  depthToEstimatedDistance,
  distanceToSteps,
} from "src/constants/config";


/* =========================================================
   HELPER FUNCTIONS
========================================================= */

/* =========================================================
   PHASE 13 SENSOR & CAMERA INTEGRITY DIAGNOSTICS
========================================================= */

/**
 * Formal Phase 13 Mathematical Specification for Pixel-Based Frame Integrity.
 *
 * Evaluates true pixel luminance and variance against Phase 13 thresholds:
 * - meanLuminance < 15: DARK (unsafe)
 * - variance < 10 (with meanLuminance < 35 for false-positive protection): OCCLUDED (unsafe)
 * - meanLuminance >= 15 && variance >= 10: NORMAL (safe)
 *
 * This function represents the exact Phase 13 pixel-level algorithm and is used
 * by automated unit/integration tests and pipelines providing true pixel data.
 */
export function evaluateFrameIntegrity(
  meanLuminance: number,
  variance: number
): CameraIntegrityDiagnostics {
  // 1. Mean luminance < 15 -> scene is too dark
  if (meanLuminance < 15) {
    return {
      state: "DARK",
      mean_luminance: meanLuminance,
      variance,
      message: "Scene too dark for reliable guidance.",
      is_valid: false,
    };
  }

  // 2. Variance < 10:
  // False-positive protection: if mean luminance is high (>= 35), it could be a well-lit uniform surface.
  // Severe obstruction is when luminance is low (< 35) and variance < 10.
  if (variance < 10) {
    if (meanLuminance < 35) {
      return {
        state: "OCCLUDED",
        mean_luminance: meanLuminance,
        variance,
        message: "Camera view obstructed or lens covered.",
        is_valid: false,
      };
    }
  }

  return {
    state: "NORMAL",
    mean_luminance: meanLuminance,
    variance,
    message: "Camera view normal.",
    is_valid: true,
  };
}

/**
 * Documented Mobile-Side Fallback Integrity Heuristic (Option B).
 *
 * IMPORTANT ARCHITECTURAL NOTE:
 * Expo Camera's takePictureAsync() outputs a compressed JPEG file on disk and does
 * NOT expose an uncompressed raw pixel buffer (RGBA/YUV) in JavaScript.
 *
 * Without adding custom native C++/Android modules (which is prohibited), the mobile
 * client cannot calculate true pixel-level luminance or variance directly.
 * Compressed JPEG payload size or byte entropy MUST NOT be conflated with pixel luminance.
 *
 * This function provides a safe, lightweight structural validation:
 * - Validates picture existence, URI, and frame dimensions.
 * - Leaves mean_luminance and variance undefined (does not generate fake numbers).
 * - Avoids false positives on ordinary valid scenes.
 * - Fails safely as DEGRADED if the frame cannot be captured or validated.
 */
export function diagnoseCapturedPicture(
  picture: { uri: string; width?: number; height?: number } | null
): CameraIntegrityDiagnostics {
  if (!picture || !picture.uri) {
    return {
      state: "DEGRADED",
      message: "No image captured from camera.",
      is_valid: false,
    };
  }

  if (
    typeof picture.width === "number" &&
    typeof picture.height === "number" &&
    (picture.width <= 0 || picture.height <= 0)
  ) {
    return {
      state: "DEGRADED",
      message: "Invalid frame dimensions.",
      is_valid: false,
    };
  }

  return {
    state: "NORMAL",
    message: "Camera frame valid (mobile fallback diagnostic).",
    is_valid: true,
  };
}

function getObjects(
  response: SmartVisionResponse | null
): DetectionObject[] {
  if (!response) {
    return [];
  }

  const list: DetectionObject[] = [];

  if (Array.isArray(response.objects)) {
    list.push(...response.objects);
  } else if (Array.isArray(response.detections)) {
    list.push(...response.detections);
  }

  if (Array.isArray(response.unknown_objects)) {
    for (const u of response.unknown_objects) {
      list.push({
        ...u,
        class_name: "Unknown Obstacle",
      });
    }
  }

  // Priority ranking: Humans > Vehicles > Hazards > Unknown Obstacles > Furniture > Other objects
  const getPriorityWeight = (obj: DetectionObject): number => {
    const name = (obj.class_name || "").toLowerCase();
    let base = 0.5;
    if (name.includes("person") || name.includes("human")) base = 1.0;
    else if (name.includes("car") || name.includes("vehicle") || name.includes("bike")) base = 0.95;
    else if (name.includes("stair") || name.includes("drop")) base = 0.90;
    else if (name.includes("door")) base = 0.70;
    else if (name.includes("chair") || name.includes("table")) base = 0.60;
    else if (name.includes("unknown")) base = 0.45;

    const risk = typeof obj.risk_score === "number" ? obj.risk_score : 0;
    const dist = typeof obj.distance_m === "number" && obj.distance_m > 0 ? obj.distance_m : 5.0;
    const proximityBonus = Math.max(0, 1.0 - dist / 5.0);

    return base * 0.4 + risk * 0.4 + proximityBonus * 0.2;
  };

  list.sort((a, b) => getPriorityWeight(b) - getPriorityWeight(a));
  return list;
}


function getObjectName(object: DetectionObject): string {
  return (
    object.class_name ||
    object.name ||
    object.label ||
    "Unknown"
  );
}


function getConfidence(object: DetectionObject): string {
  const value = Number(object.confidence);

  if (!Number.isFinite(value)) {
    return "0%";
  }

  return `${Math.round(value * 100)}%`;
}


function getSafetyLevel(
  response: SmartVisionResponse | null
): string {
  if (!response) {
    return "UNKNOWN";
  }

  if (response.safety_level) {
    return String(response.safety_level).toUpperCase();
  }

  if (response.safety?.level) {
    return String(response.safety.level).toUpperCase();
  }

  return "UNKNOWN";
}


function getNavigation(
  response: SmartVisionResponse | null
): string {
  if (!response?.navigation) {
    return "FORWARD";
  }

  if (typeof response.navigation === "string") {
    return response.navigation.toUpperCase();
  }

  if (response.navigation.action) {
    return String(response.navigation.action).toUpperCase();
  }

  if (response.navigation.direction) {
    return String(response.navigation.direction).toUpperCase();
  }

  return "FORWARD";
}


function getRiskScore(
  response: SmartVisionResponse | null
): number {
  if (!response) {
    return 0;
  }

  const direct = Number(response.safety?.risk_score);

  if (Number.isFinite(direct)) {
    return direct;
  }

  const obstacle = Number(
    response.primary_obstacle?.risk_score
  );

  if (Number.isFinite(obstacle)) {
    return obstacle;
  }

  return 0;
}


function getDirectionText(
  direction: string,
  navigationResult?: NavigationResult | null
): string {
  const value = direction.toUpperCase();

  if (navigationResult?.navigation_stage === "TURN" && navigationResult?.turn_angle_deg) {
    const deg = Math.round(navigationResult.turn_angle_deg);
    if (value === "TURN_LEFT" || value === "LEFT") {
      return `Turn Left about ${deg || 45} degrees`;
    }
    if (value === "TURN_RIGHT" || value === "RIGHT") {
      return `Turn Right about ${deg || 45} degrees`;
    }
  }

  if (navigationResult?.navigation_stage === "MOVE") {
    const steps = navigationResult.movement_steps ?? navigationResult.steps ?? 0;
    if (steps >= 2) {
      if (value === "GO_BACK" || navigationResult.movement_action === "WALK_BACK") {
        return `Move Back for about ${steps} steps`;
      }
      return `Move Forward for about ${steps} steps`;
    }
  }

  switch (value) {
    case "LEFT":
    case "MOVE_LEFT":
    case "KEEP_LEFT":
      return "Move Left";

    case "TURN_LEFT":
      return "Turn Left about 45 degrees";

    case "RIGHT":
    case "MOVE_RIGHT":
    case "KEEP_RIGHT":
      return "Move Right";

    case "TURN_RIGHT":
      return "Turn Right about 45 degrees";

    case "STOP":
      return "Stop";

    case "GO_BACK":
    case "BACK":
      return "Go Back";

    case "WAIT":
      return "Wait";

    case "SLOW_DOWN":
      return "Slow Down";

    case "FORWARD_CAREFULLY":
    case "GO_FORWARD":
      return "Move Forward Carefully";

    case "CONTINUE_FORWARD":
    case "FORWARD":
    default:
      return "Move Forward";
  }
}


let lastHapticState = "";
let lastHapticTime = 0;

function handleDetectionHaptics(response: SmartVisionResponse): void {
  const nav = getNavigation(response);
  const now = Date.now();
  const isEmergency =
    response.emergency_stop === true ||
    (response.safety_level || "").toUpperCase() === "EMERGENCY" ||
    nav === "STOP";

  const isRapid =
    response.predictive_threat?.motion_state === "RAPIDLY_APPROACHING" ||
    (response.primary_obstacle as { motion_state?: string } | undefined)?.motion_state === "RAPIDLY_APPROACHING" ||
    (response as { safety_alert?: string }).safety_alert === "WARNING_RAPID_APPROACH";

  if (isEmergency) {
    if (lastHapticState !== "EMERGENCY") {
      lastHapticState = "EMERGENCY";
      lastHapticTime = now;
      triggerHaptic("EMERGENCY");
    }
  } else if (isRapid) {
    if (lastHapticState !== "DANGER") {
      lastHapticState = "DANGER";
      lastHapticTime = now;
      triggerHaptic("DANGER");
    }
  } else if (nav === "TURN_LEFT") {
    if (lastHapticState !== "TURN_LEFT") {
      lastHapticState = "TURN_LEFT";
      lastHapticTime = now;
      triggerHaptic("TURN_LEFT");
    }
  } else if (nav === "TURN_RIGHT") {
    if (lastHapticState !== "TURN_RIGHT") {
      lastHapticState = "TURN_RIGHT";
      lastHapticTime = now;
      triggerHaptic("TURN_RIGHT");
    }
  } else if (nav === "MOVE_LEFT") {
    if (lastHapticState !== "MOVE_LEFT") {
      lastHapticState = "MOVE_LEFT";
      lastHapticTime = now;
      triggerHaptic("MOVE_LEFT");
    }
  } else if (nav === "MOVE_RIGHT") {
    if (lastHapticState !== "MOVE_RIGHT") {
      lastHapticState = "MOVE_RIGHT";
      lastHapticTime = now;
      triggerHaptic("MOVE_RIGHT");
    }
  } else if (nav === "SLOW_DOWN") {
    if (lastHapticState !== "SLOW_DOWN") {
      lastHapticState = "SLOW_DOWN";
      lastHapticTime = now;
      triggerHaptic("SLOW_DOWN");
    }
  } else if (nav === "GO_FORWARD" || nav === "FORWARD") {
    if (lastHapticState !== "FORWARD") {
      lastHapticState = "FORWARD";
      lastHapticTime = now;
      triggerHaptic("PATH_CLEAR");
    }
  } else {
    lastHapticState = nav;
  }
}

function pruneTemporaryCameraCache(): void {
  try {
    const cacheDir = Paths.cache;
    if (cacheDir && cacheDir.exists) {
      const items = cacheDir.list();
      for (const item of items) {
        if (item instanceof File) {
          const name = item.name.toLowerCase();
          if (
            name.endsWith(".jpg") ||
            name.endsWith(".jpeg") ||
            name.includes("smartvision") ||
            name.includes("camera")
          ) {
            try {
              item.delete();
            } catch {
              // Non-fatal
            }
          }
        }
      }
    }
  } catch {
    // Fail silently
  }
}

function handleDetectionSpeech(response: SmartVisionResponse): void {
  void voiceManager.handleResponse(response);
}


/* =========================================================
   CAMERA SCREEN
========================================================= */

export default function CameraScreen() {
  const [permission, requestPermission] =
    useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);

  const processingRef = useRef(false);
  const mountedRef = useRef(true);
  const frameCountRef = useRef(0);
  const latestResponseRef = useRef<SmartVisionResponse | null>(null);

  const lastNavTextRef = useRef("Move Forward");
  const lastVoiceMsgRef = useRef("");
  const lastSafetyLevelRef = useRef("SAFE");
  const lastEmergencyRef = useRef(false);

  const [processing, setProcessing] = useState(false);
  const [backendError, setBackendError] = useState("");

  // Lightweight UI display states (prevents wasted full-tree re-renders)
  const [navigationText, setNavigationText] = useState("Move Forward");
  const [voiceMessage, setVoiceMessage] = useState("");
  const [safetyLevel, setSafetyLevel] = useState("SAFE");
  const [isEmergency, setIsEmergency] = useState(false);
  const [detectedObjects, setDetectedObjects] = useState<DetectionObject[]>([]);
  const [frameDimensions, setFrameDimensions] = useState<{ width: number; height: number }>({ width: 640, height: 480 });
  const [primaryObstacleText, setPrimaryObstacleText] = useState("Clear");
  const [distanceText, setDistanceText] = useState("Path Clear");
  const [corridorClearance, setCorridorClearance] = useState({
    left: true,
    center: true,
    right: true,
  });

  // Phase 13 State Management
  const [connectionState, setConnectionState] =
    useState<ConnectionState>("CONNECTED");
  const [cameraIntegrity, setCameraIntegrity] =
    useState<CameraIntegrityState>("NORMAL");
  const [cameraIntegrityMessage, setCameraIntegrityMessage] =
    useState<string>("");

  /* =======================================================
     CLEANUP & WATCHDOG SUBSCRIPTION
  ======================================================= */
  useEffect(() => {
    mountedRef.current = true;
    connectionWatchdog.start();

    const unsubscribe = connectionWatchdog.subscribe((newState, prevState) => {
      if (!mountedRef.current) return;
      setConnectionState(newState);

      // Trigger safety alert & haptic on transition to LOST
      if (newState === "LOST" && prevState !== "LOST") {
        triggerHaptic("CONNECTION_LOST");
        void voiceManager.speakConnectionLost();
      }

      // Trigger restoration confirmation when recovered (RECOVERING -> CONNECTED)
      if (newState === "CONNECTED" && prevState === "RECOVERING") {
        void voiceManager.speakConnectionRestored();
      }
    });

    return () => {
      mountedRef.current = false;
      unsubscribe();
      connectionWatchdog.stop();
      processingRef.current = false;
      lastHapticState = "";
      lastHapticTime = 0;
      void stopSpeech();
      voiceManager.reset();
      pruneTemporaryCameraCache();
    };
  }, []);

  /* =======================================================
     AUTOMATIC CAMERA PERMISSION
  ======================================================= */
  useEffect(() => {
    if (permission && !permission.granted && permission.canAskAgain) {
      void requestPermission();
    }
  }, [permission, requestPermission]);

  useEffect(() => {
    if (permission && !permission.granted) {
      void speak(
        "Camera permission required. Please grant camera access to start detection."
      );
    }
  }, [permission?.granted]);

  /* =======================================================
     DETECT CURRENT FRAME (NON-BLOCKING, LATEST-FRAME)
  ======================================================= */
  const detectCurrentFrame = useCallback(async () => {
    if (
      processingRef.current ||
      !cameraRef.current ||
      !mountedRef.current
    ) {
      return;
    }

    processingRef.current = true;
    if (mountedRef.current) {
      setProcessing(true);
      setBackendError("");
    }

    const tStart = Date.now();
    let capturedUri: string | null = null;
    try {
      const picture =
        await cameraRef.current.takePictureAsync({
          quality: IMAGE_QUALITY,
          skipProcessing: true,
          shutterSound: false,
        });

      const tCapture = Date.now() - tStart;

      if (!picture?.uri) {
        throw new Error("Camera did not return an image.");
      }
      capturedUri = picture.uri;

      // 1. SENSOR / CAMERA INTEGRITY GUARD
      const integrity = diagnoseCapturedPicture(picture);
      if (mountedRef.current) {
        setCameraIntegrity(integrity.state);
        setCameraIntegrityMessage(integrity.message || "");
      }

      if (!integrity.is_valid) {
        triggerHaptic("CAMERA_OCCLUDED");
        void voiceManager.speakCameraOccluded();
        return;
      }

      // 2. DETECT IMAGE VIA BACKEND
      const tNetStart = Date.now();
      const result = await detectImage(picture.uri);
      const tNet = Date.now() - tNetStart;

      if (!mountedRef.current) {
        return;
      }

      latestResponseRef.current = result;

      if (result.success) {
        connectionWatchdog.recordSuccess();

        if (connectionWatchdog.isLost()) {
          setBackendError("Connection lost. Guidance paused.");
          return;
        }

        const currentSafety = getSafetyLevel(result);
        const emergencyStop =
          result.emergency_stop === true || currentSafety === "EMERGENCY";

        const rawNav = getNavigation(result);
        const navText = getDirectionText(
          rawNav,
          typeof result.navigation === "object" ? result.navigation : null
        );

        const assistive = generateAssistiveInstruction(result).text;
        const msg = result.voice_instruction || assistive || "";

        // Extract primary obstacle and distance
        let obsText = "Clear";
        let distText = "Path Clear";

        if (result.primary_obstacle) {
          const po = result.primary_obstacle;
          let name = po.class_name || (po as { name?: string }).name || (po as { label?: string }).label || "Obstacle";
          if (name.toLowerCase().includes("unknown")) {
            name = "Unknown Obstacle";
          } else {
            name = name.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());
          }
          const pos = po.position ? ` (${po.position})` : "";
          obsText = `${name}${pos}`;
          if (typeof po.distance_m === "number" && Number.isFinite(po.distance_m)) {
            distText = `${po.distance_m.toFixed(1)} m`;
          } else if (po.distance_category) {
            distText = po.distance_category.replace("_", " ");
          }
        } else if (result.objects && result.objects.length > 0) {
          const po = result.objects[0];
          let name = po.class_name || (po as { name?: string }).name || "Obstacle";
          if (name.toLowerCase().includes("unknown")) {
            name = "Unknown Obstacle";
          } else {
            name = name.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());
          }
          const pos = po.position ? ` (${po.position})` : "";
          obsText = `${name}${pos}`;
          if (typeof po.distance_m === "number" && Number.isFinite(po.distance_m)) {
            distText = `${po.distance_m.toFixed(1)} m`;
          }
        }

        // Extract corridor clearance from free_path
        let leftClear = true;
        let centerClear = true;
        let rightClear = true;
        if (result.free_path && typeof result.free_path === "object") {
          const fp = result.free_path as FreePath;
          if (typeof fp.left_clear === "boolean") leftClear = fp.left_clear;
          if (typeof fp.center_clear === "boolean") centerClear = fp.center_clear;
          if (typeof fp.right_clear === "boolean") rightClear = fp.right_clear;
        }

        setPrimaryObstacleText(obsText);
        setDistanceText(distText);
        setCorridorClearance({ left: leftClear, center: centerClear, right: rightClear });

        // Update lightweight display states
        setNavigationText(navText);
        setVoiceMessage(msg);
        setSafetyLevel(currentSafety);
        setIsEmergency(emergencyStop);
        lastNavTextRef.current = navText;
        lastVoiceMsgRef.current = msg;
        lastSafetyLevelRef.current = currentSafety;
        lastEmergencyRef.current = emergencyStop;
        setDetectedObjects(getObjects(result));
        if (result.frame?.width && result.frame?.height) {
          setFrameDimensions({
            width: result.frame.width,
            height: result.frame.height,
          });
        }
        setBackendError("");

        const tTtsStart = Date.now();
        try {
          handleDetectionSpeech(result);
          handleDetectionHaptics(result);
        } catch (speechError) {
          console.warn("[Camera] Speech/Haptic error:", speechError);
        }
        const tTts = Date.now() - tTtsStart;

        frameCountRef.current += 1;
        if (__DEV__) {
          console.log(
            `[LiveDetection] #${frameCountRef.current} capture=${tCapture}ms net=${tNet}ms tts=${tTts}ms total=${Date.now() - tStart}ms nav="${navText}"`
          );
        }
      } else {
        const errorMsg =
          result.error || "Backend could not process the frame.";
        connectionWatchdog.recordFailure(errorMsg);
        setBackendError(errorMsg);
        setDetectedObjects([]);
      }
    } catch (error) {
      if (!mountedRef.current) return;
      const message =
        error instanceof Error ? error.message : "Detection failed.";
      connectionWatchdog.recordFailure(message);
      setBackendError(message);
      setDetectedObjects([]);
    } finally {
      // Safely delete the temporary camera JPEG frame immediately
      if (capturedUri) {
        try {
          const tempFile = new File(capturedUri);
          tempFile.delete();
          if (tempFile.exists) {
            tempFile.delete();
          }
        } catch {
          // Ignore if file was already removed
        }
      }

      // Periodic sanitation every 20 frames to purge any orphaned camera temp files
      if (frameCountRef.current > 0 && frameCountRef.current % 20 === 0) {
        pruneTemporaryCameraCache();
      }

      processingRef.current = false;
      if (mountedRef.current) {
        setProcessing(false);
      }
    }
  }, []);

  /* =======================================================
     AUTOMATIC CONTINUOUS DETECTION LOOP (SELF-SCHEDULING)
     Cadence: Authoritative DETECTION_INTERVAL_MS (~2 FPS),
     no overlapping requests, minimum 50ms yield
  ======================================================= */
  useEffect(() => {
    if (!permission?.granted) {
      return;
    }

    let isLoopActive = true;
    let timerId: ReturnType<typeof setTimeout> | null = null;
    const TARGET_CADENCE_MS = DETECTION_INTERVAL_MS;

    const runLoop = async () => {
      if (!isLoopActive || !mountedRef.current) return;
      const cycleStart = Date.now();

      try {
        await detectCurrentFrame();
      } catch (err) {
        if (__DEV__) console.warn("[Camera] Loop error:", err);
      }

      if (!isLoopActive || !mountedRef.current) return;

      const elapsed = Date.now() - cycleStart;
      const delay = Math.max(50, TARGET_CADENCE_MS - elapsed);
      timerId = setTimeout(runLoop, delay);
    };

    // Auto-start continuous detection immediately
    runLoop();

    return () => {
      isLoopActive = false;
      if (timerId) clearTimeout(timerId);
    };
  }, [permission?.granted, detectCurrentFrame]);

  /* =======================================================
     PERMISSION LOADING
  ======================================================= */
  if (!permission) {
    return (
      <View style={styles.centerScreen}>
        <ActivityIndicator
          size="large"
          color={COLORS.primary}
        />
        <Text style={styles.loadingText}>
          Checking camera permission...
        </Text>
      </View>
    );
  }

  /* =======================================================
     CAMERA PERMISSION PROMPT
  ======================================================= */
  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.permissionScreen}>
        <View style={styles.permissionCard}>
          <Text style={styles.permissionIcon}>
            CAMERA
          </Text>
          <Text style={styles.permissionTitle}>
            Camera Access Required
          </Text>
          <Text style={styles.permissionText}>
            Disha requires camera access for real-time assistive guidance.
          </Text>
          <Pressable
            style={styles.primaryButton}
            onPress={requestPermission}
          >
            <Text style={styles.primaryButtonText}>
              ALLOW CAMERA
            </Text>
          </Pressable>
          <Pressable
            style={styles.secondaryButton}
            onPress={() => router.back()}
          >
            <Text style={styles.secondaryButtonText}>
              GO BACK
            </Text>
          </Pressable>
        </View>
      </SafeAreaView>
    );
  }

  const isConnectionLost = connectionState === "LOST";
  const isCameraDegraded = cameraIntegrity !== "NORMAL";

  const effectiveNavText = isConnectionLost
    ? "STOP (CONNECTION LOST)"
    : isCameraDegraded
    ? `STOP (${cameraIntegrity})`
    : navigationText;

  const effectiveVoiceText = isConnectionLost
    ? "Connection lost. Stop walking."
    : isCameraDegraded
    ? cameraIntegrityMessage || "Camera view obstructed or scene too dark. Please check camera view."
    : voiceMessage;

  /* =======================================================
     MAIN ASSISTIVE NAVIGATION UI (NO DASHBOARD)
  ======================================================= */
  return (
    <View style={styles.screen}>
      {/* 1. FULL-SCREEN LIVE CAMERA PREVIEW */}
      <CameraView
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        facing="back"
        mode="picture"
        pictureSize="640x480"
        mute={true}
        accessible={false}
        importantForAccessibility="no-hide-descendants"
        accessibilityElementsHidden={true}
      />

      {/* 1.1 REAL-TIME OBJECT BOUNDING BOXES */}
      <DetectionOverlay
        detections={detectedObjects}
        frameWidth={frameDimensions.width}
        frameHeight={frameDimensions.height}
      />

      {/* 2. TOP ACCESSIBLE CONTROLS */}
      <SafeAreaView style={styles.topSafeArea}>
        <View style={styles.topBar}>
          <Pressable
            style={styles.exitButton}
            onPress={() => router.push("/")}
            accessible={true}
            accessibilityLabel="Exit Camera"
            accessibilityRole="button"
          >
            <Text style={styles.exitButtonText}>EXIT CAMERA</Text>
          </Pressable>

          <View style={styles.statusPill}>
            <View
              style={[
                styles.statusDot,
                isConnectionLost
                  ? styles.dotLost
                  : isCameraDegraded
                  ? styles.dotDegraded
                  : styles.dotLive,
              ]}
            />
            <Text style={styles.statusText}>
              {isConnectionLost
                ? "LOST"
                : isCameraDegraded
                ? "OCCLUDED"
                : "LIVE ASSIST"}
            </Text>
          </View>

          <Pressable
            style={styles.settingsButton}
            onPress={() => router.push("/settings")}
            accessible={true}
            accessibilityLabel="Settings"
            accessibilityRole="button"
          >
            <Text style={styles.settingsButtonText}>SETTINGS</Text>
          </Pressable>
        </View>
      </SafeAreaView>

      {/* 3. SAFETY OVERLAYS (ALERT ROLES) */}
      {/* EMERGENCY STOP BANNER */}
      {isEmergency && !isConnectionLost && !isCameraDegraded && (
        <View style={styles.emergencyBanner} accessible={true} accessibilityRole="alert">
          <Text style={styles.emergencyTitle}>
            EMERGENCY STOP
          </Text>
          <Text style={styles.emergencyText}>
            Immediate obstacle detected. Stop immediately.
          </Text>
        </View>
      )}

      {/* CONNECTION LOST BANNER */}
      {isConnectionLost && (
        <View style={styles.connectionLostBanner} accessible={true} accessibilityRole="alert">
          <Text style={styles.connectionLostTitle}>
            CONNECTION LOST
          </Text>
          <Text style={styles.connectionLostText}>
            Backend connection lost. Stop walking and wait for automatic recovery.
          </Text>
        </View>
      )}

      {/* CAMERA INTEGRITY BANNER */}
      {isCameraDegraded && (
        <View style={styles.cameraDegradedBanner} accessible={true} accessibilityRole="alert">
          <Text style={styles.cameraDegradedTitle}>
            CAMERA VIEW {cameraIntegrity}
          </Text>
          <Text style={styles.cameraDegradedText}>
            {cameraIntegrityMessage || "Camera view obstructed or scene too dark. Please check camera view."}
          </Text>
        </View>
      )}

      {/* 4. BOTTOM ACCESSIBLE GUIDANCE CARD (NO DASHBOARD) */}
      <View style={styles.bottomContainer}>
        <View
          style={[
            styles.guidanceCard,
            isEmergency
              ? styles.guidanceEmergency
              : safetyLevel === "DANGER"
              ? styles.guidanceDanger
              : safetyLevel === "CAUTION"
              ? styles.guidanceCaution
              : safetyLevel === "AWARE"
              ? styles.guidanceAware
              : styles.guidanceSafe,
          ]}
          accessible={false}
          accessibilityElementsHidden={true}
          importantForAccessibility="no-hide-descendants"
        >
          <View style={styles.guidanceHeader}>
            <Text style={styles.guidanceTitle}>LIVE GUIDANCE</Text>
            <View
              style={[
                styles.safetyBadge,
                isEmergency
                  ? styles.safetyEmergency
                  : safetyLevel === "DANGER"
                  ? styles.safetyDanger
                  : safetyLevel === "CAUTION"
                  ? styles.safetyCaution
                  : safetyLevel === "AWARE"
                  ? styles.safetyAware
                  : styles.safetySafe,
              ]}
            >
              <Text style={styles.safetyBadgeText}>
                {isEmergency ? "EMERGENCY STOP" : safetyLevel}
              </Text>
            </View>
          </View>

          <Text
            style={[
              styles.navigationDirection,
              (isEmergency || isConnectionLost || isCameraDegraded) &&
                styles.navigationStop,
            ]}
          >
            {effectiveNavText}
          </Text>

          {/* Key Metrics Row: Distance & Primary Obstacle */}
          <View style={styles.metricsRow}>
            <View style={styles.metricItem}>
              <Text style={styles.metricLabel}>DISTANCE</Text>
              <Text style={styles.metricValue}>{distanceText}</Text>
            </View>
            <View style={styles.metricDivider} />
            <View style={styles.metricItem}>
              <Text style={styles.metricLabel}>OBSTACLE</Text>
              <Text style={styles.metricValue} numberOfLines={1}>{primaryObstacleText}</Text>
            </View>
          </View>

          {/* Corridor Clearance Chips */}
          <View style={styles.corridorRow}>
            <View style={[styles.corridorChip, corridorClearance.left ? styles.chipClear : styles.chipBlocked]}>
              <Text style={styles.chipText}>L: {corridorClearance.left ? "CLEAR" : "BLOCKED"}</Text>
            </View>
            <View style={[styles.corridorChip, corridorClearance.center ? styles.chipClear : styles.chipBlocked]}>
              <Text style={styles.chipText}>C: {corridorClearance.center ? "CLEAR" : "BLOCKED"}</Text>
            </View>
            <View style={[styles.corridorChip, corridorClearance.right ? styles.chipClear : styles.chipBlocked]}>
              <Text style={styles.chipText}>R: {corridorClearance.right ? "CLEAR" : "BLOCKED"}</Text>
            </View>
          </View>

          {effectiveVoiceText.length > 0 && (
            <Text style={styles.voiceGuidanceText}>
              {effectiveVoiceText}
            </Text>
          )}
        </View>
      </View>
    </View>
  );
}


/* =========================================================
   STYLES
========================================================= */

const styles = StyleSheet.create({

  screen: {
    flex: 1,
    backgroundColor: COLORS.background,
  },

  centerScreen: {
    flex: 1,
    backgroundColor: COLORS.background,
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
  },

  loadingText: {
    marginTop: 14,
    color: COLORS.white,
    fontSize: 15,
    fontWeight: "600",
  },

  permissionScreen: {
    flex: 1,
    backgroundColor: COLORS.background,
    justifyContent: "center",
    padding: 22,
  },

  permissionCard: {
    backgroundColor: COLORS.surface,
    borderRadius: 22,
    padding: 24,
    borderWidth: 1,
    borderColor: COLORS.border,
  },

  permissionIcon: {
    color: COLORS.primary,
    fontSize: 18,
    fontWeight: "900",
    marginBottom: 18,
  },

  permissionTitle: {
    color: COLORS.white,
    fontSize: 25,
    fontWeight: "900",
    marginBottom: 12,
  },

  permissionText: {
    color: COLORS.secondaryText,
    fontSize: 15,
    lineHeight: 23,
    marginBottom: 24,
  },

  primaryButton: {
    backgroundColor: COLORS.primary,
    borderRadius: 14,
    paddingVertical: 16,
    alignItems: "center",
    marginBottom: 12,
  },

  primaryButtonText: {
    color: COLORS.white,
    fontWeight: "900",
    fontSize: 14,
  },

  secondaryButton: {
    backgroundColor: COLORS.surfaceLight,
    borderRadius: 14,
    paddingVertical: 15,
    alignItems: "center",
  },

  secondaryButtonText: {
    color: COLORS.white,
    fontWeight: "900",
    fontSize: 14,
  },

  topSafeArea: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
  },

  topBar: {
    marginHorizontal: 14,
    marginTop: 8,
    backgroundColor: "rgba(7,16,28,0.90)",
    borderRadius: 18,
    padding: 12,
    flexDirection: "row",
    alignItems: "center",
    borderWidth: 1,
    borderColor: COLORS.border,
  },

  backButton: {
    backgroundColor: COLORS.surfaceLight,
    borderRadius: 10,
    paddingHorizontal: 11,
    paddingVertical: 9,
  },

  backText: {
    color: COLORS.white,
    fontSize: 11,
    fontWeight: "900",
  },

  titleContainer: {
    flex: 1,
    marginLeft: 12,
  },

  title: {
    color: COLORS.white,
    fontSize: 17,
    fontWeight: "900",
  },

  subtitle: {
    color: COLORS.secondaryText,
    fontSize: 10,
    fontWeight: "800",
    marginTop: 2,
  },

  liveIndicator: {
    borderRadius: 10,
    paddingHorizontal: 10,
    paddingVertical: 8,
  },

  liveIndicatorReady: {
    backgroundColor: COLORS.success,
  },

  liveIndicatorBusy: {
    backgroundColor: COLORS.warning,
  },

  liveText: {
    color: COLORS.white,
    fontSize: 10,
    fontWeight: "900",
  },

  statusContainer: {
    position: "absolute",
    top: 110,
    left: 14,
  },

  statusPill: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(7,16,28,0.90)",
    borderRadius: 20,
    paddingHorizontal: 12,
    paddingVertical: 9,
    borderWidth: 1,
    borderColor: COLORS.border,
  },

  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 8,
  },

  dotReady: {
    backgroundColor: COLORS.success,
  },

  dotLive: {
    backgroundColor: COLORS.success,
  },

  dotDegraded: {
    backgroundColor: COLORS.warning,
  },

  dotProcessing: {
    backgroundColor: COLORS.warning,
  },

  settingsButton: {
    backgroundColor: "rgba(255,255,255,0.12)",
    borderRadius: 12,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },

  settingsButtonText: {
    color: COLORS.white,
    fontSize: 12,
    fontWeight: "900",
  },

  statusText: {
    color: COLORS.white,
    fontSize: 11,
    fontWeight: "800",
  },

  emergencyBanner: {
    position: "absolute",
    top: 155,
    left: 14,
    right: 14,
    backgroundColor: "rgba(160,25,35,0.95)",
    borderRadius: 16,
    padding: 15,
    borderWidth: 1,
    borderColor: COLORS.danger,
  },

  emergencyTitle: {
    color: COLORS.white,
    fontSize: 18,
    fontWeight: "900",
  },

  emergencyText: {
    color: COLORS.white,
    marginTop: 4,
    lineHeight: 19,
  },

  dotLost: {
    backgroundColor: COLORS.danger,
  },

  connectionLostBanner: {
    position: "absolute",
    top: 155,
    left: 14,
    right: 14,
    backgroundColor: "rgba(180, 20, 30, 0.96)",
    borderRadius: 16,
    padding: 15,
    borderWidth: 1,
    borderColor: COLORS.danger,
    zIndex: 10,
  },

  connectionLostTitle: {
    color: COLORS.white,
    fontSize: 18,
    fontWeight: "900",
  },

  connectionLostText: {
    color: COLORS.white,
    marginTop: 4,
    lineHeight: 19,
    fontSize: 13,
  },

  cameraDegradedBanner: {
    position: "absolute",
    top: 155,
    left: 14,
    right: 14,
    backgroundColor: "rgba(190, 100, 15, 0.96)",
    borderRadius: 16,
    padding: 15,
    borderWidth: 1,
    borderColor: COLORS.warning,
    zIndex: 9,
  },

  cameraDegradedTitle: {
    color: COLORS.white,
    fontSize: 18,
    fontWeight: "900",
  },

  cameraDegradedText: {
    color: COLORS.white,
    marginTop: 4,
    lineHeight: 19,
    fontSize: 13,
  },

  bottomContainer: {
    position: "absolute",
    left: 0,
    right: 0,
    bottom: 0,
    maxHeight: "57%",
  },

  guidanceCard: {
    backgroundColor: "rgba(7,16,28,0.95)",
    borderRadius: 20,
    padding: 20,
    marginHorizontal: 16,
    marginBottom: 24,
    borderWidth: 2,
    borderColor: COLORS.border,
  },

  guidanceSafe: {
    borderColor: COLORS.safe,
    backgroundColor: "rgba(8, 26, 18, 0.95)",
  },

  guidanceAware: {
    borderColor: COLORS.aware,
    backgroundColor: "rgba(28, 26, 8, 0.95)",
  },

  guidanceCaution: {
    borderColor: COLORS.caution,
    backgroundColor: "rgba(32, 22, 10, 0.95)",
  },

  guidanceDanger: {
    borderColor: COLORS.danger,
    backgroundColor: "rgba(36, 12, 14, 0.95)",
  },

  guidanceEmergency: {
    borderColor: COLORS.emergency,
    backgroundColor: "rgba(48, 6, 8, 0.95)",
  },

  guidanceHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 10,
  },

  guidanceTitle: {
    color: COLORS.secondaryText,
    fontSize: 12,
    fontWeight: "900",
    letterSpacing: 1,
  },

  voiceGuidanceText: {
    color: COLORS.primary,
    fontSize: 16,
    fontWeight: "700",
    marginTop: 10,
    lineHeight: 22,
  },

  metricsRow: {
    flexDirection: "row",
    alignItems: "center",
    marginTop: 10,
    backgroundColor: "rgba(255, 255, 255, 0.07)",
    borderRadius: 12,
    paddingVertical: 8,
    paddingHorizontal: 12,
  },

  metricItem: {
    flex: 1,
  },

  metricLabel: {
    fontSize: 10,
    fontWeight: "800",
    color: COLORS.secondaryText,
    letterSpacing: 0.5,
  },

  metricValue: {
    fontSize: 14,
    fontWeight: "900",
    color: COLORS.white,
    marginTop: 2,
  },

  metricDivider: {
    width: 1,
    height: 24,
    backgroundColor: "rgba(255, 255, 255, 0.15)",
    marginHorizontal: 10,
  },

  corridorRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    gap: 8,
    marginTop: 8,
  },

  corridorChip: {
    flex: 1,
    paddingVertical: 5,
    borderRadius: 8,
    alignItems: "center",
    justifyContent: "center",
  },

  chipClear: {
    backgroundColor: "rgba(34, 197, 94, 0.22)",
    borderWidth: 1,
    borderColor: COLORS.safe,
  },

  chipBlocked: {
    backgroundColor: "rgba(239, 68, 68, 0.22)",
    borderWidth: 1,
    borderColor: COLORS.danger,
  },

  chipText: {
    fontSize: 11,
    fontWeight: "900",
    color: COLORS.white,
    letterSpacing: 0.5,
  },

  bottomContent: {
    paddingHorizontal: 14,
    paddingBottom: 30,
  },

  card: {
    backgroundColor: "rgba(7,16,28,0.94)",
    borderRadius: 18,
    padding: 16,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: COLORS.border,
  },

  cardHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 12,
  },

  cardTitle: {
    color: COLORS.white,
    fontSize: 14,
    fontWeight: "900",
    letterSpacing: 0.5,
  },

  safetyBadge: {
    borderRadius: 9,
    paddingHorizontal: 10,
    paddingVertical: 6,
  },

  safetySafe: {
    backgroundColor: COLORS.safe,
  },

  safetyAware: {
    backgroundColor: COLORS.aware,
  },

  safetyCaution: {
    backgroundColor: COLORS.caution,
  },

  safetyDanger: {
    backgroundColor: COLORS.danger,
  },

  safetyEmergency: {
    backgroundColor: COLORS.emergency,
  },

  safetyBadgeText: {
    color: COLORS.white,
    fontSize: 10,
    fontWeight: "900",
  },

  row: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 6,
  },

  rowLabel: {
    color: COLORS.secondaryText,
    fontSize: 13,
  },

  rowValue: {
    color: COLORS.white,
    fontSize: 13,
    fontWeight: "800",
  },

  dangerText: {
    color: COLORS.danger,
  },

  safeText: {
    color: COLORS.success,
  },

  navigationBox: {
    backgroundColor: COLORS.surfaceLight,
    borderRadius: 14,
    padding: 15,
    marginTop: 12,
    marginBottom: 8,
  },

  navigationDirection: {
    color: COLORS.primary,
    fontSize: 21,
    fontWeight: "900",
  },

  navigationStop: {
    color: COLORS.danger,
  },

  navigationReason: {
    color: COLORS.secondaryText,
    fontSize: 12,
    marginTop: 5,
    lineHeight: 18,
  },

  navigationSubtext: {
    color: COLORS.primary,
    fontSize: 13,
    fontWeight: "700",
    marginTop: 4,
  },

  obstacleRow: {
    flexDirection: "row",
    alignItems: "center",
    marginTop: 12,
  },

  obstacleInfo: {
    flex: 1,
  },

  objectName: {
    color: COLORS.white,
    fontSize: 16,
    fontWeight: "800",
  },

  objectDetails: {
    color: COLORS.secondaryText,
    fontSize: 12,
    marginTop: 3,
  },

  riskBox: {
    backgroundColor: COLORS.surfaceLight,
    borderRadius: 12,
    paddingHorizontal: 14,
    paddingVertical: 10,
    alignItems: "center",
    marginLeft: 12,
  },

  riskNumber: {
    color: COLORS.danger,
    fontSize: 18,
    fontWeight: "900",
  },

  riskLabel: {
    color: COLORS.secondaryText,
    fontSize: 9,
    fontWeight: "800",
    marginTop: 2,
  },

  countText: {
    color: COLORS.primary,
    fontSize: 15,
    fontWeight: "900",
  },

  emptyText: {
    color: COLORS.secondaryText,
    fontSize: 13,
    paddingVertical: 8,
  },

  objectRow: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 9,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },

  objectInfo: {
    flex: 1,
    marginRight: 12,
  },

  confidence: {
    color: COLORS.primary,
    fontSize: 13,
    fontWeight: "900",
  },

  voiceMessage: {
    color: COLORS.primary,
    fontSize: 15,
    lineHeight: 22,
    marginTop: 10,
    fontWeight: "700",
  },

  errorCard: {
    backgroundColor: "rgba(120,25,35,0.95)",
    borderRadius: 18,
    padding: 16,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: COLORS.danger,
  },

  errorTitle: {
    color: COLORS.white,
    fontSize: 14,
    fontWeight: "900",
  },

  errorText: {
    color: COLORS.white,
    fontSize: 13,
    lineHeight: 19,
    marginTop: 6,
  },

  errorHint: {
    color: "#ffd9dc",
    fontSize: 11,
    lineHeight: 17,
    marginTop: 8,
  },

  infoCard: {
    backgroundColor: "rgba(7,16,28,0.90)",
    borderRadius: 16,
    padding: 14,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: COLORS.border,
  },

  infoRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingVertical: 5,
  },

  infoLabel: {
    color: COLORS.secondaryText,
    fontSize: 11,
  },

  infoValue: {
    color: COLORS.white,
    fontSize: 11,
    fontWeight: "800",
  },

  scanButton: {
    backgroundColor: COLORS.primary,
    borderRadius: 15,
    minHeight: 52,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 10,
  },

  scanButtonText: {
    color: COLORS.white,
    fontSize: 14,
    fontWeight: "900",
  },

  exitButton: {
    backgroundColor: COLORS.surfaceLight,
    borderRadius: 15,
    minHeight: 50,
    alignItems: "center",
    justifyContent: "center",
  },

  exitButtonText: {
    color: COLORS.white,
    fontSize: 13,
    fontWeight: "900",
  },

  lastScanText: {
    color: COLORS.secondaryText,
    fontSize: 10,
    textAlign: "center",
    marginTop: 8,
  },

  emergencyAlertCard: {
    backgroundColor: "#d32f2f",
    borderRadius: 16,
    padding: 14,
    marginBottom: 12,
    borderWidth: 2,
    borderColor: "#ff5252",
    alignItems: "center",
  },

  emergencyAlertTitle: {
    color: COLORS.white,
    fontSize: 16,
    fontWeight: "900",
    letterSpacing: 0.8,
  },

  emergencyAlertSubtitle: {
    color: "#ffebee",
    fontSize: 11,
    fontWeight: "700",
    marginTop: 3,
  },

  approachingBadge: {
    backgroundColor: "rgba(255, 152, 0, 0.25)",
    borderColor: COLORS.warning,
    borderWidth: 1,
    borderRadius: 8,
    paddingHorizontal: 8,
    paddingVertical: 3,
    alignSelf: "flex-start",
    marginVertical: 4,
  },

  approachingBadgeText: {
    color: COLORS.warning,
    fontSize: 10,
    fontWeight: "900",
  },

  rapidApproachingBadge: {
    backgroundColor: "rgba(244, 67, 54, 0.28)",
    borderColor: COLORS.danger,
    borderWidth: 1.5,
    borderRadius: 8,
    paddingHorizontal: 8,
    paddingVertical: 3,
    alignSelf: "flex-start",
    marginVertical: 4,
  },

  rapidApproachingBadgeText: {
    color: COLORS.danger,
    fontSize: 10,
    fontWeight: "900",
    letterSpacing: 0.5,
  },

  objectDetailsHighlight: {
    color: COLORS.warning,
    fontSize: 12,
    fontWeight: "800",
    lineHeight: 18,
  },

});