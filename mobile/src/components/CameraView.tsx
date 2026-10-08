import React, {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";

import {
  ActivityIndicator,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";

import {
  CameraView as ExpoCameraView,
  useCameraPermissions,
} from "expo-camera";

import { detectImage } from "src/services/api";
import {
  DETECTION_INTERVAL_MS,
  IMAGE_QUALITY,
  MIN_CONFIDENCE,
  COLORS,
} from "src/constants/config";

import type {
  DetectionObject,
  SmartVisionResponse,
} from "src/types/smartvision";

import DetectionOverlay from "src/components/DetectionOverlay";

export interface CameraViewHandle {
  captureAndDetect: () => Promise<SmartVisionResponse | null>;
  startDetection: () => void;
  stopDetection: () => void;
}

export interface CameraViewProps {
  /**
   * Automatically start backend detection after
   * camera permission is granted.
   *
   * Default: true
   */
  autoDetect?: boolean;

  /**
   * Called whenever a successful backend response
   * is received.
   */
  onDetection?: (
    response: SmartVisionResponse
  ) => void;

  /**
   * Called when backend/API detection fails.
   */
  onError?: (message: string) => void;

  /**
   * Called whenever detection starts/stops.
   */
  onDetectionStateChange?: (
    running: boolean
  ) => void;

  /**
   * Show detection bounding boxes.
   *
   * Default: true.
   */
  showOverlay?: boolean;

  /**
   * Detection interval in milliseconds.
   *
   * Default comes from config.ts.
   */
  detectionIntervalMs?: number;

  /**
   * Camera image quality.
   *
   * Default comes from config.ts.
   */
  imageQuality?: number;

  /**
   * Minimum confidence used to display
   * detections on the overlay.
   */
  minConfidence?: number;

  /**
   * Camera facing direction.
   */
  facing?: "front" | "back";
}

/**
 * SmartVisionAI Camera Component
 *
 * Responsibilities:
 *
 * Camera
 *   ↓
 * Capture frame
 *   ↓
 * FastAPI /detect
 *   ↓
 * YOLOv8 + MiDaS + Decision Engine
 *   ↓
 * SmartVisionResponse
 *   ↓
 * Parent screen
 *   ↓
 * DetectionOverlay / Safety / Navigation / Objects
 */
const CameraView = forwardRef<
  CameraViewHandle,
  CameraViewProps
>(function CameraView(
  {
    autoDetect = true,
    onDetection,
    onError,
    onDetectionStateChange,
    showOverlay = true,
    detectionIntervalMs = DETECTION_INTERVAL_MS,
    imageQuality = IMAGE_QUALITY,
    minConfidence = MIN_CONFIDENCE,
    facing = "back",
  },
  ref
) {
  const [
    permission,
    requestPermission,
  ] = useCameraPermissions();

  const cameraRef =
    useRef<ExpoCameraView>(null);

  const intervalRef =
    useRef<ReturnType<typeof setInterval> | null>(
      null
    );

  const mountedRef = useRef(true);

  const busyRef = useRef(false);

  const runningRef = useRef(false);

  const [running, setRunning] =
    useState(false);

  const [capturing, setCapturing] =
    useState(false);

  const [response, setResponse] =
    useState<SmartVisionResponse | null>(
      null
    );

  const [error, setError] =
    useState<string | null>(null);

  /*
   * Keep mounted state safe so async camera/API
   * operations don't update an unmounted screen.
   */
  useEffect(() => {
    mountedRef.current = true;

    return () => {
      mountedRef.current = false;

      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }

      runningRef.current = false;
      busyRef.current = false;
    };
  }, []);

  /**
   * Extract detections from either:
   *
   * response.detections
   *
   * or
   *
   * response.objects
   */
  const getDetections = useCallback(
    (
      data: SmartVisionResponse
    ): DetectionObject[] => {
      const source =
        data.detections ??
        data.objects ??
        [];

      return source.filter(
        (item) =>
          Number(item.confidence ?? 0) >=
          minConfidence
      );
    },
    [minConfidence]
  );

  /**
   * Capture one camera frame and send it
   * to the FastAPI backend.
   */
  const captureAndDetect =
    useCallback(async (): Promise<SmartVisionResponse | null> => {
      if (!cameraRef.current) {
        return null;
      }

      if (busyRef.current) {
        return null;
      }

      busyRef.current = true;

      if (mountedRef.current) {
        setCapturing(true);
        setError(null);
      }

      try {
        const photo =
          await cameraRef.current.takePictureAsync(
            {
              quality: imageQuality,
              skipProcessing: true,
              shutterSound: false,
            }
          );

        if (!photo?.uri) {
          throw new Error(
            "Camera did not return an image."
          );
        }

        const result =
          await detectImage(photo.uri);

        if (!mountedRef.current) {
          return result;
        }

        if (!result.success) {
          const message =
            result.error ??
            "Backend detection failed.";

          setError(message);
          onError?.(message);

          return result;
        }

        /*
         * Filter detections only for UI.
         *
         * We keep the original backend response
         * untouched so the parent receives all
         * backend information including:
         *
         * - safety
         * - navigation
         * - depth
         * - risk
         * - emergency stop
         * - primary obstacle
         */
        const detections =
          getDetections(result);

        const filteredResult: SmartVisionResponse =
          {
            ...result,
            detections,
          };

        setResponse(filteredResult);

        onDetection?.(filteredResult);

        return filteredResult;
      } catch (err) {
        const message =
          err instanceof Error
            ? err.message
            : "Camera detection failed.";

        if (mountedRef.current) {
          setError(message);
        }

        onError?.(message);

        return null;
      } finally {
        busyRef.current = false;

        if (mountedRef.current) {
          setCapturing(false);
        }
      }
    }, [
      getDetections,
      imageQuality,
      onDetection,
      onError,
    ]);

  /**
   * Stop automatic detection.
   */
  const stopDetection =
    useCallback(() => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }

      runningRef.current = false;

      if (mountedRef.current) {
        setRunning(false);
      }

      onDetectionStateChange?.(false);
    }, [onDetectionStateChange]);

  /**
   * Start automatic detection.
   */
  const startDetection =
    useCallback(() => {
      if (runningRef.current) {
        return;
      }

      if (!permission?.granted) {
        return;
      }

      runningRef.current = true;

      if (mountedRef.current) {
        setRunning(true);
        setError(null);
      }

      onDetectionStateChange?.(true);

      /*
       * Immediately process one frame.
       */
      void captureAndDetect();

      /*
       * Then continue at the configured interval.
       *
       * Default is normally 500 ms for SmartVisionAI.
       */
      intervalRef.current =
        setInterval(() => {
          void captureAndDetect();
        }, detectionIntervalMs);
    }, [
      captureAndDetect,
      detectionIntervalMs,
      onDetectionStateChange,
      permission?.granted,
    ]);

  /**
   * Expose camera controls to parent screens.
   */
  useImperativeHandle(
    ref,
    () => ({
      captureAndDetect,
      startDetection,
      stopDetection,
    }),
    [
      captureAndDetect,
      startDetection,
      stopDetection,
    ]
  );

  /**
   * Automatically request permission when
   * the component is mounted.
   */
  useEffect(() => {
    if (!permission) {
      return;
    }

    if (!permission.granted) {
      void requestPermission();
    }
  }, [
    permission,
    requestPermission,
  ]);

  /**
   * Automatically start detection once
   * permission has been granted.
   */
  useEffect(() => {
    if (
      permission?.granted &&
      autoDetect
    ) {
      startDetection();
    }

    return () => {
      stopDetection();
    };
  }, [
    permission?.granted,
    autoDetect,
    startDetection,
    stopDetection,
  ]);

  /**
   * Permission state not loaded yet.
   */
  if (!permission) {
    return (
      <View style={styles.center}>
        <ActivityIndicator
          size="large"
          color={COLORS.primary}
        />

        <Text style={styles.infoText}>
          Checking camera permission...
        </Text>
      </View>
    );
  }

  /**
   * Camera permission denied.
   */
  if (!permission.granted) {
    return (
      <View style={styles.center}>
        <Text style={styles.permissionIcon}>
          📷
        </Text>

        <Text style={styles.title}>
          Camera Permission Required
        </Text>

        <Text style={styles.description}>
          Disha needs camera access to
          detect objects, estimate depth and
          provide navigation guidance.
        </Text>

        <Pressable
          style={styles.primaryButton}
          onPress={() =>
            void requestPermission()
          }
        >
          <Text style={styles.buttonText}>
            ALLOW CAMERA
          </Text>
        </Pressable>
      </View>
    );
  }

  const detections =
    response?.detections ??
    response?.objects ??
    [];

  return (
    <View style={styles.container}>
      {/* -------------------------------------------------- */}
      {/* REAL CAMERA                                        */}
      {/* -------------------------------------------------- */}

      <ExpoCameraView
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        facing={facing}
        mute={true}
      />

      {/* -------------------------------------------------- */}
      {/* DETECTION OVERLAY                                  */}
      {/* -------------------------------------------------- */}

      {showOverlay && (
        <DetectionOverlay
          detections={detections}
          frameWidth={
            response?.frame?.width ?? 640
          }
          frameHeight={
            response?.frame?.height ?? 480
          }
        />
      )}

      {/* -------------------------------------------------- */}
      {/* TOP STATUS                                         */}
      {/* -------------------------------------------------- */}

      <View style={styles.topStatus}>
        <View
          style={[
            styles.statusDot,
            {
              backgroundColor: running
                ? COLORS.safe
                : COLORS.danger,
            },
          ]}
        />

        <Text style={styles.statusText}>
          {running
            ? "LIVE DETECTION"
            : "DETECTION STOPPED"}
        </Text>

        {response?.perception_health && response.perception_health.status !== "HEALTHY" && (
          <View
            style={[
              styles.healthBadge,
              {
                backgroundColor:
                  response.perception_health.status === "FAILED"
                    ? COLORS.danger
                    : response.perception_health.status === "UNCERTAIN"
                      ? COLORS.caution
                      : COLORS.aware,
              },
            ]}
            accessible={true}
            accessibilityLabel={`Vision health: ${response.perception_health.status}. ${response.perception_health.message}`}
          >
            <Text style={styles.healthText}>
              {response.perception_health.status}
            </Text>
          </View>
        )}

        {capturing && (
          <ActivityIndicator
            size="small"
            color={COLORS.white}
          />
        )}
      </View>

      {/* -------------------------------------------------- */}
      {/* BACKEND ERROR                                      */}
      {/* -------------------------------------------------- */}

      {error && (
        <View style={styles.errorCard}>
          <Text style={styles.errorTitle}>
            Detection Error
          </Text>

          <Text style={styles.errorText}>
            {error}
          </Text>
        </View>
      )}

      {/* -------------------------------------------------- */}
      {/* BOTTOM CONTROLS                                    */}
      {/* -------------------------------------------------- */}

      <View style={styles.controls}>
        <View style={styles.resultInfo}>
          <Text style={styles.resultTitle}>
            Disha
          </Text>

          <Text style={styles.resultText}>
            {detections.length > 0
              ? `${detections.length} object${
                  detections.length === 1
                    ? ""
                    : "s"
                } detected`
              : running
                ? "Scanning environment..."
                : "Detection paused"}
          </Text>

          {response?.safety_level && (
            <Text
              style={styles.safetyText}
            >
              Safety:{" "}
              {response.safety_level}
            </Text>
          )}

          {response?.navigation && (
            <Text
              style={styles.navigationText}
            >
              Navigation:{" "}
              {typeof response.navigation ===
              "string"
                ? response.navigation
                : response.navigation
                    ?.direction ??
                  "UNKNOWN"}
            </Text>
          )}
        </View>

        <View style={styles.buttonRow}>
          <Pressable
            style={[
              styles.controlButton,
              running
                ? styles.stopButton
                : styles.startButton,
            ]}
            onPress={() => {
              if (running) {
                stopDetection();
              } else {
                startDetection();
              }
            }}
          >
            <Text style={styles.buttonText}>
              {running
                ? "STOP"
                : "START"}
            </Text>
          </Pressable>

          <Pressable
            style={[
              styles.controlButton,
              styles.scanButton,
            ]}
            disabled={capturing}
            onPress={() => {
              void captureAndDetect();
            }}
          >
            {capturing ? (
              <ActivityIndicator
                size="small"
                color={COLORS.white}
              />
            ) : (
              <Text
                style={styles.buttonText}
              >
                SCAN
              </Text>
            )}
          </Pressable>
        </View>
      </View>
    </View>
  );
});

export default CameraView;

/* ========================================================= */
/* STYLES                                                    */
/* ========================================================= */

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.bg,
  },

  center: {
    flex: 1,
    backgroundColor: COLORS.bg,
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
  },

  permissionIcon: {
    fontSize: 48,
    marginBottom: 16,
  },

  title: {
    color: COLORS.white,
    fontSize: 22,
    fontWeight: "900",
    textAlign: "center",
    marginBottom: 10,
  },

  description: {
    color: COLORS.muted,
    fontSize: 15,
    lineHeight: 22,
    textAlign: "center",
    marginBottom: 24,
  },

  infoText: {
    color: COLORS.muted,
    fontSize: 14,
    marginTop: 14,
  },

  primaryButton: {
    backgroundColor: COLORS.primary,
    paddingHorizontal: 28,
    paddingVertical: 15,
    borderRadius: 14,
  },

  buttonText: {
    color: COLORS.white,
    fontSize: 13,
    fontWeight: "900",
    letterSpacing: 0.5,
  },

  topStatus: {
    position: "absolute",
    top: 50,
    left: 16,
    right: 16,
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    backgroundColor:
      "rgba(5,12,22,0.82)",
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 14,
  },

  statusDot: {
    width: 9,
    height: 9,
    borderRadius: 5,
  },

  statusText: {
    flex: 1,
    color: COLORS.white,
    fontSize: 12,
    fontWeight: "900",
  },

  healthBadge: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 8,
  },

  healthText: {
    color: COLORS.white,
    fontSize: 10,
    fontWeight: "900",
    letterSpacing: 0.5,
  },

  errorCard: {
    position: "absolute",
    top: 105,
    left: 16,
    right: 16,
    backgroundColor:
      "rgba(110,20,30,0.92)",
    borderRadius: 14,
    padding: 13,
    borderWidth: 1,
    borderColor: COLORS.danger,
  },

  errorTitle: {
    color: COLORS.white,
    fontSize: 13,
    fontWeight: "900",
    marginBottom: 4,
  },

  errorText: {
    color: "#FFD9DD",
    fontSize: 12,
    lineHeight: 18,
  },

  controls: {
    position: "absolute",
    left: 12,
    right: 12,
    bottom: 18,
    backgroundColor:
      "rgba(5,12,22,0.94)",
    borderRadius: 20,
    padding: 14,
  },

  resultInfo: {
    marginBottom: 12,
  },

  resultTitle: {
    color: COLORS.white,
    fontSize: 18,
    fontWeight: "900",
  },

  resultText: {
    color: COLORS.muted,
    fontSize: 13,
    marginTop: 3,
  },

  safetyText: {
    color: COLORS.caution,
    fontSize: 12,
    fontWeight: "800",
    marginTop: 5,
  },

  navigationText: {
    color: COLORS.accent,
    fontSize: 12,
    fontWeight: "800",
    marginTop: 3,
  },

  buttonRow: {
    flexDirection: "row",
    gap: 10,
  },

  controlButton: {
    flex: 1,
    minHeight: 46,
    borderRadius: 13,
    alignItems: "center",
    justifyContent: "center",
  },

  startButton: {
    backgroundColor: COLORS.safe,
  },

  stopButton: {
    backgroundColor: COLORS.danger,
  },

  scanButton: {
    backgroundColor: COLORS.primary,
  },
});