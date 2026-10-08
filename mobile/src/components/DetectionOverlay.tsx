import React from "react";
import {
  StyleSheet,
  Text,
  View,
  type ViewStyle,
} from "react-native";

import { COLORS } from "src/constants/config";
import type {
  BoundingBox,
  DetectionObject,
  NormalizedDetection,
} from "src/types/smartvision";

interface DetectionOverlayProps {
  detections?: DetectionObject[];
  objects?: DetectionObject[];
  frameWidth?: number;
  frameHeight?: number;
}

/**
 * Convert any backend bounding-box format into:
 * x1, y1, x2, y2
 *
 * Supported formats:
 *
 * 1. [x1, y1, x2, y2]
 *
 * 2. {
 *      x1,
 *      y1,
 *      x2,
 *      y2
 *    }
 *
 * 3. {
 *      x,
 *      y,
 *      width,
 *      height
 *    }
 */
function normalizeBoundingBox(
  bbox: BoundingBox | undefined,
  frameWidth: number,
  frameHeight: number
): {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
} | null {
  if (!bbox) {
    return null;
  }

  // ---------------------------------------------------------
  // Format 1: Array
  // [x1, y1, x2, y2]
  // ---------------------------------------------------------
  if (Array.isArray(bbox)) {
    if (bbox.length < 4) {
      return null;
    }

    const values = bbox.slice(0, 4).map(Number);

    if (values.some((value) => !Number.isFinite(value))) {
      return null;
    }

    return {
      x1: values[0],
      y1: values[1],
      x2: values[2],
      y2: values[3],
    };
  }

  // ---------------------------------------------------------
  // Format 2 / 3: Object
  // ---------------------------------------------------------
  if (typeof bbox === "object") {
    /*
     * TypeScript does not allow direct access to x/y here
     * because BoundingBoxCoordinates only contains x1/y1/x2/y2.
     *
     * We therefore explicitly check which object shape
     * the backend returned.
     */

    const box = bbox as {
      x?: number;
      y?: number;
      width?: number;
      height?: number;
      x1?: number;
      y1?: number;
      x2?: number;
      y2?: number;
    };

    // -------------------------------------------------------
    // Format: x1, y1, x2, y2
    // -------------------------------------------------------
    if (
      Number.isFinite(box.x1) &&
      Number.isFinite(box.y1) &&
      Number.isFinite(box.x2) &&
      Number.isFinite(box.y2)
    ) {
      return {
        x1: Number(box.x1),
        y1: Number(box.y1),
        x2: Number(box.x2),
        y2: Number(box.y2),
      };
    }

    // -------------------------------------------------------
    // Format: x, y, width, height
    // -------------------------------------------------------
    if (
      Number.isFinite(box.x) &&
      Number.isFinite(box.y) &&
      Number.isFinite(box.width) &&
      Number.isFinite(box.height)
    ) {
      return {
        x1: Number(box.x),
        y1: Number(box.y),
        x2: Number(box.x) + Number(box.width),
        y2: Number(box.y) + Number(box.height),
      };
    }
  }

  return null;
}

export function getSafetyBadge(riskScore: number): { label: string; color: string; bgAlpha: string } {
  if (riskScore >= 0.8) return { label: "EMERGENCY", color: COLORS.emergency, bgAlpha: "rgba(185, 28, 28, 0.20)" };
  if (riskScore >= 0.6) return { label: "DANGER", color: COLORS.danger, bgAlpha: "rgba(239, 68, 68, 0.18)" };
  if (riskScore >= 0.4) return { label: "CAUTION", color: COLORS.caution, bgAlpha: "rgba(245, 158, 11, 0.16)" };
  if (riskScore >= 0.2) return { label: "AWARE", color: COLORS.aware, bgAlpha: "rgba(234, 179, 8, 0.14)" };
  return { label: "SAFE", color: COLORS.safe, bgAlpha: "rgba(34, 197, 94, 0.12)" };
}

/**
 * Convert backend detection into a safe normalized object.
 */
function normalizeDetection(
  detection: DetectionObject,
  frameWidth: number,
  frameHeight: number
): NormalizedDetection | null {
  const box = normalizeBoundingBox(
    detection.bbox,
    frameWidth,
    frameHeight
  );

  if (!box) {
    return null;
  }

  let label =
    detection.class_name ??
    detection.name ??
    detection.label ??
    "Unknown Obstacle";

  if (label.toLowerCase().includes("unknown")) {
    label = "Unknown Obstacle";
  } else {
    label = label.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());
  }

  const rawPosition = String(detection.position ?? "CENTER").toUpperCase();

  const positionLabel: "LEFT" | "CENTER" | "RIGHT" =
    rawPosition === "LEFT"
      ? "LEFT"
      : rawPosition === "RIGHT"
        ? "RIGHT"
        : "CENTER";

  const depthValue =
    typeof detection.distance_m === "number" && Number.isFinite(detection.distance_m)
      ? detection.distance_m
      : typeof detection.depth === "number"
      ? detection.depth
      : typeof detection.estimated_distance === "number"
        ? detection.estimated_distance
        : Number(detection.depth ?? detection.estimated_distance ?? 0);

  const riskScore =
    typeof detection.risk_score === "number"
      ? detection.risk_score
      : 0;

  const distanceText =
    typeof detection.distance_m === "number" && Number.isFinite(detection.distance_m) && detection.distance_m > 0
      ? `${detection.distance_m.toFixed(1)}m`
      : typeof detection.estimated_distance_m === "number" && Number.isFinite(detection.estimated_distance_m) && detection.estimated_distance_m > 0
      ? `${detection.estimated_distance_m.toFixed(1)}m`
      : detection.distance_category && detection.distance_category !== "UNKNOWN_DISTANCE"
      ? detection.distance_category.replace("_", " ")
      : "";

  const safety = getSafetyBadge(riskScore);

  return {
    ...detection,
    label,
    positionLabel,
    x1: box.x1,
    y1: box.y1,
    x2: box.x2,
    y2: box.y2,
    depthValue: Number.isFinite(depthValue) ? depthValue : 0,
    riskScore,
    distanceText,
    safetyLabel: safety.label,
  };
}

/**
 * Clamp value to a valid range.
 */
function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Convert backend coordinates to screen percentages.
 */
function getBoxStyle(
  detection: NormalizedDetection,
  frameWidth: number,
  frameHeight: number
): ViewStyle {
  const x1 = clamp(detection.x1, 0, frameWidth);
  const y1 = clamp(detection.y1, 0, frameHeight);
  const x2 = clamp(detection.x2, 0, frameWidth);
  const y2 = clamp(detection.y2, 0, frameHeight);

  const left = (x1 / frameWidth) * 100;
  const top = (y1 / frameHeight) * 100;

  const width = ((x2 - x1) / frameWidth) * 100;
  const height = ((y2 - y1) / frameHeight) * 100;

  return {
    left: `${clamp(left, 0, 100)}%`,
    top: `${clamp(top, 0, 100)}%`,
    width: `${clamp(width, 0, 100)}%`,
    height: `${clamp(height, 0, 100)}%`,
  } as ViewStyle;
}

/**
 * Select box border color according to risk.
 */
function getRiskColor(detection: NormalizedDetection): string {
  if (detection.riskScore >= 0.8) {
    return COLORS.emergency; // strong red
  }

  if (detection.riskScore >= 0.6) {
    return COLORS.danger; // red
  }

  if (detection.riskScore >= 0.4) {
    return COLORS.caution; // orange
  }

  if (detection.riskScore >= 0.2) {
    return COLORS.aware; // yellow
  }

  return COLORS.safe; // green
}

function formatConfidence(confidence: number): string {
  if (!Number.isFinite(confidence)) {
    return "0%";
  }

  const value =
    confidence <= 1
      ? confidence * 100
      : confidence;

  return `${Math.round(clamp(value, 0, 100))}%`;
}

function formatDepth(value: number): string {
  if (!Number.isFinite(value) || value <= 0) {
    return "";
  }

  return `${value.toFixed(2)}`;
}

export default function DetectionOverlay({
  detections = [],
  objects = [],
  frameWidth = 640,
  frameHeight = 480,
}: DetectionOverlayProps) {
  /*
   * Some backend versions use `objects`,
   * while others use `detections`.
   *
   * Support both.
   */
  const source = detections.length > 0 ? detections : objects;

  const normalized = source
    .map((detection) =>
      normalizeDetection(
        detection,
        frameWidth,
        frameHeight
      )
    )
    .filter(
      (detection): detection is NormalizedDetection =>
        detection !== null
    );

  if (normalized.length === 0) {
    return null;
  }

  return (
    <View
      pointerEvents="none"
      style={StyleSheet.absoluteFill}
      accessible={false}
      accessibilityElementsHidden={true}
      importantForAccessibility="no-hide-descendants"
    >
      {normalized.map((detection, index) => {
        const boxStyle = getBoxStyle(
          detection,
          frameWidth,
          frameHeight
        );

        const safety = getSafetyBadge(detection.riskScore);
        const riskColor = safety.color;

        return (
          <View
            key={`${detection.label}-${index}-${detection.x1}`}
            style={[
              styles.box,
              boxStyle,
              {
                borderColor: riskColor,
                backgroundColor: safety.bgAlpha,
              },
            ]}
          >
            {/* Header Badge: Object Name + Metric Distance + Safety Level */}
            <View
              style={[
                styles.labelContainer,
                {
                  backgroundColor: riskColor,
                },
              ]}
            >
              <Text style={styles.label}>
                {detection.label}
              </Text>

              {detection.distanceText ? (
                <View style={styles.distanceBadge}>
                  <Text style={styles.distanceText}>
                    {detection.distanceText}
                  </Text>
                </View>
              ) : null}

              <View style={styles.safetyBadge}>
                <Text style={styles.safetyTag}>
                  {detection.safetyLabel || safety.label}
                </Text>
              </View>
            </View>

            {detection.positionLabel ? (
              <View
                style={[
                  styles.positionBadge,
                  {
                    backgroundColor: "rgba(0,0,0,0.85)",
                    borderColor: riskColor,
                    borderWidth: 1.5,
                  },
                ]}
              >
                <Text style={styles.positionText}>
                  {detection.positionLabel}
                </Text>
              </View>
            ) : null}
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  box: {
    position: "absolute",
    borderWidth: 2.5,
    borderRadius: 6,
    overflow: "visible",
  },

  labelContainer: {
    position: "absolute",
    top: -30,
    left: -2,
    minHeight: 26,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
  },

  label: {
    color: COLORS.white,
    fontSize: 12,
    fontWeight: "900",
  },

  distanceBadge: {
    backgroundColor: "rgba(0, 0, 0, 0.45)",
    paddingHorizontal: 5,
    paddingVertical: 2,
    borderRadius: 4,
  },

  distanceText: {
    color: COLORS.white,
    fontSize: 11,
    fontWeight: "900",
  },

  safetyBadge: {
    backgroundColor: "rgba(255, 255, 255, 0.25)",
    paddingHorizontal: 5,
    paddingVertical: 2,
    borderRadius: 4,
  },

  safetyTag: {
    color: COLORS.white,
    fontSize: 10,
    fontWeight: "900",
    letterSpacing: 0.5,
  },

  positionBadge: {
    position: "absolute",
    right: 3,
    bottom: 3,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },

  positionText: {
    color: COLORS.white,
    fontSize: 9,
    fontWeight: "900",
    letterSpacing: 0.5,
  },
});