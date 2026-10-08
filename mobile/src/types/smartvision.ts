/**
 * SmartVisionAI
 * ============================
 * Shared TypeScript definitions
 *
 * These types are used by:
 * - API service
 * - Camera screen
 * - Finder
 * - Detection overlay
 * - Safety card
 * - Navigation card
 * - Object list
 * - Settings
 * - Backend status
 */

export type SafetyLevel =
  | "SAFE"
  | "AWARE"
  | "CAUTION"
  | "DANGER"
  | "EMERGENCY"
  | "UNKNOWN";

export type ObjectPosition = "LEFT" | "CENTER" | "RIGHT";

export type DistanceCategory =
  | "VERY_CLOSE"
  | "NEAR"
  | "FAR"
  | "UNKNOWN_DISTANCE";

export type DistanceConfidence = "HIGH" | "MEDIUM" | "LOW";

export type NavigationRelevance = "HIGH" | "MEDIUM" | "LOW" | "NONE";

export type NavigationAction =
  | "STOP"
  | "CONTINUE_FORWARD"
  | "GO_FORWARD"
  | "TURN_LEFT"
  | "TURN_RIGHT"
  | "MOVE_LEFT"
  | "MOVE_RIGHT"
  | "GO_BACK"
  | "WAIT";

export type NavigationDirection =
  | "LEFT"
  | "CENTER"
  | "RIGHT"
  | "FORWARD"
  | "BACK"
  | "NONE"
  | string;

export type NavigationStage = "TURN" | "MOVE" | "STOP" | "IDLE";
export type MovementAction =
  | "WALK_FORWARD"
  | "WALK_LEFT"
  | "WALK_RIGHT"
  | "WALK_BACK"
  | "NONE";
export type TurnDirection = "LEFT" | "RIGHT" | "NONE";

export type MotionState =
  | "STATIONARY"
  | "APPROACHING"
  | "RAPIDLY_APPROACHING"
  | "RECEDING"
  | "UNKNOWN_MOTION";

export type PersonState =
  | "PERSON_STATIC"
  | "PERSON_APPROACHING"
  | "PERSON_MOVING_AWAY"
  | "PERSON_CROSSING"
  | "PERSON_SIDE_MOVING"
  | "PERSON_GROUP"
  | "PERSON_UNCERTAIN";

export type CrowdSize =
  | "NONE"
  | "SINGLE"
  | "PAIR"
  | "GROUP"
  | "DENSE_CROWD";

export type CrowdDistribution =
  | "NONE"
  | "CENTER"
  | "LEFT"
  | "RIGHT"
  | "BOTH_SIDES"
  | "DISPERSED";

export type PredictiveThreatLevel =
  | "EMERGENCY"
  | "DANGER"
  | "CAUTION"
  | "AWARE"
  | "NONE";

export interface PredictiveThreat {
  obstacle_id?: number | string | null;
  track_id?: string;
  class_name: string;
  position: ObjectPosition | "NONE" | string;
  distance_m: number;
  motion_state: MotionState | string;
  approach_rate_mps: number;
  time_to_collision_s?: number | null;
  approach_score: number;
  threat_level: PredictiveThreatLevel | string;
  is_approaching: boolean;
  warning_issued: boolean;
}

/* =========================================================
   Phase 13 Connection & Sensor Integrity
   ========================================================= */

export type ConnectionState =
  | "CONNECTED"
  | "DEGRADED"
  | "LOST"
  | "RECOVERING";

export type CameraIntegrityState =
  | "NORMAL"
  | "DARK"
  | "OCCLUDED"
  | "DEGRADED";

export interface CameraIntegrityDiagnostics {
  state: CameraIntegrityState;
  mean_luminance?: number;
  variance?: number;
  message?: string;
  is_valid: boolean;
}

/* =========================================================
   Bounding Box
   ========================================================= */

export interface BoundingBoxCoordinates {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface BoundingBoxDimensions {
  x?: number;
  y?: number;
  width?: number;
  height?: number;

  x1?: number;
  y1?: number;
  x2?: number;
  y2?: number;
}

export type BoundingBox =
  | BoundingBoxCoordinates
  | BoundingBoxDimensions
  | number[];

/* =========================================================
   Detection Object (Known Object)
   ========================================================= */

export interface DetectionObject {
  id?: number;
  class_name?: string;
  confidence: number;
  bbox?: BoundingBoxCoordinates | BoundingBox;
  position?: ObjectPosition | string;
  distance_m?: number;
  distance_category?: DistanceCategory;
  distance_confidence?: DistanceConfidence;
  priority?: number | string;
  risk_score?: number;
  approaching?: boolean;
  approach_rate?: number;
  navigation_relevance?: NavigationRelevance;

  // Phase 11 Motion & Predictive Safety fields
  motion_state?: MotionState | string;
  approach_rate_mps?: number;
  time_to_collision_s?: number | null;
  approach_score?: number;
  is_approaching?: boolean;

  // Phase 10 Dynamic Scene & Person States
  person_state?: PersonState | string;
  is_crossing?: boolean;
  crossing_direction?: string;

  // Backward compatibility fields
  depth?: number | string;
  estimated_distance?: number | string;
  estimated_distance_m?: number;
  estimated_steps?: number;
  class_id?: number;
  name?: string;
  label?: string;
  direction?: string;
  obstacle_stability?: number;
  box?: number[];
  center?: number[];
  width?: number;
  height?: number;
  area?: number;
}

/* =========================================================
   Unknown Object Contract
   ========================================================= */

export interface UnknownObject {
  id: number;
  position: ObjectPosition;
  bbox: BoundingBoxCoordinates;
  distance_m: number;
  distance_category: DistanceCategory;
  distance_confidence: DistanceConfidence;
  risk_score: number;
  navigation_relevance: NavigationRelevance;
  confidence: number;

  // Phase 11 Motion & Predictive Safety fields
  motion_state?: MotionState | string;
  approach_rate_mps?: number;
  time_to_collision_s?: number | null;
  approach_score?: number;
  is_approaching?: boolean;
  approaching?: boolean;
  approach_rate?: number;
}

/* =========================================================
   Normalized Detection
   ========================================================= */

export interface NormalizedDetection extends DetectionObject {
  label: string;

  positionLabel:
    | "LEFT"
    | "CENTER"
    | "RIGHT";

  x1: number;
  y1: number;
  x2: number;
  y2: number;

  depthValue: number;

  riskScore: number;

  distanceText?: string;

  safetyLabel?: string;
}

/* =========================================================
   Frame Information
   ========================================================= */

export interface FrameInfo {
  width?: number;
  height?: number;
}

/* =========================================================
   Scene Depth
   ========================================================= */

export interface SceneDepth {
  average_depth?: number;
  center_depth?: number;
  left?: number;
  right?: number;
  left_depth?: number;
  center_region_depth?: number;
  right_depth?: number;
}

/* =========================================================
   Free Path
   ========================================================= */

export interface FreePath {
  left_clear?: boolean;
  center_clear?: boolean;
  right_clear?: boolean;
  left_depth?: number;
  center_depth?: number;
  right_depth?: number;
  left_risk?: number;
  center_risk?: number;
  right_risk?: number;
  best_direction?: "LEFT" | "CENTER" | "RIGHT" | "NONE" | string;

  // Backward compatibility
  left?: number;
  center?: number;
  right?: number;
  is_center_blocked?: boolean;

  // Phase 7 Safe Walking Corridor fields
  center_blocked?: boolean;
  left_blocked?: boolean;
  right_blocked?: boolean;
  center_coverage?: number;
  left_coverage?: number;
  right_coverage?: number;
  corridor_width?: number;
  no_safe_path?: boolean;
  safe_directions?: ("LEFT" | "CENTER" | "RIGHT" | string)[];
  path_confidence?: "HIGH" | "MEDIUM" | "LOW" | string;
}

/* =========================================================
   Multiple Objects Contract
   ========================================================= */

export interface MultipleObjects {
  object_count: number;
  left_obstacles: number;
  center_obstacles: number;
  right_obstacles: number;
  human_detected: boolean;
  vehicle_detected: boolean;
  highest_risk_object_id: number | null;
  closest_object_id: number | null;

  // Phase 10 Crowd & Dynamic Scene fields
  people_count?: number;
  crowd_size?: CrowdSize | string;
  crowd_distribution?: CrowdDistribution | string;
  crowd_center?: boolean;
  crowd_left?: boolean;
  crowd_right?: boolean;
  crowd_both_sides?: boolean;
  is_crowded?: boolean;
  crowd_blocking?: boolean;
  person_states?: (PersonState | string)[];
}

// Alias for backwards compatibility
export type MultiObjectReasoning = MultipleObjects & {
  important_objects?: DetectionObject[];
  center_obstacles_list?: DetectionObject[];
};

/* =========================================================
   Navigation Contract
   ========================================================= */

export interface NavigationResult {
  action?: NavigationAction;
  direction?: NavigationDirection;
  distance_m?: number;
  steps?: number;
  confidence?: DistanceConfidence | string;
  turn_required?: boolean;
  movement_required?: boolean;
  reason?: string;
  angle?: number;

  // Phase 9 turn-vs-movement separation & angle estimation
  turn_angle_deg?: number;
  turn_direction?: TurnDirection | string;
  movement_action?: MovementAction | string;
  navigation_stage?: NavigationStage | string;
  turn_confidence?: DistanceConfidence | string;

  // Phase 10 walking steps & movement distance guidance
  movement_distance_m?: number;
  movement_steps?: number;
  step_confidence?: DistanceConfidence | string;
  step_reason?: string;

  // Phase 3 Closed-Loop Navigation verification
  closed_loop_state?: string;
  environment_verified?: boolean;
}

/* =========================================================
   Step Guidance Contract
   ========================================================= */

export type StepGuidanceAction =
  | "WALK_FORWARD"
  | "WALK_BACK"
  | "STOP"
  | "NONE"
  | NavigationAction;

export interface StepGuidance {
  action: StepGuidanceAction | NavigationAction | string;
  steps: number;
  distance_m: number;
  confidence: DistanceConfidence | string;
  reason?: string;
}

/* =========================================================
   Safety
   ========================================================= */

export interface SafetyResult {
  level?: SafetyLevel | string;
  risk_score?: number;
}

/* =========================================================
   OCR Text Recognition Contract
   ========================================================= */

export interface OcrItem {
  text: string;
  confidence: number;
  position: ObjectPosition | string;
  bbox?: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  };
}

export interface OcrResult {
  detected: boolean;
  text_count: number;
  primary_text: string;
  items: OcrItem[];
  is_repeat?: boolean;
  should_speak?: boolean;
}

/* =========================================================
   Phase 2 Indoor Safety Perception Contracts
   ========================================================= */

export type DoorState = "OPEN" | "CLOSED" | "BLOCKED" | "POSSIBLE" | "NONE";

export interface DoorAnalysis {
  detected: boolean;
  state: DoorState;
  confidence: number;
  walkable: boolean;
  bbox?: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  } | null;
  interior_depth?: number;
  message?: string;
}

export type StairType = "STAIRS_DOWN" | "STAIRS_UP" | "DROP_OFF" | "NONE";
export type StairHazardLevel = "CRITICAL" | "WARNING" | "SAFE";

export interface StairAnalysis {
  detected: boolean;
  type: StairType;
  confidence: number;
  hazard_level: StairHazardLevel;
  distance_m: number;
  step_count_estimate?: number;
  message?: string;
}

export interface FloorHazard {
  class_name: string;
  bbox: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  };
  hazard_type: string;
  distance_m: number;
  risk_score: number;
  confidence: number;
  position: ObjectPosition | string;
  cand_depth?: number;
  description?: string;
}

export interface GlassObstacle {
  detected: boolean;
  confidence: number;
  position: ObjectPosition | string;
  risk_score: number;
  reason?: string;
}

export type CorridorClearanceState = "CLEAR" | "NARROW" | "BLOCKED";

export interface CorridorAnalysis {
  corridor_detected: boolean;
  walkable_width_m: number;
  clearance_state: CorridorClearanceState;
  vanishing_point?: {
    x: number;
    y: number;
  } | null;
  left_boundary_x?: number;
  right_boundary_x?: number;
  message?: string;
}

/* =========================================================
   Phase 3 Semantic Indoor Understanding & Landmark Layer
   ========================================================= */

export type SemanticLandmarkType =
  | "EXIT"
  | "ENTRANCE"
  | "RESTROOM"
  | "STAIRS"
  | "ELEVATOR"
  | "RECEPTION"
  | "ROOM"
  | "OFFICE"
  | "CAFETERIA"
  | "LOBBY"
  | "WARNING"
  | "EMERGENCY"
  | "INFORMATION"
  | "UNKNOWN_SIGN"
  | string;

export interface SemanticLandmark {
  type: SemanticLandmarkType;
  label: string;
  text: string;
  clean_text: string;
  identifier?: string | null;
  position: ObjectPosition | "CENTER" | "LEFT" | "RIGHT" | string;
  confidence: number;
  bbox?: BoundingBoxCoordinates | BoundingBox;
  is_navigable_cue: boolean;
  safety_relevance: "HIGH" | "MEDIUM" | "LOW" | string;
  should_announce: boolean;
}

/* =========================================================
   Phase 4 Closed-Loop Indoor Destination Navigation
   ========================================================= */

export type NavigationSessionState =
  | "IDLE"
  | "SEARCHING"
  | "APPROACHING"
  | "TURNING"
  | "VERIFYING"
  | "AVOIDING"
  | "ARRIVED"
  | "LOST"
  | "CANCELLED";

export type RelativeProximity =
  | "FAR"
  | "NEAR"
  | "VERY_CLOSE"
  | "UNKNOWN";

export type TurnVerificationState =
  | "NOT_REQUIRED"
  | "IN_PROGRESS"
  | "VERIFIED"
  | "FAILED"
  | "TIMEOUT";

export interface DestinationInfo {
  name: string;
  type: string;
  identifier?: string;
  raw_query?: string;
}

export interface TurnVerificationInfo {
  status: TurnVerificationState;
  pending_turn: string;
  frames_waited: number;
  visual_change_detected?: boolean;
  timeout_reached?: boolean;
}

export interface NavigationSessionInfo {
  active: boolean;
  state: NavigationSessionState;
  destination: DestinationInfo | null;
  destination_visible: boolean;
  destination_position: ObjectPosition | "UNKNOWN";
  progress: RelativeProximity;
  turn_verification: TurnVerificationInfo;
  arrival: boolean;
  arrival_announced?: boolean;
  safety_override?: boolean;
  speech_message?: string;
}

/* =========================================================
   Phase 5 Real-World Indoor Robustness & Perception Health
   ========================================================= */

export type PerceptionHealthStatus =
  | "HEALTHY"
  | "DEGRADED"
  | "UNCERTAIN"
  | "FAILED";

export type ImageQualityStatus = "GOOD" | "DEGRADED" | "POOR" | "OBSTRUCTED";

export interface ImageQualityInfo {
  status: ImageQualityStatus | string;
  brightness: number;
  contrast: number;
  blur_score: number;
  is_low_light: boolean;
  is_severely_dark: boolean;
  is_overexposed: boolean;
  is_low_contrast: boolean;
  is_blurred: boolean;
  is_obstructed: boolean;
  message?: string;
}

export interface ModelHealthInfo {
  status: PerceptionHealthStatus | string;
  yolo_valid: boolean;
  yolo_error?: string | null;
  depth_valid: boolean;
  depth_error?: string | null;
  message?: string;
}

export interface PerceptionHealthInfo {
  status: PerceptionHealthStatus;
  score: number;
  message: string;
  safety_override?: string | null;
  voice_override?: string | null;
  image_quality?: ImageQualityInfo;
  model_health?: ModelHealthInfo;
  reflection_uncertainty?: boolean;
  scene_flux?: boolean;
  scene_flux_diff?: number;
  is_crowded?: boolean;
  person_count?: number;
  stale_frame?: boolean;
  stale_age_s?: number;
}

/* =========================================================
   SmartVision Backend Response
   ========================================================= */

export interface SmartVisionResponse {
  success: boolean;

  ocr?: OcrResult;

  // Phase 2 Indoor Safety Perception Fields
  door_analysis?: DoorAnalysis;
  stair_analysis?: StairAnalysis;
  floor_hazards?: FloorHazard[];
  glass_obstacles?: GlassObstacle;
  corridor_analysis?: CorridorAnalysis;

  // Phase 3 Semantic Indoor Understanding Fields
  semantic_landmarks?: SemanticLandmark[];
  primary_semantic_sign?: SemanticLandmark | null;

  // Phase 4 Closed-Loop Indoor Destination Navigation Fields
  navigation_session?: NavigationSessionInfo;

  // Phase 5 Real-World Indoor Robustness & Perception Health Fields
  perception_health?: PerceptionHealthInfo;

  frame?: FrameInfo;

  objects?: DetectionObject[];

  unknown_objects?: UnknownObject[];

  scene?: SceneDepth;

  free_path?:
    | FreePath
    | boolean
    | Record<string, unknown>;

  multiple_objects?: MultipleObjects;

  safety_level?: SafetyLevel | string;

  emergency_stop?: boolean;

  primary_obstacle?: DetectionObject | null;

  navigation?:
    | NavigationDirection
    | NavigationResult;

  step_guidance?: StepGuidance;

  predictive_threat?: PredictiveThreat;

  approaching?: boolean;
  approach_rate?: number;
  obstacle_stability?: number;

  voice_instruction?: string;

  emergency_alert?: boolean;

  processing_time_ms?: number;

  api?: {
    endpoint?: string;
    service?: string;
    filename?: string;
    content_type?: string;
  };

  error?: string;

  // Backward compatibility aliases
  detections?: DetectionObject[];
  multi_object?: MultipleObjects;
  safety?: SafetyResult;
  navigation_confidence?: number;
  voice_message?: string;
  processing_time?: number;

  // Phase 13 Fail-Safe Connection & Sensor Integrity fields
  connection_state?: ConnectionState | string;
  camera_integrity?: CameraIntegrityState | string;
  camera_message?: string;
  guidance_suppressed?: boolean;
  connection_lost?: boolean;
  camera_occluded?: boolean;
  camera_obstructed?: boolean;
  sensor_degraded?: boolean;
  nav_state?: string;
  state_changed?: boolean;
  should_speak?: boolean;
}

/* =========================================================
   Backend Components
   ========================================================= */

export interface ComponentStatus {
  loaded?: boolean;

  is_loaded?: boolean;

  model?: string;

  confidence?: number;

  device?: string | null;

  engine?: string;

  emergency_threshold?: number;

  danger_threshold?: number;

  caution_threshold?: number;

  aware_threshold?: number;
}

/* =========================================================
   System Status
   ========================================================= */

export interface SystemComponents {
  loaded?: boolean;

  processor?: string;

  yolo?: ComponentStatus;

  midas?: ComponentStatus;

  decision_engine?: ComponentStatus;
}

export interface SystemStatusResponse {
  success: boolean;

  status: string;

  project?: string;

  components?: SystemComponents;

  error?: string;
}

/* =========================================================
   Health
   ========================================================= */

export interface HealthResponse {
  success: boolean;

  status: string;

  service?: string;

  timestamp?: string;

  error?: string;
}

/* =========================================================
   API Error
   ========================================================= */

export interface ApiError {
  success: false;

  error: string;
}

/* =========================================================
   Finder
   ========================================================= */

export interface FinderResult {
  found: boolean;

  target: string;

  detection?: NormalizedDetection;

  message: string;
}

/* =========================================================
   Camera State
   ========================================================= */

export interface CameraDetectionState {
  loading: boolean;

  lastResult: SmartVisionResponse | null;

  error: string | null;

  lastUpdated: number | null;

  // Phase 13 state fields
  connectionState?: ConnectionState;
  cameraIntegrity?: CameraIntegrityState;
}

/* =========================================================
   Settings
   ========================================================= */

export interface AppSettings {
  backendUrl: string;

  detectionEnabled: boolean;

  voiceEnabled: boolean;

  vibrationEnabled: boolean;

  confidenceThreshold: number;

  detectionIntervalMs: number;

  // Phase 13 settings
  hapticsEnabled?: boolean;
}

/* =========================================================
   Phase 12 Voice Decision Guidance
   ========================================================= */

export type VoicePriority = 1 | 2 | 3 | 4 | 5;

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

export interface VoiceManagerState {
  isSpeaking: boolean;
  lastSpokenText: string;
  lastSpokenAt: number;
  lastPriority: VoicePriority | 0;
  lastSafetyAlert: string;
  lastMotionTrend: string;
  lastDistanceCategory: string;
  lastThreatTrackId: string | number | null;
  lastAction: string;
  lastSteps: number | null;
  queuedInstruction: {
    text: string;
    priority: VoicePriority;
    action: string;
    timestamp: number;
  } | null;
}