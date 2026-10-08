/**
 * SmartVisionAI
 * API Service
 *
 * Mobile App <----HTTP----> FastAPI Backend
 *
 * Handles:
 * - Backend health check
 * - System status
 * - Image detection
 * - YOLO object detection
 * - MiDaS depth estimation
 * - Decision engine results
 * - Navigation / safety information
 * - Connection testing
 * - Request timeout handling
 */

import {
  BACKEND_URL,
  REQUEST_TIMEOUT_MS,
  getDetectUrl,
  getHealthUrl,
  getRuntimeBackendUrl,
  getSystemStatusUrl,
} from "src/constants/config";

import type {
  DetectionObject,
  HealthResponse,
  SmartVisionResponse,
  SystemStatusResponse,
} from "src/types/smartvision";

/* =========================================================
   TYPES
   ========================================================= */

export interface ApiResult<T> {
  ok: boolean;
  status: number;
  data: T | null;
  error?: string;
}

export interface BackendStatusResponse {
  success: boolean;
  status: string;
  message: string;
  backend_url: string;
  response_time_ms?: number;
  error?: string;
}

/* =========================================================
   INTERNAL HELPERS
   ========================================================= */

/**
 * Safely parse JSON response.
 */
async function parseJson<T>(response: Response): Promise<T | null> {
  try {
    return (await response.json()) as T;
  } catch {
    return null;
  }
}

/**
 * Generic HTTP request helper.
 */
async function request(
  url: string,
  options: RequestInit = {},
  timeoutMs: number = REQUEST_TIMEOUT_MS
): Promise<Response> {
  const controller = new AbortController();

  const timeout = setTimeout(() => {
    controller.abort();
  }, timeoutMs);

  try {
    return await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        Accept: "application/json",
        ...(options.headers ?? {}),
      },
    });
  } finally {
    clearTimeout(timeout);
  }
}

/**
 * Convert unknown error into readable text.
 */
function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }

  if (typeof error === "string") {
    return error;
  }

  return "Unknown network error.";
}

/* =========================================================
   BACKEND HEALTH
   ========================================================= */

/**
 * Check whether FastAPI backend is alive.
 *
 * Used by:
 * - Home screen
 * - Settings
 * - Camera status
 */
export async function checkBackendHealth(): Promise<boolean> {
  try {
    const response = await request(
      getHealthUrl(),
      {
        method: "GET",
      },
      5000
    );

    if (!response.ok) {
      return false;
    }

    const data = await parseJson<HealthResponse>(response);

    return (
      data?.success === true &&
      data?.status?.toLowerCase() === "healthy"
    );
  } catch {
    return false;
  }
}

/**
 * Get detailed backend health information.
 */
export async function getHealth(): Promise<HealthResponse> {
  try {
    const response = await request(
      getHealthUrl(),
      {
        method: "GET",
      },
      5000
    );

    const data = await parseJson<HealthResponse>(response);

    if (data) {
      return data;
    }

    return {
      success: false,
      status: "unhealthy",
      service: "SmartVisionAI Backend",
    };
  } catch (error) {
    return {
      success: false,
      status: "unreachable",
      service: getErrorMessage(error),
    };
  }
}

/* =========================================================
   CONNECTION TEST
   ========================================================= */

/**
 * Test connection to backend.
 *
 * Returns an object with `ok` so Settings can directly use:
 *
 * const result = await testConnection();
 * if (result.ok) { ... }
 */
export async function testConnection(
  backendUrl: string = BACKEND_URL
): Promise<BackendStatusResponse & { ok: boolean }> {
  const startTime = Date.now();

  const cleanUrl = backendUrl.replace(/\/+$/, "");

  const healthUrl = `${cleanUrl}/health`;

  try {
    const response = await request(
      healthUrl,
      {
        method: "GET",
      },
      5000
    );

    const responseTime = Date.now() - startTime;

    const data = await parseJson<HealthResponse>(response);

    if (response.ok && data?.success === true) {
      return {
        ok: true,
        success: true,
        status: data.status ?? "healthy",
        message: "Backend connection successful.",
        backend_url: cleanUrl,
        response_time_ms: responseTime,
      };
    }

    return {
      ok: false,
      success: false,
      status: data?.status ?? "error",
      message: `Backend returned HTTP ${response.status}.`,
      backend_url: cleanUrl,
      response_time_ms: responseTime,
      error: `HTTP ${response.status}`,
    };
  } catch (error) {
    return {
      ok: false,
      success: false,
      status: "unreachable",
      message: "Unable to connect to backend.",
      backend_url: cleanUrl,
      response_time_ms: Date.now() - startTime,
      error: getErrorMessage(error),
    };
  }
}

/* =========================================================
   SYSTEM STATUS
   ========================================================= */

/**
 * Get complete backend system status.
 *
 * Expected backend information:
 * - Processor
 * - YOLO
 * - MiDaS
 * - Decision Engine
 */
export async function getSystemStatus(
  backendUrl: string = BACKEND_URL
): Promise<SystemStatusResponse> {
  const cleanUrl = backendUrl.replace(/\/+$/, "");

  const url =
    backendUrl === BACKEND_URL
      ? getSystemStatusUrl()
      : `${cleanUrl}/system/status`;

  try {
    const response = await request(
      url,
      {
        method: "GET",
      },
      20000
    );

    const data = await parseJson<SystemStatusResponse>(response);

    if (data) {
      return {
        ...data,
        success: data.success === true,
      };
    }

    return {
      success: false,
      status: "unknown",
      project: "SmartVisionAI",
      error: `Invalid backend response (HTTP ${response.status}).`,
    };
  } catch (error) {
    return {
      success: false,
      status: "unreachable",
      project: "SmartVisionAI",
      error: getErrorMessage(error),
    };
  }
}

/* =========================================================
   IMAGE DETECTION
   ========================================================= */

/**
 * Send camera image to FastAPI backend.
 *
 * Backend pipeline:
 *
 * Camera Image
 *      ↓
 * FastAPI /detect
 *      ↓
 * YOLOv8
 *      ↓
 * MiDaS
 *      ↓
 * Decision Engine
 *      ↓
 * Safety + Navigation
 *      ↓
 * Mobile App
 */
/**
 * Low-level multipart upload using React Native's native XMLHttpRequest bridge.
 *
 * In Expo SDK 52-57, global `fetch` is intercepted by Expo's WinterCG fetch,
 * which does not support React Native FormData containing `{ uri, name, type }`
 * file attachments and throws "Unsupported FormData implementation".
 * React Native's native XMLHttpRequest directly supports `{ uri, name, type }`
 * via Android's OkHttp MultipartBody with native file streaming.
 */
function uploadFormData(
  url: string,
  formData: FormData,
  timeoutMs: number = REQUEST_TIMEOUT_MS
): Promise<{ status: number; text: string; ok: boolean }> {
  return new Promise((resolve, reject) => {
    let xhr: XMLHttpRequest | null = new XMLHttpRequest();

    const cleanup = () => {
      if (xhr) {
        xhr.onload = null;
        xhr.onerror = null;
        xhr.ontimeout = null;
        xhr = null;
      }
    };

    xhr.open("POST", url);
    xhr.timeout = timeoutMs;
    // NOTE: Do NOT set Content-Type header; React Native / OkHttp automatically
    // sets 'multipart/form-data; boundary=...' with the generated boundary.
    xhr.setRequestHeader("Accept", "application/json");

    xhr.onload = () => {
      const res = {
        status: xhr?.status ?? 0,
        text: xhr?.responseText ?? "",
        ok: (xhr?.status ?? 0) >= 200 && (xhr?.status ?? 0) < 300,
      };
      cleanup();
      resolve(res);
    };

    xhr.onerror = () => {
      const status = xhr?.status ?? 0;
      cleanup();
      const err = new Error(
        status
          ? `HTTP connection failed (status ${status})`
          : "Network unreachable or connection refused"
      );
      err.name = "NetworkError";
      reject(err);
    };

    xhr.ontimeout = () => {
      cleanup();
      const err = new Error(`Request timed out after ${timeoutMs} ms`);
      err.name = "TimeoutError";
      reject(err);
    };

    try {
      xhr.send(formData);
    } catch (sendError) {
      cleanup();
      const err = new Error(
        `Multipart/FormData send error: ${getErrorMessage(sendError)}`
      );
      err.name = "FormDataError";
      reject(err);
    }
  });
}

export async function detectImage(
  imageUri: string
): Promise<SmartVisionResponse> {
  if (!imageUri) {
    return {
      success: false,
      error: "No image was provided.",
    };
  }

  let formData: FormData;
  try {
    formData = new FormData();
    formData.append(
      "file",
      {
        uri: imageUri,
        name: `smartvision_${Date.now()}.jpg`,
        type: "image/jpeg",
      } as any
    );
    formData.append("timestamp", String(Date.now() / 1000.0));
  } catch (formError) {
    console.warn("[SmartVision API] FormData construction failed:", formError);
    return {
      success: false,
      error: `FormData construction error: ${getErrorMessage(formError)}`,
    };
  }

  const detectUrl = getDetectUrl();

  try {
    const LIVE_FRAME_TIMEOUT_MS = 4000;
    const result = await uploadFormData(
      detectUrl,
      formData,
      LIVE_FRAME_TIMEOUT_MS
    );

    let data: SmartVisionResponse;

    try {
      data = JSON.parse(result.text) as SmartVisionResponse;
    } catch (parseError) {
      console.warn(
        `[SmartVision API] Invalid JSON response (HTTP ${result.status}):`,
        result.text.slice(0, 200)
      );
      return {
        success: false,
        error: `Invalid JSON from backend (HTTP ${result.status}).`,
      };
    }

    if (!result.ok) {
      console.warn(
        `[SmartVision API] Backend HTTP ${result.status}:`,
        data.error || result.text
      );
      return {
        ...data,
        success: false,
        error:
          data.error ??
          `Backend error (HTTP ${result.status}).`,
      };
    }

    return {
      ...data,
      success: data.success !== false,
    };
  } catch (error: any) {
    const errorName = error?.name ?? "";
    const errorMessage = getErrorMessage(error);

    if (errorName === "TimeoutError") {
      console.warn("[SmartVision API] Request timed out:", detectUrl);
      return {
        success: false,
        error: "Backend request timed out.",
      };
    }

    if (errorName === "FormDataError") {
      console.warn("[SmartVision API] FormData error:", errorMessage);
      return {
        success: false,
        error: `FormData upload error: ${errorMessage}`,
      };
    }

    console.warn(
      `[SmartVision API] Network failure to ${detectUrl}:`,
      errorMessage
    );
    return {
      success: false,
      error: `Cannot reach backend at ${getRuntimeBackendUrl()}. ${errorMessage}`,
    };
  }
}

/* =========================================================
   DETECTION HELPERS
   ========================================================= */

/**
 * Return detections from either:
 *
 * response.detections
 *
 * or legacy:
 *
 * response.objects
 */
export function getDetections(
  response: SmartVisionResponse
): DetectionObject[] {
  if (Array.isArray(response.detections)) {
    return response.detections;
  }

  if (Array.isArray(response.objects)) {
    return response.objects;
  }

  return [];
}

/**
 * Get primary obstacle from backend.
 */
export function getPrimaryObstacle(
  response: SmartVisionResponse
): DetectionObject | null {
  if (response.primary_obstacle) {
    return response.primary_obstacle;
  }

  const detections = getDetections(response);

  if (detections.length === 0) {
    return null;
  }

  return [...detections].sort(
    (a, b) => (b.risk_score ?? 0) - (a.risk_score ?? 0)
  )[0];
}

/* =========================================================
   ERROR HELPERS
   ========================================================= */

/**
 * Determine whether API response contains an error.
 */
export function hasApiError(
  response: SmartVisionResponse
): boolean {
  return response.success === false || Boolean(response.error);
}

/**
 * Return readable API error.
 */
export function getApiError(
  response: SmartVisionResponse
): string | null {
  if (!hasApiError(response)) {
    return null;
  }

  return response.error ?? "SmartVisionAI backend error.";
}

/* =========================================================
   Phase 4 Closed-Loop Indoor Destination Navigation
   ========================================================= */

export async function startDestinationNavigation(
  destination: string,
  type?: string,
  identifier?: string
): Promise<{ success: boolean; message?: string; session?: any; error?: string }> {
  try {
    const url = `${getRuntimeBackendUrl()}/navigation/start`;
    const response = await request(
      url,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ destination, type, identifier }),
      },
      5000
    );
    const data = await parseJson<any>(response);
    return data ?? { success: false, error: "Invalid response" };
  } catch (error) {
    return { success: false, error: getErrorMessage(error) };
  }
}

export async function stopDestinationNavigation(): Promise<{
  success: boolean;
  message?: string;
  session?: any;
  error?: string;
}> {
  try {
    const url = `${getRuntimeBackendUrl()}/navigation/stop`;
    const response = await request(
      url,
      {
        method: "POST",
      },
      5000
    );
    const data = await parseJson<any>(response);
    return data ?? { success: false, error: "Invalid response" };
  } catch (error) {
    return { success: false, error: getErrorMessage(error) };
  }
}

export async function getDestinationNavigationStatus(): Promise<{
  success: boolean;
  session?: any;
  error?: string;
}> {
  try {
    const url = `${getRuntimeBackendUrl()}/navigation/status`;
    const response = await request(
      url,
      {
        method: "GET",
      },
      5000
    );
    const data = await parseJson<any>(response);
    return data ?? { success: false, error: "Invalid response" };
  } catch (error) {
    return { success: false, error: getErrorMessage(error) };
  }
}

/* =========================================================
   EXPORTS
   ========================================================= */

export { BACKEND_URL };