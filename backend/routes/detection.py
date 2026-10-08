"""
SmartVisionAI - Detection API Routes
====================================
Endpoints:
    POST /detect
    GET  /detect/status
    GET  /detect

Receives an image from the mobile application, passes it through:

    Image
      ↓
    YOLO
      ↓
    MiDaS
      ↓
    Decision Engine
      ↓
    Navigation + Voice Guidance

Phase 14 Real-Time Optimization:
- Non-blocking image decode inside thread worker to preserve asyncio event loop responsiveness
- Singleton FrameProcessor reuse with thread-safe warmup helper
- Retains exact response schema, backward compatibility aliases, and error handling
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, Optional

import cv2
import numpy as np

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.processors.frame_processor import FrameProcessor


router = APIRouter(
    prefix="",
    tags=["Detection"],
)


# ============================================================
# GLOBAL PROCESSOR
# ============================================================

_processor: Optional[FrameProcessor] = None
_processor_lock = threading.Lock()
_warmup_status: bool = False


def get_processor() -> FrameProcessor:
    """
    Return the shared FrameProcessor instance.

    Models are loaded only once instead of loading YOLO
    and MiDaS for every request.
    """

    global _processor

    if _processor is None:
        with _processor_lock:
            if _processor is None:
                print(
                    "[Detection] Creating FrameProcessor..."
                )
                _processor = FrameProcessor()
                print(
                    "[Detection] FrameProcessor ready."
                )

    return _processor


def warmup_processor() -> bool:
    """
    Pre-warm the shared FrameProcessor by running a dummy frame.
    Ensures first-request cold-start latency is completely eliminated.
    """
    global _warmup_status
    try:
        processor = get_processor()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = processor.process_frame(dummy_frame)
        _warmup_status = True
        print("[Detection] FrameProcessor successfully pre-warmed.")
        return True
    except Exception as exc:
        _warmup_status = False
        print(f"[Detection] Pre-warming warning (non-fatal): {exc}")
        return False


# ============================================================
# IMAGE DECODING
# ============================================================

def decode_image(
    data: bytes,
) -> np.ndarray:
    """
    Convert uploaded image bytes into an OpenCV image.
    Convert uploaded image bytes into an OpenCV image with memory guard.
    Prevents large multi-megapixel allocations (e.g. 12MP 37.7MB buffers).
    """

    if not data:
        raise ValueError(
            "Uploaded file is empty."
        )

    array = np.frombuffer(
        data,
        dtype=np.uint8,
    )

    if array.size == 0:
        raise ValueError(
            "Invalid image data."
        )

    try:
        image = cv2.imdecode(
            array,
            cv2.IMREAD_COLOR,
        )
    except (cv2.error, MemoryError) as exc:
        raise ValueError(
            f"Image decompression failed due to memory constraints: {exc}"
        ) from exc

    if image is None or image.size == 0:
        raise ValueError(
            "Unable to decode uploaded image or image is empty."
        )

    # Memory guard: downscale multi-megapixel frames (> 960px) to prevent
    # downstream OpenCV out-of-memory errors (e.g. Failed to allocate 37748736 bytes)
    h, w = image.shape[:2]
    max_dim = max(h, w)
    if max_dim > 960:
        scale = 960.0 / float(max_dim)
        new_w = max(32, int(round(w * scale)))
        new_h = max(32, int(round(h * scale)))
        downscaled = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )
        del image
        image = downscaled

    return image


def _decode_and_process(
    data: bytes,
    processor: FrameProcessor,
    destination: Optional[Dict[str, Any]] = None,
    timestamp: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Worker-thread helper: decodes image and processes frame without
    blocking the main FastAPI asynchronous event loop, ensuring prompt
    memory deallocation of intermediate buffers.
    """
    image = None
    try:
        image = decode_image(data)
        return processor.process_frame(image, destination=destination, timestamp=timestamp)
    finally:
        if image is not None:
            del image


# ============================================================
# HEALTH / STATUS
# ============================================================

@router.get(
    "/detect/status",
)
def detection_status() -> Dict[str, Any]:
    """
    Return detection pipeline status.
    """

    try:
        processor = get_processor()
        proc_status = processor.status()
        active_backend = getattr(processor, "active_backend", "pytorch")
        requested_backend = getattr(processor, "requested_backend", "pytorch")
        yolo_backend = getattr(processor.yolo, "backend", "pytorch")
        depth_backend = getattr(processor.midas, "backend", "pytorch")
        models_loaded = bool(getattr(processor, "loaded", True))
        fallback_active = bool(getattr(processor, "fallback_active", False))

        return {
            "success": True,
            "service": "SmartVisionAI Detection API",
            "loaded": models_loaded,
            "models_loaded": models_loaded,
            "active_backend": active_backend,
            "requested_backend": requested_backend,
            "yolo_backend": yolo_backend,
            "depth_backend": depth_backend,
            "fallback_active": fallback_active,
            "warmup_status": _warmup_status,
            "processor": proc_status,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(exc),
            },
        )


# ============================================================
# MAIN DETECTION ENDPOINT
# ============================================================

@router.post(
    "/detect",
)
async def detect(
    file: UploadFile = File(...),
    timestamp: Optional[float] = Form(None),
) -> Dict[str, Any]:
    """
    Process an uploaded camera frame.

    Mobile application sends:

        multipart/form-data

    with field:

        file

    Returns:

        YOLO detections
        MiDaS scene depth
        risk analysis
        safety level
        navigation instruction
        voice instruction
    """

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": "No file provided.",
            },
        )

    # --------------------------------------------------------
    # Validate content type when available
    # --------------------------------------------------------

    content_type = (
        file.content_type or ""
    ).lower()

    allowed_types = {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "image/bmp",
    }

    # Some mobile clients may not provide a content type.
    # Therefore we reject only known non-image types.
    if (
        content_type
        and content_type not in allowed_types
    ):
        if not content_type.startswith(
            "image/"
        ):
            raise HTTPException(
                status_code=415,
                detail={
                    "success": False,
                    "error":
                        "Unsupported file type.",
                    "content_type":
                        content_type,
                },
            )

    # --------------------------------------------------------
    # Read uploaded file
    # --------------------------------------------------------

    try:
        data = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error":
                    f"Unable to read uploaded file: {exc}",
            },
        )

    if not data:
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": "Uploaded file is empty.",
            },
        )

    # --------------------------------------------------------
    # Obtain processor
    # --------------------------------------------------------

    try:
        processor = get_processor()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error":
                    f"AI processor initialization failed: {exc}",
            },
        )

    # --------------------------------------------------------
    # Run heavy decode & AI processing outside async event loop
    # --------------------------------------------------------

    try:
        result = await asyncio.to_thread(
            _decode_and_process,
            data,
            processor,
            None,
            timestamp,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": str(exc),
            },
        )
    except Exception as exc:
        print(
            "[Detection] Processing error:",
            repr(exc),
        )
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error":
                    f"Frame processing failed: {exc}",
            },
        )

    # --------------------------------------------------------
    # Validate result
    # --------------------------------------------------------

    if not isinstance(
        result,
        dict,
    ):
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error":
                    "Frame processor returned invalid result.",
            },
        )

    # --------------------------------------------------------
    # Add API metadata
    # --------------------------------------------------------

    result["api"] = {
        "endpoint": "/detect",
        "filename": file.filename,
        "content_type": content_type,
    }

    return result


# ============================================================
# SIMPLE ROOT DETECTION CHECK
# ============================================================

@router.get(
    "/detect",
)
def detect_info() -> Dict[str, Any]:
    """
    Information endpoint.

    Useful for checking whether the API route exists.
    """

    return {
        "success": True,
        "service": "SmartVisionAI Detection",
        "method": "POST",
        "endpoint": "/detect",
        "upload_field": "file",
        "description":
            "Upload a camera image for object detection, "
            "depth estimation and navigation analysis.",
    }


# ============================================================
# EXPORT
# ============================================================

__all__ = [
    "router",
    "get_processor",
    "warmup_processor",
    "decode_image",
]