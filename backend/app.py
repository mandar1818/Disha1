"""
SmartVisionAI - FastAPI Application
===================================
Main backend entry point.

Architecture:

Mobile Camera
      ↓
FastAPI
      ↓
Detection Route
      ↓
Frame Processor
      ├── YOLO
      ├── MiDaS
      └── Decision Engine
      ↓
Navigation + Voice Instruction

Phase 14 Real-Time Optimization:
- Startup pre-warming via warmup_processor() to eliminate initial cold-start delay
- Preserves all health, system status, and detection endpoints
- Thread-safe non-blocking event loop execution
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.detection import router as detection_router, warmup_processor
from backend.routes.navigation import router as navigation_router


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="SmartVisionAI",
    description=(
        "AI-based real-time assistive system for visually "
        "impaired users using object detection, monocular "
        "depth estimation and intelligent navigation."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROUTES
# ============================================================

app.include_router(
    detection_router
)
app.include_router(
    navigation_router
)


# ============================================================
# ROOT
# ============================================================

@app.get(
    "/",
    tags=["System"],
)
def root():
    """
    Basic API information.
    """

    return {
        "success": True,
        "project": "SmartVisionAI",
        "version": "1.0.0",
        "description": (
            "Real-time AI assistive system using "
            "YOLO + MiDaS + Decision Engine."
        ),
        "status": "running",
        "pipeline": [
            "Camera Frame",
            "YOLO Object Detection",
            "MiDaS Depth Estimation",
            "Decision Engine",
            "Navigation Decision",
            "Voice Guidance",
        ],
        "documentation": "/docs",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/health",
    tags=["System"],
)
def health():
    """
    Lightweight health check.

    Does not trigger heavy AI re-initialization.
    """

    return {
        "success": True,
        "status": "healthy",
        "service": "SmartVisionAI Backend",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }


# ============================================================
# AI SYSTEM STATUS
# ============================================================

@app.get(
    "/system/status",
    tags=["System"],
)
def system_status():
    """
    Check the complete AI pipeline.

    This endpoint initializes the shared processor if it
    has not already been initialized.
    """

    try:
        from backend.routes.detection import (
            get_processor,
        )

        processor = get_processor()

        return {
            "success": True,
            "status": "ready",
            "project": "SmartVisionAI",
            "components": processor.status(),
        }

    except Exception as exc:
        return {
            "success": False,
            "status": "error",
            "project": "SmartVisionAI",
            "error": str(exc),
        }


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():
    """
    FastAPI startup event.

    Phase 14 Optimization: Pre-warms YOLOv8n and MiDaS_small
    so the first real client frame does not experience cold-start latency.
    """

    print("=" * 60)
    print("SmartVisionAI Backend")
    print("=" * 60)
    print("FastAPI server starting...")

    # Pre-warm AI pipeline in background/safe block
    try:
        warmup_processor()
    except Exception as exc:
        print(f"[Startup] Warmup notice: {exc}")

    print("Documentation: http://127.0.0.1:8000/docs")
    print("Health:        http://127.0.0.1:8000/health")
    print("Detection:     POST /detect")
    print("Navigation:    POST /navigation/start, POST /navigation/stop")
    print("=" * 60)


# ============================================================
# SHUTDOWN
# ============================================================

@app.on_event("shutdown")
async def shutdown_event():
    """
    FastAPI shutdown event.
    """

    print("SmartVisionAI Backend stopped.")


# ============================================================
# EXPORT
# ============================================================

__all__ = [
    "app",
]