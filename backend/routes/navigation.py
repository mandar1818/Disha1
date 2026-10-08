"""
SmartVisionAI - Closed-Loop Destination Navigation API Routes
=============================================================
Endpoints:
    POST /navigation/start   - Start active destination navigation
    POST /navigation/stop    - Cancel / stop active navigation session
    GET  /navigation/status  - Query current navigation session status

Phase 4 Implementation:
- Direct integration with FrameProcessor and NavigationSession
- Thread-safe access to singleton processor
- Supports room/office/restroom/elevator/stairs/exit target navigation
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from backend.routes.detection import get_processor


router = APIRouter(
    prefix="/navigation",
    tags=["Navigation"],
)


class StartNavigationRequest(BaseModel):
    """
    Request payload to initiate closed-loop destination navigation.
    """
    destination: Optional[str] = Field(
        default=None,
        description="Target destination query string (e.g. 'Room 302', 'Restroom', 'Elevator', 'Exit')",
    )
    name: Optional[str] = Field(
        default=None,
        description="Optional explicit destination name",
    )
    type: Optional[str] = Field(
        default=None,
        description="Optional destination type (ROOM, OFFICE, RESTROOM, ELEVATOR, STAIRS, EXIT, etc.)",
    )
    identifier: Optional[str] = Field(
        default=None,
        description="Optional numeric or alphabetic room/office identifier (e.g. '302')",
    )


@router.post("/start")
def start_navigation(request: StartNavigationRequest) -> Dict[str, Any]:
    """
    Start closed-loop destination navigation session.
    """
    target = request.destination or request.name
    if not target and not request.identifier:
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": "No destination specified. Provide 'destination', 'name', or 'identifier'.",
            },
        )

    try:
        processor = get_processor()
        dest_spec: Any = target
        if request.type or request.identifier:
            dest_spec = {
                "name": target or f"{request.type or 'Target'} {request.identifier or ''}".strip(),
                "type": request.type or "ROOM",
                "identifier": request.identifier or "",
            }

        processor.start_destination(dest_spec)
        dest_name = processor.navigation_session.destination.name if (processor.navigation_session and processor.navigation_session.destination) else str(target)
        return {
            "success": True,
            "message": f"Navigation session initiated for '{dest_name}'.",
            "session": processor.get_navigation_status(),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": f"Failed to start navigation: {exc}",
            },
        )


@router.post("/stop")
def stop_navigation() -> Dict[str, Any]:
    """
    Stop and reset the active closed-loop destination navigation session.
    """
    try:
        processor = get_processor()
        session = processor.stop_destination()
        return {
            "success": True,
            "message": "Navigation session stopped.",
            "session": processor.get_navigation_status(),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": f"Failed to stop navigation: {exc}",
            },
        )


@router.get("/status")
def navigation_status() -> Dict[str, Any]:
    """
    Query the active closed-loop destination navigation session status.
    """
    try:
        processor = get_processor()
        return {
            "success": True,
            "session": processor.get_navigation_status(),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": f"Failed to retrieve navigation status: {exc}",
            },
        )
