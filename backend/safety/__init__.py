"""
============================================================
SmartVisionAI - Safety & Robustness Module
============================================================
Provides real-world failure-safe perception health diagnostics:
- Image quality evaluation (low-light, blur, glare, obstruction)
- ML model health and exception containment (YOLO, MiDaS)
- Mirror, glass, and reflection contradiction uncertainty
- Stale frame detection and safety timeouts
- Sudden scene flux shock absorber
- Crowd density & anti-oscillation lateral stabilization
============================================================
"""

from __future__ import annotations

from .health import (
    ImageQualityEvaluator,
    ModelHealthEvaluator,
    SceneStabilityEvaluator,
    PerceptionHealthEvaluator,
    PerceptionHealthStatus,
)

__all__ = [
    "ImageQualityEvaluator",
    "ModelHealthEvaluator",
    "SceneStabilityEvaluator",
    "PerceptionHealthEvaluator",
    "PerceptionHealthStatus",
]

