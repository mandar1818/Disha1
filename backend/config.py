"""
============================================================
SmartVisionAI - Backend Configuration
============================================================
Defines active inference backends, model weights locations,
and fail-safe fallback policies.

Supported Backends:
    - "pytorch": PyTorch YOLOv8n (.pt) + Torch Hub MiDaS v2.1 Small
    - "tflite_onnx": TensorFlow Lite YOLOv8n (.tflite) + ONNX Runtime MiDaS (.onnx)
============================================================
"""

from __future__ import annotations

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------
# INFERENCE BACKEND SELECTION
# ------------------------------------------------------------
# Options: "pytorch" | "tflite_onnx"
# Default is "tflite_onnx" for Edge-AI TFLite+ONNX integration
INFERENCE_BACKEND = os.getenv("SMARTVISION_INFERENCE_BACKEND", "tflite_onnx").strip().lower()

# Automatic fallback to PyTorch if TFLite / ONNX fails or weights are missing
FALLBACK_TO_PYTORCH_ON_ERROR = True

# ------------------------------------------------------------
# MODEL PATHS & ASSETS
# ------------------------------------------------------------
# PyTorch models
YOLO_PYTORCH_PATH = str(PROJECT_ROOT / "yolov8n.pt")
YOLO_PYTORCH_MODEL = YOLO_PYTORCH_PATH
MIDAS_PYTORCH_MODEL = "MiDaS_small"

# Edge-AI TFLite & ONNX models
YOLO_TFLITE_PATH = str(PROJECT_ROOT / "edge_ai" / "models" / "yolov8n.tflite")
if not Path(YOLO_TFLITE_PATH).exists():
    # Fallback to root copy if edge_ai folder relocated
    alt_tflite = PROJECT_ROOT / "yolov8n.tflite"
    if alt_tflite.exists():
        YOLO_TFLITE_PATH = str(alt_tflite)
YOLO_TFLITE_MODEL = YOLO_TFLITE_PATH

COCO_CLASSES_PATH = str(PROJECT_ROOT / "edge_ai" / "config" / "coco_classes.txt")
COCO_CLASSES = COCO_CLASSES_PATH

MIDAS_ONNX_PATH = str(PROJECT_ROOT / "edge_ai" / "models" / "midas_small.onnx")
MIDAS_ONNX_MODEL = MIDAS_ONNX_PATH

# ------------------------------------------------------------
# INFERENCE HYPERPARAMETERS
# ------------------------------------------------------------
DEFAULT_YOLO_CONFIDENCE = 0.25
DEFAULT_YOLO_IOU_THRESHOLD = 0.45
DEFAULT_DEPTH_INPUT_SIZE = 256

