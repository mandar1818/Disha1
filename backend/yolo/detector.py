"""
============================================================
SmartVisionAI
============================================================

File:
    backend/yolo/detector.py

Purpose:
    YOLOv8 object detection service.
    Supports dual execution backends:
      - "pytorch": Ultralytics YOLOv8 (.pt)
      - "tflite": TensorFlow Lite YOLOv8 (.tflite) Edge-AI model

Responsibilities:
    - Load YOLOv8 model once (PyTorch or TFLite)
    - Detect objects from images
    - Apply confidence and NMS filtering
    - Return standardized detection dictionaries
    - Provide model status
    - Handle different image input types
    - Keep output JSON-friendly
    - Ensure identical schema across PyTorch and TFLite backends

Expected detection format:

{
    "class_name": "person",
    "class_id": 0,
    "confidence": 0.92,
    "bbox": {
        "x1": 100.0,
        "y1": 50.0,
        "x2": 300.0,
        "y2": 500.0
    }
}

============================================================
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
# ------------------------------------------------------------
# Lightweight Runtime Resolvers (No heavy imports at import-time)
# ------------------------------------------------------------
def _get_tflite_interpreter_class() -> Any:
    """
    Resolve a lightweight TFLite Interpreter without importing monolithic TensorFlow.
    Tries runtimes in order:
      1. ai_edge_litert (Official Google lightweight LiteRT for Python 3.11+)
      2. tflite_runtime (Legacy official lightweight runtime)
      3. Isolated tensorflow.lite interpreter submodule (fallback)
    """
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter
    except (ImportError, Exception):
        pass

    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter
    except (ImportError, Exception):
        pass

    try:
        from tensorflow.lite.python.interpreter import Interpreter
        return Interpreter
    except (ImportError, Exception):
        pass

    try:
        import sys
        if "tensorflow" in sys.modules:
            return sys.modules["tensorflow"].lite.Interpreter
    except Exception:
        pass

    return None


_YOLO_CLASS: Any = None

def _get_ultralytics_yolo() -> Any:
    """
    Lazy-load Ultralytics YOLO only when PyTorch backend is explicitly requested.
    Keeps memory footprint low when running TFLite Edge-AI backend.
    """
    global _YOLO_CLASS
    if _YOLO_CLASS is None:
        try:
            from ultralytics import YOLO as _cls
            _YOLO_CLASS = _cls
        except ImportError:
            _YOLO_CLASS = None
    return _YOLO_CLASS


try:
    from backend.config import (
        DEFAULT_YOLO_CONFIDENCE,
        DEFAULT_YOLO_IOU_THRESHOLD,
        YOLO_PYTORCH_PATH,
        YOLO_TFLITE_PATH,
        COCO_CLASSES_PATH,
    )
except ImportError:
    DEFAULT_YOLO_CONFIDENCE = 0.25
    DEFAULT_YOLO_IOU_THRESHOLD = 0.45
    YOLO_PYTORCH_PATH = "yolov8n.pt"
    YOLO_TFLITE_PATH = "edge_ai/models/yolov8n.tflite"
    COCO_CLASSES_PATH = "edge_ai/config/coco_classes.txt"


# ============================================================
# TYPE DEFINITIONS
# ============================================================

ImageInput = Union[
    str,
    Path,
    np.ndarray,
]


# ============================================================
# YOLO DETECTOR
# ============================================================

class YOLODetector:
    """
    YOLOv8 object detector supporting both PyTorch (.pt) and
    TensorFlow Lite (.tflite) Edge-AI inference backends.

    The class provides a stable interface for the rest of
    SmartVisionAI.
    """

    DEFAULT_MODEL = YOLO_PYTORCH_PATH
    DEFAULT_CONFIDENCE = DEFAULT_YOLO_CONFIDENCE
    DEFAULT_IOU_THRESHOLD = DEFAULT_YOLO_IOU_THRESHOLD

    # --------------------------------------------------------
    # Constructor
    # --------------------------------------------------------

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = DEFAULT_CONFIDENCE,
        device: Optional[str] = None,
        verbose: bool = False,
        backend: Optional[str] = None,
        labels_path: Optional[str] = None,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ) -> None:

        # Resolve backend
        if backend:
            self.backend = str(backend).strip().lower()
        elif model_path and str(model_path).lower().endswith(".tflite"):
            self.backend = "tflite"
        else:
            self.backend = "pytorch"

        # Resolve model path
        if model_path is not None:
            self.model_path = str(model_path)
        else:
            self.model_path = (
                YOLO_TFLITE_PATH
                if self.backend == "tflite"
                else str(self.DEFAULT_MODEL)
            )

        self.confidence = float(
            max(
                0.0,
                min(
                    1.0,
                    confidence,
                ),
            )
        )
        self.iou_threshold = float(
            max(
                0.0,
                min(
                    1.0,
                    iou_threshold,
                ),
            )
        )

        self.device = device
        self.verbose = bool(verbose)
        self.labels_path = labels_path or COCO_CLASSES_PATH

        self.model: Any = None
        self.interpreter: Any = None
        self.class_names: List[str] = []
        self.input_details: Any = None
        self.output_details: Any = None
        self.input_shape: Any = None
        self.input_w: int = 640
        self.input_h: int = 640

        self.is_loaded = False

        self._load_model()

    # ========================================================
    # MODEL LOADING
    # ========================================================

    def _load_model(self) -> None:
        """
        Load either the PyTorch YOLO model or the TFLite model.
        """
        if self.backend == "tflite":
            self._load_tflite_model()
        else:
            self._load_pytorch_model()

    def _load_pytorch_model(self) -> None:
        """
        Load the PyTorch YOLOv8 model via Ultralytics.
        """
        yolo_cls = _get_ultralytics_yolo()
        if yolo_cls is None:
            raise RuntimeError(
                "Ultralytics is not installed. "
                "Install it with: pip install ultralytics"
            )

        try:
            self.model = yolo_cls(self.model_path)
            self.is_loaded = True
            print(f"[YOLO] PyTorch model loaded: {self.model_path}")
        except Exception as error:
            self.model = None
            self.is_loaded = False
            raise RuntimeError(
                f"Failed to load PyTorch YOLO model '{self.model_path}': {error}"
            ) from error

    def _load_tflite_model(self) -> None:
        """
        Load the YOLOv8 TFLite Edge-AI model using the lightweight TFLite runtime.
        """
        InterpreterCls = _get_tflite_interpreter_class()
        if InterpreterCls is None:
            raise RuntimeError(
                "TFLite runtime is not installed. "
                "Install it with: pip install ai-edge-litert"
            )

        # Load COCO class labels
        labels_file = Path(self.labels_path)
        if labels_file.exists():
            with open(labels_file, "r", encoding="utf-8") as f:
                self.class_names = [line.strip() for line in f if line.strip()]
        else:
            # Fallback 80 COCO classes if file missing
            self.class_names = [str(i) for i in range(80)]

        try:
            num_threads = min(4, os.cpu_count() or 4)
            self.interpreter = InterpreterCls(
                model_path=self.model_path,
                num_threads=num_threads,
            )
            self.interpreter.allocate_tensors()

            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_shape = self.input_details[0]["shape"]

            # NCHW: [1, 3, H, W]
            self.input_h = int(self.input_shape[2])
            self.input_w = int(self.input_shape[3])

            # Pre-allocate reusable input buffer to eliminate per-frame allocations
            self._input_buffer = np.zeros((1, 3, self.input_h, self.input_w), dtype=np.float32)

            self.is_loaded = True
            print(
                f"[YOLO] TFLite model loaded: {self.model_path} "
                f"(shape: {self.input_shape}, classes: {len(self.class_names)})"
            )
        except Exception as error:
            self.interpreter = None
            self.is_loaded = False
            raise RuntimeError(
                f"Failed to load TFLite YOLO model '{self.model_path}': {error}"
            ) from error

    # ========================================================
    # IMAGE VALIDATION & PREPARATION
    # ========================================================

    @staticmethod
    def _validate_numpy_image(
        image: np.ndarray,
    ) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a NumPy array.")
        if image.size == 0:
            raise ValueError("Image array is empty.")
        if image.ndim not in (2, 3):
            raise ValueError("Image must have 2 or 3 dimensions.")
        return image

    def _prepare_image(
        self,
        image: ImageInput,
    ) -> Union[Path, np.ndarray]:
        if isinstance(image, Path):
            if not image.exists():
                raise FileNotFoundError(f"Image file not found: {image}")
            if not image.is_file():
                raise ValueError(f"Image path is not a file: {image}")
            return image

        if isinstance(image, str):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            if not image_path.is_file():
                raise ValueError(f"Image path is not a file: {image_path}")
            return image_path

        if isinstance(image, np.ndarray):
            return self._validate_numpy_image(image)

        raise TypeError("Unsupported image type. Use a file path or NumPy array.")

    # ========================================================
    # DETECTION (DISPATCHER)
    # ========================================================

    def detect(
        self,
        image: ImageInput,
        confidence: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in one image. Dispatches to active backend (PyTorch or TFLite).
        """
        if not self.is_loaded:
            raise RuntimeError("YOLO model is not loaded.")

        threshold = (
            self.confidence
            if confidence is None
            else float(confidence)
        )
        threshold = max(0.0, min(1.0, threshold))

        if self.backend == "tflite":
            return self._detect_tflite(image, threshold=threshold)
        else:
            return self._detect_pytorch(image, threshold=threshold)

    def detect_objects(
        self,
        image: ImageInput,
        confidence: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compatibility alias for detect().
        """
        return self.detect(image=image, confidence=confidence)

    # ========================================================
    # PYTORCH INFERENCE
    # ========================================================

    def _detect_pytorch(
        self,
        image: ImageInput,
        threshold: float,
    ) -> List[Dict[str, Any]]:
        prepared_image = self._prepare_image(image)

        try:
            results = self.model.predict(
                source=prepared_image,
                conf=threshold,
                device=self.device,
                verbose=self.verbose,
            )
        except Exception as error:
            raise RuntimeError(f"YOLO PyTorch detection failed: {error}") from error

        return self._parse_results(results)

    # ========================================================
    # TFLITE INFERENCE
    # ========================================================

    def _detect_tflite(
        self,
        image: ImageInput,
        threshold: float,
    ) -> List[Dict[str, Any]]:
        # Ensure image is a NumPy BGR array
        if isinstance(image, (str, Path)):
            frame = cv2.imread(str(image))
            if frame is None:
                raise ValueError(f"Unable to read image from path: {image}")
        elif isinstance(image, np.ndarray):
            frame = self._validate_numpy_image(image)
        else:
            raise TypeError("Unsupported image type for TFLite detector.")

        orig_h, orig_w = frame.shape[:2]

        # ----------------------------------------------------
        # Preprocessing: Aspect-ratio letterbox
        # ----------------------------------------------------
        scale = min(self.input_w / orig_w, self.input_h / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = self.input_w - new_w
        pad_h = self.input_h - new_h
        left = pad_w // 2
        top = pad_h // 2

        # Fast direct slice letterbox padding
        padded = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        padded[top : top + new_h, left : left + new_w] = resized

        # BGR -> RGB, normalize to [0.0, 1.0], HWC -> CHW into pre-allocated contiguous buffer
        padded_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        if not hasattr(self, "_input_buffer") or self._input_buffer is None:
            self._input_buffer = np.zeros((1, 3, self.input_h, self.input_w), dtype=np.float32)
        np.multiply(np.transpose(padded_rgb, (2, 0, 1)), 1.0 / 255.0, out=self._input_buffer[0], dtype=np.float32)
        del padded, padded_rgb, resized

        # ----------------------------------------------------
        # Run TFLite interpreter
        # ----------------------------------------------------
        self.interpreter.set_tensor(self.input_details[0]["index"], self._input_buffer)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])

        # Output shape: [1, 84, 8400] -> transpose to [8400, 84]
        predictions = np.squeeze(output, axis=0).T

        # ----------------------------------------------------
        # Vectorized candidate parsing & filtering
        # ----------------------------------------------------
        class_scores = predictions[:, 4:]
        max_scores = np.max(class_scores, axis=1)
        class_ids_arr = np.argmax(class_scores, axis=1)

        mask = max_scores >= threshold
        if not np.any(mask):
            return []

        filt_preds = predictions[mask, :4]
        filt_scores = max_scores[mask]
        filt_class_ids = class_ids_arr[mask]

        # Coordinates are normalized relative to 640x640: cx, cy, w, h
        cx = filt_preds[:, 0] * self.input_w
        cy = filt_preds[:, 1] * self.input_h
        w = filt_preds[:, 2] * self.input_w
        h = filt_preds[:, 3] * self.input_h

        # Un-pad and scale back to original image coordinates
        x1 = np.clip((cx - w / 2.0 - left) / scale, 0.0, float(orig_w))
        y1 = np.clip((cy - h / 2.0 - top) / scale, 0.0, float(orig_h))
        x2 = np.clip((cx + w / 2.0 - left) / scale, 0.0, float(orig_w))
        y2 = np.clip((cy + h / 2.0 - top) / scale, 0.0, float(orig_h))

        valid_box = (x2 > x1) & (y2 > y1)
        if not np.any(valid_box):
            return []

        x1 = x1[valid_box]
        y1 = y1[valid_box]
        x2 = x2[valid_box]
        y2 = y2[valid_box]
        bw = x2 - x1
        bh = y2 - y1
        filt_scores = filt_scores[valid_box]
        filt_class_ids = filt_class_ids[valid_box]

        boxes: List[List[float]] = [
            [float(bx1), float(by1), float(bbw), float(bbh)]
            for bx1, by1, bbw, bbh in zip(x1, y1, bw, bh)
        ]
        scores: List[float] = [float(s) for s in filt_scores]
        class_ids: List[int] = [int(c) for c in filt_class_ids]

        # ----------------------------------------------------
        # Non-Maximum Suppression (NMS)
        # ----------------------------------------------------
        if hasattr(cv2, "dnn") and hasattr(cv2.dnn, "NMSBoxes"):
            indices = cv2.dnn.NMSBoxes(boxes, scores, threshold, self.iou_threshold)
            surviving_indices = [
                int(idx[0] if isinstance(idx, (list, tuple, np.ndarray)) else idx)
                for idx in indices
            ]
        else:
            surviving_indices = self._custom_nms(boxes, scores, self.iou_threshold)

        objects: List[Dict[str, Any]] = []
        for idx in surviving_indices:
            b = boxes[idx]
            x1 = round(float(b[0]), 2)
            y1 = round(float(b[1]), 2)
            x2 = round(float(b[0] + b[2]), 2)
            y2 = round(float(b[1] + b[3]), 2)
            cid = class_ids[idx]
            cname = (
                self.class_names[cid]
                if cid < len(self.class_names)
                else str(cid)
            )

            objects.append({
                "class_name": cname,
                "class_id": cid,
                "confidence": round(float(scores[idx]), 4),
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
            })

        # Sort by confidence descending
        objects.sort(key=lambda x: x["confidence"], reverse=True)
        return objects

    @staticmethod
    def _custom_nms(
        boxes: List[List[float]],
        scores: List[float],
        iou_threshold: float,
    ) -> List[int]:
        """Custom pure-Python IoU NMS fallback if cv2.dnn is unavailable."""
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        keep = []
        while order:
            i = order.pop(0)
            keep.append(i)
            b1 = boxes[i]
            x1_1, y1_1, w1, h1 = b1
            x2_1, y2_1 = x1_1 + w1, y1_1 + h1
            area1 = w1 * h1

            remaining = []
            for j in order:
                b2 = boxes[j]
                x1_2, y1_2, w2, h2 = b2
                x2_2, y2_2 = x1_2 + w2, y1_2 + h2
                area2 = w2 * h2

                inter_w = max(0.0, min(x2_1, x2_2) - max(x1_1, x1_2))
                inter_h = max(0.0, min(y2_1, y2_2) - max(y1_1, y1_2))
                intersection = inter_w * inter_h
                union = area1 + area2 - intersection
                iou = intersection / union if union > 0 else 0.0

                if iou < iou_threshold:
                    remaining.append(j)
            order = remaining
        return keep

    # ========================================================
    # RESULT PARSING (PYTORCH)
    # ========================================================

    def _parse_results(
        self,
        results: Any,
    ) -> List[Dict[str, Any]]:
        objects: List[Dict[str, Any]] = []
        if results is None:
            return objects

        try:
            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue

                names = getattr(result, "names", {})
                xyxy = getattr(boxes, "xyxy", None)
                confs = getattr(boxes, "conf", None)
                class_ids = getattr(boxes, "cls", None)

                if xyxy is None:
                    continue

                for index in range(len(xyxy)):
                    box = xyxy[index]
                    try:
                        coordinates = box.detach().cpu().tolist()
                    except Exception:
                        coordinates = np.asarray(box).tolist()

                    if len(coordinates) < 4:
                        continue

                    x1, y1, x2, y2 = [float(c) for c in coordinates[:4]]

                    confidence = 0.0
                    if confs is not None:
                        try:
                            confidence = float(confs[index].detach().cpu().item())
                        except Exception:
                            try:
                                confidence = float(confs[index])
                            except Exception:
                                confidence = 0.0

                    class_id = 0
                    if class_ids is not None:
                        try:
                            class_id = int(class_ids[index].detach().cpu().item())
                        except Exception:
                            try:
                                class_id = int(class_ids[index])
                            except Exception:
                                class_id = 0

                    class_name = self._get_class_name(names, class_id)

                    if confidence < self.confidence:
                        continue

                    objects.append({
                        "class_name": class_name,
                        "class_id": class_id,
                        "confidence": round(confidence, 4),
                        "bbox": {
                            "x1": round(x1, 2),
                            "y1": round(y1, 2),
                            "x2": round(x2, 2),
                            "y2": round(y2, 2),
                        },
                    })
        except Exception as error:
            raise RuntimeError(f"Failed to parse YOLO results: {error}") from error

        return objects

    @staticmethod
    def _get_class_name(
        names: Any,
        class_id: int,
    ) -> str:
        try:
            if isinstance(names, dict):
                return str(names.get(class_id, class_id))
            if isinstance(names, list) and 0 <= class_id < len(names):
                return str(names[class_id])
        except Exception:
            pass
        return str(class_id)

    # ========================================================
    # STATUS
    # ========================================================

    def status(self) -> Dict[str, Any]:
        """
        Return detector status including active backend.
        """
        return {
            "loaded": bool(self.is_loaded),
            "model": self.model_path,
            "confidence": round(self.confidence, 3),
            "device": self.device,
            "backend": self.backend,
        }

    # ========================================================
    # WARMUP
    # ========================================================

    def warmup(
        self,
        width: int = 640,
        height: int = 640,
    ) -> bool:
        """
        Warm up YOLO with a blank image.
        """
        if not self.is_loaded:
            raise RuntimeError("YOLO model is not loaded.")

        try:
            width = max(32, int(width))
            height = max(32, int(height))
            blank = np.zeros((height, width, 3), dtype=np.uint8)

            if self.backend == "tflite":
                self.detect(blank, confidence=self.confidence)
            else:
                self.model.predict(
                    source=blank,
                    conf=self.confidence,
                    device=self.device,
                    verbose=False,
                )
            return True
        except Exception as error:
            raise RuntimeError(f"YOLO warmup failed: {error}") from error


# ============================================================
# EXPORT
# ============================================================

__all__ = [
    "YOLODetector",
]