"""
============================================================
SmartVisionAI - MiDaS Depth Estimation
============================================================
Provides monocular relative-depth estimation using MiDaS.
Supports dual execution backends:
  - "pytorch": Official Torch Hub MiDaS (MiDaS_small)
  - "onnx": High-performance ONNX Runtime MiDaS (midas_small.onnx)

Phase 14 Real-Time Optimization:
- ONNX Runtime CPUExecutionProvider integration
- PyTorch torch.inference_mode() execution with autograd bypass
- Fast bilinear interpolation for depth map scaling
- In-place tensor/array normalization to minimize memory allocations
- Preserves full relative-depth contract (0.0 = closest, 1.0 = farthest)
============================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from backend.config import (
        DEFAULT_DEPTH_INPUT_SIZE,
        MIDAS_ONNX_PATH,
        MIDAS_PYTORCH_MODEL,
    )
except ImportError:
    DEFAULT_DEPTH_INPUT_SIZE = 256
    MIDAS_PYTORCH_MODEL = "MiDaS_small"
    MIDAS_ONNX_PATH = "edge_ai/models/midas_small.onnx"


class MiDaSDepthEstimator:
    """
    MiDaS-based monocular depth estimator supporting both PyTorch
    and ONNX Runtime execution tiers.

    The output represents normalized relative depth [0.0, 1.0].
    """

    # MiDaS standard ImageNet normalization constants
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        input_size: int = DEFAULT_DEPTH_INPUT_SIZE,
        backend: Optional[str] = None,
        onnx_model_path: Optional[str] = None,
    ) -> None:

        # Resolve backend
        if backend:
            self.backend = str(backend).strip().lower()
        elif onnx_model_path or (model_name and str(model_name).lower().endswith(".onnx")):
            self.backend = "onnx"
        else:
            self.backend = "pytorch"

        self.input_size = input_size
        self.onnx_model_path = str(onnx_model_path or MIDAS_ONNX_PATH)
        self.model_name = str(model_name or MIDAS_PYTORCH_MODEL)

        if self.backend == "pytorch" and torch is not None:
            if device:
                self.device = torch.device(device)
            else:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
        else:
            self.device = "cpu"

        # PyTorch model state
        self.model = None
        self.transform = None

        # ONNX model state
        self.session = None
        self.input_name = None
        self.output_name = None

        self.is_loaded = False

        self._load_model()

    # ========================================================
    # MODEL LOADING
    # ========================================================

    def _load_model(self) -> None:
        """
        Load either the PyTorch MiDaS model or the ONNX Runtime model.
        """
        if self.backend == "onnx":
            self._load_onnx_model()
        else:
            self._load_pytorch_model()

    def _load_onnx_model(self) -> None:
        """
        Load the ONNX Runtime MiDaS model.
        """
        if ort is None:
            raise RuntimeError(
                "onnxruntime is not installed. "
                "Install it with: pip install onnxruntime"
            )

        onnx_file = Path(self.onnx_model_path)
        if not onnx_file.exists():
            raise FileNotFoundError(
                f"ONNX MiDaS model file not found: {self.onnx_model_path}"
            )

        try:
            # Configure lightweight CPU session
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                str(onnx_file),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )

            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.is_loaded = True

            print(
                f"[MiDaS] ONNX model loaded: {self.onnx_model_path} "
                f"(input: {self.input_name}, output: {self.output_name}) on CPU"
            )
        except Exception as exc:
            self.session = None
            self.is_loaded = False
            raise RuntimeError(
                f"Failed to load ONNX MiDaS model '{self.onnx_model_path}': {exc}"
            ) from exc

    def _load_pytorch_model(self) -> None:
        """
        Load lightweight MiDaS model from official Torch Hub repository.
        """
        if torch is None:
            raise RuntimeError(
                "PyTorch is not installed. "
                "Install it with: pip install torch"
            )

        print(f"[MiDaS] Loading PyTorch model: {self.model_name}")

        try:
            repo = "intel-isl/MiDaS"

            if self.model_name == "MiDaS_small":
                model_type = "MiDaS_small"
            else:
                model_type = "DPT_Hybrid"

            self.model = torch.hub.load(
                repo,
                model_type,
                trust_repo=True,
            )

            self.model.to(self.device)
            self.model.eval()

            transforms = torch.hub.load(
                repo,
                "transforms",
                trust_repo=True,
            )

            if model_type == "MiDaS_small":
                self.transform = transforms.small_transform
            else:
                self.transform = transforms.dpt_transform

            self.is_loaded = True
            print(f"[MiDaS] PyTorch model loaded successfully on {self.device}.")

        except Exception as exc:
            self.model = None
            self.transform = None
            self.is_loaded = False
            raise RuntimeError(
                f"Failed to load PyTorch MiDaS model: {exc}"
            ) from exc

    # ========================================================
    # STATUS
    # ========================================================

    def status(self) -> Dict[str, Any]:
        """
        Return MiDaS component status.
        """
        return {
            "loaded": self.is_loaded,
            "is_loaded": self.is_loaded,
            "model": (
                self.onnx_model_path
                if self.backend == "onnx"
                else self.model_name
            ),
            "device": str(self.device),
            "backend": self.backend,
        }

    # ========================================================
    # DEPTH MAP (DISPATCHER)
    # ========================================================

    def estimate_depth_map(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Generate a normalized relative depth map [0.0, 1.0].
        0.0 = very close, 1.0 = relatively far.
        """
        if not self.is_loaded:
            raise RuntimeError("MiDaS model is not loaded.")

        if image is None:
            raise ValueError("Image cannot be None.")
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a NumPy array.")
        if image.size == 0:
            raise ValueError("Image is empty.")

        if self.backend == "onnx":
            return self._estimate_depth_map_onnx(image)
        else:
            return self._estimate_depth_map_pytorch(image)

    # ========================================================
    # ONNX DEPTH ESTIMATION
    # ========================================================

    def _estimate_depth_map_onnx(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        target_height, target_width = image.shape[:2]

        # Fast compact preprocessing: resize BGR to 256x256 first, then convert BGR -> RGB
        resized_bgr = cv2.resize(
            image,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) * (1.0 / 255.0)

        # Vectorized ImageNet standardization and CHW batch transpose
        normalized = (rgb - self.IMAGENET_MEAN) / self.IMAGENET_STD
        input_tensor = np.ascontiguousarray(
            np.transpose(normalized, (2, 0, 1))[np.newaxis, ...],
            dtype=np.float32,
        )

        # Run ONNX inference
        output = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor},
        )[0]
        del rgb, resized_bgr, normalized, input_tensor

        depth = np.squeeze(output)

        # Sanitize non-finite values on compact 256x256 tensor
        depth = np.nan_to_num(
            depth,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
            copy=False,
        )

        d_min = float(depth.min())
        d_max = float(depth.max())
        diff = d_max - d_min

        # Invert disparity: MiDaS predicts inverse depth (disparity, where larger = closer).
        # Contract: 0.0 = closest, 1.0 = farthest.
        # Normalizing in 256x256 first prevents massive temporary allocations at frame resolution.
        if diff > 1e-8:
            depth = (d_max - depth) / diff
        else:
            depth = np.zeros_like(depth, dtype=np.float32)

        # Fast bilinear interpolation up to frame dimensions
        depth_resized = cv2.resize(
            depth.astype(np.float32, copy=False),
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )

        return depth_resized.astype(np.float32, copy=False)

    # ========================================================
    # PYTORCH DEPTH ESTIMATION
    # ========================================================

    def _estimate_depth_map_pytorch(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        target_height, target_width = image.shape[:2]

        # Convert BGR -> RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # MiDaS preprocessing transform
        input_batch = self.transform(rgb).to(self.device)

        # High-performance inference mode (disables autograd metadata overhead)
        with torch.inference_mode():
            prediction = self.model(input_batch)

            # Fast bilinear interpolation on CPU
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1).squeeze(0)

        depth = prediction.cpu().numpy()

        # Sanitize non-finite values
        depth = np.nan_to_num(
            depth,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
            copy=False,
        )

        minimum = float(depth.min())
        maximum = float(depth.max())
        diff = maximum - minimum

        # Invert disparity: 0.0 = closest, 1.0 = farthest
        if diff > 1e-8:
            depth = (maximum - depth) / diff
        else:
            depth = np.zeros_like(depth, dtype=np.float32)

        return depth.astype(np.float32, copy=False)

    # ========================================================
    # REGION DEPTH
    # ========================================================

    @staticmethod
    def _safe_region(
        depth_map: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> np.ndarray:
        height, width = depth_map.shape[:2]
        x1 = max(0, min(width - 1, int(x1)))
        y1 = max(0, min(height - 1, int(y1)))
        x2 = max(0, min(width, int(x2)))
        y2 = max(0, min(height, int(y2)))

        if x2 <= x1 or y2 <= y1:
            return np.array([], dtype=np.float32)

        return depth_map[y1:y2, x1:x2]

    def estimate_object_depth(
        self,
        depth_map: np.ndarray,
        bbox: Dict[str, float],
    ) -> float:
        """
        Estimate relative depth for one detected object.
        Uses the central portion of the bounding box to reduce
        the influence of background pixels.
        """
        if depth_map is None:
            raise ValueError("Depth map cannot be None.")

        height, width = depth_map.shape[:2]
        x1 = float(bbox.get("x1", 0))
        y1 = float(bbox.get("y1", 0))
        x2 = float(bbox.get("x2", width))
        y2 = float(bbox.get("y2", height))

        # Use central 60% of bounding box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = abs(x2 - x1)
        bh = abs(y2 - y1)

        rx1 = cx - bw * 0.30
        ry1 = cy - bh * 0.30
        rx2 = cx + bw * 0.30
        ry2 = cy + bh * 0.30

        region = self._safe_region(
            depth_map,
            int(rx1),
            int(ry1),
            int(rx2),
            int(ry2),
        )

        if region.size == 0:
            return 1.0

        # Median is more robust than a single pixel
        value = float(np.median(region))
        return round(max(0.0, min(1.0, value)), 4)

    # ========================================================
    # SCENE DEPTH
    # ========================================================

    def analyze_scene(
        self,
        depth_map: np.ndarray,
    ) -> Dict[str, float]:
        """
        Extract left, center and right relative-depth values.
        """
        if depth_map is None:
            raise ValueError("Depth map cannot be None.")
        if depth_map.size == 0:
            raise ValueError("Depth map is empty.")

        height, width = depth_map.shape[:2]

        left = depth_map[:, : int(width * 0.33)]
        center = depth_map[:, int(width * 0.33): int(width * 0.67)]
        right = depth_map[:, int(width * 0.67):]

        center_region = depth_map[
            int(height * 0.25): int(height * 0.85),
            int(width * 0.40): int(width * 0.60),
        ]

        return {
            "average_depth": round(float(np.mean(depth_map)), 4),
            "center_depth": round(float(np.mean(center)), 4),
            "left_depth": round(float(np.mean(left)), 4),
            "center_region_depth": round(float(np.mean(center_region)), 4),
            "right_depth": round(float(np.mean(right)), 4),
        }

    # ========================================================
    # COMPLETE ANALYSIS
    # ========================================================

    def analyze(
        self,
        image: np.ndarray,
        objects: Optional[list] = None,
    ) -> Dict[str, Any]:
        depth_map = self.estimate_depth_map(image)
        scene = self.analyze_scene(depth_map)
        analyzed_objects = []

        if objects:
            for obj in objects:
                item = dict(obj)
                bbox = item.get("bbox", {})
                item["depth"] = self.estimate_object_depth(depth_map, bbox)
                analyzed_objects.append(item)

        return {
            "depth_map": depth_map,
            "scene": scene,
            "objects": analyzed_objects,
        }

    # ========================================================
    # WARMUP
    # ========================================================

    def warmup(
        self,
        width: int = 640,
        height: int = 480,
    ) -> bool:
        """
        Warm up MiDaS with a blank frame.
        """
        if not self.is_loaded:
            raise RuntimeError("MiDaS model is not loaded.")
        try:
            blank = np.zeros((height, width, 3), dtype=np.uint8)
            self.estimate_depth_map(blank)
            return True
        except Exception as exc:
            raise RuntimeError(f"MiDaS warmup failed: {exc}") from exc


# ============================================================
# ALIAS & EXPORT
# ============================================================

MiDaS = MiDaSDepthEstimator

__all__ = [
    "MiDaSDepthEstimator",
    "MiDaS",
]