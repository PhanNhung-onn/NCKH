"""
modules/detector.py
YOLO-World open-vocabulary detector with NMS post-processing.

Uses Ultralytics YOLO-World / YOLOv8-Worldv2.
The pretrained model is automatically downloaded by Ultralytics
when a supported model name is provided.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single bounding-box detection."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def tlwh(self):
        """Top-left x/y + width/height."""
        return np.array(
            [
                self.x1,
                self.y1,
                self.x2 - self.x1,
                self.y2 - self.y1,
            ],
            dtype=np.float32,
        )

    @property
    def tlbr(self):
        """Top-left / bottom-right bounding box."""
        return np.array(
            [self.x1, self.y1, self.x2, self.y2],
            dtype=np.float32,
        )

    @property
    def area(self):
        """Bounding-box area."""
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def center(self):
        """Bounding-box center."""
        return (
            (self.x1 + self.x2) / 2.0,
            (self.y1 + self.y2) / 2.0,
        )


class YOLOWorldDetector:
    """
    YOLO-World open-vocabulary detector using Ultralytics.

    Supported pretrained models include:

        yolov8s-world.pt
        yolov8m-world.pt
        yolov8l-world.pt
        yolov8x-world.pt

        yolov8s-worldv2.pt
        yolov8m-worldv2.pt
        yolov8l-worldv2.pt
        yolov8x-worldv2.pt

    Default:
        yolov8l-worldv2.pt

    Ultralytics automatically downloads the pretrained weight
    when the model is initialized for the first time.
    """

    # Mapping kept for backward compatibility with the old project
    # names such as "yolo_world_v2_l".
    MODEL_ALIASES = {
        "yolo_world_v2_s": "yolov8s-worldv2.pt",
        "yolo_world_v2_m": "yolov8m-worldv2.pt",
        "yolo_world_v2_l": "yolov8l-worldv2.pt",
        "yolo_world_v2_x": "yolov8x-worldv2.pt",

        "yolo_world_s": "yolov8s-world.pt",
        "yolo_world_m": "yolov8m-world.pt",
        "yolo_world_l": "yolov8l-world.pt",
        "yolo_world_x": "yolov8x-world.pt",
    }

    def __init__(
        self,
        model_name: str = "yolo_world_v2_l",
        classes: Optional[List[str]] = None,
        conf: float = 0.35,
        iou: float = 0.45,
        device: str = "auto",
        img_size: int = 640,
    ):
        self.conf = conf
        self.iou = iou
        self.img_size = img_size
        self.classes = classes or ["person"]

        # --------------------------------------------------------------
        # Select device
        # --------------------------------------------------------------
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        logger.info(f"Using device: {self.device}")

        # --------------------------------------------------------------
        # Load YOLO-World
        # --------------------------------------------------------------
        self._load_model(model_name)

    # ------------------------------------------------------------------
    def _resolve_model_name(self, model_name: str) -> str:
        """
        Convert old project model names to valid Ultralytics model names.

        Examples:
            yolo_world_v2_l
                -> yolov8l-worldv2.pt

            yolov8l-worldv2.pt
                -> unchanged

            /path/to/custom_model.pt
                -> unchanged
        """

        model_name = str(model_name).strip()

        # Backward-compatible aliases
        if model_name in self.MODEL_ALIASES:
            return self.MODEL_ALIASES[model_name]

        # Already a .pt / .yaml / .yml model path/name
        if model_name.endswith((".pt", ".yaml", ".yml")):
            return model_name

        # If somebody passes the model without extension
        # and it follows an Ultralytics name, add .pt.
        if model_name.startswith("yolov8"):
            return f"{model_name}.pt"

        # Final fallback:
        # preserve the old project convention by mapping the
        # model size if possible.
        raise ValueError(
            f"Unsupported YOLO-World model name: '{model_name}'. "
            f"Use one of: {list(self.MODEL_ALIASES.keys())} "
            f"or a valid Ultralytics model filename such as "
            f"'yolov8l-worldv2.pt'."
        )

    # ------------------------------------------------------------------
    def _load_model(self, model_name: str):
        """
        Load the YOLO-World model.

        Ultralytics will automatically download official pretrained
        weights when the model filename is used and the file is not
        already available locally.
        """

        try:
            from ultralytics import YOLOWorld

            model_path = self._resolve_model_name(model_name)

            logger.info(f"Loading YOLO-World model: {model_path}")

            # ----------------------------------------------------------
            # If this is a local path, check that it exists.
            # Official model names such as yolov8l-worldv2.pt are
            # allowed to be downloaded automatically by Ultralytics.
            # ----------------------------------------------------------
            path = Path(model_path)

            is_local_path = (
                path.is_absolute()
                or "/" in model_path
                or "\\" in model_path
            )

            if is_local_path and not path.exists():
                raise FileNotFoundError(
                    f"YOLO-World weight file not found: {model_path}"
                )

            # ----------------------------------------------------------
            # Load model
            # ----------------------------------------------------------
            self.model = YOLOWorld(model_path)

            # ----------------------------------------------------------
            # Set open-vocabulary classes
            # ----------------------------------------------------------
            self.model.set_classes(self.classes)

            # ----------------------------------------------------------
            # Move model to selected device
            # ----------------------------------------------------------
            self.model.to(self.device)

            logger.info(
                f"Loaded YOLO-World: {model_path} "
                f"on {self.device} "
                f"classes={self.classes}"
            )

        except Exception as e:
            logger.exception(
                f"Failed to load YOLO-World: {e}"
            )
            raise

    # ------------------------------------------------------------------
    def set_classes(self, classes: List[str]):
        """
        Dynamically update open-vocabulary classes at runtime.
        """

        if not classes:
            raise ValueError("classes must contain at least one class.")

        self.classes = classes
        self.model.set_classes(classes)

        logger.info(
            f"YOLO-World classes updated: {self.classes}"
        )

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a BGR OpenCV frame.

        Returns:
            List[Detection]

        NMS is already handled by Ultralytics.
        """

        if frame is None:
            return []

        if not isinstance(frame, np.ndarray):
            raise TypeError(
                f"frame must be numpy.ndarray, got {type(frame)}"
            )

        if frame.size == 0:
            return []

        # --------------------------------------------------------------
        # Inference
        # --------------------------------------------------------------
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.img_size,
            verbose=False,
            device=self.device,
        )

        detections: List[Detection] = []

        if not results:
            return detections

        # --------------------------------------------------------------
        # Convert Ultralytics result -> Detection
        # --------------------------------------------------------------
        for result in results:
            boxes = result.boxes

            if boxes is None:
                continue

            for i in range(len(boxes)):
                # Bounding box
                xyxy = boxes.xyxy[i].detach().cpu().numpy()

                x1, y1, x2, y2 = map(float, xyxy)

                # Confidence
                conf = float(
                    boxes.conf[i].detach().cpu().item()
                )

                # Class ID
                cls_id = int(
                    boxes.cls[i].detach().cpu().item()
                )

                # Class name
                if 0 <= cls_id < len(self.classes):
                    cls_name = self.classes[cls_id]
                else:
                    cls_name = str(cls_id)

                det = Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                )

                if det.area > 0:
                    detections.append(det)

        return detections
