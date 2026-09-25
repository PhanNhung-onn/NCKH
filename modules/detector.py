"""
modules/detector.py
YOLO-World open-vocabulary detector with NMS post-processing.
"""

from __future__ import annotations
import numpy as np
import cv2
import logging
from dataclasses import dataclass
from typing import List

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
        return np.array([self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1])

    @property
    def tlbr(self):
        return np.array([self.x1, self.y1, self.x2, self.y2])

    @property
    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def center(self):
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class YOLOWorldDetector:
    """
    Wraps YOLO-World (via ultralytics) for open-vocabulary detection.

    Install:  pip install ultralytics>=8.2
    Models:   yolo_world_v2_s / m / l / x
    """

    def __init__(
        self,
        model_name: str = "yolo_world_v2_l",
        classes: List[str] = None,
        conf: float = 0.35,
        iou: float = 0.45,
        device: str = "auto",
        img_size: int = 640,
    ):
        self.conf = conf
        self.iou = iou
        self.img_size = img_size
        self.classes = classes or ["person"]

        import torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self._load_model(model_name)

    # ------------------------------------------------------------------
    def _load_model(self, model_name: str):
        try:
            from ultralytics import YOLOWorld
            self.model = YOLOWorld(f"{model_name}.pt")
            self.model.set_classes(self.classes)
            self.model.to(self.device)
            logger.info(f"Loaded {model_name} on {self.device}  classes={self.classes}")
        except Exception as e:
            logger.error(f"Failed to load YOLO-World: {e}")
            raise

    # ------------------------------------------------------------------
    def set_classes(self, classes: List[str]):
        """Dynamically update open-vocab classes at runtime."""
        self.classes = classes
        self.model.set_classes(classes)

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a BGR frame.
        Returns list of Detection objects (NMS already applied by ultralytics).
        """
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.img_size,
            verbose=False,
            device=self.device,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = self.classes[cls_id] if cls_id < len(self.classes) else str(cls_id)

                det = Detection(
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                )
                if det.area > 0:
                    detections.append(det)

        return detections
