"""
modules/pose_extractor.py
Extract 17 COCO keypoints per track dùng YOLOv8-pose.
Debug version.
"""

from ultralytics import YOLO
import numpy as np


COCO_17 = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


class PoseExtractor:

    def __init__(self, model_name="yolov8n-pose.pt", conf=0.5):
        print(f"[POSE] Loading model: {model_name}")

        self.model = YOLO(model_name)
        self.conf = conf

        print(f"[POSE] Confidence threshold: {self.conf}")

    def extract(self, frame, track_tlbr) -> np.ndarray | None:

        # =====================================================
        # 1. Validate frame
        # =====================================================

        if frame is None:
            print("[POSE DEBUG] frame=None")
            return None

        if frame.size == 0:
            print("[POSE DEBUG] frame empty")
            return None

        h, w = frame.shape[:2]

        # =====================================================
        # 2. Track bbox
        # =====================================================

        x1, y1, x2, y2 = [int(v) for v in track_tlbr]

        # Clamp bbox vào frame
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            print(
                f"[POSE DEBUG] Invalid bbox: "
                f"({x1}, {y1}, {x2}, {y2})"
            )
            return None

        # =====================================================
        # 3. Padding
        # =====================================================

        pad = int(max(x2 - x1, y2 - y1) * 0.15)

        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w, x2 + pad)
        cy2 = min(h, y2 + pad)

        crop = frame[cy1:cy2, cx1:cx2]

        if crop.size == 0:
            print(
                f"[POSE DEBUG] Empty crop: "
                f"({cx1}, {cy1}, {cx2}, {cy2})"
            )
            return None

        # =====================================================
        # 4. YOLOv8-Pose inference
        # =====================================================

        try:
            results = self.model(
                crop,
                verbose=False,
                conf=self.conf
            )
        except Exception as e:
            print(
                f"[POSE ERROR] inference failed: "
                f"{type(e).__name__}: {e}"
            )
            return None

        if not results:
            print("[POSE DEBUG] No YOLO results")
            return None

        result = results[0]

        # =====================================================
        # 5. Check keypoints
        # =====================================================

        if result.keypoints is None:
            print("[POSE DEBUG] keypoints=None")
            return None

        kpts = result.keypoints.data

        if kpts is None:
            print("[POSE DEBUG] keypoints.data=None")
            return None

        if len(kpts) == 0:
            print("[POSE DEBUG] YOLO found 0 persons")
            return None

        # =====================================================
        # 6. Debug shape
        # =====================================================

        print(
            f"[POSE DEBUG] "
            f"bbox=({x1},{y1},{x2},{y2}) "
            f"crop={crop.shape[:2]} "
            f"persons={len(kpts)} "
            f"kpts_shape={tuple(kpts.shape)}"
        )

        # =====================================================
        # 7. First detected person
        # =====================================================

        kp = kpts[0].cpu().numpy()

        if kp.shape != (17, 3):
            print(
                f"[POSE DEBUG] Unexpected keypoint shape: "
                f"{kp.shape}"
            )
            return None

        # =====================================================
        # 8. Convert crop coordinates -> full-frame normalized
        # =====================================================

        cw = cx2 - cx1
        ch = cy2 - cy1

        if cw <= 0 or ch <= 0:
            return None

        kp[:, 0] = (cx1 + kp[:, 0]) / w
        kp[:, 1] = (cy1 + kp[:, 1]) / h

        # =====================================================
        # 9. Debug wrist
        # =====================================================

        left_wrist = kp[9]
        right_wrist = kp[10]

        print(
            f"[POSE WRIST] "
            f"L={left_wrist} "
            f"R={right_wrist}"
        )

        return kp
