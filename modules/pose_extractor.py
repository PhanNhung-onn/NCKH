"""
modules/pose_extractor.py
Extract 17 COCO keypoints per track dùng YOLOv8-pose.
Thay thế MediaPipe — cùng chuẩn keypoint với PoseLift dataset.
"""
from ultralytics import YOLO   # đã cài sẵn trong project
import numpy as np

COCO_17 = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

class PoseExtractor:
    def __init__(self, model_name="yolov8n-pose.pt", conf=0.5):
        self.model = YOLO(model_name)   # tự tải ~6MB
        self.conf  = conf

    def extract(self, frame, track_tlbr) -> np.ndarray | None:
        """
        Crop vùng track, chạy pose estimation.
        Trả về array (17, 3) = [x_norm, y_norm, confidence] mỗi keypoint.
        Trả về None nếu không detect được.
        """
        x1, y1, x2, y2 = [int(v) for v in track_tlbr]
        h, w = frame.shape[:2]
        pad  = int(max(x2-x1, y2-y1) * 0.15)
        cx1  = max(0, x1-pad); cy1 = max(0, y1-pad)
        cx2  = min(w, x2+pad); cy2 = min(h, y2+pad)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        results = self.model(crop, verbose=False, conf=self.conf)
        if not results or results[0].keypoints is None:
            return None

        kpts = results[0].keypoints.data   # (N_person, 17, 3)
        if len(kpts) == 0:
            return None

        # Lấy person đầu tiên (crop đã isolate 1 người)
        kp = kpts[0].cpu().numpy()   # (17, 3): x, y, conf
        cw, ch = cx2-cx1, cy2-cy1

        # Normalize về toàn frame
        kp[:, 0] = (cx1 + kp[:, 0]) / w
        kp[:, 1] = (cy1 + kp[:, 1]) / h
        return kp   # (17, 3)