"""
INFERENCE PIPELINE
Live Camera Stream → Frame Reader → Preprocessing → YOLO-World Detection
→ NMS → ByteTrack → Behavior Feature Extraction → Isolation Forest/Autoencoder
→ Anomaly Scoring → Anomaly Decision → Alert System / Database Storage
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import torch

from modules.detector import YOLOWorldDetector
from modules.tracker import ByteTrackWrapper
from modules.features import BehaviorFeatureExtractor
from modules.anomaly import AnomalyScorer
from modules.alert import AlertSystem
from modules.database import DatabaseStorage
from modules.visualizer import Visualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    # Camera
    source: str = "0"                  # RTSP URL or webcam index
    width: int = 1280
    height: int = 720
    fps_limit: int = 15

    # Detection
    yolo_model: str = "yolo_world_v2_l"
    detection_classes: list = field(default_factory=lambda: [
        "person", "bag", "backpack", "handbag", "suitcase"
    ])
    conf_threshold: float = 0.35
    nms_iou: float = 0.45

    # Tracking
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    min_box_area: float = 100.0

    # Anomaly
    model_path: str = "models/anomaly_model.pkl"
    anomaly_threshold: float = 0.65
    window_size: int = 30             # frames for feature window

    # Output
    display: bool = True
    save_video: str = ""              # path to save output, "" = don't save
    db_path: str = "data/events.db"
    alert_cooldown: int = 10          # seconds between alerts per track


class RetailAnomalyPipeline:
    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self._init_components()
        self.frame_count = 0
        self.fps = 0.0
        self._t = time.time()

    # ------------------------------------------------------------------
    def _init_components(self):
        cfg = self.cfg
        logger.info("Initialising components …")

        self.detector = YOLOWorldDetector(
            model_name=cfg.yolo_model,
            classes=cfg.detection_classes,
            conf=cfg.conf_threshold,
            iou=cfg.nms_iou,
        )
        self.tracker = ByteTrackWrapper(
            track_thresh=cfg.track_thresh,
            track_buffer=cfg.track_buffer,
            match_thresh=cfg.match_thresh,
            min_box_area=cfg.min_box_area,
        )
        self.feat_extractor = BehaviorFeatureExtractor(window=cfg.window_size)
        self.anomaly_scorer = AnomalyScorer(model_path=cfg.model_path)
        self.alert_system = AlertSystem(cooldown=cfg.alert_cooldown)
        self.db = DatabaseStorage(db_path=cfg.db_path)
        self.viz = Visualizer()
        logger.info("All components ready.")

    # ------------------------------------------------------------------
    def _open_capture(self):
        src = self.cfg.source
        try:
            src = int(src)
        except ValueError:
            pass
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {src}")
        return cap

    # ------------------------------------------------------------------
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize, normalise brightness."""
        frame = cv2.resize(frame, (self.cfg.width, self.cfg.height))
        # CLAHE for low-light scenes common in retail
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return frame

    # ------------------------------------------------------------------
    def _compute_fps(self):
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            now = time.time()
            self.fps = 30 / (now - self._t)
            self._t = now

    # ------------------------------------------------------------------
    def run(self):
        cap = self._open_capture()
        writer = None

        if self.cfg.save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                self.cfg.save_video, fourcc, self.cfg.fps_limit,
                (self.cfg.width, self.cfg.height)
            )

        logger.info("Pipeline running. Press 'q' to quit.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Frame read failed – reconnecting …")
                    time.sleep(1)
                    cap = self._open_capture()
                    continue

                result_frame = self.process_frame(frame)
                self._compute_fps()

                if writer:
                    writer.write(result_frame)

                if self.cfg.display:
                    cv2.imshow("Retail Anomaly Detection", result_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            self.db.close()
            logger.info("Pipeline stopped.")

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Single-frame inference – callable from external code / tests."""
        ts = time.time()

        # 1. Preprocess
        frame = self._preprocess(frame)

        # 2. Detect
        detections = self.detector.detect(frame)           # List[Detection]

        # 3. NMS already done inside detector; track
        tracks = self.tracker.update(detections, frame)    # List[Track]

        # 4. Feature extraction per track
        features_map = self.feat_extractor.update(tracks, frame)

        # 5. Anomaly scoring
        anomaly_map = {}
        for tid, feat_vec in features_map.items():
            score, label = self.anomaly_scorer.score(feat_vec)
            anomaly_map[tid] = {"score": score, "is_anomaly": label}

        # 6. Decision + side effects
        for tid, result in anomaly_map.items():
            if result["is_anomaly"]:
                track = next((t for t in tracks if t.track_id == tid), None)
                if track:
                    # Alert
                    self.alert_system.trigger(tid, result["score"], frame, ts)
                    # Store
                    self.db.log_event(tid, result["score"], ts, track.tlbr.tolist())

        # 7. Visualise
        out = self.viz.draw(frame, tracks, anomaly_map, fps=self.fps)
        return out


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retail Anomaly Detection – Inference")
    parser.add_argument("--source", default="0", help="Camera index, RTSP URL, or video file")
    parser.add_argument("--model", default="models/anomaly_model.pkl")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--save", default="", help="Path to save output video")
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()

    cfg = PipelineConfig(
        source=args.source,
        model_path=args.model,
        display=not args.no_display,
        save_video=args.save,
        anomaly_threshold=args.threshold,
    )
    RetailAnomalyPipeline(cfg).run()
