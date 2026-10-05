"""
INFERENCE PIPELINE
==================
Live Camera Stream → Frame Reader → Preprocessing → YOLO-World Detection
→ NMS → ByteTrack + SPARTA → Pose Extraction (YOLOv8-pose)
→ Behavior Feature Extraction → CombinedScorer (Kinematic + Pose/Shopformer)
→ Anomaly Decision → Alert System / Database Storage

Thay đổi so với v1:
  - Thêm PoseExtractor (YOLOv8-pose → 17 COCO keypoints mỗi track)
  - Thêm ShopformerScorer (GCAE tokens → Transformer MSE → pose anomaly score)
  - AnomalyScorer → CombinedScorer (kết hợp kinematic + pose)
  - score_with_history() thay score() để tận dụng bbox_h history
  - Dọn track buffer khi track bị mất
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Set
import torch

from modules.detector import YOLOWorldDetector
from modules.tracker import ByteTrackWrapper
from modules.features import BehaviorFeatureExtractor
from modules.anomaly import CombinedScorer
from modules.alert import AlertSystem
from modules.database import DatabaseStorage
from modules.visualizer import Visualizer

# PoseExtractor là optional — pipeline vẫn chạy nếu thiếu
try:
    from modules.pose_extractor import PoseExtractor
    _POSE_AVAILABLE = True
except ImportError:
    _POSE_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# CẤU HÌNH
# ══════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    # Camera
    source: str  = "0"
    width:  int  = 1280
    height: int  = 720
    fps_limit: int = 15

    # Detection
    yolo_model: str = "yolo_world_v2_l"
    detection_classes: list = field(default_factory=lambda: [
        "person", "bag", "backpack", "handbag", "suitcase"
    ])
    conf_threshold: float = 0.35
    nms_iou:        float = 0.45

    # Tracking
    track_thresh: float = 0.5
    track_buffer: int   = 30
    match_thresh: float = 0.8
    min_box_area: float = 100.0

    # Pose extraction (YOLOv8-pose)
    pose_model:    str   = "yolov8n-pose.pt"   # nano = nhanh nhất
    pose_conf:     float = 0.5
    enable_pose:   bool  = True                # tắt nếu không cần

    # Anomaly — kinematic
    kinematic_model_path: str   = "models/anomaly_model.pkl"
    kinematic_threshold:  float = 0.65

    # Anomaly — pose (Shopformer)
    pose_model_path:   str   = "models/pose_tokenizer.pt"
    pose_threshold:    float = 0.60
    kinematic_weight:  float = 0.45
    pose_weight:       float = 0.55

    # Legacy alias (giữ để không break code cũ)
    model_path:        str   = ""
    anomaly_threshold: float = 0.65

    # Feature window
    window_size: int = 30

    # Output
    display:        bool = True
    save_video:     str  = ""
    db_path:        str  = "data/events.db"
    alert_cooldown: int  = 10

    def __post_init__(self):
        # Legacy: nếu dùng model_path cũ → map sang kinematic_model_path
        if self.model_path and not self.kinematic_model_path:
            self.kinematic_model_path = self.model_path
        if self.anomaly_threshold != 0.65:
            self.kinematic_threshold = self.anomaly_threshold


# ══════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════

class RetailAnomalyPipeline:

    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self._init_components()
        self.frame_count   = 0
        self.fps           = 0.0
        self._t            = time.time()
        self._active_tracks: Set[int] = set()

    # ------------------------------------------------------------------
    def _init_components(self):
        cfg = self.cfg
        logger.info("Initialising components …")

        # ── Detection ─────────────────────────────────────────────
        self.detector = YOLOWorldDetector(
            model_name=cfg.yolo_model,
            classes=cfg.detection_classes,
            conf=cfg.conf_threshold,
            iou=cfg.nms_iou,
        )

        # ── Tracking ──────────────────────────────────────────────
        self.tracker = ByteTrackWrapper(
            track_thresh=cfg.track_thresh,
            track_buffer=cfg.track_buffer,
            match_thresh=cfg.match_thresh,
            min_box_area=cfg.min_box_area,
        )

        # ── Pose extraction (optional) ────────────────────────────
        self.pose_extractor = None
        if cfg.enable_pose and _POSE_AVAILABLE:
            self.pose_extractor = PoseExtractor(
                model_name=cfg.pose_model,
                conf=cfg.pose_conf,
            )
            logger.info("PoseExtractor: ON")
        else:
            logger.info(
                "PoseExtractor: OFF"
                + (" (enable_pose=False)" if not cfg.enable_pose
                   else " (pose_extractor.py not found)")
            )

        # ── Feature extraction ────────────────────────────────────
        self.feat_extractor = BehaviorFeatureExtractor(window=cfg.window_size)

        # ── Combined scorer (kinematic + pose) ────────────────────
        self.scorer = CombinedScorer(
            kinematic_model_path=cfg.kinematic_model_path,
            pose_tokenizer_path=cfg.pose_model_path,
            kinematic_threshold=cfg.kinematic_threshold,
            pose_threshold=cfg.pose_threshold,
            kinematic_weight=cfg.kinematic_weight,
            pose_weight=cfg.pose_weight,
        )

        # ── Alert + DB + Visualizer ───────────────────────────────
        self.alert_system = AlertSystem(cooldown=cfg.alert_cooldown)
        self.db            = DatabaseStorage(db_path=cfg.db_path)
        self.viz           = Visualizer()

        logger.info("All components ready.")

    # ------------------------------------------------------------------
    def _open_capture(self):
        src = self.cfg.source
        try:
            src = int(src)
        except ValueError:
            pass
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {src}")
        return cap

    # ------------------------------------------------------------------
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.resize(frame, (self.cfg.width, self.cfg.height))
        lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l     = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # ------------------------------------------------------------------
    def _compute_fps(self):
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            now      = time.time()
            self.fps = 30 / max(now - self._t, 1e-6)
            self._t  = now

    # ------------------------------------------------------------------
    def run(self):
        cap    = self._open_capture()
        writer = None

        if self.cfg.save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                self.cfg.save_video, fourcc, self.cfg.fps_limit,
                (self.cfg.width, self.cfg.height),
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
        """Single-frame inference — callable từ external code / tests."""
        ts = time.time()

        # ── 1. Preprocess ─────────────────────────────────────────
        frame = self._preprocess(frame)

        # ── 2. Detect ─────────────────────────────────────────────
        detections = self.detector.detect(frame)

        # ── 3. Track ──────────────────────────────────────────────
        tracks = self.tracker.update(detections, frame)

        # ── 3b. Dọn buffer cho track đã mất ──────────────────────
        current_ids = {t.track_id for t in tracks}
        lost_ids    = self._active_tracks - current_ids
        for tid in lost_ids:
            self.scorer.clear_track(tid)
            self.feat_extractor._histories.pop(tid, None)
        self._active_tracks = current_ids

        # ── 4. Pose extraction (per track) ────────────────────────
        pose_map: Dict[int, Optional[np.ndarray]] = {}
        if self.pose_extractor is not None:
            for track in tracks:
                kpts = self.pose_extractor.extract(frame, track.tlbr)
                pose_map[track.track_id] = kpts   # (17, 3) hoặc None

        # ── 5. Feature extraction ─────────────────────────────────
        features_map = self.feat_extractor.update(tracks, frame)

        # ── 6. Combined scoring (kinematic + pose) ────────────────
        anomaly_map: Dict[int, dict] = {}
        for tid, feat_vec in features_map.items():
            track  = next((t for t in tracks if t.track_id == tid), None)
            bbox_h = (
                float(track.tlbr[3] - track.tlbr[1]) / frame.shape[0]
                if track else None
            )
            kpts = pose_map.get(tid)   # (17, 3) hoặc None

            final_score, is_anom, detail = self.scorer.score(
                track_id=tid,
                feature_vec=feat_vec,
                kpts=kpts,
                bbox_h=bbox_h,
            )
            anomaly_map[tid] = {
                "score":      final_score,
                "is_anomaly": is_anom,
                "action":     detail.get("action", ""),
                "k_score":    detail.get("kinematic_score"),
                "p_score":    detail.get("pose_score"),
            }

        # ── 7. Decision + side effects ────────────────────────────
        for tid, result in anomaly_map.items():
            if result["is_anomaly"]:
                track = next((t for t in tracks if t.track_id == tid), None)
                if track:
                    self.alert_system.trigger(tid, result["score"], frame, ts)
                    self.db.log_event(
                        tid, result["score"], ts, track.tlbr.tolist(),
                        extra={
                            "action":  result.get("action", ""),
                            "k_score": result.get("k_score"),
                            "p_score": result.get("p_score"),
                        },
                    )

        # ── 8. Visualise ──────────────────────────────────────────
        out = self.viz.draw(frame, tracks, anomaly_map, fps=self.fps)
        return out


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retail Anomaly Detection – Inference")
    parser.add_argument("--source",     default="0")
    parser.add_argument("--model",      default="models/anomaly_model.pkl",
                        help="Kinematic model path (.pkl)")
    parser.add_argument("--pose-model", default="models/pose_tokenizer.pt",
                        help="Shopformer pose tokenizer path (.pt)")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-pose",    action="store_true",
                        help="Tắt pose extraction (nhanh hơn, kém chính xác hơn)")
    parser.add_argument("--save",       default="")
    parser.add_argument("--threshold",  type=float, default=0.65)
    args = parser.parse_args()

    cfg = PipelineConfig(
        source=args.source,
        kinematic_model_path=args.model,
        pose_model_path=args.pose_model,
        kinematic_threshold=args.threshold,
        display=not args.no_display,
        save_video=args.save,
        enable_pose=not args.no_pose,
    )
    RetailAnomalyPipeline(cfg).run()
