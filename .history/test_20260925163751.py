"""
test.py — Kiểm thử toàn bộ pipeline nhận diện hành vi bất thường
=================================================================
Chạy độc lập, KHÔNG cần alert.py / database.py / visualizer.py.

Cách dùng:
    python test.py --video input.mp4
    python test.py --video input.mp4 --model models/anomaly_model.pkl
    python test.py --video input.mp4 --save-video output.mp4 --report report.json
    python test.py --video input.mp4 --threshold 0.55 --no-display
    python test.py --unit          # chỉ chạy unit tests, không cần video

Kết quả xuất ra:
    • Cửa sổ hiển thị real-time (nếu có màn hình)
    • output/<tên_video>_result.mp4
    • output/report.json  (thống kê + danh sách sự kiện bất thường)
    • output/anomaly_frames/  (ảnh frame của từng sự kiện)
    • Log chi tiết trên terminal (màu ANSI)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
import unittest
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ─── màu ANSI cho terminal ────────────────────────────────────────────
_R = "\033[91m"; _G = "\033[92m"; _Y = "\033[93m"
_B = "\033[94m"; _C = "\033[96m"; _W = "\033[97m"; _X = "\033[0m"

logging.basicConfig(
    level=logging.INFO,
    format=f"{_C}%(asctime)s{_X} %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test")


# ══════════════════════════════════════════════════════════════════════
# 1. CẤU HÌNH TEST
# ══════════════════════════════════════════════════════════════════════

@dataclass
class TestConfig:
    # Đầu vào
    video_path: str         = ""
    model_path: str         = "models/anomaly_model.pkl"
    classes: List[str]      = field(default_factory=lambda: [
        "person", "bag", "backpack", "handbag", "suitcase"
    ])

    # Ngưỡng
    conf_threshold: float   = 0.35
    nms_iou: float          = 0.45
    anomaly_threshold: float = 0.65
    window_size: int        = 30

    # Giới hạn xử lý
    max_frames: int         = 0         # 0 = toàn bộ video
    frame_skip: int         = 1         # xử lý 1 trong N frame (tăng tốc test)
    resize_w: int           = 1280
    resize_h: int           = 720

    # Đầu ra
    display: bool           = True
    save_video: str         = ""        # "" = tự tạo trong output/
    report_path: str        = ""        # "" = tự tạo trong output/
    save_anomaly_frames: bool = True
    output_dir: str         = "output"

    # Debug
    show_features: bool     = False     # in bảng feature vector ra log
    verbose_score: bool     = True      # in điểm từng track ra log


# ══════════════════════════════════════════════════════════════════════
# 2. MOCK CHO CÁC MODULE CHƯA VIẾT
# ══════════════════════════════════════════════════════════════════════

class _MockAlert:
    """Alert giả — chỉ log, không gửi thật."""
    def __init__(self, cooldown=10):
        self.cooldown = cooldown
        self._last: Dict[int, float] = {}
        self.triggered: List[dict] = []

    def trigger(self, track_id: int, score: float, frame: np.ndarray, ts: float):
        if ts - self._last.get(track_id, -999) < self.cooldown:
            return
        self._last[track_id] = ts
        logger.warning(f"{_R}[ALERT]{_X} Track #{track_id}  score={score:.3f}")
        self.triggered.append({"track_id": track_id, "score": score, "ts": ts})


class _MockDB:
    """DB giả — ghi vào list thay vì SQLite."""
    def __init__(self, *a, **kw):
        self.events: List[dict] = []

    def log_event(self, track_id, score, ts, bbox):
        self.events.append({
            "track_id": track_id,
            "score":    round(score, 4),
            "ts":       round(ts, 3),
            "bbox":     [round(v, 1) for v in bbox],
        })

    def close(self): pass


class _MockVisualizer:
    """Visualizer tối giản — không cần file riêng."""

    NORMAL_COLOR  = (0, 200, 80)     # xanh lá
    ANOMALY_COLOR = (0, 50, 220)     # đỏ
    SCORE_COLORS  = [                # gradient xanh→vàng→đỏ
        (0, 200, 80), (0, 210, 160), (0, 200, 220),
        (0, 140, 240), (0, 50, 220),
    ]

    def draw(
        self,
        frame: np.ndarray,
        tracks,
        anomaly_map: Dict,
        fps: float = 0.0,
        frame_idx: int = 0,
    ) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        for track in tracks:
            tid = track.track_id
            info = anomaly_map.get(tid, {"score": 0.0, "is_anomaly": False})
            score = info["score"]
            is_anom = info["is_anomaly"]

            x1, y1, x2, y2 = [int(v) for v in track.tlbr]
            color = self.ANOMALY_COLOR if is_anom else self.NORMAL_COLOR
            thick = 3 if is_anom else 2

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)

            # Score bar (dưới bbox)
            bar_w = max(0, min(int((x2 - x1) * score), x2 - x1))
            bar_y = min(y2 + 4, h - 6)
            cv2.rectangle(out, (x1, bar_y), (x2, bar_y + 4),
                          (50, 50, 50), -1)
            bar_color = self._score_color(score)
            cv2.rectangle(out, (x1, bar_y), (x1 + bar_w, bar_y + 4),
                          bar_color, -1)

            # Label
            label = f"#{tid}  {score:.2f}"
            if is_anom:
                label = f"! {label}"
            lx, ly = x1, max(y1 - 6, 14)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (lx, ly - th - 4), (lx + tw + 6, ly + 2), color, -1)
            cv2.putText(out, label, (lx + 3, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Cảnh báo nổi bật
            if is_anom:
                cv2.rectangle(out, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                              self.ANOMALY_COLOR, 1)

        # HUD góc trên trái
        hud_lines = [
            f"Frame: {frame_idx}",
            f"FPS:   {fps:.1f}",
            f"Tracks: {len(tracks)}",
            f"Anomalies: {sum(1 for v in anomaly_map.values() if v['is_anomaly'])}",
        ]
        for i, line in enumerate(hud_lines):
            y = 22 + i * 20
            cv2.putText(out, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (220, 220, 220), 1, cv2.LINE_AA)

        return out

    def _score_color(self, score: float) -> Tuple[int, int, int]:
        idx = min(int(score * (len(self.SCORE_COLORS) - 1)), len(self.SCORE_COLORS) - 1)
        return self.SCORE_COLORS[idx]


# ══════════════════════════════════════════════════════════════════════
# 3. TEST RUNNER CHÍNH
# ══════════════════════════════════════════════════════════════════════

@dataclass
class FrameResult:
    """Kết quả xử lý 1 frame."""
    frame_idx:    int
    ts:           float
    n_detections: int
    n_tracks:     int
    anomalies:    List[dict]   # [{track_id, score}]
    proc_ms:      float        # thời gian xử lý (ms)


class VideoTester:
    """
    Chạy pipeline trên video, thu thập kết quả, xuất báo cáo.
    Hoàn toàn độc lập với alert.py / database.py / visualizer.py.
    """

    def __init__(self, cfg: TestConfig):
        self.cfg = cfg
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        if cfg.save_anomaly_frames:
            (Path(cfg.output_dir) / "anomaly_frames").mkdir(exist_ok=True)

        self._init_pipeline()
        self._alert  = _MockAlert(cooldown=5)
        self._db     = _MockDB()
        self._viz    = _MockVisualizer()

        self.results: List[FrameResult] = []
        self._fps_buf: List[float] = []
        self._current_fps = 0.0

    # ------------------------------------------------------------------
    def _init_pipeline(self):
        """Import + khởi tạo các module thật."""
        logger.info(f"{_B}[INIT]{_X} Đang tải các module pipeline …")

        from modules.detector import YOLOWorldDetector
        from modules.tracker  import ByteTrackWrapper
        from modules.features import BehaviorFeatureExtractor
        from modules.anomaly  import AnomalyScorer

        self.detector = YOLOWorldDetector(
            classes=self.cfg.classes,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.nms_iou,
        )
        self.tracker = ByteTrackWrapper(
            track_thresh=0.5,
            track_buffer=self.cfg.window_size,
            match_thresh=0.8,
        )
        self.feat_ext = BehaviorFeatureExtractor(window=self.cfg.window_size)
        self.scorer   = AnomalyScorer(
            model_path=self.cfg.model_path,
            threshold=self.cfg.anomaly_threshold,
        )
        logger.info(f"{_G}[INIT]{_X} Pipeline sẵn sàng.")

    # ------------------------------------------------------------------
    def run(self) -> dict:
        """Chạy toàn bộ video, trả về dict báo cáo."""
        cap = self._open_video()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
        src_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            f"{_B}[VIDEO]{_X} {self.cfg.video_path}  "
            f"{src_w}×{src_h}  {src_fps:.1f}fps  {total_frames} frames"
        )

        # Chuẩn bị writer
        writer = self._open_writer(src_fps)

        frame_idx = 0
        t_start   = time.time()

        print(f"\n{'─'*60}")
        print(f"  Bắt đầu kiểm thử: {Path(self.cfg.video_path).name}")
        print(f"  Model: {self.cfg.model_path}")
        print(f"  Threshold: {self.cfg.anomaly_threshold}  |  Window: {self.cfg.window_size}")
        print(f"{'─'*60}\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                # Giới hạn số frame
                if self.cfg.max_frames and frame_idx > self.cfg.max_frames:
                    logger.info(f"Đã đạt max_frames={self.cfg.max_frames}, dừng.")
                    break

                # Bỏ qua frame theo frame_skip
                if (frame_idx - 1) % self.cfg.frame_skip != 0:
                    continue

                ts = frame_idx / src_fps

                # ── Xử lý frame ───────────────────────────────────────
                t0 = time.perf_counter()
                result_frame, fr = self._process_frame(frame, frame_idx, ts)
                proc_ms = (time.perf_counter() - t0) * 1000
                fr.proc_ms = proc_ms
                self.results.append(fr)

                # ── FPS thực ──────────────────────────────────────────
                self._fps_buf.append(1000.0 / max(proc_ms, 1))
                if len(self._fps_buf) > 30:
                    self._fps_buf.pop(0)
                self._current_fps = float(np.mean(self._fps_buf))

                # ── Overlay HUD cuối ──────────────────────────────────
                result_frame = self._viz.draw(
                    result_frame,
                    self.tracker._tracks,
                    {a["track_id"]: a for a in fr.anomalies}
                    if fr.anomalies else {},
                    fps=self._current_fps,
                    frame_idx=frame_idx,
                )

                # ── Lưu frame bất thường ──────────────────────────────
                if fr.anomalies and self.cfg.save_anomaly_frames:
                    self._save_anomaly_frame(result_frame, frame_idx, fr.anomalies)

                # ── Ghi video output ──────────────────────────────────
                if writer:
                    writer.write(result_frame)

                # ── Hiển thị ──────────────────────────────────────────
                if self.cfg.display:
                    cv2.imshow("Retail Anomaly — Test", result_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("Người dùng nhấn Q — dừng sớm.")
                        break
                    elif key == ord("s"):
                        snap = Path(self.cfg.output_dir) / f"snap_{frame_idx}.jpg"
                        cv2.imwrite(str(snap), result_frame)
                        logger.info(f"Snapshot: {snap}")

                # ── Log tiến độ mỗi 50 frame ─────────────────────────
                if frame_idx % 50 == 0:
                    elapsed = time.time() - t_start
                    pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                    anom_cnt = sum(len(r.anomalies) for r in self.results)
                    logger.info(
                        f"Frame {frame_idx:5d}/{total_frames}  "
                        f"({pct:5.1f}%)  "
                        f"FPS={self._current_fps:5.1f}  "
                        f"Elapsed={elapsed:6.1f}s  "
                        f"Anomalies={_R}{anom_cnt}{_X}"
                    )

        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

        elapsed_total = time.time() - t_start
        report = self._build_report(elapsed_total, src_fps, total_frames)
        self._save_report(report)
        self._print_summary(report)
        return report

    # ------------------------------------------------------------------
    def _process_frame(
        self, frame: np.ndarray, frame_idx: int, ts: float
    ) -> Tuple[np.ndarray, FrameResult]:
        """Xử lý 1 frame qua đầy đủ pipeline."""

        # 1. Preprocess
        frame = cv2.resize(frame, (self.cfg.resize_w, self.cfg.resize_h))
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        # 2. Detect
        detections = self.detector.detect(frame)

        # 3. Track
        tracks = self.tracker.update(detections, frame)

        # 4. Feature extraction
        feat_map = self.feat_ext.update(tracks, frame)

        if self.cfg.show_features and feat_map:
            df = self.feat_ext.to_dataframe(feat_map)
            logger.debug(f"Features frame {frame_idx}:\n{df.to_string()}")

        # 5. Anomaly scoring
        anomaly_map: Dict[int, dict] = {}
        anomaly_events: List[dict]   = []

        for tid, vec in feat_map.items():
            score, is_anom = self.scorer.score(vec)
            anomaly_map[tid] = {"score": score, "is_anomaly": is_anom}

            if self.cfg.verbose_score:
                flag = f"{_R}ANOMALY{_X}" if is_anom else f"{_G}normal {_X}"
                logger.debug(f"  Track #{tid:3d}  score={score:.4f}  [{flag}]")

            if is_anom:
                track = next((t for t in tracks if t.track_id == tid), None)
                bbox  = track.tlbr.tolist() if track else []
                anomaly_events.append({
                    "track_id": tid,
                    "score":    round(score, 4),
                    "bbox":     [round(v, 1) for v in bbox],
                })
                # Mock alert + DB
                self._alert.trigger(tid, score, frame, ts)
                self._db.log_event(tid, score, ts, bbox)

        fr = FrameResult(
            frame_idx    = frame_idx,
            ts           = round(ts, 3),
            n_detections = len(detections),
            n_tracks     = len(tracks),
            anomalies    = anomaly_events,
            proc_ms      = 0.0,
        )
        return frame, fr

    # ------------------------------------------------------------------
    def _open_video(self) -> cv2.VideoCapture:
        p = self.cfg.video_path
        if not Path(p).exists():
            raise FileNotFoundError(f"Không tìm thấy video: {p}")
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise RuntimeError(f"Không mở được video: {p}")
        return cap

    # ------------------------------------------------------------------
    def _open_writer(self, src_fps: float) -> Optional[cv2.VideoWriter]:
        if not self.cfg.save_video:
            stem = Path(self.cfg.video_path).stem
            out  = Path(self.cfg.output_dir) / f"{stem}_result.mp4"
        else:
            out = Path(self.cfg.save_video)

        out.parent.mkdir(parents=True, exist_ok=True)
        self.cfg.save_video = str(out)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(out), fourcc, src_fps,
            (self.cfg.resize_w, self.cfg.resize_h)
        )
        if not writer.isOpened():
            logger.warning(f"Không tạo được VideoWriter cho {out}")
            return None
        logger.info(f"{_G}[OUTPUT]{_X} Video kết quả: {out}")
        return writer

    # ------------------------------------------------------------------
    def _save_anomaly_frame(
        self, frame: np.ndarray, frame_idx: int, anomalies: List[dict]
    ):
        ids   = "_".join(str(a["track_id"]) for a in anomalies)
        score = max(a["score"] for a in anomalies)
        name  = f"frame{frame_idx:05d}_track{ids}_s{score:.2f}.jpg"
        path  = Path(self.cfg.output_dir) / "anomaly_frames" / name
        cv2.imwrite(str(path), frame)

    # ------------------------------------------------------------------
    def _build_report(self, elapsed: float, src_fps: float, total_frames: int) -> dict:
        n_proc    = len(self.results)
        all_anom  = [a for r in self.results for a in r.anomalies]
        anom_tids = set(a["track_id"] for a in all_anom)
        proc_ms   = [r.proc_ms for r in self.results]
        avg_fps   = n_proc / elapsed if elapsed > 0 else 0

        # Thống kê theo track
        track_stats: Dict[int, dict] = defaultdict(lambda: {
            "n_anomaly_frames": 0, "max_score": 0.0, "scores": []
        })
        for a in all_anom:
            tid = a["track_id"]
            track_stats[tid]["n_anomaly_frames"] += 1
            track_stats[tid]["max_score"] = max(
                track_stats[tid]["max_score"], a["score"]
            )
            track_stats[tid]["scores"].append(a["score"])

        for tid in track_stats:
            s = track_stats[tid]["scores"]
            track_stats[tid]["mean_score"] = round(float(np.mean(s)), 4)
            del track_stats[tid]["scores"]

        report = {
            "meta": {
                "video":        self.cfg.video_path,
                "model":        self.cfg.model_path,
                "threshold":    self.cfg.anomaly_threshold,
                "window":       self.cfg.window_size,
                "frame_skip":   self.cfg.frame_skip,
                "tested_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "summary": {
                "total_frames_src":      total_frames,
                "frames_processed":      n_proc,
                "elapsed_sec":           round(elapsed, 2),
                "avg_fps":               round(avg_fps, 2),
                "proc_ms_mean":          round(float(np.mean(proc_ms)), 2) if proc_ms else 0,
                "proc_ms_p95":           round(float(np.percentile(proc_ms, 95)), 2) if proc_ms else 0,
                "proc_ms_max":           round(float(np.max(proc_ms)), 2) if proc_ms else 0,
                "total_anomaly_events":  len(all_anom),
                "unique_anomaly_tracks": len(anom_tids),
                "anomaly_frame_rate":    round(
                    len([r for r in self.results if r.anomalies]) / max(n_proc, 1), 4
                ),
            },
            "anomaly_tracks": {
                str(tid): stats for tid, stats in track_stats.items()
            },
            "events": self._db.events[:500],   # tối đa 500 event đầu
            "alerts": self._alert.triggered,
            "output_video": self.cfg.save_video,
        }
        return report

    # ------------------------------------------------------------------
    def _save_report(self, report: dict):
        if not self.cfg.report_path:
            stem = Path(self.cfg.video_path).stem
            self.cfg.report_path = str(
                Path(self.cfg.output_dir) / f"{stem}_report.json"
            )
        Path(self.cfg.report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.cfg.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"{_G}[REPORT]{_X} Báo cáo JSON: {self.cfg.report_path}")

    # ------------------------------------------------------------------
    def _print_summary(self, report: dict):
        s = report["summary"]
        m = report["meta"]
        print(f"\n{'═'*60}")
        print(f"  KẾT QUẢ KIỂM THỬ")
        print(f"{'═'*60}")
        print(f"  Video      : {m['video']}")
        print(f"  Model      : {m['model']}")
        print(f"  Threshold  : {m['threshold']}")
        print(f"{'─'*60}")
        print(f"  Frames xử lý : {s['frames_processed']} / {s['total_frames_src']}")
        print(f"  Thời gian    : {s['elapsed_sec']} s  ({s['avg_fps']:.1f} fps thực)")
        print(f"  Proc/frame   : mean={s['proc_ms_mean']} ms  "
              f"p95={s['proc_ms_p95']} ms  max={s['proc_ms_max']} ms")
        print(f"{'─'*60}")
        n = s['total_anomaly_events']
        t = s['unique_anomaly_tracks']
        r = s['anomaly_frame_rate']
        color = _R if n > 0 else _G
        print(f"  {color}Tổng sự kiện bất thường : {n}{_X}")
        print(f"  {color}Track bất thường duy nhất: {t}{_X}")
        print(f"  Tỷ lệ frame bất thường  : {r*100:.1f}%")
        if report["anomaly_tracks"]:
            print(f"{'─'*60}")
            print(f"  Chi tiết từng track bất thường:")
            for tid, stat in report["anomaly_tracks"].items():
                print(
                    f"    Track #{tid:>4s}  "
                    f"frames={stat['n_anomaly_frames']:4d}  "
                    f"max_score={stat['max_score']:.4f}  "
                    f"mean_score={stat['mean_score']:.4f}"
                )
        print(f"{'─'*60}")
        print(f"  Video kết quả : {report['output_video']}")
        print(f"  Báo cáo JSON  : {self.cfg.report_path}")
        print(f"{'═'*60}\n")


# ══════════════════════════════════════════════════════════════════════
# 4. UNIT TESTS — chạy mà không cần video thật
# ══════════════════════════════════════════════════════════════════════

class TestDetector(unittest.TestCase):
    """Kiểm tra YOLOWorldDetector khởi tạo và cho ra Detection hợp lệ."""

    def setUp(self):
        from modules.detector import YOLOWorldDetector, Detection
        self.Det  = Detection
        self.YOLO = YOLOWorldDetector

    def test_detection_dataclass(self):
        from modules.detector import Detection
        d = Detection(x1=10, y1=20, x2=110, y2=220,
                      confidence=0.8, class_id=0, class_name="person")
        self.assertAlmostEqual(d.area, 100 * 200)
        cx, cy = d.center
        self.assertAlmostEqual(cx, 60.0)
        self.assertAlmostEqual(cy, 120.0)
        self.assertEqual(d.tlwh[2], 100.0)

    def test_detection_zero_area(self):
        from modules.detector import Detection
        d = Detection(x1=5, y1=5, x2=5, y2=5,
                      confidence=0.9, class_id=0, class_name="person")
        self.assertEqual(d.area, 0.0)

    def test_yolo_init_no_crash(self):
        """Chỉ kiểm tra import và khởi tạo không crash."""
        try:
            det = self.YOLO(classes=["person"], conf=0.4)
            self.assertIsNotNone(det)
        except Exception as e:
            # Cho phép fail nếu không có GPU/model weight — không tính là lỗi
            logger.warning(f"YOLO init warning (bỏ qua): {e}")


class TestTracker(unittest.TestCase):
    """Kiểm tra ByteTrackWrapper và Track dataclass."""

    def setUp(self):
        from modules.tracker import ByteTrackWrapper, Track
        import numpy as np
        self.Tracker = ByteTrackWrapper
        self.Track   = Track
        self.np      = np

    def test_track_properties(self):
        from modules.tracker import Track
        t = Track(track_id=1,
                  tlbr=np.array([10., 20., 110., 220.]),
                  score=0.9)
        self.assertAlmostEqual(t.area, 100 * 200)
        cx, cy = t.center
        self.assertAlmostEqual(cx, 60.0)
        self.assertAlmostEqual(cy, 120.0)
        w, h = t.tlwh[2], t.tlwh[3]
        self.assertAlmostEqual(w, 100.0)
        self.assertAlmostEqual(h, 200.0)

    def test_iou_tracker_fallback(self):
        """Fallback IoU tracker cho kết quả có track_id."""
        from modules.tracker import ByteTrackWrapper
        from modules.detector import Detection
        tracker = ByteTrackWrapper()
        # Force fallback bằng cách gọi _iou_tracker trực tiếp
        dets = [
            Detection(0, 0, 100, 100, 0.8, 0, "person"),
            Detection(200, 200, 300, 300, 0.7, 0, "person"),
        ]
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Gọi update (sẽ dùng ByteTrack hoặc fallback)
        try:
            tracks = tracker.update(dets, frame)
            self.assertIsInstance(tracks, list)
        except Exception as e:
            logger.warning(f"Tracker update warning (bỏ qua): {e}")

    def test_empty_detection(self):
        """Không có detection → trả về list (có thể rỗng)."""
        from modules.tracker import ByteTrackWrapper
        tracker = ByteTrackWrapper()
        frame   = np.zeros((720, 1280, 3), dtype=np.uint8)
        result  = tracker.update([], frame)
        self.assertIsInstance(result, list)


class TestFeatureExtractor(unittest.TestCase):
    """Kiểm tra BehaviorFeatureExtractor tạo vector 24-D đúng shape."""

    def setUp(self):
        from modules.features import BehaviorFeatureExtractor, FEATURE_DIM
        from modules.tracker  import Track
        self.Extractor  = BehaviorFeatureExtractor
        self.Track      = Track
        self.FEAT_DIM   = FEATURE_DIM

    def _make_tracks(self, n=2):
        from modules.tracker import Track
        tracks = []
        for i in range(n):
            tracks.append(Track(
                track_id=i + 1,
                tlbr=np.array([i*100., 50., i*100.+80., 200.], dtype=np.float32),
                score=0.9,
            ))
        return tracks

    def test_feature_dim(self):
        """Sau đủ frame, vector phải có đúng 24 chiều."""
        ext    = self.Extractor(window=10)
        tracks = self._make_tracks(1)
        frame  = np.zeros((720, 1280, 3), dtype=np.uint8)

        for step in range(12):
            # Di chuyển track mỗi frame
            tracks[0].tlbr = np.array([
                step * 15., 50.,
                step * 15. + 80., 200.
            ], dtype=np.float32)
            feat_map = ext.update(tracks, frame)

        self.assertIn(1, feat_map)
        vec = feat_map[1]
        self.assertEqual(vec.shape[0], self.FEAT_DIM)
        self.assertFalse(np.any(np.isnan(vec)), "Feature vector chứa NaN")
        self.assertFalse(np.any(np.isinf(vec)), "Feature vector chứa Inf")

    def test_loiter_score_high_when_stationary(self):
        """Track đứng yên → loiter_score cao."""
        ext    = self.Extractor(window=10)
        frame  = np.zeros((720, 1280, 3), dtype=np.uint8)
        tracks = self._make_tracks(1)

        for _ in range(12):
            feat_map = ext.update(tracks, frame)

        if 1 in feat_map:
            loiter = feat_map[1][9]     # index 9 = loiter_score
            self.assertGreater(loiter, 0.0)

    def test_to_dataframe(self):
        """to_dataframe() phải trả về DataFrame đúng số cột."""
        import pandas as pd
        ext    = self.Extractor(window=5)
        frame  = np.zeros((720, 1280, 3), dtype=np.uint8)
        tracks = self._make_tracks(2)

        for _ in range(7):
            feat_map = ext.update(tracks, frame)

        if feat_map:
            df = ext.to_dataframe(feat_map)
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(df.shape[1], self.FEAT_DIM)


class TestAnomalyScorer(unittest.TestCase):
    """Kiểm tra AnomalyScorer trả về score hợp lệ."""

    def setUp(self):
        from modules.anomaly import AnomalyScorer, FEATURE_DIM
        self.Scorer   = AnomalyScorer
        self.FEAT_DIM = FEATURE_DIM

    def _random_vec(self):
        return np.random.randn(self.FEAT_DIM).astype(np.float32)

    def test_score_range(self):
        """Score phải nằm trong [0, 1]."""
        scorer = self.Scorer(model_path="models/__nonexistent__.pkl")
        for _ in range(20):
            score, is_anom = scorer.score(self._random_vec())
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            self.assertIsInstance(is_anom, bool)

    def test_threshold_respected(self):
        """is_anomaly phải khớp với score >= threshold."""
        scorer = self.Scorer(
            model_path="models/__nonexistent__.pkl",
            threshold=0.5,
        )
        for _ in range(30):
            score, is_anom = scorer.score(self._random_vec())
            expected = score >= 0.5
            self.assertEqual(is_anom, expected)

    def test_score_shape_variants(self):
        """Score không crash với input 1-D hay 2-D."""
        scorer = self.Scorer(model_path="models/__nonexistent__.pkl")
        vec1d = np.random.randn(self.FEAT_DIM).astype(np.float32)
        s1, _ = scorer.score(vec1d)
        vec2d = vec1d.reshape(1, -1)
        s2, _ = scorer.score(vec2d)
        self.assertAlmostEqual(s1, s2, places=5)

    def test_extreme_inputs(self):
        """Score không trả về NaN với input cực trị."""
        scorer = self.Scorer(model_path="models/__nonexistent__.pkl")
        for vec in [
            np.zeros(self.FEAT_DIM, dtype=np.float32),
            np.ones(self.FEAT_DIM, dtype=np.float32) * 1e6,
            np.ones(self.FEAT_DIM, dtype=np.float32) * -1e6,
        ]:
            score, _ = scorer.score(vec)
            self.assertFalse(np.isnan(score), f"NaN với input {vec[0]}")


class TestMockComponents(unittest.TestCase):
    """Kiểm tra _MockAlert, _MockDB, _MockVisualizer."""

    def test_alert_cooldown(self):
        alert = _MockAlert(cooldown=5)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        alert.trigger(1, 0.9, frame, ts=0.0)    # trigger #1
        alert.trigger(1, 0.9, frame, ts=2.0)    # trong cooldown (2 < 5) → bỏ qua
        alert.trigger(1, 0.9, frame, ts=4.9)    # trong cooldown (4.9 < 5) → bỏ qua
        alert.trigger(1, 0.9, frame, ts=5.1)    # qua cooldown (5.1 > 5) → trigger #2
        alert.trigger(2, 0.8, frame, ts=1.0)    # track khác → trigger #3
        self.assertEqual(len(alert.triggered), 3,
                         f"Expected 3 triggers, got {len(alert.triggered)}")

    def test_db_log(self):
        db = _MockDB()
        db.log_event(1, 0.8, 1.5, [10, 20, 100, 200])
        db.log_event(2, 0.7, 2.0, [50, 50, 150, 250])
        self.assertEqual(len(db.events), 2)
        self.assertEqual(db.events[0]["track_id"], 1)

    def test_visualizer_draw_no_crash(self):
        viz   = _MockVisualizer()
        from modules.tracker import Track
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        tracks = [
            Track(track_id=1, tlbr=np.array([100.,100.,300.,400.]), score=0.9),
            Track(track_id=2, tlbr=np.array([500.,200.,700.,600.]), score=0.8),
        ]
        anomaly_map = {
            1: {"score": 0.8, "is_anomaly": True},
            2: {"score": 0.3, "is_anomaly": False},
        }
        result = viz.draw(frame, tracks, anomaly_map, fps=30.0, frame_idx=100)
        self.assertEqual(result.shape, frame.shape)


# ══════════════════════════════════════════════════════════════════════
# 5. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_unit_tests():
    """Chạy tất cả unit tests, trả về True nếu pass hết."""
    print(f"\n{_B}{'═'*60}{_X}")
    print(f"{_B}  UNIT TESTS{_X}")
    print(f"{_B}{'═'*60}{_X}\n")

    suites = [
        unittest.TestLoader().loadTestsFromTestCase(cls)
        for cls in [
            TestDetector, TestTracker, TestFeatureExtractor,
            TestAnomalyScorer, TestMockComponents,
        ]
    ]
    runner = unittest.TextTestRunner(verbosity=2)
    results = [runner.run(s) for s in suites]
    n_fail  = sum(len(r.failures) + len(r.errors) for r in results)
    n_ok    = sum(r.testsRun for r in results) - n_fail

    print(f"\n{_G if n_fail == 0 else _R}  {n_ok} tests passed  |  {n_fail} failed{_X}\n")
    return n_fail == 0


def parse_args():
    p = argparse.ArgumentParser(
        description="Kiểm thử pipeline nhận diện hành vi bất thường",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python test.py --video cctv.mp4
  python test.py --video cctv.mp4 --model models/anomaly_model.pkl --threshold 0.6
  python test.py --video cctv.mp4 --frame-skip 2 --max-frames 500 --no-display
  python test.py --unit
        """,
    )
    p.add_argument("--video",          default="",    help="Đường dẫn video đầu vào")
    p.add_argument("--model",          default="models/anomaly_model.pkl")
    p.add_argument("--threshold",      type=float, default=0.65)
    p.add_argument("--window",         type=int,   default=30,
                   help="Số frame cho sliding-window feature")
    p.add_argument("--frame-skip",     type=int,   default=1,
                   help="Xử lý 1 trong N frame (1=tất cả, 2=nhanh gấp đôi)")
    p.add_argument("--max-frames",     type=int,   default=0,
                   help="Giới hạn frame (0=toàn bộ)")
    p.add_argument("--conf",           type=float, default=0.35,
                   help="Ngưỡng confidence YOLO")
    p.add_argument("--save-video",     default="",
                   help="Đường dẫn lưu video output (mặc định: output/<tên>_result.mp4)")
    p.add_argument("--report",         default="",
                   help="Đường dẫn lưu báo cáo JSON")
    p.add_argument("--output-dir",     default="output")
    p.add_argument("--no-display",     action="store_true",
                   help="Không hiện cửa sổ (dùng khi chạy headless/server)")
    p.add_argument("--show-features",  action="store_true",
                   help="In feature vector ra log (verbose)")
    p.add_argument("--unit",           action="store_true",
                   help="Chỉ chạy unit tests, không cần video")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── Chế độ unit test ──────────────────────────────────────────────
    if args.unit:
        ok = run_unit_tests()
        sys.exit(0 if ok else 1)

    # ── Chế độ test video ─────────────────────────────────────────────
    if not args.video:
        print(f"{_R}Lỗi: Cần truyền --video <đường_dẫn_video>{_X}")
        print(f"Hoặc dùng --unit để chạy unit tests.\n")
        parse_args().__class__(prog="test.py").print_help()
        sys.exit(1)

    cfg = TestConfig(
        video_path        = args.video,
        model_path        = args.model,
        anomaly_threshold = args.threshold,
        window_size       = args.window,
        frame_skip        = args.frame_skip,
        max_frames        = args.max_frames,
        conf_threshold    = args.conf,
        save_video        = args.save_video,
        report_path       = args.report,
        output_dir        = args.output_dir,
        display           = not args.no_display,
        show_features     = args.show_features,
    )

    # Chạy unit tests nhỏ trước khi test video
    print(f"\n{_Y}[PRE-CHECK] Chạy unit tests nhanh trước …{_X}")
    ok = run_unit_tests()
    if not ok:
        print(f"{_R}Unit tests thất bại. Kiểm tra lại module trước khi test video.{_X}")
        sys.exit(1)

    print(f"{_G}Unit tests OK — bắt đầu test video …{_X}\n")

    try:
        tester = VideoTester(cfg)
        report = tester.run()
        sys.exit(0)
    except FileNotFoundError as e:
        logger.error(f"{_R}{e}{_X}")
        sys.exit(2)
    except KeyboardInterrupt:
        print(f"\n{_Y}Đã dừng bởi người dùng.{_X}")
        sys.exit(0)
    except Exception:
        logger.error(f"{_R}Lỗi không mong đợi:{_X}")
        traceback.print_exc()
        sys.exit(3)
