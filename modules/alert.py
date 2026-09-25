"""
modules/alert.py
Hệ thống cảnh báo khi phát hiện hành vi bất thường.

Hỗ trợ 4 kênh cảnh báo:
  1. Console  — in log màu ra terminal (luôn bật)
  2. Sound    — beep âm thanh qua hệ thống (Windows / Linux)
  3. Image    — lưu ảnh snapshot frame bất thường
  4. Webhook  — gửi HTTP POST JSON đến URL tuỳ chỉnh (Slack, Teams, v.v.)

Cách dùng trong inference_pipeline.py:
    from modules.alert import AlertSystem
    alert = AlertSystem(cooldown=10, save_dir="output/alerts", webhook_url="")
    alert.trigger(track_id=3, score=0.82, frame=frame, ts=time.time())
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ── màu ANSI ─────────────────────────────────────────────────────────
_R = "\033[91m"; _Y = "\033[93m"; _G = "\033[92m"; _X = "\033[0m"; _BOLD = "\033[1m"


# ══════════════════════════════════════════════════════════════════════
# CẤU HÌNH
# ══════════════════════════════════════════════════════════════════════

@dataclass
class AlertConfig:
    cooldown:       int   = 10        # giây tối thiểu giữa 2 alert cùng track
    global_cooldown:int   = 2         # giây tối thiểu giữa bất kỳ 2 alert
    save_images:    bool  = True      # lưu ảnh snapshot
    save_dir:       str   = "output/alerts"
    play_sound:     bool  = True      # beep khi có alert
    webhook_url:    str   = ""        # HTTP POST URL (Slack / Teams / custom)
    webhook_timeout:int   = 3         # giây timeout cho HTTP request
    score_levels: dict = field(default_factory=lambda: {
        # score → (nhãn, màu BGR)
        0.90: ("NGUY HIỂM CAO",  (0,   0, 220)),
        0.75: ("BẤT THƯỜNG",     (0,  80, 220)),
        0.65: ("ĐÁNG NGỜ",       (0, 160, 240)),
    })


# ══════════════════════════════════════════════════════════════════════
# EVENT
# ══════════════════════════════════════════════════════════════════════

@dataclass
class AlertEvent:
    track_id:  int
    score:     float
    ts:        float
    label:     str
    image_path: str = ""


# ══════════════════════════════════════════════════════════════════════
# ALERT SYSTEM
# ══════════════════════════════════════════════════════════════════════

class AlertSystem:
    """
    Phát cảnh báo đa kênh khi AnomalyScorer phát hiện bất thường.

    Thread-safe: webhook gửi trong background thread, không block pipeline.
    """

    def __init__(
        self,
        cooldown:    int  = 10,
        save_dir:    str  = "output/alerts",
        save_images: bool = True,
        play_sound:  bool = True,
        webhook_url: str  = "",
        on_alert:    Optional[Callable[[AlertEvent], None]] = None,
    ):
        self.cfg = AlertConfig(
            cooldown    = cooldown,
            save_dir    = save_dir,
            save_images = save_images,
            play_sound  = play_sound,
            webhook_url = webhook_url,
        )
        self._on_alert = on_alert           # callback tuỳ chỉnh từ bên ngoài

        self._last_per_track: Dict[int, float] = {}   # track_id → last ts
        self._last_global: float = -999.0
        self._history: List[AlertEvent] = []
        self._lock = threading.Lock()

        if save_images:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def trigger(
        self,
        track_id: int,
        score:    float,
        frame:    np.ndarray,
        ts:       float,
    ) -> Optional[AlertEvent]:
        """
        Kích hoạt cảnh báo cho track_id với anomaly score.
        Trả về AlertEvent nếu thực sự phát cảnh báo, None nếu trong cooldown.
        """
        with self._lock:
            # Kiểm tra cooldown
            if ts - self._last_per_track.get(track_id, -999) < self.cfg.cooldown:
                return None
            if ts - self._last_global < self.cfg.global_cooldown:
                return None

            self._last_per_track[track_id] = ts
            self._last_global = ts

        label = self._score_label(score)
        event = AlertEvent(track_id=track_id, score=score, ts=ts, label=label)

        # ── Kênh 1: Console ──────────────────────────────────────────
        self._log_console(event)

        # ── Kênh 2: Lưu ảnh ─────────────────────────────────────────
        if self.cfg.save_images and frame is not None:
            event.image_path = self._save_image(frame, event)

        # ── Kênh 3: Sound ────────────────────────────────────────────
        if self.cfg.play_sound:
            threading.Thread(target=self._beep, daemon=True).start()

        # ── Kênh 4: Webhook ──────────────────────────────────────────
        if self.cfg.webhook_url:
            threading.Thread(
                target=self._send_webhook, args=(event,), daemon=True
            ).start()

        # ── Callback tuỳ chỉnh ───────────────────────────────────────
        if self._on_alert:
            try:
                self._on_alert(event)
            except Exception as e:
                log.debug(f"on_alert callback lỗi: {e}")

        with self._lock:
            self._history.append(event)

        return event

    # ------------------------------------------------------------------
    def _score_label(self, score: float) -> str:
        for threshold in sorted(self.cfg.score_levels.keys(), reverse=True):
            if score >= threshold:
                return self.cfg.score_levels[threshold][0]
        return "ĐÁNG NGỜ"

    def _score_color(self, score: float) -> tuple:
        for threshold in sorted(self.cfg.score_levels.keys(), reverse=True):
            if score >= threshold:
                return self.cfg.score_levels[threshold][1]
        return (0, 160, 240)

    # ------------------------------------------------------------------
    def _log_console(self, event: AlertEvent):
        t = time.strftime("%H:%M:%S", time.localtime(event.ts))
        color = _R if event.score >= 0.75 else _Y
        print(
            f"\n{color}{_BOLD}{'▓'*50}{_X}\n"
            f"{color}{_BOLD}  ⚠  [{event.label}]  "
            f"Track #{event.track_id}  |  Score: {event.score:.3f}  |  {t}{_X}\n"
            f"{color}{_BOLD}{'▓'*50}{_X}\n"
        )

    # ------------------------------------------------------------------
    def _save_image(self, frame: np.ndarray, event: AlertEvent) -> str:
        """Lưu frame với overlay thông tin cảnh báo."""
        out = frame.copy()
        h, w = out.shape[:2]

        # Overlay banner phía trên
        color = self._score_color(event.score)
        cv2.rectangle(out, (0, 0), (w, 52), color, -1)
        cv2.rectangle(out, (0, 0), (w, 52), (255, 255, 255), 2)

        t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event.ts))
        cv2.putText(out,
            f"[{event.label}]  Track #{event.track_id}  Score: {event.score:.3f}",
            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
            cv2.LINE_AA,
        )
        cv2.putText(out, t_str,
            (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1,
            cv2.LINE_AA,
        )

        ts_int = int(event.ts * 1000)
        fname  = f"alert_t{event.track_id}_s{event.score:.2f}_{ts_int}.jpg"
        path   = str(Path(self.cfg.save_dir) / fname)
        cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        log.info(f"Alert image saved → {path}")
        return path

    # ------------------------------------------------------------------
    @staticmethod
    def _beep():
        """Phát âm thanh cảnh báo (không block)."""
        try:
            import winsound
            winsound.Beep(880, 400)   # 880 Hz, 400ms
        except ImportError:
            # Linux / macOS
            os.system("echo -e '\\a'")
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _send_webhook(self, event: AlertEvent):
        """Gửi HTTP POST JSON đến webhook URL trong background thread."""
        try:
            import json
            import urllib.request

            payload = json.dumps({
                "text":     f"⚠ [{event.label}] Track #{event.track_id} | Score: {event.score:.3f}",
                "track_id": event.track_id,
                "score":    round(event.score, 4),
                "label":    event.label,
                "ts":       event.ts,
                "time":     time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event.ts)),
            }).encode("utf-8")

            req = urllib.request.Request(
                self.cfg.webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.cfg.webhook_timeout) as resp:
                log.info(f"Webhook gửi OK ({resp.status})")
        except Exception as e:
            log.debug(f"Webhook lỗi: {e}")

    # ------------------------------------------------------------------
    @property
    def history(self) -> List[AlertEvent]:
        """Toàn bộ lịch sử alert trong session."""
        with self._lock:
            return list(self._history)

    def summary(self) -> dict:
        h = self.history
        return {
            "total_alerts":   len(h),
            "unique_tracks":  len({e.track_id for e in h}),
            "max_score":      max((e.score for e in h), default=0.0),
            "by_label": {
                lbl: sum(1 for e in h if e.label == lbl)
                for lbl in {e.label for e in h}
            },
        }
