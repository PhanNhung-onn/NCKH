"""
modules/visualizer.py
Vẽ kết quả detection, tracking và anomaly score lên frame video.

Các thành phần hiển thị:
  • Bounding box — màu theo mức độ bất thường
  • Score bar    — thanh màu bên dưới mỗi bbox
  • Track label  — ID + điểm số + nhãn
  • Trail        — vết di chuyển của track
  • HUD          — FPS, số track, số anomaly (góc trên trái)
  • Mini-map     — bản đồ thu nhỏ vị trí các track (góc dưới phải)
  • Alert flash  — viền đỏ nhấp nháy khi có anomaly

Cách dùng:
    from modules.visualizer import Visualizer
    viz = Visualizer(show_trail=True, show_minimap=True)
    out_frame = viz.draw(frame, tracks, anomaly_map, fps=30.0)
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════════════
# CẤU HÌNH
# ══════════════════════════════════════════════════════════════════════

@dataclass
class VisConfig:
    # Màu theo score (BGR)
    color_normal:  Tuple = (60, 200, 60)      # xanh lá
    color_suspect: Tuple = (0,  200, 240)     # vàng
    color_anomaly: Tuple = (30,  50, 220)     # đỏ

    # Ngưỡng score
    thresh_suspect: float = 0.50
    thresh_anomaly: float = 0.65

    # Trail (vết di chuyển)
    show_trail:    bool = True
    trail_len:     int  = 40          # số điểm lưu trong trail
    trail_alpha:   float = 0.6        # độ trong suốt trail cũ

    # HUD
    show_hud:      bool = True
    hud_alpha:     float = 0.55       # nền HUD

    # Mini-map
    show_minimap:  bool = True
    minimap_size:  Tuple = (180, 120) # w, h

    # Alert flash
    flash_duration: float = 0.6       # giây mỗi lần flash
    flash_color:    Tuple = (0, 0, 200)

    # Zone overlay
    show_zones:    bool = True
    zones: dict = field(default_factory=lambda: {
        "entrance":   ((0.00, 0.00, 0.20, 1.00), (200, 200,  60, 40)),
        "checkout":   ((0.80, 0.00, 1.00, 1.00), (200,  60,  60, 40)),
        "high_value": ((0.40, 0.10, 0.70, 0.50), ( 60,  60, 200, 40)),
    })


# ══════════════════════════════════════════════════════════════════════
# VISUALIZER
# ══════════════════════════════════════════════════════════════════════

class Visualizer:
    """
    Render toàn bộ thông tin pipeline lên frame OpenCV.
    Stateful: lưu trail và màu theo track_id giữa các frame.
    """

    def __init__(
        self,
        show_trail:   bool = True,
        show_minimap: bool = True,
        show_zones:   bool = False,    # tắt mặc định, bật khi đã cấu hình zone
        show_hud:     bool = True,
        config: Optional[VisConfig] = None,
    ):
        self.cfg = config or VisConfig(
            show_trail   = show_trail,
            show_minimap = show_minimap,
            show_zones   = show_zones,
            show_hud     = show_hud,
        )
        # Trail buffer: track_id → deque of (cx, cy)
        self._trails: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.cfg.trail_len)
        )
        # Màu cố định mỗi track (tránh nhấp nháy)
        self._track_colors: Dict[int, Tuple] = {}
        # Thời điểm alert gần nhất (để flash)
        self._last_alert_ts: float = 0.0

        # Palette màu phân biệt track
        self._palette = [
            (255, 100,  50), (50, 180, 255), (100, 255, 100),
            (255, 255,  50), (200,  50, 255), (50, 255, 200),
            (255, 150, 200), (150, 200, 255), (200, 255, 150),
        ]
        self._palette_idx = 0

    # ------------------------------------------------------------------
    def draw(
        self,
        frame:       np.ndarray,
        tracks:      list,                      # List[Track]
        anomaly_map: Dict[int, dict],           # {track_id: {score, is_anomaly}}
        fps:         float = 0.0,
        frame_idx:   int   = 0,
        extra_text:  str   = "",
    ) -> np.ndarray:
        """
        Hàm chính — vẽ tất cả lên frame và trả về frame mới.
        Không thay đổi frame gốc (copy trước khi vẽ).
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # ── Zone overlay (mờ, dưới cùng) ─────────────────────────────
        if self.cfg.show_zones:
            out = self._draw_zones(out, w, h)

        # ── Trail ─────────────────────────────────────────────────────
        if self.cfg.show_trail:
            self._update_trails(tracks)
            out = self._draw_trails(out)

        # ── Tracks + bboxes ───────────────────────────────────────────
        has_anomaly = False
        for track in tracks:
            tid  = track.track_id
            info = anomaly_map.get(tid, {"score": 0.0, "is_anomaly": False})
            score    = info.get("score", 0.0)
            is_anom  = info.get("is_anomaly", False)
            if is_anom:
                has_anomaly = True
            self._draw_track(out, track, score, is_anom)

        # ── Alert flash (viền đỏ toàn màn hình) ──────────────────────
        if has_anomaly:
            self._last_alert_ts = time.time()
        if time.time() - self._last_alert_ts < self.cfg.flash_duration:
            out = self._draw_flash(out, w, h)

        # ── Mini-map ──────────────────────────────────────────────────
        if self.cfg.show_minimap and tracks:
            out = self._draw_minimap(out, tracks, anomaly_map, w, h)

        # ── HUD ───────────────────────────────────────────────────────
        if self.cfg.show_hud:
            n_anom = sum(1 for v in anomaly_map.values() if v.get("is_anomaly"))
            out = self._draw_hud(out, fps, frame_idx, len(tracks), n_anom, extra_text)

        return out

    # ------------------------------------------------------------------
    # ── TRACK + BBOX ──────────────────────────────────────────────────

    def _draw_track(
        self,
        out: np.ndarray,
        track,
        score: float,
        is_anomaly: bool,
    ):
        tid = track.track_id
        x1, y1, x2, y2 = [int(v) for v in track.tlbr]
        h, w = out.shape[:2]

        # Clip to frame
        x1c, y1c = max(x1, 0), max(y1, 0)
        x2c, y2c = min(x2, w - 1), min(y2, h - 1)
        if x2c <= x1c or y2c <= y1c:
            return

        color = self._color_for_score(score)

        # Thickness theo mức độ
        thick = 3 if is_anomaly else 2

        # ── Bounding box ──────────────────────────────────────────────
        cv2.rectangle(out, (x1c, y1c), (x2c, y2c), color, thick)

        # Góc nhọn trang trí (chỉ khi bất thường)
        if is_anomaly:
            self._draw_corner_brackets(out, x1c, y1c, x2c, y2c, color)

        # ── Score bar (dưới bbox) ─────────────────────────────────────
        bar_y1 = min(y2c + 3, h - 8)
        bar_y2 = min(y2c + 9, h - 2)
        bar_w  = x2c - x1c
        cv2.rectangle(out, (x1c, bar_y1), (x2c, bar_y2), (30, 30, 30), -1)
        fill = max(0, min(int(bar_w * score), bar_w))
        if fill > 0:
            cv2.rectangle(out, (x1c, bar_y1), (x1c + fill, bar_y2), color, -1)

        # ── Label ─────────────────────────────────────────────────────
        prefix = "⚠ " if is_anomaly else ""
        label  = f"{prefix}#{tid}  {score:.2f}"
        lx = x1c
        ly = max(y1c - 7, 16)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(out, (lx, ly - th - 4), (lx + tw + 8, ly + 2), color, -1)
        cv2.putText(out, label, (lx + 4, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    def _draw_corner_brackets(self, out, x1, y1, x2, y2, color, size=14, thick=3):
        """Vẽ 4 góc hình chữ L thay cho rectangle thông thường — trông chuyên nghiệp hơn."""
        pts = [
            # top-left
            ((x1, y1 + size), (x1, y1), (x1 + size, y1)),
            # top-right
            ((x2 - size, y1), (x2, y1), (x2, y1 + size)),
            # bottom-left
            ((x1, y2 - size), (x1, y2), (x1 + size, y2)),
            # bottom-right
            ((x2 - size, y2), (x2, y2), (x2, y2 - size)),
        ]
        for p1, corner, p2 in pts:
            cv2.line(out, p1, corner, color, thick, cv2.LINE_AA)
            cv2.line(out, corner, p2, color, thick, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # ── TRAIL ─────────────────────────────────────────────────────────

    def _update_trails(self, tracks):
        for t in tracks:
            cx, cy = int(t.center[0]), int(t.center[1])
            self._trails[t.track_id].append((cx, cy))

    def _draw_trails(self, out: np.ndarray) -> np.ndarray:
        overlay = out.copy()
        for tid, pts in self._trails.items():
            if len(pts) < 2:
                continue
            color = self._get_track_color(tid)
            pts_list = list(pts)
            for i in range(1, len(pts_list)):
                alpha = self.cfg.trail_alpha * (i / len(pts_list))
                c = tuple(int(v * alpha) for v in color)
                cv2.line(overlay, pts_list[i - 1], pts_list[i], c, 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)
        return out

    # ------------------------------------------------------------------
    # ── ZONE OVERLAY ──────────────────────────────────────────────────

    def _draw_zones(self, out: np.ndarray, w: int, h: int) -> np.ndarray:
        overlay = out.copy()
        for name, (coords, rgba) in self.cfg.zones.items():
            x0, y0, x1, y1 = coords
            px1, py1 = int(x0 * w), int(y0 * h)
            px2, py2 = int(x1 * w), int(y1 * h)
            bgr = (rgba[2], rgba[1], rgba[0])
            alpha = rgba[3] / 255.0
            cv2.rectangle(overlay, (px1, py1), (px2, py2), bgr, -1)
            cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
            cv2.rectangle(out, (px1, py1), (px2, py2), bgr, 1)
            cv2.putText(out, name.upper(), (px1 + 6, py1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, bgr, 1, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------------
    # ── ALERT FLASH ───────────────────────────────────────────────────

    def _draw_flash(self, out: np.ndarray, w: int, h: int) -> np.ndarray:
        # Nhấp nháy 2 Hz
        if int(time.time() * 2) % 2 == 0:
            cv2.rectangle(out, (0, 0), (w - 1, h - 1), self.cfg.flash_color, 6)
            cv2.putText(out, "⚠ ANOMALY DETECTED",
                        (w // 2 - 150, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                        (255, 255, 255), 2, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------------
    # ── HUD (góc trên trái) ───────────────────────────────────────────

    def _draw_hud(
        self,
        out: np.ndarray,
        fps: float,
        frame_idx: int,
        n_tracks: int,
        n_anomaly: int,
        extra: str = "",
    ) -> np.ndarray:
        lines = [
            f"FPS    {fps:5.1f}",
            f"Frame  {frame_idx:6d}",
            f"Tracks {n_tracks:4d}",
        ]
        if n_anomaly > 0:
            lines.append(f"ANOM   {n_anomaly:4d}  !")
        if extra:
            lines.append(extra[:28])

        pad    = 8
        lh     = 20
        box_h  = pad * 2 + lh * len(lines)
        box_w  = 160

        # Nền mờ
        overlay = out.copy()
        cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, self.cfg.hud_alpha, out, 1 - self.cfg.hud_alpha, 0, out)
        cv2.rectangle(out, (8, 8), (8 + box_w, 8 + box_h), (100, 100, 100), 1)

        for i, line in enumerate(lines):
            color = (60, 60, 220) if "ANOM" in line and n_anomaly > 0 else (200, 220, 200)
            cv2.putText(out, line,
                        (8 + pad, 8 + pad + lh * (i + 1) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------------
    # ── MINI-MAP (góc dưới phải) ──────────────────────────────────────

    def _draw_minimap(
        self,
        out: np.ndarray,
        tracks: list,
        anomaly_map: Dict[int, dict],
        w: int,
        h: int,
    ) -> np.ndarray:
        mw, mh = self.cfg.minimap_size
        margin = 10
        mx = w - mw - margin
        my = h - mh - margin

        # Nền
        overlay = out.copy()
        cv2.rectangle(overlay, (mx, my), (mx + mw, my + mh), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)
        cv2.rectangle(out, (mx, my), (mx + mw, my + mh), (80, 80, 80), 1)
        cv2.putText(out, "MAP", (mx + 4, my + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

        # Chấm mỗi track
        for track in tracks:
            cx, cy = track.center
            # Chuyển toạ độ frame → minimap
            px = int(mx + (cx / w) * mw)
            py = int(my + (cy / h) * mh)
            info    = anomaly_map.get(track.track_id, {})
            score   = info.get("score", 0.0)
            is_anom = info.get("is_anomaly", False)
            color   = self._color_for_score(score)
            radius  = 5 if is_anom else 3
            cv2.circle(out, (px, py), radius, color, -1)
            if is_anom:
                cv2.circle(out, (px, py), radius + 3, color, 1)

        return out

    # ------------------------------------------------------------------
    # ── HELPER ────────────────────────────────────────────────────────

    def _color_for_score(self, score: float) -> Tuple:
        if score >= self.cfg.thresh_anomaly:
            return self.cfg.color_anomaly
        if score >= self.cfg.thresh_suspect:
            return self.cfg.color_suspect
        return self.cfg.color_normal

    def _get_track_color(self, tid: int) -> Tuple:
        if tid not in self._track_colors:
            self._track_colors[tid] = self._palette[
                self._palette_idx % len(self._palette)
            ]
            self._palette_idx += 1
        return self._track_colors[tid]

    # ------------------------------------------------------------------
    def reset_trails(self):
        """Xoá toàn bộ trail — gọi khi camera scene thay đổi."""
        self._trails.clear()

    def remove_track(self, track_id: int):
        """Xoá trail của 1 track khi nó bị mất."""
        self._trails.pop(track_id, None)
