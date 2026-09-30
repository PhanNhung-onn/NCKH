"""
modules/anomaly.py
Anomaly Scoring Model — Isolation Forest + Autoencoder ensemble.

Training produces a .pkl artifact consumed here at inference time.

v2: Thêm shelf_return_signal (index 24) và score adjustment theo hướng 1.
    Người đặt hàng lại → shelf_return_signal > 1.05 → score giảm 40%.
"""

from __future__ import annotations
import numpy as np
import logging
import pickle
from pathlib import Path
from collections import defaultdict, deque
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)

FEATURE_DIM     = 24   # số chiều gốc từ features.py
FEATURE_DIM_EXT = 26   # sau khi thêm shelf_return_signal + reach_events

# ── Ngưỡng shelf interaction ──────────────────────────────────────────
_SHELF_RETURN_THRESH  = 0.97   # bbox_h sau / trước >= 0.97 → đặt hàng lại
_SHELF_PICKUP_THRESH  = 0.88   # bbox_h sau / trước <  0.88 → lấy hàng
_PLACE_BACK_DISCOUNT  = 0.55   # nhân vào score khi phát hiện place_back
_PICKUP_PENALTY       = 1.15   # nhân vào score khi phát hiện pick_up
_MIN_REACH_EVENTS     = 1      # cần ít nhất N reach event để tin tưởng signal


class AnomalyScorer:
    """
    Ensemble scorer:
      • Isolation Forest  → fast, works out-of-the-box with no labels
      • Autoencoder       → captures temporal reconstruction error
    Final score = weighted average + shelf interaction adjustment.
    """

    def __init__(
        self,
        model_path: str   = "models/anomaly_model.pkl",
        threshold:  float = 0.65,
        if_weight:  float = 0.4,
        ae_weight:  float = 0.6,
    ):
        self.threshold = threshold
        self.if_weight = if_weight
        self.ae_weight = ae_weight
        self._if      = None
        self._ae      = None
        self._scaler  = None

        # Buffer bbox_h per track để tính shelf signal nội bộ
        # (dùng khi caller chỉ truyền feature_vec, không truyền bbox series)
        self._bbox_h_buf: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))

        self._load(model_path)

    # ──────────────────────────────────────────────────────────────────
    def _load(self, path: str):
        p = Path(path)
        if not p.exists():
            logger.warning(
                f"Model file {path} not found. "
                "Scoring sẽ dùng fallback chưa train. "
                "Chạy training_pipeline.py trước."
            )
            self._init_untrained()
            return

        with open(p, "rb") as f:
            bundle = pickle.load(f)

        self._if     = bundle.get("isolation_forest")
        self._scaler = bundle.get("scaler")

        ae_state = bundle.get("autoencoder_state")
        if ae_state is not None:
            self._ae = _build_autoencoder(FEATURE_DIM)
            import torch
            self._ae.load_state_dict(torch.load(ae_state, map_location="cpu"))
            self._ae.eval()

        logger.info(f"Anomaly models loaded from {path}")

    # ──────────────────────────────────────────────────────────────────
    def _init_untrained(self):
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        self._if     = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        self._scaler = StandardScaler()
        X_dummy = np.random.randn(200, FEATURE_DIM)
        self._scaler.fit(X_dummy)
        self._if.fit(X_dummy)
        logger.info("Dùng fallback models (chưa train).")

    # ──────────────────────────────────────────────────────────────────
    def score(self, feature_vec: np.ndarray) -> Tuple[float, bool]:
        """
        API gốc — không cần track_id.
        Shelf signal được tính từ feature_vec[11] (bbox_h, index 11).

        Input có thể là vector 1-D hoặc batch 2-D dạng (1, D).
        Chuẩn hóa về 1-D để hai dạng input cho cùng một kết quả.
        """
        feature_vec = np.asarray(feature_vec, dtype=np.float32).reshape(-1)

        # Chỉ dùng FEATURE_DIM feature đầu tiên; nếu thiếu thì pad 0.
        if feature_vec.size < FEATURE_DIM:
            x_features = np.pad(
                feature_vec,
                (0, FEATURE_DIM - feature_vec.size),
                mode="constant",
            )
        else:
            x_features = feature_vec[:FEATURE_DIM]

        raw_score = self._compute_base_score(x_features)

        # Luôn truyền vector 1-D đã chuẩn hóa.
        shelf_signal, reach_events = _compute_shelf_signal_from_feature(x_features)
        adjusted = self._apply_shelf_adjustment(
            raw_score, shelf_signal, reach_events
        )

        return adjusted, adjusted >= self.threshold

    # ──────────────────────────────────────────────────────────────────
    def score_with_history(
        self,
        track_id:    int,
        feature_vec: np.ndarray,
        bbox_h:      Optional[float] = None,
    ) -> Tuple[float, bool, str]:
        """
        API nâng cao — nhận track_id để duy trì lịch sử bbox_h riêng.
        Chính xác hơn score() vì dùng chuỗi bbox_h thực thay vì
        ước tính từ feature vector một frame.

        Returns: (score, is_anomaly, action_label)
        """
        # Cập nhật buffer bbox_h
        if bbox_h is not None:
            self._bbox_h_buf[track_id].append(bbox_h)

        raw_score = self._compute_base_score(feature_vec[:FEATURE_DIM])

        # Tính shelf signal từ buffer lịch sử
        shelf_signal, reach_events, action = self._shelf_signal_from_buf(track_id)
        adjusted = self._apply_shelf_adjustment(raw_score, shelf_signal, reach_events, action)

        return adjusted, adjusted >= self.threshold, action

    # ──────────────────────────────────────────────────────────────────
    def clear_track(self, track_id: int):
        """Xoá buffer khi track bị mất."""
        self._bbox_h_buf.pop(track_id, None)

    # ──────────────────────────────────────────────────────────────────
    # ── CORE SCORING ──────────────────────────────────────────────────

    def _compute_base_score(self, x24: np.ndarray) -> float:
        """Tính điểm gốc từ IF + AE, chưa áp dụng shelf adjustment."""
        x = x24.reshape(1, -1)

        if self._scaler is not None:
            x = self._scaler.transform(x)

        score = 0.0

        if self._if is not None:
            raw      = -float(self._if.decision_function(x)[0])
            if_score = self._sigmoid(raw)
            score   += self.if_weight * if_score
        else:
            score += self.if_weight * 0.5

        if self._ae is not None:
            import torch
            with torch.no_grad():
                t     = torch.tensor(x, dtype=torch.float32)
                recon = self._ae(t)
                mse   = float(torch.nn.functional.mse_loss(recon, t).item())
            ae_score = self._sigmoid(mse * 10)
            score   += self.ae_weight * ae_score
        else:
            score = score / (self.if_weight if self._if else 1.0)

        return float(np.clip(score, 0.0, 1.0))

    # ──────────────────────────────────────────────────────────────────
    # ── SHELF INTERACTION ─────────────────────────────────────────────

    def _shelf_signal_from_buf(
        self, track_id: int
    ) -> Tuple[float, int, str]:
        """
        Tính shelf_signal từ buffer bbox_h của track.

        Returns:
            shelf_signal  — tỷ lệ bbox_h[-1] / bbox_h[0]  (> 1 = đặt lại)
            reach_events  — số lần phát hiện tương tác với kệ
            action_label  — "place_back" | "pick_up" | "examining" | "walking"
        """
        buf = self._bbox_h_buf.get(track_id)
        if buf is None or len(buf) < 5:
            return 1.0, 0, "walking"

        heights = np.array(buf, dtype=np.float32)
        return _analyse_height_series(heights)

    # ──────────────────────────────────────────────────────────────────
    def _apply_shelf_adjustment(
        self,
        base_score:   float,
        shelf_signal: float,
        reach_events: int,
        action:       str = "",
    ) -> float:
        """
        Điều chỉnh score dựa trên shelf interaction signal.

        Logic (ưu tiên action label nếu có):
          • reach_events < 1      → chưa có tương tác kệ → giữ nguyên
          • action == place_back  → đặt hàng lại         → giảm 45%
          • action == pick_up     → lấy hàng             → tăng 15%
          • còn lại               → không đổi
        """
        if reach_events < _MIN_REACH_EVENTS:
            return base_score

        # Dùng action label (chính xác hơn shelf_signal đơn thuần)
        effective_action = action if action else (
            "place_back" if shelf_signal >= _SHELF_RETURN_THRESH else
            "pick_up"    if shelf_signal <  _SHELF_PICKUP_THRESH else
            "neutral"
        )

        if effective_action == "place_back":
            adjusted = base_score * _PLACE_BACK_DISCOUNT
            logger.debug(f"PLACE-BACK  {base_score:.3f} → {adjusted:.3f}")
        elif effective_action == "pick_up":
            adjusted = min(base_score * _PICKUP_PENALTY, 1.0)
            logger.debug(f"PICK-UP     {base_score:.3f} → {adjusted:.3f}")
        else:
            adjusted = base_score

        return float(np.clip(adjusted, 0.0, 1.0))

    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))


# ══════════════════════════════════════════════════════════════════════
# HÀM PHÂN TÍCH HEIGHT SERIES — dùng cho cả 2 API
# ══════════════════════════════════════════════════════════════════════

def _analyse_height_series(
    heights: np.ndarray,
) -> Tuple[float, int, str]:
    """
    Phân tích chuỗi bbox_h để phát hiện tương tác kệ hàng.

    Nguyên lý:
      - Người vươn tay/cúi xuống → bbox_h GIẢM đột ngột (dip)
      - Nếu sau dip, bbox_h hồi phục về gần giá trị ban đầu
        → người đặt hàng lại (shelf_signal > 1.0)
      - Nếu sau dip, bbox_h KHÔNG hồi phục (vẫn nhỏ hơn)
        → người cầm hàng, bbox nhỏ hơn (shelf_signal < 1.0)

    Returns: (shelf_signal, reach_events, action_label)
    """
    if len(heights) < 5:
        return 1.0, 0, "walking"

    # Smooth nhẹ để lọc noise
    kernel = np.ones(3) / 3
    smooth = np.convolve(heights, kernel, mode="same")

    diffs  = np.diff(smooth)

    # Reach kiểu V (dip + hồi phục) → place_back
    reach_v = int(np.sum((diffs[:-1] < -0.008) & (diffs[1:] > 0.005)))

    # Reach kiểu L (dip + giữ thấp) → pick_up
    mid_idx  = len(smooth) // 2
    min_pre  = float(np.min(smooth[:mid_idx]))
    min_post = float(np.min(smooth[mid_idx:]))
    reach_l  = 1 if (min_pre - min_post) > 0.04 else 0

    reach_events = reach_v + reach_l

    # Baseline = max của nửa đầu (đỉnh cao nhất trước khi tương tác)
    # Dùng max thay vì mean để tránh bị smooth kéo xuống
    mid      = max(4, len(smooth) // 2)
    pre_h    = float(np.max(smooth[:mid]))
    # Post = max của 4 frame cuối (sau tương tác)
    post_h   = float(np.max(smooth[-4:]))
    shelf_signal = post_h / (pre_h + 1e-6)

    # Action label
    if reach_events == 0:
        speed_proxy = float(np.std(smooth[:, np.newaxis] if smooth.ndim > 1 else smooth))
        action = "walking" if speed_proxy > 0.03 else "examining"
    elif shelf_signal > _SHELF_RETURN_THRESH:
        action = "place_back"
    elif shelf_signal < _SHELF_PICKUP_THRESH:
        action = "pick_up"
    else:
        action = "examining"

    return shelf_signal, reach_events, action


def _compute_shelf_signal_from_feature(
    feature_vec: np.ndarray,
) -> Tuple[float, int]:
    """
    Ước tính shelf signal từ feature vector 1 frame.
    Kém chính xác hơn _shelf_signal_from_buf nhưng không cần track_id.

    Dùng:
      index 11 = bbox_h hiện tại
      index 22 = jerk_ratio (cao → bbox thay đổi đột ngột → reach event)
    """
    if len(feature_vec) < 23:
        return 1.0, 0

    bbox_h     = float(feature_vec[11])
    jerk_ratio = float(feature_vec[22])
    stop_frac  = float(feature_vec[20]) / 30.0   # stop_frames / window

    # Ước tính reach_events từ jerk_ratio
    reach_events = 1 if jerk_ratio > 0.3 else 0

    # Ước tính shelf_signal: nếu jerk cao + stop nhiều → có tương tác
    # Không đủ thông tin để phân biệt pick_up / place_back từ 1 frame
    # → trả về neutral
    shelf_signal = 1.0

    return shelf_signal, reach_events


# ══════════════════════════════════════════════════════════════════════
# AUTOENCODER
# ══════════════════════════════════════════════════════════════════════

def _build_autoencoder(input_dim: int = 24):
    """Lightweight MLP autoencoder for anomaly detection."""
    import torch
    import torch.nn as nn

    class Autoencoder(nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(d, 16), nn.ReLU(),
                nn.Linear(16, 8), nn.ReLU(),
                nn.Linear(8, 4),
            )
            self.dec = nn.Sequential(
                nn.Linear(4, 8),  nn.ReLU(),
                nn.Linear(8, 16), nn.ReLU(),
                nn.Linear(16, d),
            )

        def forward(self, x):
            return self.dec(self.enc(x))

    return Autoencoder(input_dim)