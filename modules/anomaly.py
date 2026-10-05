"""
modules/anomaly.py
==================
Anomaly Scoring Model — 2 scorer song song:

  1. AnomalyScorer   — Isolation Forest + Autoencoder trên 24-D kinematic features
                       (giữ nguyên từ v2, thêm shelf_return_signal adjustment)

  2. ShopformerScorer — GCAE pose tokenizer + Transformer reconstruction error
                        (mới, dành riêng cho phát hiện hành vi giấu hàng)

Final score = max(kinematic_score, pose_score) hoặc weighted average tuỳ config.

Training:
  - AnomalyScorer:     chạy training_pipeline.py (Stage 1+2 cũ)
  - ShopformerScorer:  chạy training_pipeline.py --stage pose (Stage 3+4 mới)
"""

from __future__ import annotations

import logging
import pickle
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FEATURE_DIM     = 24   # kinematic features từ features.py
FEATURE_DIM_EXT = 26

# ── Ngưỡng shelf interaction ──────────────────────────────────────────
_SHELF_RETURN_THRESH = 0.97
_SHELF_PICKUP_THRESH = 0.88
_PLACE_BACK_DISCOUNT = 0.55
_PICKUP_PENALTY      = 1.15
_MIN_REACH_EVENTS    = 1


# ══════════════════════════════════════════════════════════════════════
# ANOMALY SCORER (kinematic — giữ nguyên từ v2)
# ══════════════════════════════════════════════════════════════════════

class AnomalyScorer:
    """
    Ensemble scorer dựa trên kinematic features (24-D):
      • Isolation Forest  → fast, unsupervised
      • Autoencoder MLP   → temporal reconstruction error
    Kết hợp với shelf interaction adjustment.
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
        self._if       = None
        self._ae       = None
        self._scaler   = None
        self._bbox_h_buf: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
        self._load(model_path)

    # ------------------------------------------------------------------
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

        logger.info(f"AnomalyScorer loaded from {path}")

    # ------------------------------------------------------------------
    def _init_untrained(self):
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        self._if     = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        self._scaler = StandardScaler()
        X_dummy = np.random.randn(200, FEATURE_DIM)
        self._scaler.fit(X_dummy)
        self._if.fit(X_dummy)
        logger.info("AnomalyScorer: dùng fallback models (chưa train).")

    # ------------------------------------------------------------------
    def score(self, feature_vec: np.ndarray) -> Tuple[float, bool]:
        """API gốc — không cần track_id."""
        feature_vec = np.asarray(feature_vec, dtype=np.float32).reshape(-1)
        if feature_vec.size < FEATURE_DIM:
            x_features = np.pad(feature_vec, (0, FEATURE_DIM - feature_vec.size))
        else:
            x_features = feature_vec[:FEATURE_DIM]

        raw_score = self._compute_base_score(x_features)
        shelf_signal, reach_events = _compute_shelf_signal_from_feature(x_features)
        adjusted = self._apply_shelf_adjustment(raw_score, shelf_signal, reach_events)
        return adjusted, adjusted >= self.threshold

    # ------------------------------------------------------------------
    def score_with_history(
        self,
        track_id:    int,
        feature_vec: np.ndarray,
        bbox_h:      Optional[float] = None,
    ) -> Tuple[float, bool, str]:
        """API nâng cao — dùng lịch sử bbox_h để nhận diện shelf action."""
        if bbox_h is not None:
            self._bbox_h_buf[track_id].append(bbox_h)

        raw_score = self._compute_base_score(feature_vec[:FEATURE_DIM])
        shelf_signal, reach_events, action = self._shelf_signal_from_buf(track_id)
        adjusted = self._apply_shelf_adjustment(raw_score, shelf_signal, reach_events, action)
        return adjusted, adjusted >= self.threshold, action

    # ------------------------------------------------------------------
    def clear_track(self, track_id: int):
        self._bbox_h_buf.pop(track_id, None)

    # ------------------------------------------------------------------
    def _compute_base_score(self, x24: np.ndarray) -> float:
        x = x24.reshape(1, -1)
        if self._scaler is not None:
            x = self._scaler.transform(x)

        score = 0.0
        if self._if is not None:
            raw      = -float(self._if.decision_function(x)[0])
            if_score = _sigmoid(raw)
            score   += self.if_weight * if_score
        else:
            score += self.if_weight * 0.5

        if self._ae is not None:
            import torch
            with torch.no_grad():
                t     = torch.tensor(x, dtype=torch.float32)
                recon = self._ae(t)
                mse   = float(torch.nn.functional.mse_loss(recon, t).item())
            ae_score = _sigmoid(mse * 10)
            score   += self.ae_weight * ae_score
        else:
            score = score / (self.if_weight if self._if else 1.0)

        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    def _shelf_signal_from_buf(self, track_id: int) -> Tuple[float, int, str]:
        buf = self._bbox_h_buf.get(track_id)
        if buf is None or len(buf) < 5:
            return 1.0, 0, "walking"
        return _analyse_height_series(np.array(buf, dtype=np.float32))

    # ------------------------------------------------------------------
    def _apply_shelf_adjustment(
        self,
        base_score:   float,
        shelf_signal: float,
        reach_events: int,
        action:       str = "",
    ) -> float:
        if reach_events < _MIN_REACH_EVENTS:
            return base_score

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


# ══════════════════════════════════════════════════════════════════════
# SHOPFORMER SCORER (pose-based — MỚI)
# ══════════════════════════════════════════════════════════════════════

class ShopformerScorer:
    """
    Scorer dựa trên pose tokens từ PoseTokenizer.

    Dùng reconstruction error của Shopformer Transformer làm anomaly score.
    Train trên normal pose sequences → MSE thấp.
    Shoplifting sequences → MSE cao → score cao.

    Được dùng SONG SONG với AnomalyScorer.
    Final score = max(kinematic_score, pose_score) hoặc weighted.
    """

    def __init__(
        self,
        tokenizer_path: str   = "models/pose_tokenizer.pt",
        threshold:      float = 0.60,   # thấp hơn AnomalyScorer vì pose cụ thể hơn
        device:         str   = "cpu",
    ):
        self.threshold = threshold
        self.device    = device
        self._tokenizer = None
        self._load(tokenizer_path)

    # ------------------------------------------------------------------
    def _load(self, path: str):
        try:
            from modules.pose_tokenizer import PoseTokenizer
            self._tokenizer = PoseTokenizer(device=self.device)
            self._tokenizer.load(path)
            logger.info(f"ShopformerScorer loaded from {path}")
        except ImportError:
            logger.warning("ShopformerScorer: pose_tokenizer.py không tìm thấy.")
        except Exception as e:
            logger.warning(f"ShopformerScorer: {e}")

    # ------------------------------------------------------------------
    def update_and_score(
        self,
        track_id: int,
        kpts:     np.ndarray,   # (17, 3) từ PoseExtractor
    ) -> Tuple[Optional[float], Optional[bool]]:
        """
        Nhận keypoints mỗi frame, trả về score khi đủ sequence.

        Returns:
            (score, is_anomaly) hoặc (None, None) nếu chưa đủ frames.
        """
        if self._tokenizer is None:
            return None, None

        tokens = self._tokenizer.update(track_id, kpts)
        if tokens is None:
            return None, None

        score = self._tokenizer.anomaly_score(tokens, track_id=track_id)
        return score, score >= self.threshold

    # ------------------------------------------------------------------
    def clear_track(self, track_id: int):
        if self._tokenizer:
            self._tokenizer.clear_track(track_id)

    # ------------------------------------------------------------------
    @property
    def is_available(self) -> bool:
        return self._tokenizer is not None


# ══════════════════════════════════════════════════════════════════════
# COMBINED SCORER — dùng cả 2 scorer
# ══════════════════════════════════════════════════════════════════════

class CombinedScorer:
    """
    Kết hợp AnomalyScorer (kinematic) và ShopformerScorer (pose).

    Chiến lược:
        - Nếu pose score có sẵn: final = max(kinematic, pose)
        - Nếu không có pose:     final = kinematic (fallback)

    Dùng max thay vì average để bắt được cả 2 loại bất thường:
        - Trajectory bất thường (kinematic cao)
        - Hành vi giấu hàng (pose cao)
    """

    def __init__(
        self,
        kinematic_model_path:  str   = "models/anomaly_model.pkl",
        pose_tokenizer_path:   str   = "models/pose_tokenizer.pt",
        kinematic_threshold:   float = 0.65,
        pose_threshold:        float = 0.60,
        kinematic_weight:      float = 0.45,
        pose_weight:           float = 0.55,   # pose quan trọng hơn vì cụ thể hơn
        device:                str   = "cpu",
    ):
        self.kinematic_weight = kinematic_weight
        self.pose_weight      = pose_weight

        self.kinematic = AnomalyScorer(
            model_path=kinematic_model_path,
            threshold=kinematic_threshold,
        )
        self.pose = ShopformerScorer(
            tokenizer_path=pose_tokenizer_path,
            threshold=pose_threshold,
            device=device,
        )

    # ------------------------------------------------------------------
    def score(
        self,
        track_id:    int,
        feature_vec: np.ndarray,    # 24-D kinematic
        kpts:        Optional[np.ndarray] = None,  # (17,3) pose
        bbox_h:      Optional[float]      = None,
    ) -> Tuple[float, bool, dict]:
        """
        Tính score kết hợp.

        Returns:
            (final_score, is_anomaly, detail_dict)
            detail_dict có 'kinematic_score', 'pose_score', 'action'
        """
        # ── Kinematic score ───────────────────────────────────────
        k_score, _, action = self.kinematic.score_with_history(
            track_id, feature_vec, bbox_h
        )

        # ── Pose score ────────────────────────────────────────────
        p_score = None
        if kpts is not None and self.pose.is_available:
            p_score, _ = self.pose.update_and_score(track_id, kpts)

        # ── Combine ───────────────────────────────────────────────
        if p_score is not None:
            # Weighted max: ưu tiên score cao nhất có trọng số
            final = (
                self.kinematic_weight * k_score
                + self.pose_weight    * p_score
            )
            # Nếu 1 trong 2 rất cao → giữ nguyên giá trị cao đó
            final = max(final, k_score * 0.7, p_score * 0.8)
            final = float(np.clip(final, 0.0, 1.0))
        else:
            final = k_score

        # Dùng ngưỡng của kinematic (conservative)
        is_anom = final >= self.kinematic.threshold

        detail = {
            "kinematic_score": round(k_score, 4),
            "pose_score":      round(p_score, 4) if p_score is not None else None,
            "action":          action,
            "pose_available":  p_score is not None,
        }
        return final, is_anom, detail

    # ------------------------------------------------------------------
    def clear_track(self, track_id: int):
        self.kinematic.clear_track(track_id)
        self.pose.clear_track(track_id)


# ══════════════════════════════════════════════════════════════════════
# HÀM TIỆN ÍCH
# ══════════════════════════════════════════════════════════════════════

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _analyse_height_series(heights: np.ndarray) -> Tuple[float, int, str]:
    """Phân tích chuỗi bbox_h để phát hiện tương tác kệ hàng."""
    if len(heights) < 5:
        return 1.0, 0, "walking"

    kernel = np.ones(3) / 3
    smooth = np.convolve(heights, kernel, mode="same")
    diffs  = np.diff(smooth)

    reach_v = int(np.sum((diffs[:-1] < -0.008) & (diffs[1:] > 0.005)))

    mid_idx  = len(smooth) // 2
    min_pre  = float(np.min(smooth[:mid_idx]))
    min_post = float(np.min(smooth[mid_idx:]))
    reach_l  = 1 if (min_pre - min_post) > 0.04 else 0
    reach_events = reach_v + reach_l

    mid    = max(4, len(smooth) // 2)
    pre_h  = float(np.max(smooth[:mid]))
    post_h = float(np.max(smooth[-4:]))
    shelf_signal = post_h / (pre_h + 1e-6)

    if reach_events == 0:
        speed_proxy = float(np.std(smooth))
        action = "walking" if speed_proxy > 0.03 else "examining"
    elif shelf_signal >= _SHELF_RETURN_THRESH:
        action = "place_back"
    elif shelf_signal < _SHELF_PICKUP_THRESH:
        action = "pick_up"
    else:
        action = "examining"

    return shelf_signal, reach_events, action


def _compute_shelf_signal_from_feature(feature_vec: np.ndarray) -> Tuple[float, int]:
    if len(feature_vec) < 23:
        return 1.0, 0
    jerk_ratio   = float(feature_vec[22])
    reach_events = 1 if jerk_ratio > 0.3 else 0
    return 1.0, reach_events


def _build_autoencoder(input_dim: int = 24):
    """MLP Autoencoder cho kinematic features."""
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
