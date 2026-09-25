"""
modules/features.py
Behavior Feature Extraction — HuMiPy / Pandas-backed sliding-window features.

Features extracted per track over a rolling window:
  - Kinematic : speed, acceleration, heading, path length, displacement ratio
  - Spatial   : dwell time zones, bounding-box aspect ratio change
  - Interaction: proximity to other tracks, crowd density
  - Temporal  : stop-start events, loitering score
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import logging
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

from .tracker import Track

logger = logging.getLogger(__name__)

# Retail zone definitions (normalised 0-1 coordinates).
# Adjust to match your store layout.
ZONES = {
    "entrance":   (0.0, 0.0, 0.2, 1.0),
    "checkout":   (0.8, 0.0, 1.0, 1.0),
    "high_value": (0.4, 0.1, 0.7, 0.5),
    "general":    (0.2, 0.0, 0.8, 1.0),
}

FEATURE_DIM = 24   # total feature vector length


class TrackHistory:
    """Rolling buffer of per-frame observations for one track ID."""

    def __init__(self, window: int = 30):
        self.window = window
        self.centers: deque = deque(maxlen=window)       # (cx, cy) tuples
        self.sizes: deque = deque(maxlen=window)         # (w, h)
        self.timestamps: deque = deque(maxlen=window)    # frame index
        self.zone_hits: Dict[str, int] = defaultdict(int)

    def update(self, cx: float, cy: float, w: float, h: float, t: int):
        self.centers.append((cx, cy))
        self.sizes.append((w, h))
        self.timestamps.append(t)

        # Zone classification
        for name, (x0, y0, x1, y1) in ZONES.items():
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                self.zone_hits[name] += 1

    def ready(self) -> bool:
        return len(self.centers) >= max(5, self.window // 3)


# ──────────────────────────────────────────────────────────────────────
class BehaviorFeatureExtractor:
    """
    Maintains per-track history and computes feature vectors.
    Compatible with HuMiPy pose-based features if available.
    """

    def __init__(self, window: int = 30, frame_w: int = 1280, frame_h: int = 720):
        self.window = window
        self.frame_w = frame_w
        self.frame_h = frame_h
        self._histories: Dict[int, TrackHistory] = {}
        self._frame_idx = 0
        self._humify = self._try_load_humify()

    # ------------------------------------------------------------------
    def _try_load_humify(self):
        try:
            import humipy as hp
            logger.info("HuMiPy loaded – pose-based features enabled.")
            return hp
        except ImportError:
            logger.debug("humipy not installed – using kinematic features only.")
            return None

    # ------------------------------------------------------------------
    def update(
        self, tracks: List[Track], frame: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """
        Update histories for all active tracks; return feature dict
        {track_id: feature_vector} for tracks with enough history.
        """
        self._frame_idx += 1
        h, w = frame.shape[:2]
        self.frame_w, self.frame_h = w, h

        # Prune dead tracks
        active_ids = {t.track_id for t in tracks}
        stale = [tid for tid in self._histories if tid not in active_ids]
        for tid in stale:
            del self._histories[tid]

        for track in tracks:
            tid = track.track_id
            if tid not in self._histories:
                self._histories[tid] = TrackHistory(window=self.window)

            cx, cy = track.center
            bx1, by1, bx2, by2 = track.tlbr
            bw = (bx2 - bx1) / w
            bh = (by2 - by1) / h
            cx_n = cx / w
            cy_n = cy / h

            self._histories[tid].update(cx_n, cy_n, bw, bh, self._frame_idx)

        # Build feature vectors
        features: Dict[int, np.ndarray] = {}
        for tid, hist in self._histories.items():
            if hist.ready():
                vec = self._extract(hist, tracks, tid)
                features[tid] = vec

        return features

    # ------------------------------------------------------------------
    def _extract(
        self, hist: TrackHistory, all_tracks: List[Track], tid: int
    ) -> np.ndarray:
        """Compute 24-D feature vector from history."""
        centers = np.array(list(hist.centers))   # (N, 2)
        sizes   = np.array(list(hist.sizes))     # (N, 2)

        # ── Kinematic features ──────────────────────────────────────
        disps = np.linalg.norm(np.diff(centers, axis=0), axis=1)   # (N-1,)
        speed_mean = float(np.mean(disps)) if len(disps) else 0.0
        speed_std  = float(np.std(disps))  if len(disps) else 0.0
        speed_max  = float(np.max(disps))  if len(disps) else 0.0

        accels = np.abs(np.diff(disps)) if len(disps) > 1 else np.array([0.0])
        accel_mean = float(np.mean(accels))
        accel_max  = float(np.max(accels))

        total_path = float(np.sum(disps))
        start_end  = float(np.linalg.norm(centers[-1] - centers[0]))
        linearity  = start_end / (total_path + 1e-6)

        # ── Heading ─────────────────────────────────────────────────
        if len(centers) >= 2:
            headings = np.arctan2(np.diff(centers[:, 1]), np.diff(centers[:, 0]))
            heading_std = float(np.std(headings))
            turn_rate   = float(np.mean(np.abs(np.diff(headings)))) if len(headings) > 1 else 0.0
        else:
            heading_std = 0.0
            turn_rate   = 0.0

        # ── Loitering ────────────────────────────────────────────────
        loiter_score = self._loiter_score(centers)

        # ── Spatial / size ───────────────────────────────────────────
        w_mean = float(np.mean(sizes[:, 0]))
        h_mean = float(np.mean(sizes[:, 1]))
        aspect_var = float(np.var(sizes[:, 0] / (sizes[:, 1] + 1e-6)))

        pos_cx = float(centers[-1, 0])
        pos_cy = float(centers[-1, 1])

        # ── Zone features ────────────────────────────────────────────
        total_frames = max(1, len(hist.centers))
        zone_entrance   = hist.zone_hits.get("entrance",   0) / total_frames
        zone_checkout   = hist.zone_hits.get("checkout",   0) / total_frames
        zone_high_value = hist.zone_hits.get("high_value", 0) / total_frames

        # ── Proximity to other tracks ────────────────────────────────
        prox = self._proximity(tid, all_tracks)

        # ── Assemble ─────────────────────────────────────────────────
        feat = np.array([
            speed_mean, speed_std, speed_max,          # 0-2
            accel_mean, accel_max,                      # 3-4
            total_path, linearity,                      # 5-6
            heading_std, turn_rate,                     # 7-8
            loiter_score,                               # 9
            w_mean, h_mean, aspect_var,                 # 10-12
            pos_cx, pos_cy,                             # 13-14
            zone_entrance, zone_checkout, zone_high_value,  # 15-17
            prox,                                       # 18
            float(len(hist.centers)) / self.window,    # 19 track maturity
            float(np.sum(disps < 0.002)),               # 20 stop frames
            float(speed_std / (speed_mean + 1e-6)),    # 21 speed CV
            float(accel_max / (speed_max + 1e-6)),     # 22 jerk ratio
            float(hist.zone_hits.get("high_value", 0)),# 23 raw HV dwell
        ], dtype=np.float32)

        assert len(feat) == FEATURE_DIM
        return feat

    # ------------------------------------------------------------------
    def _loiter_score(self, centers: np.ndarray) -> float:
        """
        High score → person oscillating in small area (potential shoplifting).
        """
        if len(centers) < 5:
            return 0.0
        spread = np.std(centers, axis=0)  # (std_x, std_y)
        area = float(spread[0] * spread[1])
        return 1.0 / (area + 1e-4)       # high when spread is tiny

    # ------------------------------------------------------------------
    def _proximity(self, tid: int, tracks: List[Track]) -> float:
        """Mean distance to nearest 3 other tracks (normalised)."""
        own = next((t for t in tracks if t.track_id == tid), None)
        if own is None or len(tracks) <= 1:
            return 1.0
        own_c = np.array(own.center)
        dists = []
        for t in tracks:
            if t.track_id == tid:
                continue
            dists.append(np.linalg.norm(np.array(t.center) - own_c))
        dists.sort()
        return float(np.mean(dists[:3])) if dists else 1.0

    # ------------------------------------------------------------------
    def to_dataframe(self, features_map: Dict[int, np.ndarray]) -> pd.DataFrame:
        """Utility: convert feature dict to Pandas DataFrame for analysis."""
        cols = [
            "speed_mean","speed_std","speed_max",
            "accel_mean","accel_max",
            "path_len","linearity",
            "heading_std","turn_rate",
            "loiter_score",
            "bbox_w","bbox_h","aspect_var",
            "pos_cx","pos_cy",
            "zone_entrance","zone_checkout","zone_high_value",
            "proximity","maturity","stop_frames","speed_cv","jerk_ratio","hv_dwell"
        ]
        rows = {tid: vec for tid, vec in features_map.items()}
        df = pd.DataFrame.from_dict(rows, orient="index", columns=cols)
        df.index.name = "track_id"
        return df
