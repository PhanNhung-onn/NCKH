"""
modules/tracker.py
ByteTrack multi-object tracker integrated with SPARTA.

Dependencies:
    pip install bytetracker          # or use ultralytics built-in BYTETracker
    pip install sparta-track         # SPARTA re-ID / sparse attention tracker
"""

from __future__ import annotations
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from .detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Active track produced by the tracker."""
    track_id: int
    tlbr: np.ndarray          # [x1, y1, x2, y2]
    score: float
    class_name: str = "person"
    age: int = 0              # frames since first seen
    hits: int = 0             # total matched frames
    state: str = "active"     # active | lost | removed

    @property
    def tlwh(self):
        x1, y1, x2, y2 = self.tlbr
        return np.array([x1, y1, x2 - x1, y2 - y1])

    @property
    def center(self):
        x1, y1, x2, y2 = self.tlbr
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self):
        x1, y1, x2, y2 = self.tlbr
        return (x2 - x1) * (y2 - y1)


# ──────────────────────────────────────────────────────────────────────
class ByteTrackWrapper:
    """
    Thin wrapper that exposes a unified update() API for ByteTrack.

    Falls back to a simple IoU-based tracker if ByteTrack is not installed
    (useful for quick dev / testing without GPU).
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        min_box_area: float = 100.0,
        frame_rate: int = 15,
        use_sparta: bool = True,
    ):
        self.min_box_area = min_box_area
        self.use_sparta = use_sparta
        self._tracks: List[Track] = []
        self._next_id = 1

        self._bt = self._init_bytetrack(
            track_thresh, track_buffer, match_thresh, frame_rate
        )
        self._sparta = self._init_sparta() if use_sparta else None

    # ------------------------------------------------------------------
    def _init_bytetrack(self, track_thresh, track_buffer, match_thresh, fps):
        try:
            from bytetracker import BYTETracker

            class _Args:
                pass

            args = _Args()
            args.track_thresh = track_thresh
            args.track_buffer = track_buffer
            args.match_thresh = match_thresh
            args.mot20 = False

            tracker = BYTETracker(args, frame_rate=fps)
            logger.info("ByteTrack initialised.")
            return tracker
        except ImportError:
            logger.warning("bytetracker not installed – using fallback IoU tracker.")
            return None

    # ------------------------------------------------------------------
    def _init_sparta(self):
        """
        SPARTA (Sparse Probabilistic Re-ID Attention) optional integration.
        Provides appearance-based re-identification to recover tracks after
        occlusion — critical in crowded retail environments.
        """
        try:
            from sparta import SPARTAReID
            reid = SPARTAReID(feature_dim=256, device="cpu")
            logger.info("SPARTA ReID initialised.")
            return reid
        except ImportError:
            logger.warning("sparta-track not installed – ReID disabled.")
            return None

    # ------------------------------------------------------------------
    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        """
        Feed new detections and return active tracks.
        """
        if not detections:
            return self._tracks

        # Build numpy array [x1,y1,x2,y2,score] expected by ByteTrack
        det_array = np.array([
            [d.x1, d.y1, d.x2, d.y2, d.confidence]
            for d in detections
            if d.area >= self.min_box_area
        ], dtype=np.float32)

        if det_array.ndim == 1:
            det_array = det_array.reshape(-1, 5)

        if self._bt is not None:
            raw_tracks = self._bt.update(det_array, [frame.shape[0], frame.shape[1]], [frame.shape[0], frame.shape[1]])
            self._tracks = [
                Track(
                    track_id=int(t.track_id),
                    tlbr=t.tlbr,
                    score=float(t.score),
                    age=t.frame_id,
                    hits=t.tracklet_len,
                )
                for t in raw_tracks
            ]
        else:
            # Fallback: very simple IoU tracker
            self._tracks = self._iou_tracker(det_array, detections)

        # SPARTA re-ID pass (refine IDs after occlusion)
        if self._sparta and len(self._tracks) > 0:
            self._tracks = self._apply_sparta(frame, self._tracks)

        return self._tracks

    # ------------------------------------------------------------------
    def _apply_sparta(self, frame: np.ndarray, tracks: List[Track]) -> List[Track]:
        try:
            crops = []
            for t in tracks:
                x1, y1, x2, y2 = [int(v) for v in t.tlbr]
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size > 0:
                    crops.append(crop)
                else:
                    crops.append(np.zeros((64, 32, 3), dtype=np.uint8))

            refined_ids = self._sparta.match(crops, [t.track_id for t in tracks])
            for t, rid in zip(tracks, refined_ids):
                t.track_id = rid
        except Exception as e:
            logger.debug(f"SPARTA pass failed: {e}")
        return tracks

    # ------------------------------------------------------------------
    def _iou_tracker(self, det_array: np.ndarray, detections: List[Detection]) -> List[Track]:
        """Minimal IoU-based fallback tracker."""
        import scipy.optimize

        if not self._tracks:
            tracks = []
            for i, d in enumerate(detections):
                tracks.append(Track(
                    track_id=self._next_id,
                    tlbr=np.array([d.x1, d.y1, d.x2, d.y2]),
                    score=d.confidence,
                    class_name=d.class_name,
                ))
                self._next_id += 1
            return tracks

        def iou(b1, b2):
            ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
            ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
            a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
            return inter / (a1 + a2 - inter + 1e-6)

        cost = np.zeros((len(self._tracks), len(detections)))
        for i, t in enumerate(self._tracks):
            for j, d in enumerate(detections):
                cost[i, j] = 1 - iou(t.tlbr, [d.x1, d.y1, d.x2, d.y2])

        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)
        matched_t, matched_d = set(), set()
        new_tracks = []

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 0.7:
                t = self._tracks[r]
                d = detections[c]
                t.tlbr = np.array([d.x1, d.y1, d.x2, d.y2])
                t.score = d.confidence
                t.hits += 1
                new_tracks.append(t)
                matched_t.add(r); matched_d.add(c)

        for j, d in enumerate(detections):
            if j not in matched_d:
                new_tracks.append(Track(
                    track_id=self._next_id,
                    tlbr=np.array([d.x1, d.y1, d.x2, d.y2]),
                    score=d.confidence,
                    class_name=d.class_name,
                ))
                self._next_id += 1

        return new_tracks
