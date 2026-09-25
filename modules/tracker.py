"""
modules/tracker.py

ByteTrack multi-object tracker integrated with SPARTA.

ByteTrack implementation:
    Ultralytics built-in BYTETracker

Installation:
    python -m pip install -U ultralytics

No need for:
    bytetracker
    lap==0.4.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np

from .detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Active track produced by ByteTrack."""

    track_id: int
    tlbr: np.ndarray          # [x1, y1, x2, y2]
    score: float
    class_name: str = "person"
    age: int = 0              # frames since first seen
    hits: int = 0              # total matched frames
    state: str = "active"      # active | lost | removed

    @property
    def tlwh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.tlbr
        return np.array(
            [x1, y1, x2 - x1, y2 - y1],
            dtype=np.float32,
        )

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.tlbr
        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        )

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.tlbr
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


# ──────────────────────────────────────────────────────────────────────


class ByteTrackWrapper:
    """
    Unified ByteTrack wrapper.

    Uses Ultralytics' built-in BYTETracker.

    The rest of the project only sees:

        detections -> update() -> List[Track]

    Therefore the detector / SPARTA / behavior-analysis modules
    do not need to know which ByteTrack implementation is used.
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
        self.frame_rate = frame_rate
        self.use_sparta = use_sparta

        self._tracks: List[Track] = []

        # Initialize Ultralytics ByteTrack.
        self._bt = self._init_bytetrack(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
        )

        # Optional SPARTA Re-ID.
        self._sparta = (
            self._init_sparta()
            if use_sparta
            else None
        )

    # ------------------------------------------------------------------
    def _init_bytetrack(
        self,
        track_thresh: float,
        track_buffer: int,
        match_thresh: float,
    ):
        """
        Initialize Ultralytics BYTETracker.

        This avoids the old `bytetracker` package and therefore
        avoids the problematic `lap==0.4.0` dependency.
        """

        try:
            import torch

            from ultralytics.engine.results import Boxes
            from ultralytics.trackers.byte_tracker import BYTETracker
            from ultralytics.utils import (
                IterableSimpleNamespace,
                YAML,
            )
            from ultralytics.utils import ROOT

            # Load Ultralytics' official ByteTrack configuration.
            yaml_path = ROOT / "cfg" / "trackers" / "bytetrack.yaml"
            config = YAML.load(yaml_path)

            # Override the parameters requested by our project.
            config["track_high_thresh"] = track_thresh
            config["track_buffer"] = track_buffer
            config["match_thresh"] = match_thresh

            # Build the namespace expected by BYTETracker.
            args = IterableSimpleNamespace(**config)

            tracker = BYTETracker(args)

            # Store classes used by update().
            self._torch = torch
            self._Boxes = Boxes

            logger.info(
                "Ultralytics ByteTrack initialised successfully."
            )

            return tracker

        except ImportError as e:
            logger.error(
                "Ultralytics is not installed correctly: %s",
                e,
            )
            logger.error(
                "Install it with: "
                "python -m pip install -U ultralytics"
            )
            return None

        except Exception as e:
            logger.exception(
                "Failed to initialise Ultralytics ByteTrack: %s",
                e,
            )
            return None

    # ------------------------------------------------------------------
    def _init_sparta(self):
        """
        Optional SPARTA Re-ID integration.

        SPARTA is not required for ByteTrack itself.
        If the package is unavailable, tracking still works.
        """

        try:
            from sparta import SPARTAReID

            reid = SPARTAReID(
                feature_dim=256,
                device="cpu",
            )

            logger.info("SPARTA ReID initialised.")
            return reid

        except ImportError:
            logger.warning(
                "SPARTA ReID not installed - "
                "appearance Re-ID disabled."
            )
            return None

        except Exception as e:
            logger.warning(
                "SPARTA initialisation failed: %s",
                e,
            )
            return None

    # ------------------------------------------------------------------
    def update(
        self,
        detections: List[Detection],
        frame: np.ndarray,
    ) -> List[Track]:
        """
        Feed detections from the current frame into ByteTrack.

        Parameters
        ----------
        detections:
            Detection objects produced by the detector.

        frame:
            Current BGR image from OpenCV.

        Returns
        -------
        List[Track]
            Currently active ByteTrack tracks.
        """

        if self._bt is None:
            logger.error(
                "ByteTrack is unavailable. "
                "Returning empty track list."
            )
            return []

        # --------------------------------------------------------------
        # Filter detections by minimum bounding-box area.
        # --------------------------------------------------------------

        valid_detections = [
            d
            for d in detections
            if d.area >= self.min_box_area
        ]

        # --------------------------------------------------------------
        # Build Ultralytics Boxes tensor.
        #
        # Format:
        # [x1, y1, x2, y2, confidence, class_id]
        # --------------------------------------------------------------

        if valid_detections:

            data = np.array(
                [
                    [
                        d.x1,
                        d.y1,
                        d.x2,
                        d.y2,
                        d.confidence,
                        0,  # person class
                    ]
                    for d in valid_detections
                ],
                dtype=np.float32,
            )

        else:
            # Empty detection tensor.
            data = np.empty(
                (0, 6),
                dtype=np.float32,
            )

        boxes_tensor = self._torch.from_numpy(data)

        # Ultralytics Boxes requires the original image dimensions.
        boxes = self._Boxes(
            boxes_tensor,
            orig_shape=(
                frame.shape[0],
                frame.shape[1],
            ),
        )

        # --------------------------------------------------------------
        # Run ByteTrack.
        #
        # Current Ultralytics API:
        #
        #     tracker.update(Boxes)
        #
        # Returns:
        #
        # [x1, y1, x2, y2, track_id, score, class_id, index]
        # --------------------------------------------------------------

        try:
            raw_tracks = self._bt.update(boxes)

        except Exception as e:
            logger.exception(
                "ByteTrack update failed: %s",
                e,
            )
            return self._tracks

        # --------------------------------------------------------------
        # Convert Ultralytics tracks into our project's Track objects.
        # --------------------------------------------------------------

        tracks: List[Track] = []

        for row in raw_tracks:

            # Expected output:
            #
            # x1, y1, x2, y2, track_id, score, class_id, detection_idx

            x1 = float(row[0])
            y1 = float(row[1])
            x2 = float(row[2])
            y2 = float(row[3])

            track_id = int(row[4])
            score = float(row[5])

            class_id = int(row[6]) if len(row) > 6 else 0

            # Convert class ID to the project-level class name.
            class_name = self._class_name(
                class_id
            )

            tracks.append(
                Track(
                    track_id=track_id,
                    tlbr=np.array(
                        [x1, y1, x2, y2],
                        dtype=np.float32,
                    ),
                    score=score,
                    class_name=class_name,
                    age=0,
                    hits=1,
                    state="active",
                )
            )

        self._tracks = tracks

        # --------------------------------------------------------------
        # Optional SPARTA Re-ID.
        # --------------------------------------------------------------

        if self._sparta and self._tracks:
            self._tracks = self._apply_sparta(
                frame,
                self._tracks,
            )

        return self._tracks

    # ------------------------------------------------------------------
    @staticmethod
    def _class_name(class_id: int) -> str:
        """
        Convert detector class ID to project class name.

        RetailGuard currently focuses on people, so class 0 is treated
        as person. Additional classes can be added later.
        """

        if class_id == 0:
            return "person"

        return str(class_id)

    # ------------------------------------------------------------------
    def _apply_sparta(
        self,
        frame: np.ndarray,
        tracks: List[Track],
    ) -> List[Track]:
        """
        Apply optional SPARTA appearance-based Re-ID.
        """

        try:
            crops = []

            for track in tracks:

                x1, y1, x2, y2 = [
                    int(v)
                    for v in track.tlbr
                ]

                # Clamp coordinates to image bounds.
                x1 = max(0, min(x1, frame.shape[1]))
                x2 = max(0, min(x2, frame.shape[1]))
                y1 = max(0, min(y1, frame.shape[0]))
                y2 = max(0, min(y2, frame.shape[0]))

                crop = frame[y1:y2, x1:x2]

                if crop.size > 0:
                    crops.append(crop)
                else:
                    crops.append(
                        np.zeros(
                            (64, 32, 3),
                            dtype=np.uint8,
                        )
                    )

            refined_ids = self._sparta.match(
                crops,
                [t.track_id for t in tracks],
            )

            for track, refined_id in zip(
                tracks,
                refined_ids,
            ):
                track.track_id = int(refined_id)

        except Exception as e:
            logger.debug(
                "SPARTA pass failed: %s",
                e,
            )

        return tracks
