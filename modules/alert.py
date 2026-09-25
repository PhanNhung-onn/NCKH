"""
modules/alert.py

Alert system for RetailGuard-AI inference pipeline.

Responsibilities:
    - Receive anomaly events from the inference pipeline.
    - Apply per-track cooldown to avoid repeated alerts.
    - Log alerts to console.
    - Keep a lightweight in-memory alert history.

The module is intentionally independent from the database.
DatabaseStorage is responsible for persistent event storage.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class AlertEvent:
    """Represents one anomaly alert."""

    track_id: int
    score: float
    timestamp: float


class AlertSystem:
    """
    Handle anomaly alerts with per-track cooldown.

    Parameters
    ----------
    cooldown:
        Minimum number of seconds between two alerts
        for the same track ID.
    """

    def __init__(self, cooldown: int = 10):
        self.cooldown = max(0, int(cooldown))

        # Last alert timestamp for each track.
        self._last_alert: Dict[int, float] = {}

        # In-memory history for debugging/testing.
        self.history: List[AlertEvent] = []

        logger.info(
            f"AlertSystem initialised "
            f"(cooldown={self.cooldown}s)"
        )

    # ------------------------------------------------------------------
    def _can_alert(
        self,
        track_id: int,
        timestamp: float,
    ) -> bool:
        """
        Check whether this track is allowed to trigger an alert.
        """

        last_time = self._last_alert.get(track_id)

        # No previous alert for this track.
        if last_time is None:
            return True

        elapsed = timestamp - last_time

        return elapsed >= self.cooldown

    # ------------------------------------------------------------------
    def trigger(
        self,
        track_id: int,
        score: float,
        frame: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> bool:
        """
        Trigger an anomaly alert.

        Parameters
        ----------
        track_id:
            ByteTrack track ID.

        score:
            Combined anomaly score produced by AnomalyScorer.

        frame:
            Current video frame.
            Currently retained only for API compatibility and
            future snapshot/image alert functionality.

        timestamp:
            Unix timestamp. If omitted, time.time() is used.

        Returns
        -------
        bool
            True  -> alert was actually triggered.
            False -> suppressed because of cooldown.
        """

        if timestamp is None:
            timestamp = time.time()

        timestamp = float(timestamp)

        track_id = int(track_id)
        score = float(score)

        # --------------------------------------------------------------
        # Cooldown
        # --------------------------------------------------------------

        if not self._can_alert(
            track_id,
            timestamp,
        ):
            logger.debug(
                "Alert suppressed by cooldown: "
                f"track_id={track_id}, "
                f"score={score:.4f}"
            )

            return False

        # --------------------------------------------------------------
        # Record alert
        # --------------------------------------------------------------

        self._last_alert[track_id] = timestamp

        event = AlertEvent(
            track_id=track_id,
            score=score,
            timestamp=timestamp,
        )

        self.history.append(event)

        # --------------------------------------------------------------
        # Console/log alert
        # --------------------------------------------------------------

        logger.warning(
            "ANOMALY ALERT | "
            f"track_id={track_id} | "
            f"score={score:.4f} | "
            f"timestamp={timestamp:.3f}"
        )

        return True

    # ------------------------------------------------------------------
    def reset_track(self, track_id: int) -> None:
        """
        Remove cooldown state for one track.

        Useful when a tracked person disappears and a new
        tracking session should start cleanly.
        """

        self._last_alert.pop(
            int(track_id),
            None
        )

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all cooldown state and alert history."""

        self._last_alert.clear()
        self.history.clear()

    # ------------------------------------------------------------------
    def get_history(self) -> List[AlertEvent]:
        """Return a copy of the alert history."""

        return list(self.history)
