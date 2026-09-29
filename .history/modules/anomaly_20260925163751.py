"""
modules/anomaly.py
Anomaly Scoring Model — Isolation Forest + Autoencoder ensemble.

Training produces a .pkl artifact consumed here at inference time.
"""

from __future__ import annotations
import numpy as np
import logging
import pickle
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

FEATURE_DIM = 24


class AnomalyScorer:
    """
    Ensemble scorer:
      • Isolation Forest  → fast, works out-of-the-box with no labels
      • Autoencoder       → captures temporal reconstruction error
    Final score = weighted average of normalised individual scores.
    """

    def __init__(
        self,
        model_path: str = "models/anomaly_model.pkl",
        threshold: float = 0.65,
        if_weight: float = 0.4,
        ae_weight: float = 0.6,
    ):
        self.threshold = threshold
        self.if_weight = if_weight
        self.ae_weight = ae_weight
        self._if = None
        self._ae = None
        self._scaler = None
        self._load(model_path)

    # ------------------------------------------------------------------
    def _load(self, path: str):
        p = Path(path)
        if not p.exists():
            logger.warning(
                f"Model file {path} not found. "
                "Scoring will use an untrained fallback (random-like). "
                "Run training_pipeline.py first."
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

    # ------------------------------------------------------------------
    def _init_untrained(self):
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        self._if     = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        self._scaler = StandardScaler()
        # Fit on random data so it doesn't crash
        X_dummy = np.random.randn(200, FEATURE_DIM)
        self._scaler.fit(X_dummy)
        self._if.fit(X_dummy)
        logger.info("Using untrained fallback models (run training_pipeline.py to train).")

    # ------------------------------------------------------------------
    def score(self, feature_vec: np.ndarray) -> Tuple[float, bool]:
        """
        Returns (score ∈ [0,1], is_anomaly: bool).
        Higher score → more anomalous.
        """
        x = feature_vec.reshape(1, -1)

        if self._scaler is not None:
            x = self._scaler.transform(x)

        score = 0.0

        # Isolation Forest score (decision_function → negative → anomaly)
        if self._if is not None:
            raw = -float(self._if.decision_function(x)[0])  # higher = more anomalous
            if_score = self._sigmoid(raw)
            score += self.if_weight * if_score
        else:
            if_score = 0.5

        # Autoencoder reconstruction error
        if self._ae is not None:
            import torch
            with torch.no_grad():
                t = torch.tensor(x, dtype=torch.float32)
                recon = self._ae(t)
                mse = float(torch.nn.functional.mse_loss(recon, t).item())
            ae_score = self._sigmoid(mse * 10)
            score += self.ae_weight * ae_score
        else:
            score = score / (self.if_weight if self._if else 1.0)

        # Normalise to [0,1]
        score = float(np.clip(score, 0.0, 1.0))
        return score, score >= self.threshold

    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))


# ──────────────────────────────────────────────────────────────────────
def _build_autoencoder(input_dim: int = 24):
    """Lightweight MLP autoencoder for anomaly detection."""
    import torch
    import torch.nn as nn

    class Autoencoder(nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(d, 16), nn.ReLU(),
                nn.Linear(16, 8),  nn.ReLU(),
                nn.Linear(8, 4),
            )
            self.decoder = nn.Sequential(
                nn.Linear(4, 8),  nn.ReLU(),
                nn.Linear(8, 16), nn.ReLU(),
                nn.Linear(16, d),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return Autoencoder(input_dim)
