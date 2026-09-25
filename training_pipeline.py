"""
TRAINING PIPELINE
Video Data → OpenCV+PyTorch Detection+Tracking → Behavior Feature Extraction
→ HuMiPy/Pandas Build Dataset → Scikit-learn/PyTorch Train Model
→ Evaluation → pickle/TorchScript → Model Export
"""

from __future__ import annotations
import argparse
import logging
import pickle
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from modules.detector import YOLOWorldDetector
from modules.tracker import ByteTrackWrapper
from modules.features import BehaviorFeatureExtractor, FEATURE_DIM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# STEP 1 – Extract features from a list of video files
# ══════════════════════════════════════════════════════════════════════

def extract_features_from_videos(
    video_paths: List[str],
    classes: List[str] = None,
    window: int = 30,
    max_frames: Optional[int] = None,
) -> pd.DataFrame:
    """Run detection+tracking+feature extraction on every video."""
    classes = classes or ["person", "bag", "backpack"]

    detector = YOLOWorldDetector(classes=classes)
    tracker  = ByteTrackWrapper()
    feat_ext = BehaviorFeatureExtractor(window=window)

    rows: List[dict] = []
    col_names = [
        "speed_mean","speed_std","speed_max",
        "accel_mean","accel_max",
        "path_len","linearity",
        "heading_std","turn_rate",
        "loiter_score",
        "bbox_w","bbox_h","aspect_var",
        "pos_cx","pos_cy",
        "zone_entrance","zone_checkout","zone_high_value",
        "proximity","maturity","stop_frames","speed_cv","jerk_ratio","hv_dwell",
        "video_source", "track_id", "frame_end",
    ]

    for vpath in video_paths:
        logger.info(f"Processing {vpath} …")
        cap = cv2.VideoCapture(vpath)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break
            frame_idx += 1

            dets   = detector.detect(frame)
            tracks = tracker.update(dets, frame)
            feats  = feat_ext.update(tracks, frame)

            for tid, vec in feats.items():
                row = dict(zip(col_names[:FEATURE_DIM], vec.tolist()))
                row["video_source"] = Path(vpath).name
                row["track_id"]     = tid
                row["frame_end"]    = frame_idx
                rows.append(row)

        cap.release()
        logger.info(f"  → {frame_idx} frames, {len(rows)} feature rows so far")

    df = pd.DataFrame(rows, columns=col_names)
    logger.info(f"Total feature rows extracted: {len(df)}")
    return df


# ══════════════════════════════════════════════════════════════════════
# STEP 2 – Train Isolation Forest
# ══════════════════════════════════════════════════════════════════════

def train_isolation_forest(X: np.ndarray, contamination: float = 0.05):
    logger.info("Training Isolation Forest …")
    clf = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X)
    logger.info("Isolation Forest trained.")
    return clf


# ══════════════════════════════════════════════════════════════════════
# STEP 3 – Train Autoencoder
# ══════════════════════════════════════════════════════════════════════

def _build_autoencoder(d: int) -> nn.Module:
    class AE(nn.Module):
        def __init__(self):
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
    return AE()


def train_autoencoder(
    X_train: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "cpu",
) -> nn.Module:
    logger.info(f"Training Autoencoder ({epochs} epochs) …")
    model = _build_autoencoder(FEATURE_DIM).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    t = torch.tensor(X_train, dtype=torch.float32)
    ds = TensorDataset(t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        for (batch,) in dl:
            batch = batch.to(device)
            recon = model(batch)
            loss  = loss_fn(recon, batch)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(batch)
        ep_loss /= len(X_train)
        if ep % 10 == 0:
            logger.info(f"  Epoch {ep:3d}/{epochs}  loss={ep_loss:.6f}")

    model.eval()
    logger.info("Autoencoder trained.")
    return model


# ══════════════════════════════════════════════════════════════════════
# STEP 4 – Evaluate  (requires some labelled anomaly rows)
# ══════════════════════════════════════════════════════════════════════

def evaluate(
    iforest: IsolationForest,
    ae: nn.Module,
    scaler: StandardScaler,
    X: np.ndarray,
    y: np.ndarray,   # 1=anomaly, 0=normal
    device: str = "cpu",
):
    X_s = scaler.transform(X)
    t   = torch.tensor(X_s, dtype=torch.float32).to(device)

    with torch.no_grad():
        recon = ae(t).cpu().numpy()
    ae_scores = np.mean((X_s - recon) ** 2, axis=1)

    if_raw    = -iforest.decision_function(X_s)

    combined  = 0.4 * _norm(if_raw) + 0.6 * _norm(ae_scores)

    roc  = roc_auc_score(y, combined) if len(np.unique(y)) > 1 else float("nan")
    ap   = average_precision_score(y, combined) if len(np.unique(y)) > 1 else float("nan")
    logger.info(f"Evaluation  ROC-AUC={roc:.4f}  AP={ap:.4f}")
    return {"roc_auc": roc, "average_precision": ap}


def _norm(x: np.ndarray) -> np.ndarray:
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-9)


# ══════════════════════════════════════════════════════════════════════
# STEP 5 – Export
# ══════════════════════════════════════════════════════════════════════

def export_model(
    iforest: IsolationForest,
    ae: nn.Module,
    scaler: StandardScaler,
    out_path: str = "models/anomaly_model.pkl",
):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ae_state_path = str(out_path).replace(".pkl", "_ae.pt")
    torch.save(ae.state_dict(), ae_state_path)

    bundle = {
        "isolation_forest":   iforest,
        "scaler":             scaler,
        "autoencoder_state":  ae_state_path,
        "feature_dim":        FEATURE_DIM,
        "created":            time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info(f"Model exported → {out_path}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main(args):
    # 1. Feature extraction
    if args.feature_csv and Path(args.feature_csv).exists():
        logger.info(f"Loading pre-extracted features from {args.feature_csv}")
        df = pd.read_csv(args.feature_csv)
    else:
        video_list = args.videos.split(",") if args.videos else []
        if not video_list:
            raise ValueError("Provide --videos or --feature-csv")
        df = extract_features_from_videos(
            video_list,
            max_frames=args.max_frames,
            window=args.window,
        )
        if args.save_features:
            Path(args.save_features).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.save_features, index=False)
            logger.info(f"Features saved → {args.save_features}")

    feat_cols = df.columns[:FEATURE_DIM].tolist()
    X = df[feat_cols].values.astype(np.float32)

    # Handle NaN / Inf
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

    # Labels (optional)
    y = df["label"].values if "label" in df.columns else np.zeros(len(X))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # 3. Train
    iforest = train_isolation_forest(X_train_s, contamination=args.contamination)
    ae      = train_autoencoder(
        X_train_s, epochs=args.epochs, lr=args.lr, device=args.device
    )

    # 4. Evaluate
    if len(np.unique(y_test)) > 1:
        evaluate(iforest, ae, scaler, X_test, y_test, device=args.device)
    else:
        logger.info("No labelled anomalies found – skipping evaluation metrics.")

    # 5. Export
    export_model(iforest, ae, scaler, out_path=args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retail Anomaly – Training Pipeline")
    parser.add_argument("--videos",        default="",   help="Comma-separated video paths")
    parser.add_argument("--feature-csv",   default="",   help="Pre-extracted CSV to skip detection")
    parser.add_argument("--save-features", default="data/features.csv")
    parser.add_argument("--output",        default="models/anomaly_model.pkl")
    parser.add_argument("--window",        type=int,   default=30)
    parser.add_argument("--max-frames",    type=int,   default=None)
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--device",        default="cpu")
    args = parser.parse_args()
    main(args)
