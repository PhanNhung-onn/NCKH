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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# STEP 1 – Resolve video inputs
# ══════════════════════════════════════════════════════════════════════

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}


def resolve_video_paths(inputs: List[str]) -> List[str]:
    """
    Accept:
      - individual video files
      - directories containing videos
      - comma-separated paths

    Returns a sorted list of video files.
    """

    video_paths = []

    for item in inputs:
        item = item.strip()

        if not item:
            continue

        path = Path(item)

        # ----------------------------------------------------------
        # Single video file
        # ----------------------------------------------------------
        if path.is_file():
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                video_paths.append(str(path))
            else:
                logger.warning(
                    f"Skipping unsupported file type: {path}"
                )

        # ----------------------------------------------------------
        # Directory
        # ----------------------------------------------------------
        elif path.is_dir():
            logger.info(f"Searching videos in directory: {path}")

            for file_path in sorted(path.rglob("*")):
                if (
                    file_path.is_file()
                    and file_path.suffix.lower() in VIDEO_EXTENSIONS
                ):
                    video_paths.append(str(file_path))

        else:
            logger.warning(
                f"Path does not exist: {path}"
            )

    # Remove duplicates while preserving sorted order
    video_paths = sorted(set(video_paths))

    logger.info(
        f"Found {len(video_paths)} video file(s) to process."
    )

    if video_paths:
        logger.info(f"First video: {video_paths[0]}")

    return video_paths


# ══════════════════════════════════════════════════════════════════════
# STEP 2 – Extract features
# ══════════════════════════════════════════════════════════════════════

def extract_features_from_videos(
    video_paths: List[str],
    classes: List[str] = None,
    window: int = 30,
    max_frames: Optional[int] = None,
) -> pd.DataFrame:

    """Run detection + tracking + feature extraction on every video."""

    classes = classes or ["person", "bag", "backpack"]

    detector = YOLOWorldDetector(classes=classes)
    tracker = ByteTrackWrapper()
    feat_ext = BehaviorFeatureExtractor(window=window)

    rows: List[dict] = []

    col_names = [
        "speed_mean",
        "speed_std",
        "speed_max",
        "accel_mean",
        "accel_max",
        "path_len",
        "linearity",
        "heading_std",
        "turn_rate",
        "loiter_score",
        "bbox_w",
        "bbox_h",
        "aspect_var",
        "pos_cx",
        "pos_cy",
        "zone_entrance",
        "zone_checkout",
        "zone_high_value",
        "proximity",
        "maturity",
        "stop_frames",
        "speed_cv",
        "jerk_ratio",
        "hv_dwell",
        "video_source",
        "track_id",
        "frame_end",
    ]

    total_frames = 0
    successful_videos = 0
    failed_videos = 0

    for video_number, vpath in enumerate(video_paths, start=1):

        logger.info(
            f"[{video_number}/{len(video_paths)}] Processing {vpath}"
        )

        cap = cv2.VideoCapture(vpath)

        if not cap.isOpened():
            logger.error(
                f"Cannot open video: {vpath}"
            )
            failed_videos += 1
            continue

        frame_idx = 0
        video_feature_count_before = len(rows)

        try:
            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                if max_frames is not None and frame_idx >= max_frames:
                    break

                frame_idx += 1
                total_frames += 1

                # --------------------------------------------------
                # Detection
                # --------------------------------------------------
                dets = detector.detect(frame)

                # --------------------------------------------------
                # Tracking
                # --------------------------------------------------
                tracks = tracker.update(dets, frame)

                # --------------------------------------------------
                # Behavior feature extraction
                # --------------------------------------------------
                feats = feat_ext.update(tracks, frame)

                for tid, vec in feats.items():

                    row = dict(
                        zip(
                            col_names[:FEATURE_DIM],
                            vec.tolist()
                        )
                    )

                    row["video_source"] = Path(vpath).name
                    row["track_id"] = tid
                    row["frame_end"] = frame_idx

                    rows.append(row)

        except Exception as exc:

            logger.exception(
                f"Error while processing {vpath}: {exc}"
            )

            failed_videos += 1
            cap.release()
            continue

        finally:
            cap.release()

        video_features = len(rows) - video_feature_count_before

        if frame_idx > 0:
            successful_videos += 1

        logger.info(
            f"    Frames read: {frame_idx}"
        )

        logger.info(
            f"    Features extracted: {video_features}"
        )

        logger.info(
            f"    Total feature rows so far: {len(rows)}"
        )

    logger.info("=" * 70)
    logger.info("FEATURE EXTRACTION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Videos processed successfully: {successful_videos}")
    logger.info(f"Videos failed: {failed_videos}")
    logger.info(f"Total frames read: {total_frames}")
    logger.info(f"Total feature rows: {len(rows)}")

    df = pd.DataFrame(rows, columns=col_names)

    if df.empty:
        logger.error(
            "No features were extracted from any video."
        )

    return df


# ══════════════════════════════════════════════════════════════════════
# STEP 3 – Train Isolation Forest
# ══════════════════════════════════════════════════════════════════════

def train_isolation_forest(
    X: np.ndarray,
    contamination: float = 0.05
):
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
# STEP 4 – Train Autoencoder
# ══════════════════════════════════════════════════════════════════════

def _build_autoencoder(d: int) -> nn.Module:

    class AE(nn.Module):

        def __init__(self):
            super().__init__()

            self.enc = nn.Sequential(
                nn.Linear(d, 16),
                nn.ReLU(),

                nn.Linear(16, 8),
                nn.ReLU(),

                nn.Linear(8, 4),
            )

            self.dec = nn.Sequential(
                nn.Linear(4, 8),
                nn.ReLU(),

                nn.Linear(8, 16),
                nn.ReLU(),

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

    logger.info(
        f"Training Autoencoder ({epochs} epochs) ..."
    )

    model = _build_autoencoder(FEATURE_DIM).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    loss_fn = nn.MSELoss()

    t = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    ds = TensorDataset(t)

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True
    )

    model.train()

    for ep in range(1, epochs + 1):

        ep_loss = 0.0

        for (batch,) in dl:

            batch = batch.to(device)

            recon = model(batch)

            loss = loss_fn(
                recon,
                batch
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            ep_loss += loss.item() * len(batch)

        ep_loss /= len(X_train)

        if ep % 10 == 0:
            logger.info(
                f"  Epoch {ep:3d}/{epochs} "
                f"loss={ep_loss:.6f}"
            )

    model.eval()

    logger.info("Autoencoder trained.")

    return model


# ══════════════════════════════════════════════════════════════════════
# STEP 5 – Evaluation
# ══════════════════════════════════════════════════════════════════════

def _norm(x: np.ndarray) -> np.ndarray:

    mn, mx = x.min(), x.max()

    return (x - mn) / (mx - mn + 1e-9)


def evaluate(
    iforest: IsolationForest,
    ae: nn.Module,
    scaler: StandardScaler,
    X: np.ndarray,
    y: np.ndarray,
    device: str = "cpu",
):

    X_s = scaler.transform(X)

    t = torch.tensor(
        X_s,
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():

        recon = ae(t).cpu().numpy()

    ae_scores = np.mean(
        (X_s - recon) ** 2,
        axis=1
    )

    if_raw = -iforest.decision_function(X_s)

    combined = (
        0.4 * _norm(if_raw)
        + 0.6 * _norm(ae_scores)
    )

    roc = (
        roc_auc_score(y, combined)
        if len(np.unique(y)) > 1
        else float("nan")
    )

    ap = (
        average_precision_score(y, combined)
        if len(np.unique(y)) > 1
        else float("nan")
    )

    logger.info(
        f"Evaluation ROC-AUC={roc:.4f} "
        f"AP={ap:.4f}"
    )

    return {
        "roc_auc": roc,
        "average_precision": ap
    }


# ══════════════════════════════════════════════════════════════════════
# STEP 6 – Export
# ══════════════════════════════════════════════════════════════════════

def export_model(
    iforest: IsolationForest,
    ae: nn.Module,
    scaler: StandardScaler,
    out_path: str = "models/anomaly_model.pkl",
):

    Path(out_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    ae_state_path = str(
        out_path
    ).replace(".pkl", "_ae.pt")

    torch.save(
        ae.state_dict(),
        ae_state_path
    )

    bundle = {
        "isolation_forest": iforest,
        "scaler": scaler,
        "autoencoder_state": ae_state_path,
        "feature_dim": FEATURE_DIM,
        "created": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }

    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)

    logger.info(
        f"Model exported → {out_path}"
    )


# ══════════════════════════════════════════════════════════════════════
# STEP 7 – MAIN
# ══════════════════════════════════════════════════════════════════════

def main(args):

    # --------------------------------------------------------------
    # 1. Feature extraction
    # --------------------------------------------------------------

    if (
        args.feature_csv
        and Path(args.feature_csv).exists()
    ):

        logger.info(
            f"Loading pre-extracted features "
            f"from {args.feature_csv}"
        )

        df = pd.read_csv(
            args.feature_csv
        )

    else:

        if not args.videos:

            raise ValueError(
                "Provide --videos or --feature-csv"
            )

        # ----------------------------------------------------------
        # NEW:
        # Resolve files/directories automatically.
        # ----------------------------------------------------------

        raw_inputs = args.videos.split(",")

        video_list = resolve_video_paths(
            raw_inputs
        )

        if not video_list:

            raise RuntimeError(
                "No video files found. "
                "Check --videos path."
            )

        df = extract_features_from_videos(
            video_list,
            max_frames=args.max_frames,
            window=args.window,
        )

        if args.save_features:

            Path(
                args.save_features
            ).parent.mkdir(
                parents=True,
                exist_ok=True
            )

            df.to_csv(
                args.save_features,
                index=False
            )

            logger.info(
                f"Features saved → "
                f"{args.save_features}"
            )

    # --------------------------------------------------------------
    # Safety check
    # --------------------------------------------------------------

    if df.empty:

        raise RuntimeError(
            "No features were extracted. "
            "Please check video files, detector, "
            "tracker, and feature extractor."
        )

    if len(df) < 2:

        raise RuntimeError(
            f"Only {len(df)} feature row(s) were extracted. "
            "At least 2 rows are required for train/test split."
        )

    # --------------------------------------------------------------
    # 2. Prepare features
    # --------------------------------------------------------------

    feat_cols = (
        df.columns[:FEATURE_DIM]
        .tolist()
    )

    X = df[
        feat_cols
    ].values.astype(
        np.float32
    )

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    # --------------------------------------------------------------
    # Labels
    # --------------------------------------------------------------

    y = (
        df["label"].values
        if "label" in df.columns
        else np.zeros(len(X))
    )

    # --------------------------------------------------------------
    # 3. Train / test split
    # --------------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )
    )

    # --------------------------------------------------------------
    # 4. Scale
    # --------------------------------------------------------------

    scaler = StandardScaler()

    X_train_s = scaler.fit_transform(
        X_train
    )

    X_test_s = scaler.transform(
        X_test
    )

    # --------------------------------------------------------------
    # 5. Train models
    # --------------------------------------------------------------

    iforest = train_isolation_forest(
        X_train_s,
        contamination=args.contamination
    )

    ae = train_autoencoder(
        X_train_s,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device
    )

    # --------------------------------------------------------------
    # 6. Evaluate
    # --------------------------------------------------------------

    if len(np.unique(y_test)) > 1:

        evaluate(
            iforest,
            ae,
            scaler,
            X_test,
            y_test,
            device=args.device
        )

    else:

        logger.info(
            "No labelled anomalies found – "
            "skipping evaluation metrics."
        )

    # --------------------------------------------------------------
    # 7. Export
    # --------------------------------------------------------------

    export_model(
        iforest,
        ae,
        scaler,
        out_path=args.output
    )


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Retail Anomaly – Training Pipeline"
    )

    parser.add_argument(
        "--videos",
        default="",
        help=(
            "Video file, directory, "
            "or comma-separated video paths"
        )
    )

    parser.add_argument(
        "--feature-csv",
        default="",
        help="Pre-extracted CSV to skip detection"
    )

    parser.add_argument(
        "--save-features",
        default="data/features.csv"
    )

    parser.add_argument(
        "--output",
        default="models/anomaly_model.pkl"
    )

    parser.add_argument(
        "--window",
        type=int,
        default=30
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None
    )

    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--device",
        default="cpu"
    )

    args = parser.parse_args()

    main(args)
