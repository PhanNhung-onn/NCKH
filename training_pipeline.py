"""
TRAINING PIPELINE
=================
Stage 1+2 (kinematic — giữ nguyên):
    Video → Detection → Tracking → Feature Extraction
    → Train Isolation Forest + MLP Autoencoder → Export .pkl

Stage 3+4 (pose — MỚI):
    Video → Detection → Tracking → Pose Extraction (YOLOv8-pose)
    → Build pose sequences → Train GCAE (encoder+decoder)
    → Freeze encoder → Train Shopformer Transformer
    → Export pose_tokenizer.pt

Cách chạy:
    # Chỉ kinematic (cũ):
    python training_pipeline.py --videos data/normal_videos --output models/anomaly_model.pkl

    # Kinematic + Pose (đầy đủ):
    python training_pipeline.py --videos data/normal_videos --output models/anomaly_model.pkl --train-pose

    # Chỉ pose (nếu đã có kinematic model):
    python training_pipeline.py --videos data/normal_videos --stage pose --pose-output models/pose_tokenizer.pt
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
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ══════════════════════════════════════════════════════════════════════
# STEP 0 — Resolve video paths (giữ nguyên)
# ══════════════════════════════════════════════════════════════════════

def resolve_video_paths(inputs: List[str]) -> List[str]:
    video_paths = []
    for item in inputs:
        item = item.strip()
        if not item:
            continue
        path = Path(item)
        if path.is_file():
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                video_paths.append(str(path))
            else:
                logger.warning(f"Skipping unsupported file type: {path}")
        elif path.is_dir():
            logger.info(f"Searching videos in directory: {path}")
            for fp in sorted(path.rglob("*")):
                if fp.is_file() and fp.suffix.lower() in VIDEO_EXTENSIONS:
                    video_paths.append(str(fp))
        else:
            logger.warning(f"Path does not exist: {path}")
    video_paths = sorted(set(video_paths))
    logger.info(f"Found {len(video_paths)} video file(s).")
    return video_paths


# ══════════════════════════════════════════════════════════════════════
# STEP 1 — Kinematic feature extraction (giữ nguyên)
# ══════════════════════════════════════════════════════════════════════

def extract_features_from_videos(
    video_paths: List[str],
    classes:     List[str]       = None,
    window:      int             = 30,
    max_frames:  Optional[int]   = None,
) -> pd.DataFrame:
    """Trích xuất 24-D kinematic features từ video."""
    classes  = classes or ["person", "bag", "backpack"]
    detector = YOLOWorldDetector(classes=classes)
    tracker  = ByteTrackWrapper()
    feat_ext = BehaviorFeatureExtractor(window=window)

    col_names = [
        "speed_mean","speed_std","speed_max","accel_mean","accel_max",
        "path_len","linearity","heading_std","turn_rate","loiter_score",
        "bbox_w","bbox_h","aspect_var","pos_cx","pos_cy",
        "zone_entrance","zone_checkout","zone_high_value",
        "proximity","maturity","stop_frames","speed_cv","jerk_ratio","hv_dwell",
        "video_source","track_id","frame_end",
    ]

    rows: List[dict] = []
    total_frames = 0
    successful_videos = 0
    failed_videos = 0

    for video_number, vpath in enumerate(video_paths, start=1):
        logger.info(f"[{video_number}/{len(video_paths)}] {vpath}")
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            logger.error(f"Cannot open: {vpath}")
            failed_videos += 1
            continue

        frame_idx = 0
        rows_before = len(rows)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if max_frames is not None and frame_idx >= max_frames:
                    break
                frame_idx  += 1
                total_frames += 1

                dets   = detector.detect(frame)
                tracks = tracker.update(dets, frame)
                feats  = feat_ext.update(tracks, frame)

                for tid, vec in feats.items():
                    row = dict(zip(col_names[:FEATURE_DIM], vec.tolist()))
                    row["video_source"] = Path(vpath).name
                    row["track_id"]     = tid
                    row["frame_end"]    = frame_idx
                    rows.append(row)
        except Exception as exc:
            logger.exception(f"Error processing {vpath}: {exc}")
            failed_videos += 1
        finally:
            cap.release()

        if frame_idx > 0:
            successful_videos += 1
        logger.info(f"  Frames: {frame_idx}  Features: {len(rows)-rows_before}")

    logger.info(
        f"Summary: {successful_videos} ok / {failed_videos} failed / "
        f"{total_frames} frames / {len(rows)} feature rows"
    )
    return pd.DataFrame(rows, columns=col_names)


# ══════════════════════════════════════════════════════════════════════
# STEP 2 — Pose sequence extraction (MỚI)
# ══════════════════════════════════════════════════════════════════════

def extract_pose_sequences(
    video_paths:  List[str],
    classes:      List[str]     = None,
    seq_len:      int           = 12,
    max_frames:   Optional[int] = None,
    pose_model:   str           = "yolov8n-pose.pt",
    pose_conf:    float         = 0.5,
) -> np.ndarray:
    """
    Trích xuất pose sequences từ video.

    Returns:
        np.ndarray (N_sequences, seq_len, 17, 3)
        Mỗi sequence = 12 frame liên tiếp của 1 track.
    """
    try:
        from modules.pose_extractor import PoseExtractor
    except ImportError:
        raise RuntimeError(
            "pose_extractor.py không tìm thấy. "
            "Đặt file vào modules/ trước khi train pose model."
        )

    classes      = classes or ["person", "bag", "backpack"]
    detector     = YOLOWorldDetector(classes=classes)
    tracker      = ByteTrackWrapper()
    pose_ext     = PoseExtractor(model_name=pose_model, conf=pose_conf)

    from collections import defaultdict, deque
    # Buffer: track_id → deque of (17, 3)
    kpt_bufs: dict = defaultdict(lambda: deque(maxlen=seq_len))
    sequences: List[np.ndarray] = []

    total_frames = 0

    for vpath in video_paths:
        logger.info(f"[Pose] Processing {vpath} …")
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            logger.error(f"Cannot open: {vpath}")
            continue

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if max_frames is not None and frame_idx >= max_frames:
                    break
                frame_idx    += 1
                total_frames += 1

                dets   = detector.detect(frame)
                tracks = tracker.update(dets, frame)

                for track in tracks:
                    kpts = pose_ext.extract(frame, track.tlbr)
                    if kpts is None:
                        # Nếu không detect được pose → dùng zeros (mask)
                        kpts = np.zeros((17, 3), dtype=np.float32)

                    kpt_bufs[track.track_id].append(kpts)

                    # Khi đủ seq_len → lưu sequence
                    if len(kpt_bufs[track.track_id]) == seq_len:
                        seq = np.stack(kpt_bufs[track.track_id])  # (T, 17, 3)
                        sequences.append(seq)
        except Exception as e:
            logger.exception(f"Error: {e}")
        finally:
            cap.release()

        logger.info(f"  Frames: {frame_idx}  Sequences so far: {len(sequences)}")

    if not sequences:
        raise RuntimeError("Không trích xuất được pose sequence nào. Kiểm tra video và pose model.")

    data = np.stack(sequences, axis=0)   # (N, T, 17, 3)
    logger.info(f"Pose sequences: {data.shape}  ({total_frames} frames)")
    return data


# ══════════════════════════════════════════════════════════════════════
# STEP 3 — Train GCAE (Stage 1 Shopformer)
# ══════════════════════════════════════════════════════════════════════

def train_gcae(
    sequences:  np.ndarray,    # (N, T, 17, 3)
    epochs:     int   = 30,
    lr:         float = 1e-3,
    batch_size: int   = 64,
    device:     str   = "cpu",
) -> nn.Module:
    """
    Train Graph Convolutional Autoencoder trên normal pose sequences.
    Sau khi train, encoder được lấy ra làm tokenizer.
    """
    from modules.pose_tokenizer import GCAEFull

    logger.info(f"Training GCAE ({epochs} epochs, {len(sequences)} sequences) …")

    model   = GCAEFull().to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # (N, T, 17, 3)
    data = torch.tensor(sequences, dtype=torch.float32)
    ds   = TensorDataset(data)
    dl   = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model.train()
    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        for (batch,) in dl:
            batch = batch.to(device)
            recon = model(batch)
            loss  = loss_fn(recon, batch)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(batch)
        ep_loss /= len(sequences)
        if ep % 5 == 0:
            logger.info(f"  GCAE Epoch {ep:3d}/{epochs}  loss={ep_loss:.6f}")

    model.eval()
    logger.info("GCAE trained.")
    return model


# ══════════════════════════════════════════════════════════════════════
# STEP 4 — Train Shopformer Transformer (Stage 2)
# ══════════════════════════════════════════════════════════════════════

def train_shopformer(
    sequences:   np.ndarray,   # (N, T, 17, 3)
    gcae_model:  nn.Module,
    epochs:      int   = 20,
    lr:          float = 5e-5,
    batch_size:  int   = 64,
    device:      str   = "cpu",
) -> nn.Module:
    """
    Freeze GCAE encoder → train Shopformer Transformer.
    Input: pose sequences → tokens → reconstruct → MSE loss.
    """
    from modules.pose_tokenizer import ShopformerTransformer

    logger.info(f"Training Shopformer Transformer ({epochs} epochs) …")

    # Freeze encoder
    encoder = gcae_model.encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    transformer = ShopformerTransformer().to(device)
    opt         = torch.optim.Adam(transformer.parameters(), lr=lr)
    loss_fn     = nn.MSELoss()

    data = torch.tensor(sequences, dtype=torch.float32)
    ds   = TensorDataset(data)
    dl   = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    transformer.train()
    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        for (batch,) in dl:
            batch = batch.to(device)
            with torch.no_grad():
                tokens = encoder(batch)             # (B, N_tokens, embed_dim)
            recon, _ = transformer(tokens)
            loss     = loss_fn(recon, tokens)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(batch)
        ep_loss /= len(sequences)
        if ep % 5 == 0:
            logger.info(f"  Shopformer Epoch {ep:3d}/{epochs}  loss={ep_loss:.6f}")

    transformer.eval()
    logger.info("Shopformer Transformer trained.")
    return transformer


# ══════════════════════════════════════════════════════════════════════
# STEP 5 — Kinematic model functions (giữ nguyên)
# ══════════════════════════════════════════════════════════════════════

def train_isolation_forest(X: np.ndarray, contamination: float = 0.05):
    logger.info("Training Isolation Forest …")
    clf = IsolationForest(
        n_estimators=200, max_samples="auto",
        contamination=contamination, random_state=42, n_jobs=-1,
    )
    clf.fit(X)
    logger.info("Isolation Forest trained.")
    return clf


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
        def forward(self, x): return self.dec(self.enc(x))
    return AE()


def train_autoencoder(
    X_train: np.ndarray,
    epochs: int = 50, lr: float = 1e-3,
    batch_size: int = 64, device: str = "cpu",
) -> nn.Module:
    logger.info(f"Training MLP Autoencoder ({epochs} epochs) …")
    model   = _build_autoencoder(FEATURE_DIM).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    t  = torch.tensor(X_train, dtype=torch.float32)
    dl = DataLoader(TensorDataset(t), batch_size=batch_size, shuffle=True)
    model.train()
    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        for (batch,) in dl:
            batch = batch.to(device)
            loss  = loss_fn(model(batch), batch)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(batch)
        ep_loss /= len(X_train)
        if ep % 10 == 0:
            logger.info(f"  MLP AE Epoch {ep:3d}/{epochs}  loss={ep_loss:.6f}")
    model.eval()
    logger.info("MLP Autoencoder trained.")
    return model


def _norm(x: np.ndarray) -> np.ndarray:
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-9)


def evaluate(iforest, ae, scaler, X, y, device="cpu"):
    X_s = scaler.transform(X)
    t   = torch.tensor(X_s, dtype=torch.float32).to(device)
    with torch.no_grad():
        recon = ae(t).cpu().numpy()
    ae_scores = np.mean((X_s - recon) ** 2, axis=1)
    if_raw    = -iforest.decision_function(X_s)
    combined  = 0.4 * _norm(if_raw) + 0.6 * _norm(ae_scores)
    roc = roc_auc_score(y, combined) if len(np.unique(y)) > 1 else float("nan")
    ap  = average_precision_score(y, combined) if len(np.unique(y)) > 1 else float("nan")
    logger.info(f"Evaluation  ROC-AUC={roc:.4f}  AP={ap:.4f}")
    return {"roc_auc": roc, "average_precision": ap}


def export_kinematic_model(iforest, ae, scaler, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ae_state_path = out_path.replace(".pkl", "_ae.pt")
    torch.save(ae.state_dict(), ae_state_path)
    bundle = {
        "isolation_forest":  iforest,
        "scaler":            scaler,
        "autoencoder_state": ae_state_path,
        "feature_dim":       FEATURE_DIM,
        "created":           time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info(f"Kinematic model exported → {out_path}")


def export_pose_model(gcae, transformer, out_path: str):
    """Lưu encoder + transformer vào 1 file .pt."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder":     gcae.encoder.state_dict(),
        "transformer": transformer.state_dict(),
        "created":     time.strftime("%Y-%m-%d %H:%M:%S"),
    }, out_path)
    logger.info(f"Pose tokenizer exported → {out_path}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main(args):
    video_list = []
    if not args.feature_csv or not Path(args.feature_csv).exists():
        if not args.videos:
            raise ValueError("Cần --videos hoặc --feature-csv")
        video_list = resolve_video_paths(args.videos.split(","))
        if not video_list:
            raise RuntimeError("Không tìm thấy video nào.")

    run_kinematic = args.stage in ("kinematic", "all")
    run_pose      = args.stage in ("pose", "all") or args.train_pose

    # ── KINEMATIC STAGE ───────────────────────────────────────────
    if run_kinematic:
        logger.info("=" * 60)
        logger.info("KINEMATIC STAGE")
        logger.info("=" * 60)

        if args.feature_csv and Path(args.feature_csv).exists():
            logger.info(f"Loading features from {args.feature_csv}")
            df = pd.read_csv(args.feature_csv)
        else:
            df = extract_features_from_videos(
                video_list, max_frames=args.max_frames, window=args.window
            )
            if args.save_features:
                Path(args.save_features).parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(args.save_features, index=False)
                logger.info(f"Features saved → {args.save_features}")

        if df.empty or len(df) < 2:
            raise RuntimeError(f"Chỉ có {len(df)} feature rows — không đủ để train.")

        X = np.nan_to_num(
            df[df.columns[:FEATURE_DIM]].values.astype(np.float32),
            nan=0.0, posinf=1.0, neginf=0.0,
        )
        y = df["label"].values if "label" in df.columns else np.zeros(len(X))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler    = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        iforest = train_isolation_forest(X_train_s, contamination=args.contamination)
        ae      = train_autoencoder(X_train_s, epochs=args.epochs, lr=args.lr, device=args.device)

        if len(np.unique(y_test)) > 1:
            evaluate(iforest, ae, scaler, X_test, y_test, device=args.device)
        else:
            logger.info("Không có nhãn anomaly → bỏ qua evaluation.")

        export_kinematic_model(iforest, ae, scaler, out_path=args.output)

    # ── POSE STAGE ────────────────────────────────────────────────
    if run_pose:
        logger.info("=" * 60)
        logger.info("POSE STAGE (GCAE + Shopformer)")
        logger.info("=" * 60)

        if not video_list:
            video_list = resolve_video_paths(args.videos.split(","))

        sequences = extract_pose_sequences(
            video_list,
            max_frames=args.max_frames,
            pose_model=args.pose_detector,
            pose_conf=args.pose_conf,
        )

        # Stage 3: Train GCAE
        gcae = train_gcae(
            sequences,
            epochs=args.gcae_epochs,
            lr=args.lr,
            device=args.device,
        )

        # Stage 4: Freeze encoder → Train Transformer
        transformer = train_shopformer(
            sequences,
            gcae_model=gcae,
            epochs=args.shopformer_epochs,
            lr=args.shopformer_lr,
            device=args.device,
        )

        export_pose_model(gcae, transformer, out_path=args.pose_output)

    logger.info("Training hoàn tất.")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retail Anomaly – Training Pipeline")

    # Input
    parser.add_argument("--videos",       default="",  help="Video file, dir, hoặc comma-separated")
    parser.add_argument("--feature-csv",  default="",  help="CSV kinematic đã trích xuất sẵn")
    parser.add_argument("--save-features",default="data/features.csv")

    # Stage control
    parser.add_argument("--stage",      default="all",
                        choices=["all", "kinematic", "pose"],
                        help="Stage nào cần train")
    parser.add_argument("--train-pose", action="store_true",
                        help="Shortcut để thêm pose stage (--stage all)")

    # Kinematic output
    parser.add_argument("--output",       default="models/anomaly_model.pkl")

    # Pose output
    parser.add_argument("--pose-output",  default="models/pose_tokenizer.pt")

    # Kinematic hyperparams
    parser.add_argument("--window",        type=int,   default=30)
    parser.add_argument("--max-frames",    type=int,   default=None)
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--epochs",        type=int,   default=50,  help="MLP AE epochs")
    parser.add_argument("--lr",            type=float, default=1e-3)

    # Pose hyperparams
    parser.add_argument("--pose-detector",    default="yolov8n-pose.pt")
    parser.add_argument("--pose-conf",        type=float, default=0.5)
    parser.add_argument("--gcae-epochs",      type=int,   default=30)
    parser.add_argument("--shopformer-epochs",type=int,   default=20)
    parser.add_argument("--shopformer-lr",    type=float, default=5e-5)

    # Device
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()
    main(args)
