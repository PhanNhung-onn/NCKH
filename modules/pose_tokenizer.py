"""
modules/pose_tokenizer.py
=========================
GCAE-based Pose Tokenizer + Shopformer Transformer.

Kiến trúc theo paper:
    Shopformer: Transformer-Based Framework for Detecting Shoplifting
    via Human Pose — CVPR Workshops 2025
    github.com/TeCSAR-UNCC/Shopformer

Pipeline:
    Pose sequence (T=12, 17 kpts, 3 ch)
         ↓  GCAEEncoder (ST-GCN)
    2 tokens × 144-D                ← optimal theo ablation study
         ↓  Shopformer Transformer (encoder-decoder)
    Reconstruction MSE              ← anomaly score

Hai giai đoạn train (xem training_pipeline.py):
    Stage 1 — Train GCAE toàn bộ (encoder + decoder) unsupervised
    Stage 2 — Freeze encoder, train Transformer trên normal sequences

Cách dùng inference:
    from modules.pose_tokenizer import PoseTokenizer
    tok = PoseTokenizer()
    tok.load("models/pose_tokenizer.pt")
    tokens = tok.update(track_id, kpts_17x3)    # None cho đến khi đủ 12 frame
    score  = tok.anomaly_score(tokens)           # MSE reconstruction error
"""

from __future__ import annotations

import logging
import math
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Hằng số theo paper ───────────────────────────────────────────────
SEQ_LEN   = 12    # T: số frame mỗi sequence
N_TOKENS  = 2     # optimal theo ablation (AUC-ROC 69.15%)
EMBED_DIM = 144   # token embedding size (8 channels × 17 kpts ≈ 136 → pad 144)
N_KPT     = 17    # COCO keypoints
IN_CH     = 3     # x, y, confidence

# Kết nối xương người (COCO 17 keypoints)
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # đầu
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # tay
    (5, 11), (6, 12), (11, 12),                  # thân
    (11, 13), (13, 15), (12, 14), (14, 16),     # chân
]


# ══════════════════════════════════════════════════════════════════════
# BUILD ADJACENCY MATRIX
# ══════════════════════════════════════════════════════════════════════

def _build_adj(K: int = N_KPT) -> torch.Tensor:
    """Normalized adjacency matrix cho skeleton graph."""
    A = torch.eye(K)
    for i, j in COCO_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    D = A.sum(dim=1, keepdim=True).clamp(min=1.0)
    return A / D


# ══════════════════════════════════════════════════════════════════════
# ST-GCN BLOCK
# ══════════════════════════════════════════════════════════════════════

class STGCNBlock(nn.Module):
    """
    Spatial-Temporal Graph Convolutional block.
    Spatial GCN → Temporal Conv → Residual.
    """

    def __init__(self, in_ch: int, out_ch: int, A: torch.Tensor):
        super().__init__()
        self.register_buffer("A", A)
        self.gcn  = nn.Linear(in_ch, out_ch, bias=False)
        self.tcn  = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.res  = (
            nn.Linear(in_ch, out_ch, bias=False)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, K, C)
        B, T, K, C = x.shape
        A = self.A  # (K, K)

        # ── Spatial GCN ───────────────────────────────────────────
        xr = x.reshape(B * T, K, C)
        xs = torch.bmm(
            A.unsqueeze(0).expand(B * T, -1, -1), xr
        )                                           # (B*T, K, C)
        xs = self.gcn(xs)                           # (B*T, K, out_ch)
        xs = xs.reshape(B, T, K, -1)

        # ── Temporal Conv (per joint) ─────────────────────────────
        out_ch = xs.shape[-1]
        xt = xs.permute(0, 2, 3, 1).reshape(B * K, out_ch, T)
        xt = self.tcn(xt)                           # (B*K, out_ch, T)
        xt = self.bn(xt)
        xt = xt.reshape(B, K, out_ch, T).permute(0, 3, 1, 2)  # (B, T, K, out_ch)

        # ── Residual ─────────────────────────────────────────────
        res = self.res(x)
        return self.relu(xt + res)


# ══════════════════════════════════════════════════════════════════════
# GCAE ENCODER
# ══════════════════════════════════════════════════════════════════════

class GCAEEncoder(nn.Module):
    """
    Encoder của Graph Convolutional Autoencoder.

    Input:  (B, T, K, in_ch)
    Output: (B, N_tokens, embed_dim)
    """

    def __init__(
        self,
        in_ch:     int = IN_CH,
        n_tokens:  int = N_TOKENS,
        embed_dim: int = EMBED_DIM,
    ):
        super().__init__()
        self.n_tokens  = n_tokens
        self.embed_dim = embed_dim

        A = _build_adj()
        self.blocks = nn.Sequential(
            STGCNBlock(in_ch, 8,  A),
            STGCNBlock(8,     16, A),
        )
        # (K × 16) → embed_dim per token
        self.proj = nn.Linear(N_KPT * 16, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, K, in_ch)
        x = self.blocks(x)                  # (B, T, K, 16)
        B, T, K, C = x.shape

        seg = max(1, T // self.n_tokens)
        tokens = []
        for i in range(self.n_tokens):
            s = i * seg
            e = s + seg if i < self.n_tokens - 1 else T
            seg_feat = x[:, s:e, :, :]     # (B, seg, K, 16)
            pooled   = seg_feat.mean(dim=1) # (B, K, 16)
            flat     = pooled.reshape(B, -1)
            token    = self.proj(flat)      # (B, embed_dim)
            tokens.append(token)

        return torch.stack(tokens, dim=1)   # (B, N_tokens, embed_dim)


# ══════════════════════════════════════════════════════════════════════
# GCAE DECODER
# ══════════════════════════════════════════════════════════════════════

class GCAEDecoder(nn.Module):
    """
    Decoder của GCAE — dùng trong Stage 1 training.

    Input:  (B, N_tokens, embed_dim)
    Output: (B, T, K, in_ch)
    """

    def __init__(
        self,
        in_ch:     int = IN_CH,
        n_tokens:  int = N_TOKENS,
        embed_dim: int = EMBED_DIM,
        seq_len:   int = SEQ_LEN,
    ):
        super().__init__()
        self.seq_len  = seq_len
        self.n_tokens = n_tokens

        A = _build_adj()
        self.proj = nn.Linear(embed_dim, N_KPT * 16)
        self.blocks = nn.Sequential(
            STGCNBlock(16, 8,    A),
            STGCNBlock(8,  in_ch, A),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, N_tokens, embed_dim)
        B = tokens.shape[0]
        seg = self.seq_len // self.n_tokens

        frames = []
        for i in range(self.n_tokens):
            t = tokens[:, i, :]             # (B, embed_dim)
            f = self.proj(t)                # (B, K*16)
            f = f.reshape(B, 1, N_KPT, 16) # (B, 1, K, 16)
            f = f.expand(-1, seg, -1, -1)   # (B, seg, K, 16)
            frames.append(f)

        x = torch.cat(frames, dim=1)        # (B, T, K, 16)
        x = self.blocks(x)                  # (B, T, K, in_ch)
        return x


# ══════════════════════════════════════════════════════════════════════
# SHOPFORMER TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class ShopformerTransformer(nn.Module):
    """
    Transformer encoder-decoder xử lý pose tokens.

    Optimal config theo ablation study (Table S3):
        n_layers=2, n_heads=2, ff_dim=64, embed_dim=144

    Train trên normal sequences → reconstruction error thấp.
    Shoplifting → MSE cao → anomaly score cao.
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        n_tokens:  int = N_TOKENS,
        n_layers:  int = 2,
        n_heads:   int = 2,
        ff_dim:    int = 64,
        dropout:   float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_tokens  = n_tokens

        # Positional encoding
        self.pos_enc = nn.Parameter(
            torch.zeros(1, n_tokens, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_enc, std=0.02)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Transformer decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)

    def forward(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (B, N_tokens, embed_dim)
        Returns:
            recon:  (B, N_tokens, embed_dim) — reconstructed tokens
            memory: (B, N_tokens, embed_dim) — encoder output (latent)
        """
        x      = tokens + self.pos_enc
        memory = self.encoder(x)
        recon  = self.decoder(x, memory)
        return recon, memory

    def reconstruction_error(self, tokens: torch.Tensor) -> torch.Tensor:
        """MSE giữa tokens gốc và tokens tái tạo — dùng làm anomaly score."""
        recon, _ = self(tokens)
        return nn.functional.mse_loss(recon, tokens, reduction="none").mean(dim=(1, 2))


# ══════════════════════════════════════════════════════════════════════
# POSE TOKENIZER — HIGH-LEVEL WRAPPER
# ══════════════════════════════════════════════════════════════════════

class PoseTokenizer:
    """
    High-level wrapper tích hợp GCAEEncoder + ShopformerTransformer.

    Luồng:
        1. Nhận keypoints (17, 3) mỗi frame qua update()
        2. Khi đủ SEQ_LEN frame → GCAEEncoder tạo ra 2 tokens
        3. ShopformerTransformer tính MSE reconstruction error
        4. MSE → anomaly_score (sigmoid normalize)

    Stateful: duy trì buffer per track_id giữa các frame.
    """

    def __init__(
        self,
        seq_len:   int   = SEQ_LEN,
        n_tokens:  int   = N_TOKENS,
        embed_dim: int   = EMBED_DIM,
        device:    str   = "cpu",
    ):
        self.seq_len  = seq_len
        self.device   = device

        self.encoder     = GCAEEncoder(n_tokens=n_tokens, embed_dim=embed_dim).to(device)
        self.transformer = ShopformerTransformer(embed_dim=embed_dim, n_tokens=n_tokens).to(device)

        self.encoder.eval()
        self.transformer.eval()

        # Buffer per track_id: deque of (17, 3) arrays
        self._bufs: Dict[int, deque] = {}

        # Score history per track (smooth bằng rolling mean)
        self._score_hist: Dict[int, deque] = {}

    # ------------------------------------------------------------------
    def update(
        self, track_id: int, kpts: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Nhận (17, 3) keypoints của 1 frame.
        Khi đủ seq_len → trả về token array (N_tokens, embed_dim).
        Chưa đủ → trả về None.
        """
        if track_id not in self._bufs:
            self._bufs[track_id] = deque(maxlen=self.seq_len)
        self._bufs[track_id].append(kpts.astype(np.float32))

        if len(self._bufs[track_id]) < self.seq_len:
            return None

        seq = np.stack(self._bufs[track_id])  # (T, 17, 3)
        t   = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        # t: (1, T, 17, 3) → encoder expects (B, T, K, C)

        with torch.no_grad():
            tokens = self.encoder(t)           # (1, N_tokens, embed_dim)

        return tokens.squeeze(0).cpu().numpy() # (N_tokens, embed_dim)

    # ------------------------------------------------------------------
    def anomaly_score(
        self,
        tokens: np.ndarray,
        track_id: Optional[int] = None,
        smooth_window: int = 5,
    ) -> float:
        """
        Tính anomaly score từ tokens (N_tokens, embed_dim).

        Args:
            tokens:        output từ update()
            track_id:      nếu cung cấp → smooth score theo lịch sử
            smooth_window: số score gần nhất để lấy trung bình

        Returns:
            score ∈ [0, 1] — cao = bất thường
        """
        t = torch.tensor(tokens, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mse = self.transformer.reconstruction_error(t)  # (1,)
        raw_score = float(mse.item())

        # Normalize: sigmoid với scale factor
        # MSE trên normal data thường < 0.1; anomaly > 0.3
        score = 1.0 / (1.0 + np.exp(-(raw_score - 0.15) * 20))
        score = float(np.clip(score, 0.0, 1.0))

        # Smooth theo lịch sử track
        if track_id is not None:
            if track_id not in self._score_hist:
                self._score_hist[track_id] = deque(maxlen=smooth_window)
            self._score_hist[track_id].append(score)
            score = float(np.mean(self._score_hist[track_id]))

        return score

    # ------------------------------------------------------------------
    def clear_track(self, track_id: int):
        """Xoá buffer khi track bị mất."""
        self._bufs.pop(track_id, None)
        self._score_hist.pop(track_id, None)

    # ------------------------------------------------------------------
    def load(self, path: str):
        """
        Load weights đã train từ file.
        File .pt chứa dict với 2 key: 'encoder' và 'transformer'.
        """
        p = Path(path)
        if not p.exists():
            logger.warning(f"PoseTokenizer: {path} not found — dùng untrained weights.")
            return
        ckpt = torch.load(path, map_location=self.device)
        if "encoder" in ckpt:
            self.encoder.load_state_dict(ckpt["encoder"])
        if "transformer" in ckpt:
            self.transformer.load_state_dict(ckpt["transformer"])
        self.encoder.eval()
        self.transformer.eval()
        logger.info(f"PoseTokenizer loaded from {path}")

    def save(self, path: str):
        """Lưu weights encoder + transformer."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "encoder":     self.encoder.state_dict(),
            "transformer": self.transformer.state_dict(),
        }, path)
        logger.info(f"PoseTokenizer saved → {path}")


# ══════════════════════════════════════════════════════════════════════
# GCAE FULL MODEL (dùng trong Stage 1 training)
# ══════════════════════════════════════════════════════════════════════

class GCAEFull(nn.Module):
    """
    Autoencoder hoàn chỉnh (encoder + decoder) cho Stage 1.
    Train để tái tạo pose sequences → encoder học representation tốt.
    """

    def __init__(self, in_ch=IN_CH, n_tokens=N_TOKENS,
                 embed_dim=EMBED_DIM, seq_len=SEQ_LEN):
        super().__init__()
        self.encoder = GCAEEncoder(in_ch, n_tokens, embed_dim)
        self.decoder = GCAEDecoder(in_ch, n_tokens, embed_dim, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, K, in_ch)
        tokens = self.encoder(x)    # (B, N_tokens, embed_dim)
        recon  = self.decoder(tokens)   # (B, T, K, in_ch)
        return recon
