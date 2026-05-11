# -*- coding: utf-8 -*-
"""
model_multimodal.py

Multimodal model (improved, drop-in):
- Sequence encoder = (local pretrained checkpoint) + (local LoRA adapter) -> merged -> frozen
- Signal encoder = CNN + BiLSTM (captures temporal asymmetry)
- Fusion = gated fusion with temperature-scaled softmax (more stable)
- Heads:
    - binary damage (logit)
    - damage position regression in [0,1] (sigmoid)
- Optional: per-sample signal normalization, local token pooling for seq

Offline-safe:
- local_files_only=True
- trust_remote_code=False
- never loads any remote repo name
"""

from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

try:
    from peft import PeftModel
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False


def _assert_local_dir(path: str, what: str):
    if path is None or len(path) == 0:
        raise ValueError(f"{what} is empty")
    if "zhihan1996" in path or "DNABERT-2-117M" in path:
        raise ValueError(
            f"[FATAL] {what} points to an HF repo string ({path}). "
            f"Must be a LOCAL directory path."
        )


def load_frozen_merged_seq_encoder(
    pretrained_dir: str,
    adapter_dir: str,
    device: torch.device,
) -> nn.Module:
    """
    Load base model from local pretrained_dir, then load LoRA adapter from local adapter_dir,
    merge adapter into base, freeze everything.
    """
    _assert_local_dir(pretrained_dir, "pretrained_dir")
    _assert_local_dir(adapter_dir, "adapter_dir")
    if not HAS_PEFT:
        raise ImportError("peft is required: pip install peft")

    base = AutoModel.from_pretrained(
        pretrained_dir,
        local_files_only=True,
        trust_remote_code=False,
    ).to(device)

    model = PeftModel.from_pretrained(
        base,
        adapter_dir,
        local_files_only=True,
    )

    # Merge LoRA into base weights and remove adapter modules
    model = model.merge_and_unload()
    model.eval()

    # Freeze
    for p in model.parameters():
        p.requires_grad = False

    return model


# =============================================================================
# Signal encoder (CNN)
# =============================================================================
class SignalEncoder(nn.Module):
    def __init__(
        self,
        signal_len: int,
        out_dim: int,
        lstm_hidden: int = 128,
        use_lstm: bool = True,
        use_residual: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_lstm = use_lstm
        self.use_residual = use_residual

        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        if use_residual:
            self.res_proj = nn.Conv1d(32, 64, kernel_size=1)

        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=64,
                hidden_size=lstm_hidden,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )
            self.lstm_dropout = nn.Dropout(dropout)
            proj_in_dim = 2 * lstm_hidden
        else:
            self.lstm = None
            proj_in_dim = 64

        self.proj = nn.Sequential(
            nn.Linear(proj_in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        x = signal.unsqueeze(1)              # (B, 1, L)
        x1 = self.act(self.conv1(x))          # (B, 32, L)
        x2 = self.act(self.conv2(x1))         # (B, 64, L)

        if self.use_residual:
            x2 = x2 + self.res_proj(x1)

        x2 = self.dropout(x2)                 # dropout now active

        if self.use_lstm:
            x2 = x2.transpose(1, 2)           # (B, L, 64)
            lstm_out, _ = self.lstm(x2)
            lstm_out = self.lstm_dropout(lstm_out)
            pooled = lstm_out.mean(dim=1)
        else:
            pooled = x2.mean(dim=-1)

        return self.proj(pooled)


# =============================================================================
# Main multimodal model
# =============================================================================

class MultimodalDNADamageModel(nn.Module):
    """
    Outputs dict keys (stable for train/inference/interpretability):
      - binary_logits: (B,)
      - pos_pred: (B,) in [0,1]
      - fused_emb: (B, D)
      - seq_emb: (B, D) normalized
      - sig_emb: (B, D) normalized
      - fusion_attention: (B, 2)
      - sequence_attention: (optional)
    """

    def __init__(
        self,
        pretrained_dir: str,
        adapter_dir: str,
        max_signal_len: int = 256,
        fusion_dim: int = 256,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,

        # --- fusion & pooling ---
        fusion_temperature: float = 1.0,
        use_signal_layernorm: bool = False,
        use_local_seq_pool: bool = True,
        local_pool_tokens: int = 16,

        # --- signal encoder toggles ---
        use_lstm: bool = True,
        use_residual: bool = False,
        lstm_hidden: int = 128,
        freeze_seq_encoder: bool = True,
    ):
        super().__init__()

        self.fusion_temperature = fusion_temperature
        self.use_signal_layernorm = use_signal_layernorm
        self.use_local_seq_pool = use_local_seq_pool
        self.local_pool_tokens = local_pool_tokens
        self.dropout = dropout

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ============================
        # Frozen sequence encoder
        # ============================
        self.seq_encoder = load_frozen_merged_seq_encoder(
            pretrained_dir=pretrained_dir,
            adapter_dir=adapter_dir,
            device=device,
        )
        
        if freeze_seq_encoder:
            for param in self.seq_encoder.parameters():
                param.requires_grad = False

        hidden = getattr(self.seq_encoder.config, "hidden_size", None)
        if hidden is None:
            raise ValueError("Cannot infer hidden_size from seq_encoder.config")

        self.seq_proj = nn.Sequential(
            nn.Linear(hidden, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        # ============================
        # Signal encoder (CNN + optional BiLSTM)
        # ============================
        self.sig_encoder = SignalEncoder(
            signal_len=max_signal_len,
            out_dim=fusion_dim,
            lstm_hidden=lstm_hidden,
            use_lstm=use_lstm,
            use_residual=use_residual,
            dropout=dropout,
        )

        # ============================
        # Fusion gate
        # ============================
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 2),
        )

        # ============================
        # Task adapters
        # ============================
        self.cls_adapter = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pos_adapter = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ============================
        # Heads
        # ============================
        self.cls_head = nn.Linear(fusion_dim, 1)
        self.pos_head = nn.Linear(fusion_dim, 1)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _maybe_norm_signal(self, signal: torch.Tensor) -> torch.Tensor:
        if not self.use_signal_layernorm:
            return signal
        mean = signal.mean(dim=1, keepdim=True)
        std = signal.std(dim=1, keepdim=True).clamp_min(1e-6)
        return (signal - mean) / std

    def _pool_sequence(self, seq_out) -> torch.Tensor:
        h = seq_out.last_hidden_state  # (B, T, H)
        cls = h[:, 0]

        if not self.use_local_seq_pool:
            return cls

        T = h.size(1)
        k = max(1, min(self.local_pool_tokens, T - 1))
        local = h[:, 1:1 + k].mean(dim=1)
        return cls + local

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        signal: torch.Tensor,
        output_attentions: bool = False,
        mode: str = "fused",
    ) -> Dict[str, torch.Tensor]:

        # ----- Sequence -----
        with torch.no_grad():
            seq_out = self.seq_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                return_dict=True,
            )
            pooled = self._pool_sequence(seq_out)

        seq_emb = self.seq_proj(pooled)

        sequence_attention = None
        if output_attentions and getattr(seq_out, "attentions", None):
            sequence_attention = seq_out.attentions[-1].mean(dim=1)

        # ----- Signal -----
        signal = self._maybe_norm_signal(signal)
        sig_emb = self.sig_encoder(signal)

        # ----- Fusion -----
        if mode == "seq_only":
            fused = seq_emb
            fusion_w = torch.tensor([[1.0, 0.0]], device=seq_emb.device).repeat(seq_emb.size(0), 1)

        elif mode == "signal_only":
            fused = sig_emb
            fusion_w = torch.tensor([[0.0, 1.0]], device=sig_emb.device).repeat(sig_emb.size(0), 1)

        else:
            gate_logits = self.gate(torch.cat([seq_emb, sig_emb], dim=1))
            tau = max(self.fusion_temperature, 1e-6)
            fusion_w = F.softmax(gate_logits / tau, dim=1)
            fused = fusion_w[:, 0:1] * seq_emb + fusion_w[:, 1:2] * sig_emb

        # ----- Heads -----
        cls_feat = self.cls_adapter(fused)
        pos_feat = self.pos_adapter(fused)

        binary_logits = self.cls_head(cls_feat).squeeze(-1)
        pos_pred = torch.sigmoid(self.pos_head(pos_feat).squeeze(-1))

        return {
            "binary_logits": binary_logits,
            "pos_pred": pos_pred,
            "fused_emb": fused,
            "seq_emb": F.normalize(seq_emb, dim=1),
            "sig_emb": F.normalize(sig_emb, dim=1),
            "fusion_attention": fusion_w,
            "sequence_attention": sequence_attention,
        }

