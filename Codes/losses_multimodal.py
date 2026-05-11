# -*- coding: utf-8 -*-
"""
losses_multimodal.py

Clean multi-task losses for DNA damage detection:
- Classification: BCE or Focal
- Position regression: SmoothL1 / MSE (positives only)
- Cross-modal alignment: configurable contrastive objectives

Design goals:
- Single compute_loss entry point
- Modular contrastive losses
- DDP-safe (zero-loss fallbacks)
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Classification losses
# =============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs = torch.sigmoid(logits)

        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = focal_weight * bce

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


# =============================================================================
# Position loss
# =============================================================================

class PositionLoss(nn.Module):
    """
    Regression loss for damage position.
    Applied ONLY to valid damaged samples.
    """

    def __init__(self, loss_type: str = "smoothl1", beta: float = 0.05):
        super().__init__()
        self.loss_type = loss_type
        self.beta = float(beta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.numel() == 0:
            # DDP-safe zero
            return pred.sum() * 0.0

        if self.loss_type == "mse":
            return F.mse_loss(pred, target)
        else:
            return F.smooth_l1_loss(pred, target, beta=self.beta)


# =============================================================================
# Contrastive losses
# =============================================================================

class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = float(temperature)

    def forward(
        self,
        seq_emb: torch.Tensor,
        sig_emb: torch.Tensor,
        labels: torch.Tensor = None,  # <-- accept labels
        **kwargs
    ) -> torch.Tensor:
        sim = torch.matmul(seq_emb, sig_emb.T) / self.temperature
        targets = torch.arange(sim.size(0), device=sim.device)
        return 0.5 * (
            F.cross_entropy(sim, targets) +
            F.cross_entropy(sim.T, targets)
        )

class LabelAwareContrastiveLoss(nn.Module):
    """
    Positives = same damage label
    Negatives = different label
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, seq_emb, sig_emb, labels, **kwargs):
        B = seq_emb.size(0)
        if B < 2:
            return seq_emb.sum() * 0.0

        sim = torch.matmul(seq_emb, sig_emb.T) / self.temperature

        labels = labels.view(-1, 1)
        same_label = (labels == labels.T).float()
        self_mask = torch.eye(B, device=sim.device, dtype=sim.dtype)
        
        exp_sim = torch.exp(sim - sim.max(dim=1, keepdim=True)[0].detach())
        pos_mask = same_label * (1 - self_mask)
        pos_count = pos_mask.sum(dim=1)
        valid = pos_count > 0

        if not valid.any():
            targets = torch.arange(B, device=sim.device)
            return 0.5 * (
                F.cross_entropy(sim, targets) +
                F.cross_entropy(sim.T, targets)
            )

        pos_sum = (exp_sim * pos_mask).sum(dim=1)
        all_sum = (exp_sim * (1 - self_mask)).sum(dim=1)

        loss = -torch.log(pos_sum / (all_sum + 1e-8) + 1e-8)
        return (loss * valid.float()).sum() / valid.sum()


class DamageFocusedContrastiveLoss(nn.Module):
    """
    RECOMMENDED:
    Only aligns damaged samples (label==1).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, seq_emb, sig_emb, labels, **kwargs):
        mask = labels > 0
        if mask.sum() < 2:
            return seq_emb.sum() * 0.0

        seq = seq_emb[mask]
        sig = sig_emb[mask]
        sim = torch.matmul(seq, sig.T) / self.temperature
        targets = torch.arange(sim.size(0), device=sim.device)

        return 0.5 * (
            F.cross_entropy(sim, targets) +
            F.cross_entropy(sim.T, targets)
        )


class HardNegativeContrastiveLoss(nn.Module):
    """
    Contrastive loss with hard negative mining.
    """

    def __init__(self, temperature: float = 0.07, hard_negative_weight: float = 0.5):
        super().__init__()
        self.temperature = float(temperature)
        self.hard_negative_weight = float(hard_negative_weight)

    def forward(self, seq_emb, sig_emb, labels, **kwargs):
        B = seq_emb.size(0)
        if B < 2:
            return seq_emb.sum() * 0.0

        sim = torch.matmul(seq_emb, sig_emb.T) / self.temperature

        labels_col = labels.view(-1, 1).float()
        labels_row = labels.view(1, -1).float()
        diff_label = (labels_col != labels_row).float()

        self_mask = torch.eye(B, device=sim.device, dtype=sim.dtype)

        neg_sim = sim * diff_label - 1e9 * (1 - diff_label)
        hardest_neg = neg_sim.max(dim=1)[0]

        pos_sim = sim.diag()

        loss = -torch.log(
            torch.exp(pos_sim) /
            (torch.exp(pos_sim) + self.hard_negative_weight * torch.exp(hardest_neg) + 1e-8)
        )

        return loss.mean()


# =============================================================================
# Contrastive loss factory
# =============================================================================

def get_contrastive_loss(loss_type: str, temperature: float) -> Optional[nn.Module]:
    if loss_type in (None, "none"):
        return None
    if loss_type == "infonce":
        return InfoNCELoss(temperature)
    if loss_type == "label_aware":
        return LabelAwareContrastiveLoss(temperature)
    if loss_type == "damage_focused":
        return DamageFocusedContrastiveLoss(temperature)
    if loss_type == "hard_negative":
        return HardNegativeContrastiveLoss(temperature)
    raise ValueError(f"Unknown contrastive loss: {loss_type}")


# =============================================================================
# Unified loss interface
# =============================================================================

def compute_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    damage_positions: torch.Tensor,
    cls_weight: float = 1.0,
    pos_weight: float = 0.5,
    contrastive_weight: float = 0.1,
    use_focal: bool = True,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    pos_loss_type: str = "smoothl1",
    pos_beta: float = 0.05,
    temperature: float = 0.07,
    contrastive_type: str = "damage_focused",
) -> Tuple[torch.Tensor, Dict[str, float]]:

    logits = outputs["binary_logits"]
    pos_pred = outputs["pos_pred"]
    seq_emb = outputs["seq_emb"]
    sig_emb = outputs["sig_emb"]

    # ---- classification ----
    if use_focal:
        cls_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        loss_cls = cls_fn(logits, labels)
    else:
        loss_cls = F.binary_cross_entropy_with_logits(
            logits, labels.float()
        )

    # ---- position (positives only) ----
    pos_mask = (labels > 0) & (damage_positions >= 0)
    if pos_mask.any():
        loss_pos = PositionLoss(loss_type=pos_loss_type, beta=pos_beta)(
            pos_pred[pos_mask],
            damage_positions[pos_mask],
        )
    else:
        loss_pos = pos_pred.sum() * 0.0

    # ---- contrastive ----
    con_fn = get_contrastive_loss(contrastive_type, temperature)
    if contrastive_weight > 0 and con_fn is not None:
        loss_con = con_fn(seq_emb, sig_emb, labels)
    else:
        loss_con = seq_emb.sum() * 0.0

    total = (
        cls_weight * loss_cls +
        pos_weight * loss_pos +
        contrastive_weight * loss_con
    )

    return total, {
        "loss_total": float(total.detach()),
        "loss_cls": float(loss_cls.detach()),
        "loss_pos": float(loss_pos.detach()),
        "loss_contrast": float(loss_con.detach()),
    }
