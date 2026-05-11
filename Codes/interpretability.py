# -*- coding: utf-8 -*-
"""
interpretability.py

Interpretability utilities for MultimodalDNADamageModel.

Adds a light-but-useful set of analyses that are stable in practice:
1) Fusion gate reliance (w_seq / w_sig) distribution and relation to confidence.
2) Cross-modal swap tests (sequence/signal) to test whether fusion is causal.
3) Integrated Gradients (signal) for classification head (and optionally position head).
4) Token-level attributions on the *sequence encoder* via Integrated Gradients on inputs_embeds.
5) Optional in-silico mutagenesis on raw sequences (character-level) as a robust sanity-check.

Notes
- The training model forward() wraps the sequence encoder in torch.no_grad().
  For sequence attributions we therefore bypass model.forward() and run the
  sequence encoder directly with gradients enabled.
- Designed to be offline-safe (no web, no remote repos).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, List

import os
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Helpers
# =============================================================================

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _safe_sigmoid(x: torch.Tensor) -> torch.Tensor:
    # stable sigmoid for float16-ish logits
    return torch.sigmoid(x.float()).to(x.dtype)

def _pick_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")

def _default_baseline_signal(signal: torch.Tensor) -> torch.Tensor:
    # baseline = zeros (works well when signals are normalized)
    return torch.zeros_like(signal)

def _sequence_pool_from_last_hidden(model, last_hidden_state: torch.Tensor) -> torch.Tensor:
    """
    Replicate model's sequence pooling behavior without depending on model.forward().
    Uses:
      - CLS token always
      - optional local mean pooling if model.use_local_seq_pool is True (tokens 1..K)
    """
    h = last_hidden_state  # (B, T, H)
    cls = h[:, 0]
    use_local = bool(getattr(model, "use_local_seq_pool", False))
    if not use_local:
        return cls

    k_req = int(getattr(model, "local_pool_tokens", 16))
    T = h.size(1)
    k = max(1, min(k_req, T - 1))
    local = h[:, 1:1 + k].mean(dim=1)
    return cls + local


# =============================================================================
# 1) Fusion gate reliance
# =============================================================================

@torch.no_grad()
def collect_fusion_gate_stats(
    model,
    loader,
    mode: str = "fused",
    max_batches: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Collect fusion weights and probabilities across a loader.
    Returns arrays: w_seq, w_sig, prob, label (if available)
    """
    device = _pick_device(model)
    model.eval()

    w_seq, w_sig, prob, label = [], [], [], []
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        signal = batch["signal"].to(device)
        y = batch.get("label", None)
        if y is not None:
            y = y.to(device)

        out = model(input_ids, attn, signal, output_attentions=False, mode=mode)
        p = _safe_sigmoid(out["binary_logits"])

        fw = out.get("fusion_attention", None)
        if fw is None:
            continue

        fw = fw.float()
        w_seq.append(fw[:, 0].detach().cpu())
        w_sig.append(fw[:, 1].detach().cpu())
        prob.append(p.detach().cpu())
        if y is not None:
            label.append(y.detach().cpu())

    out_dict = {
        "w_seq": torch.cat(w_seq).numpy() if w_seq else np.array([]),
        "w_sig": torch.cat(w_sig).numpy() if w_sig else np.array([]),
        "prob": torch.cat(prob).numpy() if prob else np.array([]),
    }
    if label:
        out_dict["label"] = torch.cat(label).numpy()
    return out_dict


def plot_fusion_gate_reliance(stats: Dict[str, np.ndarray], out_dir: str, prefix: str = "fusion"):
    _ensure_dir(out_dir)
    w_sig = stats.get("w_sig", np.array([]))
    prob = stats.get("prob", np.array([]))
    label = stats.get("label", None)

    if w_sig.size == 0 or prob.size == 0:
        return

    # Histogram of w_sig
    plt.figure()
    plt.hist(w_sig, bins=40)
    plt.xlabel("Fusion gate weight on signal (w_sig)")
    plt.ylabel("Count")
    plt.title("Fusion gate reliance distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_w_sig_hist.svg"), dpi=200)
    plt.close()

    # w_sig vs confidence (prob)
    plt.figure()
    if label is None:
        plt.scatter(w_sig, prob, s=6, alpha=0.5)
    else:
        mask1 = label == 1
        mask0 = label == 0
        plt.scatter(w_sig[mask0], prob[mask0], s=6, alpha=0.4, label="undamaged")
        plt.scatter(w_sig[mask1], prob[mask1], s=6, alpha=0.4, label="damaged")
        plt.legend(frameon=False)

    plt.xlabel("w_sig")
    plt.ylabel("P(damage)")
    plt.title("Modality reliance vs prediction confidence")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_w_sig_vs_prob.svg"), dpi=200)
    plt.close()


# =============================================================================
# 2) Cross-modal swap tests
# =============================================================================

# @torch.no_grad()
# def cross_modal_swap_test(
#     model,
#     batch: Dict[str, torch.Tensor],
#     swap: str = "signal",   # "signal" | "sequence"
#     n_perm: int = 1,
#     mode: str = "fused",
# ) -> Dict[str, float]:
#     """
#     Swap one modality within the batch and measure prediction changes.
#     Returns mean absolute probability shift.
#     """
#     device = _pick_device(model)
#     model.eval()

#     input_ids = batch["input_ids"].to(device)
#     attn = batch["attention_mask"].to(device)
#     signal = batch["signal"].to(device)

#     out0 = model(input_ids, attn, signal, output_attentions=False, mode=mode)
#     p0 = _safe_sigmoid(out0["binary_logits"]).float()

#     shifts = []
#     B = input_ids.size(0)
#     for _ in range(max(1, n_perm)):
#         perm = torch.randperm(B, device=device)
#         if swap == "signal":
#             signal_sw = signal[perm]
#             out = model(input_ids, attn, signal_sw, output_attentions=False, mode=mode)
#         else:
#             input_sw = input_ids[perm]
#             attn_sw = attn[perm]
#             out = model(input_sw, attn_sw, signal, output_attentions=False, mode=mode)

#         p = _safe_sigmoid(out["binary_logits"]).float()
#         shifts.append((p - p0).abs().mean().item())

#     return {
#         "swap": swap,
#         "mean_abs_prob_shift": float(np.mean(shifts)),
#     }

@torch.no_grad()
def cross_modal_swap_test(
    model,
    batch,
    swap="signal",
    n_perm=1,
    mode="fused",
):
    device = _pick_device(model)
    model.eval()

    input_ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    signal = batch["signal"].to(device)

    out0 = model(input_ids, attn, signal, output_attentions=False, mode=mode)
    p0 = _safe_sigmoid(out0["binary_logits"]).float()

    shifts = []
    B = input_ids.size(0)

    for _ in range(max(1, n_perm)):
        perm = torch.randperm(B, device=device)

        if swap == "signal":
            signal_sw = signal[perm]
            out = model(input_ids, attn, signal_sw, output_attentions=False, mode=mode)
        else:
            input_sw = input_ids[perm]
            attn_sw = attn[perm]
            out = model(input_sw, attn_sw, signal, output_attentions=False, mode=mode)

        p = _safe_sigmoid(out["binary_logits"]).float()

        shifts.append((p - p0).abs().detach().cpu().numpy())

    return {
        "swap": swap,
        "shifts": np.concatenate(shifts),
    }


# =============================================================================
# 3) Integrated Gradients on signal
# =============================================================================

def integrated_gradients_signal(
    model,
    input_ids,
    attention_mask,
    signal,
    baseline=None,
    steps: int = 50,
    target: str = "binary",
    mode: str = "fused",
):
    """
    Integrated Gradients for the signal input.

    Returns:
      ig: np.ndarray shaped like signal (without batch dim)
      p_bin: float P(damage) on the *original* input (for display)
    """
    device = signal.device

    # IMPORTANT: keep model in eval for determinism, but LSTM backward needs training mode
    model.eval()

    if baseline is None:
        baseline = torch.zeros_like(signal)

    # ----- prediction on original input (for reporting) -----
    with torch.no_grad():
        out0 = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            signal=signal,
            mode=mode,
        )
        p_bin = float(torch.sigmoid(out0["binary_logits"].float()).mean().item())

    # ----- IG accumulation -----
    # cuDNN RNN backward requires training mode
    was_training = model.sig_encoder.training
    model.sig_encoder.train()

    try:
        grads = []
        # interpolate from baseline -> signal
        for i in range(1, steps + 1):
            alpha = float(i) / float(steps)
            s = (baseline + alpha * (signal - baseline)).detach().clone().requires_grad_(True)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                signal=s,
                mode=mode,
            )

            if target == "binary":
                score = out["binary_logits"].sum()
            else:
                # pos head is sigmoid already in your model output
                score = out["pos_pred"].sum()

            model.zero_grad(set_to_none=True)
            score.backward()
            grads.append(s.grad.detach())

        avg_grad = torch.mean(torch.stack(grads, dim=0), dim=0)
        ig = (signal - baseline) * avg_grad

    finally:
        # restore original mode
        if was_training:
            model.sig_encoder.train()
        else:
            model.sig_encoder.eval()

    return ig.squeeze(0).detach().cpu().numpy(), p_bin



def plot_signal_ig(signal: np.ndarray, ig: np.ndarray, out_path: str, title: str):
    plt.figure(figsize=(10, 3))
    plt.plot(signal, linewidth=1)
    # Scale IG for overlay visibility (robust scaling)
    scale = np.percentile(np.abs(ig), 95) + 1e-8
    overlay = ig / scale
    plt.plot(overlay, linewidth=1)
    plt.title(title)
    plt.xlabel("Time index")
    plt.ylabel("Signal / scaled IG")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# =============================================================================
# 4) Token-level IG via inputs_embeds
# =============================================================================

def integrated_gradients_sequence_tokens(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    signal: torch.Tensor,
    target: str = "binary",          # "binary" | "pos"
    steps: int = 32,
    mode: str = "fused",
):
    """
    Token IG on the sequence encoder by interpolating input embeddings.
    IMPORTANT: We DETACH the signal branch so we never backprop through cuDNN RNN.
    Returns:
      - token_attrib: (T,) numpy
      - pred_val: float (P(damage) or pos)
    """
    device = _pick_device(model)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    signal = signal.to(device)

    # If user asks token IG but inference is signal-only, attribution is undefined.
    if mode == "signal_only":
        T = input_ids.size(1)
        return np.zeros((T,), dtype=np.float32), float("nan")

    model.eval()

    # ---- Precompute signal embedding as a CONSTANT (no grad, no cudnn backward) ----
    with torch.no_grad():
        if hasattr(model, "_maybe_norm_signal"):
            signal_in = model._maybe_norm_signal(signal)
        else:
            signal_in = signal
        sig_emb_const = model.sig_encoder(signal_in).detach()  # (B, D)

    # Embeddings to attribute
    emb_layer = model.seq_encoder.get_input_embeddings()
    embeds = emb_layer(input_ids)  # (B,T,H)

    # baseline = PAD embedding (fallback to id=0)
    pad_id = getattr(getattr(model.seq_encoder, "config", None), "pad_token_id", None)
    if pad_id is None:
        pad_id = 0
    baseline_ids = torch.full_like(input_ids, int(pad_id))
    baseline_embeds = emb_layer(baseline_ids)

    alphas = torch.linspace(0.0, 1.0, steps, device=device).view(-1, 1, 1, 1)  # (S,1,1,1)
    emb_steps = baseline_embeds.unsqueeze(0) + alphas * (embeds.unsqueeze(0) - baseline_embeds.unsqueeze(0))  # (S,B,T,H)

    grads = []
    pred_val = None

    # Make sure gradients are enabled even if caller was under no_grad
    grad_prev = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    try:
        for s in range(steps):
            emb_s = emb_steps[s].clone().detach().requires_grad_(True)

            # run seq encoder with grad (bypassing model.forward no_grad)
            seq_out = model.seq_encoder(
                inputs_embeds=emb_s,
                attention_mask=attention_mask,
                output_attentions=False,
                return_dict=True,
            )
            pooled = _sequence_pool_from_last_hidden(model, seq_out.last_hidden_state)
            seq_emb = model.seq_proj(pooled)

            # ---- fuse using CONSTANT signal embedding ----
            if mode == "seq_only":
                fused = seq_emb
            else:
                gate_logits = model.gate(torch.cat([seq_emb, sig_emb_const], dim=1))
                tau = float(getattr(model, "fusion_temperature", 1.0))
                tau = max(tau, 1e-6)
                w = F.softmax(gate_logits / tau, dim=1)
                fused = w[:, 0:1] * seq_emb + w[:, 1:2] * sig_emb_const

            cls_feat = model.cls_adapter(fused) if hasattr(model, "cls_adapter") else fused
            pos_feat = model.pos_adapter(fused) if hasattr(model, "pos_adapter") else fused

            model.zero_grad(set_to_none=True)

            if target == "pos":
                pos = torch.sigmoid(model.pos_head(pos_feat)).squeeze(-1)  # (B,)
                score_tensor = pos.sum()
                pred_val = float(pos.detach().mean().item())
            else:
                logits = model.cls_head(cls_feat).squeeze(-1)               # (B,)
                prob = torch.sigmoid(logits)
                score_tensor = prob.sum()
                pred_val = float(prob.detach().mean().item())

            score_tensor.backward()
            grads.append(emb_s.grad.detach())

        grads = torch.stack(grads, dim=0)     # (S,B,T,H)
        avg_grads = grads.mean(dim=0)         # (B,T,H)
        ig = (embeds - baseline_embeds) * avg_grads  # (B,T,H)

        tok_attr = ig.norm(p=2, dim=-1)       # (B,T)
        tok_attr = tok_attr.squeeze(0)
        tok_attr = tok_attr * attention_mask.squeeze(0).float()
        return _to_numpy(tok_attr), float(pred_val)

    finally:
        torch.set_grad_enabled(grad_prev)


def plot_token_attributions(token_attr: np.ndarray, out_path: str, title: str):
    plt.figure(figsize=(10, 2.5))
    plt.imshow(token_attr[None, :], aspect="auto")
    plt.yticks([])
    plt.xlabel("Token index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# =============================================================================
# 5) In-silico mutagenesis (character-level)
# =============================================================================

@torch.no_grad()
def in_silico_mutagenesis(
    model,
    tokenizer,
    seq_str: str,
    signal: torch.Tensor,
    max_len: int,
    alphabet: str = "ACGT",
    center_window: Optional[Tuple[int, int]] = None,  # (start,end) inclusive
    device: Optional[torch.device] = None,
    mode: str = "fused",
) -> Dict[str, np.ndarray]:
    """
    Character-level mutagenesis: substitute each position with A/C/G/T.
    Returns:
      - delta_prob: (L, |alphabet|) change in P(damage) relative to reference
    """
    if device is None:
        device = _pick_device(model)
    model.eval()

    seq_str = str(seq_str)
    L = len(seq_str)

    if center_window is None:
        start, end = 0, L - 1
    else:
        start, end = center_window
        start = max(0, int(start))
        end = min(L - 1, int(end))

    # reference prob
    enc_ref = tokenizer(seq_str, truncation=True, padding=False, max_length=max_len, return_tensors="pt")
    input_ids_ref = enc_ref["input_ids"].to(device)
    attn_ref = enc_ref["attention_mask"].to(device)
    sig = signal.to(device).unsqueeze(0) if signal.dim() == 1 else signal.to(device)

    out_ref = model(input_ids_ref, attn_ref, sig, output_attentions=False, mode=mode)
    p_ref = float(_safe_sigmoid(out_ref["binary_logits"]).detach().mean().item())

    delta = np.zeros((end - start + 1, len(alphabet)), dtype=np.float32)

    seq_chars = list(seq_str)
    for i, pos in enumerate(range(start, end + 1)):
        original = seq_chars[pos]
        for j, a in enumerate(alphabet):
            if a == original:
                delta[i, j] = 0.0
                continue
            seq_chars[pos] = a
            mutated = "".join(seq_chars)
            enc = tokenizer(mutated, truncation=True, padding=False, max_length=max_len, return_tensors="pt")
            out = model(enc["input_ids"].to(device), enc["attention_mask"].to(device), sig, output_attentions=False, mode=mode)
            p = float(_safe_sigmoid(out["binary_logits"]).detach().mean().item())
            delta[i, j] = p - p_ref
        seq_chars[pos] = original

    return {"delta_prob": delta, "alphabet": np.array(list(alphabet)), "start": start, "end": end, "p_ref": p_ref}


def plot_mutagenesis(delta_prob: np.ndarray, alphabet: List[str], out_path: str, title: str):
    plt.figure(figsize=(10, 3.2))
    plt.imshow(delta_prob.T, aspect="auto")
    plt.yticks(range(len(alphabet)), alphabet)
    plt.xlabel("Position (relative window)")
    plt.title(title)
    plt.colorbar(label="ΔP(damage)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# =============================================================================
# Entry point for inference script
# =============================================================================

def interpret_sample(
    model,
    tokenizer,
    sample: Dict[str, torch.Tensor],
    seq_str: Optional[str],
    out_dir: str,
    sample_id: str,
    max_seq_len: int,
    mode: str = "fused",
):
    """
    Run a bundle of interpretability analyses for a single sample.
    Saves figures into out_dir/sample_id_*.svg
    """
    _ensure_dir(out_dir)
    device = _pick_device(model)

    input_ids = sample["input_ids"].unsqueeze(0).to(device)
    attn = sample["attention_mask"].unsqueeze(0).to(device)
    signal = sample["signal"].unsqueeze(0).to(device)

    # --- Signal IG (binary) ---
    ig_sig, p_bin = integrated_gradients_signal(
        model, input_ids, attn, signal, target="binary", steps=48, mode=mode
    )
    plot_signal_ig(
        signal=_to_numpy(signal.squeeze(0)),
        ig=ig_sig,
        out_path=os.path.join(out_dir, f"{sample_id}_signal_ig_binary.svg"),
        title=f"Signal IG (binary), P(damage)={p_bin:.3f}",
    )

    # --- Token IG (binary) ---
    tok_attr, p_bin2 = integrated_gradients_sequence_tokens(
        model, input_ids, attn, signal, target="binary", steps=24, mode=mode
    )
    plot_token_attributions(
        tok_attr,
        out_path=os.path.join(out_dir, f"{sample_id}_token_ig_binary.svg"),
        title=f"Token IG (binary), P(damage)={p_bin2:.3f}",
    )

    # --- Optional mutagenesis if seq_str provided ---
    if seq_str is not None and tokenizer is not None:
        try:
            mut = in_silico_mutagenesis(
                model=model,
                tokenizer=tokenizer,
                seq_str=seq_str,
                signal=signal.squeeze(0),
                max_len=max_seq_len,
                alphabet="ACGT",
                center_window=None,
                device=device,
                mode=mode,
            )
            plot_mutagenesis(
                delta_prob=mut["delta_prob"],
                alphabet=[str(x) for x in mut["alphabet"].tolist()],
                out_path=os.path.join(out_dir, f"{sample_id}_mutagenesis.svg"),
                title=f"In-silico mutagenesis (ΔP), ref P={mut['p_ref']:.3f}",
            )
        except Exception:
            # keep interpretability robust; don't crash inference
            pass

    return {
        "p_damage": float(p_bin),
    }
