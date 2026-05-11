# -*- coding: utf-8 -*-
"""
inference_multimodal.py

Offline-safe inference + interpretability for MultimodalDNADamageModel.

Fixes (minimal, but important):
- add --seed (was missing)
- build tokenizer (was missing)
- treat --output as a FILE path; use its directory for artifacts
- fix undefined vars: df_out/out_dir -> df_pred/out_dir
- safe model ctor via keyword args (avoid signature mismatch bugs)
- robust checkpoint loading (model_state/state_dict)
"""

from __future__ import annotations

import os
import argparse
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from dataset_multimodal import MultimodalDNADataset, collate_fn
from model_multimodal import MultimodalDNADamageModel

from interpretability import (
    collect_fusion_gate_stats,
    plot_fusion_gate_reliance,
    cross_modal_swap_test,
    plot_embedding_space,
    interpret_sample,
)

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def _safe_load_tokenizer(tokenizer_dir: str):
    # Prefer local, no remote code. If your tokenizer was saved from DNABERT-2 remote tokenizer,
    # local loading is still fine as long as files exist.
    try:
        tok = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
    except Exception:
        # Fallback (still local-only) in case tokenizer needs remote_code artifacts
        tok = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            local_files_only=True,
            trust_remote_code=True,
            use_fast=True,
        )
    if tok.pad_token is None:
        tok.pad_token = "[PAD]"
    return tok

def _load_ckpt(path: str, device: torch.device) -> Dict:
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict):
        # Sometimes people save raw state_dict
        ckpt = {"model_state": ckpt}
    return ckpt

def _get_state_dict(ckpt: Dict) -> Dict[str, torch.Tensor]:
    # common keys
    if "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
        return ckpt["model_state"]
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    # maybe the whole dict is a state dict
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt  # type: ignore
    raise ValueError(f"Could not find model weights in checkpoint keys: {list(ckpt.keys())}")

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode: str = "fused",
    threshold: float = 0.5,
) -> Tuple[List[Dict], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    model.eval()
    
    all_rows: List[Dict] = []
    fused_list, seq_list, sig_list = [], [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn = batch["attention_mask"].to(device, non_blocking=True)
        signal = batch["signal"].to(device, non_blocking=True)

        labels = batch.get("label", None)
        pos_t = batch.get("damage_position", None)
        if labels is not None:
            labels = labels.to(device, non_blocking=True)
        if pos_t is not None:
            pos_t = pos_t.to(device, non_blocking=True)

        out = model(input_ids, attn, signal, output_attentions=False, mode=mode)
        
        prob = torch.sigmoid(out["binary_logits"].float())
        pred = (prob >= threshold).long()

        pos_pred = out.get("pos_pred", None)
        if pos_pred is None:
            pos_pred = torch.zeros_like(prob)

        # embeddings (optional)
        if "fused_emb" in out:
            fused_list.append(out["fused_emb"].detach().cpu())
        if "seq_emb" in out:
            seq_list.append(out["seq_emb"].detach().cpu())
        if "sig_emb" in out:
            sig_list.append(out["sig_emb"].detach().cpu())

        B = input_ids.size(0)
        for i in range(B):
            row = {
                "prob_damage": float(prob[i].item()),
                "pred_label": int(pred[i].item()),
                "pos_pred": float(pos_pred[i].item()),
            }
            if labels is not None:
                row["label"] = int(labels[i].item())
            if pos_t is not None:
                row["pos_target"] = float(pos_t[i].item())
            all_rows.append(row)

    fused = torch.cat(fused_list, dim=0).numpy() if fused_list else None
    seq = torch.cat(seq_list, dim=0).numpy() if seq_list else None
    sig = torch.cat(sig_list, dim=0).numpy() if sig_list else None
    return all_rows, fused, seq, sig

# -----------------------------
# Args
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--pretrained_dir", required=True)
    p.add_argument("--adapter_dir", required=True)
    p.add_argument("--output", default="predictions.csv")

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--max_signal_len", type=int, default=256)
    p.add_argument("--fusion_dim", type=int, default=256)
    p.add_argument("--mode", type=str, default="fused", choices=["fused", "seq_only", "signal_only"])
    p.add_argument("--normalize_signal", action="store_true")

    p.add_argument("--fusion_temperature", type=float, default=1.0)
    p.add_argument("--use_signal_layernorm", action="store_true")
    p.add_argument("--use_local_seq_pool", action="store_true")
    p.add_argument("--local_pool_tokens", type=int, default=16)
    p.add_argument("--use_lstm", action="store_true")
    p.add_argument("--use_residual", action="store_true")
    p.add_argument("--lstm_hidden", type=int, default=128)

    p.add_argument("--save_embeddings", action="store_true")
    p.add_argument("--interpret", action="store_true")
    p.add_argument("--interpret_samples", type=int, default=50)
    p.add_argument("--interpret_max_batches", type=int, default=50)

    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()

# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # output is a FILE path; use its directory for artifacts
    out_dir = os.path.dirname(os.path.abspath(args.output))
    ensure_dir(out_dir)

    # tokenizer (needed for interpretability token IG, and sometimes for dataset)
    tokenizer = _safe_load_tokenizer(args.pretrained_dir)

    # checkpoint
    ckpt = _load_ckpt(args.checkpoint, device=device)
    state_dict = _get_state_dict(ckpt)

    # model (keyword args to avoid signature mismatch)
    model = MultimodalDNADamageModel(
        pretrained_dir=args.pretrained_dir,
        adapter_dir=args.adapter_dir,
        max_signal_len=args.max_signal_len,
        fusion_dim=args.fusion_dim,
        dropout=0.0,  # inference
        device=device,
        fusion_temperature=args.fusion_temperature,
        use_signal_layernorm=args.use_signal_layernorm,
        use_local_seq_pool=args.use_local_seq_pool,
        local_pool_tokens=args.local_pool_tokens,
        use_lstm=args.use_lstm,
        use_residual=args.use_residual,
        lstm_hidden=args.lstm_hidden,
    ).to(device)

    # load weights
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # dataset + loader
    ds = MultimodalDNADataset(
        data_path=args.data,
        tokenizer_dir=args.pretrained_dir,   # dataset expects a dir; keep minimal change
        max_seq_len=args.max_seq_len,
        max_signal_len=args.max_signal_len,
        normalize_signal=args.normalize_signal,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # -----------------------------
    # Inference
    # -----------------------------
    rows, fused, seq, sig = run_inference(
        model=model,
        loader=loader,
        device=device,
        mode=args.mode,
        threshold=args.threshold,
    )
    df_pred = pd.DataFrame(rows)
    df_pred.to_csv(args.output, index=False)
    print(f"[Inference] saved: {args.output}")

    # -----------------------------
    # Save embeddings (optional)
    # -----------------------------
    if args.save_embeddings and fused is not None:
        npz_path = os.path.join(out_dir, "embeddings.npz")
        np.savez_compressed(npz_path, fused=fused, seq=seq, sig=sig)
        print(f"[Inference] Saved embeddings to: {npz_path}")

        label_col = df_pred["label"].to_numpy() if "label" in df_pred.columns else None
        plot_embedding_space(
            fused_embs=fused,
            labels=label_col,
            out_path=os.path.join(out_dir, "embedding_pca.svg"),
            title="Fused embedding space (PCA)",
        )

    # -----------------------------
    # Interpretability (optional)
    # -----------------------------
    if args.interpret:
        interp_dir = ensure_dir(os.path.join(out_dir, "interpretability"))

        # 1) Fusion gate stats
        stats = collect_fusion_gate_stats(
            model=model,
            loader=loader,
            mode=args.mode,
            max_batches=args.interpret_max_batches,
        )
        plot_fusion_gate_reliance(stats, out_dir=interp_dir, prefix="fusion")

        # 2) Swap tests (first batch)
        first_batch = next(iter(loader))
        swap_sig = cross_modal_swap_test(model, first_batch, swap="signal", n_perm=2, mode=args.mode)
        swap_seq = cross_modal_swap_test(model, first_batch, swap="sequence", n_perm=2, mode=args.mode)
        with open(os.path.join(interp_dir, "swap_tests.txt"), "w", encoding="utf-8") as f:
            f.write(f"swap_signal mean_abs_prob_shift: {swap_sig['mean_abs_prob_shift']:.6f}\n")
            f.write(f"swap_sequence mean_abs_prob_shift: {swap_seq['mean_abs_prob_shift']:.6f}\n")
        print(f"[✓] Saved swap test summary to: {os.path.join(interp_dir, 'swap_tests.txt')}")

        # 3) Per-sample attributions
        n = min(args.interpret_samples, len(ds))
        pick = np.random.choice(len(ds), size=n, replace=False)

        col_seq = getattr(ds, "col_seq", None)

        for idx in pick:
            item = ds[int(idx)]
            seq_str = None
            try:
                if col_seq is not None and hasattr(ds, "df"):
                    seq_str = str(ds.df.loc[int(idx), col_seq])
            except Exception:
                seq_str = None

            sample_id = f"sample_{int(idx)}"
            interpret_sample(
                model=model,
                tokenizer=tokenizer,
                sample=item,
                seq_str=seq_str,
                out_dir=interp_dir,
                sample_id=sample_id,
                max_seq_len=args.max_seq_len,
                mode=args.mode,
            )

        print(f"[✓] Saved interpretability outputs to: {interp_dir}")

if __name__ == "__main__":
    main()
