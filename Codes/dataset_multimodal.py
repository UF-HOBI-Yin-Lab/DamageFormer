# -*- coding: utf-8 -*-
"""
dataset_multimodal.py

Multimodal dataset:
- sequence -> tokenizer (from tokenizer_dir = pretrained checkpoint dir)
- signal -> fixed length tensor
- label -> 0/1
- damage_position -> normalized [0,1] for label=1 else -1

Offline-safe:
- local_files_only=True
- trust_remote_code=False
"""

from __future__ import annotations
import os
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def _infer_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _parse_signal(signal_val, max_signal_len: int) -> torch.Tensor:
    if signal_val is None:
        return torch.zeros(max_signal_len, dtype=torch.float32)
    if isinstance(signal_val, float) and np.isnan(signal_val):
        return torch.zeros(max_signal_len, dtype=torch.float32)

    if isinstance(signal_val, np.ndarray):
        arr = signal_val.astype(np.float32).flatten()
    elif isinstance(signal_val, (list, tuple)):
        arr = np.asarray(signal_val, dtype=np.float32).flatten()
    else:
        s = str(signal_val).strip()
        if len(s) == 0 or s.lower() in {"nan", "none", "null"}:
            return torch.zeros(max_signal_len, dtype=torch.float32)
        s = s.strip("[]")
        arr = np.fromstring(s, sep=",", dtype=np.float32)
        if arr.size == 0:
            arr = np.fromstring(s, sep=" ", dtype=np.float32)

    if arr.size == 0:
        arr = np.zeros(max_signal_len, dtype=np.float32)
    elif arr.size < max_signal_len:
        arr = np.pad(arr, (0, max_signal_len - arr.size))
    else:
        arr = arr[:max_signal_len]

    return torch.tensor(arr, dtype=torch.float32)


def _load_dataframe(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if ext == ".tsv" else ",")
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(path)


class MultimodalDNADataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer_dir: str,
        max_seq_len: int = 128,
        max_signal_len: int = 256,
        normalize_signal: bool = False,
    ):
        if not os.path.exists(data_path):
            raise FileNotFoundError(data_path)
        if not os.path.isdir(tokenizer_dir):
            raise FileNotFoundError(tokenizer_dir)

        self.df = _load_dataframe(data_path)
        print(f"[Dataset] Loaded {len(self.df)} samples from {data_path}")
        print(f"[Dataset] Columns: {list(self.df.columns)}")

        self.col_seq = _infer_col(self.df, ["Sequence_new", "sequence_new", "sequence", "Sequence"])
        self.col_sig = _infer_col(self.df, ["Signal", "signal"])
        self.col_label = _infer_col(self.df, ["Label", "label"])
        self.col_pos = _infer_col(self.df, ["Damage_Position", "damage_position", "DamagePosition"])
        self.col_blen = _infer_col(self.df, ["Basecall_Length", "basecall_length", "seq_len"])

        if any(v is None for v in [self.col_seq, self.col_sig, self.col_label]):
            raise ValueError(f"Missing required columns. seq={self.col_seq}, sig={self.col_sig}, label={self.col_label}")

        self.max_seq_len = int(max_seq_len)
        self.max_signal_len = int(max_signal_len)
        self.normalize_signal = bool(normalize_signal)

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            use_fast=True,
            local_files_only=True,
            trust_remote_code=True,#False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        if self.normalize_signal:
            self._compute_signal_stats()

        self._print_stats()

    def _compute_signal_stats(self):
        vals = []
        for i in range(min(len(self.df), 10000)):
            sig = self.df.iloc[i][self.col_sig]
            if isinstance(sig, np.ndarray):
                vals.extend(sig.flatten().tolist())
            elif isinstance(sig, (list, tuple)):
                vals.extend(list(sig))
        if vals:
            self.signal_mean = float(np.mean(vals))
            self.signal_std = float(np.std(vals)) + 1e-8
        else:
            self.signal_mean = 0.0
            self.signal_std = 1.0
        print(f"[Dataset] Signal mean={self.signal_mean:.4f}, std={self.signal_std:.4f}")

    def _print_stats(self):
        labels = self.df[self.col_label].values
        pos = int((labels == 1).sum())
        neg = int((labels == 0).sum())
        print(f"[Dataset] Pos={pos}, Neg={neg}, ratio={pos/(neg+1e-8):.3f}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        seq = str(row[self.col_seq])

        enc = self.tokenizer(
            seq,
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_len,
            return_attention_mask=True,
            return_tensors="pt",
        )

        signal = _parse_signal(row[self.col_sig], self.max_signal_len)
        if self.normalize_signal:
            signal = (signal - self.signal_mean) / self.signal_std

        label = int(row[self.col_label])

        raw_pos = -1
        if self.col_pos is not None and not pd.isna(row[self.col_pos]):
            raw_pos = int(row[self.col_pos])

        base_len = (
            int(row[self.col_blen])
            if self.col_blen is not None and not pd.isna(row[self.col_blen])
            else len(seq)
        )

        # normalized position only meaningful for damaged and valid
        if label == 1 and 0 <= raw_pos < base_len:
            norm_pos = float(raw_pos) / float(max(base_len - 1, 1))
            norm_pos = max(0.0, min(1.0, norm_pos))
        else:
            norm_pos = -1.0

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "signal": signal,
            "label": torch.tensor(label, dtype=torch.long),
            "damage_position": torch.tensor(norm_pos, dtype=torch.float32),
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "signal": torch.stack([b["signal"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "damage_position": torch.stack([b["damage_position"] for b in batch]),
    }
