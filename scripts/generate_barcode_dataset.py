#!/usr/bin/env python3

import argparse
import csv
import math
import os
import time
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

@dataclass(frozen=True)
class Sample:
    image_path: Path
    symbology: str
    value: str

DEFAULT_ALPHABET = "0123456789"

def build_vocab(samples: Sequence[Sample]) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = set()
    for s in samples:
        for ch in s.value:
            chars.add(ch)
    alphabet = sorted(chars)
    for ch in DEFAULT_ALPHABET:
        alphabet.append(ch)
    
    seen = set()
    deduped: List[str] = []
    for ch in alphabet:
        if ch not in seen:
            seen.add(ch)
            deduped.append(ch)

    char2idx = {ch: i + 1 for i, ch in enumerate(deduped)}
    idx2char = {i: ch for ch, i in char2idx.items()}
    return char2idx, idx2char

def read_labels_csv(dataset_root: Path, split: str) -> List[Sample]:
    labels_path = dataset_root / split / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels.csv: {labels_path}")

    samples: List[Sample] = []
    with labels_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(Sample(
                image_path=dataset_root / split / row["filename"], 
                symbology=row["symbology"], 
                value=row["value"]
            ))
    return samples

class BarcodeCtcDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], char2idx: Dict[str, int], height: int) -> None:
        self.samples = list(samples)
        self.char2idx = char2idx
        self.height = int(height)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = self.height / float(h)
        new_w = max(1, int(round(w * scale)))
        img = img.resize((new_w, self.height), resample=Image.Resampling.BILINEAR)
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def _encode(self, text: str) -> torch.Tensor:
        ids = [self.char2idx[ch] for ch in text if ch in self.char2idx]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        s = self.samples[idx]
        # Added original string (s.value) to return for validation comparison
        return self._load_image(s.image_path), self._encode(s.value), s.symbology, s.value

def ctc_collate(batch):
    xs, ys, syms, values = zip(*batch)
    x_w_lens = torch.tensor([x.shape[-1] for x in xs], dtype=torch.long)
    max_w = max(x.shape[-1] for x in xs)
    
    padded_xs = [F.pad(x, (0, max_w - x.shape[-1], 0, 0), value=1.0) for x in xs]
    x_batch = torch.stack(padded_xs, dim=0)
    
    y_lens = torch.tensor([y.numel() for y in ys], dtype=torch.long)
    y_concat = torch.cat(ys, dim=0) if len(ys) > 0 else torch.empty((0,), dtype=torch.long)
    return x_batch, x_w_lens, y_concat, y_lens, list(syms), list(values)

class ConvEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, 1, 1), nn.BatchNorm2d(base), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(base, base * 2, 3, 1, 1), nn.BatchNorm2d(base * 2), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(base * 2, base * 4, 3, 1, 1), nn.BatchNorm2d(base * 4), nn.ReLU(True),
            nn.Conv2d(base * 4, base * 4, 3, 1, 1), nn.BatchNorm2d(base * 4), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)), 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CtcRecognizer(nn.Module):
    def __init__(self, vocab_size: int, enc_base: int = 32, rnn_hidden: int = 256) -> None:
        super().__init__()
        self.encoder = ConvEncoder(3, enc_base)
        self.rnn = nn.LSTM(enc_base * 4, rnn_hidden, 2, bidirectional=True, batch_first=True, dropout=0.1)
        self.head = nn.Linear(rnn_hidden * 2, vocab_size)

    def forward(self, x: torch.Tensor, x_w_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(x)
        feat = feat.mean(dim=2).permute(0, 2, 1) # B, W, C
        seq_out, _ = self.rnn(feat)
        logits = self.head(seq_out)
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2).contiguous()
        input_lengths = torch.div(x_w_lens, 4, rounding_mode="floor").clamp_min(1)
        return log_probs, input_lengths

def greedy_ctc_decode(log_probs_tbc: torch.Tensor, idx2char: Dict[int, str]) -> List[str]:
    preds = log_probs_tbc.argmax(dim=-1)
    out = []
    for b in range(preds.shape[1]):
        seq, chars, prev = preds[:, b].tolist(), [], None
        for p in seq:
            if p != 0 and p != prev: 
                chars.append(idx2char.get(p, ""))
            prev = p
        out.append("".join(chars))
    return out

OLD_LR = 1e-3

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="my_dataset")
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=8e-3)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--outdir", type=str, default="checkpoints")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data Loading
    train_samples = read_labels_csv(Path(args.data), "train")
    val_samples = read_labels_csv(Path(args.data), "val")
    char2idx, idx2char = build_vocab(train_samples + val_samples)
    v_size = max(char2idx.values()) + 1

    cpus = multiprocessing.cpu_count()
    train_loader = DataLoader(
        BarcodeCtcDataset(train_samples, char2idx, args.height),
        batch_size=args.batch, shuffle=True, 
        num_workers=cpus, pin_memory=True, collate_fn=ctc_collate
    )
    val_loader = DataLoader(
        BarcodeCtcDataset(val_samples, char2idx, args.height),
        batch_size=args.batch, shuffle=False, 
        num_workers=cpus, pin_memory=True, collate_fn=ctc_collate
    )

    model = CtcRecognizer(v_size).to(device)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1, anneal_strategy='cos'
    )

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # --- TRAINING PHASE ---
        model.train()
        t0 = time.time()
        train_loss = 0.0
        
        for xb, x_w_lens, y_concat, y_lens, _, _ in train_loader:
            xb, y_concat, y_lens = xb.to(device), y_concat.to(device), y_lens.to(device)
            
            log_probs, input_lens = model(xb, x_w_lens)
            loss = ctc(log_probs, y_concat, input_lens.to(device), y_lens)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        correct_sequences = 0
        total_sequences = 0
        
        with torch.no_grad():
            for xb, x_w_lens, y_concat, y_lens, _, targets in val_loader:
                xb, y_concat, y_lens = xb.to(device), y_concat.to(device), y_lens.to(device)
                
                log_probs, input_lens = model(xb, x_w_lens)
                loss = ctc(log_probs, y_concat, input_lens.to(device), y_lens)
                val_loss += loss.item()
                
                # Decode and calculate accuracy
                preds = greedy_ctc_decode(log_probs, idx2char)
                for p, t in zip(preds, targets):
                    if p == t:
                        correct_sequences += 1
                    total_sequences += 1

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        accuracy = (correct_sequences / total_sequences) * 100
        
        print(f"Epoch [{epoch}/{args.epochs}] | Train Loss: {avg_train:.4f} | "
              f"Val Loss: {avg_val:.4f} | Acc: {accuracy:.2f}% | Time: {time.time()-t0:.2f}s")

        # Save Checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(args.outdir, "best_model.pt"))
            print(f"  --> Saved new best model to {args.outdir}")

if __name__ == "__main__":
    main()