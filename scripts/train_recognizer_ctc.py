#!/usr/bin/env python3

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import mlflow


@dataclass(frozen=True)
class Sample:
    image_path: Path
    symbology: str
    value: str


DEFAULT_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. $/+%/"


def build_vocab(samples: Sequence[Sample]) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = set()
    for s in samples:
        for ch in s.value:
            chars.add(ch)

    # Ensure a stable order.
    # CTC uses 0 as blank by convention here.
    alphabet = sorted(chars)

    # If dataset is tiny, we still want the full expected charset so we don't
    # crash when a new char appears later.
    for ch in DEFAULT_ALPHABET:
        alphabet.append(ch)

    # Deduplicate while keeping order
    seen = set()
    deduped: List[str] = []
    for ch in alphabet:
        if ch not in seen:
            seen.add(ch)
            deduped.append(ch)

    char2idx = {ch: i + 1 for i, ch in enumerate(deduped)}  # 0 reserved for CTC blank
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
            rel = row["filename"]
            sym = row["symbology"]
            val = row["value"]
            samples.append(Sample(image_path=dataset_root / split / rel, symbology=sym, value=val))
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
        # CHW
        arr = np.transpose(arr, (2, 0, 1))
        x = torch.from_numpy(arr)
        return x

    def _encode(self, text: str) -> torch.Tensor:
        ids: List[int] = []
        for ch in text:
            if ch not in self.char2idx:
                raise ValueError(f"Encountered unseen character {ch!r} in label {text!r}")
            ids.append(self.char2idx[ch])
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        s = self.samples[idx]
        x = self._load_image(s.image_path)
        y = self._encode(s.value)
        return x, y, s.symbology


def ctc_collate(batch: Sequence[Tuple[torch.Tensor, torch.Tensor, str]]):
    xs, ys, syms = zip(*batch)

    max_w = max(x.shape[-1] for x in xs)
    padded_xs = []
    for x in xs:
        pad_w = max_w - x.shape[-1]
        if pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, 0), value=1.0)  # white padding
        padded_xs.append(x)

    x_batch = torch.stack(padded_xs, dim=0)

    y_lens = torch.tensor([y.numel() for y in ys], dtype=torch.long)
    y_concat = torch.cat(ys, dim=0) if len(ys) > 0 else torch.empty((0,), dtype=torch.long)

    return x_batch, y_concat, y_lens, list(syms)


class ConvEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(base, base * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(base * 2, base * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(base * 4, base * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # reduce height more than width
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CtcRecognizer(nn.Module):
    def __init__(self, vocab_size_including_blank: int, enc_base: int = 32, rnn_hidden: int = 256) -> None:
        super().__init__()
        self.encoder = ConvEncoder(in_ch=3, base=enc_base)
        enc_out_ch = enc_base * 4
        self.rnn = nn.LSTM(
            input_size=enc_out_ch,
            hidden_size=rnn_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1,
        )
        self.head = nn.Linear(rnn_hidden * 2, vocab_size_including_blank)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, 3, H, W]
        feat = self.encoder(x)  # [B, C, H', W']
        b, c, h, w = feat.shape
        # pool height away -> [B, C, W]
        feat = feat.mean(dim=2)
        # [B, W, C]
        seq = feat.permute(0, 2, 1)
        seq_out, _ = self.rnn(seq)
        logits = self.head(seq_out)  # [B, T, V]

        # CTC wants [T, B, V]
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2).contiguous()
        input_lengths = torch.full((b,), fill_value=log_probs.shape[0], dtype=torch.long, device=log_probs.device)
        return log_probs, input_lengths


def greedy_ctc_decode(log_probs_tbc: torch.Tensor, idx2char: Dict[int, str]) -> List[str]:
    # log_probs: [T, B, V]
    preds = log_probs_tbc.argmax(dim=-1)  # [T, B]
    out: List[str] = []
    for b in range(preds.shape[1]):
        seq = preds[:, b].tolist()
        prev = None
        chars: List[str] = []
        for p in seq:
            if p == 0:
                prev = p
                continue
            if prev == p:
                continue
            chars.append(idx2char.get(p, ""))
            prev = p
        out.append("".join(chars))
    return out


def edit_distance(a: str, b: str) -> int:
    # simple Levenshtein DP
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, start=1):
            cur = dp[j]
            cost = 0 if ca == cb else 1
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + cost,
            )
            prev = cur
    return dp[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a CTC barcode recognizer on cropped images")
    ap.add_argument("--data", type=str, default="my_dataset", help="Dataset root with train/val/test")
    ap.add_argument("--height", type=int, default=64, help="Resize images to this height")
    ap.add_argument("--batch", type=int, default=16, help="Batch size")
    ap.add_argument("--epochs", type=int, default=10, help="Epochs")
    ap.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|mps|cuda")
    ap.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking")
    ap.add_argument("--mlflow-tracking-uri", type=str, default="", help="MLflow tracking URI (optional)")
    ap.add_argument("--mlflow-experiment", type=str, default="barcode-ctc", help="MLflow experiment name")
    ap.add_argument("--mlflow-run-name", type=str, default="", help="MLflow run name (optional)")
    args = ap.parse_args()

    data_root = Path(args.data)

    train_samples = read_labels_csv(data_root, "train")
    val_samples = read_labels_csv(data_root, "val")

    char2idx, idx2char = build_vocab(train_samples + val_samples)
    vocab_size_including_blank = max(char2idx.values()) + 1

    train_ds = BarcodeCtcDataset(train_samples, char2idx=char2idx, height=args.height)
    val_ds = BarcodeCtcDataset(val_samples, char2idx=char2idx, height=args.height)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, collate_fn=ctc_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=ctc_collate)

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model = CtcRecognizer(vocab_size_including_blank=vocab_size_including_blank).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    mlflow_enabled = bool(args.mlflow)
    if mlflow_enabled and args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    if mlflow_enabled:
        mlflow.set_experiment(args.mlflow_experiment)

    def _train_and_eval() -> Path:
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            total_items = 0
            for xb, y_concat, y_lens, _syms in train_loader:
                xb = xb.to(device)
                y_concat = y_concat.to(device)
                y_lens = y_lens.to(device)

                log_probs, input_lens = model(xb)
                loss = ctc(log_probs, y_concat, input_lens, y_lens)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                opt.step()

                bs = xb.shape[0]
                total_loss += float(loss.item()) * bs
                total_items += bs

            train_loss = total_loss / max(1, total_items)

            model.eval()
            val_loss = 0.0
            val_items = 0
            n_exact = 0
            n_chars = 0
            n_edits = 0

            with torch.no_grad():
                for xb, y_concat, y_lens, _syms in val_loader:
                    xb = xb.to(device)
                    y_concat = y_concat.to(device)
                    y_lens = y_lens.to(device)

                    log_probs, input_lens = model(xb)
                    loss = ctc(log_probs, y_concat, input_lens, y_lens)

                    bs = xb.shape[0]
                    val_loss += float(loss.item()) * bs
                    val_items += bs

                    pred_texts = greedy_ctc_decode(log_probs, idx2char)

                    # unpack targets back into strings
                    y_cpu = y_concat.detach().cpu()
                    lens_cpu = y_lens.detach().cpu().tolist()
                    offset = 0
                    tgt_texts: List[str] = []
                    for ln in lens_cpu:
                        ids = y_cpu[offset : offset + ln].tolist()
                        offset += ln
                        tgt_texts.append("".join(idx2char[i] for i in ids))

                    for p, t in zip(pred_texts, tgt_texts):
                        if p == t:
                            n_exact += 1
                        n_edits += edit_distance(p, t)
                        n_chars += max(1, len(t))

            val_loss = val_loss / max(1, val_items)
            exact_acc = n_exact / max(1, val_items)
            cer = n_edits / max(1, n_chars)

            if mlflow_enabled:
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("exact_acc", exact_acc, step=epoch)
                mlflow.log_metric("cer", cer, step=epoch)

            print(
                f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} exact={exact_acc:.3f} CER={cer:.3f} vocab={vocab_size_including_blank}"
            )

        out_path = Path("ctc_recognizer.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "height": args.height,
            },
            out_path,
        )
        return out_path

    if mlflow_enabled:
        run_name = args.mlflow_run_name if args.mlflow_run_name else None
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "data": str(args.data),
                    "height": int(args.height),
                    "batch": int(args.batch),
                    "epochs": int(args.epochs),
                    "lr": float(args.lr),
                    "device": str(device),
                    "vocab_size_including_blank": int(vocab_size_including_blank),
                }
            )

            out_path = _train_and_eval()
            mlflow.log_artifact(str(out_path))
            print(f"saved: {out_path}")
    else:
        out_path = _train_and_eval()
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
