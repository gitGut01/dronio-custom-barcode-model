#!/usr/bin/env python3

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import mlflow

from transformer_model.data import BarcodeCtcDataset, ctc_collate, read_labels_csv
from transformer_model.decode import beam_ctc_decode, greedy_ctc_decode
from transformer_model.model import TransformerCtcRecognizer
from transformer_model.vocab import build_vocab_from_alphabet, code128_alphabet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="my_dataset")
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--outdir", type=str, default="checkpoints_transformer")

    ap.add_argument("--model-weights", type=str, default="")

    ap.add_argument("--enc-base", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=6)
    ap.add_argument("--ff", type=int, default=1536)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--decode", type=str, default="beam", choices=["beam", "greedy"])
    ap.add_argument("--beam-width", type=int, default=10)

    ap.add_argument("--num-workers", type=int, default=4)

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--log-interval", type=int, default=100)

    ap.add_argument("--tb", action="store_true")
    ap.add_argument("--tb-logdir", type=str, default="runs/barcode-transformer-ctc")
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--mlflow-tracking-uri", type=str, default="")
    ap.add_argument("--mlflow-experiment", type=str, default="barcode-transformer-ctc")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"

    train_samples = read_labels_csv(Path(args.data), "train")
    val_samples = read_labels_csv(Path(args.data), "val")

    char2idx, idx2char = build_vocab_from_alphabet(code128_alphabet())

    vocab_size = max(char2idx.values()) + 1 if len(char2idx) > 0 else 1

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        BarcodeCtcDataset(train_samples, char2idx, args.height),
        batch_size=args.batch,
        shuffle=True,
        **loader_kwargs,
        collate_fn=ctc_collate,
    )
    val_loader = DataLoader(
        BarcodeCtcDataset(val_samples, char2idx, args.height),
        batch_size=args.batch,
        shuffle=False,
        **loader_kwargs,
        collate_fn=ctc_collate,
    )

    model = TransformerCtcRecognizer(
        vocab_size=vocab_size,
        enc_base=args.enc_base,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.ff,
        dropout=args.dropout,
    ).to(device)

    if args.model_weights:
        ckpt = torch.load(args.model_weights, map_location="cpu")
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"--- Loaded checkpoint: {args.model_weights} ---")
        if missing:
            print(f"  > Missing keys: {len(missing)}")
        if unexpected:
            print(f"  > Unexpected keys: {len(unexpected)}")

    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = GradScaler(device.type, enabled=use_amp)

    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.2,
        anneal_strategy="cos",
    )

    writer = SummaryWriter(args.tb_logdir) if args.tb else None

    if args.mlflow:
        if args.mlflow_tracking_uri:
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        mlflow.start_run()

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0

        step_t0 = time.perf_counter()

        for i, (xb, x_w_lens, y_concat, y_lens, _, _) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)
            y_concat = y_concat.to(device, non_blocking=True)
            y_lens = y_lens.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device.type, enabled=use_amp):
                log_probs, input_lens = model(xb, x_w_lens)
                loss = ctc(log_probs, y_concat, input_lens.to(device), y_lens)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            if args.log_interval > 0 and i % args.log_interval == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                dt = max(1e-9, time.perf_counter() - step_t0)
                imgs_per_sec = float(xb.size(0)) / dt
                print(
                    f"Epoch {epoch} [{i}/{len(train_loader)}] Loss: {loss.item():.4f} | "
                    f"{imgs_per_sec:.1f} img/s | {dt*1000.0:.1f} ms/step"
                )

                if writer:
                    global_step = (epoch - 1) * len(train_loader) + i
                    writer.add_scalar("Throughput/Train_ImgsPerSec", imgs_per_sec, global_step)
                    writer.add_scalar("Throughput/Train_MsPerStep", dt * 1000.0, global_step)

                if args.mlflow:
                    global_step = (epoch - 1) * len(train_loader) + i
                    mlflow.log_metric("train_imgs_per_sec", imgs_per_sec, step=global_step)
                    mlflow.log_metric("train_ms_per_step", dt * 1000.0, step=global_step)

                step_t0 = time.perf_counter()

        model.eval()
        val_loss = 0.0
        n_exact = 0
        total_v = 0

        with torch.no_grad():
            val_t0 = time.perf_counter()
            n_val_imgs = 0
            for xb, x_w_lens, y_concat, y_lens, _, targets in val_loader:
                xb = xb.to(device, non_blocking=True)
                y_concat = y_concat.to(device, non_blocking=True)
                y_lens = y_lens.to(device, non_blocking=True)

                with autocast(device.type, enabled=use_amp):
                    log_probs, input_lens = model(xb, x_w_lens)
                    loss = ctc(log_probs, y_concat, input_lens.to(device), y_lens)
                val_loss += loss.item()

                n_val_imgs += int(xb.size(0))

                if args.decode == "beam":
                    preds = beam_ctc_decode(log_probs, idx2char, beam_width=args.beam_width, blank=0)
                else:
                    preds = greedy_ctc_decode(log_probs, idx2char)
                for p, t in zip(preds, targets):
                    if p == t:
                        n_exact += 1
                    total_v += 1

        avg_train = train_loss / max(1, len(train_loader))
        avg_val = val_loss / max(1, len(val_loader))
        acc = (n_exact / total_v) * 100 if total_v > 0 else 0.0

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        val_dt = max(1e-9, time.perf_counter() - val_t0)
        val_imgs_per_sec = float(n_val_imgs) / val_dt if n_val_imgs > 0 else 0.0

        print(
            f"--- Epoch {epoch} Summary | Val Loss: {avg_val:.4f} | Acc: {acc:.2f}% | "
            f"Time: {time.time()-t0:.2f}s | Val {val_imgs_per_sec:.1f} img/s ---"
        )

        if writer:
            writer.add_scalar("Loss/Train", avg_train, epoch)
            writer.add_scalar("Loss/Val", avg_val, epoch)
            writer.add_scalar("Accuracy/Val", acc, epoch)
            writer.add_scalar("Throughput/Val_ImgsPerSec", val_imgs_per_sec, epoch)

        if args.mlflow:
            mlflow.log_metric("train_loss", avg_train, step=epoch)
            mlflow.log_metric("val_loss", avg_val, step=epoch)
            mlflow.log_metric("accuracy", acc, step=epoch)
            mlflow.log_metric("val_imgs_per_sec", val_imgs_per_sec, step=epoch)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            ckpt_path = os.path.join(args.outdir, "best_model.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "char2idx": char2idx,
                    "idx2char": idx2char,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  --> Saved new best model to {ckpt_path}")

    if writer:
        writer.close()

    if args.mlflow:
        mlflow.end_run()


if __name__ == "__main__":
    main()
