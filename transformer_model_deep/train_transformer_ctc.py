#!/usr/bin/env python3

import argparse
import os
import time
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import mlflow

from transformer_model_deep.data import BarcodeCtcDataset, ctc_collate, read_labels_csv
from transformer_model_deep.decode import beam_ctc_decode, greedy_ctc_decode
from transformer_model_deep.model import TransformerCtcRecognizer
from transformer_model_deep.vocab import build_vocab_from_alphabet, code128_alphabet


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def _cer(pred: str, target: str) -> tuple[int, int]:
    return _edit_distance(pred, target), len(target)


def _estimate_resized_width(path: Path, target_h: int) -> int:
    try:
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
        if h <= 0:
            return 1
        scale = float(target_h) / float(h)
        return max(1, int(round(float(w) * scale)))
    except Exception:
        return 1


class _BucketBatchSampler:
    def __init__(
        self,
        widths: Sequence[int],
        batch_size: int,
        shuffle: bool,
        bucket_size: int,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if bucket_size <= 0:
            raise ValueError("bucket_size must be positive")
        if len(widths) == 0:
            raise ValueError("widths must be non-empty")

        self.widths = list(int(w) for w in widths)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.bucket_size = int(bucket_size)
        self.drop_last = bool(drop_last)

    def __iter__(self) -> Iterable[List[int]]:
        n = len(self.widths)
        indices = list(range(n))
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            perm = torch.randperm(n, generator=g).tolist()
            indices = [indices[i] for i in perm]

        chunks: List[List[int]] = []
        for i in range(0, n, self.bucket_size):
            chunk = indices[i : i + self.bucket_size]
            chunk.sort(key=lambda j: self.widths[j])
            chunks.append(chunk)

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            perm_chunks = torch.randperm(len(chunks), generator=g).tolist()
            chunks = [chunks[i] for i in perm_chunks]

        batch: List[int] = []
        for chunk in chunks:
            for idx in chunk:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        n = len(self.widths)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="my_dataset")
    ap.add_argument("--height", type=int, default=96)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--outdir", type=str, default="checkpoints_transformer_deep")

    ap.add_argument("--model-weights", type=str, default="")
    ap.add_argument("--resume", type=str, default="")

    ap.add_argument("--scheduler", type=str, default="onecycle", choices=["none", "onecycle", "cosine"])

    ap.add_argument("--enc-base", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--ff", type=int, default=1536)
    ap.add_argument("--dropout", type=float, default=0.15)

    ap.add_argument("--decode", type=str, default="greedy", choices=["beam", "greedy"])
    ap.add_argument("--beam-width", type=int, default=10)

    ap.add_argument("--num-workers", type=int, default=2)

    ap.add_argument("--bucket", action="store_true")
    ap.add_argument("--bucket-size", type=int, default=4096)

    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--aug-prob", type=float, default=0.9)

    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--viz-n", type=int, default=8)

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--log-interval", type=int, default=100)
    ap.add_argument("--sync-timing", action="store_true")

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

        # Prefer newer PyTorch API if available
        if hasattr(torch, "set_float32_matmul_precision"):
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        # Enable TF32 where supported (older API)
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True

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
        loader_kwargs["prefetch_factor"] = 1 # 2

    train_ds = BarcodeCtcDataset(
        train_samples,
        char2idx,
        args.height,
        augment=bool(args.aug),
        augment_prob=float(args.aug_prob),
    )
    val_ds = BarcodeCtcDataset(
        val_samples,
        char2idx,
        args.height,
        augment=False,
    )

    if bool(args.bucket):
        print("--- Building width buckets for training ---")
        train_widths = [_estimate_resized_width(s.image_path, int(args.height)) for s in train_ds.samples]
        train_sampler = _BucketBatchSampler(
            widths=train_widths,
            batch_size=int(args.batch),
            shuffle=True,
            bucket_size=int(args.bucket_size),
            drop_last=False,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            **loader_kwargs,
            collate_fn=ctc_collate,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch,
            shuffle=True,
            **loader_kwargs,
            collate_fn=ctc_collate,
        )
    val_loader = DataLoader(
        val_ds,
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

    start_epoch = 1
    global_step = 0
    resume_ckpt = None
    resume_path = args.resume or args.model_weights
    if resume_path:
        resume_ckpt = torch.load(resume_path, map_location="cpu")
        state = resume_ckpt.get("model", resume_ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"--- Loaded checkpoint: {resume_path} ---")
        if missing:
            print(f"  > Missing keys: {len(missing)}")
        if unexpected:
            print(f"  > Unexpected keys: {len(unexpected)}")

        if args.resume:
            if isinstance(resume_ckpt, dict):
                start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
                global_step = int(resume_ckpt.get("global_step", 0))

    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = GradScaler(device.type, enabled=use_amp)

    if args.resume and isinstance(resume_ckpt, dict):
        opt_state = resume_ckpt.get("optimizer")
        if opt_state is not None:
            try:
                optimizer.load_state_dict(opt_state)
                print("--- Restored optimizer state ---")
            except Exception as e:
                print(f"--- Failed to restore optimizer state: {e} ---")

        scaler_state = resume_ckpt.get("scaler")
        if use_amp and scaler_state is not None:
            try:
                scaler.load_state_dict(scaler_state)
                print("--- Restored GradScaler state ---")
            except Exception as e:
                print(f"--- Failed to restore GradScaler state: {e} ---")

    total_steps = len(train_loader) * args.epochs
    scheduler = None
    if args.scheduler != "none":
        last_epoch = int(global_step) - 1
        if args.scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=args.lr,
                total_steps=total_steps,
                pct_start=0.2,
                anneal_strategy="cos",
                last_epoch=last_epoch,
            )
        elif args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=0.0,
                last_epoch=last_epoch,
            )

        if args.resume and isinstance(resume_ckpt, dict):
            sched_state = resume_ckpt.get("scheduler")
            if sched_state is not None:
                try:
                    scheduler.load_state_dict(sched_state)
                    print("--- Restored scheduler state ---")
                except Exception as e:
                    print(f"--- Failed to restore scheduler state: {e} ---")

    tb_logdir = None
    if args.tb:
        tb_logdir = str(Path(args.tb_logdir).expanduser().resolve())
        os.makedirs(tb_logdir, exist_ok=True)
        print(f"--- TensorBoard logdir: {tb_logdir} ---")
    writer = SummaryWriter(tb_logdir, flush_secs=10) if args.tb else None

    if args.mlflow:
        if args.mlflow_tracking_uri:
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        mlflow.start_run()

    best_val_loss = float("inf")

    def _make_grid_2col(imgs: list[torch.Tensor], pad: int = 2, pad_value: float = 0.0) -> torch.Tensor:
        if not imgs:
            return torch.empty((3, 1, 1), dtype=torch.float32)
        c = int(imgs[0].shape[0])
        h = max(int(t.shape[1]) for t in imgs)
        w = max(int(t.shape[2]) for t in imgs)

        padded = []
        for t in imgs:
            if t.shape[0] != c:
                raise ValueError("All images must have same channels")
            dh = h - int(t.shape[1])
            dw = w - int(t.shape[2])
            if dh < 0 or dw < 0:
                raise ValueError("Unexpected negative padding")
            padded.append(torch.nn.functional.pad(t, (0, dw, 0, dh), value=pad_value))

        rows = []
        for i in range(0, len(padded), 2):
            left = padded[i]
            right = padded[i + 1] if i + 1 < len(padded) else torch.full_like(left, pad_value)
            if pad > 0:
                spacer = torch.full((c, h, pad), pad_value, dtype=left.dtype, device=left.device)
                row = torch.cat([left, spacer, right], dim=2)
            else:
                row = torch.cat([left, right], dim=2)
            rows.append(row)

        if pad > 0:
            hsp = torch.full((c, pad, rows[0].shape[2]), pad_value, dtype=rows[0].dtype, device=rows[0].device)
            out = torch.cat([r if j == 0 else torch.cat([hsp, r], dim=1) for j, r in enumerate(rows)], dim=1)
        else:
            out = torch.cat(rows, dim=1)

        return out

    def _log_viz(epoch_i: int) -> None:
        if not writer or not bool(args.viz):
            return
        if not hasattr(val_ds, "load_resized_rgb_u8"):
            return

        n = int(args.viz_n)
        if n <= 0:
            return

        n = min(n, len(val_ds))
        if n <= 0:
            return

        aug = None
        if hasattr(train_ds, "get_augmenter"):
            aug = train_ds.get_augmenter()

        imgs = []
        for i in range(n):
            s = val_ds.samples[i]
            rgb = val_ds.load_resized_rgb_u8(s.image_path)
            rgb_aug = val_ds.apply_augment_to_rgb_u8(rgb, augmenter=aug)

            t0 = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            t1 = torch.from_numpy(rgb_aug).permute(2, 0, 1).float() / 255.0
            imgs.append(t0)
            imgs.append(t1)

        grid = _make_grid_2col(imgs, pad=2, pad_value=0.0)
        writer.add_image("Viz/Val_Orig_Aug", grid, global_step=epoch_i)
        writer.flush()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0

        log_t0 = time.perf_counter()
        last_log_i = -1

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
            if scheduler is not None:
                scheduler.step()

            global_step += 1

            train_loss += loss.item()
            if args.log_interval > 0 and i % args.log_interval == 0:
                lr = float(optimizer.param_groups[0]["lr"]) if len(optimizer.param_groups) > 0 else float("nan")
                if bool(args.sync_timing) and device.type == "cuda":
                    torch.cuda.synchronize(device)
                dt = max(1e-9, time.perf_counter() - log_t0)
                steps = max(1, i - last_log_i)
                imgs_per_sec = (float(xb.size(0)) * float(steps)) / dt
                ms_per_step = (dt / float(steps)) * 1000.0
                print(
                    f"Epoch {epoch} [{i}/{len(train_loader)}] Loss: {loss.item():.4f} | "
                    f"LR: {lr:.6g} | {imgs_per_sec:.1f} img/s | {ms_per_step:.1f} ms/step"
                )

                if writer:
                    writer.add_scalar("LR/Train", lr, global_step)
                    writer.add_scalar("Throughput/Train_ImgsPerSec", imgs_per_sec, global_step)
                    writer.add_scalar("Throughput/Train_MsPerStep", ms_per_step, global_step)

                if args.mlflow:
                    mlflow.log_metric("lr", lr, step=global_step)
                    mlflow.log_metric("train_imgs_per_sec", imgs_per_sec, step=global_step)
                    mlflow.log_metric("train_ms_per_step", ms_per_step, step=global_step)

                log_t0 = time.perf_counter()
                last_log_i = i

        model.eval()
        val_loss = 0.0
        n_exact = 0
        total_v = 0
        total_edits = 0
        total_chars = 0

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
                    ed, ch = _cer(p, t)
                    total_edits += ed
                    total_chars += ch
                    total_v += 1

        avg_train = train_loss / max(1, len(train_loader))
        avg_val = val_loss / max(1, len(val_loader))
        epoch_lr = float(optimizer.param_groups[0]["lr"]) if len(optimizer.param_groups) > 0 else float("nan")
        acc = (n_exact / total_v) * 100 if total_v > 0 else 0.0
        cer = (total_edits / max(1, total_chars)) if total_chars > 0 else 0.0

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        val_dt = max(1e-9, time.perf_counter() - val_t0)
        val_imgs_per_sec = float(n_val_imgs) / val_dt if n_val_imgs > 0 else 0.0

        print(
            f"--- Epoch {epoch} Summary | Val Loss: {avg_val:.4f} | Acc: {acc:.2f}% | "
            f"CER: {cer:.4f} | Time: {time.time()-t0:.2f}s | Val {val_imgs_per_sec:.1f} img/s ---"
        )

        if writer:
            writer.add_scalar("Loss/Train", avg_train, epoch)
            writer.add_scalar("Loss/Val", avg_val, epoch)
            writer.add_scalar("LR/Epoch", epoch_lr, epoch)
            writer.add_scalar("Accuracy/Val", acc, epoch)
            writer.add_scalar("CER/Val", cer, epoch)
            writer.add_scalar("Throughput/Val_ImgsPerSec", val_imgs_per_sec, epoch)
            writer.flush()

        if args.mlflow:
            mlflow.log_metric("train_loss", avg_train, step=epoch)
            mlflow.log_metric("val_loss", avg_val, step=epoch)
            mlflow.log_metric("lr_epoch", epoch_lr, step=epoch)
            mlflow.log_metric("accuracy", acc, step=epoch)
            mlflow.log_metric("cer", cer, step=epoch)
            mlflow.log_metric("val_imgs_per_sec", val_imgs_per_sec, step=epoch)

        last_ckpt_path = os.path.join(args.outdir, "last_model.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if use_amp else None,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "epoch": epoch,
                "global_step": global_step,
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            },
            last_ckpt_path,
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            ckpt_path = os.path.join(args.outdir, "best_model.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if use_amp else None,
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "epoch": epoch,
                    "global_step": global_step,
                    "char2idx": char2idx,
                    "idx2char": idx2char,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  --> Saved new best model to {ckpt_path}")

        _log_viz(epoch)

    if writer:
        writer.close()

    if args.mlflow:
        mlflow.end_run()


if __name__ == "__main__":
    main()
