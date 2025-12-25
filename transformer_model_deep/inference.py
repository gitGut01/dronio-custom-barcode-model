#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

import torch

from transformer_model_deep.decode import greedy_ctc_decode
from transformer_model_deep.model import TransformerCtcRecognizer
from transformer_model_deep.vocab import build_vocab_from_alphabet, code128_alphabet


def _load_image_rgb(path: Path, height: int) -> Tuple[torch.Tensor, int]:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = float(height) / float(h)
    new_w = max(1, int(round(w * scale)))
    img = img.resize((new_w, int(height)), resample=Image.Resampling.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    x = torch.from_numpy(arr)
    return x, int(new_w)


def _infer_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _extract_model_args(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    args = ckpt.get("args") if isinstance(ckpt, dict) else None
    if isinstance(args, dict):
        return args
    return {}


def _get_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError("Unsupported checkpoint format")


def _get_idx2char(ckpt: Any) -> Dict[int, str]:
    if isinstance(ckpt, dict):
        idx2char = ckpt.get("idx2char")
        if isinstance(idx2char, dict) and idx2char:
            out: Dict[int, str] = {}
            for k, v in idx2char.items():
                out[int(k)] = str(v)
            return out

        char2idx = ckpt.get("char2idx")
        if isinstance(char2idx, dict) and char2idx:
            return {int(i): str(ch) for ch, i in char2idx.items()}

    _, idx2char = build_vocab_from_alphabet(code128_alphabet())
    return idx2char


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--height", type=int, default=96)
    ap.add_argument("--decode", type=str, default="greedy", choices=["greedy", "beam"])
    ap.add_argument("--beam-width", type=int, default=10)

    ap.add_argument("--enc-base", type=int, default=None)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--nhead", type=int, default=None)
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--ff", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)

    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(str(img_path))

    device = _infer_device(args.device)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    ckpt_args = _extract_model_args(ckpt) if isinstance(ckpt, dict) else {}

    idx2char = _get_idx2char(ckpt)
    vocab_size = max(idx2char.keys(), default=0) + 1

    def pick(name: str, cli_val, default_val):
        if cli_val is not None:
            return cli_val
        if name in ckpt_args and ckpt_args[name] is not None:
            return ckpt_args[name]
        return default_val

    model = TransformerCtcRecognizer(
        vocab_size=vocab_size,
        enc_base=int(pick("enc_base", args.enc_base, 48)),
        d_model=int(pick("d_model", args.d_model, 384)),
        nhead=int(pick("nhead", args.nhead, 8)),
        num_layers=int(pick("num_layers", args.num_layers, 6)),
        dim_feedforward=int(pick("ff", args.ff, 1536)),
        dropout=float(pick("dropout", args.dropout, 0.1)),
    ).to(device)

    state = _get_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint (showing up to 20): {unexpected[:20]}")

    model.eval()

    x, w = _load_image_rgb(img_path, args.height)
    xb = x.unsqueeze(0).to(device)
    x_w_lens = torch.tensor([w], dtype=torch.long)

    with torch.no_grad():
        log_probs_tbc, _ = model(xb, x_w_lens)

    if args.decode == "beam":
        try:
            from transformer_model_deep.decode import beam_ctc_decode

            pred = beam_ctc_decode(log_probs_tbc, idx2char, beam_width=int(args.beam_width), blank=0)[0]
        except Exception as e:
            raise RuntimeError(
                "Beam decoding requires torchaudio to be installed and importable. "
                "Either install torchaudio or use --decode greedy."
            ) from e
    else:
        pred = greedy_ctc_decode(log_probs_tbc, idx2char)[0]

    print(pred)


if __name__ == "__main__":
    main()
