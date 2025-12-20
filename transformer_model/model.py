# placeholder

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("Expected x to be (B, T, C)")
        t = x.size(1)
        if t > self.pe.size(0):
            raise ValueError(f"Sequence length {t} exceeds max_len {self.pe.size(0)}")
        return x + self.pe[:t].unsqueeze(0)


class TransformerCtcRecognizer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        enc_base: int = 32,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = ConvEncoder(3, enc_base)
        self.proj = nn.Linear(enc_base * 4, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, x_w_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(x)
        feat = feat.mean(dim=2).permute(0, 2, 1)

        seq = self.proj(feat)
        seq = self.pos(seq)

        input_lengths = torch.div(x_w_lens, 4, rounding_mode="floor").clamp_min(1)
        t = seq.size(1)
        device = seq.device

        arange = torch.arange(t, device=device).unsqueeze(0)
        key_padding_mask = arange >= input_lengths.to(device).unsqueeze(1)

        enc = self.transformer(seq, src_key_padding_mask=key_padding_mask)
        logits = self.head(enc)
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2).contiguous()
        return log_probs, input_lengths