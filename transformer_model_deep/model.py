# placeholder

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones((dim)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + x


class ModernConvEncoderW4(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 64, depth: int = 3) -> None:
        super().__init__()
        self.stem_conv = nn.Conv2d(in_ch, base, kernel_size=4, stride=(4, 2), padding=0)
        self.stem_norm = nn.LayerNorm(base)

        self.stage1 = nn.Sequential(*[ConvNeXtBlock(base) for _ in range(depth)])
        self.downsample = nn.Conv2d(base, base * 4, kernel_size=2, stride=2)
        self.stage2 = nn.Sequential(*[ConvNeXtBlock(base * 4) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.stem_norm(x)
        x = x.permute(0, 3, 1, 2)

        x = self.stage1(x)
        x = self.downsample(x)
        x = self.stage2(x)
        return x


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
        self.encoder = ModernConvEncoderW4(3, base=enc_base)
        self._downsample_factor_w = 4
        enc_out_ch = enc_base * 4

        self.proj = nn.Linear(enc_out_ch, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)

        self.height_attn = nn.Conv2d(enc_out_ch, 1, kernel_size=1)

        self.seq_dwconv = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        self.seq_act = nn.GELU()

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
        attn_logits = self.height_attn(feat)
        attn = F.softmax(attn_logits, dim=2)
        feat = (feat * attn).sum(dim=2)
        feat = feat.permute(0, 2, 1)

        seq = self.proj(feat)

        seq_res = seq
        seq = self.seq_dwconv(seq.transpose(1, 2)).transpose(1, 2)
        seq = self.seq_act(seq)
        seq = seq + seq_res

        seq = self.pos(seq)

        t = seq.size(1)
        input_lengths = (
            torch.div(x_w_lens, self._downsample_factor_w, rounding_mode="floor")
            .clamp_min(1)
            .clamp_max(t)
        )
        device = seq.device

        arange = torch.arange(t, device=device).unsqueeze(0)
        key_padding_mask = arange >= input_lengths.to(device).unsqueeze(1)

        enc = self.transformer(seq, src_key_padding_mask=key_padding_mask)
        logits = self.head(enc)
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2).contiguous()
        return log_probs, input_lengths