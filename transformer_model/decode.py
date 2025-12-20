from __future__ import annotations

from typing import Dict, List

import torch


def greedy_ctc_decode(log_probs_tbc: torch.Tensor, idx2char: Dict[int, str]) -> List[str]:
    preds = log_probs_tbc.argmax(dim=-1)
    out: List[str] = []
    for b in range(preds.shape[1]):
        seq = preds[:, b].tolist()
        chars: List[str] = []
        prev = None
        for p in seq:
            if p != 0 and p != prev:
                chars.append(idx2char.get(p, ""))
            prev = p
        out.append("".join(chars))
    return out
