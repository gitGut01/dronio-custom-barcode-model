from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

import torch

from torchaudio.models.decoder import ctc_decoder


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


@lru_cache(maxsize=32)
def _get_torchaudio_ctc_decoder(tokens: tuple[str, ...], beam_width: int, blank: int):
    blank_token = tokens[blank]
    return ctc_decoder(
        tokens=list(tokens),
        lexicon=None,
        lm=None,
        blank_token=blank_token,
        beam_size=beam_width,
    )


def beam_ctc_decode(
    log_probs_tbc: torch.Tensor,
    idx2char: Dict[int, str],
    beam_width: int = 10,
    blank: int = 0,
) -> List[str]:
    if log_probs_tbc.dim() != 3:
        raise ValueError("Expected log_probs_tbc to be (T, B, C)")

    _, bsz, vocab = log_probs_tbc.shape
    if blank < 0 or blank >= vocab:
        raise ValueError("blank index out of range")

    tokens: List[str] = []
    for i in range(vocab):
        if i == blank:
            tokens.append("<blk>")
        else:
            tok = idx2char.get(i, "")
            tokens.append(tok if tok else f"<{i}>")

    decoder = _get_torchaudio_ctc_decoder(tuple(tokens), beam_width, blank)

    emissions_btc = log_probs_tbc.permute(1, 0, 2).detach().float().cpu()
    hypos = decoder(emissions_btc)

    out: List[str] = []
    for hlist in hypos:
        if not hlist:
            out.append("")
            continue

        best = hlist[0]

        ids = getattr(best, "tokens", None)
        if ids is not None:
            s = "".join(tokens[int(i)] for i in ids if int(i) != blank)
            out.append(s.replace("<blk>", ""))
            continue

        words = getattr(best, "words", None)
        if words is not None:
            out.append("".join(w for w in words if w != "<blk>"))
            continue

        out.append("")

    return out
