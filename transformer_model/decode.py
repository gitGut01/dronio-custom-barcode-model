from __future__ import annotations

from typing import Dict, List

import torch


def _log_add_exp(a: float, b: float) -> float:
    if a == -float("inf"):
        return b
    if b == -float("inf"):
        return a
    m = a if a > b else b
    return m + float(torch.log1p(torch.exp(torch.tensor(-abs(a - b), dtype=torch.float32))).item())


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


def beam_ctc_decode(
    log_probs_tbc: torch.Tensor,
    idx2char: Dict[int, str],
    beam_width: int = 10,
    blank: int = 0,
) -> List[str]:
    if log_probs_tbc.dim() != 3:
        raise ValueError("Expected log_probs_tbc to be (T, B, C)")
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")

    t, bsz, vocab = log_probs_tbc.shape
    if blank < 0 or blank >= vocab:
        raise ValueError("blank index out of range")

    out: List[str] = []
    log_probs_tbc = log_probs_tbc.detach().float().cpu()

    for b in range(bsz):
        beams: Dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -float("inf"))}
        for ti in range(t):
            lp = log_probs_tbc[ti, b]
            next_beams: Dict[tuple[int, ...], tuple[float, float]] = {}

            def _get(prefix: tuple[int, ...]) -> tuple[float, float]:
                return next_beams.get(prefix, (-float("inf"), -float("inf")))

            for prefix, (p_b, p_nb) in beams.items():
                p_total = _log_add_exp(p_b, p_nb)

                p_b2, p_nb2 = _get(prefix)
                p_b2 = _log_add_exp(p_b2, p_total + float(lp[blank].item()))
                next_beams[prefix] = (p_b2, p_nb2)

                for c in range(vocab):
                    if c == blank:
                        continue
                    p_c = float(lp[c].item())
                    end = prefix[-1] if prefix else None
                    if c == end:
                        new_prefix = prefix
                        p_b2, p_nb2 = _get(new_prefix)
                        p_nb2 = _log_add_exp(p_nb2, p_b + p_c)
                        next_beams[new_prefix] = (p_b2, p_nb2)

                        new_prefix = prefix + (c,)
                        p_b2, p_nb2 = _get(new_prefix)
                        p_nb2 = _log_add_exp(p_nb2, p_nb + p_c)
                        next_beams[new_prefix] = (p_b2, p_nb2)
                    else:
                        new_prefix = prefix + (c,)
                        p_b2, p_nb2 = _get(new_prefix)
                        p_nb2 = _log_add_exp(p_nb2, p_total + p_c)
                        next_beams[new_prefix] = (p_b2, p_nb2)

            beams = dict(
                sorted(next_beams.items(), key=lambda kv: _log_add_exp(kv[1][0], kv[1][1]), reverse=True)[:beam_width]
            )

        best = max(beams.items(), key=lambda kv: _log_add_exp(kv[1][0], kv[1][1]))[0]
        out.append("".join(idx2char.get(i, "") for i in best))

    return out
