from __future__ import annotations

from typing import Dict, List, Tuple


def code128_alphabet_old() -> str:
    return "".join(chr(i) for i in range(32, 127))


def code128_alphabet() -> str:
    tags = ["[EAN13]", "[EAN8]", "[UPCA]", "[CODE39]", "[CODE128]"]
    chars = [chr(i) for i in range(32, 127)]

    full_vocab = tags + chars
    char_to_idx = {char: i for i, char in enumerate(full_vocab)}
    return char_to_idx


def build_vocab_from_alphabet(alphabet: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    # Reserve 0 for the CTC blank. Characters start from 1.
    seen = set()
    deduped: List[str] = []
    for ch in alphabet:
        if ch not in seen:
            seen.add(ch)
            deduped.append(ch)

    char2idx = {ch: i + 1 for i, ch in enumerate(deduped)}
    idx2char = {i: ch for ch, i in char2idx.items()}
    return char2idx, idx2char
