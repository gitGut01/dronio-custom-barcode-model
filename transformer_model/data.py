from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    image_path: Path
    symbology: str
    value: str


def read_labels_csv(dataset_root: Path, split: str) -> List[Sample]:
    labels_path = dataset_root / split / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels.csv: {labels_path}")

    samples: List[Sample] = []
    print(f"--- Loading {split} dataset ---")
    with labels_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            samples.append(
                Sample(
                    image_path=dataset_root / split / row["filename"],
                    symbology=row.get("symbology", ""),
                    value=row.get("value", ""),
                )
            )
            if (i + 1) % 250000 == 0:
                print(f"  > Processed {i + 1} rows...")
    print(f"Successfully loaded {len(samples)} samples for {split}.")
    return samples


class BarcodeCtcDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], char2idx: Dict[str, int], height: int) -> None:
        self.samples = list(samples)
        self.char2idx = char2idx
        self.height = int(height)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> torch.Tensor:
        try:
            img = Image.open(path).convert("RGB")
            w, h = img.size
            scale = self.height / float(h)
            new_w = max(1, int(round(w * scale)))
            img = img.resize((new_w, self.height), resample=Image.Resampling.BILINEAR)
            arr = np.asarray(img).astype(np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))
            return torch.from_numpy(arr)
        except Exception:
            return torch.ones((3, self.height, self.height * 2))

    def _encode(self, text: str) -> torch.Tensor:
        ids = [self.char2idx[ch] for ch in text if ch in self.char2idx]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        s = self.samples[idx]
        return self._load_image(s.image_path), self._encode(s.value), s.symbology, s.value


def ctc_collate(batch):
    xs, ys, syms, values = zip(*batch)
    x_w_lens = torch.tensor([x.shape[-1] for x in xs], dtype=torch.long)
    max_w = max(x.shape[-1] for x in xs)
    padded_xs = [F.pad(x, (0, max_w - x.shape[-1], 0, 0), value=1.0) for x in xs]
    x_batch = torch.stack(padded_xs, dim=0)
    y_lens = torch.tensor([y.numel() for y in ys], dtype=torch.long)
    y_concat = torch.cat(ys, dim=0) if len(ys) > 0 else torch.empty((0,), dtype=torch.long)
    return x_batch, x_w_lens, y_concat, y_lens, list(syms), list(values)
