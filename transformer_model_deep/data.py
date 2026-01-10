from __future__ import annotations

import csv
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


try:
    import albumentations as A  # type: ignore
except Exception:
    A = None


try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


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
    def __init__(
        self,
        samples: Sequence[Sample],
        char2idx: Dict[str, int],
        height: int,
        augment: bool = False,
        augment_prob: float = 0.9,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.samples = list(samples)
        self.char2idx = char2idx
        self.height = int(height)
        self.augment = bool(augment)
        self.augment_prob = float(augment_prob)

        self.cache_dir: Optional[Path] = Path(cache_dir).expanduser().resolve() if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._aug = None
        if self.augment:
            if A is None or cv2 is None:
                raise RuntimeError("Augmentations requested but albumentations/opencv not available")

            self._aug = A.Compose(
                [
                    A.Affine(
                        scale=(0.85, 1.15),
                        translate_percent=(0.0, 0.02),
                        rotate=(-18, 18),
                        shear=(-12, 12),
                        interpolation=cv2.INTER_LINEAR,
                        mode=cv2.BORDER_REFLECT_101,
                        p=0.95,
                    ),
                    A.Perspective(scale=(0.02, 0.08), keep_size=True, p=0.6),
                    A.OneOf(
                        [
                            A.MotionBlur(blur_limit=(7, 21), p=1.0),
                            A.GaussianBlur(blur_limit=(3, 11), p=1.0),
                        ],
                        p=0.75,
                    ),
                    A.OneOf(
                        [
                            A.Downscale(scale_min=0.35, scale_max=0.85, interpolation=cv2.INTER_AREA, p=1.0),
                            A.ImageCompression(quality_lower=25, quality_upper=85, p=1.0),
                        ],
                        p=0.8,
                    ),
                ]
            )

    def get_augmenter(self):
        return self._aug

    def load_resized_rgb_u8(self, path: Path) -> np.ndarray:
        if cv2 is None:
            raise RuntimeError("cv2 is required for image loading")

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read image: {path}")

        h, w = bgr.shape[:2]
        scale = self.height / float(h)
        new_w = max(1, int(round(w * scale)))
        interp = cv2.INTER_AREA if new_w < w else cv2.INTER_LINEAR
        bgr = cv2.resize(bgr, (new_w, self.height), interpolation=interp)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def apply_augment_to_rgb_u8(self, rgb_u8: np.ndarray, augmenter: Optional[object] = None) -> np.ndarray:
        aug = augmenter if augmenter is not None else self._aug
        if aug is None:
            return rgb_u8
        if cv2 is None:
            raise RuntimeError("cv2 is required for augmentations")

        bgr = rgb_u8[:, :, ::-1]
        out = aug(image=bgr)
        bgr = out["image"]
        return bgr[:, :, ::-1]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> torch.Tensor:
        try:
            if cv2 is None:
                raise RuntimeError("cv2 is required for image loading")

            bgr = None
            if self.cache_dir is not None:
                key = f"{str(path)}|h={self.height}"
                digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
                cache_path = self.cache_dir / f"{digest}.npy"
                if cache_path.exists():
                    bgr = np.load(str(cache_path))
                else:
                    src = cv2.imread(str(path), cv2.IMREAD_COLOR)
                    if src is None:
                        raise FileNotFoundError(f"Failed to read image: {path}")

                    h, w = src.shape[:2]
                    scale = self.height / float(h)
                    new_w = max(1, int(round(w * scale)))
                    interp = cv2.INTER_AREA if new_w < w else cv2.INTER_LINEAR
                    bgr = cv2.resize(src, (new_w, self.height), interpolation=interp)

                    tmp_dir = str(self.cache_dir)
                    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".npy", dir=tmp_dir)
                    os.close(fd)
                    try:
                        np.save(tmp_path, bgr)
                        os.replace(tmp_path, cache_path)
                    finally:
                        if os.path.exists(tmp_path):
                            try:
                                os.remove(tmp_path)
                            except OSError:
                                pass
            else:
                bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise FileNotFoundError(f"Failed to read image: {path}")

                h, w = bgr.shape[:2]
                scale = self.height / float(h)
                new_w = max(1, int(round(w * scale)))
                interp = cv2.INTER_AREA if new_w < w else cv2.INTER_LINEAR
                bgr = cv2.resize(bgr, (new_w, self.height), interpolation=interp)

            if self._aug is not None and np.random.rand() < self.augment_prob:
                out = self._aug(image=bgr)
                bgr = out["image"]

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            arr = rgb.astype(np.float32) / 255.0
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
