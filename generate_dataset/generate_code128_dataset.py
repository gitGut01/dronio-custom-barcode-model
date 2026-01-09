#!/usr/bin/env python3
 
import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
 
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from tqdm import tqdm
import os
import shutil
 
# --- Dependency Checks ---
try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "opencv-python is required for this script. Install dependencies from requirements.txt"
    ) from e
 
# Specific path check for Mac (Homebrew) or general Linux
gs_path = shutil.which('gs') 
if gs_path:
    os.environ['TREEPOEM_GHOSTSCRIPT'] = gs_path
else:
    print("Warning: Could not find 'gs' executable in PATH. Treepoem might fail.")
 
try:
    import treepoem  # type: ignore
except Exception as e:  # pragma: no cover
    treepoem = None
 
 
@dataclass
class SampleSpec:
    text: str
    codewords: List[int]
 
 
def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
 
 
def _code128_char_to_code_b(ch: str) -> int:
    o = ord(ch)
    if o < 32 or o > 126:
        raise ValueError("Code128 set B supports ASCII 32..126")
    return o - 32
 
 
def _is_digit(ch: str) -> bool:
    return "0" <= ch <= "9"
 
 
def _encode_code128_optimal(text: str) -> List[int]:
    """
    Calculates the optimal codewords for Code 128.
    This is used to populate the CSV labels with ground truth data.
    """
    if not text:
        raise ValueError("text must be non-empty")
 
    for ch in text:
        o = ord(ch)
        if o < 32 or o > 126:
            raise ValueError("All characters must be ASCII 32..126 for this generator")
 
    n = len(text)
    INF = 10**9
 
    dp_b = [INF] * (n + 1)
    dp_c = [INF] * (n + 1)
    back_b: List[Optional[Tuple[str, int, List[int]]]] = [None] * (n + 1)
    back_c: List[Optional[Tuple[str, int, List[int]]]] = [None] * (n + 1)
 
    dp_b[0] = 1
    back_b[0] = ("STARTB", -1, [104])
 
    if n >= 2 and _is_digit(text[0]) and _is_digit(text[1]):
        dp_c[0] = 1
        back_c[0] = ("STARTC", -1, [105])
 
    for i in range(n + 1):
        if dp_b[i] < INF:
            if i < n:
                cw = _code128_char_to_code_b(text[i])
                cost = dp_b[i] + 1
                if cost < dp_b[i + 1]:
                    dp_b[i + 1] = cost
                    back_b[i + 1] = ("B", i, [cw])
 
            if i < n - 1 and _is_digit(text[i]) and _is_digit(text[i + 1]):
                cost = dp_b[i] + 1
                if cost < dp_c[i]:
                    dp_c[i] = cost
                    back_c[i] = ("SWITCH_C", i, [99])
 
        if dp_c[i] < INF:
            if i < n - 1 and _is_digit(text[i]) and _is_digit(text[i + 1]):
                pair = int(text[i : i + 2])
                cost = dp_c[i] + 1
                if cost < dp_c[i + 2]:
                    dp_c[i + 2] = cost
                    back_c[i + 2] = ("C", i, [pair])
            if i < n:
                cost = dp_c[i] + 1
                if cost < dp_b[i]:
                    dp_b[i] = cost
                    back_b[i] = ("SWITCH_B", i, [100])
 
    best_set = "B" if dp_b[n] <= dp_c[n] else "C"
    end_cost = dp_b[n] if best_set == "B" else dp_c[n]
    if end_cost >= INF:
        raise RuntimeError("Failed to encode text")
 
    out_rev: List[int] = []
    pos = n
    cur = best_set
    while True:
        if cur == "B":
            b = back_b[pos]
        else:
            b = back_c[pos]
        if b is None:
            raise RuntimeError("Broken backpointer")
        kind, prev_pos, emitted = b
        out_rev.extend(reversed(emitted))
        if kind in {"STARTB", "STARTC"}:
            break
        if kind == "SWITCH_C":
            cur = "B"
        elif kind == "SWITCH_B":
            cur = "C"
        else:
            pass
        pos = prev_pos
 
    codewords = list(reversed(out_rev))
    if not codewords or codewords[0] not in (104, 105):
        raise RuntimeError("Missing start code")
 
    checksum = codewords[0]
    for i, cw in enumerate(codewords[1:], start=1):
        checksum += cw * i
    checksum %= 103
 
    codewords = codewords + [checksum, 106]
    return codewords
 
 
def _codewords_to_pua(codewords: List[int]) -> str:
    for cw in codewords:
        if cw < 0 or cw > 106:
            raise ValueError("Codeword out of range 0..106")
    return "".join(chr(0xE000 + cw) for cw in codewords)


_CODE128_PATTERNS: List[str] = [
    "212222",
    "222122",
    "222221",
    "121223",
    "121322",
    "131222",
    "122213",
    "122312",
    "132212",
    "221213",
    "221312",
    "231212",
    "112232",
    "122132",
    "122231",
    "113222",
    "123122",
    "123221",
    "223211",
    "221132",
    "221231",
    "213212",
    "223112",
    "312131",
    "311222",
    "321122",
    "321221",
    "312212",
    "322112",
    "322211",
    "212123",
    "212321",
    "232121",
    "111323",
    "131123",
    "131321",
    "112313",
    "132113",
    "132311",
    "211313",
    "231113",
    "231311",
    "112133",
    "112331",
    "132131",
    "113123",
    "113321",
    "133121",
    "313121",
    "211331",
    "231131",
    "213113",
    "213311",
    "213131",
    "311123",
    "311321",
    "331121",
    "312113",
    "312311",
    "332111",
    "314111",
    "221411",
    "431111",
    "111224",
    "111422",
    "121124",
    "121421",
    "141122",
    "141221",
    "112214",
    "112412",
    "122114",
    "122411",
    "142112",
    "142211",
    "241211",
    "221114",
    "413111",
    "241112",
    "134111",
    "111242",
    "121142",
    "121241",
    "114212",
    "124112",
    "124211",
    "411212",
    "421112",
    "421211",
    "212141",
    "214121",
    "412121",
    "111143",
    "111341",
    "131141",
    "114113",
    "114311",
    "411113",
    "411311",
    "113141",
    "114131",
    "311141",
    "411131",
    "211412",
    "211214",
    "211232",
    "2331112",
]


def _render_code128_from_codewords(codewords: List[int]) -> Image.Image:
    if not codewords or codewords[-1] != 106:
        raise ValueError("Expected codewords to include Stop (106)")
    for cw in codewords:
        if cw < 0 or cw > 106:
            raise ValueError("Codeword out of range 0..106")

    module = random.choice([2, 3, 4])
    height = random.randint(60, 120)
    quiet = 10 * module

    total_modules = 0
    for cw in codewords:
        total_modules += sum(int(d) for d in _CODE128_PATTERNS[cw])
    w = quiet * 2 + total_modules * module
    h = height

    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    x = quiet

    for cw in codewords:
        pattern = _CODE128_PATTERNS[cw]
        is_bar = True
        for d in pattern:
            width = int(d) * module
            if is_bar:
                draw.rectangle((x, 0, x + width - 1, h), fill=(0, 0, 0))
            x += width
            is_bar = not is_bar

    return img


def _render_code128_raw(codewords: List[int]) -> Image.Image:
    if treepoem is not None:
        data = " ".join(str(int(x)) for x in codewords)
        try:
            img = treepoem.generate_barcode(
                barcode_type="code128raw",
                data=data,
                options={
                    "includetext": False,
                },
            )
            return img.convert("RGB")
        except Exception:
            pass

    return _render_code128_from_codewords(codewords)
 
 
def random_background(size: Tuple[int, int]) -> Image.Image:
    w, h = size
    choice = random.random()
 
    if choice < 0.4:
        base = Image.new("RGB", (w, h), tuple(int(x) for x in np.random.randint(0, 255, size=3)))
        enhancer = ImageEnhance.Brightness(base)
        return enhancer.enhance(random.uniform(1.1, 1.6))
 
    if choice < 0.75:
        arr = np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
        arr = cv2.GaussianBlur(arr, (0, 0), sigmaX=random.uniform(1.0, 4.0))
        arr = cv2.addWeighted(arr, 0.35, np.full_like(arr, 235), 0.65, 0)
        return Image.fromarray(arr, mode="RGB")
 
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    c1 = np.random.randint(80, 240, size=3)
    c2 = np.random.randint(80, 240, size=3)
    for y in range(h):
        t = y / max(1, h - 1)
        arr[y, :, :] = (c1 * (1 - t) + c2 * t).astype(np.uint8)
    arr = cv2.GaussianBlur(arr, (0, 0), sigmaX=random.uniform(0.5, 2.0))
    return Image.fromarray(arr, mode="RGB")
 
 
def place_barcode_on_canvas(
    barcode_img: Image.Image,
    canvas_size: Tuple[int, int],
    max_margin_ratio: float,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    canvas_w, canvas_h = canvas_size
    bg = random_background((canvas_w, canvas_h))
 
    b = barcode_img.copy()
 
    min_b_w = int(math.ceil(canvas_w / (1.0 + 2.0 * max_margin_ratio)))
    min_b_h = int(math.ceil(canvas_h / (1.0 + 2.0 * max_margin_ratio)))
 
    bw, bh = b.size
    scale = max(min_b_w / bw, min_b_h / bh)
    scale *= random.uniform(1.0, 1.15)
    new_w = min(canvas_w, max(1, int(round(bw * scale))))
    new_h = min(canvas_h, max(1, int(round(bh * scale))))
    b = b.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
 
    max_mx = int(round(max_margin_ratio * new_w))
    max_my = int(round(max_margin_ratio * new_h))
 
    left_margin = random.randint(0, min(max_mx, max(0, canvas_w - new_w)))
    top_margin = random.randint(0, min(max_my, max(0, canvas_h - new_h)))
 
    x0 = left_margin
    y0 = top_margin
    x1 = x0 + new_w
    y1 = y0 + new_h
 
    if x1 > canvas_w:
        x0 = max(0, (canvas_w - new_w) // 2)
        x1 = x0 + new_w
    if y1 > canvas_h:
        y0 = max(0, (canvas_h - new_h) // 2)
        y1 = y0 + new_h
 
    bg.paste(b, (x0, y0))
    return bg, (x0, y0, x1, y1)
 
 
def _add_reflection(pil_img: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    x0, y0, x1, y1 = bbox
    img = pil_img.copy()
    draw = ImageDraw.Draw(img, "RGBA")
 
    if random.random() < 0.5:
        band_w = random.randint(max(10, (x1 - x0) // 10), max(20, (x1 - x0) // 4))
        alpha = random.randint(30, 90)
        x_start = random.randint(x0 - band_w, x1)
        poly = [
            (x_start, y0),
            (x_start + band_w, y0),
            (x_start + band_w + (x1 - x0) // 4, y1),
            (x_start + (x1 - x0) // 4, y1),
        ]
        draw.polygon(poly, fill=(255, 255, 255, alpha))
    else:
        cx = random.randint(x0, x1)
        cy = random.randint(y0, y1)
        r = random.randint(max(8, (x1 - x0) // 20), max(16, (x1 - x0) // 8))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 255, 255, random.randint(40, 110)))
 
    return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
 
 
def _damage_barcode(pil_img: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    x0, y0, x1, y1 = bbox
    img = pil_img.copy()
    draw = ImageDraw.Draw(img, "RGBA")
 
    n = random.randint(1, 6)
    for _ in range(n):
        if random.random() < 0.5:
            x = random.randint(x0, x1)
            y = random.randint(y0, y1)
            x2 = x + random.randint(-80, 80)
            y2 = y + random.randint(-20, 20)
            width = random.randint(1, 4)
            draw.line((x, y, x2, y2), fill=(255, 255, 255, random.randint(60, 160)), width=width)
        else:
            bx0 = random.randint(x0, max(x0, x1 - 10))
            by0 = random.randint(y0, max(y0, y1 - 10))
            bx1 = min(x1, bx0 + random.randint(6, max(8, (x1 - x0) // 8)))
            by1 = min(y1, by0 + random.randint(6, max(8, (y1 - y0) // 8)))
            draw.rectangle((bx0, by0, bx1, by1), fill=(255, 255, 255, random.randint(50, 140)))
 
    return img
 
 
def _low_light(pil_img: Image.Image) -> Image.Image:
    img = pil_img.copy()
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.35, 0.9))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.2))
    return img
 
 
def apply_pipeline(
    pil_img: Image.Image,
    bbox: Tuple[int, int, int, int],
) -> Image.Image:
    if random.random() < 0.65:
        pil_img = _low_light(pil_img)
    if random.random() < 0.55:
        pil_img = _damage_barcode(pil_img, bbox)
    if random.random() < 0.5:
        pil_img = _add_reflection(pil_img, bbox)
 
    np_img = np.array(pil_img)[:, :, ::-1].copy()
    rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
 
 
def _sample_size(
    size_min: int,
    size_max: int,
    landscape_prob: float,
    aspect_min: float,
    aspect_max: float,
) -> Tuple[int, int]:
    base = random.randint(size_min, size_max)
    aspect = random.uniform(aspect_min, aspect_max)
    landscape = random.random() < landscape_prob
 
    if landscape:
        w = int(round(base * aspect))
        h = base
    else:
        w = base
        h = int(round(base * aspect))
 
    return max(96, w), max(96, h)
 
 
_DIGITS = "0123456789"
_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_ALNUM = _UPPER + _DIGITS


def _rand_from(alphabet: str, n: int) -> str:
    return "".join(random.choice(alphabet) for _ in range(n))


def _random_text_warehouse(min_len: int, max_len: int) -> str:
    r = random.random()

    # numeric-only (dominant; encourages Set C)
    if r < 0.85:
        n = random.randint(max(min_len, 10), min(max_len, 30))
        if random.random() < 0.8 and (n % 2 == 1):
            n = min(max_len, n + 1)
        return _rand_from(_DIGITS, n)

    # uppercase prefix + digits (common mixed format)
    if r < 0.95:
        prefix_len = random.choice([2, 3, 4])
        digits_len = random.randint(8, 20)
        s = _rand_from(_UPPER, prefix_len) + _rand_from(_DIGITS, digits_len)
        return s[:max_len]

    # mixed uppercase alphanumeric
    n = random.randint(max(min_len, 6), min(max_len, 18))
    return _rand_from(_ALNUM, n)
 
 
def _make_sample(min_len: int, max_len: int) -> SampleSpec:
    text = _random_text_warehouse(min_len, max_len)
    codewords = _encode_code128_optimal(text)
    return SampleSpec(text=text, codewords=codewords)
 
 
def generate_split(
    out_dir: Path,
    split: str,
    n: int,
    seed: int,
    size_min: int,
    size_max: int,
    max_margin_ratio: float,
    landscape_prob: float,
    aspect_min: float,
    aspect_max: float,
    min_len: int,
    max_len: int,
) -> None:
    images_dir = out_dir / split / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / split / "labels.csv"
 
    _seed_everything(seed)
 
    with labels_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "symbology", "value", "human_text", "codewords"])
 
        for i in tqdm(range(n), desc=f"{split}"):
            spec = _make_sample(min_len=min_len, max_len=max_len)
            barcode_img = _render_code128_raw(spec.codewords)
 
            canvas_size = _sample_size(
                size_min=size_min,
                size_max=size_max,
                landscape_prob=landscape_prob,
                aspect_min=aspect_min,
                aspect_max=aspect_max,
            )
 
            composed, bbox = place_barcode_on_canvas(
                barcode_img=barcode_img,
                canvas_size=canvas_size,
                max_margin_ratio=max_margin_ratio,
            )
 
            final_img = apply_pipeline(composed, bbox=bbox)
 
            filename = f"{split}_{i:07d}.png"
            final_img.save(images_dir / filename, format="PNG", optimize=True)
 
            value = _codewords_to_pua(spec.codewords)
            writer.writerow(
                [
                    f"images/{filename}",
                    "code128",
                    value,
                    spec.text,
                    " ".join(str(x) for x in spec.codewords),
                ]
            )
 
 
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="code128_dataset")
    ap.add_argument("--train", type=int, default=20)
    ap.add_argument("--val", type=int, default=20)
    ap.add_argument("--test", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)
 
    ap.add_argument("--size-min", type=int, default=256)
    ap.add_argument("--size-max", type=int, default=640)
    ap.add_argument("--max-margin-ratio", type=float, default=0.25)
    ap.add_argument("--landscape-prob", type=float, default=0.7)
    ap.add_argument("--aspect-min", type=float, default=1.2)
    ap.add_argument("--aspect-max", type=float, default=3.2)
 
    ap.add_argument("--min-len", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=20)
 
    args = ap.parse_args()
 
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    generate_split(
        out_dir=out_dir,
        split="train",
        n=int(args.train),
        seed=int(args.seed),
        size_min=int(args.size_min),
        size_max=int(args.size_max),
        max_margin_ratio=float(args.max_margin_ratio),
        landscape_prob=float(args.landscape_prob),
        aspect_min=float(args.aspect_min),
        aspect_max=float(args.aspect_max),
        min_len=int(args.min_len),
        max_len=int(args.max_len),
    )
    generate_split(
        out_dir=out_dir,
        split="val",
        n=int(args.val),
        seed=int(args.seed) + 1,
        size_min=int(args.size_min),
        size_max=int(args.size_max),
        max_margin_ratio=float(args.max_margin_ratio),
        landscape_prob=float(args.landscape_prob),
        aspect_min=float(args.aspect_min),
        aspect_max=float(args.aspect_max),
        min_len=int(args.min_len),
        max_len=int(args.max_len),
    )
    generate_split(
        out_dir=out_dir,
        split="test",
        n=int(args.test),
        seed=int(args.seed) + 2,
        size_min=int(args.size_min),
        size_max=int(args.size_max),
        max_margin_ratio=float(args.max_margin_ratio),
        landscape_prob=float(args.landscape_prob),
        aspect_min=float(args.aspect_min),
        aspect_max=float(args.aspect_max),
        min_len=int(args.min_len),
        max_len=int(args.max_len),
    )
 
 
if __name__ == "__main__":
    main()