#!/usr/bin/env python3

import argparse
import csv
import math
import os
import random
import shutil
import string
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm import tqdm


try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "opencv-python is required for this script. Install dependencies from requirements.txt"
    ) from e


try:
    import barcode  # type: ignore
    from barcode.writer import ImageWriter  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "python-barcode is required for this script. Install dependencies from requirements.txt"
    ) from e


try:
    import albumentations as A  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "albumentations is required for this script. Install dependencies from requirements.txt"
    ) from e


try:
    import treepoem  # type: ignore

    _HAS_TREEPOEM = True
except Exception:
    _HAS_TREEPOEM = False


SUPPORTED_SYMBOLOGIES = {
    "ean13": "ean13",
    "ean8": "ean8",
    "upca": "upc",
    "code128": "code128",
    "code39": "code39",
    "itf": "itf",
    # gs1-databar (optional via treepoem)
    "gs1_databar": "gs1databaromni",
}


@dataclass
class SampleSpec:
    symbology: str
    value: str


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _digits(n: int) -> str:
    return "".join(random.choice(string.digits) for _ in range(n))


def _alnum(n: int) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(n))



def code128_alphabet() -> str:
    return "".join(chr(i) for i in range(32, 127))


def _code128_chars() -> str:
    return code128_alphabet()


def _code128_random(n: int) -> str:
    alphabet = _code128_chars()
    return "".join(random.choice(alphabet) for _ in range(n))


def _random_value(sym: str) -> str:
    if sym == "ean13":
        # python-barcode computes checksum if omit last digit, but easiest is provide 12 digits.
        return _digits(12)
    if sym == "ean8":
        return _digits(7)
    if sym == "upca":
        return _digits(11)
    if sym == "code128":
        return _code128_random(random.randint(6, 18))
    if sym == "code39":
        # Code39 traditionally supports A-Z 0-9 space - . $ / + %
        alphabet = string.ascii_uppercase + string.digits + "-. $/+%"
        return "".join(random.choice(alphabet) for _ in range(random.randint(6, 18))).strip()
    if sym == "itf":
        # ITF generally needs even number of digits
        length = random.choice([6, 8, 10, 12, 14])
        if length % 2 == 1:
            length += 1
        return _digits(length)
    if sym == "gs1_databar":
        # For treepoem / BWIPP you can often pass a GTIN (14) or similar.
        return _digits(14)

    raise ValueError(f"Unsupported symbology: {sym}")


def _render_barcode_python_barcode(
    sym: str,
    value: str,
    module_width: float,
    write_text: bool,
) -> Image.Image:
    writer = ImageWriter()
    module_height = random.randint(30, 80)
    font_size = random.randint(8, 16)
    text_distance = 0
    if write_text:
        # Give the label some room so it doesn't touch/overlap the bars.
        # (Downstream resizing/blur can otherwise make it look like overlap.)
        module_height = random.randint(24, 70)
        font_size = random.randint(10, 18)
        text_distance = random.randint(6, 12)

    # Tweak writer options for a crisp base image
    writer.set_options(
        {
            "module_width": module_width,
            "module_height": module_height,
            "quiet_zone": 2.0,
            "font_size": font_size,
            "text_distance": text_distance,
            "write_text": write_text,
            "background": "white",
            "foreground": "black",
        }
    )

    bclass = barcode.get_barcode_class(SUPPORTED_SYMBOLOGIES[sym])
    b = bclass(value, writer=writer)
    pil_img = b.render(
        writer_options={
            "write_text": write_text,
            "text_distance": text_distance,
            "font_size": font_size,
            "module_height": module_height,
        }
    )
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return pil_img


def _render_barcode_treepoem(sym: str, value: str) -> Image.Image:
    if not _HAS_TREEPOEM:
        raise RuntimeError(
            "treepoem is not installed; install it (and Ghostscript) to enable gs1_databar"
        )

    # BWIPP names
    bwipp_name = SUPPORTED_SYMBOLOGIES[sym]
    img = treepoem.generate_barcode(
        barcode_type=bwipp_name,
        data=value,
        options={
            "includetext": False,
        },
    )
    pil_img = img.convert("RGB")
    return pil_img


def render_barcode(sym: str, value: str, under_text_prob: float) -> Image.Image:
    # module_width influences density of bars for python-barcode
    module_width = random.choice([0.18, 0.2, 0.25, 0.3, 0.35])

    if sym == "gs1_databar":
        img = _render_barcode_treepoem(sym, value)
        return img

    write_text = under_text_prob > 0 and random.random() < under_text_prob
    img = _render_barcode_python_barcode(
        sym,
        value,
        module_width=module_width,
        write_text=write_text,
    )
    return img


def random_background(size: Tuple[int, int]) -> Image.Image:
    w, h = size
    choice = random.random()

    if choice < 0.4:
        # solid
        base = Image.new("RGB", (w, h), tuple(int(x) for x in np.random.randint(0, 255, size=3)))
        # lighten it a bit to keep barcode visible
        enhancer = ImageEnhance.Brightness(base)
        return enhancer.enhance(random.uniform(1.1, 1.6))

    if choice < 0.75:
        # mild texture (noise)
        arr = np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
        arr = cv2.GaussianBlur(arr, (0, 0), sigmaX=random.uniform(1.0, 4.0))
        arr = cv2.addWeighted(arr, 0.35, np.full_like(arr, 235), 0.65, 0)
        return Image.fromarray(arr, mode="RGB")

    # gradient-ish background
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
    """Return composed image and barcode bounding box on the canvas."""

    canvas_w, canvas_h = canvas_size
    bg = random_background((canvas_w, canvas_h))

    b = barcode_img.copy()

    # Ensure barcode occupies majority of image: margin <= max_margin_ratio * barcode size
    # Equivalent: canvas_dim <= barcode_dim * (1 + 2*max_margin_ratio)
    # So barcode_dim >= canvas_dim / (1 + 2*max_margin_ratio)
    min_b_w = int(math.ceil(canvas_w / (1.0 + 2.0 * max_margin_ratio)))
    min_b_h = int(math.ceil(canvas_h / (1.0 + 2.0 * max_margin_ratio)))

    # Resize barcode to satisfy min dims but preserve aspect ratio.
    bw, bh = b.size
    scale = max(min_b_w / bw, min_b_h / bh)
    # Also allow some variation up to near full frame
    scale *= random.uniform(1.0, 1.15)
    new_w = min(canvas_w, max(1, int(round(bw * scale))))
    new_h = min(canvas_h, max(1, int(round(bh * scale))))
    b = b.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)

    # Choose margins within constraint.
    max_mx = int(round(max_margin_ratio * new_w))
    max_my = int(round(max_margin_ratio * new_h))

    left_margin = random.randint(0, min(max_mx, max(0, canvas_w - new_w)))
    top_margin = random.randint(0, min(max_my, max(0, canvas_h - new_h)))

    x0 = left_margin
    y0 = top_margin
    x1 = x0 + new_w
    y1 = y0 + new_h

    # If barcode doesn't fit due to rounding, center it
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
        # diagonal specular highlight band over barcode region
        band_w = random.randint(max(10, (x1 - x0) // 10), max(20, (x1 - x0) // 4))
        alpha = random.randint(30, 90)
        # create polygon across bbox
        x_start = random.randint(x0 - band_w, x1)
        poly = [
            (x_start, y0),
            (x_start + band_w, y0),
            (x_start + band_w + (x1 - x0) // 4, y1),
            (x_start + (x1 - x0) // 4, y1),
        ]
        draw.polygon(poly, fill=(255, 255, 255, alpha))
    else:
        # small glare spot
        cx = random.randint(x0, x1)
        cy = random.randint(y0, y1)
        r = random.randint(max(8, (x1 - x0) // 20), max(16, (x1 - x0) // 8))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 255, 255, random.randint(40, 110)))

    return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))


def _damage_barcode(pil_img: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    x0, y0, x1, y1 = bbox
    img = pil_img.copy()
    draw = ImageDraw.Draw(img, "RGBA")

    # scratches / occlusions inside bbox
    n = random.randint(1, 6)
    for _ in range(n):
        if random.random() < 0.5:
            # scratch line
            x = random.randint(x0, x1)
            y = random.randint(y0, y1)
            x2 = x + random.randint(-80, 80)
            y2 = y + random.randint(-20, 20)
            width = random.randint(1, 4)
            draw.line((x, y, x2, y2), fill=(255, 255, 255, random.randint(60, 160)), width=width)
        else:
            # blotch
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


def _grain_and_artifacts(np_img: np.ndarray) -> np.ndarray:
    # Add grain
    if random.random() < 0.9:
        sigma = random.uniform(3.0, 18.0)
        noise = np.random.normal(0, sigma, size=np_img.shape).astype(np.float32)
        np_img = np.clip(np_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # JPEG compression artifacts
    if random.random() < 0.8:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(25, 85)]
        ok, enc = cv2.imencode(".jpg", np_img, encode_param)
        if ok:
            np_img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # Chromatic aberration (small channel shift)
    if random.random() < 0.35:
        shift = random.randint(1, 3)
        b, g, r = cv2.split(np_img)
        r = np.roll(r, shift, axis=1)
        b = np.roll(b, -shift, axis=0)
        np_img = cv2.merge([b, g, r])

    return np_img


def _rolling_shutter_warp(
    np_img: np.ndarray,
    max_shift_px: float,
    wobble_px: float,
    freq_min: float,
    freq_max: float,
) -> np.ndarray:
    h, w = np_img.shape[:2]
    if h <= 1 or w <= 1:
        return np_img

    max_shift_px = float(max(0.0, max_shift_px))
    wobble_px = float(max(0.0, wobble_px))
    if max_shift_px == 0.0 and wobble_px == 0.0:
        return np_img

    t = (np.arange(h, dtype=np.float32) / float(h - 1)) - 0.5
    base = t * np.float32(random.uniform(-max_shift_px, max_shift_px))
    dx = base
    if wobble_px > 0.0:
        freq = float(random.uniform(freq_min, freq_max))
        phase = float(random.uniform(0.0, 2.0 * math.pi))
        dx = dx + np.float32(wobble_px) * np.sin((t + 0.5) * (2.0 * math.pi * freq) + phase).astype(
            np.float32
        )

    map_x = (np.arange(w, dtype=np.float32)[None, :] + dx[:, None]).astype(np.float32)
    map_y = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1).astype(np.float32)

    return cv2.remap(
        np_img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def _directional_motion_blur(
    np_img: np.ndarray,
    length_min: int,
    length_max: int,
    angle_bias_deg: float,
    angle_jitter_deg: float,
) -> np.ndarray:
    length_min = int(max(1, length_min))
    length_max = int(max(length_min, length_max))
    k = int(random.randint(length_min, length_max))
    if k <= 1:
        return np_img
    if k % 2 == 0:
        k += 1

    angle = float(angle_bias_deg + random.uniform(-angle_jitter_deg, angle_jitter_deg))
    theta = math.radians(angle)

    kernel = np.zeros((k, k), dtype=np.float32)
    cx = (k - 1) / 2.0
    cy = (k - 1) / 2.0
    dx = math.cos(theta) * (k - 1) / 2.0
    dy = math.sin(theta) * (k - 1) / 2.0
    x0 = int(round(cx - dx))
    y0 = int(round(cy - dy))
    x1 = int(round(cx + dx))
    y1 = int(round(cy + dy))
    cv2.line(kernel, (x0, y0), (x1, y1), 1.0, thickness=1)

    s = float(kernel.sum())
    if s <= 0.0:
        return np_img
    kernel /= s

    return cv2.filter2D(np_img, ddepth=-1, kernel=kernel)


def build_augmentation_pipeline() -> A.Compose:
    # Directional motion blur + skew + resize/aspect changes
    return A.Compose(
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


def apply_pipeline(
    pil_img: Image.Image,
    bbox: Tuple[int, int, int, int],
    out_size: Tuple[int, int],
    video_artifacts_prob: float,
    rolling_shutter_prob: float,
    rolling_shutter_max_shift_px: float,
    rolling_shutter_wobble_px: float,
    motion_blur_prob: float,
    motion_blur_len_min: int,
    motion_blur_len_max: int,
    motion_blur_angle_bias_deg: float,
    motion_blur_angle_jitter_deg: float,
) -> Image.Image:
    # PIL-side effects
    if random.random() < 0.65:
        pil_img = _low_light(pil_img)
    if random.random() < 0.55:
        pil_img = _damage_barcode(pil_img, bbox)
    if random.random() < 0.5:
        pil_img = _add_reflection(pil_img, bbox)

    # Albumentations / CV side
    np_img = np.array(pil_img)[:, :, ::-1].copy()  # BGR

    aug = build_augmentation_pipeline()
    np_img = aug(image=np_img)["image"]

    do_video = random.random() < float(np.clip(video_artifacts_prob, 0.0, 1.0))
    if do_video:
        if random.random() < float(np.clip(rolling_shutter_prob, 0.0, 1.0)):
            np_img = _rolling_shutter_warp(
                np_img,
                max_shift_px=float(rolling_shutter_max_shift_px),
                wobble_px=float(rolling_shutter_wobble_px),
                freq_min=1.0,
                freq_max=3.0,
            )

        if random.random() < float(np.clip(motion_blur_prob, 0.0, 1.0)):
            np_img = _directional_motion_blur(
                np_img,
                length_min=int(motion_blur_len_min),
                length_max=int(motion_blur_len_max),
                angle_bias_deg=float(motion_blur_angle_bias_deg),
                angle_jitter_deg=float(motion_blur_angle_jitter_deg),
            )

    np_img = _grain_and_artifacts(np_img)

    # Final resize to requested output size
    np_img = cv2.resize(np_img, out_size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _sample_size(
    size_min: int,
    size_max: int,
    landscape_prob: float,
    aspect_min: float,
    aspect_max: float,
) -> Tuple[int, int]:
    if size_min <= 0 or size_max <= 0:
        raise ValueError("size_min and size_max must be > 0")
    if size_min > size_max:
        raise ValueError("size_min must be <= size_max")
    if not (0.0 <= landscape_prob <= 1.0):
        raise ValueError("landscape_prob must be in [0, 1]")
    if aspect_min < 1.0 or aspect_max < 1.0 or aspect_min > aspect_max:
        raise ValueError("aspect_min/aspect_max must be >= 1 and aspect_min <= aspect_max")

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


def generate_split(
    out_dir: Path,
    split: str,
    n: int,
    size_min: int,
    size_max: int,
    sym_probs: Dict[str, float],
    seed: int,
    max_margin_ratio: float,
    under_text_prob: float,
    landscape_prob: float,
    aspect_min: float,
    aspect_max: float,
    video_artifacts_prob: float,
    rolling_shutter_prob: float,
    rolling_shutter_max_shift_px: float,
    rolling_shutter_wobble_px: float,
    motion_blur_prob: float,
    motion_blur_len_min: int,
    motion_blur_len_max: int,
    motion_blur_angle_bias_deg: float,
    motion_blur_angle_jitter_deg: float,
) -> None:
    images_dir = out_dir / split / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    labels_path = out_dir / split / "labels.csv"

    # normalize weights
    keys = list(sym_probs.keys())
    weights = np.array([sym_probs[k] for k in keys], dtype=np.float64)
    if (weights <= 0).any():
        raise ValueError("All symbology probabilities must be > 0")
    weights = weights / weights.sum()

    skipped_gs1 = False

    with labels_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "symbology", "value"])

        for i in tqdm(range(n), desc=f"{split}"):
            sym = str(np.random.choice(keys, p=weights))
            if sym == "gs1_databar" and not _HAS_TREEPOEM:
                skipped_gs1 = True
                sym = random.choice([s for s in keys if s != "gs1_databar"])

            value = _random_value(sym)
            spec = SampleSpec(symbology=sym, value=value)

            canvas_size = _sample_size(
                size_min=size_min,
                size_max=size_max,
                landscape_prob=landscape_prob,
                aspect_min=aspect_min,
                aspect_max=aspect_max,
            )

            # generate and compose
            barcode_img = render_barcode(spec.symbology, spec.value, under_text_prob=under_text_prob)
            composed, bbox = place_barcode_on_canvas(
                barcode_img=barcode_img,
                canvas_size=canvas_size,
                max_margin_ratio=max_margin_ratio,
            )

            out_size = _sample_size(
                size_min=size_min,
                size_max=size_max,
                landscape_prob=landscape_prob,
                aspect_min=aspect_min,
                aspect_max=aspect_max,
            )

            final_img = apply_pipeline(
                composed,
                bbox=bbox,
                out_size=out_size,
                video_artifacts_prob=float(video_artifacts_prob),
                rolling_shutter_prob=float(rolling_shutter_prob),
                rolling_shutter_max_shift_px=float(rolling_shutter_max_shift_px),
                rolling_shutter_wobble_px=float(rolling_shutter_wobble_px),
                motion_blur_prob=float(motion_blur_prob),
                motion_blur_len_min=int(motion_blur_len_min),
                motion_blur_len_max=int(motion_blur_len_max),
                motion_blur_angle_bias_deg=float(motion_blur_angle_bias_deg),
                motion_blur_angle_jitter_deg=float(motion_blur_angle_jitter_deg),
            )

            filename = f"{split}_{i:07d}.jpg"
            final_img.save(
                images_dir / filename,
                format="JPEG",
                quality=85,
                optimize=True,
                progressive=True,
            )
            writer.writerow([f"images/{filename}", spec.symbology, spec.value])

    if skipped_gs1:
        warnings.warn(
            "gs1_databar requested but treepoem/Ghostscript not available; those samples were generated using other symbologies instead.",
            stacklevel=1,
        )


def parse_symbology_probs(s: str) -> Dict[str, float]:
    """Format: ean13=1,ean8=1,upca=1,code128=1,code39=1,itf=1,gs1_databar=0.5"""
    out: Dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split("=")
        k = k.strip()
        v = v.strip()
        if k not in SUPPORTED_SYMBOLOGIES:
            raise ValueError(f"Unknown symbology '{k}'. Supported: {sorted(SUPPORTED_SYMBOLOGIES.keys())}")
        out[k] = float(v)
    if not out:
        raise ValueError("No symbology probabilities provided")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic 1D barcode dataset")
    ap.add_argument("--out", type=str, default="my_dataset", help="Output directory")
    ap.add_argument("--train", type=int, default=0, help="Number of training samples")
    ap.add_argument("--val", type=int, default=0, help="Number of validation samples")
    ap.add_argument("--test", type=int, default=0, help="Number of test samples")
    ap.add_argument(
        "--zip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Zip the output directory after generation (creates <out>.zip)",
    )
    ap.add_argument("--size-min", type=int, default=80, help="Min side length for generated images")
    ap.add_argument("--size-max", type=int, default=300, help="Max side length for generated images")
    ap.add_argument(
        "--under-text-prob",
        type=float,
        default=0.30,
        help="Probability to render the human-readable value under the barcode (python-barcode only)",
    )
    ap.add_argument(
        "--max-margin-ratio",
        type=float,
        default=0.2,
        help="Max margin as fraction of barcode size (0.5 means margin <= 0.5 * barcode dimension)",
    )
    ap.add_argument(
        "--landscape-prob",
        type=float,
        default=1.0,
        help="Probability that generated images are wider than tall",
    )
    ap.add_argument(
        "--aspect-min",
        type=float,
        default=1.25,
        help="Minimum aspect ratio (>=1). With --landscape-prob, controls how wide images tend to be",
    )
    ap.add_argument(
        "--aspect-max",
        type=float,
        default=2.0,
        help="Maximum aspect ratio (>=1). With --landscape-prob, clamps how wide/tall images can get",
    )
    ap.add_argument("--seed", type=int, default=1337, help="Random seed")
    ap.add_argument(
        "--symbology-probs",
        type=str,
        default="ean13=0.08,ean8=0.06,upca=0.08,itf=0.06,code128=0.50,code39=0.20",
        help="Comma-separated probabilities per symbology",
    )

    ap.add_argument(
        "--video-artifacts-prob",
        type=float,
        default=0.70,
        help="Fraction of samples that receive additional video-like artifacts (rolling shutter + directional motion blur)",
    )
    ap.add_argument(
        "--rolling-shutter-prob",
        type=float,
        default=0.3,
        help="Probability of rolling shutter given video artifacts are enabled for a sample",
    )
    ap.add_argument(
        "--rolling-shutter-max-shift-px",
        type=float,
        default=10.0,
        help="Maximum row-dependent horizontal shift in pixels for rolling shutter",
    )
    ap.add_argument(
        "--rolling-shutter-wobble-px",
        type=float,
        default=1.0,
        help="Additional sinusoidal wobble amplitude in pixels for rolling shutter",
    )
    ap.add_argument(
        "--dir-motion-blur-prob",
        type=float,
        default=0.6,
        help="Probability of directional motion blur given video artifacts are enabled for a sample",
    )
    ap.add_argument(
        "--dir-motion-blur-len-min",
        type=int,
        default=1,
        help="Minimum kernel length (pixels) for directional motion blur",
    )
    ap.add_argument(
        "--dir-motion-blur-len-max",
        type=int,
        default=2,
        help="Maximum kernel length (pixels) for directional motion blur",
    )
    ap.add_argument(
        "--dir-motion-blur-angle-bias-deg",
        type=float,
        default=0.0,
        help="Angle bias in degrees for directional motion blur (0 means horizontal blur)",
    )
    ap.add_argument(
        "--dir-motion-blur-angle-jitter-deg",
        type=float,
        default=0.0,
        help="Angle jitter in degrees around the bias for directional motion blur",
    )

    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    _seed_everything(args.seed)

    sym_probs = parse_symbology_probs(args.symbology_probs)

    # Warn early if user requested gs1_databar but it isn't available
    if ("gs1_databar" in sym_probs) and (not _HAS_TREEPOEM):
        warnings.warn(
            "treepoem is not installed; gs1_databar will be skipped. Install treepoem + Ghostscript to enable it.",
            stacklevel=1,
        )

    # Use different seeds per split for stability
    generate_split(
        out_dir=out_dir,
        split="train",
        n=int(args.train),
        size_min=int(args.size_min),
        size_max=int(args.size_max),
        sym_probs=sym_probs,
        seed=args.seed + 1,
        max_margin_ratio=float(args.max_margin_ratio),
        under_text_prob=float(args.under_text_prob),
        landscape_prob=float(args.landscape_prob),
        aspect_min=float(args.aspect_min),
        aspect_max=float(args.aspect_max),
        video_artifacts_prob=float(args.video_artifacts_prob),
        rolling_shutter_prob=float(args.rolling_shutter_prob),
        rolling_shutter_max_shift_px=float(args.rolling_shutter_max_shift_px),
        rolling_shutter_wobble_px=float(args.rolling_shutter_wobble_px),
        motion_blur_prob=float(args.dir_motion_blur_prob),
        motion_blur_len_min=int(args.dir_motion_blur_len_min),
        motion_blur_len_max=int(args.dir_motion_blur_len_max),
        motion_blur_angle_bias_deg=float(args.dir_motion_blur_angle_bias_deg),
        motion_blur_angle_jitter_deg=float(args.dir_motion_blur_angle_jitter_deg),
    )
    _seed_everything(args.seed + 2)
    generate_split(
        out_dir=out_dir,
        split="val",
        n=int(args.val),
        size_min=int(args.size_min),
        size_max=int(args.size_max),
        sym_probs=sym_probs,
        seed=args.seed + 2,
        max_margin_ratio=float(args.max_margin_ratio),
        under_text_prob=float(args.under_text_prob),
        landscape_prob=float(args.landscape_prob),
        aspect_min=float(args.aspect_min),
        aspect_max=float(args.aspect_max),
        video_artifacts_prob=float(args.video_artifacts_prob),
        rolling_shutter_prob=float(args.rolling_shutter_prob),
        rolling_shutter_max_shift_px=float(args.rolling_shutter_max_shift_px),
        rolling_shutter_wobble_px=float(args.rolling_shutter_wobble_px),
        motion_blur_prob=float(args.dir_motion_blur_prob),
        motion_blur_len_min=int(args.dir_motion_blur_len_min),
        motion_blur_len_max=int(args.dir_motion_blur_len_max),
        motion_blur_angle_bias_deg=float(args.dir_motion_blur_angle_bias_deg),
        motion_blur_angle_jitter_deg=float(args.dir_motion_blur_angle_jitter_deg),
    )
    _seed_everything(args.seed + 3)
    generate_split(
        out_dir=out_dir,
        split="test",
        n=int(args.test),
        size_min=int(args.size_min),
        size_max=int(args.size_max),
        sym_probs=sym_probs,
        seed=args.seed + 3,
        max_margin_ratio=float(args.max_margin_ratio),
        under_text_prob=float(args.under_text_prob),
        landscape_prob=float(args.landscape_prob),
        aspect_min=float(args.aspect_min),
        aspect_max=float(args.aspect_max),
        video_artifacts_prob=float(args.video_artifacts_prob),
        rolling_shutter_prob=float(args.rolling_shutter_prob),
        rolling_shutter_max_shift_px=float(args.rolling_shutter_max_shift_px),
        rolling_shutter_wobble_px=float(args.rolling_shutter_wobble_px),
        motion_blur_prob=float(args.dir_motion_blur_prob),
        motion_blur_len_min=int(args.dir_motion_blur_len_min),
        motion_blur_len_max=int(args.dir_motion_blur_len_max),
        motion_blur_angle_bias_deg=float(args.dir_motion_blur_angle_bias_deg),
        motion_blur_angle_jitter_deg=float(args.dir_motion_blur_angle_jitter_deg),
    )

    if bool(args.zip):
        zip_path = out_dir.with_name(out_dir.name + ".zip")
        if zip_path.exists():
            zip_path.unlink()
        shutil.make_archive(
            base_name=str(zip_path.with_suffix("")),
            format="zip",
            root_dir=str(out_dir.parent),
            base_dir=str(out_dir.name),
        )


if __name__ == "__main__":
    main()