"""PIL-space corruption functions for the transform-robustness battery.

Trimmed from the main research codebase's `lorc/robustness_battery.py` down
to just the pieces `evaluate_wildfake.py` here needs: the 14 non-clean
condition functions, and the "condition first, then a final q=96 JPEG pass"
stacking order (matches training's own mandatory-JPEG convention, so
evaluating any other way would test the model on an artifact distribution
it never saw).
"""
import io

import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms


def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def tf_jpeg(q):
    return lambda img: jpeg_compress(img, q)


def tf_blur(sigma):
    return lambda img: img.filter(ImageFilter.GaussianBlur(radius=sigma))


def tf_resize(scale):
    def _fn(img):
        w, h = img.size
        small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
        return small.resize((w, h), Image.BICUBIC)
    return _fn


def tf_noise(sigma, seed=None):
    def _fn(img):
        rng = np.random.RandomState(seed)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.clip(arr + rng.normal(0, sigma, arr.shape).astype(np.float32), 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))
    return _fn


def tf_jitter(strength, seed=None):
    import torch
    jitter = transforms.ColorJitter(brightness=strength, contrast=strength, saturation=strength)
    def _fn(img):
        if seed is not None:
            torch.manual_seed(seed)
        return jitter(img)
    return _fn


def tf_crop(frac):
    def _fn(img):
        w, h = img.size
        nw, nh = int(w * frac), int(h * frac)
        left, top = (w - nw) // 2, (h - nh) // 2
        return img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.BICUBIC)
    return _fn


def stack_baseline_last(fn):
    """The condition's own corruption first, then a final q=96 JPEG pass —
    standardizes the final encoding every image ends up in, matching
    training's own transform-then-JPEG convention."""
    return lambda img: jpeg_compress(fn(img), 96)


# The 14 non-clean conditions: (name, raw corruption fn, family)
RAW_CONDITIONS = [
    ("JPEG q=90",        tf_jpeg(90),              "JPEG"),
    ("JPEG q=70",        tf_jpeg(70),              "JPEG"),
    ("JPEG q=50",        tf_jpeg(50),              "JPEG"),
    ("JPEG q=30",        tf_jpeg(30),              "JPEG"),
    ("Blur sigma=0.5",   tf_blur(0.5),             "Blur"),
    ("Blur sigma=1.0",   tf_blur(1.0),             "Blur"),
    ("Blur sigma=2.0",   tf_blur(2.0),             "Blur"),
    ("Resize 0.5x",      tf_resize(0.5),           "Resize"),
    ("Resize 0.25x",     tf_resize(0.25),          "Resize"),
    ("Noise sigma=0.02", tf_noise(0.02, seed=42),  "Noise"),
    ("Noise sigma=0.05", tf_noise(0.05, seed=42),  "Noise"),
    ("Noise sigma=0.10", tf_noise(0.10, seed=42),  "Noise"),
    ("ColorJitter +-20%", tf_jitter(0.2, seed=42), "ColorJitter"),
    ("CenterCrop 80%",   tf_crop(0.8),             "CenterCrop"),
]

RAW_BY_NAME = {name: fn for name, fn, _family in RAW_CONDITIONS}
FAMILY_BY_NAME = {name: family for name, _fn, family in RAW_CONDITIONS}
