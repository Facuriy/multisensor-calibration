"""Checkerboard detection on raw thermal mono16/count images.

Unlike ``thermal_checker.py``, this module does not invert a colormap. It uses
the scalar thermal image directly and performs flat-fielding/re-stretching in
float precision before quantizing to 8 bit for OpenCV's checkerboard detectors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from thermal_checker import detect_classic, detect_sb


def load_thermal_raw(path: str | Path) -> np.ndarray:
    """Load a 2D raw thermal array from ``.npy`` or 16-bit TIFF/PNG."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext in (".tif", ".tiff", ".png"):
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        if arr is None:
            raise OSError(f"could not read {path}")
        if arr.ndim == 3:
            arr = arr[..., 0]
    else:
        raise ValueError(
            "Unsupported thermal raw container. Export R-JPEG/vendor formats to "
            ".npy, .tif, .tiff or 16-bit .png first."
        )
    if arr.ndim != 2:
        raise ValueError(f"expected 2D thermal raw image, got shape {arr.shape}")
    return arr.astype(np.float32)


def remove_bad_pixels(img: np.ndarray, k: int = 3, thresh: float = 6.0) -> np.ndarray:
    """Replace outlier pixels by a local median using a MAD threshold."""
    src = img.astype(np.float32)
    med = cv2.medianBlur(src, k)
    diff = np.abs(src - med)
    mad = float(np.median(diff)) + 1e-6
    bad = diff > thresh * mad
    out = src.copy()
    out[bad] = med[bad]
    return out


def destripe(img: np.ndarray) -> np.ndarray:
    """Remove simple row/column fixed-pattern offsets."""
    src = img.astype(np.float32)
    g = src - np.median(src, axis=0, keepdims=True) + np.median(src)
    g = g - np.median(g, axis=1, keepdims=True) + np.median(g)
    return g.astype(np.float32)


def flat_field_raw(img: np.ndarray, sigma_frac: float = 0.12) -> np.ndarray:
    """Subtract low-frequency thermal background in native counts."""
    src = img.astype(np.float32)
    h, w = src.shape
    sigma = max(3, int(sigma_frac * max(h, w)))
    bg = cv2.GaussianBlur(src, (0, 0), sigma)
    return src - bg


def robust01(img: np.ndarray, lo: float = 1, hi: float = 99) -> np.ndarray:
    src = img.astype(np.float32)
    finite = np.isfinite(src)
    if not finite.any():
        return np.zeros(src.shape, dtype=np.float32)
    a, b = np.percentile(src[finite], (lo, hi))
    if b <= a:
        b = a + 1.0
    return np.clip((src - a) / (b - a), 0, 1).astype(np.float32)


def preprocess_raw_variants(
    img: np.ndarray,
    do_badpix: bool = True,
    do_destripe: bool = False,
) -> list[tuple[str, np.ndarray]]:
    """Create 8-bit detector inputs from raw thermal counts."""
    work = img.astype(np.float32)
    if do_badpix:
        work = remove_bad_pixels(work)
    if do_destripe:
        work = destripe(work)

    resid = flat_field_raw(work)
    base = robust01(resid, 1, 99)
    base16 = (base * 65535.0).astype(np.uint16)
    base8 = (base * 255.0).astype(np.uint8)

    def clahe16(clip: float, grid: int) -> np.ndarray:
        out = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(base16)
        return (out / 257.0).astype(np.uint8)

    bil = cv2.bilateralFilter(base8, 7, 40, 7)
    return [
        ("raw_ff_clahe16", clahe16(2.0, 8)),
        ("raw_ff_clahe16_fine", clahe16(3.0, 4)),
        ("raw_ff_bilateral_clahe8", cv2.createCLAHE(2.0, (8, 8)).apply(bil)),
        ("raw_ff_clahe8", cv2.createCLAHE(2.0, (8, 8)).apply(base8)),
        ("raw_ff", base8),
    ]


def detect_board_raw(
    img: np.ndarray,
    pattern_size: tuple[int, int],
    do_badpix: bool = True,
    do_destripe: bool = False,
) -> tuple[bool, np.ndarray | None, dict[str, Any]]:
    """Detect checkerboard corners in raw thermal image coordinates."""
    variants = preprocess_raw_variants(img, do_badpix=do_badpix, do_destripe=do_destripe)
    for name, u8 in variants:
        corners, sweep = detect_sb(u8, pattern_size)
        if corners is not None:
            return True, corners, {"method": "SB", "variant": name, "sweep": sweep}
    corners, method = detect_classic(variants[0][1], pattern_size)
    if corners is not None:
        return True, corners, {"method": method, "variant": variants[0][0]}
    return False, None, {"method": "none"}

