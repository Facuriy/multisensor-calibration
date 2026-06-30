"""Thermal checkerboard helpers for color-mapped thermal previews.

The thermal preview images in this project are saved with a visible colormap
applied. Converting those images directly to grayscale loses the scalar thermal
ordering. This module first inverts the colormap approximately and then runs
checkerboard detection on several contrast-normalized variants.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _cv2_lut(cid: int) -> np.ndarray:
    ramp = np.arange(256, dtype=np.uint8).reshape(-1, 1)
    return cv2.applyColorMap(ramp, cid)[:, 0, ::-1].astype(np.float32)


CMAPS = {
    "inferno": _cv2_lut(cv2.COLORMAP_INFERNO),
    "magma": _cv2_lut(cv2.COLORMAP_MAGMA),
    "plasma": _cv2_lut(cv2.COLORMAP_PLASMA),
    "viridis": _cv2_lut(cv2.COLORMAP_VIRIDIS),
    "turbo": _cv2_lut(cv2.COLORMAP_TURBO),
    "jet": _cv2_lut(cv2.COLORMAP_JET),
    "hot": _cv2_lut(cv2.COLORMAP_HOT),
}


def detect_colormap(rgb: np.ndarray, step: int = 6) -> tuple[str, float]:
    """Return the OpenCV colormap name that best explains an RGB image."""
    sub = rgb[::step, ::step].reshape(-1, 3).astype(np.float32)
    best_name = "inferno"
    best_err = float("inf")
    for name, lut in CMAPS.items():
        lut2 = (lut**2).sum(1)
        d = (sub**2).sum(1, keepdims=True) - 2 * sub @ lut.T + lut2[None, :]
        err = np.sqrt(np.clip(d.min(1), 0, None)).mean()
        if err < best_err:
            best_name = name
            best_err = float(err)
    return best_name, best_err


def recover_scalar(rgb: np.ndarray, cmap: str = "auto", n: int = 512) -> tuple[np.ndarray, dict[str, Any]]:
    """Approximate the original thermal scalar in [0, 1] from RGB colormap data."""
    if cmap == "auto":
        cmap, residual = detect_colormap(rgb)
    else:
        residual = float("nan")
    if cmap not in CMAPS:
        raise ValueError(f"unsupported colormap: {cmap}")

    base = CMAPS[cmap]
    xs = np.linspace(0, 255, n)
    lut = np.stack([np.interp(xs, np.arange(256), base[:, c]) for c in range(3)], 1).astype(np.float32)

    h, w, _ = rgb.shape
    pix = rgb.reshape(-1, 3).astype(np.float32)
    out = np.empty(pix.shape[0], np.float32)
    lut2 = (lut**2).sum(1)
    chunk = 300_000
    for start in range(0, pix.shape[0], chunk):
        p = pix[start : start + chunk]
        d = (p**2).sum(1, keepdims=True) - 2 * p @ lut.T + lut2[None, :]
        out[start : start + chunk] = np.argmin(d, 1) / float(n - 1)
    return out.reshape(h, w), {"cmap": cmap, "residual": residual}


def to_u8(src: np.ndarray, lo_pct: float = 1, hi_pct: float = 99) -> np.ndarray:
    arr = src.astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(arr[finite], (lo_pct, hi_pct))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def flat_field(scalar: np.ndarray, sigma_frac: float = 0.12) -> np.ndarray:
    h, w = scalar.shape
    sigma = max(3, int(sigma_frac * max(h, w)))
    bg = cv2.GaussianBlur(scalar.astype(np.float32), (0, 0), sigma)
    out = scalar.astype(np.float32) - bg
    return (out - out.min()) / (out.max() - out.min() + 1e-6)


def clahe(u8: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(u8)


def unsharp(u8: np.ndarray, sigma: float = 2.0, amount: float = 1.5) -> np.ndarray:
    blur = cv2.GaussianBlur(u8, (0, 0), sigma)
    return cv2.addWeighted(u8, 1 + amount, blur, -amount, 0)


def preprocess_variants(scalar: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Return normalized thermal scalar variants ordered by expected usefulness."""
    ff = flat_field(scalar)
    raw8 = to_u8(scalar)
    ff8 = to_u8(ff)
    bil = cv2.bilateralFilter(ff8, 7, 40, 7)
    variants = [
        ("scalar_flatfield_clahe", clahe(ff8, 2.0, 8)),
        ("scalar_flatfield_clahe_fine", clahe(ff8, 3.0, 4)),
        ("scalar_flatfield_bilateral_clahe", clahe(bil, 2.0, 8)),
        ("scalar_flatfield_unsharp", unsharp(ff8)),
        ("scalar_raw_clahe", clahe(raw8, 2.0, 8)),
        ("scalar_flatfield", ff8),
        ("scalar_raw_stretch", raw8),
    ]
    return variants


SB_FLAGS = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY


def _rotate(img: np.ndarray, deg: float) -> tuple[np.ndarray, np.ndarray]:
    if deg == 0:
        return img, np.eye(2, 3, dtype=np.float32)
    h, w = img.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    rot = cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rot, mat.astype(np.float32)


def _invert_affine(mat: np.ndarray) -> np.ndarray:
    aff = np.vstack([mat, [0, 0, 1]])
    return np.linalg.inv(aff)[:2].astype(np.float32)


def detect_sb(
    u8: np.ndarray,
    pattern_size: tuple[int, int],
    scales: tuple[float, ...] = (1.0, 2.0, 1.5, 0.5),
    rotations: tuple[float, ...] = (0, 8, -8, 15, -15),
    try_invert: bool = True,
) -> tuple[np.ndarray | None, str | None]:
    """Find checker corners with scale/rotation/polarity sweep."""
    for scale in scales:
        img_s = cv2.resize(
            u8,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC if scale >= 1 else cv2.INTER_AREA,
        )
        for deg in rotations:
            img_r, mat = _rotate(img_s, deg)
            polarities = [(False, img_r)]
            if try_invert:
                polarities.append((True, cv2.bitwise_not(img_r)))
            for inverted, pol in polarities:
                ok, corners = cv2.findChessboardCornersSB(pol, pattern_size, flags=SB_FLAGS)
                if not ok or corners is None:
                    continue
                inv = _invert_affine(mat)
                pts = corners.reshape(-1, 2)
                pts = (inv[:, :2] @ pts.T + inv[:, 2:3]).T
                pts /= scale
                tag = f"s{scale:g}_r{deg:g}_{'inv' if inverted else 'pos'}"
                return pts.reshape(-1, 1, 2).astype(np.float32), tag
    return None, None


def detect_classic(u8: np.ndarray, pattern_size: tuple[int, int]) -> tuple[np.ndarray | None, str | None]:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    for inverted, pol in ((False, u8), (True, cv2.bitwise_not(u8))):
        ok, corners = cv2.findChessboardCorners(pol, pattern_size, flags=flags)
        if not ok or corners is None:
            continue
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
        cv2.cornerSubPix(pol, corners, (11, 11), (-1, -1), term)
        return corners, f"classic_{'inv' if inverted else 'pos'}"
    return None, None


def detect_board(
    rgb: np.ndarray,
    pattern_size: tuple[int, int],
    cmap: str = "auto",
) -> tuple[bool, np.ndarray | None, dict[str, Any]]:
    """Detect a thermal checkerboard in a color-mapped RGB image."""
    scalar, scalar_info = recover_scalar(rgb, cmap=cmap)
    for name, u8 in preprocess_variants(scalar):
        corners, sweep = detect_sb(u8, pattern_size)
        if corners is not None:
            return True, corners, {"method": "SB", "variant": name, "sweep": sweep, **scalar_info}
    best_u8 = preprocess_variants(scalar)[0][1]
    corners, method = detect_classic(best_u8, pattern_size)
    if corners is not None:
        return True, corners, {"method": method, "variant": "scalar_flatfield_clahe", **scalar_info}
    return False, None, {"method": "none", **scalar_info}

