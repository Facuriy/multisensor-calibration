#!/usr/bin/env python3
"""Shared checkerboard detection utilities for multisensor calibration.

This module provides preprocessing and quality-scoring helpers that complement
the per-script detection code across calibration pipelines.  All functions
operate on uint8 grayscale images and are sensor-agnostic.

Key additions over bare OpenCV calls:

  local_normalize_checker   — spatial local-mean/std normalization; nearly
                              invariant to smooth illumination gradients, which
                              enhances the sharp black/white corner transitions
                              even under uneven lighting.

  tophat_enhance            — morphological white-hat / black-hat combination;
                              removes slow shading and boosts fine checker-square
                              structure.

  gradient_variance_roi     — finds image blocks with high gradient-energy
                              variance, which correlates with periodic patterns
                              like a checkerboard.  Returns a bounding-box hint
                              so the detector can focus there first.

  grid_homography_residual  — fits a homography from grid indices to detected
                              pixel positions and reports reprojection residuals.
                              Low residuals confirm geometric consistency of a
                              detection; high residuals flag false or poorly
                              localized corners before they reach calibration.

  extra_detector_variants   — convenience wrapper that returns the new
                              preprocessing variants as (name, uint8) pairs,
                              ready to be appended to any existing variant list.
"""

from __future__ import annotations

import cv2
import numpy as np


def local_normalize_checker(u8: np.ndarray, block_size: int = 0) -> np.ndarray:
    """Local mean-subtraction + std normalization tuned for checkerboard patterns.

    Unlike CLAHE (which redistributes the global histogram in tiles),
    this subtracts the local mean and divides by the local std within a
    spatial window.  The result is nearly invariant to slow illumination
    gradients while preserving the sharp black/white transitions at corner
    junctions that the OpenCV detectors rely on.

    Particularly effective for:
    - Photonfocus multispectral preview images (uneven spectral weighting)
    - Checkerboards partially in shadow or under directional lighting
    - Low-contrast boards (dark on dark, bright on bright)

    Args:
        u8:         Grayscale uint8 image.
        block_size: Square window side in pixels (should be odd).
                    0 → auto (min(h, w) // 8, clipped to ≥7, forced odd).

    Returns:
        Normalized uint8 image in the same spatial shape as the input.
        The mapping [-3σ, +3σ] → [0, 255] is applied after normalization.
    """
    if block_size <= 0:
        block_size = max(7, (min(u8.shape[:2]) // 8) | 1)
    ksize = (block_size, block_size)
    src = u8.astype(np.float32)
    local_mean = cv2.boxFilter(src, cv2.CV_32F, ksize)
    local_sq = cv2.boxFilter(src * src, cv2.CV_32F, ksize)
    # max(..., 0) prevents negative variance from floating-point rounding
    local_std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 0.0) + 1e-4)
    normed = (src - local_mean) / local_std  # typically in [-5, +5]
    # Clip [-3, +3] → [0, 255]
    return np.clip((normed + 3.0) * (255.0 / 6.0), 0.0, 255.0).astype(np.uint8)


def tophat_enhance(u8: np.ndarray, kernel_size: int = 0) -> np.ndarray:
    """Morphological top-hat / bottom-hat enhancement for checkerboard contrast.

    White top-hat (WTH) isolates bright regions smaller than the structuring
    element; black top-hat (BTH) isolates dark ones.  Adding WTH and subtracting
    BTH from the original removes slow shading and amplifies fine periodic
    structure such as the squares of a checkerboard.

    Particularly effective when:
    - The board surface is slightly reflective or shows specular highlights
    - There is a slow illumination gradient across the board
    - The board is slightly out of focus (the operation sharpens edges)

    Args:
        u8:          Grayscale uint8 image.
        kernel_size: Structuring element side in pixels (odd, ≥5).
                     0 → auto (min(h, w) // 14, clipped to ≥5, forced odd).

    Returns:
        Enhanced uint8 image.
    """
    if kernel_size <= 0:
        kernel_size = max(5, (min(u8.shape[:2]) // 14) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    wth = cv2.morphologyEx(u8, cv2.MORPH_TOPHAT, kernel)
    bth = cv2.morphologyEx(u8, cv2.MORPH_BLACKHAT, kernel)
    # add wth (brightens bright edges) and subtract bth (darkens dark edges)
    return cv2.add(cv2.subtract(u8, bth), wth)


def gradient_variance_roi(
    img: np.ndarray,
    grid_rows: int = 4,
    grid_cols: int = 4,
    top_n: int = 4,
    margin_frac: float = 0.10,
) -> list[float] | None:
    """Estimate the likely checkerboard region using gradient-variance analysis.

    Divides the image into a (grid_rows × grid_cols) grid and scores each
    block by the variance of its squared-gradient energy.  A regular
    checkerboard pattern produces strong, spatially periodic gradient energy
    whose variance within a block stands out from uniform background areas.

    This estimate is used as a spatial prior: try the returned crop first
    before falling back to the full image.  It is especially useful for large
    RGB images (2448 × 2048) where the board occupies a small fraction and the
    full-image scan is expensive.

    For Photonfocus sensors the board may be partially outside the frame; this
    function will still identify the visible corner as the high-gradient region.

    Args:
        img:         Grayscale uint8 image.
        grid_rows:   Number of vertical grid divisions.
        grid_cols:   Number of horizontal grid divisions.
        top_n:       Number of top-scoring blocks to include in the ROI.
        margin_frac: Fractional padding added around the selected blocks.

    Returns:
        [x0, y0, x1, y1] bounding box in image pixel coordinates, or None
        when the estimated ROI covers more than 80 % of the image (uninformative)
        or the image is too small for reliable block analysis.
    """
    h, w = img.shape[:2]
    bh = h // grid_rows
    bw = w // grid_cols
    if bh < 8 or bw < 8:
        return None

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    energy = gx * gx + gy * gy

    scores: list[tuple[float, int, int]] = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            block = energy[r * bh : (r + 1) * bh, c * bw : (c + 1) * bw]
            scores.append((float(np.var(block)), r, c))

    scores.sort(reverse=True)
    best = scores[:top_n]
    min_r = min(r for _, r, c in best)
    max_r = max(r for _, r, c in best)
    min_c = min(c for _, r, c in best)
    max_c = max(c for _, r, c in best)

    pad_x = int(w * margin_frac)
    pad_y = int(h * margin_frac)
    x0 = float(max(0, min_c * bw - pad_x))
    y0 = float(max(0, min_r * bh - pad_y))
    x1 = float(min(w, (max_c + 1) * bw + pad_x))
    y1 = float(min(h, (max_r + 1) * bh + pad_y))

    # Don't return a hint that covers almost the whole image — it's not useful
    if (x1 - x0) * (y1 - y0) > 0.80 * float(w * h):
        return None
    return [x0, y0, x1, y1]


def grid_homography_residual(
    corners: np.ndarray,
    pattern: tuple[int, int],
) -> dict[str, float]:
    """Measure geometric consistency of detected checkerboard corners.

    Fits a homography H mapping grid-index coordinates to image pixels
    and reports the reprojection residuals.  A well-detected board has
    mean_px < 0.5 px and max_px < 2 px; values above ~4 px typically
    indicate a false or badly localized detection.

    This metric is more sensitive than the coarse checker_score because
    it captures local distortions and mis-orderings that do not strongly
    affect the bounding-box-based score.

    Args:
        corners: Corner positions, shape (N, 2), row-major order as
                 returned by cv2.findChessboardCorners /
                 findChessboardCornersSB (left→right within each row,
                 then top→bottom rows).
        pattern: (cols, rows) number of internal corners.

    Returns:
        Dict with keys:
          mean_px  — mean reprojection error across all corners
          max_px   — worst-corner reprojection error
          rms_px   — RMS reprojection error
          ok       — True when max_px < 2.0 px (reliable detection)
    """
    cols, rows = pattern
    n = cols * rows
    if corners.shape[0] != n:
        return {"mean_px": float("inf"), "max_px": float("inf"), "rms_px": float("inf"), "ok": False}

    # Normalized object grid: column index, row index (no physical scale needed)
    obj = np.array(
        [[float(c), float(r)] for r in range(rows) for c in range(cols)],
        dtype=np.float64,
    )
    img_pts = corners.reshape(-1, 2).astype(np.float64)

    H, _ = cv2.findHomography(
        obj.reshape(-1, 1, 2),
        img_pts.reshape(-1, 1, 2),
        0,  # DLT — no RANSAC, we trust the full set
    )
    if H is None:
        return {"mean_px": float("inf"), "max_px": float("inf"), "rms_px": float("inf"), "ok": False}

    proj = cv2.perspectiveTransform(obj.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(proj - img_pts, axis=1)
    mean_err = float(np.mean(err))
    max_err = float(np.max(err))
    rms_err = float(np.sqrt(np.mean(err**2)))
    return {
        "mean_px": mean_err,
        "max_px": max_err,
        "rms_px": rms_err,
        "ok": max_err < 2.0,
    }


def extra_detector_variants(u8: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Return additional preprocessed variants for the checkerboard detector.

    Produces four new uint8 images that complement the standard CLAHE /
    unsharp-mask preprocessing already present in the calibration scripts:

      local_norm          — adaptive local normalization (invariant to gradients)
      tophat              — morphological enhancement (removes slow shading)
      local_norm_tophat   — both combined (strong enhancement, good for dark scenes)
      local_norm_inv      — inverted local norm (dark-on-light board variants)

    These are particularly useful for:
    - Photonfocus VIS / NIR sensors with uneven spectral band weighting
    - Partially shadowed boards (common when the robot is close and the board
      is near the ground)
    - Boards with lower contrast than expected (worn paint, partial occlusion)

    Args:
        u8: Grayscale uint8 image — should be the preview/mean image, already
            converted to uint8 (e.g. after robust_u8 or normalize_u8).

    Returns:
        List of (name, processed_uint8) pairs ready to append to any variant
        list.  Names do not contain spaces so they are safe to use as CSV/JSON
        keys.
    """
    lnorm = local_normalize_checker(u8)
    tophat = tophat_enhance(u8)
    lnorm_tophat = tophat_enhance(lnorm)
    lnorm_inv = cv2.bitwise_not(lnorm)
    return [
        ("local_norm", lnorm),
        ("tophat", tophat),
        ("local_norm_tophat", lnorm_tophat),
        ("local_norm_inv", lnorm_inv),
    ]
