#!/usr/bin/env python3
"""Calibrate camera intrinsics from multisensor checkerboard ROS1 bags.

This is the production entry point for per-camera K/distortion calibration.
It scans calibration bags, keeps one best checkerboard detection per bag/sensor
by default, calibrates intrinsics, removes obvious outlier views, and writes
JSON/CSV/preview QA artifacts.

The 2026-06-23 board has 10 x 7 total squares, therefore 9 x 6 internal
corners. Only internal corners are used because the outer border is irregular.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep the calibration directory itself importable for thermal_checker_raw.py,
# which intentionally imports thermal_checker as a sibling module.
CALIB_DIR = Path(__file__).resolve().parent
if str(CALIB_DIR) not in sys.path:
    sys.path.insert(0, str(CALIB_DIR))

from src.extraction.extract_all_bag_images import decode_ros_image, photonfocus_preview, robust_preview  # noqa: E402
from src.calibration.thermal_checker import detect_board as detect_thermal_colormap  # noqa: E402
from src.calibration.thermal_checker_raw import detect_board_raw, preprocess_raw_variants, robust01  # noqa: E402


DEFAULT_MANIFEST = Path("data/calibration/new_session/20260623/bag_manifest_20260623.csv")
DEFAULT_CHECKER = Path("data/calibration/new_session/20260623/checkerboard_config.json")
DEFAULT_INITIAL_INTRINSICS = Path("data/matrices/initial_camera_intrinsics_from_report.json")
DEFAULT_OUT = Path("runs/calibration_intrinsics_checkerboard_bags")
DEFAULT_RESCUE_GUIDE_SUMMARY = Path(
    "runs/calibration_20260623_refined_multisensor_detection/refined_multisensor_detection_summary.json"
)

TOPICS = {
    "rgb": "/ssf/BFS_usb_0/image_raw",
    "vis": "/ssf/photonfocus_camera_vis_node/image_raw",
    "nir": "/ssf/photonfocus_camera_nir_node/image_raw",
    "thermal_c": "/ssf/thermalgrabber_ros/image_deg_celsius",
    "thermal_raw": "/ssf/thermalgrabber_ros/image_mono16",
}

COUNT_COLUMNS = {
    "rgb": "rgb_msgs",
    "vis": "vis_msgs",
    "nir": "nir_msgs",
    "thermal_c": "thermal_c_msgs",
    "thermal_raw": "thermal_raw_msgs",
}

INITIAL_INTRINSIC_KEYS = {
    "rgb": "rgb_blackfly_bfs_u3_50s5c_c",
    "vis": "vis_photonfocus_hs03_single_band_preview",
    "nir": "nir_photonfocus_hs02_single_band_preview",
    "thermal_c": "thermal_tau2_640_19mm",
    "thermal_raw": "thermal_tau2_640_19mm",
}


@dataclass
class Detection:
    sensor: str
    bag: str
    bag_path: str
    label: str
    frame_index: int
    stamp_ns: int
    image_size_wh: tuple[int, int]
    corners: np.ndarray
    method: str
    variant: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    preview_path: str
    encoding: str = ""
    confidence: str = "subpixel_full"
    rescue: bool = False
    usable_for_intrinsics: bool = True
    usable_for_registration: bool = True


def load_checker_config(path: Path) -> tuple[tuple[int, int], float]:
    if not path.exists():
        return (9, 6), 0.04
    data = json.loads(path.read_text(encoding="utf-8"))
    pattern = tuple(int(v) for v in data.get("internal_corners", [9, 6]))
    square_m = float(data.get("square_size_m", 0.04))
    return (pattern[0], pattern[1]), square_m


def object_points(pattern: tuple[int, int], square_size_m: float) -> np.ndarray:
    cols, rows = pattern
    obj = np.zeros((cols * rows, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj[:, :2] = grid * float(square_size_m)
    return obj


def stamp_ns(msg: Any, bag_ts: int) -> int:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return int(bag_ts)
    sec = int(getattr(stamp, "sec", 0))
    nsec = int(getattr(stamp, "nanosec", getattr(stamp, "nsec", 0)))
    out = sec * 1_000_000_000 + nsec
    return int(out if out > 0 else bag_ts)


def robust_u8(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    src = img.astype(np.float32)
    finite = np.isfinite(src)
    if not finite.any():
        return np.zeros(src.shape[:2], dtype=np.uint8)
    lo, hi = np.percentile(src[finite], [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((src - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def clahe(u8: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(u8)


def unsharp(u8: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(u8, (0, 0), 2.0)
    return cv2.addWeighted(u8, 2.1, blur, -1.1, 0)


def visible_variants(img: np.ndarray, sensor: str, include_bands: bool = False) -> list[tuple[str, np.ndarray]]:
    """Return detector inputs in the coordinate system to be calibrated."""
    variants: list[tuple[str, np.ndarray]] = []

    if sensor in ("vis", "nir") and img.ndim == 2:
        # Photonfocus frames are multispectral mosaics. The calibrated image
        # plane is the demosaiced/downsampled preview plane used downstream.
        pf = photonfocus_preview(img)
        base = robust_u8(pf)
        variants.extend(
            [
                ("pf_mean_stretch", base),
                ("pf_mean_clahe", clahe(base, 2.0, 8)),
                ("pf_mean_clahe_fine", clahe(base, 3.0, 4)),
                ("pf_mean_unsharp", unsharp(base)),
            ]
        )

        if include_bands:
            # Individual mosaic bands rescue cases where one band has much
            # stronger checker contrast than the average. They are slower, so
            # keep them behind --deep.
            pattern = 4
            usable_h = (img.shape[0] // pattern) * pattern
            usable_w = (img.shape[1] // pattern) * pattern
            raw = img[:usable_h, :usable_w]
            for r in range(pattern):
                for c in range(pattern):
                    band = cv2.medianBlur(raw[r::pattern, c::pattern].astype(np.float32), 3)
                    b8 = robust_u8(band)
                    variants.append((f"pf_band_{r}_{c}_clahe", clahe(b8, 2.0, 8)))
        return variants

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    base = robust_u8(gray)
    variants.extend(
        [
            ("gray_stretch", base),
            ("gray_unsharp", unsharp(base)),
            ("gray_equalized", cv2.equalizeHist(base)),
            ("gray_clahe", clahe(base, 2.0, 8)),
            ("gray_clahe_fine", clahe(base, 3.0, 4)),
            ("gray_blur_clahe", clahe(cv2.GaussianBlur(base, (5, 5), 0), 2.0, 8)),
        ]
    )
    return variants


def search_regions(shape: tuple[int, int], crops: bool) -> list[tuple[int, int, int, int, str]]:
    h, w = shape
    regions = [(0, 0, w, h, "full")]
    if not crops:
        return regions
    regions.extend(
        [
            (0, 0, w // 2, h, "left"),
            (w // 2, 0, w, h, "right"),
            (0, 0, w, h // 2, "top"),
            (0, h // 2, w, h, "bottom"),
            (w // 6, h // 6, 5 * w // 6, 5 * h // 6, "center"),
            (0, 0, int(w * 0.68), int(h * 0.68), "top_left"),
            (int(w * 0.32), 0, w, int(h * 0.68), "top_right"),
            (0, int(h * 0.32), int(w * 0.68), h, "bottom_left"),
            (int(w * 0.32), int(h * 0.32), w, h, "bottom_right"),
        ]
    )
    return [(x0, y0, x1, y1, name) for x0, y0, x1, y1, name in regions if x1 > x0 and y1 > y0]


def local_detail_u8(img: np.ndarray, sigma: float = 9.0) -> np.ndarray:
    src = img.astype(np.float32)
    bg = cv2.GaussianBlur(src, (0, 0), sigma)
    return robust_u8(src - bg, 1.0, 99.0)


def load_rescue_guides(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("results", [])
    return {
        str(row.get("label_norm") or row.get("label_raw") or row.get("label") or ""): row
        for row in rows
        if row.get("label_norm") or row.get("label_raw") or row.get("label")
    }


def rescue_guide_box(
    label: str,
    sensor: str,
    guides: dict[str, dict[str, Any]],
    shape_hw: tuple[int, int],
) -> list[float] | None:
    row = guides.get(label, {})
    box = row.get("projected_boxes_from_rgb", {}).get(sensor)
    if box is None:
        box = row.get("detections", {}).get(sensor, {}).get("panel_box_xyxy")
    if box is None:
        return None
    h, w = shape_hw
    x0, y0, x1, y1 = [float(v) for v in box]
    x0 = max(0.0, min(float(w - 1), x0))
    x1 = max(0.0, min(float(w - 1), x1))
    y0 = max(0.0, min(float(h - 1), y0))
    y1 = max(0.0, min(float(h - 1), y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def rescale_box(box: list[float] | None, from_shape_hw: tuple[int, int], to_shape_hw: tuple[int, int]) -> list[float] | None:
    if box is None:
        return None
    fh, fw = from_shape_hw
    th, tw = to_shape_hw
    if fw <= 0 or fh <= 0:
        return None
    sx = float(tw) / float(fw)
    sy = float(th) / float(fh)
    return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]


def expanded_box(box: list[float] | None, shape_hw: tuple[int, int], frac: float = 0.65) -> tuple[int, int, int, int] | None:
    if box is None:
        return None
    h, w = shape_hw
    x0, y0, x1, y1 = [float(v) for v in box]
    if not all(np.isfinite([x0, y0, x1, y1])) or x1 <= x0 or y1 <= y0:
        return None
    px = max(14, int((x1 - x0) * frac))
    py = max(14, int((y1 - y0) * frac))
    rx0 = max(0, int(math.floor(x0)) - px)
    ry0 = max(0, int(math.floor(y0)) - py)
    rx1 = min(w, int(math.ceil(x1)) + px)
    ry1 = min(h, int(math.ceil(y1)) + py)
    if rx1 - rx0 < 35 or ry1 - ry0 < 25:
        return None
    return rx0, ry0, rx1, ry1


def rescue_regions(shape_hw: tuple[int, int], guide: list[float] | None, include_broad: bool) -> list[tuple[int, int, int, int, str]]:
    h, w = shape_hw
    out: list[tuple[int, int, int, int, str]] = []
    for frac, name in ((0.25, "guided_tight"), (0.70, "guided"), (1.10, "guided_wide")):
        roi = expanded_box(guide, shape_hw, frac)
        if roi is not None:
            out.append((*roi, name))
    if include_broad or not out:
        out.extend(search_regions(shape_hw, crops=True))
    uniq: list[tuple[int, int, int, int, str]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for x0, y0, x1, y1, name in out:
        key = (x0, y0, x1, y1)
        if key not in seen and x1 > x0 and y1 > y0:
            uniq.append((x0, y0, x1, y1, name))
            seen.add(key)
    return uniq


def photonfocus_rescue_bases(raw: np.ndarray, sensor: str) -> list[tuple[str, np.ndarray, bool]]:
    """Return Photonfocus bases. The boolean marks the calibrated preview plane."""
    if raw.ndim != 2:
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if raw.ndim == 3 else raw
        return [("gray", gray, True)]
    out: list[tuple[str, np.ndarray, bool]] = []
    for pat in (4, 5):
        work = raw[:1024, :] if pat == 4 else raw
        h = (work.shape[0] // pat) * pat
        w = (work.shape[1] // pat) * pat
        if h <= 0 or w <= 0:
            continue
        work = work[:h, :w]
        bands = [cv2.medianBlur(work[r::pat, c::pat].astype(np.float32), 3) for r in range(pat) for c in range(pat)]
        stack = np.stack(bands, axis=-1)
        intrinsic_plane = pat == 4
        out.extend(
            [
                (f"mean{pat}", np.mean(stack, axis=-1), intrinsic_plane),
                (f"median{pat}", np.median(stack, axis=-1), intrinsic_plane),
                (f"max{pat}", np.max(stack, axis=-1), intrinsic_plane),
                (f"min{pat}", np.min(stack, axis=-1), intrinsic_plane),
                (f"range{pat}", np.max(stack, axis=-1) - np.min(stack, axis=-1), intrinsic_plane),
                (f"std{pat}", np.std(stack, axis=-1), intrinsic_plane),
            ]
        )
        idxs = [0, pat + 1, 2 * pat + 2, pat * pat - 1, 1, pat, max(0, pat * pat // 2)]
        for idx in idxs:
            if 0 <= idx < len(bands):
                mean = np.mean(stack, axis=-1)
                std = np.std(stack, axis=-1) + 1e-3
                out.append((f"band{pat}_{idx:02d}", bands[idx], intrinsic_plane))
                out.append((f"zband{pat}_{idx:02d}", (bands[idx] - mean) / std, intrinsic_plane))
        pairs = [(0, pat * pat - 1), (pat + 1, min(len(bands) - 1, 2 * pat + 2)), (1, pat)]
        for a_idx, b_idx in pairs:
            if a_idx < len(bands) and b_idx < len(bands):
                out.append(
                    (
                        f"ndi{pat}_{a_idx:02d}_{b_idx:02d}",
                        (bands[a_idx] - bands[b_idx]) / (bands[a_idx] + bands[b_idx] + 1.0),
                        intrinsic_plane,
                    )
                )
    return out


def variant_quality(u8: np.ndarray, guide: list[float] | None) -> float:
    roi = expanded_box(guide, u8.shape[:2], 0.35)
    crop = u8 if roi is None else u8[roi[1] : roi[3], roi[0] : roi[2]]
    if crop.size == 0:
        return -1e9
    gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)
    sat = float(np.mean((crop <= 2) | (crop >= 253)))
    return float(np.std(crop)) + 0.25 * float(np.mean(np.abs(gx) + np.abs(gy))) - 18.0 * sat


def rescue_variants(
    img: np.ndarray,
    sensor: str,
    guide: list[float] | None,
    guide_shape_hw: tuple[int, int],
    max_variants: int,
) -> list[tuple[str, np.ndarray, bool, list[float] | None]]:
    """Return rescue detector variants plus whether each can be used for K calibration."""
    raw_variants: list[tuple[str, np.ndarray, bool]] = []
    if sensor in ("vis", "nir"):
        raw_variants = photonfocus_rescue_bases(img, sensor)
    elif sensor in ("thermal_c", "thermal_raw") and img.ndim == 2:
        raw_variants = [(name, u8, True) for name, u8 in preprocess_raw_variants(img.astype(np.float32))]
        raw_variants.extend(
            [
                ("raw_p01_99", robust_u8(img, 1, 99), True),
                ("raw_p05_95", robust_u8(img, 5, 95), True),
                ("raw_detail", local_detail_u8(img, 13), True),
            ]
        )
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        raw_variants = [("gray", gray, True)]

    expanded: list[tuple[str, np.ndarray, bool, list[float] | None, float]] = []
    for name, base, intrinsic_plane in raw_variants:
        base_u8 = base if base.dtype == np.uint8 else robust_u8(base)
        detector_inputs = [
            (f"{name}_norm", base_u8),
            (f"{name}_clahe", clahe(base_u8, 2.4, 8)),
            (f"{name}_clahe_fine", clahe(base_u8, 3.0, 4)),
            (f"{name}_unsharp", unsharp(base_u8)),
            (f"{name}_detail", local_detail_u8(base_u8, 7)),
            (f"{name}_bilateral", cv2.bilateralFilter(base_u8, 5, 25, 5)),
        ]
        for vname, u8 in detector_inputs:
            vguide = rescale_box(guide, guide_shape_hw, u8.shape[:2]) if guide is not None else None
            expanded.append((vname, u8, intrinsic_plane, vguide, variant_quality(u8, vguide)))

    expanded.sort(key=lambda item: (item[4], 1 if item[2] else 0), reverse=True)
    out: list[tuple[str, np.ndarray, bool, list[float] | None]] = []
    seen: set[str] = set()
    for name, u8, intrinsic_plane, vguide, _score in expanded:
        if name in seen:
            continue
        out.append((name, u8, intrinsic_plane, vguide))
        seen.add(name)
        if max_variants > 0 and len(out) >= max_variants:
            break
    return out


def rescue_patterns(primary: tuple[int, int]) -> list[tuple[int, int]]:
    patterns = [
        primary,
        (primary[0], max(3, primary[1] - 1)),
        (max(3, primary[0] - 1), max(3, primary[1] - 1)),
        (primary[0], max(3, primary[1] - 2)),
        (max(3, primary[0] - 1), max(3, primary[1] - 2)),
        (max(3, primary[0] - 2), max(3, primary[1] - 2)),
        (max(3, primary[0] - 3), max(3, primary[1] - 3)),
    ]
    out: list[tuple[int, int]] = []
    for pat in patterns:
        if pat not in out and pat[0] >= 3 and pat[1] >= 3:
            out.append(pat)
    return out


def canonicalize_grid(corners: np.ndarray, pattern: tuple[int, int]) -> tuple[np.ndarray, bool, float]:
    cols, rows = pattern
    pts = corners.reshape(rows, cols, 2).astype(np.float32)
    row_vec = pts[-1, 0] - pts[0, 0]
    col_vec = pts[0, -1] - pts[0, 0]
    cross = float(col_vec[0] * row_vec[1] - col_vec[1] * row_vec[0])
    rotation_ok = bool(cross > 0 and np.linalg.norm(col_vec) >= 0.55 * np.linalg.norm(row_vec))
    if cross < 0:
        pts = pts[::-1, :, :]
        row_vec = pts[-1, 0] - pts[0, 0]
        col_vec = pts[0, -1] - pts[0, 0]
        cross = float(col_vec[0] * row_vec[1] - col_vec[1] * row_vec[0])
        rotation_ok = bool(cross > 0 and np.linalg.norm(col_vec) >= 0.55 * np.linalg.norm(row_vec))
    angle = float(np.degrees(np.arctan2(float(col_vec[1]), float(col_vec[0]))))
    return pts.reshape(-1, 2), rotation_ok, angle


def draw_rescue_preview_u8(
    u8: np.ndarray,
    corners: np.ndarray | None,
    pattern: tuple[int, int] | None,
    title: str,
    out_path: Path,
) -> None:
    color = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    if corners is not None:
        pts = corners.astype(np.int32)
        for i, p in enumerate(pts):
            c = (0, 255, 0)
            if i == 0:
                c = (0, 0, 255)
            elif pattern is not None and i == pattern[0] - 1:
                c = (255, 0, 0)
            cv2.circle(color, tuple(p), 3, c, -1, cv2.LINE_AA)
        hull = cv2.convexHull(pts.reshape(-1, 1, 2).astype(np.float32)).astype(np.int32)
        cv2.polylines(color, [hull], True, (0, 220, 0), 2, cv2.LINE_AA)
    h, w = color.shape[:2]
    scale = min(900 / max(w, 1), 650 / max(h, 1), 3.0)
    interp = cv2.INTER_NEAREST if scale > 1.0 else cv2.INTER_AREA
    view = cv2.resize(color, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=interp)
    cv2.putText(view, title[:150], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), view)


SB_FLAGS_FAST = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY
SB_FLAGS_DEEP = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY


def rotate_image(img: np.ndarray, deg: float) -> tuple[np.ndarray, np.ndarray]:
    if deg == 0:
        return img, np.eye(2, 3, dtype=np.float32)
    h, w = img.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0).astype(np.float32)
    rot = cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rot, mat


def invert_affine(mat: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.vstack([mat, [0, 0, 1]]))[:2].astype(np.float32)


def detect_u8_checker(
    u8: np.ndarray,
    pattern: tuple[int, int],
    scales: tuple[float, ...],
    rotations: tuple[float, ...],
    try_invert: bool,
    exhaustive: bool,
    allow_sb: bool = True,
) -> tuple[np.ndarray | None, str]:
    classic_fast_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    # Try the cheap detector first. In the 20260623 RGB captures this often
    # succeeds and is far faster than exhaustive SB on large frames.
    for scale in scales:
        scaled = cv2.resize(
            u8,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC if scale >= 1 else cv2.INTER_AREA,
        )
        for inverted, src in ((False, scaled), (True, cv2.bitwise_not(scaled))):
            ok, corners = cv2.findChessboardCorners(src, pattern, flags=classic_fast_flags)
            if ok and corners is not None:
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 1e-4)
                cv2.cornerSubPix(src, corners, (9, 9), (-1, -1), term)
                pts = corners.reshape(-1, 2).astype(np.float32) / scale
                return pts, f"classic_fast_s{scale:g}_{'inv' if inverted else 'pos'}"

    if allow_sb:
        sb_flags = SB_FLAGS_DEEP if exhaustive else SB_FLAGS_FAST
        for scale in scales:
            scaled = cv2.resize(
                u8,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC if scale >= 1 else cv2.INTER_AREA,
            )
            for deg in rotations:
                rot, mat = rotate_image(scaled, deg)
                candidates = [(False, rot)]
                if try_invert:
                    candidates.append((True, cv2.bitwise_not(rot)))
                for inverted, src in candidates:
                    ok, corners = cv2.findChessboardCornersSB(src, pattern, flags=sb_flags)
                    if ok and corners is not None:
                        pts = corners.reshape(-1, 2).astype(np.float32)
                        inv = invert_affine(mat)
                        pts = (inv[:, :2] @ pts.T + inv[:, 2:3]).T / scale
                        tag = f"SB_s{scale:g}_r{deg:g}_{'inv' if inverted else 'pos'}"
                        return pts.astype(np.float32), tag

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    for inverted, src in ((False, u8), (True, cv2.bitwise_not(u8))):
        ok, corners = cv2.findChessboardCorners(src, pattern, flags=flags)
        if ok and corners is not None:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 1e-4)
            cv2.cornerSubPix(src, corners, (9, 9), (-1, -1), term)
            return corners.reshape(-1, 2).astype(np.float32), f"classic_{'inv' if inverted else 'pos'}"
    return None, ""


def checker_score(corners: np.ndarray, image_size_wh: tuple[int, int], u8: np.ndarray) -> float:
    w, h = image_size_wh
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    area_ratio = max(0.0, float((x1 - x0) * (y1 - y0)) / max(1.0, float(w * h)))
    margin = min(x0, y0, w - x1, h - y1) / max(1.0, min(w, h))
    margin_score = float(np.clip(margin * 8.0, -1.0, 1.0))
    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(corners.reshape(-1, 1, 2).astype(np.float32)).astype(np.int32)
    cv2.fillConvexPoly(mask, hull, 255)
    lap = cv2.Laplacian(u8, cv2.CV_32F)
    sharp = float(np.nanstd(lap[mask > 0])) if np.any(mask > 0) else 0.0
    return area_ratio * 100.0 + margin_score + min(sharp / 150.0, 1.0)


def draw_preview(
    img: np.ndarray,
    sensor: str,
    corners: np.ndarray | None,
    label: str,
    out_path: Path,
) -> None:
    prev = robust_preview(img, sensor)
    if prev.ndim == 2:
        prev = cv2.cvtColor(prev, cv2.COLOR_GRAY2BGR)
    h, w = prev.shape[:2]
    scale = min(900 / max(w, 1), 650 / max(h, 1), 1.0)
    view = cv2.resize(prev, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    if corners is not None:
        pts = (corners * scale).astype(np.int32)
        for i, p in enumerate(pts):
            cv2.circle(view, tuple(p), 3, (0, 255, 255), -1, cv2.LINE_AA)
            if i in (0, len(pts) - 1):
                cv2.circle(view, tuple(p), 7, (0, 0, 255) if i == 0 else (0, 255, 0), 2, cv2.LINE_AA)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.putText(view, label[:150], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), view)


def detect_in_image(
    img: np.ndarray,
    sensor: str,
    pattern: tuple[int, int],
    crops: bool,
    deep: bool,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if sensor in ("thermal_c", "thermal_raw") and img.ndim == 2:
        variants = preprocess_raw_variants(img.astype(np.float32))
        if deep:
            ok, corners, info = detect_board_raw(img.astype(np.float32), pattern, do_badpix=True, do_destripe=False)
            if ok and corners is not None:
                u8 = variants[0][1]
                return corners.reshape(-1, 2).astype(np.float32), {"method": "thermal_raw", **info, "score_u8": u8}
            return None, {"method": "thermal_raw_none", **info}
        for name, u8 in variants[:2]:
            corners, method = detect_u8_checker(
                u8,
                pattern,
                scales=(1.0, 0.5),
                rotations=(0,),
                try_invert=True,
                exhaustive=False,
                allow_sb=False,
            )
            if corners is not None:
                return corners.reshape(-1, 2).astype(np.float32), {
                    "method": f"thermal_fast_{method}",
                    "variant": name,
                    "score_u8": u8,
                }
        return None, {"method": "thermal_fast_none"}

    if sensor.startswith("thermal") and img.ndim == 3:
        rgb = img[:, :, ::-1] if img.shape[2] == 3 else img
        ok, corners, info = detect_thermal_colormap(rgb, pattern)
        if ok and corners is not None:
            return corners.reshape(-1, 2).astype(np.float32), {"method": "thermal_colormap", **info}
        return None, {"method": "thermal_colormap_none", **info}

    if sensor == "rgb":
        scales = (0.25, 0.5, 1.0, 1.5) if deep else (0.25, 0.5)
    else:
        scales = (1.0, 2.0, 0.5, 1.5) if deep else (1.0, 0.5)
    rotations = (0, 6, -6, 12, -12) if deep else (0,)
    best: tuple[float, np.ndarray, dict[str, Any]] | None = None
    for variant_name, u8 in visible_variants(img, sensor, include_bands=deep):
        for x0, y0, x1, y1, region_name in search_regions(u8.shape[:2], crops=crops):
            crop = u8[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            corners, method = detect_u8_checker(
                crop,
                pattern,
                scales=scales,
                rotations=rotations,
                try_invert=True,
                exhaustive=deep,
            )
            if corners is None:
                continue
            corners[:, 0] += x0
            corners[:, 1] += y0
            score = checker_score(corners, (u8.shape[1], u8.shape[0]), u8)
            info = {
                "method": method,
                "variant": variant_name,
                "region": region_name,
                "score_u8": u8,
                "score": score,
            }
            if best is None or score > best[0]:
                best = (score, corners, info)
    if best is None:
        return None, {"method": "none"}
    return best[1], best[2]


def candidate_indices(n: int, max_frames: int, every_n: int, sample_mode: str) -> set[int]:
    if n <= 0:
        return set()
    if sample_mode == "first":
        stop = n if max_frames <= 0 else min(n, max_frames * max(1, every_n))
        base = set(range(0, stop, max(1, every_n)))
        if max_frames > 0:
            base = set(sorted(base)[:max_frames])
        return {i for i in base if 0 <= i < n}
    if every_n > 1:
        base = set(range(0, n, every_n))
    else:
        base = set(range(n))
    if max_frames > 0 and len(base) > max_frames:
        base = set(np.linspace(0, n - 1, max_frames, dtype=int).tolist())
    base.update([0, n // 2, n - 1])
    return {i for i in base if 0 <= i < n}


def read_manifest(path: Path, include_all: bool, include_test: bool, limit_bags: int) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not include_all:
        rows = [r for r in rows if str(r.get("include_default", "")).lower() == "yes"]
    if not include_test:
        rows = [r for r in rows if str(r.get("height_level", "")).lower() != "test"]
    if limit_bags > 0:
        rows = rows[:limit_bags]
    return rows


def scan_bag_sensor(
    row: dict[str, str],
    sensor: str,
    topic: str,
    args: argparse.Namespace,
    typestore,
    pattern: tuple[int, int],
) -> Detection | None:
    bag = Path(row["bag_path"])
    label = row.get("label_norm") or row.get("label_raw") or bag.stem
    if not bag.exists():
        return None

    declared_count = int(float(row.get(COUNT_COLUMNS.get(sensor, ""), 0) or 0))
    wanted = candidate_indices(declared_count, args.max_frames_per_bag, args.every_n, args.sample_mode) if declared_count else set()
    max_wanted = max(wanted) if wanted else None
    best: Detection | None = None
    seen = 0
    checked = 0

    with Reader(bag) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            return None
        for conn, ts, raw in reader.messages(connections=conns):
            use_frame = seen in wanted if wanted else (args.every_n <= 1 or seen % args.every_n == 0)
            if use_frame:
                msg = typestore.deserialize_ros1(raw, conn.msgtype)
                img = decode_ros_image(msg)
                if img is not None:
                    corners, info = detect_in_image(img, sensor, pattern, crops=args.crops, deep=args.deep)
                    checked += 1
                    if corners is not None:
                        if "score_u8" in info:
                            score_u8 = info.pop("score_u8")
                        else:
                            score_u8 = robust_u8(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img)
                        if "score" in info:
                            score = float(info["score"])
                        else:
                            score = checker_score(corners, (img.shape[1], img.shape[0]), score_u8)
                        x0, y0 = corners.min(axis=0)
                        x1, y1 = corners.max(axis=0)
                        preview_rel = Path("previews") / sensor / f"{label}_{bag.stem}_{seen:05d}.jpg"
                        det = Detection(
                            sensor=sensor,
                            bag=bag.name,
                            bag_path=str(bag),
                            label=label,
                            frame_index=seen,
                            stamp_ns=stamp_ns(msg, int(ts)),
                            image_size_wh=(int(score_u8.shape[1]), int(score_u8.shape[0])),
                            corners=corners.astype(np.float32),
                            method=str(info.get("method", "")),
                            variant=str(info.get("variant", info.get("region", ""))),
                            score=score,
                            bbox_xyxy=(float(x0), float(y0), float(x1), float(y1)),
                            preview_path=preview_rel.as_posix(),
                            encoding=str(getattr(msg, "encoding", "")),
                        )
                        if best is None or det.score > best.score:
                            best = det
                if args.max_frames_per_bag > 0 and checked >= args.max_frames_per_bag:
                    # When declared counts are absent, this bounds the scan.
                    pass
            seen += 1
            if max_wanted is not None and seen > max_wanted:
                break
            if not wanted and args.max_frames_per_bag > 0 and checked >= args.max_frames_per_bag:
                break

    if best is not None:
        # Re-read the best frame only to write a preview. This avoids keeping
        # large raw frames alive while scanning.
        with Reader(bag) as reader:
            conns = [c for c in reader.connections if c.topic == topic]
            idx = 0
            for conn, ts, raw in reader.messages(connections=conns):
                if idx == best.frame_index:
                    msg = typestore.deserialize_ros1(raw, conn.msgtype)
                    img = decode_ros_image(msg)
                    if img is not None:
                        label_text = f"{sensor} {best.label} frame={best.frame_index} {best.method} {best.variant}"
                        draw_preview(img, sensor, best.corners, label_text, args.out / best.preview_path)
                    break
                idx += 1
    return best


def scan_bag_sensor_rescue(
    row: dict[str, str],
    sensor: str,
    topic: str,
    args: argparse.Namespace,
    typestore,
    pattern: tuple[int, int],
    guides: dict[str, dict[str, Any]],
) -> tuple[Detection | None, list[dict[str, Any]]]:
    """Guided heavy recovery pass for difficult cameras.

    Full 9x6 detections in the calibrated image plane can be promoted for
    intrinsics. Partial detections and alternate Photonfocus planes are kept
    only as QA/registration evidence.
    """
    bag = Path(row["bag_path"])
    label = row.get("label_norm") or row.get("label_raw") or bag.stem
    if not bag.exists():
        return None, []

    declared_count = int(float(row.get(COUNT_COLUMNS.get(sensor, ""), 0) or 0))
    wanted = candidate_indices(declared_count, args.rescue_max_frames_per_bag, args.every_n, args.sample_mode) if declared_count else set()
    max_wanted = max(wanted) if wanted else None
    candidates: list[dict[str, Any]] = []
    best_promotable: dict[str, Any] | None = None
    seen = 0
    checked = 0

    if sensor == "rgb":
        scales = (0.25, 0.5, 1.0)
    else:
        scales = (1.0, 1.5, 0.5) if not args.rescue_exhaustive else (1.0, 1.5, 2.0, 0.5)
    rotations = (0, 6, -6) if not args.rescue_exhaustive else (0, 6, -6, 12, -12)

    with Reader(bag) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            return None, []
        for conn, ts, raw in reader.messages(connections=conns):
            use_frame = seen in wanted if wanted else (args.every_n <= 1 or seen % args.every_n == 0)
            if use_frame:
                msg = typestore.deserialize_ros1(raw, conn.msgtype)
                img = decode_ros_image(msg)
                checked += 1
                if img is not None:
                    if sensor in ("vis", "nir") and img.ndim == 2:
                        guide_shape = photonfocus_preview(img).shape[:2]
                    else:
                        guide_shape = img.shape[:2]
                    guide = rescue_guide_box(label, sensor, guides, guide_shape)
                    for variant_name, u8, intrinsic_plane, vguide in rescue_variants(
                        img,
                        sensor,
                        guide,
                        guide_shape,
                        max_variants=args.rescue_max_variants,
                    ):
                        regions = rescue_regions(u8.shape[:2], vguide, include_broad=args.rescue_broad or vguide is None)
                        patterns_to_try = rescue_patterns(pattern)
                        if sensor.startswith("thermal") and not args.rescue_exhaustive:
                            patterns_to_try = [p for p in (pattern, (pattern[0], max(3, pattern[1] - 1)), (max(3, pattern[0] - 1), max(3, pattern[1] - 1))) if p[0] >= 3 and p[1] >= 3]
                        for x0, y0, x1, y1, region_name in regions:
                            crop = u8[y0:y1, x0:x1]
                            if crop.size == 0:
                                continue
                            for pat in patterns_to_try:
                                pts, method = detect_u8_checker(
                                    crop,
                                    pat,
                                    scales=scales,
                                    rotations=rotations,
                                    try_invert=True,
                                    exhaustive=args.rescue_exhaustive,
                                    allow_sb=(not sensor.startswith("thermal")) or args.rescue_exhaustive,
                                )
                                if pts is None:
                                    continue
                                pts[:, 0] += x0
                                pts[:, 1] += y0
                                pts, rotation_ok, angle = canonicalize_grid(pts, pat)
                                score = checker_score(pts, (u8.shape[1], u8.shape[0]), u8)
                                score += float(pat[0] * pat[1]) * 0.35
                                if pat == pattern:
                                    score += 50.0
                                if rotation_ok:
                                    score += 15.0
                                usable_for_intrinsics = bool(pat == pattern and intrinsic_plane and rotation_ok)
                                confidence = "subpixel_full_rescue" if usable_for_intrinsics else "subpixel_partial_rescue"
                                if pat == pattern and not intrinsic_plane:
                                    confidence = "subpixel_full_alt_plane"
                                if not rotation_ok:
                                    confidence += "_rotation_warning"
                                x_min, y_min = pts.min(axis=0)
                                x_max, y_max = pts.max(axis=0)
                                rec: dict[str, Any] = {
                                    "sensor": sensor,
                                    "label": label,
                                    "bag": bag.name,
                                    "bag_path": str(bag),
                                    "frame_index": seen,
                                    "stamp_ns": stamp_ns(msg, int(ts)),
                                    "encoding": str(getattr(msg, "encoding", "")),
                                    "image_size_wh": [int(u8.shape[1]), int(u8.shape[0])],
                                    "pattern_internal_corners": [int(pat[0]), int(pat[1])],
                                    "method": method,
                                    "variant": variant_name,
                                    "region": region_name,
                                    "score": float(score),
                                    "bbox_xyxy": [float(x_min), float(y_min), float(x_max), float(y_max)],
                                    "corners_px": pts.astype(float).tolist(),
                                    "confidence": confidence,
                                    "rotation_ok": bool(rotation_ok),
                                    "orientation_deg": float(angle),
                                    "intrinsic_plane": bool(intrinsic_plane),
                                    "usable_for_intrinsics": usable_for_intrinsics,
                                    "usable_for_registration": bool(rotation_ok),
                                    "_u8": u8,
                                    "_corners": pts.astype(np.float32),
                                }
                                candidates.append(rec)
                                if usable_for_intrinsics and (best_promotable is None or score > float(best_promotable["score"])):
                                    best_promotable = rec
                                if usable_for_intrinsics and not args.rescue_exhaustive_after_full:
                                    break
                            if best_promotable is not None and not args.rescue_exhaustive_after_full:
                                break
                        if best_promotable is not None and not args.rescue_exhaustive_after_full:
                            break
                if args.rescue_max_frames_per_bag > 0 and checked >= args.rescue_max_frames_per_bag:
                    pass
            seen += 1
            if max_wanted is not None and seen > max_wanted:
                break
            if not wanted and args.rescue_max_frames_per_bag > 0 and checked >= args.rescue_max_frames_per_bag:
                break
            if best_promotable is not None and not args.rescue_exhaustive_after_full:
                break

    if not candidates:
        return None, []

    candidates.sort(key=lambda c: float(c["score"]), reverse=True)
    keep = candidates[: max(1, args.rescue_keep_candidates)]
    if best_promotable is not None and not any(rec is best_promotable for rec in keep):
        keep.append(best_promotable)

    for rank, rec in enumerate(keep, start=1):
        preview_rel = Path("rescue_previews") / sensor / f"{label}_{bag.stem}_{rec['frame_index']:05d}_{rank:02d}.jpg"
        title = (
            f"{sensor} {label} f={rec['frame_index']} {rec['confidence']} "
            f"{rec['pattern_internal_corners']} {rec['variant']}/{rec['region']}"
        )
        draw_rescue_preview_u8(
            rec["_u8"],
            rec["_corners"],
            tuple(rec["pattern_internal_corners"]),
            title,
            args.out / preview_rel,
        )
        rec["preview_path"] = preview_rel.as_posix()

    # Strip numpy payloads before returning report records.
    report = []
    for rec in keep:
        clean = {k: v for k, v in rec.items() if not k.startswith("_")}
        report.append(clean)

    promoted: Detection | None = None
    if best_promotable is not None:
        if "preview_path" not in best_promotable:
            preview_rel = Path("rescue_previews") / sensor / f"{label}_{bag.stem}_{best_promotable['frame_index']:05d}_promoted.jpg"
            draw_rescue_preview_u8(
                best_promotable["_u8"],
                best_promotable["_corners"],
                tuple(best_promotable["pattern_internal_corners"]),
                f"{sensor} {label} promoted rescue",
                args.out / preview_rel,
            )
            best_promotable["preview_path"] = preview_rel.as_posix()
        promoted = Detection(
            sensor=sensor,
            bag=bag.name,
            bag_path=str(bag),
            label=label,
            frame_index=int(best_promotable["frame_index"]),
            stamp_ns=int(best_promotable["stamp_ns"]),
            image_size_wh=(int(best_promotable["image_size_wh"][0]), int(best_promotable["image_size_wh"][1])),
            corners=np.asarray(best_promotable["_corners"], dtype=np.float32),
            method=str(best_promotable["method"]),
            variant=f"rescue:{best_promotable['variant']}:{best_promotable['region']}",
            score=float(best_promotable["score"]),
            bbox_xyxy=tuple(float(v) for v in best_promotable["bbox_xyxy"]),
            preview_path=str(best_promotable["preview_path"]),
            encoding=str(best_promotable["encoding"]),
            confidence=str(best_promotable["confidence"]),
            rescue=True,
            usable_for_intrinsics=True,
            usable_for_registration=True,
        )
    return promoted, report


def reprojection_errors(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    rvecs: Any,
    tvecs: Any,
    k: np.ndarray,
    dist: np.ndarray,
) -> list[dict[str, float]]:
    out = []
    for obj, img, rv, tv in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, rv, tv, k, dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - img.reshape(-1, 2), axis=1)
        out.append(
            {
                "mean_px": float(np.mean(err)),
                "median_px": float(np.median(err)),
                "rms_px": float(np.sqrt(np.mean(err**2))),
                "max_px": float(np.max(err)),
            }
        )
    return out


def default_initial_k(image_size_wh: tuple[int, int]) -> np.ndarray:
    w, h = image_size_wh
    f = float(max(w, h) * 1.4)
    return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def load_initial_k(path: Path, sensor: str, image_size_wh: tuple[int, int]) -> tuple[np.ndarray | None, str]:
    if not path.exists():
        return None, ""
    data = json.loads(path.read_text(encoding="utf-8"))
    key = INITIAL_INTRINSIC_KEYS.get(sensor, "")
    item = data.get(key)
    if not item:
        return None, ""
    res = tuple(int(v) for v in item.get("resolution_px", []))
    if len(res) == 2 and res != tuple(image_size_wh):
        # Do not silently use raw-mosaic intrinsics in the demosaiced preview
        # plane or vice versa.
        return None, f"{key}:resolution_mismatch_{res[0]}x{res[1]}"
    return np.asarray(item["K_initial"], dtype=np.float64), key


def calibration_flags(model: str) -> int:
    if model == "standard":
        return 0
    if model == "zero_tangent":
        return cv2.CALIB_ZERO_TANGENT_DIST
    if model == "fixed_principal_zero_tangent_k1k2":
        return cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    if model == "rational":
        return cv2.CALIB_RATIONAL_MODEL
    raise ValueError(f"unknown model: {model}")


def run_calibrate(
    detections: list[Detection],
    pattern: tuple[int, int],
    square_size_m: float,
    model: str,
    min_views: int,
    max_outlier_passes: int,
    initial_intrinsics_path: Path,
    sensor: str,
) -> dict[str, Any]:
    full_count = pattern[0] * pattern[1]
    usable = [d for d in detections if d.usable_for_intrinsics and d.corners.reshape(-1, 2).shape[0] == full_count]
    if len(usable) < min_views:
        return {
            "status": "insufficient_views",
            "n_views": len(usable),
            "n_views_input": len(detections),
            "min_views": min_views,
        }

    # Use the most common image size. Mixed sizes indicate different decoded
    # image products and must not be mixed in one K calibration.
    size_counts: dict[tuple[int, int], int] = {}
    for d in usable:
        size_counts[d.image_size_wh] = size_counts.get(d.image_size_wh, 0) + 1
    image_size = sorted(size_counts.items(), key=lambda kv: kv[1], reverse=True)[0][0]
    active = [d for d in usable if d.image_size_wh == image_size]
    if len(active) < min_views:
        return {
            "status": "insufficient_views_for_common_image_size",
            "n_views": len(active),
            "min_views": min_views,
            "image_size_wh": list(image_size),
            "all_image_sizes": {f"{k[0]}x{k[1]}": v for k, v in size_counts.items()},
        }

    flags = calibration_flags(model)
    k_loaded, k_source = load_initial_k(initial_intrinsics_path, sensor, image_size)
    k0 = k_loaded.copy() if k_loaded is not None else default_initial_k(image_size)
    dist0 = np.zeros((8 if model == "rational" else 5, 1), dtype=np.float64)
    obj_base = object_points(pattern, square_size_m)
    removed: list[dict[str, Any]] = []

    for pass_idx in range(max(1, max_outlier_passes + 1)):
        objpoints = [obj_base.copy() for _ in active]
        imgpoints = [d.corners.reshape(-1, 1, 2).astype(np.float32) for d in active]
        rms, k, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints,
            imgpoints,
            image_size,
            k0.copy(),
            dist0.copy(),
            flags=flags,
        )
        errors = reprojection_errors(objpoints, imgpoints, rvecs, tvecs, k, dist)
        means = np.array([e["mean_px"] for e in errors], dtype=np.float64)
        if pass_idx >= max_outlier_passes or len(active) <= min_views:
            break
        med = float(np.median(means))
        mad = float(np.median(np.abs(means - med)))
        robust_sigma = 1.4826 * mad if mad > 1e-9 else float(np.std(means))
        threshold = max(1.5, 2.5 * med, med + 3.0 * robust_sigma)
        worst = int(np.argmax(means))
        if means[worst] <= threshold:
            break
        removed.append({"label": active[worst].label, "bag": active[worst].bag, "mean_px": float(means[worst]), "threshold_px": threshold})
        active.pop(worst)

    views = []
    for det, err, rv, tv in zip(active, errors, rvecs, tvecs):
        views.append(
            {
                "label": det.label,
                "bag": det.bag,
                "frame_index": det.frame_index,
                "stamp_ns": det.stamp_ns,
                "method": det.method,
                "variant": det.variant,
                "confidence": det.confidence,
                "rescue": det.rescue,
                "score": det.score,
                "bbox_xyxy": list(det.bbox_xyxy),
                "preview_path": det.preview_path,
                "rvec": np.asarray(rv).reshape(-1).tolist(),
                "tvec_m": np.asarray(tv).reshape(-1).tolist(),
                **err,
            }
        )

    mean_px = [v["mean_px"] for v in views]
    return {
        "status": "ok",
        "model": model,
        "pattern_internal_corners": list(pattern),
        "square_size_m": square_size_m,
        "image_size_wh": list(image_size),
        "n_views_input": len(detections),
        "n_views_usable": len(usable),
        "n_views_used": len(active),
        "removed_outliers": removed,
        "rms_px": float(rms),
        "mean_view_error_px": float(np.mean(mean_px)),
        "median_view_error_px": float(np.median(mean_px)),
        "max_view_error_px": float(np.max(mean_px)),
        "K": np.asarray(k).tolist(),
        "dist_coeffs": np.asarray(dist).reshape(-1).tolist(),
        "flags": int(flags),
        "K_initial": np.asarray(k0).tolist(),
        "K_initial_source": k_source or "heuristic_from_image_size",
        "views": views,
    }


def make_contact_sheet(detections: list[Detection], out_path: Path, root: Path, title: str) -> None:
    thumbs = []
    for det in detections:
        img = cv2.imread(str(root / det.preview_path))
        if img is None:
            continue
        img = cv2.resize(img, (260, 180), interpolation=cv2.INTER_AREA)
        label = np.full((34, 260, 3), 24, dtype=np.uint8)
        cv2.putText(label, f"{det.label[:22]} e={det.score:.1f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
        thumbs.append(np.vstack([label, img]))
    if not thumbs:
        return
    cols = 4
    rows = []
    blank = np.full_like(thumbs[0], 35)
    for i in range(0, len(thumbs), cols):
        chunk = thumbs[i : i + cols]
        while len(chunk) < cols:
            chunk.append(blank.copy())
        rows.append(np.hstack(chunk))
    sheet = np.vstack(rows)
    header = np.full((42, sheet.shape[1], 3), 12, dtype=np.uint8)
    cv2.putText(header, title[:120], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack([header, sheet]))


def write_detection_tables(out: Path, detections_by_sensor: dict[str, list[Detection]]) -> None:
    rows = []
    for sensor, detections in detections_by_sensor.items():
        for d in detections:
            rows.append(
                {
                    "sensor": sensor,
                    "label": d.label,
                    "bag": d.bag,
                    "frame_index": d.frame_index,
                    "stamp_ns": d.stamp_ns,
                    "image_width": d.image_size_wh[0],
                    "image_height": d.image_size_wh[1],
                    "method": d.method,
                    "variant": d.variant,
                    "confidence": d.confidence,
                    "rescue": d.rescue,
                    "usable_for_intrinsics": d.usable_for_intrinsics,
                    "usable_for_registration": d.usable_for_registration,
                    "score": f"{d.score:.6f}",
                    "bbox_x0": f"{d.bbox_xyxy[0]:.3f}",
                    "bbox_y0": f"{d.bbox_xyxy[1]:.3f}",
                    "bbox_x1": f"{d.bbox_xyxy[2]:.3f}",
                    "bbox_y1": f"{d.bbox_xyxy[3]:.3f}",
                    "preview_path": d.preview_path,
                    "encoding": d.encoding,
                }
            )
    if not rows:
        return
    with (out / "detections.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_rescue_outputs(out: Path, rescue_candidates: list[dict[str, Any]]) -> None:
    if not rescue_candidates:
        return
    (out / "rescue_candidates.json").write_text(json.dumps(rescue_candidates, indent=2), encoding="utf-8")
    flat_rows = []
    fields = [
        "sensor",
        "label",
        "bag",
        "frame_index",
        "stamp_ns",
        "image_width",
        "image_height",
        "pattern",
        "method",
        "variant",
        "region",
        "confidence",
        "rotation_ok",
        "intrinsic_plane",
        "usable_for_intrinsics",
        "usable_for_registration",
        "score",
        "bbox_x0",
        "bbox_y0",
        "bbox_x1",
        "bbox_y1",
        "preview_path",
    ]
    for c in rescue_candidates:
        bbox = c.get("bbox_xyxy") or [None, None, None, None]
        size = c.get("image_size_wh") or [None, None]
        pat = c.get("pattern_internal_corners") or []
        flat_rows.append(
            {
                "sensor": c.get("sensor"),
                "label": c.get("label"),
                "bag": c.get("bag"),
                "frame_index": c.get("frame_index"),
                "stamp_ns": c.get("stamp_ns"),
                "image_width": size[0],
                "image_height": size[1],
                "pattern": "x".join(str(v) for v in pat),
                "method": c.get("method"),
                "variant": c.get("variant"),
                "region": c.get("region"),
                "confidence": c.get("confidence"),
                "rotation_ok": c.get("rotation_ok"),
                "intrinsic_plane": c.get("intrinsic_plane"),
                "usable_for_intrinsics": c.get("usable_for_intrinsics"),
                "usable_for_registration": c.get("usable_for_registration"),
                "score": f"{float(c.get('score', 0.0)):.6f}",
                "bbox_x0": "" if bbox[0] is None else f"{float(bbox[0]):.3f}",
                "bbox_y0": "" if bbox[1] is None else f"{float(bbox[1]):.3f}",
                "bbox_x1": "" if bbox[2] is None else f"{float(bbox[2]):.3f}",
                "bbox_y1": "" if bbox[3] is None else f"{float(bbox[3]):.3f}",
                "preview_path": c.get("preview_path", ""),
            }
        )
    with (out / "rescue_candidates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)

    thumbs = []
    for c in rescue_candidates[:80]:
        rel = c.get("preview_path")
        if not rel:
            continue
        img = cv2.imread(str(out / rel))
        if img is None:
            continue
        img = cv2.resize(img, (280, 190), interpolation=cv2.INTER_AREA)
        label = np.full((42, 280, 3), 22, dtype=np.uint8)
        text = f"{c.get('sensor')} {str(c.get('label'))[:17]} {c.get('confidence')}"
        cv2.putText(label, text[:38], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(label, f"pat={c.get('pattern_internal_corners')} e={float(c.get('score', 0.0)):.1f}", (6, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        thumbs.append(np.vstack([label, img]))
    if thumbs:
        cols = 4
        rows = []
        blank = np.full_like(thumbs[0], 35)
        for i in range(0, len(thumbs), cols):
            chunk = thumbs[i : i + cols]
            while len(chunk) < cols:
                chunk.append(blank.copy())
            rows.append(np.hstack(chunk))
        sheet = np.vstack(rows)
        header = np.full((44, sheet.shape[1], 3), 10, dtype=np.uint8)
        cv2.putText(header, "rescue checker candidates: review before using partial/model detections", (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out / "rescue_candidates_contactsheet.jpg"), np.vstack([header, sheet]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checker-config", type=Path, default=DEFAULT_CHECKER)
    parser.add_argument("--initial-intrinsics", type=Path, default=DEFAULT_INITIAL_INTRINSICS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sensors", default="rgb,vis,nir,thermal_c,thermal_raw")
    parser.add_argument("--include-all", action="store_true", help="Use all manifest rows instead of include_default=yes only.")
    parser.add_argument("--include-test", action="store_true", help="Include rows marked as height_level=test.")
    parser.add_argument("--limit-bags", type=int, default=0)
    parser.add_argument("--max-frames-per-bag", type=int, default=14)
    parser.add_argument("--every-n", type=int, default=1)
    parser.add_argument("--sample-mode", choices=["first", "even"], default="first", help="first is correct for one-static-pose-per-bag calibration captures.")
    parser.add_argument("--min-views", type=int, default=8)
    parser.add_argument(
        "--model",
        choices=["standard", "zero_tangent", "fixed_principal_zero_tangent_k1k2", "rational"],
        default="fixed_principal_zero_tangent_k1k2",
    )
    parser.add_argument("--outlier-passes", type=int, default=2)
    parser.add_argument("--crops", action="store_true", help="Search broad image crops in addition to the full image.")
    parser.add_argument("--deep", action="store_true", help="Use extra scales/rotations. Slower but useful for difficult frames.")
    parser.add_argument("--rescue", action="store_true", help="Run a guided heavy recovery pass for failed VIS/NIR/thermal detections.")
    parser.add_argument("--rescue-always", action="store_true", help="Also run rescue on sensors that already have a clean detection, for QA only.")
    parser.add_argument("--rescue-guide-summary", type=Path, default=DEFAULT_RESCUE_GUIDE_SUMMARY)
    parser.add_argument("--rescue-max-frames-per-bag", type=int, default=3)
    parser.add_argument("--rescue-max-variants", type=int, default=28)
    parser.add_argument("--rescue-keep-candidates", type=int, default=5)
    parser.add_argument("--rescue-broad", action="store_true", help="Allow broad image crops even when an RGB-guided ROI exists.")
    parser.add_argument("--rescue-exhaustive", action="store_true", help="Use exhaustive SB during rescue. Much slower; keep off for normal runs.")
    parser.add_argument("--rescue-exhaustive-after-full", action="store_true", help="Keep searching after the first full promotable rescue detection.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    pattern, square_size_m = load_checker_config(args.checker_config)
    sensors = [s.strip() for s in args.sensors.split(",") if s.strip()]
    unknown = [s for s in sensors if s not in TOPICS]
    if unknown:
        raise SystemExit(f"Unknown sensors: {unknown}. Valid: {sorted(TOPICS)}")

    rows = read_manifest(args.manifest, include_all=args.include_all, include_test=args.include_test, limit_bags=args.limit_bags)
    typestore = get_typestore(Stores.ROS1_NOETIC)
    rescue_guides = load_rescue_guides(args.rescue_guide_summary) if args.rescue else {}
    detections_by_sensor: dict[str, list[Detection]] = {s: [] for s in sensors}
    failures: list[dict[str, Any]] = []
    rescue_candidates: list[dict[str, Any]] = []

    for i, row in enumerate(rows, start=1):
        label = row.get("label_norm") or row.get("label_raw") or Path(row.get("bag_path", "")).stem
        print(f"[{i}/{len(rows)}] {label}")
        for sensor in sensors:
            topic = TOPICS[sensor]
            try:
                det = scan_bag_sensor(row, sensor, topic, args, typestore, pattern)
            except Exception as exc:
                failures.append({"label": label, "sensor": sensor, "bag_path": row.get("bag_path"), "error": repr(exc)})
                print(f"  {sensor}: ERROR {exc!r}")
                continue
            if args.rescue and sensor != "rgb" and (det is None or args.rescue_always):
                try:
                    promoted, rescue_records = scan_bag_sensor_rescue(row, sensor, topic, args, typestore, pattern, rescue_guides)
                    rescue_candidates.extend(rescue_records)
                    if det is None and promoted is not None:
                        det = promoted
                        print(f"  {sensor}: rescue promoted frame={det.frame_index} score={det.score:.2f} {det.method}/{det.variant}")
                    elif rescue_records:
                        usable = sum(1 for r in rescue_records if r.get("usable_for_intrinsics"))
                        print(f"  {sensor}: rescue candidates={len(rescue_records)} promotable={usable}")
                except Exception as exc:
                    failures.append({"label": label, "sensor": sensor, "bag_path": row.get("bag_path"), "error": f"rescue:{exc!r}"})
                    print(f"  {sensor}: rescue ERROR {exc!r}")
            if det is None:
                failures.append({"label": label, "sensor": sensor, "bag_path": row.get("bag_path"), "error": "no_detection_or_missing_topic"})
                print(f"  {sensor}: no detection")
            else:
                detections_by_sensor[sensor].append(det)
                print(f"  {sensor}: ok frame={det.frame_index} score={det.score:.2f} {det.method}/{det.variant}")

    calibrations: dict[str, Any] = {}
    for sensor, detections in detections_by_sensor.items():
        detections = sorted(detections, key=lambda d: d.label)
        detections_by_sensor[sensor] = detections
        make_contact_sheet(detections, args.out / f"{sensor}_detections_contactsheet.jpg", args.out, f"{sensor} checker detections")
        calib = run_calibrate(
            detections,
            pattern=pattern,
            square_size_m=square_size_m,
            model=args.model,
            min_views=args.min_views,
            max_outlier_passes=args.outlier_passes,
            initial_intrinsics_path=args.initial_intrinsics,
            sensor=sensor,
        )
        calibrations[sensor] = calib
        (args.out / f"{sensor}_intrinsics.json").write_text(json.dumps(calib, indent=2), encoding="utf-8")
        if calib.get("status") == "ok":
            with (args.out / f"{sensor}_views.csv").open("w", newline="", encoding="utf-8") as f:
                fields = [
                    "label",
                    "bag",
                    "frame_index",
                    "mean_px",
                    "median_px",
                    "rms_px",
                    "max_px",
                    "method",
                    "variant",
                    "confidence",
                    "rescue",
                    "preview_path",
                ]
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for v in calib["views"]:
                    writer.writerow({k: v.get(k, "") for k in fields})

    write_detection_tables(args.out, detections_by_sensor)
    write_rescue_outputs(args.out, rescue_candidates)
    summary = {
        "checker": {"internal_corners": list(pattern), "square_size_m": square_size_m},
        "manifest": str(args.manifest),
        "sensors": sensors,
        "n_bags": len(rows),
        "detections": {sensor: len(dets) for sensor, dets in detections_by_sensor.items()},
        "rescue_candidates": len(rescue_candidates),
        "rescue_promoted_detections": {
            sensor: sum(1 for d in dets if d.rescue) for sensor, dets in detections_by_sensor.items()
        },
        "calibration_status": {sensor: c.get("status") for sensor, c in calibrations.items()},
        "calibrations": calibrations,
        "failures": failures,
        "notes": [
            "VIS/NIR intrinsics are for the Photonfocus demosaiced/downsampled preview image plane used by the RGB-master pipeline.",
            "thermal_c and thermal_raw intrinsics are calibrated in their scalar image coordinate systems.",
            "Default behavior keeps one best checkerboard detection per bag/sensor to avoid overweighting static duplicate frames.",
            "Rescue detections are written separately. Only full 9x6 subpixel detections in the calibrated image plane are promoted into intrinsics.",
        ],
    }
    (args.out / "intrinsics_calibration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n_bags", "detections", "calibration_status")}, indent=2))


if __name__ == "__main__":
    main()
