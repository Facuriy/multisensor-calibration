#!/usr/bin/env python3
"""Guided thermal panel/checker detection for the 20260623 calibration session.

This pass deliberately searches only near the expected target location. The
guide comes from RGB/VIS panel boxes projected into thermal coordinates through
the current RGB/VIS/Thermal homographies.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from thermal_checker import detect_sb as detect_thermal_sb
from thermal_checker import preprocess_variants as thermal_preprocess_variants
from thermal_checker import recover_scalar as recover_thermal_scalar


DEFAULT_REFINED = Path(
    "runs/calibration_20260623_refined_multisensor_detection/"
    "refined_multisensor_detection_summary.json"
)
DEFAULT_HOMOGRAPHIES = Path("data/calibration/new_session/20260623/homographies_20260623_to_vis.json")
DEFAULT_PREVIEW_ROOT = Path("runs/calibration_20260623_full_review/per_bag")
DEFAULT_OUT = Path("runs/calibration_20260623_thermal_guided_panel")
THERMAL_SIZE = (640, 512)  # width, height


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_homographies(path: Path) -> dict[str, np.ndarray]:
    data = load_json(path)
    h = data["homographies"]
    h_rgb_vis = np.asarray(h["rgb_to_vis"], dtype=np.float64)
    h_vis = np.eye(3, dtype=np.float64)
    h_th_vis = np.asarray(h["thermal_deg_to_vis"], dtype=np.float64)
    h_vis_th = np.linalg.inv(h_th_vis)
    return {
        "rgb_to_thermal": h_vis_th @ h_rgb_vis,
        "vis_to_thermal": h_vis_th @ h_vis,
        "thermal_to_vis": h_th_vis,
    }


def project_box(box: list[float] | None, h: np.ndarray, shape_hw: tuple[int, int]) -> list[float] | None:
    if box is None:
        return None
    x0, y0, x1, y1 = [float(v) for v in box]
    if x1 <= x0 or y1 <= y0:
        return None
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, h).reshape(-1, 2)
    height, width = shape_hw
    bx0 = max(0.0, float(np.nanmin(dst[:, 0])))
    by0 = max(0.0, float(np.nanmin(dst[:, 1])))
    bx1 = min(float(width - 1), float(np.nanmax(dst[:, 0])))
    by1 = min(float(height - 1), float(np.nanmax(dst[:, 1])))
    if bx1 <= bx0 or by1 <= by0:
        return None
    return [bx0, by0, bx1, by1]


def union_boxes(boxes: list[list[float] | None], shape_hw: tuple[int, int]) -> list[float] | None:
    valid = [b for b in boxes if b is not None]
    if not valid:
        return None
    height, width = shape_hw
    arr = np.asarray(valid, dtype=np.float64)
    x0 = max(0.0, float(np.min(arr[:, 0])))
    y0 = max(0.0, float(np.min(arr[:, 1])))
    x1 = min(float(width - 1), float(np.max(arr[:, 2])))
    y1 = min(float(height - 1), float(np.max(arr[:, 3])))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def expand_box(box: list[float], shape_hw: tuple[int, int], frac: float = 0.25, min_pad: int = 24) -> tuple[int, int, int, int]:
    height, width = shape_hw
    x0, y0, x1, y1 = box
    pad_x = max(min_pad, int((x1 - x0) * frac))
    pad_y = max(min_pad, int((y1 - y0) * frac))
    return (
        max(0, int(math.floor(x0)) - pad_x),
        max(0, int(math.floor(y0)) - pad_y),
        min(width, int(math.ceil(x1)) + pad_x),
        min(height, int(math.ceil(y1)) + pad_y),
    )


def normalize_u8(src: np.ndarray, lo_pct: float = 2, hi_pct: float = 98) -> np.ndarray:
    arr = src.astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(arr[finite], [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((arr - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def local_contrast(gray: np.ndarray, sigma: float = 9.0) -> np.ndarray:
    src = gray.astype(np.float32)
    blur = cv2.GaussianBlur(src, (0, 0), sigma)
    return normalize_u8(src - blur, 1, 99)


def signal_images(img_bgr: np.ndarray, scalar: np.ndarray | None = None) -> list[tuple[str, np.ndarray]]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _b, g, r = cv2.split(img_bgr)
    signals = [
        ("gray", gray),
        ("red", r),
        ("hsv_v", hsv[:, :, 2]),
        ("lab_l", lab[:, :, 0]),
        ("rg_diff", normalize_u8(r.astype(np.float32) - g.astype(np.float32), 1, 99)),
        ("local_gray", local_contrast(gray)),
    ]
    if scalar is not None:
        try:
            for name, sig in thermal_preprocess_variants(scalar):
                signals.append((name, sig))
        except Exception:
            pass
    else:
        try:
            scalar_auto, _info = recover_thermal_scalar(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), cmap="auto")
            for name, sig in thermal_preprocess_variants(scalar_auto):
                signals.append((name, sig))
        except Exception:
            # The detector must still work on non-colormapped or malformed previews.
            pass
    return [(name, normalize_u8(sig, 1, 99)) for name, sig in signals]


def threshold_variants(signal: np.ndarray) -> list[tuple[str, np.ndarray]]:
    blur = cv2.GaussianBlur(signal, (5, 5), 0)
    variants: list[tuple[str, np.ndarray]] = []

    _thr, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu_hot", otsu))
    variants.append(("otsu_cold", cv2.bitwise_not(otsu)))

    for pct in (60, 70, 80):
        hi = int(np.percentile(blur, pct))
        lo = int(np.percentile(blur, 100 - pct))
        variants.append((f"p{pct}_hot", (blur >= hi).astype(np.uint8) * 255))
        variants.append((f"p{100-pct}_cold", (blur <= lo).astype(np.uint8) * 255))

    for block in (41,):
        if min(signal.shape[:2]) > block:
            ad = cv2.adaptiveThreshold(
                blur,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block,
                2,
            )
            variants.append((f"adaptive{block}_hot", ad))
            variants.append((f"adaptive{block}_cold", cv2.bitwise_not(ad)))

    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = normalize_u8(np.abs(gx) + np.abs(gy), 5, 99)
    edge_thr = int(np.percentile(mag, 80))
    variants.append(("edge_p80", (mag >= edge_thr).astype(np.uint8) * 255))
    return variants


def clean_mask(mask: np.ndarray, close_size: int, open_size: int) -> np.ndarray:
    out = mask.astype(np.uint8)
    if close_size > 1:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8), iterations=1)
    if open_size > 1:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8), iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    return out


def box_iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1.0, (bx1 - bx0) * (by1 - by0))
    return float(inter / (area_a + area_b - inter))


def score_component(
    box: list[float],
    area: int,
    guide: list[float],
    roi_area: float,
) -> float:
    x0, y0, x1, y1 = box
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    gw, gh = max(1.0, guide[2] - guide[0]), max(1.0, guide[3] - guide[1])
    bbox_area = bw * bh
    fill = min(1.0, area / max(1.0, bbox_area))
    iou = box_iou(box, guide)
    cc = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5])
    gc = np.array([(guide[0] + guide[2]) * 0.5, (guide[1] + guide[3]) * 0.5])
    center_penalty = np.linalg.norm(cc - gc) / max(1.0, math.hypot(gw, gh))
    size_penalty = abs(math.log(max(0.1, bw / gw))) + abs(math.log(max(0.1, bh / gh)))
    area_frac = min(1.0, area / max(1.0, roi_area))
    aspect = bw / bh
    aspect_penalty = 0.0 if 0.55 <= aspect <= 4.0 else 1.0
    return 4.0 * iou + 1.2 * fill + 0.8 * area_frac - 0.8 * center_penalty - 0.5 * size_penalty - aspect_penalty


def detect_panel(img: np.ndarray, guide: list[float], scalar: np.ndarray | None = None) -> dict[str, Any]:
    shape = img.shape[:2]
    rx0, ry0, rx1, ry1 = expand_box(guide, shape, 0.25, 28)
    roi = img[ry0:ry1, rx0:rx1]
    roi_area = max(1.0, float((rx1 - rx0) * (ry1 - ry0)))
    guide_roi = [guide[0] - rx0, guide[1] - ry0, guide[2] - rx0, guide[3] - ry0]
    best: dict[str, Any] | None = None
    for sig_name, sig_full in signal_images(img, scalar):
        sig = sig_full[ry0:ry1, rx0:rx1]
        for thr_name, raw_mask in threshold_variants(sig):
            for close_size, open_size in ((7, 3), (13, 5)):
                mask = clean_mask(raw_mask, close_size, open_size)
                n, labels, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
                for comp in range(1, n):
                    area = int(stats[comp, cv2.CC_STAT_AREA])
                    bx = int(stats[comp, cv2.CC_STAT_LEFT])
                    by = int(stats[comp, cv2.CC_STAT_TOP])
                    bw = int(stats[comp, cv2.CC_STAT_WIDTH])
                    bh = int(stats[comp, cv2.CC_STAT_HEIGHT])
                    if area < 120 or bw < 20 or bh < 15:
                        continue
                    if area > roi_area * 0.88:
                        continue
                    bbox_area = float(bw * bh)
                    if bbox_area > roi_area * 0.70:
                        continue
                    box_roi = [float(bx), float(by), float(bx + bw), float(by + bh)]
                    gw = max(1.0, guide_roi[2] - guide_roi[0])
                    gh = max(1.0, guide_roi[3] - guide_roi[1])
                    if bw > gw * 1.55 or bh > gh * 1.55:
                        continue
                    if bw < gw * 0.18 or bh < gh * 0.18:
                        continue
                    if box_iou(box_roi, guide_roi) < 0.05:
                        continue
                    score = score_component(box_roi, area, guide_roi, roi_area)
                    if best is None or score > best["score"]:
                        comp_mask = (labels == comp).astype(np.uint8) * 255
                        best = {
                            "score": float(score),
                            "method": f"{sig_name}_{thr_name}_close{close_size}_open{open_size}",
                            "panel_box_xyxy": [
                                float(rx0 + bx),
                                float(ry0 + by),
                                float(rx0 + bx + bw),
                                float(ry0 + by + bh),
                            ],
                            "area_px": area,
                            "roi_xyxy": [rx0, ry0, rx1, ry1],
                            "mask": comp_mask,
                            "signal_name": sig_name,
                            "mask_name": thr_name,
                        }
    if best is None:
        return {
            "detected": False,
            "method": "projected_fallback",
            "panel_box_xyxy": guide,
            "score": 0.0,
            "roi_xyxy": [rx0, ry0, rx1, ry1],
            "reason": "no_threshold_component",
        }
    best["detected"] = True
    best["reason"] = ""
    return best


def checker_variants(gray: np.ndarray) -> list[tuple[str, np.ndarray]]:
    base = normalize_u8(gray, 2, 98)
    variants = [
        ("norm", base),
        ("clahe", cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(base)),
        ("local", local_contrast(base)),
    ]
    out = []
    for name, img in variants:
        out.append((name, img))
        out.append((name + "_inv", 255 - img))
    return out


def try_checker(img: np.ndarray, box: list[float], scalar: np.ndarray | None = None, scalar_info: dict[str, Any] | None = None) -> dict[str, Any]:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = expand_box(box, (h, w), 0.18, 18)
    crop_bgr = img[y0:y1, x0:x1]
    if crop_bgr.size == 0:
        return {"detected": False}

    patterns = [(9, 6), (9, 5), (8, 5)]
    best = None
    if scalar is not None:
        scalar_crop = scalar[y0:y1, x0:x1]
        scalar_variants = thermal_preprocess_variants(scalar_crop)[:3]
        for variant_name, variant in scalar_variants:
            for pattern in ((9, 5), (9, 6), (8, 5)):
                corners, sweep = detect_thermal_sb(
                    variant,
                    pattern,
                    scales=(1.0, 2.0),
                    rotations=(0, 8, -8),
                    try_invert=True,
                )
                if corners is None:
                    continue
                pts = corners.reshape(-1, 2).astype(np.float64)
                pts[:, 0] += x0
                pts[:, 1] += y0
                area = float(cv2.contourArea(cv2.convexHull(pts.astype(np.float32))))
                score = area + 1500.0 * pattern[0] * pattern[1]
                if best is None or score > best["score"]:
                    best = {
                        "detected": True,
                        "pattern_internal_corners": list(pattern),
                        "corners_px": pts.tolist(),
                        "method": f"thermal_scalar_SB_{variant_name}_{sweep}",
                        "score": float(score),
                        "roi_xyxy": [x0, y0, x1, y1],
                        "cmap": (scalar_info or {}).get("cmap"),
                        "cmap_residual": (scalar_info or {}).get("residual"),
                    }
    if best is not None:
        return best

    max_side = max(crop_bgr.shape[:2])
    scale_down = 1.0
    if max_side > 360:
        scale_down = 360.0 / max_side
        crop_bgr = cv2.resize(crop_bgr, None, fx=scale_down, fy=scale_down, interpolation=cv2.INTER_AREA)
    gray_full = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    flags_sb = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    for pattern in patterns:
        for name, variant in checker_variants(gray_full):
            for scale in (1.0,):
                if scale != 1.0:
                    work = cv2.resize(variant, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                else:
                    work = variant
                ok, corners = cv2.findChessboardCornersSB(work, pattern, flags=flags_sb)
                method = "sb"
                if not ok:
                    ok, corners = cv2.findChessboardCorners(work, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
                    method = "classic"
                if not ok or corners is None:
                    continue
                pts = corners.reshape(-1, 2).astype(np.float64) / (scale * scale_down)
                pts[:, 0] += x0
                pts[:, 1] += y0
                area = float(cv2.contourArea(cv2.convexHull(pts.astype(np.float32))))
                score = area + 1000.0 * pattern[0] * pattern[1]
                if best is None or score > best["score"]:
                    best = {
                        "detected": True,
                        "pattern_internal_corners": list(pattern),
                        "corners_px": pts.tolist(),
                        "method": f"{method}_{name}_s{scale:g}",
                        "score": float(score),
                        "roi_xyxy": [x0, y0, x1, y1],
                    }
    return best if best is not None else {"detected": False}


def draw_review(img: np.ndarray, result: dict[str, Any], out_path: Path) -> None:
    canvas = img.copy()
    guide = result.get("guide_box_xyxy")
    if guide:
        x0, y0, x1, y1 = [int(round(v)) for v in guide]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 255, 255), 2, cv2.LINE_AA)
    box = result.get("panel_box_xyxy")
    if box:
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        color = (0, 255, 255) if result.get("panel_detected") else (0, 128, 255)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 3, cv2.LINE_AA)
    checker = result.get("checker")
    if checker and checker.get("detected"):
        pts = np.asarray(checker["corners_px"], dtype=np.int32)
        for p in pts:
            cv2.circle(canvas, tuple(p), 2, (0, 255, 0), -1, cv2.LINE_AA)
        hull = cv2.convexHull(pts.astype(np.float32)).astype(np.int32)
        cv2.polylines(canvas, [hull], True, (0, 255, 0), 2, cv2.LINE_AA)
    text = f"{result['label_norm']} | panel={result.get('panel_detected')} | checker={checker.get('detected') if checker else False}"
    cv2.putText(canvas, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def draw_binary_review(img: np.ndarray, result: dict[str, Any], panel: dict[str, Any], out_path: Path) -> None:
    roi = panel.get("roi_xyxy")
    mask = panel.get("mask")
    if roi is None or mask is None:
        return
    x0, y0, x1, y1 = [int(v) for v in roi]
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return
    mask_bgr = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    overlay = crop.copy()
    color = np.zeros_like(crop)
    color[:, :, 1] = 255
    alpha_mask = (mask > 0)[:, :, None].astype(np.float32)
    overlay = np.clip(overlay * (1.0 - 0.45 * alpha_mask) + color * (0.45 * alpha_mask), 0, 255).astype(np.uint8)
    parts = []
    for label, part in (("thermal ROI", crop), ("binary mask", mask_bgr), ("mask overlay", overlay)):
        tile = fit_tile(part, (320, 220), label)
        parts.append(tile)
    canvas = np.hstack(parts)
    text = f"{result['label_norm']} | {result.get('panel_method')}"
    cv2.putText(canvas, text[:95], (10, canvas.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def fit_tile(img: np.ndarray, size: tuple[int, int], label: str) -> np.ndarray:
    width, height = size
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    h, w = img.shape[:2]
    scale = min(width / max(w, 1), (height - 28) / max(h, 1))
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    x = (width - small.shape[1]) // 2
    y = 28 + (height - 28 - small.shape[0]) // 2
    canvas[y:y + small.shape[0], x:x + small.shape[1]] = small
    cv2.putText(canvas, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 2, cv2.LINE_AA)
    return canvas


def make_contact_pages(
    results: list[dict[str, Any]],
    out_dir: Path,
    key: str = "review_path",
    prefix: str = "thermal_guided_review_page",
    cols: int = 4,
    rows: int = 4,
) -> list[str]:
    paths = [Path(r[key]) for r in results if r.get(key)]
    pages = []
    per_page = cols * rows
    for page_idx in range(0, len(paths), per_page):
        imgs = []
        for path in paths[page_idx:page_idx + per_page]:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            tile = cv2.resize(img, (320, 256), interpolation=cv2.INTER_AREA)
            imgs.append(tile)
        if not imgs:
            continue
        while len(imgs) < per_page:
            imgs.append(np.full((256, 320, 3), 245, dtype=np.uint8))
        grid_rows = [np.hstack(imgs[i:i + cols]) for i in range(0, per_page, cols)]
        page = np.vstack(grid_rows)
        out_path = out_dir / f"{prefix}_{len(pages)+1:02d}.jpg"
        cv2.imwrite(str(out_path), page)
        pages.append(str(out_path))
    return pages


def process_row(
    row: dict[str, Any],
    homographies: dict[str, np.ndarray],
    preview_root: Path,
    out_dir: Path,
    use_scalar: bool = False,
) -> dict[str, Any]:
    label = str(row.get("label_norm") or row.get("label_raw") or row.get("bag"))
    bag = str(row.get("bag") or "")
    folder = preview_root / Path(bag).stem
    thermal_path = folder / "thermal_c.jpg"
    img = cv2.imread(str(thermal_path), cv2.IMREAD_COLOR)
    if img is None:
        return {"label_norm": label, "bag": bag, "panel_detected": False, "reason": "missing_thermal_preview"}
    shape = img.shape[:2]
    dets = row.get("detections", {})
    rgb_box = (dets.get("rgb") or {}).get("panel_box_xyxy")
    vis_box = (dets.get("vis") or {}).get("panel_box_xyxy")
    thermal_old_box = (dets.get("thermal_c") or {}).get("panel_box_xyxy")
    boxes = [
        project_box(rgb_box, homographies["rgb_to_thermal"], shape),
        project_box(vis_box, homographies["vis_to_thermal"], shape),
    ]
    # Keep prior thermal box only when it is not essentially the whole frame.
    if thermal_old_box is not None:
        tx0, ty0, tx1, ty1 = [float(v) for v in thermal_old_box]
        area_frac = ((tx1 - tx0) * (ty1 - ty0)) / max(1.0, shape[0] * shape[1])
        if area_frac < 0.75:
            boxes.append([tx0, ty0, tx1, ty1])
    guide = union_boxes(boxes, shape)
    if guide is None:
        guide = [0.0, 0.0, float(shape[1] - 1), float(shape[0] - 1)]

    scalar = None
    scalar_info: dict[str, Any] = {}
    if use_scalar:
        try:
            scalar, scalar_info = recover_thermal_scalar(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap="auto")
        except Exception as exc:
            scalar_info = {"error": str(exc)}

    panel = detect_panel(img, guide, scalar)
    checker_box = guide
    panel_box = panel.get("panel_box_xyxy")
    if panel_box is not None:
        px0, py0, px1, py1 = [float(v) for v in panel_box]
        gx0, gy0, gx1, gy1 = [float(v) for v in guide]
        panel_area = max(1.0, (px1 - px0) * (py1 - py0))
        guide_area = max(1.0, (gx1 - gx0) * (gy1 - gy0))
        if panel_area <= guide_area * 1.15:
            checker_box = panel_box
    checker = try_checker(img, checker_box, scalar, scalar_info)
    result = {
        "label_norm": label,
        "bag": bag,
        "thermal_path": str(thermal_path),
        "guide_box_xyxy": [float(v) for v in guide],
        "panel_detected": bool(panel.get("detected")),
        "panel_box_xyxy": panel.get("panel_box_xyxy"),
        "panel_method": panel.get("method"),
        "panel_score": panel.get("score"),
        "panel_area_px": panel.get("area_px"),
        "roi_xyxy": panel.get("roi_xyxy"),
        "checker": checker,
        "thermal_scalar": scalar_info,
        "reason": panel.get("reason", ""),
    }
    review_path = out_dir / "bags" / Path(bag).stem / "thermal_guided_detection.jpg"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    draw_review(img, result, review_path)
    result["review_path"] = str(review_path)
    binary_path = out_dir / "bags" / Path(bag).stem / "thermal_guided_binary.jpg"
    draw_binary_review(img, result, panel, binary_path)
    if binary_path.exists():
        result["binary_review_path"] = str(binary_path)
    return result


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "label_norm",
        "bag",
        "panel_detected",
        "panel_method",
        "panel_score",
        "panel_area_px",
        "checker_detected",
        "checker_pattern",
        "checker_method",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            chk = r.get("checker") or {}
            writer.writerow(
                {
                    "label_norm": r.get("label_norm"),
                    "bag": r.get("bag"),
                    "panel_detected": r.get("panel_detected"),
                    "panel_method": r.get("panel_method"),
                    "panel_score": r.get("panel_score"),
                    "panel_area_px": r.get("panel_area_px"),
                    "checker_detected": chk.get("detected", False),
                    "checker_pattern": "x".join(map(str, chk.get("pattern_internal_corners", []))) if chk.get("detected") else "",
                    "checker_method": chk.get("method", ""),
                    "reason": r.get("reason", ""),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refined-summary", type=Path, default=DEFAULT_REFINED)
    parser.add_argument("--homographies", type=Path, default=DEFAULT_HOMOGRAPHIES)
    parser.add_argument("--preview-root", type=Path, default=DEFAULT_PREVIEW_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--use-scalar",
        action="store_true",
        help="Also invert the thermal colormap and try scalar-based variants. Slower.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    homographies = load_homographies(args.homographies)
    data = load_json(args.refined_summary)
    results = [
        process_row(row, homographies, args.preview_root, args.out_dir, use_scalar=args.use_scalar)
        for row in data.get("results", [])
    ]
    pages = make_contact_pages(results, args.out_dir)
    binary_pages = make_contact_pages(
        results,
        args.out_dir,
        key="binary_review_path",
        prefix="thermal_guided_binary_page",
        cols=1,
        rows=5,
    )
    summary = {
        "source": str(args.refined_summary),
        "homographies": str(args.homographies),
        "preview_root": str(args.preview_root),
        "n_bags": len(results),
        "panel_detected": sum(1 for r in results if r.get("panel_detected")),
        "checker_detected": sum(1 for r in results if (r.get("checker") or {}).get("detected")),
        "review_pages": pages,
        "binary_review_pages": binary_pages,
        "results": results,
    }
    with (args.out_dir / "thermal_guided_detection_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_csv(args.out_dir / "thermal_guided_detection_table.csv", results)

    print(f"Thermal guided panels: {summary['panel_detected']} / {summary['n_bags']}")
    print(f"Thermal guided checkers: {summary['checker_detected']} / {summary['n_bags']}")
    print(f"Wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
