#!/usr/bin/env python3
"""VIS/NIR partial-grid recovery campaign for 2026-06-23 Photonfocus bags.

The regular chessboard detector fails when the checkerboard is cropped, even
when the visible squares are crisp.  This campaign scans every frame, keeps
high-quality candidate frames, then tries:
  * Photonfocus band summaries and individual bands.
  * z-scored bands, normalized band differences, PCA-like components.
  * CLAHE, standardization, local detail, unsharp, bilateral filtering.
  * OpenCV full/partial checker patterns.
  * Model-based partial square-grid templates.

Outputs are strong/partial model candidates for VIS/NIR, with review sheets.
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

from src.extraction.extract_all_bag_images import decode_ros_image  # noqa: E402


MANIFEST = Path("data/calibration/new_session/20260623/bag_manifest_20260623.csv")
BAG_CACHE = Path("runs/calibration_20260623_raw_bag_cache")
BASE_SUMMARY = Path("runs/calibration_20260623_refined_multisensor_detection/refined_multisensor_detection_summary.json")
OUT = Path("runs/calibration_20260623_photonfocus_partial_grid_campaign")

TOPICS = {
    "vis": "/ssf/photonfocus_camera_vis_node/image_raw",
    "nir": "/ssf/photonfocus_camera_nir_node/image_raw",
}

EXPECTED_MOSAIC = {"vis": 4, "nir": 5}


@dataclass
class FrameCandidate:
    sensor: str
    frame_index: int
    stamp_ns: int
    raw: np.ndarray
    quality: float


@dataclass
class Template:
    squares: tuple[int, int]
    width: int
    height: int
    image: np.ndarray


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_manifest(path: Path, include_nondefault: bool) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    out = []
    for row in rows:
        if row.get("label_norm") == "test":
            continue
        if not include_nondefault and row.get("include_default") != "yes":
            continue
        out.append(row)
    return out


def base_by_label(path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(path)
    return {str(r.get("label_norm")): r for r in data.get("results", [])}


def guide_box(label: str, sensor: str, base: dict[str, dict[str, Any]], shape_hw: tuple[int, int]) -> list[float] | None:
    row = base.get(label, {})
    boxes = row.get("projected_boxes_from_rgb", {})
    box = boxes.get(sensor)
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


def expand_box(box: list[float] | None, shape_hw: tuple[int, int], frac: float) -> tuple[int, int, int, int]:
    h, w = shape_hw
    if box is None:
        return (0, 0, w, h)
    x0, y0, x1, y1 = box
    px = max(12, int((x1 - x0) * frac))
    py = max(12, int((y1 - y0) * frac))
    return (
        max(0, int(math.floor(x0)) - px),
        max(0, int(math.floor(y0)) - py),
        min(w, int(math.ceil(x1)) + px),
        min(h, int(math.ceil(y1)) + py),
    )


def normalize_u8(img: np.ndarray, lo: float = 1, hi: float = 99) -> np.ndarray:
    src = img.astype(np.float32)
    finite = np.isfinite(src)
    if not finite.any():
        return np.zeros(src.shape[:2], dtype=np.uint8)
    a, b = np.percentile(src[finite], (lo, hi))
    if b <= a:
        b = a + 1.0
    return np.clip((src - a) * 255.0 / (b - a), 0, 255).astype(np.uint8)


def zscore_u8(img: np.ndarray) -> np.ndarray:
    src = img.astype(np.float32)
    mu = float(np.mean(src))
    sd = float(np.std(src)) + 1e-6
    return normalize_u8((src - mu) / sd, 0.5, 99.5)


def local_detail(img: np.ndarray, sigma: float = 7.0) -> np.ndarray:
    src = img.astype(np.float32)
    bg = cv2.GaussianBlur(src, (0, 0), sigma)
    return normalize_u8(src - bg, 1, 99)


def unsharp(u8: np.ndarray) -> np.ndarray:
    return cv2.addWeighted(u8, 1.8, cv2.GaussianBlur(u8, (0, 0), 1.5), -0.8, 0)


def extract_bands(raw: np.ndarray, pattern: int) -> list[np.ndarray]:
    h = (raw.shape[0] // pattern) * pattern
    w = (raw.shape[1] // pattern) * pattern
    work = raw[:h, :w].astype(np.float32)
    return [work[ro::pattern, co::pattern] for ro in range(pattern) for co in range(pattern)]


def band_bases(raw: np.ndarray, sensor: str, roi: tuple[int, int, int, int] | None = None) -> list[tuple[str, np.ndarray]]:
    pattern = EXPECTED_MOSAIC[sensor]
    bands = extract_bands(raw, pattern)
    stack = np.stack(bands, axis=-1)
    bases: list[tuple[str, np.ndarray]] = [
        (f"mean{pattern}", np.mean(stack, axis=-1)),
        (f"median{pattern}", np.median(stack, axis=-1)),
        (f"max{pattern}", np.max(stack, axis=-1)),
        (f"min{pattern}", np.min(stack, axis=-1)),
        (f"range{pattern}", np.max(stack, axis=-1) - np.min(stack, axis=-1)),
        (f"std{pattern}", np.std(stack, axis=-1)),
    ]

    # Rank individual bands by contrast in the guided ROI.
    if roi is None:
        roi = (0, 0, stack.shape[1], stack.shape[0])
    x0, y0, x1, y1 = roi
    band_scores = []
    for i, b in enumerate(bands):
        crop = b[y0:y1, x0:x1]
        band_scores.append((float(np.std(crop)) + 0.02 * float(np.ptp(crop)), i))
    for _score, i in sorted(band_scores, reverse=True)[: min(10, len(bands))]:
        bases.append((f"band{pattern}_{i:02d}", bands[i]))
        bases.append((f"zband{pattern}_{i:02d}", (bands[i] - np.mean(stack, axis=-1)) / (np.std(stack, axis=-1) + 1e-3)))

    # Normalized differences between high-contrast bands.
    top = [i for _score, i in sorted(band_scores, reverse=True)[:6]]
    for a_i, a in enumerate(top[:4]):
        for b in top[a_i + 1 : a_i + 4]:
            num = bands[a] - bands[b]
            den = bands[a] + bands[b] + 1.0
            bases.append((f"ndi{pattern}_{a:02d}_{b:02d}", num / den))

    # PCA-like components on standardized bands.
    flat = stack.reshape(-1, stack.shape[-1]).astype(np.float32)
    flat = flat - flat.mean(axis=0, keepdims=True)
    flat = flat / (flat.std(axis=0, keepdims=True) + 1e-6)
    try:
        _mean, eigvec = cv2.PCACompute(flat, mean=None, maxComponents=3)
        comps = flat @ eigvec.T
        for ci in range(comps.shape[1]):
            bases.append((f"pca{pattern}_{ci}", comps[:, ci].reshape(stack.shape[:2])))
    except cv2.error:
        pass
    return bases


def variants(raw: np.ndarray, sensor: str, roi: tuple[int, int, int, int] | None) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    for name, base in band_bases(raw, sensor, roi):
        u = normalize_u8(base, 1, 99)
        out.append((f"{name}_norm", u))
        out.append((f"{name}_clahe", cv2.createCLAHE(2.2, (6, 6)).apply(u)))
        out.append((f"{name}_detail", local_detail(u, 7)))
        out.append((f"{name}_unsharp", unsharp(u)))
        out.append((f"{name}_bilateral", cv2.bilateralFilter(u, 5, 25, 5)))
    # Deduplicate names.
    seen = set()
    dedup = []
    for name, u in out:
        if name not in seen:
            dedup.append((name, u))
            seen.add(name)
    return dedup


def quality_score(raw: np.ndarray, sensor: str, base: dict[str, dict[str, Any]], label: str) -> float:
    mean = normalize_u8(np.mean(np.stack(extract_bands(raw, EXPECTED_MOSAIC[sensor]), axis=-1), axis=-1), 1, 99)
    guide = guide_box(label, sensor, base, mean.shape[:2])
    x0, y0, x1, y1 = expand_box(guide, mean.shape[:2], 0.20)
    crop = mean[y0:y1, x0:x1]
    gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.std(crop)) + 0.35 * float(np.mean(np.abs(gx) + np.abs(gy)))


def collect_frames(
    bag_path: Path,
    label: str,
    sensors: list[str],
    base: dict[str, dict[str, Any]],
    top_frames: int,
) -> dict[str, list[FrameCandidate]]:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    top = {s: [] for s in sensors}
    with Reader(bag_path) as reader:
        available = {c.topic: c for c in reader.connections}
        conns = []
        topic_to_sensor = {}
        for s in sensors:
            topic = TOPICS[s]
            if topic in available:
                conns.append(available[topic])
                topic_to_sensor[topic] = s
        counts = {s: 0 for s in sensors}
        for conn, ts, raw in reader.messages(connections=conns):
            sensor = topic_to_sensor[conn.topic]
            idx = counts[sensor]
            counts[sensor] += 1
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            img = decode_ros_image(msg)
            if img is None or img.ndim != 2:
                continue
            q = quality_score(img, sensor, base, label)
            cand = FrameCandidate(sensor, idx, int(ts), img, q)
            bucket = top[sensor]
            bucket.append(cand)
            bucket.sort(key=lambda c: c.quality, reverse=True)
            del bucket[top_frames:]
    return top


def make_template(nx: int, ny: int, width: int, height: int) -> np.ndarray:
    small = np.zeros((ny, nx), dtype=np.uint8)
    for y in range(ny):
        for x in range(nx):
            small[y, x] = 255 if ((x + y) % 2) else 0
    return cv2.GaussianBlur(cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST), (3, 3), 0)


def build_templates(sensor: str) -> list[Template]:
    templates: list[Template] = []
    # Visible square grids. Full checkerboard is 10x7 squares, but many frames
    # show only 10x3..10x5 or 8/9 columns because the target is clipped.
    for nx in range(6, 11):
        for ny in range(3, 8):
            if nx < 6 or ny < 3:
                continue
            width_range = range(90, 520, 10) if sensor == "vis" else range(80, 410, 10)
            for width in width_range:
                nominal_h = width * ny / nx
                for hs in (0.72, 0.86, 1.0, 1.15):
                    height = int(round(nominal_h * hs))
                    if height < 35:
                        continue
                    templates.append(Template((nx, ny), int(width), int(height), make_template(nx, ny, int(width), int(height))))
    return templates


def score_cells(u8: np.ndarray, box: tuple[int, int, int, int], squares: tuple[int, int]) -> float:
    nx, ny = squares
    x0, y0, x1, y1 = box
    crop = u8[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0
    h, w = crop.shape[:2]
    means = np.zeros((ny, nx), dtype=np.float32)
    for y in range(ny):
        for x in range(nx):
            cx0 = int(round(x * w / nx + 0.18 * w / nx))
            cx1 = int(round((x + 1) * w / nx - 0.18 * w / nx))
            cy0 = int(round(y * h / ny + 0.18 * h / ny))
            cy1 = int(round((y + 1) * h / ny - 0.18 * h / ny))
            means[y, x] = float(np.mean(crop[cy0:max(cy0 + 1, cy1), cx0:max(cx0 + 1, cx1)]))
    parity = np.fromfunction(lambda yy, xx: ((xx + yy) % 2) * 2 - 1, means.shape).astype(np.float32)
    centered = means - float(np.mean(means))
    denom = float(np.sqrt(np.sum(centered * centered) * np.sum(parity * parity))) + 1e-6
    corr = abs(float(np.sum(centered * parity) / denom))
    diff = abs(float(np.mean(means[parity > 0]) - np.mean(means[parity < 0])))
    return 100.0 * corr + 0.25 * diff


def template_search(u8: np.ndarray, roi: tuple[int, int, int, int], templates: list[Template]) -> dict[str, Any] | None:
    rx0, ry0, rx1, ry1 = roi
    crop = cv2.GaussianBlur(u8[ry0:ry1, rx0:rx1], (3, 3), 0)
    h, w = crop.shape[:2]
    best: dict[str, Any] | None = None
    for templ in templates:
        if templ.width >= w or templ.height >= h:
            continue
        res = cv2.matchTemplate(crop, templ.image, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if abs(min_val) > abs(max_val):
            loc = min_loc
            corr = -float(min_val)
            polarity = "inv"
        else:
            loc = max_loc
            corr = float(max_val)
            polarity = "pos"
        x0 = rx0 + int(loc[0])
        y0 = ry0 + int(loc[1])
        box = (x0, y0, x0 + templ.width, y0 + templ.height)
        cell = score_cells(u8, box, templ.squares)
        # Prefer larger grids when scores are similar.
        size_bonus = 1.2 * (templ.squares[0] - 1) * (templ.squares[1] - 1)
        score = 100.0 * corr + cell + size_bonus
        if best is None or score > float(best["score"]):
            best = {
                "detected": True,
                "method": "partial_template_grid",
                "visible_grid_squares": list(templ.squares),
                "pattern_internal_corners": [templ.squares[0] - 1, templ.squares[1] - 1],
                "box_xyxy": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                "corr": corr,
                "cell_score": cell,
                "score": score,
                "polarity": polarity,
            }
    return best


def detect_opencv(gray: np.ndarray, roi: tuple[int, int, int, int]) -> dict[str, Any] | None:
    patterns = [(9, 6), (9, 5), (8, 5), (9, 4), (8, 4), (7, 4), (9, 3), (8, 3), (7, 3), (6, 3)]
    rx0, ry0, rx1, ry1 = roi
    crop = gray[ry0:ry1, rx0:rx1]
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS
    sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_EXHAUSTIVE
    best = None
    for scale in (1.0, 1.4, 2.0):
        work = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1.0 else crop
        for inv, src in ((False, work), (True, 255 - work)):
            for pattern in patterns:
                ok, corners = cv2.findChessboardCorners(src, pattern, flags=flags)
                alg = "classic"
                if not ok:
                    ok, corners = cv2.findChessboardCornersSB(src, pattern, flags=sb_flags)
                    alg = "sb"
                if not ok or corners is None:
                    continue
                pts = corners.reshape(-1, 2).astype(np.float32) / scale
                pts[:, 0] += rx0
                pts[:, 1] += ry0
                area = float(cv2.contourArea(cv2.convexHull(pts.astype(np.float32))))
                score = area + 2000.0 * pattern[0] * pattern[1]
                rec = {
                    "detected": True,
                    "method": f"opencv_{alg}_p{pattern[0]}x{pattern[1]}_s{scale:g}_{'inv' if inv else 'norm'}",
                    "visible_grid_squares": [pattern[0] + 1, pattern[1] + 1],
                    "pattern_internal_corners": list(pattern),
                    "corners_px": pts.tolist(),
                    "score": score,
                    "corr": None,
                    "cell_score": None,
                }
                if best is None or score > float(best["score"]):
                    best = rec
    return best


def corners_from_template(det: dict[str, Any]) -> np.ndarray:
    nx, ny = det["visible_grid_squares"]
    x0, y0, x1, y1 = det["box_xyxy"]
    xs = np.linspace(x0, x1, nx + 1, dtype=np.float32)[1:-1]
    ys = np.linspace(y0, y1, ny + 1, dtype=np.float32)[1:-1]
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)


def confidence(sensor: str, det: dict[str, Any]) -> str:
    if det["method"].startswith("opencv"):
        cols, rows = det["pattern_internal_corners"]
        return "subpixel_strong" if cols >= 7 and rows >= 3 else "subpixel_partial"
    corr = float(det.get("corr") or 0.0)
    score = float(det.get("score") or 0.0)
    nx, ny = det["visible_grid_squares"]
    if corr >= 0.58 and score >= 150 and nx >= 8 and ny >= 4:
        return "model_strong"
    if corr >= 0.42 and score >= 118 and nx >= 7 and ny >= 3:
        return "model_partial"
    return "weak_reject"


def draw_review(u8: np.ndarray, det: dict[str, Any], path: Path, title: str) -> None:
    color = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    if "corners_px" in det and det["corners_px"] is not None:
        pts = np.asarray(det["corners_px"], dtype=np.int32)
        for i, p in enumerate(pts):
            cv2.circle(color, tuple(p), 3, (0, 255, 0) if i else (0, 0, 255), -1, cv2.LINE_AA)
        hull = cv2.convexHull(pts.astype(np.float32)).astype(np.int32)
        cv2.polylines(color, [hull], True, (0, 220, 0), 2)
    else:
        x0, y0, x1, y1 = det["box_xyxy"]
        nx, ny = det["visible_grid_squares"]
        cv2.rectangle(color, (x0, y0), (x1, y1), (0, 210, 255), 2)
        for x in np.linspace(x0, x1, nx + 1):
            cv2.line(color, (int(round(x)), y0), (int(round(x)), y1), (0, 120, 255), 1)
        for y in np.linspace(y0, y1, ny + 1):
            cv2.line(color, (x0, int(round(y))), (x1, int(round(y))), (0, 120, 255), 1)
    cv2.putText(color, title[:100], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
    h, w = color.shape[:2]
    scale = min(600 / max(w, 1), 420 / max(h, 1), 3.0)
    view = cv2.resize(color, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), view)


def process_sensor(
    label: str,
    sensor: str,
    frames: list[FrameCandidate],
    base: dict[str, dict[str, Any]],
    templates: list[Template],
    out_dir: Path,
    max_variants: int,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    attempts = []
    for frame in frames:
        preview_shape = np.mean(np.stack(extract_bands(frame.raw, EXPECTED_MOSAIC[sensor]), axis=-1), axis=-1).shape[:2]
        gbox = guide_box(label, sensor, base, preview_shape)
        roi = expand_box(gbox, preview_shape, 0.40)
        vars_ = variants(frame.raw, sensor, roi)
        for vi, (variant_name, u8) in enumerate(vars_):
            if vi >= max_variants:
                break
            roi = expand_box(gbox, u8.shape[:2], 0.40)
            det = detect_opencv(u8, roi)
            if det is None:
                det = template_search(u8, roi, templates)
            if det is None:
                continue
            conf = confidence(sensor, det)
            if conf == "weak_reject":
                attempts.append({"frame_index": frame.frame_index, "variant": variant_name, "confidence": conf, "score": det.get("score"), "corr": det.get("corr")})
                continue
            if "corners_px" not in det or det["corners_px"] is None:
                det["corners_px"] = corners_from_template(det).tolist()
            score = float(det.get("score") or 0.0)
            conf_bonus = {"subpixel_strong": 1e6, "subpixel_partial": 8e5, "model_strong": 5e5, "model_partial": 2e5}.get(conf, 0)
            combined = conf_bonus + score + 0.1 * frame.quality
            rec = {
                "checker_detected": True,
                "sensor": sensor,
                "label_norm": label,
                "frame_index": frame.frame_index,
                "stamp_ns": frame.stamp_ns,
                "variant": variant_name,
                "confidence": conf,
                "method": f"{variant_name}_{det['method']}",
                "score": score,
                "combined_score": combined,
                "corr": det.get("corr"),
                "cell_score": det.get("cell_score"),
                "visible_grid_squares": det.get("visible_grid_squares"),
                "pattern_internal_corners": det.get("pattern_internal_corners"),
                "corners_px": det["corners_px"],
                "roi_xyxy": list(roi),
                "warning": "partial/model grid candidate; visually review before calibration use",
            }
            attempts.append({k: v for k, v in rec.items() if k != "corners_px"})
            if best is None or combined > float(best["combined_score"]):
                review = out_dir / "bags" / label / f"{sensor}_partial_grid_best.jpg"
                draw_review(u8, rec, review, f"{label} {sensor} f{frame.frame_index} {conf} {variant_name}")
                rec["review_path"] = str(review)
                best = rec
            if conf == "subpixel_strong":
                break
    return {
        "checker_detected": best is not None,
        "best": best,
        "n_frames_scored": len(frames),
        "attempts_top": sorted(attempts, key=lambda r: float(r.get("combined_score", r.get("score") or 0)), reverse=True)[:20],
    }


def process_bag(
    row: dict[str, str],
    bag_path: Path,
    sensors: list[str],
    base: dict[str, dict[str, Any]],
    templates: dict[str, list[Template]],
    out_dir: Path,
    top_frames: int,
    max_variants: int,
) -> dict[str, Any]:
    label = row["label_norm"]
    frames = collect_frames(bag_path, label, sensors, base, top_frames)
    detections = {}
    for sensor in sensors:
        detections[sensor] = process_sensor(label, sensor, frames.get(sensor, []), base, templates[sensor], out_dir, max_variants)
    return {**row, "bag_path_local": str(bag_path), "processed": True, "detections": detections}


def make_pages(results: list[dict[str, Any]], sensors: list[str], out_dir: Path) -> None:
    strips = []
    for row in results:
        cells = []
        for sensor in sensors:
            best = (row.get("detections", {}).get(sensor, {}) or {}).get("best") or {}
            img = cv2.imread(best.get("review_path", "")) if best.get("review_path") else None
            if img is None:
                img = np.full((210, 360, 3), 245, dtype=np.uint8)
                cv2.putText(img, f"{sensor}: none", (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
            img = cv2.resize(img, (360, 210), interpolation=cv2.INTER_AREA)
            cells.append(img)
        strip = np.hstack(cells)
        canvas = np.full((240, strip.shape[1], 3), 250, dtype=np.uint8)
        canvas[30:, :] = strip
        status = " ".join(f"{s}:{((row.get('detections',{}).get(s,{}) or {}).get('best') or {}).get('confidence','-')}" for s in sensors)
        cv2.putText(canvas, f"{row.get('label_norm')} | {status}"[:130], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 2)
        strips.append(canvas)
    for i, start in enumerate(range(0, len(strips), 4), 1):
        batch = strips[start:start+4]
        while batch and len(batch) < 4:
            batch.append(np.full_like(batch[0], 250))
        if batch:
            cv2.imwrite(str(out_dir / f"photonfocus_partial_grid_page_{i:02d}.jpg"), np.vstack(batch))


def write_outputs(results: list[dict[str, Any]], sensors: list[str], out_dir: Path, args: argparse.Namespace) -> None:
    counts = {}
    for sensor in sensors:
        counts[sensor] = {}
        for conf in ("subpixel_strong", "subpixel_partial", "model_strong", "model_partial"):
            counts[sensor][conf] = sum(1 for r in results if (((r.get("detections", {}).get(sensor, {}) or {}).get("best") or {}).get("confidence") == conf))
    summary = {
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "counts": counts,
        "n_bags": len(results),
        "results": results,
    }
    (out_dir / "photonfocus_partial_grid_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "photonfocus_partial_grid_table.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["label_norm", "bag"]
        for sensor in sensors:
            fields += [f"{sensor}_confidence", f"{sensor}_frame", f"{sensor}_pattern", f"{sensor}_variant", f"{sensor}_score", f"{sensor}_corr", f"{sensor}_review"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            rec = {"label_norm": row.get("label_norm"), "bag": row.get("bag")}
            for sensor in sensors:
                best = ((row.get("detections", {}).get(sensor, {}) or {}).get("best") or {})
                rec[f"{sensor}_confidence"] = best.get("confidence", "")
                rec[f"{sensor}_frame"] = best.get("frame_index", "")
                rec[f"{sensor}_pattern"] = "x".join(map(str, best.get("pattern_internal_corners", [])))
                rec[f"{sensor}_variant"] = best.get("variant", "")
                rec[f"{sensor}_score"] = best.get("score", "")
                rec[f"{sensor}_corr"] = best.get("corr", "")
                rec[f"{sensor}_review"] = best.get("review_path", "")
            writer.writerow(rec)
    make_pages(results, sensors, out_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--bag-cache", type=Path, default=BAG_CACHE)
    parser.add_argument("--base-summary", type=Path, default=BASE_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--sensors", nargs="+", choices=sorted(TOPICS), default=["vis", "nir"])
    parser.add_argument("--include-nondefault", action="store_true")
    parser.add_argument("--only-label", default="")
    parser.add_argument("--top-frames", type=int, default=8)
    parser.add_argument("--max-variants", type=int, default=40)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest, args.include_nondefault)
    if args.only_label:
        rows = [r for r in rows if r.get("label_norm") == args.only_label or r.get("bag") == args.only_label]
    base = base_by_label(args.base_summary)
    templates = {s: build_templates(s) for s in args.sensors}

    results = []
    for i, row in enumerate(rows, 1):
        label = row["label_norm"]
        checkpoint = args.out_dir / "checkpoints" / f"{label}.json"
        if args.resume and checkpoint.exists():
            print(f"[{i}/{len(rows)}] {label}: checkpoint")
            results.append(load_json(checkpoint))
            continue
        bag_path = args.bag_cache / row["bag"]
        if not bag_path.exists():
            rec = {**row, "processed": False, "reason": f"missing_local_bag:{bag_path}", "detections": {}}
            results.append(rec)
            continue
        print(f"[{i}/{len(rows)}] {label}: VIS/NIR partial search")
        rec = process_bag(row, bag_path, args.sensors, base, templates, args.out_dir, args.top_frames, args.max_variants)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        results.append(rec)
        write_outputs(results, args.sensors, args.out_dir, args)
    write_outputs(results, args.sensors, args.out_dir, args)
    counts = load_json(args.out_dir / "photonfocus_partial_grid_summary.json").get("counts", {})
    print(json.dumps({"n_bags": len(results), "counts": counts, "out": str(args.out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
