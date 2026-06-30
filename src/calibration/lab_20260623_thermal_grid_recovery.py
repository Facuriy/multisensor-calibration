#!/usr/bin/env python3
"""Thermal checkerboard recovery lab for hard 2026-06-23 frames.

This is deliberately a small, visual workbench.  It tries standard detector
preprocessing, binary/morphology/watershed diagnostics, and a physical checker
template fit that can recover a full grid when humans can see the squares but
OpenCV's chessboard detector refuses the frame.
"""

from __future__ import annotations

import csv
import json
import math
import re
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

from src.calibration.thermal_checker import recover_scalar  # noqa: E402
from src.calibration.thermal_checker_raw import preprocess_raw_variants, robust01  # noqa: E402
from src.extraction.extract_all_bag_images import decode_ros_image  # noqa: E402


MANIFEST = Path("data/calibration/new_session/20260623/bag_manifest_20260623.csv")
BAG_CACHE = Path("runs/calibration_20260623_raw_bag_cache")
FULL_REVIEW = Path("runs/calibration_20260623_full_review/per_bag")
OUT = Path("runs/calibration_20260623_thermal_grid_recovery_lab")

TOPICS = {
    "thermal_raw": "/ssf/thermalgrabber_ros/image_mono16",
    "thermal_c": "/ssf/thermalgrabber_ros/image_deg_celsius",
}

CASES = {
    "calib_p04_low_top": {
        "thermal_raw": [30, 34, 35, 36],
        "thermal_c": [34, 35],
    },
    "calib_p08_low_bottomleft": {
        "thermal_raw": [17, 18, 19, 34],
        "thermal_c": [17, 18],
    },
    "calib_p12_low_tilt": {
        "thermal_raw": [34, 45, 52, 54],
        "thermal_c": [34, 52, 54],
    },
}

GRID_MODELS = [(10, 7), (10, 6), (9, 6)]


@dataclass
class SourceImage:
    label: str
    bag: str
    source: str
    frame_index: int | None
    image: np.ndarray


def read_manifest() -> dict[str, dict[str, str]]:
    rows = csv.DictReader(MANIFEST.open(encoding="utf-8"))
    return {row["label_norm"]: row for row in rows}


def read_frame(bag_path: Path, sensor: str, target_index: int) -> np.ndarray | None:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    topic = TOPICS[sensor]
    with Reader(bag_path) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            return None
        idx = 0
        for conn, _ts, raw in reader.messages(connections=conns):
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            if idx == target_index:
                return decode_ros_image(msg)
            idx += 1
    return None


def load_sources(labels: list[str]) -> list[SourceImage]:
    manifest = read_manifest()
    sources: list[SourceImage] = []
    for label in labels:
        row = manifest[label]
        bag = row["bag"]
        bag_path = BAG_CACHE / bag
        for sensor, frames in CASES[label].items():
            for frame in frames:
                img = read_frame(bag_path, sensor, frame)
                if img is not None:
                    sources.append(SourceImage(label, bag, sensor, frame, img.astype(np.float32)))
        jpg = FULL_REVIEW / Path(bag).stem / "thermal_c.jpg"
        if jpg.exists():
            bgr = cv2.imread(str(jpg), cv2.IMREAD_COLOR)
            if bgr is not None:
                rgb = bgr[:, :, ::-1]
                scalar, _scalar_info = recover_scalar(rgb, cmap="auto")
                sources.append(SourceImage(label, bag, "jpg_scalar_recovered", None, scalar.astype(np.float32)))
                sources.append(SourceImage(label, bag, "jpg_gray", None, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)))
    return sources


def normalize_u8(img: np.ndarray, lo: float = 1, hi: float = 99) -> np.ndarray:
    src = img.astype(np.float32)
    finite = np.isfinite(src)
    if not finite.any():
        return np.zeros(src.shape[:2], dtype=np.uint8)
    a, b = np.percentile(src[finite], (lo, hi))
    if b <= a:
        b = a + 1.0
    return np.clip((src - a) * 255.0 / (b - a), 0, 255).astype(np.uint8)


def local_detail(img: np.ndarray, sigma: float) -> np.ndarray:
    src = img.astype(np.float32)
    bg = cv2.GaussianBlur(src, (0, 0), sigma)
    return normalize_u8(src - bg, 1, 99)


def variants(src: SourceImage) -> list[tuple[str, np.ndarray]]:
    img = src.image.astype(np.float32)
    out: list[tuple[str, np.ndarray]] = []
    if src.source.startswith("thermal"):
        for name, u8 in preprocess_raw_variants(img):
            out.append((name, u8))
        out.extend(
            [
                ("raw_p01_99", normalize_u8(img, 1, 99)),
                ("raw_p05_95", normalize_u8(img, 5, 95)),
                ("detail_s9", local_detail(img, 9)),
                ("detail_s17", local_detail(img, 17)),
            ]
        )
    else:
        u8 = normalize_u8(img, 1, 99)
        out.extend(
            [
                ("jpg_norm", u8),
                ("jpg_clahe", cv2.createCLAHE(2.5, (8, 8)).apply(u8)),
                ("jpg_detail_s9", local_detail(u8, 9)),
                ("jpg_unsharp", cv2.addWeighted(u8, 2.0, cv2.GaussianBlur(u8, (0, 0), 1.8), -1.0, 0)),
            ]
        )
    # De-noise and local contrast versions that help thresholding.
    enriched: list[tuple[str, np.ndarray]] = []
    for name, u8 in out:
        enriched.append((name, u8))
        enriched.append((f"{name}_bilateral", cv2.bilateralFilter(u8, 5, 30, 5)))
    return enriched


def checker_template(nx: int, ny: int, width: int, height: int) -> np.ndarray:
    small = np.zeros((ny, nx), dtype=np.uint8)
    for y in range(ny):
        for x in range(nx):
            small[y, x] = 255 if ((x + y) % 2) else 0
    templ = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    return cv2.GaussianBlur(templ, (3, 3), 0)


def cell_score(crop: np.ndarray, nx: int, ny: int) -> float:
    h, w = crop.shape[:2]
    means = np.zeros((ny, nx), dtype=np.float32)
    for y in range(ny):
        for x in range(nx):
            x0 = int(round(x * w / nx + 0.18 * w / nx))
            x1 = int(round((x + 1) * w / nx - 0.18 * w / nx))
            y0 = int(round(y * h / ny + 0.18 * h / ny))
            y1 = int(round((y + 1) * h / ny - 0.18 * h / ny))
            cell = crop[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)]
            means[y, x] = float(np.mean(cell))
    parity = np.fromfunction(lambda yy, xx: ((xx + yy) % 2) * 2 - 1, means.shape).astype(np.float32)
    centered = means - float(np.mean(means))
    denom = float(np.sqrt(np.sum(centered * centered) * np.sum(parity * parity))) + 1e-6
    corr = abs(float(np.sum(centered * parity) / denom))
    diff = abs(float(np.mean(means[parity > 0]) - np.mean(means[parity < 0])))
    return 100.0 * corr + 0.25 * diff


def template_search(u8: np.ndarray) -> list[dict[str, Any]]:
    img = cv2.GaussianBlur(u8, (3, 3), 0)
    h, w = img.shape[:2]
    candidates: list[dict[str, Any]] = []
    for nx, ny in GRID_MODELS:
        widths = range(120, min(430, w - 20), 18)
        for tw in widths:
            nominal_h = tw * ny / nx
            for scale_h in (0.65, 0.8, 1.0, 1.18):
                th = int(round(nominal_h * scale_h))
                if th < 50 or th >= h - 12:
                    continue
                templ = checker_template(nx, ny, tw, th)
                if templ.shape[0] >= img.shape[0] or templ.shape[1] >= img.shape[1]:
                    continue
                res = cv2.matchTemplate(img, templ, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if abs(min_val) > abs(max_val):
                    loc = min_loc
                    corr = -float(min_val)
                    polarity = "inv"
                else:
                    loc = max_loc
                    corr = float(max_val)
                    polarity = "pos"
                x0, y0 = loc
                crop = img[y0 : y0 + th, x0 : x0 + tw]
                model_score = cell_score(crop, nx, ny)
                score = 100.0 * corr + model_score
                candidates.append(
                    {
                        "grid_squares": [nx, ny],
                        "pattern_internal_corners": [nx - 1, ny - 1],
                        "x0": int(x0),
                        "y0": int(y0),
                        "x1": int(x0 + tw),
                        "y1": int(y0 + th),
                        "corr": corr,
                        "polarity": polarity,
                        "cell_score": model_score,
                        "score": score,
                    }
                )
    candidates.sort(key=lambda r: float(r["score"]), reverse=True)
    # Suppress near-duplicates.
    uniq: list[dict[str, Any]] = []
    for cand in candidates:
        cx = 0.5 * (cand["x0"] + cand["x1"])
        cy = 0.5 * (cand["y0"] + cand["y1"])
        if all(math.hypot(cx - 0.5 * (u["x0"] + u["x1"]), cy - 0.5 * (u["y0"] + u["y1"])) > 22 for u in uniq):
            uniq.append(cand)
        if len(uniq) >= 8:
            break
    return uniq


def corners_from_candidate(cand: dict[str, Any]) -> np.ndarray:
    nx, ny = cand["grid_squares"]
    xs = np.linspace(cand["x0"], cand["x1"], nx + 1, dtype=np.float32)[1:-1]
    ys = np.linspace(cand["y0"], cand["y1"], ny + 1, dtype=np.float32)[1:-1]
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)


def try_rectified_sb(u8: np.ndarray, cand: dict[str, Any]) -> tuple[bool, np.ndarray | None, str]:
    nx, ny = cand["grid_squares"]
    pattern = (nx - 1, ny - 1)
    crop = u8[cand["y0"] : cand["y1"], cand["x0"] : cand["x1"]]
    if crop.size == 0:
        return False, None, ""
    warp = cv2.resize(crop, (nx * 34, ny * 34), interpolation=cv2.INTER_CUBIC)
    variants = [
        ("rect_norm", warp),
        ("rect_clahe", cv2.createCLAHE(2.0, (4, 4)).apply(warp)),
        ("rect_inv", 255 - warp),
    ]
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    for name, src in variants:
        ok, corners = cv2.findChessboardCornersSB(src, pattern, flags=flags)
        if ok and corners is not None:
            pts = corners.reshape(-1, 2).astype(np.float32)
            pts[:, 0] = cand["x0"] + pts[:, 0] * (cand["x1"] - cand["x0"]) / float(nx * 34)
            pts[:, 1] = cand["y0"] + pts[:, 1] * (cand["y1"] - cand["y0"]) / float(ny * 34)
            return True, pts, name
    return False, None, ""


def watershed_debug(u8: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(u8, (3, 3), 0)
    _thr, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - binary
    # Pick polarity with more compact foreground.
    if np.mean(inv > 0) < np.mean(binary > 0):
        binary = inv
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    _ret, sure = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
    sure = sure.astype(np.uint8)
    unknown = cv2.subtract(cv2.dilate(opening, kernel, iterations=2), sure)
    _n, markers = cv2.connectedComponents(sure)
    markers = markers + 1
    markers[unknown == 255] = 0
    color = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color, markers)
    color[markers == -1] = (0, 0, 255)
    color[opening > 0] = cv2.addWeighted(color, 0.65, np.dstack([opening] * 3), 0.35, 0)[opening > 0]
    return color


def draw_overlay(u8: np.ndarray, cand: dict[str, Any], corners: np.ndarray, method: str) -> np.ndarray:
    color = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    x0, y0, x1, y1 = [cand[k] for k in ("x0", "y0", "x1", "y1")]
    cv2.rectangle(color, (x0, y0), (x1, y1), (0, 210, 255), 2)
    nx, ny = cand["grid_squares"]
    for x in np.linspace(x0, x1, nx + 1):
        cv2.line(color, (int(round(x)), y0), (int(round(x)), y1), (0, 120, 255), 1)
    for y in np.linspace(y0, y1, ny + 1):
        cv2.line(color, (x0, int(round(y))), (x1, int(round(y))), (0, 120, 255), 1)
    for i, p in enumerate(corners.astype(np.int32)):
        col = (0, 255, 0)
        if i == 0:
            col = (0, 0, 255)
        cv2.circle(color, tuple(p), 2, col, -1, cv2.LINE_AA)
    text = f"{method} grid={nx}x{ny} score={cand['score']:.1f} corr={cand['corr']:.2f}"
    cv2.putText(color, text[:95], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)
    return color


def fit_tile(img: np.ndarray, label: str, size: tuple[int, int] = (420, 320)) -> np.ndarray:
    tw, th = size
    canvas = np.full((th, tw, 3), 245, dtype=np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    h, w = bgr.shape[:2]
    scale = min(tw / max(w, 1), (th - 28) / max(h, 1))
    small = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    x = (tw - small.shape[1]) // 2
    y = 28 + (th - 28 - small.shape[0]) // 2
    canvas[y : y + small.shape[0], x : x + small.shape[1]] = small
    cv2.putText(canvas, label[:55], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 2, cv2.LINE_AA)
    return canvas


def process_source(src: SourceImage, out_dir: Path) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    best_images: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    all_attempts: list[dict[str, Any]] = []
    for variant_name, u8 in variants(src):
        cands = template_search(u8)
        if not cands:
            continue
        cand = cands[0]
        ok_sb, sb_corners, sb_method = try_rectified_sb(u8, cand)
        if ok_sb and sb_corners is not None:
            corners = sb_corners
            method = f"{variant_name}_template_rectified_sb_{sb_method}"
            cand = {**cand, "score": float(cand["score"]) + 40.0}
        else:
            corners = corners_from_candidate(cand)
            method = f"{variant_name}_template_model_grid"
        attempt = {
            "variant": variant_name,
            "method": method,
            "detected": True,
            "n_corners": int(corners.shape[0]),
            "corners_px": corners.tolist(),
            **cand,
        }
        all_attempts.append({k: v for k, v in attempt.items() if k != "corners_px"})
        if best is None or float(attempt["score"]) > float(best["score"]):
            best = attempt
            best_images = (u8, watershed_debug(u8), draw_overlay(u8, cand, corners, method))
    if best is None or best_images is None:
        return {
            "label_norm": src.label,
            "bag": src.bag,
            "source": src.source,
            "frame_index": src.frame_index,
            "detected": False,
            "attempts": all_attempts,
        }

    stem = f"{src.label}_{src.source}_{'jpg' if src.frame_index is None else f'f{src.frame_index:04d}'}"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    review_path = out_dir / f"{stem}_review.jpg"
    page = np.hstack(
        [
            fit_tile(normalize_u8(src.image, 1, 99), "raw / jpg input"),
            fit_tile(best_images[0], "best enhanced"),
            fit_tile(best_images[1], "watershed / binary diagnostic"),
            fit_tile(best_images[2], "recovered grid"),
        ]
    )
    cv2.imwrite(str(review_path), page)
    corr = float(best.get("corr", 0.0))
    rectified = "rectified_sb" in str(best.get("method", ""))
    if rectified or corr >= 0.55:
        confidence = "strong"
    elif corr >= 0.38:
        confidence = "partial_visual_check"
    else:
        confidence = "weak_reject"
    best["confidence"] = confidence
    best["usable_for_calibration"] = confidence == "strong"
    best["review_path"] = str(review_path)
    return {
        "label_norm": src.label,
        "bag": src.bag,
        "source": src.source,
        "frame_index": src.frame_index,
        "detected": True,
        "best": best,
        "attempts": all_attempts[:12],
    }


def make_summary_pages(results: list[dict[str, Any]], out_dir: Path) -> None:
    imgs = []
    for row in results:
        p = (row.get("best") or {}).get("review_path")
        img = cv2.imread(p) if p else None
        if img is not None:
            imgs.append(cv2.resize(img, (1120, 220), interpolation=cv2.INTER_AREA))
    for i, start in enumerate(range(0, len(imgs), 4), 1):
        batch = imgs[start : start + 4]
        if not batch:
            continue
        while len(batch) < 4:
            batch.append(np.full_like(batch[0], 245))
        cv2.imwrite(str(out_dir / f"thermal_grid_recovery_page_{i:02d}.jpg"), np.vstack(batch))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    labels = list(CASES)
    sources = load_sources(labels)
    results = []
    for i, src in enumerate(sources, 1):
        print(f"[{i}/{len(sources)}] {src.label} {src.source} {src.frame_index}")
        results.append(process_source(src, OUT))
    make_summary_pages(results, OUT)
    strong = sum(1 for r in results if (r.get("best") or {}).get("confidence") == "strong")
    partial = sum(1 for r in results if (r.get("best") or {}).get("confidence") == "partial_visual_check")
    weak = sum(1 for r in results if (r.get("best") or {}).get("confidence") == "weak_reject")
    summary = {
        "description": "Thermal square-grid recovery lab using RAW/Celsius and JPG scalar inputs.",
        "grid_models": GRID_MODELS,
        "n_sources": len(results),
        "detected": sum(1 for r in results if r.get("detected")),
        "confidence_counts": {"strong": strong, "partial_visual_check": partial, "weak_reject": weak},
        "results": results,
    }
    (OUT / "thermal_grid_recovery_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fields = ["label_norm", "bag", "source", "frame_index", "detected", "confidence", "usable_for_calibration", "grid", "method", "score", "corr", "review_path"]
    with (OUT / "thermal_grid_recovery_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            best = row.get("best") or {}
            writer.writerow(
                {
                    "label_norm": row.get("label_norm"),
                    "bag": row.get("bag"),
                    "source": row.get("source"),
                    "frame_index": row.get("frame_index"),
                    "detected": row.get("detected"),
                    "confidence": best.get("confidence", ""),
                    "usable_for_calibration": best.get("usable_for_calibration", ""),
                    "grid": "x".join(map(str, best.get("grid_squares", []))),
                    "method": best.get("method", ""),
                    "score": best.get("score", ""),
                    "corr": best.get("corr", ""),
                    "review_path": best.get("review_path", ""),
                }
            )
    print(json.dumps({"n_sources": len(results), "detected": summary["detected"], "out": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
