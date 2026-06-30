#!/usr/bin/env python3
"""Low-memory multiframe calibration target campaign for 2026-06-23 bags.

The goal is recall, not speed: scan every selected image frame, keep the best
quality candidates per sensor, try many enhancement variants, and write
checkpointed per-bag outputs.  It is intentionally sequential and bounded so it
can run on Windows without exhausting RAM.
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
from src.calibration.thermal_checker_raw import preprocess_raw_variants, robust01  # noqa: E402


MANIFEST = Path("data/calibration/new_session/20260623/bag_manifest_20260623.csv")
BASE_SUMMARY = Path("runs/calibration_20260623_refined_multisensor_detection/refined_multisensor_detection_summary.json")
DEEP_PF_SUMMARY = Path("runs/calibration_20260623_deep_photonfocus_detection/deep_photonfocus_detection_summary.json")
OUT = Path("runs/calibration_20260623_multiframe_campaign")

TOPICS = {
    "rgb": "/ssf/BFS_usb_0/image_raw",
    "vis": "/ssf/photonfocus_camera_vis_node/image_raw",
    "nir": "/ssf/photonfocus_camera_nir_node/image_raw",
    "thermal_c": "/ssf/thermalgrabber_ros/image_deg_celsius",
    "thermal_raw": "/ssf/thermalgrabber_ros/image_mono16",
}

PATTERNS = [(9, 6), (9, 5), (8, 5)]
PRIMARY_PATTERN = (9, 6)


@dataclass
class FrameCandidate:
    sensor: str
    frame_index: int
    stamp_ns: int
    encoding: str
    image: np.ndarray
    quality: float
    source: str = "frame"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def results_by_label(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("label_norm")): row for row in summary.get("results", [])}


def normalize_u8(image: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    src = image.astype(np.float32)
    finite = np.isfinite(src)
    if not finite.any():
        return np.zeros(src.shape[:2], dtype=np.uint8)
    a, b = np.percentile(src[finite], (lo, hi))
    if b <= a:
        b = a + 1.0
    return np.clip((src - a) * 255.0 / (b - a), 0, 255).astype(np.uint8)


def clahe(u8: np.ndarray, clip: float = 2.5, grid: int = 8) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(u8)


def unsharp(u8: np.ndarray, amount: float = 1.2, sigma: float = 1.6) -> np.ndarray:
    blur = cv2.GaussianBlur(u8, (0, 0), sigma)
    return cv2.addWeighted(u8, 1.0 + amount, blur, -amount, 0)


def local_detail_u8(img: np.ndarray, sigma: float = 9.0) -> np.ndarray:
    src = img.astype(np.float32)
    bg = cv2.GaussianBlur(src, (0, 0), sigma)
    return normalize_u8(src - bg, 1, 99)


def safe_box(box: list[float] | None, shape_hw: tuple[int, int], pad_frac: float = 0.45) -> tuple[int, int, int, int] | None:
    if box is None:
        return None
    h, w = shape_hw
    x0, y0, x1, y1 = [float(v) for v in box]
    if not all(np.isfinite([x0, y0, x1, y1])) or x1 <= x0 or y1 <= y0:
        return None
    px = max(16, int((x1 - x0) * pad_frac))
    py = max(16, int((y1 - y0) * pad_frac))
    bx0 = max(0, int(math.floor(x0)) - px)
    by0 = max(0, int(math.floor(y0)) - py)
    bx1 = min(w, int(math.ceil(x1)) + px)
    by1 = min(h, int(math.ceil(y1)) + py)
    if bx1 - bx0 < 35 or by1 - by0 < 25:
        return None
    return bx0, by0, bx1, by1


def crop_list(shape_hw: tuple[int, int], guide: list[float] | None) -> list[tuple[int, int, int, int, str]]:
    h, w = shape_hw
    crops: list[tuple[int, int, int, int, str]] = []
    g = safe_box(guide, shape_hw, 0.70)
    if g is not None:
        crops.append((*g, "guided"))
    crops.append((0, 0, w, h, "full"))
    crops.extend(
        [
            (0, 0, w // 2, h, "left"),
            (w // 2, 0, w, h, "right"),
            (0, 0, w, h // 2, "top"),
            (0, h // 2, w, h, "bottom"),
            (0, 0, int(0.72 * w), int(0.72 * h), "top_left"),
            (int(0.28 * w), 0, w, int(0.72 * h), "top_right"),
            (0, int(0.28 * h), int(0.72 * w), h, "bottom_left"),
            (int(0.28 * w), int(0.28 * h), w, h, "bottom_right"),
        ]
    )
    uniq: list[tuple[int, int, int, int, str]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for x0, y0, x1, y1, name in crops:
        key = (x0, y0, x1, y1)
        if key not in seen and x1 - x0 >= 35 and y1 - y0 >= 25:
            uniq.append((x0, y0, x1, y1, name))
            seen.add(key)
    return uniq


def photonfocus_bases(raw: np.ndarray, sensor: str) -> list[tuple[str, np.ndarray]]:
    if raw.ndim != 2:
        return [("gray", cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if raw.ndim == 3 else raw)]
    patterns = (4, 5) if sensor in ("vis", "nir") else (4,)
    out: list[tuple[str, np.ndarray]] = []
    for pat in patterns:
        work = raw[:1024, :] if pat == 4 else raw
        h = (work.shape[0] // pat) * pat
        w = (work.shape[1] // pat) * pat
        if h <= 0 or w <= 0:
            continue
        work = work[:h, :w]
        bands = [work[ro::pat, co::pat].astype(np.float32) for ro in range(pat) for co in range(pat)]
        stack = np.stack(bands, axis=-1)
        out.extend(
            [
                (f"mean{pat}", np.mean(stack, axis=-1)),
                (f"median{pat}", np.median(stack, axis=-1)),
                (f"max{pat}", np.max(stack, axis=-1)),
                (f"std{pat}", np.std(stack, axis=-1)),
                (f"range{pat}", np.max(stack, axis=-1) - np.min(stack, axis=-1)),
            ]
        )
        idxs = [0, pat + 1, 2 * pat + 2, pat * pat - 1, 1, pat]
        for idx in idxs:
            if 0 <= idx < len(bands):
                out.append((f"band{pat}_{idx:02d}", bands[idx]))
        # A few normalized band contrasts act like synthetic indices.
        pairs = [(0, pat * pat - 1), (pat + 1, 2 * pat + 2), (1, pat)]
        for a, b in pairs:
            if a < len(bands) and b < len(bands):
                num = bands[a] - bands[b]
                den = bands[a] + bands[b] + 1.0
                out.append((f"ndi{pat}_{a:02d}_{b:02d}", num / den))
    return out


def gray_variants(sensor: str, image: np.ndarray, exhaustive_variants: bool) -> list[tuple[str, np.ndarray]]:
    if sensor in ("vis", "nir"):
        bases = photonfocus_bases(image, sensor)
        preferred: list[tuple[str, np.ndarray]] = []
        base_names = {name: arr for name, arr in bases}
        keys = (
            "mean4",
            "median4",
            "max4",
            "range4",
            "std4",
            "band4_05",
            "band4_10",
            "mean5",
            "median5",
            "range5",
            "std5",
            "band5_06",
            "band5_12",
            "ndi4_00_15",
            "ndi5_00_24",
        )
        for key in keys:
            if key in base_names:
                preferred.append((key, base_names[key]))
        bases = preferred if not exhaustive_variants else bases
    elif sensor.startswith("thermal"):
        bases = [(name, img) for name, img in preprocess_raw_variants(image)]
        bases.extend(
            [
                ("raw_p01_99", normalize_u8(image, 1, 99)),
                ("raw_p05_95", normalize_u8(image, 5, 95)),
                ("raw_detail", local_detail_u8(image, 13)),
            ]
        )
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        bases = [("gray", gray)]

    variants: list[tuple[str, np.ndarray]] = []
    for name, base in bases:
        u8 = base if base.dtype == np.uint8 else normalize_u8(base)
        variants.append((f"{name}_norm", u8))
        variants.append((f"{name}_clahe", clahe(u8, 2.5, 8)))
        variants.append((f"{name}_clahe_fine", clahe(u8, 3.0, 4)))
        variants.append((f"{name}_unsharp", unsharp(u8)))
        variants.append((f"{name}_detail", local_detail_u8(u8, 7)))
        if sensor.startswith("thermal"):
            variants.append((f"{name}_blur", cv2.GaussianBlur(u8, (3, 3), 0)))
    # Deduplicate by name while preserving order.
    out: list[tuple[str, np.ndarray]] = []
    seen: set[str] = set()
    for name, arr in variants:
        if name not in seen:
            out.append((name, arr))
            seen.add(name)
    return out


def image_quality(sensor: str, image: np.ndarray, guide: list[float] | None) -> float:
    variants = gray_variants(sensor, image, exhaustive_variants=False)
    if not variants:
        return -1e9
    gray = variants[0][1]
    roi = safe_box(guide, gray.shape[:2], 0.35)
    if roi is not None:
        x0, y0, x1, y1 = roi
        gray = gray[y0:y1, x0:x1]
    if gray.size == 0:
        return -1e9
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    contrast = float(np.std(gray))
    grad = float(np.mean(np.abs(gx) + np.abs(gy)))
    sat = float(np.mean((gray <= 2) | (gray >= 253)))
    return contrast + 0.35 * grad - 25.0 * sat


def remap_affine_points(pts: np.ndarray, scale: float, x0: int, y0: int) -> np.ndarray:
    out = pts.reshape(-1, 2).astype(np.float32)
    out /= float(scale)
    out[:, 0] += x0
    out[:, 1] += y0
    return out


def order_and_score(pts: np.ndarray, pattern: tuple[int, int], shape_hw: tuple[int, int]) -> tuple[np.ndarray, float, float, bool]:
    cols, rows = pattern
    grid = pts.reshape(rows, cols, 2).astype(np.float32)
    row_vec = grid[-1, 0] - grid[0, 0]
    col_vec = grid[0, -1] - grid[0, 0]
    cross = float(col_vec[0] * row_vec[1] - col_vec[1] * row_vec[0])
    if cross < 0:
        grid = grid[::-1, :, :]
        row_vec = grid[-1, 0] - grid[0, 0]
        col_vec = grid[0, -1] - grid[0, 0]
        cross = float(col_vec[0] * row_vec[1] - col_vec[1] * row_vec[0])
    angle = float(np.degrees(np.arctan2(float(col_vec[1]), float(col_vec[0]))))
    dx = np.linalg.norm(np.diff(grid, axis=1), axis=2)
    dy = np.linalg.norm(np.diff(grid, axis=0), axis=2)
    spacing = float(np.median(np.r_[dx.reshape(-1), dy.reshape(-1)]))
    x0, y0 = np.min(grid[:, :, 0]), np.min(grid[:, :, 1])
    x1, y1 = np.max(grid[:, :, 0]), np.max(grid[:, :, 1])
    area = max(1.0, float((x1 - x0) * (y1 - y0)))
    img_area = max(1.0, float(shape_hw[0] * shape_hw[1]))
    primary_bonus = 120.0 if pattern == PRIMARY_PATTERN else 0.0
    quality = primary_bonus + spacing + 100.0 * min(1.0, area / img_area) + 0.02 * area
    rotation_ok = bool(cross > 0 and np.linalg.norm(col_vec) >= 0.75 * np.linalg.norm(row_vec))
    return grid.reshape(-1, 2), angle, quality, rotation_ok


def detect_checker(sensor: str, gray: np.ndarray, guide: list[float] | None, deep: bool) -> dict[str, Any]:
    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS
    sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY
    if deep:
        sb_flags |= cv2.CALIB_CB_EXHAUSTIVE
    scales = (1.0, 1.5) if max(gray.shape[:2]) < 900 else (0.5, 1.0)
    best: dict[str, Any] | None = None
    crops = crop_list(gray.shape[:2], guide)
    if guide is not None:
        # Keep the normal campaign bounded. The broad crops are useful only
        # when there is no prior projected panel box.
        crops = [c for c in crops if c[4] == "guided"]
        if not sensor.startswith("thermal"):
            crops += [c for c in crop_list(gray.shape[:2], guide) if c[4] == "full"]
    for x0, y0, x1, y1, crop_name in crops:
        crop = gray[y0:y1, x0:x1]
        for scale in scales:
            work = cv2.resize(
                crop,
                (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
                interpolation=cv2.INTER_CUBIC if scale >= 1 else cv2.INTER_AREA,
            )
            for inverted, src in ((False, work), (True, 255 - work)):
                for pattern in PATTERNS:
                    ok, corners = cv2.findChessboardCorners(src, pattern, flags=classic_flags)
                    alg = "classic"
                    use_sb = not sensor.startswith("thermal") or deep
                    if not ok and use_sb and (pattern == PRIMARY_PATTERN or deep):
                        ok, corners = cv2.findChessboardCornersSB(src, pattern, flags=sb_flags)
                        alg = "sb_deep" if deep else "sb"
                    if not ok or corners is None:
                        continue
                    pts = remap_affine_points(corners, scale, x0, y0)
                    try:
                        if alg == "classic":
                            local = corners.astype(np.float32)
                            cv2.cornerSubPix(
                                src,
                                local,
                                (7, 7),
                                (-1, -1),
                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3),
                            )
                            pts = remap_affine_points(local, scale, x0, y0)
                    except Exception:
                        pass
                    pts, angle, score, rotation_ok = order_and_score(pts, pattern, gray.shape[:2])
                    rec = {
                        "checker_detected": True,
                        "pattern_internal_corners": list(pattern),
                        "corners_px": pts.tolist(),
                        "orientation_deg": angle,
                        "rotation_ok": rotation_ok,
                        "score": score,
                        "sweep": f"{alg}_{crop_name}_s{scale:g}_{'inv' if inverted else 'norm'}",
                    }
                    if best is None or score > float(best["score"]):
                        best = rec
                    if pattern == PRIMARY_PATTERN and rotation_ok and crop_name == "guided":
                        return rec
    return best or {"checker_detected": False, "score": 0.0}


def panel_from_corners(corners: list[list[float]], pad_frac: float = 0.18) -> list[float]:
    pts = np.asarray(corners, dtype=np.float32)
    x0, y0 = np.min(pts[:, 0]), np.min(pts[:, 1])
    x1, y1 = np.max(pts[:, 0]), np.max(pts[:, 1])
    px = (x1 - x0) * pad_frac
    py = (y1 - y0) * pad_frac
    return [float(x0 - px), float(y0 - py), float(x1 + px), float(y1 + py)]


def draw_review(sensor: str, gray: np.ndarray, det: dict[str, Any], out_path: Path, title: str) -> None:
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if det.get("checker_detected"):
        pts = np.asarray(det["corners_px"], dtype=np.int32)
        for i, p in enumerate(pts):
            col = (0, 255, 0)
            if i == 0:
                col = (0, 0, 255)
            elif i == PRIMARY_PATTERN[0] - 1:
                col = (255, 0, 0)
            cv2.circle(color, tuple(p), 3, col, -1, cv2.LINE_AA)
        hull = cv2.convexHull(pts.astype(np.float32)).astype(np.int32)
        cv2.polylines(color, [hull], True, (0, 220, 0), 2, cv2.LINE_AA)
    h, w = color.shape[:2]
    scale = min(760 / max(w, 1), 480 / max(h, 1), 3.0)
    view = cv2.resize(color, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_NEAREST if scale > 1 else cv2.INTER_AREA)
    text = f"{sensor} {title[:62]} checker={det.get('checker_detected')}"
    cv2.putText(view, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), view)


def make_composites(candidates: list[FrameCandidate]) -> list[FrameCandidate]:
    if len(candidates) < 2:
        return []
    imgs = [c.image.astype(np.float32) for c in candidates]
    shapes = {im.shape for im in imgs}
    if len(shapes) != 1:
        return []
    stack = np.stack(imgs, axis=0)
    stamp = int(candidates[0].stamp_ns)
    sensor = candidates[0].sensor
    return [
        FrameCandidate(sensor, -1, stamp, "composite", np.median(stack, axis=0).astype(np.float32), candidates[0].quality, "median_frames"),
        FrameCandidate(sensor, -2, stamp, "composite", np.mean(stack, axis=0).astype(np.float32), candidates[0].quality, "mean_frames"),
    ]


def collect_top_frames(
    bag_path: Path,
    sensors: list[str],
    guides: dict[str, list[float] | None],
    max_quality_frames: int,
    frame_step: int,
) -> dict[str, list[FrameCandidate]]:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    top: dict[str, list[FrameCandidate]] = {sensor: [] for sensor in sensors}
    with Reader(bag_path) as reader:
        available = {c.topic: c for c in reader.connections}
        selected = []
        topic_to_sensor = {}
        for sensor in sensors:
            topic = TOPICS[sensor]
            if topic in available:
                selected.append(available[topic])
                topic_to_sensor[topic] = sensor
        counters = {sensor: 0 for sensor in sensors}
        for conn, ts, raw in reader.messages(connections=selected):
            sensor = topic_to_sensor[conn.topic]
            idx = counters[sensor]
            counters[sensor] += 1
            if frame_step > 1 and idx % frame_step != 0:
                continue
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            img = decode_ros_image(msg)
            if img is None:
                continue
            quality = image_quality(sensor, img, guides.get(sensor))
            cand = FrameCandidate(sensor, idx, int(ts), str(getattr(msg, "encoding", "")), img, quality)
            bucket = top[sensor]
            bucket.append(cand)
            bucket.sort(key=lambda c: c.quality, reverse=True)
            del bucket[max_quality_frames:]
    return top


def best_existing(label: str, sensor: str, base: dict[str, dict[str, Any]], deep_pf: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    choices = []
    b = base.get(label, {}).get("detections", {}).get(sensor)
    d = deep_pf.get(label, {}).get("detections", {}).get(sensor)
    if b and b.get("checker_detected"):
        choices.append(("base", b))
    if d and d.get("checker_detected"):
        choices.append(("deep_pf", d))
    if not choices:
        return None
    choices.sort(key=lambda x: float(x[1].get("quality", 0.0)), reverse=True)
    source, det = choices[0]
    return {
        "checker_detected": True,
        "source": source,
        "method": det.get("method", ""),
        "frame_index": det.get("frame_index"),
        "stamp_ns": det.get("stamp_ns"),
        "corners_px": det.get("corners_px"),
        "panel_box_xyxy": det.get("panel_box_xyxy"),
        "pattern_internal_corners": det.get("pattern_internal_corners") or list(PRIMARY_PATTERN),
        "score": det.get("quality", 0.0),
        "rotation_ok": det.get("rotation_ok", False),
    }


def sensor_guides(label: str, base: dict[str, dict[str, Any]]) -> dict[str, list[float] | None]:
    row = base.get(label, {})
    boxes = row.get("projected_boxes_from_rgb", {})
    dets = row.get("detections", {})
    guides: dict[str, list[float] | None] = {}
    for sensor in TOPICS:
        guide = boxes.get(sensor)
        if guide is None:
            guide = dets.get(sensor, {}).get("panel_box_xyxy")
        guides[sensor] = guide
    return guides


def process_sensor(
    label: str,
    sensor: str,
    candidates: list[FrameCandidate],
    guide: list[float] | None,
    out_dir: Path,
    exhaustive_variants: bool,
    deep: bool,
    max_variants_per_frame: int,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    attempts = []
    all_candidates = candidates + make_composites(candidates)
    for cand_i, cand in enumerate(all_candidates):
        variants = gray_variants(sensor, cand.image, exhaustive_variants=exhaustive_variants)
        for var_i, (variant_name, gray) in enumerate(variants):
            if var_i >= max_variants_per_frame:
                break
            # Keep very broad full-image attempts for fallback, but guide first.
            det = detect_checker(sensor, gray, guide, deep=deep)
            rec = {
                "sensor": sensor,
                "frame_index": cand.frame_index,
                "stamp_ns": cand.stamp_ns,
                "encoding": cand.encoding,
                "frame_source": cand.source,
                "frame_quality": cand.quality,
                "variant": variant_name,
                **det,
            }
            attempts.append({k: v for k, v in rec.items() if k != "corners_px"})
            if det.get("checker_detected"):
                rec["method"] = f"{variant_name}_{det.get('sweep', '')}"
                rec["panel_box_xyxy"] = panel_from_corners(det["corners_px"])
                score = float(det.get("score", 0.0)) + 5.0 * float(cand.quality)
                rec["combined_score"] = score
                if best is None or score > float(best.get("combined_score", -1e18)):
                    best = rec
                    review_path = out_dir / "bags" / label / f"{sensor}_best.jpg"
                    draw_review(sensor, gray, rec, review_path, f"{variant_name} f{cand.frame_index}")
                    best["review_path"] = str(review_path)
                if det.get("pattern_internal_corners") == list(PRIMARY_PATTERN) and det.get("rotation_ok") and guide is not None:
                    break
        if best is not None and best.get("pattern_internal_corners") == list(PRIMARY_PATTERN) and best.get("rotation_ok"):
            # Do not spend all variants once a strong full checker is found.
            break
        if cand_i >= len(candidates) + 1 and not deep:
            break
    return {
        "checker_detected": bool(best),
        "best": best,
        "n_frame_candidates": len(candidates),
        "n_attempts": len(attempts),
        "top_attempts": sorted(attempts, key=lambda r: float(r.get("score", 0.0)), reverse=True)[:20],
    }


def process_bag(
    row: dict[str, str],
    bag_path: Path,
    base: dict[str, dict[str, Any]],
    deep_pf: dict[str, dict[str, Any]],
    out_dir: Path,
    sensors: list[str],
    max_quality_frames: int,
    frame_step: int,
    exhaustive_variants: bool,
    deep: bool,
    prefer_existing: bool,
    max_variants_per_frame: int,
) -> dict[str, Any]:
    label = row["label_norm"]
    guides = sensor_guides(label, base)
    top_frames = collect_top_frames(bag_path, sensors, guides, max_quality_frames, frame_step)
    detections: dict[str, Any] = {}
    for sensor in sensors:
        existing = best_existing(label, sensor, base, deep_pf) if prefer_existing else None
        current = process_sensor(
            label,
            sensor,
            top_frames.get(sensor, []),
            guides.get(sensor),
            out_dir,
            exhaustive_variants=exhaustive_variants,
            deep=deep,
            max_variants_per_frame=max_variants_per_frame,
        )
        if existing and not current.get("checker_detected"):
            current["best"] = existing
            current["checker_detected"] = True
            current["used_existing_fallback"] = True
        elif existing and current.get("best") and float(existing.get("score", 0.0)) > float(current["best"].get("combined_score", 0.0)):
            current["new_detection_kept_as_alternative"] = current["best"]
            current["best"] = existing
            current["used_existing_fallback"] = True
        detections[sensor] = current
    return {
        **row,
        "bag_path_local": str(bag_path),
        "processed": True,
        "detections": detections,
    }


def write_outputs(results: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        sensor: sum(1 for row in results if row.get("detections", {}).get(sensor, {}).get("checker_detected"))
        for sensor in args.sensors
    }
    summary = {
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "n_bags": len(results),
        "checker_counts": counts,
        "results": results,
    }
    (out_dir / "multiframe_campaign_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fields = ["label_norm", "bag", "processed"]
    for sensor in args.sensors:
        fields.extend([f"{sensor}_checker", f"{sensor}_method", f"{sensor}_frame", f"{sensor}_score", f"{sensor}_review"])
    with (out_dir / "multiframe_campaign_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            rec = {"label_norm": row.get("label_norm"), "bag": row.get("bag"), "processed": row.get("processed")}
            for sensor in args.sensors:
                det = row.get("detections", {}).get(sensor, {})
                best = det.get("best") or {}
                rec[f"{sensor}_checker"] = det.get("checker_detected", False)
                rec[f"{sensor}_method"] = best.get("method", best.get("source", ""))
                rec[f"{sensor}_frame"] = best.get("frame_index", "")
                rec[f"{sensor}_score"] = best.get("combined_score", best.get("score", ""))
                rec[f"{sensor}_review"] = best.get("review_path", "")
            writer.writerow(rec)

    compatible = {"results": []}
    for row in results:
        dets = {}
        for sensor in args.sensors:
            best = (row.get("detections", {}).get(sensor, {}) or {}).get("best") or {}
            dets[sensor] = {
                "checker_detected": bool(best.get("checker_detected")),
                "method": best.get("method", ""),
                "frame_index": best.get("frame_index"),
                "stamp_ns": best.get("stamp_ns"),
                "pattern_internal_corners": best.get("pattern_internal_corners"),
                "corners_px": best.get("corners_px"),
                "panel_box_xyxy": best.get("panel_box_xyxy"),
                "score": best.get("combined_score", best.get("score", 0.0)),
                "review_path": best.get("review_path", ""),
            }
        compatible["results"].append({"label_norm": row.get("label_norm"), "bag": row.get("bag"), "detections": dets})
    compatible["checker_counts"] = counts
    compatible["source"] = str(out_dir / "multiframe_campaign_summary.json")
    (out_dir / "multiframe_candidates_for_registration.json").write_text(json.dumps(compatible, indent=2), encoding="utf-8")

    make_contact_sheets(results, out_dir, args.sensors)


def make_contact_sheets(results: list[dict[str, Any]], out_dir: Path, sensors: list[str]) -> None:
    rows = []
    for row in results:
        cells = []
        for sensor in sensors:
            best = (row.get("detections", {}).get(sensor, {}) or {}).get("best") or {}
            path = best.get("review_path")
            img = cv2.imread(path) if path else None
            if img is None:
                img = np.full((210, 300, 3), 240, dtype=np.uint8)
                cv2.putText(img, f"{sensor}: no new review", (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 2)
            img = cv2.resize(img, (320, 210), interpolation=cv2.INTER_AREA)
            cells.append(img)
        strip = np.hstack(cells)
        label = f"{row.get('label_norm')} | " + " ".join(
            f"{s}:{int(bool(row.get('detections', {}).get(s, {}).get('checker_detected')))}" for s in sensors
        )
        canvas = np.full((strip.shape[0] + 30, strip.shape[1], 3), 250, dtype=np.uint8)
        canvas[30:, :] = strip
        cv2.putText(canvas, label[:120], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
        rows.append(canvas)
    for page_i, start in enumerate(range(0, len(rows), 4), 1):
        batch = rows[start : start + 4]
        if not batch:
            continue
        while len(batch) < 4:
            batch.append(np.full_like(batch[0], 250))
        cv2.imwrite(str(out_dir / f"multiframe_campaign_page_{page_i:02d}.jpg"), np.vstack(batch))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--bag-cache", type=Path, required=True)
    parser.add_argument("--base-summary", type=Path, default=BASE_SUMMARY)
    parser.add_argument("--deep-pf-summary", type=Path, default=DEEP_PF_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--sensors", nargs="+", choices=sorted(TOPICS), default=["vis", "nir", "thermal_raw", "thermal_c"])
    parser.add_argument("--include-nondefault", action="store_true")
    parser.add_argument("--only-label", default="")
    parser.add_argument("--max-quality-frames", type=int, default=6)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--exhaustive-variants", action="store_true")
    parser.add_argument("--deep", action="store_true", help="Use exhaustive SB on top frames. Slower but higher recall.")
    parser.add_argument("--max-variants-per-frame", type=int, default=8)
    parser.add_argument("--no-existing-fallback", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = read_manifest(args.manifest, include_nondefault=args.include_nondefault)
    if args.only_label:
        rows = [r for r in rows if r.get("label_norm") == args.only_label or r.get("bag") == args.only_label]
    base = results_by_label(load_json(args.base_summary))
    deep_pf = results_by_label(load_json(args.deep_pf_summary))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for i, row in enumerate(rows, 1):
        label = row["label_norm"]
        checkpoint = args.out_dir / "checkpoints" / f"{label}.json"
        if args.resume and checkpoint.exists():
            results.append(load_json(checkpoint))
            print(f"[{i}/{len(rows)}] {label}: checkpoint")
            continue
        bag_path = args.bag_cache / row["bag"]
        if not bag_path.exists():
            rec = {**row, "processed": False, "reason": f"missing_local_bag:{bag_path}", "detections": {}}
            results.append(rec)
            continue
        print(f"[{i}/{len(rows)}] {label}: scanning {', '.join(args.sensors)}")
        rec = process_bag(
            row,
            bag_path,
            base,
            deep_pf,
            args.out_dir,
            args.sensors,
            max(1, args.max_quality_frames),
            max(1, args.frame_step),
            exhaustive_variants=args.exhaustive_variants,
            deep=args.deep,
            prefer_existing=not args.no_existing_fallback,
            max_variants_per_frame=max(1, args.max_variants_per_frame),
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        results.append(rec)
        write_outputs(results, args.out_dir, args)
    write_outputs(results, args.out_dir, args)
    print(json.dumps({"processed": sum(1 for r in results if r.get("processed")), "checker_counts": {
        sensor: sum(1 for row in results if row.get("detections", {}).get(sensor, {}).get("checker_detected"))
        for sensor in args.sensors
    }, "out_dir": str(args.out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
