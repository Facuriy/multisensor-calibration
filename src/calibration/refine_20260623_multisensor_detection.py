#!/usr/bin/env python3
"""Robust multisensor checkerboard/panel detection for the 2026-06-23 dataset."""

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

from src.extraction.extract_all_bag_images import decode_ros_image, photonfocus_preview, robust_preview  # noqa: E402


TOPICS = {
    "rgb": "/ssf/BFS_usb_0/image_raw",
    "vis": "/ssf/photonfocus_camera_vis_node/image_raw",
    "nir": "/ssf/photonfocus_camera_nir_node/image_raw",
    "thermal_c": "/ssf/thermalgrabber_ros/image_deg_celsius",
}

PATTERN = (9, 6)
COLS, ROWS = PATTERN


@dataclass
class Detection:
    checker_detected: bool
    panel_detected: bool
    method: str
    frame_index: int | None
    stamp_ns: int | None
    corners_px: np.ndarray | None
    panel_box_xyxy: list[float] | None
    orientation_deg: float | None
    rotation_ok: bool
    quality: float
    source_shape: tuple[int, int] | None
    reason: str = ""
    processed_image: np.ndarray | None = None


def read_topic_messages(bag: Path, topics: dict[str, str]) -> dict[str, list[tuple[int, object]]]:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    out = {k: [] for k in topics}
    topic_to_key = {v: k for k, v in topics.items()}
    with Reader(bag) as reader:
        conns = [c for c in reader.connections if c.topic in topic_to_key]
        if not conns:
            return out
        for conn, ts, raw in reader.messages(connections=conns):
            key = topic_to_key[conn.topic]
            out[key].append((int(ts), typestore.deserialize_ros1(raw, conn.msgtype)))
    return out


def sampled_indices(count: int, max_frames: int) -> set[int]:
    if count <= 0:
        return set()
    max_frames = max(1, min(max_frames, count))
    center = count // 2
    order = [center, center - 1, center + 1, center - 2, center + 2, 0, count - 1]
    if max_frames > len(order):
        order.extend(np.linspace(0, count - 1, max_frames, dtype=int).tolist())
    out = []
    seen = set()
    for idx in order:
        if 0 <= idx < count and idx not in seen:
            out.append(idx)
            seen.add(idx)
        if len(out) >= max_frames:
            break
    return set(out)


def read_sampled_messages(bag: Path, topics: dict[str, str], row: dict[str, str], max_frames: int) -> dict[str, list[tuple[int, object]]]:
    """Read only a small time window around the middle of the bag."""
    typestore = get_typestore(Stores.ROS1_NOETIC)
    topic_to_key = {v: k for k, v in topics.items()}
    with Reader(bag) as reader:
        conns = [c for c in reader.connections if c.topic in topic_to_key]
        if not conns:
            return {k: [] for k in topics}
        center = (int(reader.start_time) + int(reader.end_time)) // 2
        for half_window_ns in (400_000_000, 900_000_000, 1_800_000_000, None):
            raw_by_key: dict[str, list[tuple[int, Any]]] = {k: [] for k in topics}
            kwargs = {}
            if half_window_ns is not None:
                kwargs = {"start": center - half_window_ns, "stop": center + half_window_ns}
            for conn, ts, raw in reader.messages(connections=conns, **kwargs):
                key = topic_to_key[conn.topic]
                raw_by_key[key].append((int(ts), raw, conn.msgtype))
            if all(raw_by_key[k] for k in topics if int(row.get({"rgb": "rgb_msgs", "vis": "vis_msgs", "nir": "nir_msgs", "thermal_c": "thermal_c_msgs"}[k], "0") or 0) > 0):
                out: dict[str, list[tuple[int, object]]] = {k: [] for k in topics}
                for key, items in raw_by_key.items():
                    items = sorted(items, key=lambda item: abs(item[0] - center))[:max(1, max_frames)]
                    items = sorted(items, key=lambda item: item[0])
                    out[key] = [(ts, typestore.deserialize_ros1(raw, msgtype)) for ts, raw, msgtype in items]
                return out
        return {k: [] for k in topics}


def nearest_index(items: list[tuple[int, object]], stamp_ns: int | None) -> int:
    if not items:
        return -1
    if stamp_ns is None:
        return len(items) // 2
    return int(np.argmin([abs(ts - stamp_ns) for ts, _msg in items]))


def frame_indices(n: int, center: int | None, max_frames: int) -> list[int]:
    if n <= 0:
        return []
    indices: list[int] = []
    if center is None or center < 0:
        center = n // 2
    for off in (0, -1, 1, -2, 2, -4, 4):
        idx = center + off
        if 0 <= idx < n:
            indices.append(idx)
    if len(indices) < max_frames:
        spread = np.linspace(0, n - 1, min(max_frames, n), dtype=int).tolist()
        indices.extend(spread)
    seen = set()
    uniq = []
    for idx in indices:
        if idx not in seen:
            uniq.append(idx)
            seen.add(idx)
    return uniq[:max_frames]


def normalize_u8(image: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    src = image.astype(np.float32)
    finite = np.isfinite(src)
    if not finite.any():
        return np.zeros(src.shape, dtype=np.uint8)
    lo, hi = np.percentile(src[finite], [low, high])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((src - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def clahe(image: np.ndarray, clip: float = 3.0, tile: int = 8) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(image)


def unsharp(image: np.ndarray, amount: float = 1.0, sigma: float = 2.0) -> np.ndarray:
    return cv2.addWeighted(image, 1.0 + amount, cv2.GaussianBlur(image, (0, 0), sigma), -amount, 0)


def local_contrast(image: np.ndarray) -> np.ndarray:
    src = image.astype(np.float32)
    blur = cv2.GaussianBlur(src, (0, 0), 9)
    detail = src - blur
    return normalize_u8(detail, 1, 99)


def photonfocus_candidates(raw: np.ndarray) -> list[tuple[str, np.ndarray]]:
    raw = raw[:1024, :]
    usable_h = (raw.shape[0] // 4) * 4
    usable_w = (raw.shape[1] // 4) * 4
    raw = raw[:usable_h, :usable_w]
    bands = [raw[ro::4, co::4] for ro in range(4) for co in range(4)]
    stack = np.stack([b.astype(np.float32) for b in bands], axis=-1)
    candidates: list[tuple[str, np.ndarray]] = [
        ("mean4", photonfocus_preview(raw)),
        ("median4", np.median(stack, axis=-1)),
        ("max4", np.max(stack, axis=-1)),
        ("std4", np.std(stack, axis=-1)),
    ]
    # Diagonal and center-like bands often carry the cleanest checker contrast.
    for idx in [0, 5, 10, 15, 1, 4, 6, 9, 11, 14]:
        candidates.append((f"band{idx:02d}", bands[idx]))
    return candidates


def gray_candidates(sensor: str, img: np.ndarray) -> list[tuple[str, np.ndarray]]:
    if sensor in ("vis", "nir"):
        bases = photonfocus_candidates(img)
    elif sensor == "thermal_c":
        bases = [
            ("thermal_p01_99", normalize_u8(img, 1, 99)),
            ("thermal_p05_95", normalize_u8(img, 5, 95)),
            ("thermal_p10_90", normalize_u8(img, 10, 90)),
            ("thermal_inv_p05_95", 255 - normalize_u8(img, 5, 95)),
        ]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        bases = [("gray", gray)]

    out: list[tuple[str, np.ndarray]] = []
    for name, base in bases:
        u = base if base.dtype == np.uint8 else normalize_u8(base)
        out.append((f"{name}_norm", u))
        out.append((f"{name}_clahe", clahe(u, 3.0, 8)))
        out.append((f"{name}_unsharp", unsharp(u, 0.8, 2.0)))
        if sensor == "thermal_c":
            out.append((f"{name}_local", local_contrast(u)))
            out.append((f"{name}_blur", cv2.GaussianBlur(u, (3, 3), 0)))
    if sensor in ("vis", "nir"):
        # Keep the expensive detector bounded. These variants gave the best
        # signal in spot checks; all share the same 1/4 Photonfocus geometry.
        preferred = []
        keys = (
            "mean4_norm", "mean4_clahe", "median4_norm",
            "max4_norm", "band05_norm", "band10_norm", "band15_norm",
        )
        by_name = {name: img for name, img in out}
        for key in keys:
            if key in by_name:
                preferred.append((key, by_name[key]))
        return preferred
    if sensor == "thermal_c":
        preferred = []
        keys = (
            "thermal_p05_95_clahe", "thermal_p05_95_local",
            "thermal_p10_90_clahe", "thermal_inv_p05_95_clahe",
            "thermal_p01_99_clahe", "thermal_p05_95_unsharp",
        )
        by_name = {name: img for name, img in out}
        for key in keys:
            if key in by_name:
                preferred.append((key, by_name[key]))
        return preferred
    return out[:4]


def roi_crops(shape: tuple[int, int], projected_box: list[float] | None) -> list[tuple[int, int, int, int, str]]:
    h, w = shape
    crops = [(0, 0, w, h, "full")]
    if projected_box is not None:
        x0, y0, x1, y1 = projected_box
        pad_x = max(25, int((x1 - x0) * 0.55))
        pad_y = max(25, int((y1 - y0) * 0.55))
        crops.insert(0, (max(0, int(x0) - pad_x), max(0, int(y0) - pad_y),
                         min(w, int(x1) + pad_x), min(h, int(y1) + pad_y), "guided"))
    # Broad crops for edge poses.
    crops.extend([
        (0, 0, w // 2, h, "left_half"),
        (w // 2, 0, w, h, "right_half"),
        (0, 0, w, h // 2, "top_half"),
        (0, h // 2, w, h, "bottom_half"),
        (0, 0, int(w * 0.7), int(h * 0.7), "top_left"),
        (int(w * 0.3), 0, w, int(h * 0.7), "top_right"),
        (0, int(h * 0.3), int(w * 0.7), h, "bottom_left"),
        (int(w * 0.3), int(h * 0.3), w, h, "bottom_right"),
    ])
    uniq = []
    seen = set()
    for x0, y0, x1, y1, name in crops:
        if x1 - x0 < 40 or y1 - y0 < 30:
            continue
        key = (x0, y0, x1, y1)
        if key not in seen:
            uniq.append((x0, y0, x1, y1, name))
            seen.add(key)
    return uniq


def try_checker_on_gray(gray: np.ndarray, projected_box: list[float] | None, sensor: str) -> tuple[np.ndarray | None, str]:
    sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY
    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS
    if sensor == "thermal_c":
        scales = (1.0,)
    elif sensor in ("vis", "nir"):
        scales = (1.0,)
    else:
        scales = (0.25, 0.5, 0.75, 1.0)

    crops_all = roi_crops(gray.shape[:2], projected_box)
    if projected_box is not None:
        crops = [c for c in crops_all if c[4] == "guided"] + [c for c in crops_all if c[4] == "full"]
    elif sensor == "thermal_c":
        crops = [c for c in crops_all if c[4] == "full"]
    elif sensor in ("vis", "nir"):
        crops = [c for c in crops_all if c[4] == "full"]
    else:
        crops = [c for c in crops_all if c[4] in ("full", "left_half", "right_half", "top_half", "bottom_half")]

    for x0, y0, x1, y1, crop_name in crops:
        crop = gray[y0:y1, x0:x1]
        for scale in scales:
            resized = cv2.resize(
                crop,
                (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
            )
            for inv in (False, True):
                src = 255 - resized if inv else resized
                if sensor == "thermal_c":
                    # Thermal corners are blurred; classic is faster and often
                    # less stubborn here. Full thermal checker is optional
                    # because panel localization is the primary fallback.
                    ok, corners = cv2.findChessboardCorners(src, PATTERN, flags=classic_flags)
                    alg = "classic"
                elif sensor == "rgb":
                    ok, corners = cv2.findChessboardCorners(src, PATTERN, flags=classic_flags)
                    alg = "classic"
                else:
                    ok, corners = cv2.findChessboardCorners(src, PATTERN, flags=classic_flags)
                    alg = "classic"
                    if not ok:
                        ok, corners = cv2.findChessboardCornersSB(src, PATTERN, flags=sb_flags)
                        alg = "sb"
                if ok and corners is not None:
                    pts = corners.reshape(-1, 2).astype(np.float32) / scale
                    if sensor == "rgb":
                        try:
                            local = pts.reshape(-1, 1, 2).astype(np.float32)
                            cv2.cornerSubPix(
                                crop,
                                local,
                                (7, 7),
                                (-1, -1),
                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01),
                            )
                            pts = local.reshape(-1, 2)
                        except Exception:
                            pass
                    pts[:, 0] += x0
                    pts[:, 1] += y0
                    return pts, f"{alg}_{crop_name}_s{scale:g}_{'inv' if inv else 'norm'}"
    return None, ""


def canonicalize_and_score(corners: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, float, float, bool]:
    pts = corners.reshape(ROWS, COLS, 2).astype(np.float32)
    # Prefer row-major order with the first row visually above the last row.
    row_vec = pts[-1, 0] - pts[0, 0]
    col_vec = pts[0, -1] - pts[0, 0]
    cross = float(col_vec[0] * row_vec[1] - col_vec[1] * row_vec[0])
    if cross < 0:
        pts = pts[::-1, :, :]
        row_vec = pts[-1, 0] - pts[0, 0]
        col_vec = pts[0, -1] - pts[0, 0]
        cross = float(col_vec[0] * row_vec[1] - col_vec[1] * row_vec[0])
    if np.linalg.norm(col_vec) < np.linalg.norm(row_vec):
        # This should not happen for a 9x6 pattern unless ordering is rotated or
        # detection is bad. Keep points, but flag it.
        rotation_ok = False
    else:
        rotation_ok = cross > 0
    angle = math.degrees(math.atan2(float(col_vec[1]), float(col_vec[0])))
    diffs_x = np.linalg.norm(np.diff(pts, axis=1), axis=2)
    diffs_y = np.linalg.norm(np.diff(pts, axis=0), axis=2)
    spacing = float(np.median(np.r_[diffs_x.reshape(-1), diffs_y.reshape(-1)]))
    bbox = [pts[:, :, 0].min(), pts[:, :, 1].min(), pts[:, :, 0].max(), pts[:, :, 1].max()]
    bbox_area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    image_area = max(1.0, shape[0] * shape[1])
    quality = spacing + 80.0 * min(1.0, bbox_area / image_area)
    return pts.reshape(-1, 2), angle, quality, rotation_ok


def panel_from_checker(corners: np.ndarray, pad_frac: float = 0.18) -> list[float]:
    x0, y0 = corners[:, 0].min(), corners[:, 1].min()
    x1, y1 = corners[:, 0].max(), corners[:, 1].max()
    pad_x = (x1 - x0) * pad_frac
    pad_y = (y1 - y0) * pad_frac
    return [float(x0 - pad_x), float(y0 - pad_y), float(x1 + pad_x), float(y1 + pad_y)]


def project_box(box: list[float] | None, H: np.ndarray | None, shape: tuple[int, int]) -> list[float] | None:
    if box is None or H is None:
        return None
    x0, y0, x1, y1 = box
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    h, w = shape
    bx0 = max(0.0, float(np.nanmin(dst[:, 0])))
    by0 = max(0.0, float(np.nanmin(dst[:, 1])))
    bx1 = min(float(w - 1), float(np.nanmax(dst[:, 0])))
    by1 = min(float(h - 1), float(np.nanmax(dst[:, 1])))
    if bx1 <= bx0 or by1 <= by0:
        return None
    return [bx0, by0, bx1, by1]


def load_homographies(path: Path | None) -> dict[str, np.ndarray]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    h = data.get("homographies", data)
    H_rgb_vis = np.asarray(h["rgb_to_vis"], dtype=np.float64)
    H_nir_vis = np.asarray(h["nir_to_vis"], dtype=np.float64)
    H_th_vis = np.asarray(h["thermal_deg_to_vis"], dtype=np.float64)
    return {
        "rgb_to_vis": H_rgb_vis,
        "rgb_to_nir": np.linalg.inv(H_nir_vis) @ H_rgb_vis,
        "rgb_to_thermal_c": np.linalg.inv(H_th_vis) @ H_rgb_vis,
    }


def detect_panel_by_contrast(gray: np.ndarray, projected_box: list[float] | None, sensor: str) -> tuple[list[float] | None, str, float]:
    h, w = gray.shape[:2]
    crops = roi_crops((h, w), projected_box)
    best = None
    for x0, y0, x1, y1, crop_name in crops[:4 if projected_box else len(crops)]:
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        work = clahe(normalize_u8(crop), 3.0, 8)
        if sensor == "thermal_c":
            work = local_contrast(work)
        gx = cv2.Sobel(work, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(work, cv2.CV_32F, 0, 1, ksize=3)
        mag = normalize_u8(np.abs(gx) + np.abs(gy), 5, 99)
        thr = max(45, int(np.percentile(mag, 88)))
        mask = (mag >= thr).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for comp in range(1, n):
            area = int(stats[comp, cv2.CC_STAT_AREA])
            bx = int(stats[comp, cv2.CC_STAT_LEFT])
            by = int(stats[comp, cv2.CC_STAT_TOP])
            bw = int(stats[comp, cv2.CC_STAT_WIDTH])
            bh = int(stats[comp, cv2.CC_STAT_HEIGHT])
            if area < 80 or bw < 25 or bh < 18:
                continue
            aspect = bw / max(1, bh)
            if not (0.7 <= aspect <= 4.5):
                continue
            score = area + 15.0 * min(bw, bh)
            if projected_box:
                pcx = (projected_box[0] + projected_box[2]) * 0.5
                pcy = (projected_box[1] + projected_box[3]) * 0.5
                ccx = x0 + bx + bw * 0.5
                ccy = y0 + by + bh * 0.5
                score -= 0.8 * math.hypot(ccx - pcx, ccy - pcy)
            if best is None or score > best[0]:
                pad = 0.12
                best = (
                    score,
                    [
                        float(max(0, x0 + bx - bw * pad)),
                        float(max(0, y0 + by - bh * pad)),
                        float(min(w - 1, x0 + bx + bw * (1 + pad))),
                        float(min(h - 1, y0 + by + bh * (1 + pad))),
                    ],
                    f"contrast_{crop_name}",
                )
    if best is None:
        if projected_box is not None:
            return projected_box, "projected_rgb_box", 1.0
        return None, "no_panel_candidate", 0.0
    return best[1], best[2], float(best[0])


def detect_sensor(sensor: str, messages: list[tuple[int, object]], ref_stamp: int | None,
                  projected_box: list[float] | None, max_frames: int) -> Detection:
    if not messages:
        return Detection(False, False, "missing_topic", None, None, None, None, None, False, 0.0, None, "missing_topic")
    center = nearest_index(messages, ref_stamp)
    best: Detection | None = None
    for idx in frame_indices(len(messages), center, max_frames):
        stamp, msg = messages[idx]
        img = decode_ros_image(msg)
        if img is None:
            continue
        candidates = gray_candidates(sensor, img)
        search_box = projected_box
        if sensor == "rgb" and search_box is None and candidates:
            hint_box, _hint_method, _hint_score = detect_panel_by_contrast(candidates[0][1], None, sensor)
            search_box = hint_box
        if sensor == "thermal_c":
            thermal_gray = candidates[0][1]
            panel_box, panel_method, panel_score = detect_panel_by_contrast(thermal_gray, search_box, sensor)
            if panel_box is not None:
                return Detection(
                    checker_detected=False,
                    panel_detected=True,
                    method=panel_method,
                    frame_index=idx,
                    stamp_ns=stamp,
                    corners_px=None,
                    panel_box_xyxy=panel_box,
                    orientation_deg=None,
                    rotation_ok=False,
                    quality=panel_score,
                    source_shape=thermal_gray.shape[:2],
                    reason="thermal_panel_only",
                    processed_image=thermal_gray,
                )
            continue
        for variant_name, gray in candidates:
            corners, method = try_checker_on_gray(gray, search_box, sensor)
            if corners is None:
                continue
            corners, angle, quality, rotation_ok = canonicalize_and_score(corners, gray.shape[:2])
            det = Detection(
                checker_detected=True,
                panel_detected=True,
                method=f"{variant_name}_{method}",
                frame_index=idx,
                stamp_ns=stamp,
                corners_px=corners,
                panel_box_xyxy=panel_from_checker(corners),
                orientation_deg=angle,
                rotation_ok=rotation_ok,
                quality=quality,
                source_shape=gray.shape[:2],
                processed_image=gray,
            )
            if best is None or det.quality > best.quality:
                best = det
                # Strong direct detections are good enough; avoid spending time
                # on every band.
                if quality > 14 and rotation_ok:
                    return best
        # If checker failed on this frame, keep a panel fallback from the most
        # promising base image.
        if best is None:
            fallback_gray = candidates[0][1]
            panel_box, panel_method, panel_score = detect_panel_by_contrast(fallback_gray, search_box, sensor)
            if panel_box is not None:
                best = Detection(
                    checker_detected=False,
                    panel_detected=True,
                    method=panel_method,
                    frame_index=idx,
                    stamp_ns=stamp,
                    corners_px=None,
                    panel_box_xyxy=panel_box,
                    orientation_deg=None,
                    rotation_ok=False,
                    quality=panel_score,
                    source_shape=fallback_gray.shape[:2],
                    reason="panel_only",
                    processed_image=fallback_gray,
                )
    if best is not None:
        return best
    return Detection(False, False, "not_found", None, None, None, None, None, False, 0.0, None, "not_found")


def draw_detection(det: Detection, sensor: str, out_path: Path) -> None:
    if det.processed_image is None:
        canvas = np.full((220, 320, 3), 35, dtype=np.uint8)
        cv2.putText(canvas, f"{sensor}: {det.reason}", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 255), 2)
        cv2.imwrite(str(out_path), canvas)
        return
    gray = det.processed_image
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if det.panel_box_xyxy is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in det.panel_box_xyxy]
        cv2.rectangle(color, (x0, y0), (x1, y1), (0, 200, 255), 2)
    if det.corners_px is not None:
        pts = det.corners_px.astype(int)
        for i, p in enumerate(pts):
            col = (0, 255, 0)
            if i == 0:
                col = (0, 0, 255)
            elif i == COLS - 1:
                col = (255, 0, 0)
            elif i == (ROWS - 1) * COLS:
                col = (255, 255, 0)
            cv2.circle(color, tuple(p), 3, col, -1, cv2.LINE_AA)
    h, w = color.shape[:2]
    scale = min(520 / max(h, 1), 720 / max(w, 1), 2.5)
    view = cv2.resize(color, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_NEAREST if scale > 1 else cv2.INTER_AREA)
    text = f"{sensor} checker={det.checker_detected} panel={det.panel_detected} rot_ok={det.rotation_ok} {det.method[:55]}"
    cv2.putText(view, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), view)


def detection_to_json(det: Detection) -> dict[str, Any]:
    def floats(values):
        return [float(v) for v in values] if values is not None else None

    return {
        "checker_detected": det.checker_detected,
        "panel_detected": det.panel_detected,
        "method": det.method,
        "frame_index": det.frame_index,
        "stamp_ns": det.stamp_ns,
        "corners_px": [[float(x), float(y)] for x, y in det.corners_px] if det.corners_px is not None else None,
        "panel_box_xyxy": floats(det.panel_box_xyxy),
        "orientation_deg": float(det.orientation_deg) if det.orientation_deg is not None else None,
        "rotation_ok": det.rotation_ok,
        "quality": float(det.quality),
        "source_shape_hw": list(det.source_shape) if det.source_shape else None,
        "reason": det.reason,
    }


def make_pages(results: list[dict], out: Path) -> None:
    rows = []
    for res in results:
        bag_dir = out / "bags" / Path(res["bag_path"]).stem
        cells = []
        for sensor in ("rgb", "vis", "nir", "thermal_c"):
            p = bag_dir / f"{sensor}_refined_detection.jpg"
            if p.exists():
                img = cv2.imread(str(p))
                img = cv2.resize(img, (310, 210), interpolation=cv2.INTER_AREA)
            else:
                img = np.full((210, 310, 3), 35, dtype=np.uint8)
            cells.append(img)
        strip = np.hstack(cells)
        dets = res["detections"]
        label = (
            f"{res['label_norm']} | "
            f"RGB {dets['rgb']['checker_detected']}/{dets['rgb']['panel_detected']}  "
            f"VIS {dets['vis']['checker_detected']}/{dets['vis']['panel_detected']}  "
            f"NIR {dets['nir']['checker_detected']}/{dets['nir']['panel_detected']}  "
            f"TH {dets['thermal_c']['checker_detected']}/{dets['thermal_c']['panel_detected']}"
        )
        lab = np.full((34, strip.shape[1], 3), 18, dtype=np.uint8)
        cv2.putText(lab, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 1, cv2.LINE_AA)
        rows.append(np.vstack([lab, strip]))
    for page, start in enumerate(range(0, len(rows), 5), start=1):
        cv2.imwrite(str(out / f"refined_detection_page_{page:02d}.jpg"), np.vstack(rows[start:start + 5]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--homographies", type=Path, default=Path("data/matrices/mixed_vis_nir_thermal_homographies.json"))
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--max-frames", type=int, default=9)
    parser.add_argument("--only-label-contains", default="")
    parser.add_argument("--limit-bags", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    H = load_homographies(args.homographies)

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    if not args.include_test:
        rows = [r for r in rows if r.get("height_level") != "test"]
    if args.only_label_contains:
        rows = [r for r in rows if args.only_label_contains.lower() in r.get("label_norm", "").lower()]
    if args.limit_bags > 0:
        rows = rows[: args.limit_bags]

    results = []
    for i, row in enumerate(rows, start=1):
        bag = Path(row["bag_path"])
        print(f"[{i}/{len(rows)}] {row['label_norm']} {bag.name}")
        bag_out = args.out / "bags" / bag.stem
        bag_out.mkdir(parents=True, exist_ok=True)
        messages = read_sampled_messages(bag, TOPICS, row, args.max_frames)

        detections: dict[str, Detection] = {}
        rgb_det = detect_sensor("rgb", messages["rgb"], None, None, args.max_frames)
        detections["rgb"] = rgb_det
        draw_detection(rgb_det, "rgb", bag_out / "rgb_refined_detection.jpg")

        projected: dict[str, list[float] | None] = {"rgb": rgb_det.panel_box_xyxy}
        if rgb_det.panel_box_xyxy:
            # Need representative shapes for target sensors before projecting.
            for sensor in ("vis", "nir", "thermal_c"):
                shape = None
                if messages[sensor]:
                    img0 = decode_ros_image(messages[sensor][nearest_index(messages[sensor], rgb_det.stamp_ns)][1])
                    if img0 is not None:
                        if sensor in ("vis", "nir"):
                            shape = photonfocus_preview(img0).shape[:2]
                        else:
                            shape = img0.shape[:2]
                key = f"rgb_to_{sensor}"
                projected[sensor] = project_box(rgb_det.panel_box_xyxy, H.get(key), shape) if shape else None
        else:
            projected.update({"vis": None, "nir": None, "thermal_c": None})

        for sensor in ("vis", "nir", "thermal_c"):
            det = detect_sensor(sensor, messages[sensor], rgb_det.stamp_ns, projected.get(sensor), args.max_frames)
            detections[sensor] = det
            draw_detection(det, sensor, bag_out / f"{sensor}_refined_detection.jpg")

        result = {
            **row,
            "detections": {sensor: detection_to_json(det) for sensor, det in detections.items()},
            "projected_boxes_from_rgb": projected,
        }
        results.append(result)

    summary = {
        "pattern_internal_corners": list(PATTERN),
        "square_size_m": 0.04,
        "n_bags": len(results),
        "checker_counts": {
            sensor: sum(1 for r in results if r["detections"][sensor]["checker_detected"])
            for sensor in TOPICS
        },
        "panel_counts": {
            sensor: sum(1 for r in results if r["detections"][sensor]["panel_detected"])
            for sensor in TOPICS
        },
        "rotation_warnings": {
            sensor: [
                r["label_norm"] for r in results
                if r["detections"][sensor]["checker_detected"] and not r["detections"][sensor]["rotation_ok"]
            ]
            for sensor in TOPICS
        },
        "results": results,
    }
    (args.out / "refined_multisensor_detection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    flat = []
    for r in results:
        rec = {"bag": Path(r["bag_path"]).name, "label_norm": r["label_norm"], "height_level": r.get("height_level", ""), "pose": r.get("pose", "")}
        for sensor in TOPICS:
            d = r["detections"][sensor]
            rec[f"{sensor}_checker"] = d["checker_detected"]
            rec[f"{sensor}_panel"] = d["panel_detected"]
            rec[f"{sensor}_method"] = d["method"]
            rec[f"{sensor}_rot_ok"] = d["rotation_ok"]
            rec[f"{sensor}_angle_deg"] = d["orientation_deg"]
            rec[f"{sensor}_quality"] = d["quality"]
        flat.append(rec)
    with (args.out / "refined_multisensor_detection_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        writer.writeheader()
        writer.writerows(flat)
    make_pages(results, args.out)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
