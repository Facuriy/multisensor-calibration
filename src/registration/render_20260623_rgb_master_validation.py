#!/usr/bin/env python3
"""Render RGB-master registration validation sheets for 20260623."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_H = Path("runs/calibration_20260623_rgb_master_homographies/homographies_20260623_to_rgb.json")
DEFAULT_REFINED = Path(
    "runs/calibration_20260623_refined_multisensor_detection/"
    "refined_multisensor_detection_summary.json"
)
DEFAULT_INPUT = Path("runs/calibration_20260623_full_review/per_bag")
DEFAULT_OUT = Path("runs/calibration_20260623_rgb_master_validation")
DEFAULT_LABELS = [
    "calib_p01_low_center",
    "calib_p03_low_right",
    "calib_p07_low_topright",
    "calib_p10_low_roll_plus",
    "calib_p12_low_tilt",
    "calib_p13_mid_center",
    "calib_p19_mid_topright",
    "calib_p25_high_center",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rows_by_label(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(r.get("label_norm") or r.get("label_raw") or r.get("bag")): r for r in doc.get("results", [])}


def read_bgr(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def load_h(path: Path) -> dict[str, np.ndarray]:
    raw = load_json(path)["homographies"]
    return {
        "vis": np.asarray(raw["vis_to_rgb"], dtype=np.float64),
        "nir": np.asarray(raw["nir_to_rgb"], dtype=np.float64),
        "thermal": np.asarray(raw["thermal_deg_to_rgb"], dtype=np.float64),
    }


def warp_to_rgb(img: np.ndarray, h: np.ndarray, rgb_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = rgb_shape
    warped = cv2.warpPerspective(img, h, (width, height), flags=cv2.INTER_LINEAR)
    mask_src = np.full(img.shape[:2], 255, dtype=np.uint8)
    mask = cv2.warpPerspective(mask_src, h, (width, height), flags=cv2.INTER_NEAREST)
    return warped, mask > 0


def checker_corners(row: dict[str, Any], sensor: str) -> np.ndarray | None:
    det = (row.get("detections") or {}).get(sensor) or {}
    corners = det.get("corners_px")
    if corners is None:
        return None
    arr = np.asarray(corners, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return None
    return arr


def transform_points(points: np.ndarray, h: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float64), h).reshape(-1, 2)


def corner_metrics(row: dict[str, Any], homographies: dict[str, np.ndarray]) -> dict[str, Any]:
    rgb = checker_corners(row, "rgb")
    if rgb is None:
        return {}
    out: dict[str, Any] = {}
    for sensor in ("vis", "nir"):
        src = checker_corners(row, sensor)
        if src is None:
            continue
        n = min(len(src), len(rgb))
        if n <= 0:
            continue
        pred = transform_points(src[:n], homographies[sensor])
        err = np.linalg.norm(pred - rgb[:n], axis=1)
        out[f"{sensor}_to_rgb_first_n"] = {
            "n": int(len(err)),
            "mean_px": float(np.mean(err)),
            "median_px": float(np.median(err)),
            "p90_px": float(np.percentile(err, 90)),
            "max_px": float(np.max(err)),
            "note": "first-n order metric; authoritative metrics are in homographies_20260623_to_rgb.json",
        }
    return out


def fit_tile(img: np.ndarray, size: tuple[int, int], label: str) -> np.ndarray:
    width, height = size
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    h, w = img.shape[:2]
    scale = min(width / max(w, 1), (height - 32) / max(h, 1))
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    x = (width - small.shape[1]) // 2
    y = 32 + (height - 32 - small.shape[0]) // 2
    canvas[y : y + small.shape[0], x : x + small.shape[1]] = small
    cv2.putText(canvas, label[:42], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 2, cv2.LINE_AA)
    return canvas


def overlay_rgb_ref(rgb: np.ndarray, other: np.ndarray) -> np.ndarray:
    return cv2.addWeighted(rgb, 0.62, other, 0.38, 0.0)


def render_one(label: str, row: dict[str, Any], input_dir: Path, homographies: dict[str, np.ndarray], out_dir: Path) -> dict[str, Any]:
    folder = input_dir / Path(str(row.get("bag") or "")).stem
    rgb = read_bgr(folder / "rgb.jpg")
    vis = read_bgr(folder / "vis.jpg")
    nir = read_bgr(folder / "nir.jpg")
    thermal = read_bgr(folder / "thermal_c.jpg")
    if rgb is None:
        return {"label": label, "rendered": False, "reason": "missing_rgb"}

    rgb_shape = rgb.shape[:2]
    warped: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for name, img in (("vis", vis), ("nir", nir), ("thermal", thermal)):
        if img is None:
            continue
        warped[name], masks[name] = warp_to_rgb(img, homographies[name], rgb_shape)

    tiles = [fit_tile(rgb, (420, 280), "RGB reference")]
    for name, title in (("vis", "VIS -> RGB"), ("nir", "NIR -> RGB"), ("thermal", "Thermal -> RGB")):
        if name in warped:
            tiles.append(fit_tile(warped[name], (420, 280), title))
    for name, title in (("vis", "RGB + VIS"), ("nir", "RGB + NIR"), ("thermal", "RGB + Thermal")):
        if name in warped:
            tiles.append(fit_tile(overlay_rgb_ref(rgb, warped[name]), (420, 280), title))
    while len(tiles) % 3:
        tiles.append(np.full_like(tiles[0], 245))
    sheet = np.vstack([np.hstack(tiles[i : i + 3]) for i in range(0, len(tiles), 3)])
    cv2.putText(sheet, label, (12, sheet.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
    out_path = out_dir / f"{label}_rgb_master.jpg"
    cv2.imwrite(str(out_path), sheet)

    valid_area = {}
    for name, mask in masks.items():
        valid_area[name] = float(np.mean(mask))
    if masks:
        common = np.ones(rgb_shape, dtype=bool)
        for mask in masks.values():
            common &= mask
        valid_area["common_vis_nir_thermal"] = float(np.mean(common))

    return {
        "label": label,
        "bag": row.get("bag"),
        "rendered": True,
        "path": str(out_path),
        "valid_area_fraction": valid_area,
        "corner_metrics_px": corner_metrics(row, homographies),
    }


def make_contact(results: list[dict[str, Any]], out_dir: Path) -> str | None:
    imgs = []
    for r in results:
        if not r.get("rendered"):
            continue
        img = cv2.imread(str(r["path"]), cv2.IMREAD_COLOR)
        if img is not None:
            imgs.append(cv2.resize(img, (1260, int(img.shape[0] * (1260 / img.shape[1]))), interpolation=cv2.INTER_AREA))
    if not imgs:
        return None
    page_paths = []
    for i in range(0, len(imgs), 4):
        page = np.vstack(imgs[i : i + 4])
        p = out_dir / f"rgb_master_contactsheet_{len(page_paths)+1:02d}.jpg"
        cv2.imwrite(str(p), page)
        page_paths.append(str(p))
    return page_paths[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--homographies", type=Path, default=DEFAULT_H)
    parser.add_argument("--refined-summary", type=Path, default=DEFAULT_REFINED)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--labels", nargs="*", default=DEFAULT_LABELS)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    homographies = load_h(args.homographies)
    rows = rows_by_label(load_json(args.refined_summary))
    results = []
    for label in args.labels:
        row = rows.get(label)
        if row is None:
            results.append({"label": label, "rendered": False, "reason": "missing_label"})
            continue
        result = render_one(label, row, args.input_dir, homographies, args.out_dir)
        results.append(result)
        print(result)
    contact = make_contact(results, args.out_dir)
    summary = {
        "homographies": str(args.homographies),
        "results": results,
        "contactsheet": contact,
    }
    with (args.out_dir / "rgb_master_validation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

