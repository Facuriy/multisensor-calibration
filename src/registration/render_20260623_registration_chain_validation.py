#!/usr/bin/env python3
"""Render 20260623 RGB/NIR/Thermal -> VIS homography validation sheets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_HOMOGRAPHIES = Path("data/matrices/mixed_vis_nir_thermal_homographies.json")
DEFAULT_REFINED = Path(
    "runs/calibration_20260623_refined_multisensor_detection/"
    "refined_multisensor_detection_summary.json"
)
DEFAULT_INPUT = Path("runs/calibration_20260623_full_review/per_bag")
DEFAULT_OUT = Path("runs/calibration_20260623_registration_chain_validation")
DEFAULT_LABELS = [
    "calib_p01_low_center",
    "calib_p12_low_tilt",
    "calib_p13_mid_center",
    "calib_p24_mid_tilt",
    "calib_p25_high_center",
    "calib_p36_high_tilt",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_homographies(path: Path) -> dict[str, np.ndarray]:
    data = load_json(path)
    raw = data["homographies"]
    return {
        "rgb": np.asarray(raw["rgb_to_vis"], dtype=np.float64),
        "nir": np.asarray(raw["nir_to_vis"], dtype=np.float64),
        "thermal": np.asarray(raw["thermal_deg_to_vis"], dtype=np.float64),
    }


def rows_by_label(refined: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("label_norm") or row.get("label_raw") or row.get("bag")): row
        for row in refined.get("results", [])
    }


def read_bgr(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def warp_to_vis(img: np.ndarray, h: np.ndarray, vis_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = vis_shape
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
    pts = points.reshape(-1, 1, 2).astype(np.float64)
    out = cv2.perspectiveTransform(pts, h).reshape(-1, 2)
    return out


def corner_metrics(row: dict[str, Any], homographies: dict[str, np.ndarray]) -> dict[str, Any]:
    vis = checker_corners(row, "vis")
    metrics: dict[str, Any] = {}
    if vis is None:
        return metrics
    for sensor in ("rgb", "nir"):
        src = checker_corners(row, sensor)
        if src is None or len(src) != len(vis):
            continue
        pred = transform_points(src, homographies[sensor])
        err = np.linalg.norm(pred - vis, axis=1)
        metrics[f"{sensor}_to_vis_checker"] = {
            "n": int(len(err)),
            "mean_px": float(np.mean(err)),
            "median_px": float(np.median(err)),
            "p90_px": float(np.percentile(err, 90)),
            "max_px": float(np.max(err)),
        }
    return metrics


def fit_to_tile(img: np.ndarray, size: tuple[int, int], label: str) -> np.ndarray:
    tile_w, tile_h = size
    canvas = np.full((tile_h, tile_w, 3), 245, dtype=np.uint8)
    h, w = img.shape[:2]
    scale = min(tile_w / max(w, 1), (tile_h - 34) / max(h, 1))
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x = (tile_w - new_w) // 2
    y = 34 + (tile_h - 34 - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = small
    cv2.putText(canvas, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (25, 25, 25), 2, cv2.LINE_AA)
    return canvas


def make_overlay(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cv2.addWeighted(a, 0.55, b, 0.45, 0.0)


def render_one(
    label: str,
    row: dict[str, Any],
    input_dir: Path,
    homographies: dict[str, np.ndarray],
    out_dir: Path,
) -> dict[str, Any]:
    bag = str(row.get("bag") or "")
    bag_stem = Path(bag).stem
    folder = input_dir / bag_stem
    vis = read_bgr(folder / "vis.jpg")
    rgb = read_bgr(folder / "rgb.jpg")
    nir = read_bgr(folder / "nir.jpg")
    thermal = read_bgr(folder / "thermal_c.jpg")
    if vis is None:
        return {"label": label, "rendered": False, "reason": "missing_vis_preview", "bag": bag}

    vis_shape = vis.shape[:2]
    warped: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for name, img in (("rgb", rgb), ("nir", nir), ("thermal", thermal)):
        if img is None:
            continue
        warped[name], masks[name] = warp_to_vis(img, homographies[name], vis_shape)

    tiles = [fit_to_tile(vis, (360, 220), "VIS reference")]
    if "rgb" in warped:
        tiles.append(fit_to_tile(warped["rgb"], (360, 220), "RGB -> VIS"))
    if "nir" in warped:
        tiles.append(fit_to_tile(warped["nir"], (360, 220), "NIR -> VIS"))
    if "thermal" in warped:
        tiles.append(fit_to_tile(warped["thermal"], (360, 220), "Thermal -> VIS"))
    if "rgb" in warped:
        tiles.append(fit_to_tile(make_overlay(vis, warped["rgb"]), (360, 220), "VIS + RGB"))
    if "nir" in warped:
        tiles.append(fit_to_tile(make_overlay(vis, warped["nir"]), (360, 220), "VIS + NIR"))

    while len(tiles) % 3:
        tiles.append(np.full_like(tiles[0], 245))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    sheet = np.vstack(rows)
    cv2.putText(sheet, label, (12, sheet.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)

    out_path = out_dir / f"{label}_registration_chain.jpg"
    cv2.imwrite(str(out_path), sheet)

    valid_area = {}
    for name, mask in masks.items():
        valid_area[name] = float(np.mean(mask))
    if masks:
        common = np.ones(vis_shape, dtype=bool)
        for mask in masks.values():
            common &= mask
        valid_area["common_rgb_nir_thermal"] = float(np.mean(common))

    return {
        "label": label,
        "bag": bag,
        "rendered": True,
        "path": str(out_path),
        "valid_area_fraction": valid_area,
        "checker_metrics_px": corner_metrics(row, homographies),
    }


def make_contactsheet(rendered: list[dict[str, Any]], out_dir: Path) -> Path | None:
    imgs = []
    for item in rendered:
        if not item.get("rendered"):
            continue
        img = cv2.imread(str(item["path"]), cv2.IMREAD_COLOR)
        if img is not None:
            imgs.append(img)
    if not imgs:
        return None
    width = 1200
    resized = []
    for img in imgs:
        scale = width / img.shape[1]
        resized.append(cv2.resize(img, (width, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA))
    sheet = np.vstack(resized)
    out = out_dir / "registration_chain_contactsheet.jpg"
    cv2.imwrite(str(out), sheet)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--homographies", type=Path, default=DEFAULT_HOMOGRAPHIES)
    parser.add_argument("--refined-summary", type=Path, default=DEFAULT_REFINED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--labels", nargs="*", default=DEFAULT_LABELS)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    homographies = load_homographies(args.homographies)
    refined = rows_by_label(load_json(args.refined_summary))

    results = []
    for label in args.labels:
        row = refined.get(label)
        if row is None:
            results.append({"label": label, "rendered": False, "reason": "label_not_found"})
            continue
        result = render_one(label, row, args.input_dir, homographies, args.out_dir)
        results.append(result)
        print(result)

    contact = make_contactsheet(results, args.out_dir)
    summary = {
        "homographies": str(args.homographies),
        "input_dir": str(args.input_dir),
        "labels": args.labels,
        "results": results,
        "contactsheet": str(contact) if contact else None,
    }
    with (args.out_dir / "registration_chain_validation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
