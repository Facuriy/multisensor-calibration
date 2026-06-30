#!/usr/bin/env python3
"""Render final 20260623 multisensor validation sheets in RGB coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_FINAL = Path("data/calibration/new_session/20260623/calibration_20260623_final_candidate.json")
DEFAULT_REFINED = Path(
    "runs/calibration_20260623_refined_multisensor_detection/"
    "refined_multisensor_detection_summary.json"
)
DEFAULT_INPUT = Path("runs/calibration_20260623_full_review/per_bag")
DEFAULT_OUT = Path("runs/calibration_20260623_final_rgb_validation")
DEFAULT_LABELS = [
    "calib_p01_low_center",
    "calib_p05_low_bottom",
    "calib_p10_low_roll_plus",
    "calib_p12_low_tilt",
    "calib_p13_mid_center",
    "calib_p24_mid_tilt",
    "calib_p25_high_center",
    "calib_p34_high_roll_minus",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_by_label(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(r.get("label_norm") or r.get("label_raw") or r.get("bag")): r for r in doc.get("results", [])}


def load_final_h(path: Path) -> dict[str, np.ndarray]:
    doc = load_json(path)
    regs = doc["target_plane_registration_to_rgb"]
    return {
        "vis": np.asarray(regs["vis_to_rgb"]["H_sensor_to_rgb"], dtype=np.float64),
        "nir": np.asarray(regs["nir_to_rgb"]["H_sensor_to_rgb"], dtype=np.float64),
        "thermal": np.asarray(regs["thermal_to_rgb"]["H_sensor_to_rgb"], dtype=np.float64),
    }


def read_bgr(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def robust_gray_bgr(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    lo, hi = np.percentile(gray, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    gray = np.clip((gray.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def warp_to_rgb(img: np.ndarray, h: np.ndarray, rgb_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = rgb_shape
    warped = cv2.warpPerspective(img, h, (width, height), flags=cv2.INTER_LINEAR)
    mask_src = np.full(img.shape[:2], 255, dtype=np.uint8)
    mask = cv2.warpPerspective(mask_src, h, (width, height), flags=cv2.INTER_NEAREST)
    return warped, mask > 0


def edge_overlay(rgb: np.ndarray, warped: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb_g = cv2.cvtColor(robust_gray_bgr(rgb), cv2.COLOR_BGR2GRAY)
    other_g = cv2.cvtColor(robust_gray_bgr(warped), cv2.COLOR_BGR2GRAY)
    rgb_e = cv2.Canny(rgb_g, 60, 160)
    other_e = cv2.Canny(other_g, 50, 150)
    out = np.zeros_like(rgb)
    out[..., 1] = rgb_e
    out[..., 2] = other_e
    out[~mask] = (245, 245, 245)
    return out


def false_color_overlay(rgb: np.ndarray, warped: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb_g = cv2.cvtColor(robust_gray_bgr(rgb), cv2.COLOR_BGR2GRAY)
    other_g = cv2.cvtColor(robust_gray_bgr(warped), cv2.COLOR_BGR2GRAY)
    out = np.zeros_like(rgb)
    out[..., 1] = rgb_g
    out[..., 0] = other_g
    out[..., 2] = other_g
    blended = cv2.addWeighted(rgb, 0.30, out, 0.70, 0)
    blended[~mask] = (245, 245, 245)
    return blended


def fit_tile(img: np.ndarray, size: tuple[int, int], label: str) -> np.ndarray:
    width, height = size
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    h, w = img.shape[:2]
    scale = min(width / max(w, 1), (height - 34) / max(h, 1))
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    x = (width - small.shape[1]) // 2
    y = 34 + (height - 34 - small.shape[0]) // 2
    canvas[y : y + small.shape[0], x : x + small.shape[1]] = small
    cv2.putText(canvas, label[:48], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 2, cv2.LINE_AA)
    return canvas


def draw_mask_contour(img: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = img.copy()
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 3, cv2.LINE_AA)
    return out


def render_one(
    label: str,
    row: dict[str, Any],
    input_dir: Path,
    homographies: dict[str, np.ndarray],
    out_dir: Path,
) -> dict[str, Any]:
    folder = input_dir / Path(str(row.get("bag") or "")).stem
    rgb = read_bgr(folder / "rgb.jpg")
    if rgb is None:
        return {"label": label, "rendered": False, "reason": "missing_rgb", "folder": str(folder)}

    sources = {
        "vis": read_bgr(folder / "vis.jpg"),
        "nir": read_bgr(folder / "nir.jpg"),
        "thermal": read_bgr(folder / "thermal_c.jpg"),
    }
    rgb_shape = rgb.shape[:2]
    warped: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for name, img in sources.items():
        if img is None:
            continue
        warped[name], masks[name] = warp_to_rgb(img, homographies[name], rgb_shape)

    tiles = [fit_tile(rgb, (430, 292), "RGB reference")]
    for name in ("vis", "nir", "thermal"):
        if name not in warped:
            continue
        normal = cv2.addWeighted(rgb, 0.58, warped[name], 0.42, 0)
        normal = draw_mask_contour(normal, masks[name], (0, 255, 255))
        tiles.append(fit_tile(warped[name], (430, 292), f"{name.upper()} warped to RGB"))
        tiles.append(fit_tile(normal, (430, 292), f"RGB + {name.upper()}"))
        tiles.append(fit_tile(false_color_overlay(rgb, warped[name], masks[name]), (430, 292), f"green RGB / magenta {name.upper()}"))
        tiles.append(fit_tile(edge_overlay(rgb, warped[name], masks[name]), (430, 292), f"green RGB edges / red {name.upper()}"))

    if masks:
        common = np.ones(rgb_shape, dtype=bool)
        for mask in masks.values():
            common &= mask
        common_vis = rgb.copy()
        common_vis[~common] = (235, 235, 235)
        tiles.append(fit_tile(draw_mask_contour(common_vis, common, (0, 180, 0)), (430, 292), "common valid area"))
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 245))
    sheet = np.vstack([np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles), 4)])
    cv2.putText(sheet, label, (14, sheet.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (10, 10, 10), 2, cv2.LINE_AA)
    out_path = out_dir / f"{label}_final_rgb_validation.jpg"
    cv2.imwrite(str(out_path), sheet)

    valid_area = {name: float(np.mean(mask)) for name, mask in masks.items()}
    if masks:
        common = np.ones(rgb_shape, dtype=bool)
        for mask in masks.values():
            common &= mask
        valid_area["common"] = float(np.mean(common))
    return {
        "label": label,
        "bag": row.get("bag"),
        "rendered": True,
        "path": str(out_path),
        "valid_area_fraction": valid_area,
        "available_sensors": sorted(warped),
    }


def make_contact(results: list[dict[str, Any]], out_dir: Path) -> list[str]:
    imgs = []
    for r in results:
        if not r.get("rendered"):
            continue
        img = cv2.imread(str(r["path"]), cv2.IMREAD_COLOR)
        if img is not None:
            scale = 1720 / img.shape[1]
            imgs.append(cv2.resize(img, (1720, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA))
    out_paths = []
    for i in range(0, len(imgs), 3):
        page = np.vstack(imgs[i : i + 3])
        path = out_dir / f"final_rgb_validation_contactsheet_{len(out_paths)+1:02d}.jpg"
        cv2.imwrite(str(path), page)
        out_paths.append(str(path))
    return out_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-calibration", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--refined-summary", type=Path, default=DEFAULT_REFINED)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--labels", nargs="*", default=DEFAULT_LABELS)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    homographies = load_final_h(args.final_calibration)
    rows = rows_by_label(load_json(args.refined_summary))
    results = []
    for label in args.labels:
        row = rows.get(label)
        if row is None:
            results.append({"label": label, "rendered": False, "reason": "missing_label"})
            continue
        result = render_one(label, row, args.input_dir, homographies, args.out_dir)
        print(result)
        results.append(result)
    contacts = make_contact(results, args.out_dir)
    summary = {
        "final_calibration": str(args.final_calibration),
        "labels": args.labels,
        "results": results,
        "contactsheets": contacts,
    }
    (args.out_dir / "final_rgb_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
