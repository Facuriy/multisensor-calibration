#!/usr/bin/env python3
"""Refine target-plane homographies to VIS from 20260623 checker detections."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.calibrate_rgb_intrinsics_from_detections import (  # noqa: E402
    choose_one_per_label,
    collect_refined,
    collect_scan,
)


DEFAULT_SCAN = Path("runs/calibration_20260623_checker_lidar_scan/scan_summary.json")
DEFAULT_REFINED = Path(
    "runs/calibration_20260623_refined_multisensor_detection/"
    "refined_multisensor_detection_summary.json"
)
DEFAULT_DEEP = Path("runs/calibration_20260623_deep_photonfocus_detection/deep_photonfocus_detection_summary.json")
DEFAULT_BASE = Path("data/matrices/mixed_vis_nir_thermal_homographies.json")
DEFAULT_OUT = Path("runs/calibration_20260623_refined_vis_homographies")

RGB_PATTERN = (9, 6)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_base_homographies(path: Path) -> dict[str, np.ndarray]:
    data = load_json(path)
    h = data["homographies"]
    return {
        "rgb_to_vis": np.asarray(h["rgb_to_vis"], dtype=np.float64),
        "nir_to_vis": np.asarray(h["nir_to_vis"], dtype=np.float64),
        "thermal_deg_to_vis": np.asarray(h["thermal_deg_to_vis"], dtype=np.float64),
        "thermal_mono16_to_vis": np.asarray(h["thermal_mono16_to_vis"], dtype=np.float64),
    }


def load_rgb_by_label(scan_path: Path, refined_path: Path) -> dict[str, np.ndarray]:
    detections = choose_one_per_label(collect_scan(scan_path, RGB_PATTERN) + collect_refined(refined_path, RGB_PATTERN))
    return {d.label: d.corners.reshape(-1, 2).astype(np.float64) for d in detections}


def load_deep_by_label(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    data = load_json(path)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for row in data.get("results", []):
        label = str(row.get("label_norm") or row.get("label_raw") or row.get("bag"))
        out[label] = {}
        for sensor in ("vis", "nir"):
            det = row.get("detections", {}).get(sensor, {})
            if not det.get("checker_detected"):
                continue
            pattern = det.get("pattern_internal_corners")
            corners = det.get("corners_px")
            if not pattern or corners is None:
                continue
            arr = np.asarray(corners, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != 2:
                continue
            if len(arr) != int(pattern[0]) * int(pattern[1]):
                continue
            out[label][sensor] = {
                "pattern": (int(pattern[0]), int(pattern[1])),
                "corners": arr,
                "method": det.get("method", ""),
            }
    return out


def subgrid(points: np.ndarray, full_pattern: tuple[int, int], sub_pattern: tuple[int, int], col0: int, row0: int) -> np.ndarray:
    full_cols, _ = full_pattern
    sub_cols, sub_rows = sub_pattern
    idx = []
    for row in range(sub_rows):
        for col in range(sub_cols):
            idx.append((row0 + row) * full_cols + (col0 + col))
    return points[np.asarray(idx, dtype=int)]


def transform_points(points: np.ndarray, h: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float64), h).reshape(-1, 2)


def best_subgrid_pair(
    source: np.ndarray,
    source_pattern: tuple[int, int],
    target: np.ndarray,
    target_pattern: tuple[int, int],
    common_pattern: tuple[int, int],
    h_hint: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    src_cols, src_rows = source_pattern
    dst_cols, dst_rows = target_pattern
    c_cols, c_rows = common_pattern
    best = None
    for s_row in range(src_rows - c_rows + 1):
        for s_col in range(src_cols - c_cols + 1):
            src_sub = subgrid(source, source_pattern, common_pattern, s_col, s_row)
            pred = transform_points(src_sub, h_hint)
            for d_row in range(dst_rows - c_rows + 1):
                for d_col in range(dst_cols - c_cols + 1):
                    dst_sub = subgrid(target, target_pattern, common_pattern, d_col, d_row)
                    err = np.linalg.norm(pred - dst_sub, axis=1)
                    score = float(np.median(err))
                    if best is None or score < best[0]:
                        best = (score, src_sub, dst_sub, s_col, s_row, d_col, d_row)
    if best is None:
        raise ValueError("No subgrid candidate")
    score, src_sub, dst_sub, s_col, s_row, d_col, d_row = best
    return src_sub, dst_sub, {
        "hint_median_error_px": score,
        "source_offset_col_row": [int(s_col), int(s_row)],
        "target_offset_col_row": [int(d_col), int(d_row)],
        "common_pattern": list(common_pattern),
    }


def reprojection_metrics(src: np.ndarray, dst: np.ndarray, h: np.ndarray) -> dict[str, float]:
    pred = transform_points(src, h)
    err = np.linalg.norm(pred - dst, axis=1)
    return {
        "n": int(len(err)),
        "mean_px": float(np.mean(err)),
        "median_px": float(np.median(err)),
        "p90_px": float(np.percentile(err, 90)),
        "max_px": float(np.max(err)),
    }


def fit_homography(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, mask = cv2.findHomography(src.astype(np.float64), dst.astype(np.float64), cv2.RANSAC, 3.0)
    if h is None:
        h, mask = cv2.findHomography(src.astype(np.float64), dst.astype(np.float64), 0)
    if h is None:
        raise RuntimeError("findHomography failed")
    h = h / h[2, 2]
    if mask is None:
        mask = np.ones((len(src), 1), dtype=np.uint8)
    return h, mask.reshape(-1).astype(bool)


def refine_rgb_to_vis(
    rgb_by_label: dict[str, np.ndarray],
    deep: dict[str, dict[str, dict[str, Any]]],
    h_hint: np.ndarray,
    max_hint_px: float,
) -> dict[str, Any]:
    all_src = []
    all_dst = []
    pairs = []
    rejected = []
    for label, sensors in sorted(deep.items()):
        if "vis" not in sensors or label not in rgb_by_label:
            continue
        target = sensors["vis"]["corners"]
        target_pattern = sensors["vis"]["pattern"]
        common_pattern = target_pattern
        src_sub, dst_sub, info = best_subgrid_pair(
            rgb_by_label[label],
            RGB_PATTERN,
            target,
            target_pattern,
            common_pattern,
            h_hint,
        )
        if info["hint_median_error_px"] > max_hint_px:
            rejected.append({"label": label, "reason": "hint_error_too_high", **info})
            continue
        all_src.append(src_sub)
        all_dst.append(dst_sub)
        pairs.append({"label": label, "target_pattern": list(target_pattern), **info})
    if not all_src:
        raise RuntimeError("No RGB->VIS checker pairs")
    src = np.vstack(all_src)
    dst = np.vstack(all_dst)
    h, inliers = fit_homography(src, dst)
    return {
        "H": h,
        "n_pairs": len(pairs),
        "n_points": int(len(src)),
        "inliers": int(inliers.sum()),
        "old_metrics_px": reprojection_metrics(src, dst, h_hint),
        "new_metrics_px": reprojection_metrics(src, dst, h),
        "pairs": pairs,
        "rejected_pairs": rejected,
    }


def refine_nir_to_vis(deep: dict[str, dict[str, dict[str, Any]]], h_hint: np.ndarray, max_hint_px: float) -> dict[str, Any]:
    all_src = []
    all_dst = []
    pairs = []
    rejected = []
    for label, sensors in sorted(deep.items()):
        if "vis" not in sensors or "nir" not in sensors:
            continue
        source = sensors["nir"]["corners"]
        target = sensors["vis"]["corners"]
        source_pattern = sensors["nir"]["pattern"]
        target_pattern = sensors["vis"]["pattern"]
        common_pattern = (min(source_pattern[0], target_pattern[0]), min(source_pattern[1], target_pattern[1]))
        if common_pattern[0] < 6 or common_pattern[1] < 4:
            continue
        src_sub, dst_sub, info = best_subgrid_pair(
            source,
            source_pattern,
            target,
            target_pattern,
            common_pattern,
            h_hint,
        )
        if info["hint_median_error_px"] > max_hint_px:
            rejected.append({
                "label": label,
                "reason": "hint_error_too_high",
                "source_pattern": list(source_pattern),
                "target_pattern": list(target_pattern),
                **info,
            })
            continue
        all_src.append(src_sub)
        all_dst.append(dst_sub)
        pairs.append({
            "label": label,
            "source_pattern": list(source_pattern),
            "target_pattern": list(target_pattern),
            **info,
        })
    if not all_src:
        raise RuntimeError("No NIR->VIS checker pairs")
    src = np.vstack(all_src)
    dst = np.vstack(all_dst)
    h, inliers = fit_homography(src, dst)
    return {
        "H": h,
        "n_pairs": len(pairs),
        "n_points": int(len(src)),
        "inliers": int(inliers.sum()),
        "old_metrics_px": reprojection_metrics(src, dst, h_hint),
        "new_metrics_px": reprojection_metrics(src, dst, h),
        "pairs": pairs,
        "rejected_pairs": rejected,
    }


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-summary", type=Path, default=DEFAULT_SCAN)
    parser.add_argument("--refined-summary", type=Path, default=DEFAULT_REFINED)
    parser.add_argument("--deep-summary", type=Path, default=DEFAULT_DEEP)
    parser.add_argument("--base-homographies", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-rgb-hint-px", type=float, default=30.0)
    parser.add_argument("--max-nir-hint-px", type=float, default=20.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = load_base_homographies(args.base_homographies)
    rgb = load_rgb_by_label(args.scan_summary, args.refined_summary)
    deep = load_deep_by_label(args.deep_summary)

    rgb_to_vis = refine_rgb_to_vis(rgb, deep, base["rgb_to_vis"], args.max_rgb_hint_px)
    nir_to_vis = refine_nir_to_vis(deep, base["nir_to_vis"], args.max_nir_hint_px)

    output = {
        "reference": "vis",
        "status": "candidate_target_plane_20260623",
        "notes": [
            "Refined from 20260623 checker detections.",
            "RGB->VIS uses RGB 9x6 detections and VIS 9x6/9x5/8x5 deep detections.",
            "NIR->VIS uses NIR and VIS deep detections with matching subgrids.",
            "Thermal->VIS is kept from historical aluminum-panel homography.",
            "These are 2D target-plane homographies, not full physical 3D extrinsics.",
        ],
        "homographies": {
            "vis": np.eye(3, dtype=np.float64),
            "rgb_to_vis": rgb_to_vis["H"],
            "nir_to_vis": nir_to_vis["H"],
            "thermal_deg_to_vis": base["thermal_deg_to_vis"],
            "thermal_mono16_to_vis": base["thermal_mono16_to_vis"],
            "ouster_depth_to_vis": None,
        },
        "metrics": {
            "rgb_to_vis": {k: v for k, v in rgb_to_vis.items() if k != "H"},
            "nir_to_vis": {k: v for k, v in nir_to_vis.items() if k != "H"},
        },
        "base_homographies": str(args.base_homographies),
        "filters": {
            "max_rgb_hint_px": args.max_rgb_hint_px,
            "max_nir_hint_px": args.max_nir_hint_px,
        },
    }
    with (args.out_dir / "homographies_20260623_to_vis.json").open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(output), f, indent=2)

    print("RGB->VIS")
    print(" old", rgb_to_vis["old_metrics_px"])
    print(" new", rgb_to_vis["new_metrics_px"])
    print(" H")
    print(rgb_to_vis["H"])
    print("NIR->VIS")
    print(" old", nir_to_vis["old_metrics_px"])
    print(" new", nir_to_vis["new_metrics_px"])
    print(" H")
    print(nir_to_vis["H"])
    print(f"Wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
