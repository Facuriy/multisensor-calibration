#!/usr/bin/env python3
"""Build RGB-master 2D registration homographies for the 20260623 session.

The previous pragmatic 2D chain used VIS as the reference image. This script
re-expresses the target-plane homographies with RGB as the reference:

    VIS/NIR/Thermal -> RGB

These are still planar homographies, not full physical 3D extrinsics.
"""

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

from src.registration.refine_20260623_vis_homographies import (  # noqa: E402
    RGB_PATTERN,
    best_subgrid_pair,
    fit_homography,
    load_deep_by_label,
    load_rgb_by_label,
    reprojection_metrics,
)


DEFAULT_SCAN = Path("runs/calibration_20260623_checker_lidar_scan/scan_summary.json")
DEFAULT_REFINED = Path(
    "runs/calibration_20260623_refined_multisensor_detection/"
    "refined_multisensor_detection_summary.json"
)
DEFAULT_DEEP = Path("runs/calibration_20260623_deep_photonfocus_detection/deep_photonfocus_detection_summary.json")
DEFAULT_VIS_H = Path("data/calibration/new_session/20260623/homographies_20260623_to_vis.json")
DEFAULT_OUT = Path("runs/calibration_20260623_rgb_master_homographies")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def transform_points(points: np.ndarray, h: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float64), h).reshape(-1, 2)


def invert_h(h: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(h)
    return inv / inv[2, 2]


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    h = a @ b
    return h / h[2, 2]


def direct_sensor_to_rgb(
    sensor: str,
    rgb_by_label: dict[str, np.ndarray],
    deep: dict[str, dict[str, dict[str, Any]]],
    h_hint: np.ndarray,
    max_hint_px: float,
    min_common: tuple[int, int] = (6, 4),
) -> dict[str, Any]:
    all_src = []
    all_dst = []
    pairs = []
    rejected = []
    for label, sensors in sorted(deep.items()):
        if sensor not in sensors or label not in rgb_by_label:
            continue
        src = sensors[sensor]["corners"]
        src_pattern = sensors[sensor]["pattern"]
        dst = rgb_by_label[label]
        common = (min(src_pattern[0], RGB_PATTERN[0]), min(src_pattern[1], RGB_PATTERN[1]))
        if common[0] < min_common[0] or common[1] < min_common[1]:
            continue
        src_sub, dst_sub, info = best_subgrid_pair(src, src_pattern, dst, RGB_PATTERN, common, h_hint)
        rec = {
            "label": label,
            "source_pattern": list(src_pattern),
            "target_pattern": list(RGB_PATTERN),
            **info,
        }
        if info["hint_median_error_px"] > max_hint_px:
            rejected.append({**rec, "reason": "hint_error_too_high"})
            continue
        all_src.append(src_sub)
        all_dst.append(dst_sub)
        pairs.append(rec)

    if not all_src:
        return {
            "H_direct": h_hint,
            "n_pairs": 0,
            "n_points": 0,
            "inliers": 0,
            "hint_metrics_px": None,
            "direct_metrics_px": None,
            "pairs": [],
            "rejected_pairs": rejected,
            "status": "composed_only_no_direct_pairs",
        }
    src_pts = np.vstack(all_src)
    dst_pts = np.vstack(all_dst)
    h_direct, inliers = fit_homography(src_pts, dst_pts)
    return {
        "H_direct": h_direct,
        "n_pairs": len(pairs),
        "n_points": int(len(src_pts)),
        "inliers": int(inliers.sum()),
        "hint_metrics_px": reprojection_metrics(src_pts, dst_pts, h_hint),
        "direct_metrics_px": reprojection_metrics(src_pts, dst_pts, h_direct),
        "pairs": pairs,
        "rejected_pairs": rejected,
        "status": "direct_fit",
    }


def select_homography(composed: np.ndarray, direct_result: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    hint = direct_result.get("hint_metrics_px")
    direct = direct_result.get("direct_metrics_px")
    if not hint or not direct:
        return composed, {"selected": "composed", "reason": "no_direct_validation"}
    hint_med = float(hint["median_px"])
    direct_med = float(direct["median_px"])
    if direct_med <= hint_med:
        return np.asarray(direct_result["H_direct"], dtype=np.float64), {
            "selected": "direct",
            "reason": "direct_median_lte_composed",
            "composed_median_px": hint_med,
            "direct_median_px": direct_med,
        }
    return composed, {
        "selected": "composed",
        "reason": "composed_median_better",
        "composed_median_px": hint_med,
        "direct_median_px": direct_med,
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
    parser.add_argument("--vis-homographies", type=Path, default=DEFAULT_VIS_H)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-vis-hint-px", type=float, default=45.0)
    parser.add_argument("--max-nir-hint-px", type=float, default=60.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    vis_doc = load_json(args.vis_homographies)
    h = vis_doc["homographies"]
    rgb_to_vis = np.asarray(h["rgb_to_vis"], dtype=np.float64)
    nir_to_vis = np.asarray(h["nir_to_vis"], dtype=np.float64)
    thermal_to_vis = np.asarray(h["thermal_deg_to_vis"], dtype=np.float64)
    thermal_mono_to_vis = np.asarray(h.get("thermal_mono16_to_vis", h["thermal_deg_to_vis"]), dtype=np.float64)

    vis_to_rgb_composed = invert_h(rgb_to_vis)
    nir_to_rgb_composed = compose(vis_to_rgb_composed, nir_to_vis)
    thermal_to_rgb_composed = compose(vis_to_rgb_composed, thermal_to_vis)
    thermal_mono_to_rgb_composed = compose(vis_to_rgb_composed, thermal_mono_to_vis)

    rgb_by_label = load_rgb_by_label(args.scan_summary, args.refined_summary)
    deep = load_deep_by_label(args.deep_summary)

    vis_direct = direct_sensor_to_rgb(
        "vis",
        rgb_by_label,
        deep,
        vis_to_rgb_composed,
        args.max_vis_hint_px,
        min_common=(7, 4),
    )
    nir_direct = direct_sensor_to_rgb(
        "nir",
        rgb_by_label,
        deep,
        nir_to_rgb_composed,
        args.max_nir_hint_px,
        min_common=(6, 4),
    )
    vis_selected, vis_selection = select_homography(vis_to_rgb_composed, vis_direct)
    nir_selected, nir_selection = select_homography(nir_to_rgb_composed, nir_direct)

    out = {
        "reference": "rgb",
        "status": "candidate_rgb_master_target_plane_20260623",
        "notes": [
            "RGB is the master 2D reference for pragmatic pixel registration.",
            "VIS->RGB and NIR->RGB include direct checker-based fits when enough pairs pass filters.",
            "Thermal->RGB is composed from historical Thermal->VIS and 20260623 VIS->RGB.",
            "These are target-plane homographies, not full physical 3D extrinsics.",
            "Ouster->RGB remains the physical 3D extrinsic in active_calibration.json.",
        ],
        "image_sizes_wh": {
            "rgb": [2448, 2048],
            "vis_preview": [512, 256],
            "nir_preview": [409, 217],
            "thermal": [640, 512],
        },
        "homographies": {
            "rgb": np.eye(3, dtype=np.float64),
            "vis_to_rgb_composed": vis_to_rgb_composed,
            "vis_to_rgb_direct": vis_direct["H_direct"],
            "vis_to_rgb": vis_selected,
            "nir_to_rgb_composed": nir_to_rgb_composed,
            "nir_to_rgb_direct": nir_direct["H_direct"],
            "nir_to_rgb": nir_selected,
            "thermal_deg_to_rgb": thermal_to_rgb_composed,
            "thermal_mono16_to_rgb": thermal_mono_to_rgb_composed,
        },
        "metrics": {
            "vis_to_rgb": {**{k: v for k, v in vis_direct.items() if k != "H_direct"}, "selection": vis_selection},
            "nir_to_rgb": {**{k: v for k, v in nir_direct.items() if k != "H_direct"}, "selection": nir_selection},
        },
        "source_vis_homographies": str(args.vis_homographies),
    }

    out_path = args.out_dir / "homographies_20260623_to_rgb.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(out), f, indent=2)

    print("VIS->RGB", out["metrics"]["vis_to_rgb"])
    print("NIR->RGB", out["metrics"]["nir_to_rgb"])
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
