#!/usr/bin/env python3
"""Estimate a diagnostic Thermal->VIS homography from guided thermal checkers."""

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
    best_subgrid_pair,
    fit_homography,
    reprojection_metrics,
)


DEFAULT_THERMAL = Path("runs/calibration_20260623_thermal_guided_panel/thermal_guided_detection_summary.json")
DEFAULT_DEEP = Path("runs/calibration_20260623_deep_photonfocus_detection/deep_photonfocus_detection_summary.json")
DEFAULT_HOMOGRAPHIES = Path("data/calibration/new_session/20260623/homographies_20260623_to_vis.json")
DEFAULT_OUT = Path("runs/calibration_20260623_thermal_to_vis_guided_checker")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_vis_deep(path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(path)
    out: dict[str, dict[str, Any]] = {}
    for row in data.get("results", []):
        label = str(row.get("label_norm") or row.get("label_raw") or row.get("bag"))
        det = row.get("detections", {}).get("vis", {})
        if not det.get("checker_detected"):
            continue
        pattern = tuple(int(v) for v in det.get("pattern_internal_corners", []))
        corners = np.asarray(det.get("corners_px"), dtype=np.float64)
        if len(pattern) != 2 or corners.ndim != 2 or corners.shape[1] != 2:
            continue
        out[label] = {"pattern": pattern, "corners": corners, "method": det.get("method", "")}
    return out


def load_thermal_candidates(path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(path)
    out: dict[str, dict[str, Any]] = {}
    for row in data.get("results", []):
        label = str(row.get("label_norm"))
        det = row.get("checker") or {}
        if not det.get("detected"):
            continue
        pattern = tuple(int(v) for v in det.get("pattern_internal_corners", []))
        corners = np.asarray(det.get("corners_px"), dtype=np.float64)
        if len(pattern) != 2 or corners.ndim != 2 or corners.shape[1] != 2:
            continue
        out[label] = {"pattern": pattern, "corners": corners, "method": det.get("method", "")}
    return out


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
    parser.add_argument("--thermal-summary", type=Path, default=DEFAULT_THERMAL)
    parser.add_argument("--deep-summary", type=Path, default=DEFAULT_DEEP)
    parser.add_argument("--homographies", type=Path, default=DEFAULT_HOMOGRAPHIES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-hint-px", type=float, default=35.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hdata = load_json(args.homographies)
    h_hist = np.asarray(hdata["homographies"]["thermal_deg_to_vis"], dtype=np.float64)
    thermal = load_thermal_candidates(args.thermal_summary)
    vis = load_vis_deep(args.deep_summary)

    all_src = []
    all_dst = []
    pairs = []
    rejected = []
    for label in sorted(set(thermal) & set(vis)):
        src = thermal[label]
        dst = vis[label]
        common = (min(src["pattern"][0], dst["pattern"][0]), min(src["pattern"][1], dst["pattern"][1]))
        if common[0] < 7 or common[1] < 4:
            continue
        src_sub, dst_sub, info = best_subgrid_pair(
            src["corners"],
            src["pattern"],
            dst["corners"],
            dst["pattern"],
            common,
            h_hist,
        )
        rec = {
            "label": label,
            "thermal_pattern": list(src["pattern"]),
            "vis_pattern": list(dst["pattern"]),
            "thermal_method": src["method"],
            "vis_method": dst["method"],
            **info,
        }
        if info["hint_median_error_px"] > args.max_hint_px:
            rejected.append({**rec, "reason": "hint_error_too_high"})
            continue
        all_src.append(src_sub)
        all_dst.append(dst_sub)
        pairs.append(rec)

    if not all_src:
        raise SystemExit("No usable Thermal/VIS checker pairs")
    src_pts = np.vstack(all_src)
    dst_pts = np.vstack(all_dst)
    h_new, inliers = fit_homography(src_pts, dst_pts)
    out = {
        "status": "diagnostic_not_active",
        "notes": [
            "Estimated from guided thermal checker candidates.",
            "Do not activate without visual review; thermal checker corners are weak.",
        ],
        "n_pairs": len(pairs),
        "n_points": int(len(src_pts)),
        "inliers": int(inliers.sum()),
        "old_H": h_hist,
        "new_H": h_new,
        "old_metrics_px": reprojection_metrics(src_pts, dst_pts, h_hist),
        "new_metrics_px": reprojection_metrics(src_pts, dst_pts, h_new),
        "pairs": pairs,
        "rejected_pairs": rejected,
    }
    with (args.out_dir / "thermal_to_vis_guided_checker_summary.json").open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(out), f, indent=2)
    print("Pairs:", len(pairs), "points:", len(src_pts), "inliers:", int(inliers.sum()))
    print("old", out["old_metrics_px"])
    print("new", out["new_metrics_px"])
    print("H")
    print(h_new)
    print(f"Wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
