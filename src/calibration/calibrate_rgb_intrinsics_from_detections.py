#!/usr/bin/env python3
"""Calibrate RGB intrinsics from saved 20260623 checker detections."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_REFINED = Path(
    "runs/calibration_20260623_refined_multisensor_detection/"
    "refined_multisensor_detection_summary.json"
)
DEFAULT_SCAN = Path(
    "runs/calibration_20260623_checker_lidar_scan/scan_summary.json"
)
DEFAULT_OUT = Path("runs/calibration_20260623_rgb_intrinsics_from_detections")


@dataclass
class ViewDetection:
    label: str
    bag: str
    source: str
    frame_index: int | None
    stamp_ns: int | None
    corners: np.ndarray
    source_shape_hw: tuple[int, int] | None
    method: str


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_corners(raw: Any, n_expected: int) -> np.ndarray | None:
    arr = np.asarray(raw, dtype=np.float32)
    if arr.shape != (n_expected, 2):
        return None
    if not np.isfinite(arr).all():
        return None
    return arr.reshape(-1, 1, 2)


def parse_shape_hw(raw: Any) -> tuple[int, int] | None:
    if raw is None:
        return None
    if len(raw) != 2:
        return None
    h, w = int(raw[0]), int(raw[1])
    if h <= 0 or w <= 0:
        return None
    return h, w


def collect_refined(path: Path, pattern: tuple[int, int]) -> list[ViewDetection]:
    if not path.exists():
        return []
    data = load_json(path)
    n_expected = pattern[0] * pattern[1]
    detections: list[ViewDetection] = []
    for row in data.get("results", []):
        det = row.get("detections", {}).get("rgb", {})
        if not det.get("checker_detected", False):
            continue
        corners = normalize_corners(det.get("corners_px"), n_expected)
        if corners is None:
            continue
        detections.append(
            ViewDetection(
                label=str(row.get("label_norm") or row.get("label_raw") or row.get("bag")),
                bag=str(row.get("bag") or ""),
                source="refined",
                frame_index=det.get("frame_index"),
                stamp_ns=det.get("stamp_ns"),
                corners=corners,
                source_shape_hw=parse_shape_hw(det.get("source_shape_hw")),
                method=str(det.get("method") or ""),
            )
        )
    return detections


def collect_scan(path: Path, pattern: tuple[int, int]) -> list[ViewDetection]:
    if not path.exists():
        return []
    data = load_json(path)
    n_expected = pattern[0] * pattern[1]
    detections: list[ViewDetection] = []
    for row in data.get("results", []):
        det = row.get("camera_detections", {}).get("rgb", {})
        if not det.get("detected", False):
            continue
        corners = normalize_corners(det.get("corners_px"), n_expected)
        if corners is None:
            continue
        detections.append(
            ViewDetection(
                label=str(row.get("label_norm") or row.get("label_raw") or row.get("bag")),
                bag=str(row.get("bag") or ""),
                source="scan",
                frame_index=det.get("frame_index"),
                stamp_ns=det.get("stamp_ns"),
                corners=corners,
                source_shape_hw=parse_shape_hw(det.get("source_shape_hw")),
                method="scan_summary",
            )
        )
    return detections


def choose_one_per_label(candidates: list[ViewDetection]) -> list[ViewDetection]:
    grouped: dict[str, list[ViewDetection]] = {}
    for det in candidates:
        grouped.setdefault(det.label, []).append(det)

    selected: list[ViewDetection] = []
    for label, items in sorted(grouped.items()):
        # The first scan was the most exhaustive for RGB. Use refined only to fill gaps.
        items = sorted(
            items,
            key=lambda d: (
                0 if d.source == "scan" else 1,
                -board_area_px(d.corners),
                d.frame_index if d.frame_index is not None else 10**9,
            ),
        )
        selected.append(items[0])
    return selected


def board_area_px(corners: np.ndarray) -> float:
    pts = corners.reshape(-1, 2)
    hull = cv2.convexHull(pts.astype(np.float32))
    return float(cv2.contourArea(hull))


def make_object_points(pattern: tuple[int, int], square_size_m: float) -> np.ndarray:
    cols, rows = pattern
    obj = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj[:, :2] = grid * float(square_size_m)
    return obj


def infer_image_size(detections: list[ViewDetection]) -> tuple[int, int]:
    shapes = [d.source_shape_hw for d in detections if d.source_shape_hw is not None]
    if shapes:
        h, w = max(set(shapes), key=shapes.count)
        return w, h

    all_pts = np.concatenate([d.corners.reshape(-1, 2) for d in detections], axis=0)
    max_x, max_y = np.nanmax(all_pts, axis=0)
    # RGB camera in this dataset is 2448 x 2048. Keep this fallback explicit.
    if max_x <= 2448 and max_y <= 2048:
        return 2448, 2048
    return int(np.ceil(max_x + 1)), int(np.ceil(max_y + 1))


def calibrate(
    detections: list[ViewDetection],
    pattern: tuple[int, int],
    square_size_m: float,
    image_size_wh: tuple[int, int],
    flags: int,
    k0: np.ndarray | None = None,
    d0: np.ndarray | None = None,
) -> dict[str, Any]:
    obj = make_object_points(pattern, square_size_m)
    objpoints = [obj.copy() for _ in detections]
    imgpoints = [d.corners.astype(np.float32) for d in detections]

    w, h = image_size_wh
    if k0 is None:
        fx0 = fy0 = 0.5 * (w + h) * 1.6
        k0 = np.array([[fx0, 0.0, w / 2.0], [0.0, fy0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    if d0 is None:
        d0 = np.zeros((5, 1), dtype=np.float64)

    rms, k, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size_wh, k0, d0, flags=flags
    )
    errors = reprojection_errors(objpoints, imgpoints, rvecs, tvecs, k, dist)
    return {
        "rms": float(rms),
        "K": k,
        "dist": dist,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "errors": errors,
    }


def initial_camera_matrix(image_size_wh: tuple[int, int], focal_px: float = 3623.1884057971015) -> np.ndarray:
    w, h = image_size_wh
    return np.array(
        [[focal_px, 0.0, w / 2.0], [0.0, focal_px, h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def run_model(
    name: str,
    detections: list[ViewDetection],
    pattern: tuple[int, int],
    square_size_m: float,
    image_size_wh: tuple[int, int],
    flags: int,
    min_views: int,
) -> dict[str, Any]:
    k0 = initial_camera_matrix(image_size_wh)
    d0 = np.zeros((5, 1), dtype=np.float64)
    initial = calibrate(detections, pattern, square_size_m, image_size_wh, flags, k0.copy(), d0.copy())
    keep = robust_keep_mask(initial["errors"], min(min_views, len(detections)))
    final_detections = [d for d, used in zip(detections, keep) if used]
    final = calibrate(final_detections, pattern, square_size_m, image_size_wh, flags, k0.copy(), d0.copy())

    final_errors_full = list(initial["errors"])
    for idx, err in zip(np.flatnonzero(keep), final["errors"]):
        final_errors_full[int(idx)] = err

    return {
        "name": name,
        "flags": int(flags),
        "initial": initial,
        "final": final,
        "keep": keep,
        "final_errors_full": final_errors_full,
        "final_detections": final_detections,
        "outliers": [d.label for d, used in zip(detections, keep) if not used],
    }


def model_summary(model: dict[str, Any]) -> dict[str, Any]:
    final = model["final"]
    errors = final["errors"]
    k = final["K"]
    dist = final["dist"].reshape(-1)
    return {
        "name": model["name"],
        "flags": model["flags"],
        "n_used_final": len(model["final_detections"]),
        "outlier_labels": model["outliers"],
        "rms_px": float(final["rms"]),
        "mean_view_error_px": float(np.mean([e["mean_px"] for e in errors])),
        "median_view_error_px": float(np.median([e["mean_px"] for e in errors])),
        "max_view_error_px": float(np.max([e["mean_px"] for e in errors])),
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
        "dist_coeffs": dist,
        "K": k,
    }


def reprojection_errors(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    k: np.ndarray,
    dist: np.ndarray,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for obj, img, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, rvec, tvec, k, dist)
        delta = np.linalg.norm(proj.reshape(-1, 2) - img.reshape(-1, 2), axis=1)
        rows.append(
            {
                "mean_px": float(np.mean(delta)),
                "median_px": float(np.median(delta)),
                "max_px": float(np.max(delta)),
                "std_px": float(np.std(delta)),
            }
        )
    return rows


def robust_keep_mask(errors: list[dict[str, float]], min_views: int) -> np.ndarray:
    mean_err = np.asarray([e["mean_px"] for e in errors], dtype=np.float64)
    med = float(np.median(mean_err))
    mad = float(np.median(np.abs(mean_err - med)))
    sigma = 1.4826 * mad if mad > 1e-9 else float(np.std(mean_err))
    threshold = max(3.0, med + 3.0 * sigma, 3.0 * med)
    keep = mean_err <= threshold
    if int(keep.sum()) < min_views:
        order = np.argsort(mean_err)
        keep[:] = False
        keep[order[:min_views]] = True
    return keep


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    return value


def write_views_csv(path: Path, detections: list[ViewDetection], errors: list[dict[str, float]], keep: np.ndarray) -> None:
    fields = [
        "label",
        "bag",
        "source",
        "frame_index",
        "stamp_ns",
        "method",
        "board_area_px",
        "used_final",
        "mean_px",
        "median_px",
        "max_px",
        "std_px",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for det, err, used in zip(detections, errors, keep):
            row = {
                "label": det.label,
                "bag": det.bag,
                "source": det.source,
                "frame_index": det.frame_index,
                "stamp_ns": det.stamp_ns,
                "method": det.method,
                "board_area_px": board_area_px(det.corners),
                "used_final": bool(used),
            }
            row.update(err)
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refined-summary", type=Path, default=DEFAULT_REFINED)
    parser.add_argument("--scan-summary", type=Path, default=DEFAULT_SCAN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pattern-cols", type=int, default=9)
    parser.add_argument("--pattern-rows", type=int, default=6)
    parser.add_argument("--square-size-m", type=float, default=0.04)
    parser.add_argument("--min-views", type=int, default=18)
    args = parser.parse_args()

    pattern = (args.pattern_cols, args.pattern_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    candidates = collect_scan(args.scan_summary, pattern) + collect_refined(args.refined_summary, pattern)
    detections = choose_one_per_label(candidates)
    if len(detections) < args.min_views:
        raise SystemExit(f"Need at least {args.min_views} views, got {len(detections)}")

    image_size_wh = infer_image_size(detections)
    base_flags = cv2.CALIB_USE_INTRINSIC_GUESS
    model_specs = [
        ("free_5coeff", base_flags),
        ("fixed_pp_5coeff", base_flags | cv2.CALIB_FIX_PRINCIPAL_POINT),
        (
            "fixed_pp_zero_tangent_5coeff",
            base_flags | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_ZERO_TANGENT_DIST,
        ),
        (
            "fixed_pp_zero_tangent_fix_aspect_5coeff",
            base_flags
            | cv2.CALIB_FIX_PRINCIPAL_POINT
            | cv2.CALIB_ZERO_TANGENT_DIST
            | cv2.CALIB_FIX_ASPECT_RATIO,
        ),
        (
            "fixed_pp_zero_tangent_fix_k3",
            base_flags
            | cv2.CALIB_FIX_PRINCIPAL_POINT
            | cv2.CALIB_ZERO_TANGENT_DIST
            | cv2.CALIB_FIX_K3,
        ),
    ]
    models = [
        run_model(name, detections, pattern, args.square_size_m, image_size_wh, flags, args.min_views)
        for name, flags in model_specs
    ]
    comparisons = [model_summary(model) for model in models]
    free_rms = comparisons[0]["rms_px"]
    preferred_order = [
        "fixed_pp_zero_tangent_fix_k3",
        "fixed_pp_zero_tangent_fix_aspect_5coeff",
        "fixed_pp_zero_tangent_5coeff",
    ]
    selected = None
    for preferred_name in preferred_order:
        for model, row in zip(models, comparisons):
            if model["name"] == preferred_name and row["rms_px"] <= free_rms + 0.75:
                selected = model
                break
        if selected is not None:
            break
    if selected is None:
        selected = min(models, key=lambda model: model_summary(model)["rms_px"])
    selected_summary = model_summary(selected)
    final = selected["final"]
    summary = {
        "pattern_internal_corners": list(pattern),
        "square_size_m": args.square_size_m,
        "image_size_wh": list(image_size_wh),
        "n_candidates": len(candidates),
        "n_unique_views": len(detections),
        "selected_model": selected["name"],
        "n_used_final": selected_summary["n_used_final"],
        "outlier_labels": selected_summary["outlier_labels"],
        "model_comparison": comparisons,
        "final": selected_summary,
    }

    with (args.out_dir / "rgb_intrinsics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2)
    write_views_csv(
        args.out_dir / "rgb_intrinsics_views.csv",
        detections,
        selected["final_errors_full"],
        selected["keep"],
    )
    with (args.out_dir / "rgb_intrinsics_model_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "name",
            "n_used_final",
            "rms_px",
            "mean_view_error_px",
            "median_view_error_px",
            "max_view_error_px",
            "fx",
            "fy",
            "cx",
            "cy",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in comparisons:
            writer.writerow({k: row[k] for k in fields})
    np.save(args.out_dir / "rgb_K.npy", final["K"])
    np.save(args.out_dir / "rgb_dist_coeffs.npy", final["dist"])

    calib_doc = {
        "camera": "rgb",
        "source": "20260623 checkerboard detections",
        "selected_model": selected["name"],
        "pattern_internal_corners": list(pattern),
        "square_size_m": args.square_size_m,
        "image_size_wh": list(image_size_wh),
        "K": to_jsonable(final["K"]),
        "dist_coeffs": to_jsonable(final["dist"].reshape(-1)),
        "rms_px": final["rms"],
        "n_views": len(selected["final_detections"]),
        "outlier_labels": selected_summary["outlier_labels"],
        "status": "candidate_fixed_principal_point",
    }
    active_dir = Path("data/calibration/new_session/20260623")
    active_dir.mkdir(parents=True, exist_ok=True)
    with (active_dir / "rgb_intrinsics_20260623.json").open("w", encoding="utf-8") as f:
        json.dump(calib_doc, f, indent=2)

    print(f"RGB intrinsics written to {args.out_dir}")
    print(f"views: {len(detections)} unique, {len(selected['final_detections'])} used")
    print("model comparison:")
    for row in comparisons:
        print(
            f"  {row['name']}: rms={row['rms_px']:.4f}, "
            f"fx={row['fx']:.1f}, fy={row['fy']:.1f}, cx={row['cx']:.1f}, cy={row['cy']:.1f}"
        )
    print(f"selected_model: {selected['name']}")
    print(f"rms_px: {final['rms']:.4f}")
    print("K:")
    print(final["K"])
    print("dist:", final["dist"].reshape(-1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
