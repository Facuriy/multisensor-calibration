#!/usr/bin/env python3
"""Refine 20260623 Ouster->RGB extrinsics from RGB checker poses and LiDAR planes.

Transform convention:
    X_rgb = T_cam_lidar @ X_lidar
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.calibrate_rgb_intrinsics_from_detections import (
    ViewDetection,
    choose_one_per_label,
    collect_refined,
    collect_scan,
)


DEFAULT_SCAN = Path("runs/calibration_20260623_checker_lidar_scan/scan_summary.json")
DEFAULT_FALLBACK = Path("runs/calibration_20260623_lidar_intensity_fallback/fallback_intensity_summary.json")
DEFAULT_REFINED = Path(
    "runs/calibration_20260623_refined_multisensor_detection/"
    "refined_multisensor_detection_summary.json"
)
DEFAULT_INTRINSICS = Path("data/calibration/new_session/20260623/rgb_intrinsics_20260623.json")
DEFAULT_BASE_T = Path("runs/calibration/multipose_6dof_refined_20260528/T_cam_lidar_multipose_6dof.npy")
DEFAULT_COMPARE_T = [
    Path("runs/calibration/T_from_api_values_effective_20260528/T_cam_lidar_effective.npy"),
    Path("runs/calibration/T_checker_intensity_refined_txyz_20260528/T_cam_lidar_checker_intensity_refined.npy"),
]
DEFAULT_OUT = Path("runs/calibration_20260623_ouster_rgb_multipose_6dof")


@dataclass(frozen=True)
class LidarPlane:
    label: str
    bag: str
    source: str
    normal_lidar: np.ndarray
    d_lidar: float
    centroid_lidar: np.ndarray
    n_points: int
    rms_m: float


@dataclass(frozen=True)
class PoseFrame:
    label: str
    bag: str
    rgb_source: str
    lidar_source: str
    corners_cam: np.ndarray
    center_cam: np.ndarray
    normal_cam: np.ndarray
    plane_normal_lidar: np.ndarray
    plane_d_lidar: float
    plane_center_lidar: np.ndarray
    n_lidar_pts: int
    plane_rms_m: float


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = load_json(path)
    k = np.asarray(data["K"], dtype=np.float64)
    dist = np.asarray(data.get("dist_coeffs", []), dtype=np.float64).reshape(-1, 1)
    if k.shape != (3, 3):
        raise ValueError(f"{path} does not contain a 3x3 K")
    if dist.size == 0:
        dist = np.zeros((5, 1), dtype=np.float64)
    return k, dist


def load_transform(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        t = np.load(path)
    else:
        t = np.loadtxt(path)
    t = np.asarray(t, dtype=np.float64)
    if t.shape != (4, 4):
        raise ValueError(f"{path} is not a 4x4 transform")
    return t


def save_transform(out: Path, stem: str, t: np.ndarray) -> None:
    np.save(out / f"{stem}.npy", t)
    np.savetxt(out / f"{stem}.txt", t, fmt="%.10f")


def object_points(pattern: tuple[int, int], square_m: float) -> np.ndarray:
    cols, rows = pattern
    obj = np.zeros((rows * cols, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(square_m)
    return obj


def normalize_plane(raw: dict[str, Any]) -> tuple[np.ndarray, float, np.ndarray, int, float] | None:
    if not raw or not raw.get("detected", False):
        return None
    n = np.asarray(raw.get("normal_xyz"), dtype=np.float64).reshape(3)
    d = float(raw.get("d"))
    norm = float(np.linalg.norm(n))
    if norm < 1e-9:
        return None
    n /= norm
    d /= norm
    c = np.asarray(raw.get("centroid_xyz"), dtype=np.float64).reshape(3)
    n_points = int(raw.get("points", 0))
    rms_m = float(raw.get("rms_m", math.nan))
    return n, d, c, n_points, rms_m


def load_lidar_planes(scan_path: Path, fallback_path: Path) -> dict[str, LidarPlane]:
    planes: dict[str, LidarPlane] = {}
    if scan_path.exists():
        data = load_json(scan_path)
        for row in data.get("results", []):
            label = str(row.get("label_norm") or row.get("label_raw") or row.get("bag"))
            lidar = row.get("lidar") or {}
            parsed = normalize_plane(lidar.get("plane") or {})
            if parsed is None:
                continue
            n, d, c, n_points, rms_m = parsed
            planes[label] = LidarPlane(
                label=label,
                bag=str(row.get("bag") or ""),
                source="guided",
                normal_lidar=n,
                d_lidar=d,
                centroid_lidar=c,
                n_points=n_points,
                rms_m=rms_m,
            )
    if fallback_path.exists():
        data = load_json(fallback_path)
        for row in data.get("results", []):
            label = str(row.get("label_norm") or row.get("label_raw") or row.get("bag"))
            if label in planes:
                continue
            parsed = normalize_plane(row.get("plane") or {})
            if parsed is None:
                continue
            n, d, c, n_points, rms_m = parsed
            planes[label] = LidarPlane(
                label=label,
                bag=str(row.get("bag") or ""),
                source="fallback_intensity",
                normal_lidar=n,
                d_lidar=d,
                centroid_lidar=c,
                n_points=n_points,
                rms_m=rms_m,
            )
    return planes


def solve_board_pose(
    det: ViewDetection,
    obj_pts: np.ndarray,
    k: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts,
        det.corners.astype(np.float32),
        k,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    r, _ = cv2.Rodrigues(rvec)
    corners_cam = (obj_pts @ r.T + tvec.reshape(1, 3)).astype(np.float64)
    center_cam = corners_cam.mean(axis=0)
    normal_cam = r[:, 2].astype(np.float64)
    normal_cam /= np.linalg.norm(normal_cam)
    return corners_cam, center_cam, normal_cam


def build_frames(
    rgb_detections: list[ViewDetection],
    lidar_planes: dict[str, LidarPlane],
    obj_pts: np.ndarray,
    k: np.ndarray,
    dist: np.ndarray,
) -> list[PoseFrame]:
    frames: list[PoseFrame] = []
    for det in rgb_detections:
        plane = lidar_planes.get(det.label)
        if plane is None:
            continue
        pose = solve_board_pose(det, obj_pts, k, dist)
        if pose is None:
            continue
        corners_cam, center_cam, normal_cam = pose
        frames.append(
            PoseFrame(
                label=det.label,
                bag=det.bag or plane.bag,
                rgb_source=det.source,
                lidar_source=plane.source,
                corners_cam=corners_cam,
                center_cam=center_cam,
                normal_cam=normal_cam,
                plane_normal_lidar=plane.normal_lidar,
                plane_d_lidar=plane.d_lidar,
                plane_center_lidar=plane.centroid_lidar,
                n_lidar_pts=plane.n_points,
                plane_rms_m=plane.rms_m,
            )
        )
    return frames


def compose_delta(base_t: np.ndarray, params: np.ndarray) -> np.ndarray:
    r_delta = Rotation.from_rotvec(params[:3]).as_matrix()
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r_delta @ base_t[:3, :3]
    out[:3, 3] = base_t[:3, 3] + params[3:6]
    return out


def transformed_lidar_plane_in_cam(frame: PoseFrame, t_cam_lidar: np.ndarray) -> tuple[np.ndarray, float]:
    r = t_cam_lidar[:3, :3]
    t = t_cam_lidar[:3, 3]
    n_cam = r @ frame.plane_normal_lidar
    n_cam /= np.linalg.norm(n_cam)
    d_cam = frame.plane_d_lidar - float(n_cam @ t)
    if float(n_cam @ frame.normal_cam) < 0.0:
        n_cam = -n_cam
        d_cam = -d_cam
    return n_cam, d_cam


def residual_vector(
    params: np.ndarray,
    base_t: np.ndarray,
    frames: list[PoseFrame],
    normal_weight: float,
    centroid_weight: float,
    prior_weight: float,
) -> np.ndarray:
    t_cam_lidar = compose_delta(base_t, params)
    r = t_cam_lidar[:3, :3]
    t = t_cam_lidar[:3, 3]
    parts: list[np.ndarray] = []
    for frame in frames:
        n_cam, d_cam = transformed_lidar_plane_in_cam(frame, t_cam_lidar)
        parts.append(frame.corners_cam @ n_cam + d_cam)
        parts.append((n_cam - frame.normal_cam) * normal_weight)
        lidar_center_cam = r @ frame.plane_center_lidar + t
        parts.append((lidar_center_cam - frame.center_cam) * centroid_weight)
    if prior_weight > 0.0:
        parts.append(params[3:6] * prior_weight)
    return np.concatenate(parts)


def optimize_transform(
    base_t: np.ndarray,
    frames: list[PoseFrame],
    max_rot_deg: float,
    max_trans_m: float,
    normal_weight: float,
    centroid_weight: float,
    prior_weight: float,
    max_nfev: int,
) -> tuple[np.ndarray, np.ndarray, Any]:
    x0 = np.zeros(6, dtype=np.float64)
    rot_bound = math.radians(max_rot_deg)
    lower = np.array([-rot_bound, -rot_bound, -rot_bound, -max_trans_m, -max_trans_m, -max_trans_m])
    upper = np.array([rot_bound, rot_bound, rot_bound, max_trans_m, max_trans_m, max_trans_m])
    result = least_squares(
        residual_vector,
        x0,
        bounds=(lower, upper),
        args=(base_t, frames, normal_weight, centroid_weight, prior_weight),
        loss="soft_l1",
        f_scale=0.025,
        max_nfev=max_nfev,
        x_scale=np.array([0.03, 0.03, 0.03, 0.03, 0.03, 0.03]),
    )
    return compose_delta(base_t, result.x), result.x, result


def evaluate_transform(name: str, t_cam_lidar: np.ndarray, frames: list[PoseFrame]) -> dict[str, Any]:
    r = t_cam_lidar[:3, :3]
    t = t_cam_lidar[:3, 3]
    per_pose: list[dict[str, Any]] = []
    all_corner_abs: list[float] = []
    for frame in frames:
        n_cam, d_cam = transformed_lidar_plane_in_cam(frame, t_cam_lidar)
        corner_dist = frame.corners_cam @ n_cam + d_cam
        center_dist = float(frame.center_cam @ n_cam + d_cam)
        lidar_center_cam = r @ frame.plane_center_lidar + t
        centroid_err = float(np.linalg.norm(lidar_center_cam - frame.center_cam))
        dot = float(np.clip(n_cam @ frame.normal_cam, -1.0, 1.0))
        angle_deg = math.degrees(math.acos(dot))
        all_corner_abs.extend(np.abs(corner_dist).tolist())
        per_pose.append(
            {
                "label": frame.label,
                "bag": frame.bag,
                "rgb_source": frame.rgb_source,
                "lidar_source": frame.lidar_source,
                "corner_plane_rms_m": float(np.sqrt(np.mean(corner_dist**2))),
                "corner_plane_median_abs_m": float(np.median(np.abs(corner_dist))),
                "center_plane_abs_m": abs(center_dist),
                "normal_angle_deg": angle_deg,
                "centroid_error_m": centroid_err,
                "n_lidar_pts": frame.n_lidar_pts,
                "lidar_plane_rms_m": frame.plane_rms_m,
            }
        )
    arr = np.asarray(all_corner_abs, dtype=np.float64)
    pose_rms = np.asarray([p["corner_plane_rms_m"] for p in per_pose], dtype=np.float64)
    angles = np.asarray([p["normal_angle_deg"] for p in per_pose], dtype=np.float64)
    cent = np.asarray([p["centroid_error_m"] for p in per_pose], dtype=np.float64)
    return {
        "name": name,
        "n_poses": len(frames),
        "corner_plane_abs_mean_m": float(arr.mean()),
        "corner_plane_abs_median_m": float(np.median(arr)),
        "pose_corner_plane_rms_mean_m": float(pose_rms.mean()),
        "pose_corner_plane_rms_median_m": float(np.median(pose_rms)),
        "normal_angle_mean_deg": float(angles.mean()),
        "normal_angle_median_deg": float(np.median(angles)),
        "centroid_error_mean_m": float(cent.mean()),
        "centroid_error_median_m": float(np.median(cent)),
        "per_pose": per_pose,
    }


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in metrics.items() if k != "per_pose"}


def reject_outliers(frames: list[PoseFrame], metrics: dict[str, Any], min_frames: int) -> tuple[list[PoseFrame], list[str]]:
    per_pose = metrics["per_pose"]
    rms = np.asarray([p["corner_plane_rms_m"] for p in per_pose], dtype=np.float64)
    med = float(np.median(rms))
    threshold = max(0.06, 3.0 * med)
    keep_labels = {
        p["label"]
        for p in per_pose
        if p["corner_plane_rms_m"] <= threshold and p["normal_angle_deg"] <= 30.0
    }
    if len(keep_labels) < min_frames:
        keep_labels = {p["label"] for p in sorted(per_pose, key=lambda x: x["corner_plane_rms_m"])[:min_frames]}
    kept = [f for f in frames if f.label in keep_labels]
    rejected = [f.label for f in frames if f.label not in keep_labels]
    return kept, rejected


def cross_validate(
    base_t: np.ndarray,
    frames: list[PoseFrame],
    folds: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if folds < 2 or len(frames) < folds:
        return []
    out = []
    for fold in range(folds):
        train = [f for i, f in enumerate(frames) if i % folds != fold]
        valid = [f for i, f in enumerate(frames) if i % folds == fold]
        t_fold, delta, result = optimize_transform(
            base_t,
            train,
            args.max_rot_deg,
            args.max_trans_m,
            args.normal_weight,
            args.centroid_weight,
            args.prior_weight,
            args.max_nfev,
        )
        train_metrics = evaluate_transform(f"fold_{fold}_train", t_fold, train)
        valid_metrics = evaluate_transform(f"fold_{fold}_valid", t_fold, valid)
        out.append(
            {
                "fold": fold,
                "train_labels": [f.label for f in train],
                "valid_labels": [f.label for f in valid],
                "delta_rotvec": delta[:3].tolist(),
                "delta_deg_xyz": np.degrees(delta[:3]).tolist(),
                "delta_t_m": delta[3:6].tolist(),
                "success": bool(result.success),
                "cost": float(result.cost),
                "train": compact_metrics(train_metrics),
                "valid": compact_metrics(valid_metrics),
            }
        )
    return out


def frames_to_json(frames: list[PoseFrame]) -> list[dict[str, Any]]:
    out = []
    for frame in frames:
        out.append(
            {
                "label": frame.label,
                "bag": frame.bag,
                "rgb_source": frame.rgb_source,
                "lidar_source": frame.lidar_source,
                "center_cam": frame.center_cam.tolist(),
                "normal_cam": frame.normal_cam.tolist(),
                "plane_normal_lidar": frame.plane_normal_lidar.tolist(),
                "plane_d_lidar": frame.plane_d_lidar,
                "plane_center_lidar": frame.plane_center_lidar.tolist(),
                "n_lidar_pts": frame.n_lidar_pts,
                "plane_rms_m": frame.plane_rms_m,
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-summary", type=Path, default=DEFAULT_SCAN)
    parser.add_argument("--fallback-summary", type=Path, default=DEFAULT_FALLBACK)
    parser.add_argument("--refined-summary", type=Path, default=DEFAULT_REFINED)
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument("--base-T", type=Path, default=DEFAULT_BASE_T)
    parser.add_argument("--compare-T", type=Path, action="append", default=DEFAULT_COMPARE_T.copy())
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pattern-cols", type=int, default=9)
    parser.add_argument("--pattern-rows", type=int, default=6)
    parser.add_argument("--square-size-m", type=float, default=0.04)
    parser.add_argument("--max-rot-deg", type=float, default=15.0)
    parser.add_argument("--max-trans-m", type=float, default=0.50)
    parser.add_argument("--normal-weight", type=float, default=0.15)
    parser.add_argument("--centroid-weight", type=float, default=0.05)
    parser.add_argument("--prior-weight", type=float, default=0.03)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-nfev", type=int, default=1200)
    parser.add_argument("--min-frames-after-reject", type=int, default=18)
    args = parser.parse_args()

    pattern = (args.pattern_cols, args.pattern_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    k, dist = load_intrinsics(args.intrinsics)
    obj = object_points(pattern, args.square_size_m)
    rgb = choose_one_per_label(collect_scan(args.scan_summary, pattern) + collect_refined(args.refined_summary, pattern))
    planes = load_lidar_planes(args.scan_summary, args.fallback_summary)
    frames_all = build_frames(rgb, planes, obj, k, dist)
    if len(frames_all) < args.min_frames_after_reject:
        raise SystemExit(f"Need at least {args.min_frames_after_reject} RGB+LiDAR frames, got {len(frames_all)}")

    base_t = load_transform(args.base_T)
    t_first, delta_first, result_first = optimize_transform(
        base_t,
        frames_all,
        args.max_rot_deg,
        args.max_trans_m,
        args.normal_weight,
        args.centroid_weight,
        args.prior_weight,
        args.max_nfev,
    )
    first_metrics = evaluate_transform("first_pass_all_frames", t_first, frames_all)
    frames_used, rejected_labels = reject_outliers(frames_all, first_metrics, args.min_frames_after_reject)

    t_opt, delta, result = optimize_transform(
        base_t,
        frames_used,
        args.max_rot_deg,
        args.max_trans_m,
        args.normal_weight,
        args.centroid_weight,
        args.prior_weight,
        args.max_nfev,
    )
    save_transform(args.out_dir, "T_cam_lidar_20260623_multipose_6dof", t_opt)

    comparisons = [
        evaluate_transform("A_base_previous_multipose", base_t, frames_used),
        evaluate_transform("D_20260623_multipose_6dof", t_opt, frames_used),
    ]
    for i, path in enumerate(args.compare_T):
        if path.exists():
            comparisons.insert(1 + i, evaluate_transform(f"B_compare_{i}_{path.stem}", load_transform(path), frames_used))

    cv = cross_validate(base_t, frames_used, args.folds, args)
    summary = {
        "inputs": {
            "scan_summary": str(args.scan_summary),
            "fallback_summary": str(args.fallback_summary),
            "refined_summary": str(args.refined_summary),
            "intrinsics": str(args.intrinsics),
            "base_T": str(args.base_T),
            "compare_T": [str(p) for p in args.compare_T],
        },
        "pattern_internal_corners": list(pattern),
        "square_size_m": args.square_size_m,
        "n_rgb_detections": len(rgb),
        "n_lidar_planes": len(planes),
        "n_frames_all": len(frames_all),
        "n_frames_used": len(frames_used),
        "rejected_labels": rejected_labels,
        "used_labels": [f.label for f in frames_used],
        "params": {
            "max_rot_deg": args.max_rot_deg,
            "max_trans_m": args.max_trans_m,
            "normal_weight": args.normal_weight,
            "centroid_weight": args.centroid_weight,
            "prior_weight": args.prior_weight,
            "folds": args.folds,
        },
        "first_pass": {
            "success": bool(result_first.success),
            "cost": float(result_first.cost),
            "delta_deg_xyz": np.degrees(delta_first[:3]).tolist(),
            "delta_t_m": delta_first[3:6].tolist(),
            "metrics": compact_metrics(first_metrics),
        },
        "optimization": {
            "success": bool(result.success),
            "message": str(result.message),
            "cost": float(result.cost),
            "nfev": int(result.nfev),
            "delta_rotvec": delta[:3].tolist(),
            "delta_deg_xyz": np.degrees(delta[:3]).tolist(),
            "delta_t_m": delta[3:6].tolist(),
            "hit_bounds": {
                "rotation": bool(np.any(np.isclose(np.abs(delta[:3]), math.radians(args.max_rot_deg), atol=1e-5))),
                "translation": bool(np.any(np.isclose(np.abs(delta[3:6]), args.max_trans_m, atol=1e-5))),
            },
        },
        "T_cam_lidar_20260623_multipose_6dof": t_opt.tolist(),
        "comparison": comparisons,
        "comparison_compact": [compact_metrics(m) for m in comparisons],
        "cross_validation": cv,
        "frames_all": frames_to_json(frames_all),
    }
    with (args.out_dir / "multipose_6dof_20260623_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (args.out_dir / "multipose_input_frames_20260623.json").open("w", encoding="utf-8") as f:
        json.dump(frames_to_json(frames_all), f, indent=2)

    print(f"RGB detections: {len(rgb)}")
    print(f"LiDAR planes:   {len(planes)}")
    print(f"Frames all/used:{len(frames_all)}/{len(frames_used)}")
    if rejected_labels:
        print("Rejected:", ", ".join(rejected_labels))
    print("Delta deg xyz:", " ".join(f"{v:+.4f}" for v in np.degrees(delta[:3])))
    print("Delta t m:    ", " ".join(f"{v:+.4f}" for v in delta[3:6]))
    print(f"Success: {result.success} cost={result.cost:.6g} nfev={result.nfev}")
    print("Comparison:")
    for metrics in comparisons:
        compact = compact_metrics(metrics)
        print(
            f"  {metrics['name']}: "
            f"corner_med={compact['corner_plane_abs_median_m']*1000:.1f}mm "
            f"pose_rms_med={compact['pose_corner_plane_rms_median_m']*1000:.1f}mm "
            f"normal_med={compact['normal_angle_median_deg']:.2f}deg "
            f"centroid_med={compact['centroid_error_median_m']*1000:.1f}mm"
        )
    print(f"Wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
