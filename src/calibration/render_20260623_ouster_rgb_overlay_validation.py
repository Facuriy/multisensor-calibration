#!/usr/bin/env python3
"""Render visual Ouster->RGB overlay checks for the 20260623 calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.extract_ouster_panel_from_manual_intensity_roi import CLOUD_TOPIC, cloud_to_organized  # noqa: E402
from src.calibration.scan_20260623_checker_lidar import RGB_TOPIC  # noqa: E402
from src.extraction.extract_all_bag_images import decode_ros_image, robust_preview  # noqa: E402


DEFAULT_SCAN = Path("runs/calibration_20260623_checker_lidar_scan/scan_summary.json")
DEFAULT_CALIB = Path("data/calibration/active_calibration.json")
DEFAULT_BASE_T = Path("runs/calibration/multipose_6dof_refined_20260528/T_cam_lidar_multipose_6dof.npy")
DEFAULT_OUT = Path("runs/calibration_20260623_overlay_validation")
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


def load_transform(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        out = np.load(path)
    else:
        out = np.loadtxt(path)
    out = np.asarray(out, dtype=np.float64)
    if out.shape != (4, 4):
        raise ValueError(f"{path} is not a 4x4 transform")
    return out


def calibration_from_active(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = load_json(path)
    rgb = data["intrinsics"]["rgb"]
    k = np.asarray(rgb["K"], dtype=np.float64)
    dist = np.asarray(rgb.get("dist_coeffs", []), dtype=np.float64).reshape(-1, 1)
    if dist.size == 0:
        dist = np.zeros((5, 1), dtype=np.float64)
    t = np.asarray(data["extrinsics"]["T_cam_lidar"]["matrix"], dtype=np.float64)
    return k, dist, t


def collect_topic_messages(bag: Path, topics: list[str]) -> dict[str, list[tuple[int, object]]]:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    out = {topic: [] for topic in topics}
    with Reader(bag) as reader:
        conns = [conn for conn in reader.connections if conn.topic in topics]
        for conn, ts, raw in reader.messages(connections=conns):
            out[conn.topic].append((int(ts), typestore.deserialize_ros1(raw, conn.msgtype)))
    return out


def nearest(items: list[tuple[int, object]], stamp_ns: int) -> tuple[int, object] | None:
    if not items:
        return None
    return min(items, key=lambda item: abs(item[0] - stamp_ns))


def project_lidar_points(
    xyz_lidar: np.ndarray,
    intensity: np.ndarray,
    t_cam_lidar: np.ndarray,
    k: np.ndarray,
    dist: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = xyz_lidar.reshape(-1, 3).astype(np.float64)
    inten = intensity.reshape(-1).astype(np.float64)
    finite = np.isfinite(pts).all(axis=1)
    finite &= np.linalg.norm(pts, axis=1) > 0.05
    finite &= np.linalg.norm(pts, axis=1) < 8.0
    pts = pts[finite]
    inten = inten[finite]
    cam = (t_cam_lidar[:3, :3] @ pts.T).T + t_cam_lidar[:3, 3]
    front = cam[:, 2] > 0.05
    pts = pts[front]
    inten = inten[front]
    cam = cam[front]
    uv, _ = cv2.projectPoints(cam, np.zeros(3), np.zeros(3), k, dist)
    uv = uv.reshape(-1, 2)
    h, w = image_shape
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv[inside], cam[inside], inten[inside]


def colors_from_values(values: np.ndarray, cmap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    if len(values) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    vals = values.astype(np.float64)
    lo, hi = np.percentile(vals, [5, 95])
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((vals - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(norm.reshape(-1, 1), cmap).reshape(-1, 3)


def draw_points(
    image: np.ndarray,
    uv: np.ndarray,
    values: np.ndarray,
    radius: int,
    alpha: float,
    cmap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    if len(uv) == 0:
        return image.copy()
    overlay = image.copy()
    colors = colors_from_values(values, cmap)
    for (u, v), color in zip(uv.astype(int), colors):
        cv2.circle(overlay, (int(u), int(v)), radius, tuple(int(c) for c in color), -1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)


def draw_checker(image: np.ndarray, corners: np.ndarray | None) -> np.ndarray:
    out = image.copy()
    if corners is None:
        return out
    pts = corners.reshape(-1, 2).astype(int)
    for p in pts:
        cv2.circle(out, tuple(p), 3, (0, 255, 255), -1, cv2.LINE_AA)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 3, cv2.LINE_AA)
    return out


def crop_around(corners: np.ndarray | None, uv: np.ndarray, shape: tuple[int, int], margin: int = 260) -> tuple[int, int, int, int]:
    h, w = shape
    pts = []
    if corners is not None:
        pts.append(corners.reshape(-1, 2))
    if len(uv):
        pts.append(uv.reshape(-1, 2))
    if not pts:
        return 0, 0, w, h
    all_pts = np.concatenate(pts, axis=0)
    x0 = max(0, int(np.floor(np.nanmin(all_pts[:, 0]))) - margin)
    y0 = max(0, int(np.floor(np.nanmin(all_pts[:, 1]))) - margin)
    x1 = min(w, int(np.ceil(np.nanmax(all_pts[:, 0]))) + margin)
    y1 = min(h, int(np.ceil(np.nanmax(all_pts[:, 1]))) + margin)
    return x0, y0, x1, y1


def label_row_by_norm(scan: dict[str, Any], label: str) -> dict[str, Any] | None:
    for row in scan.get("results", []):
        if row.get("label_norm") == label:
            return row
    return None


def get_rgb_corners(row: dict[str, Any]) -> np.ndarray | None:
    det = (row.get("camera_detections") or {}).get("rgb") or {}
    if not det.get("detected"):
        return None
    pts = np.asarray(det.get("corners_px"), dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return None
    return pts


def projection_bbox_metrics(uv: np.ndarray, corners: np.ndarray | None) -> dict[str, Any]:
    if corners is None or len(uv) == 0:
        return {
            "projected_count": int(len(uv)),
            "inside_checker_bbox_fraction": None,
            "centroid_offset_px": None,
        }
    pts = corners.reshape(-1, 2)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    inside = (uv[:, 0] >= x0) & (uv[:, 0] <= x1) & (uv[:, 1] >= y0) & (uv[:, 1] <= y1)
    checker_center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float64)
    uv_center = np.median(uv, axis=0)
    return {
        "projected_count": int(len(uv)),
        "inside_checker_bbox_fraction": float(np.mean(inside)),
        "centroid_offset_px": (uv_center - checker_center).tolist(),
    }


def render_one(
    row: dict[str, Any],
    k: np.ndarray,
    dist: np.ndarray,
    t_new: np.ndarray,
    t_base: np.ndarray,
    out_dir: Path,
    bag_cache: Path | None,
) -> dict[str, Any]:
    bag = Path(row["bag_path"])
    if bag_cache is not None:
        cached = bag_cache / bag.name
        if cached.exists():
            bag = cached
    label = str(row["label_norm"])
    if not bag.exists():
        return {
            "label": label,
            "rendered": False,
            "reason": "bag_path_not_visible_to_python",
            "bag_path": str(bag),
        }
    lidar = row.get("lidar") or {}
    if not lidar.get("cloud_stamp_ns"):
        return {"label": label, "rendered": False, "reason": "missing_guided_cloud_stamp"}
    rgb_det = ((row.get("camera_detections") or {}).get("rgb") or {})
    if not rgb_det.get("stamp_ns"):
        return {"label": label, "rendered": False, "reason": "missing_rgb_stamp"}

    messages = collect_topic_messages(bag, [RGB_TOPIC, CLOUD_TOPIC])
    rgb_msg = nearest(messages[RGB_TOPIC], int(rgb_det["stamp_ns"]))
    cloud_msg = nearest(messages[CLOUD_TOPIC], int(lidar["cloud_stamp_ns"]))
    if rgb_msg is None or cloud_msg is None:
        return {"label": label, "rendered": False, "reason": "missing_bag_messages"}

    rgb = decode_ros_image(rgb_msg[1])
    rgb = robust_preview(rgb, "rgb")
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    cloud = cloud_to_organized(cloud_msg[1])
    xyz = np.dstack([cloud["x"], cloud["y"], cloud["z"]])
    intensity = cloud.get("intensity", np.zeros(cloud["x"].shape, dtype=np.float32))
    roi = lidar.get("roi_xyxy_ouster") or [0, 0, xyz.shape[1], xyz.shape[0]]
    x0, y0, x1, y1 = [int(v) for v in roi]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(xyz.shape[1], x1), min(xyz.shape[0], y1)
    xyz_roi = xyz[y0:y1, x0:x1]
    int_roi = intensity[y0:y1, x0:x1]
    plane = (lidar.get("plane") or {})
    if plane.get("detected") and "normal_xyz" in plane and "d" in plane:
        n = np.asarray(plane["normal_xyz"], dtype=np.float64)
        n_norm = float(np.linalg.norm(n))
        if n_norm > 1e-9:
            n = n / n_norm
            d = float(plane["d"]) / n_norm
            plane_dist = np.abs(xyz_roi.reshape(-1, 3) @ n + d).reshape(xyz_roi.shape[:2])
            threshold = max(0.03, float(plane.get("rms_m", 0.015)) * 2.5)
            inlier = plane_dist < threshold
            xyz_roi = xyz_roi.copy()
            int_roi = int_roi.copy()
            xyz_roi[~inlier] = np.nan
            int_roi[~inlier] = np.nan

    corners = get_rgb_corners(row)
    base_uv, base_cam, _ = project_lidar_points(xyz_roi, int_roi, t_base, k, dist, rgb.shape[:2])
    new_uv, new_cam, _ = project_lidar_points(xyz_roi, int_roi, t_new, k, dist, rgb.shape[:2])

    base_img = draw_checker(draw_points(rgb, base_uv, base_cam[:, 2] if len(base_cam) else np.array([]), 4, 0.95), corners)
    new_img = draw_checker(draw_points(rgb, new_uv, new_cam[:, 2] if len(new_cam) else np.array([]), 4, 0.95), corners)

    cx0, cy0, cx1, cy1 = crop_around(corners, new_uv, rgb.shape[:2])
    base_crop = base_img[cy0:cy1, cx0:cx1]
    new_crop = new_img[cy0:cy1, cx0:cx1]
    target_h = 700
    scale = min(target_h / max(base_crop.shape[0], 1), 1.0)
    if scale < 1.0:
        size = (int(base_crop.shape[1] * scale), int(base_crop.shape[0] * scale))
        base_crop = cv2.resize(base_crop, size, interpolation=cv2.INTER_AREA)
        new_crop = cv2.resize(new_crop, size, interpolation=cv2.INTER_AREA)
    pad = np.full((base_crop.shape[0], 24, 3), 255, dtype=np.uint8)
    pair = np.hstack([base_crop, pad, new_crop])
    cv2.putText(pair, f"{label}  previous", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(
        pair,
        "20260623 candidate",
        (base_crop.shape[1] + 42, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    out_path = out_dir / f"{label}_overlay_compare.jpg"
    cv2.imwrite(str(out_path), pair)
    return {
        "label": label,
        "bag": str(bag),
        "rendered": True,
        "path": str(out_path),
        "base_projected_points": int(len(base_uv)),
        "new_projected_points": int(len(new_uv)),
        "base_bbox_metrics": projection_bbox_metrics(base_uv, corners),
        "new_bbox_metrics": projection_bbox_metrics(new_uv, corners),
        "crop_xyxy": [cx0, cy0, cx1, cy1],
    }


def make_contactsheet(items: list[dict[str, Any]], out_dir: Path) -> Path | None:
    paths = [Path(item["path"]) for item in items if item.get("rendered")]
    images = [cv2.imread(str(path)) for path in paths]
    images = [img for img in images if img is not None]
    if not images:
        return None
    width = 1600
    resized = []
    for img in images:
        scale = width / img.shape[1]
        resized.append(cv2.resize(img, (width, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA))
    sheet = np.vstack(resized)
    out = out_dir / "overlay_validation_contactsheet.jpg"
    cv2.imwrite(str(out), sheet)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-summary", type=Path, default=DEFAULT_SCAN)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIB)
    parser.add_argument("--base-T", type=Path, default=DEFAULT_BASE_T)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bag-cache", type=Path, default=DEFAULT_OUT / "bag_cache")
    parser.add_argument("--labels", nargs="*", default=DEFAULT_LABELS)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scan = load_json(args.scan_summary)
    k, dist, t_new = calibration_from_active(args.calibration)
    if args.base_T.exists():
        t_base = load_transform(args.base_T)
    else:
        # The historical baseline run is intentionally not part of the cleaned
        # public workspace. Keep the renderer usable by falling back to the
        # active candidate when the old baseline is absent.
        t_base = t_new.copy()

    results = []
    for label in args.labels:
        row = label_row_by_norm(scan, label)
        if row is None:
            results.append({"label": label, "rendered": False, "reason": "label_not_found"})
            continue
        results.append(render_one(row, k, dist, t_new, t_base, args.out_dir, args.bag_cache))
        print(results[-1])

    sheet = make_contactsheet(results, args.out_dir)
    summary = {"labels": args.labels, "results": results, "contactsheet": str(sheet) if sheet else None}
    with (args.out_dir / "overlay_validation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
