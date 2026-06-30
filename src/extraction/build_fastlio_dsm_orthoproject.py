#!/usr/bin/env python3
"""Metric DSM + RGB orthoprojector from FAST-LIO/GNSS pose.

This is the next step after the visual `vertical_strip` mosaic.  The geometry is
not estimated from 2D image homographies.  Instead:

1. FAST-LIO /cloud_registered points are aligned to UTM with GNSS.
2. A DSM grid is built from the LiDAR map in UTM.
3. RGB frames are projected onto that DSM using FAST-LIO pose(t),
   Ouster->RGB extrinsics and RGB intrinsics.

The default GPS alignment is rigid SE(2) without scale because FAST-LIO already
has metric scale.  A similarity mode is kept for diagnostics only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation as R
from shapely.geometry import LineString, Point

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction.export_fastlio_gps_qgis import (  # noqa: E402
    CLOUD_TOPIC,
    associate_gps,
    load_gps,
    load_odometry,
    msg_stamp_ns,
    pointcloud_xyz_i,
    similarity_2d,
    transform_xy,
    utm_epsg,
)
from src.extraction.extract_all_bag_images import decode_ros_image  # noqa: E402


RGB_TOPIC = "/ssf/BFS_usb_0/image_raw"


def rigid_2d(src_xy: np.ndarray, dst_xy: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src_c = src_xy.mean(axis=0)
    dst_c = dst_xy.mean(axis=0)
    xs = src_xy - src_c
    xd = dst_xy - dst_c
    u, _s, vt = np.linalg.svd(xs.T @ xd)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        vt[-1] *= -1
        rot = u @ vt
    trans = dst_c - (src_c @ rot)
    return 1.0, rot, trans


def inverse_transform_xy(xy_utm: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return ((xy_utm - trans) / scale) @ rot.T


def alignment_residuals(src_xy: np.ndarray, dst_xy: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> dict[str, float]:
    pred = transform_xy(src_xy, scale, rot, trans)
    err = np.linalg.norm(pred - dst_xy, axis=1)
    return {
        "mean_m": float(err.mean()),
        "median_m": float(np.median(err)),
        "max_m": float(err.max()),
        "p90_m": float(np.percentile(err, 90)),
    }


def load_calibration(path: Path) -> dict[str, np.ndarray]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    rgb = doc["intrinsics"]["rgb"]
    return {
        "K": np.asarray(rgb["K"], dtype=np.float64),
        "dist": np.asarray(rgb.get("dist_coeffs", [0, 0, 0, 0, 0]), dtype=np.float64).reshape(-1, 1),
        "T_cam_lidar": np.asarray(doc["ouster_rgb"]["T_cam_lidar"]["matrix"], dtype=np.float64),
    }


def pose_matrix(row: dict[str, float]) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.from_quat([row["qx"], row["qy"], row["qz"], row["qw"]]).as_matrix()
    out[:3, 3] = [row["x"], row["y"], row["z"]]
    return out


def nearest_index(stamps: np.ndarray, stamp_ns: int, max_dt_ms: float) -> int | None:
    if len(stamps) == 0:
        return None
    idx = int(np.argmin(np.abs(stamps - int(stamp_ns))))
    if abs(int(stamps[idx]) - int(stamp_ns)) / 1e6 > max_dt_ms:
        return None
    return idx


def load_rgb_frames(original_bag: Path, start_ns: int, stop_ns: int, every: int) -> tuple[np.ndarray, list[np.ndarray]]:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    stamps: list[int] = []
    images: list[np.ndarray] = []
    idx = 0
    with Reader(original_bag) as reader:
        conns = [c for c in reader.connections if c.topic == RGB_TOPIC]
        for conn, ts, raw in reader.messages(connections=conns):
            if int(ts) < start_ns or int(ts) > stop_ns:
                continue
            if idx % max(1, every) != 0:
                idx += 1
                continue
            idx += 1
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            img = decode_ros_image(msg)
            if img is None or img.ndim != 3:
                continue
            stamps.append(msg_stamp_ns(msg, int(ts)))
            images.append(img)
    return np.asarray(stamps, dtype=np.int64), images


def collect_map_points(
    fastlio_bag: Path,
    scale: float,
    rot: np.ndarray,
    trans: np.ndarray,
    cloud_every: int,
    max_points_total: int,
    z_clip_pct: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    xyz_chunks: list[np.ndarray] = []
    int_chunks: list[np.ndarray] = []
    seen = 0
    with Reader(fastlio_bag) as reader:
        conns = [c for c in reader.connections if c.topic == CLOUD_TOPIC]
        for conn, _ts, raw in reader.messages(connections=conns):
            if seen % max(1, cloud_every) != 0:
                seen += 1
                continue
            seen += 1
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            xyz_local, intensity = pointcloud_xyz_i(msg)
            if len(xyz_local) == 0:
                continue
            xy_utm = transform_xy(xyz_local[:, :2].astype(np.float64), scale, rot, trans)
            xyz_chunks.append(np.column_stack([xy_utm, xyz_local[:, 2]]).astype(np.float32))
            int_chunks.append(intensity.astype(np.float32))
    if not xyz_chunks:
        raise RuntimeError("no FAST-LIO cloud points found")
    xyz = np.vstack(xyz_chunks)
    intensity = np.concatenate(int_chunks)
    zlo, zhi = np.percentile(xyz[:, 2], z_clip_pct)
    keep = (xyz[:, 2] >= zlo) & (xyz[:, 2] <= zhi)
    xyz, intensity = xyz[keep], intensity[keep]
    if max_points_total > 0 and len(xyz) > max_points_total:
        rng = np.random.default_rng(31)
        take = rng.choice(len(xyz), int(max_points_total), replace=False)
        xyz, intensity = xyz[take], intensity[take]
    return xyz, intensity


def distance_to_polyline_sq(points_xy: np.ndarray, line_xy: np.ndarray) -> np.ndarray:
    best = np.full(len(points_xy), np.inf, dtype=np.float64)
    if len(line_xy) < 2 or len(points_xy) == 0:
        return best
    px, py = points_xy[:, 0], points_xy[:, 1]
    for a, b in zip(line_xy[:-1], line_xy[1:]):
        vx, vy = float(b[0] - a[0]), float(b[1] - a[1])
        seg_len2 = vx * vx + vy * vy
        if seg_len2 <= 1e-12:
            dx, dy = px - a[0], py - a[1]
            d2 = dx * dx + dy * dy
        else:
            t = ((px - a[0]) * vx + (py - a[1]) * vy) / seg_len2
            t = np.clip(t, 0.0, 1.0)
            qx = a[0] + t * vx
            qy = a[1] + t * vy
            dx, dy = px - qx, py - qy
            d2 = dx * dx + dy * dy
        best = np.minimum(best, d2)
    return best


def filter_to_trajectory_corridor(
    xyz_utm: np.ndarray,
    intensity: np.ndarray,
    trajectory_utm: np.ndarray,
    half_width_m: float,
    end_pad_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if half_width_m <= 0 or len(xyz_utm) == 0:
        return xyz_utm, intensity, {"enabled": False, "points_before": int(len(xyz_utm)), "points_after": int(len(xyz_utm))}
    xmin, ymin = trajectory_utm.min(axis=0) - (half_width_m + end_pad_m)
    xmax, ymax = trajectory_utm.max(axis=0) + (half_width_m + end_pad_m)
    bbox = (xyz_utm[:, 0] >= xmin) & (xyz_utm[:, 0] <= xmax) & (xyz_utm[:, 1] >= ymin) & (xyz_utm[:, 1] <= ymax)
    cand = np.flatnonzero(bbox)
    keep = np.zeros(len(xyz_utm), bool)
    if len(cand):
        d2 = distance_to_polyline_sq(xyz_utm[cand, :2].astype(np.float64), trajectory_utm.astype(np.float64))
        keep[cand] = d2 <= half_width_m * half_width_m
    stats = {
        "enabled": True,
        "half_width_m": float(half_width_m),
        "end_pad_m": float(end_pad_m),
        "points_before": int(len(xyz_utm)),
        "bbox_candidates": int(len(cand)),
        "points_after": int(keep.sum()),
    }
    return xyz_utm[keep], intensity[keep], stats


def nearest_fill(arr: np.ndarray, valid: np.ndarray, radius_px: int) -> tuple[np.ndarray, np.ndarray]:
    if radius_px <= 0 or not np.any(valid):
        return arr, valid
    dist, idx = distance_transform_edt(~valid, return_indices=True)
    fill = (~valid) & (dist <= radius_px)
    out = arr.copy()
    out[fill] = arr[idx[0][fill], idx[1][fill]]
    return out, valid | fill


def build_dsm(
    xyz_utm: np.ndarray,
    intensity: np.ndarray,
    resolution_m: float,
    fill_radius_px: int,
    surface_mode: str,
) -> dict[str, Any]:
    e, n, z = xyz_utm[:, 0], xyz_utm[:, 1], xyz_utm[:, 2]
    pad = resolution_m * 2
    emin, emax = float(e.min() - pad), float(e.max() + pad)
    nmin, nmax = float(n.min() - pad), float(n.max() + pad)
    width = max(1, int(math.ceil((emax - emin) / resolution_m)))
    height = max(1, int(math.ceil((nmax - nmin) / resolution_m)))
    col = np.clip(((e - emin) / resolution_m).astype(np.int32), 0, width - 1)
    row = np.clip(((nmax - n) / resolution_m).astype(np.int32), 0, height - 1)
    flat = row * width + col
    order = np.lexsort((z, flat))
    flat_s, z_s, int_s = flat[order], z[order], intensity[order]
    starts = np.r_[0, np.flatnonzero(flat_s[1:] != flat_s[:-1]) + 1]
    stops = np.r_[starts[1:], len(flat_s)]
    dsm = np.full(height * width, np.nan, np.float32)
    inten = np.full(height * width, np.nan, np.float32)
    dens = np.zeros(height * width, np.uint16)
    for a, b in zip(starts, stops):
        cell = flat_s[a]
        dens[cell] = min(b - a, np.iinfo(np.uint16).max)
        if surface_mode == "p95" and b - a >= 4:
            q = float(np.percentile(z_s[a:b], 95))
            k = a + int(np.argmin(np.abs(z_s[a:b] - q)))
        else:
            k = b - 1
        dsm[cell] = z_s[k]
        inten[cell] = int_s[k]
    dsm = dsm.reshape(height, width)
    inten = inten.reshape(height, width)
    dens = dens.reshape(height, width)
    valid = np.isfinite(dsm)
    dsm_filled, valid_filled = nearest_fill(dsm, valid, fill_radius_px)
    inten_filled, _ = nearest_fill(inten, valid, fill_radius_px)
    ground = float(np.nanpercentile(dsm[valid], 5)) if np.any(valid) else 0.0
    transform = from_origin(emin, nmax, resolution_m, resolution_m)
    return {
        "dsm": dsm,
        "dsm_filled": dsm_filled,
        "height_rel": (dsm_filled - ground).astype(np.float32),
        "intensity": inten_filled.astype(np.float32),
        "density": dens,
        "valid": valid,
        "valid_filled": valid_filled,
        "bounds": (emin, nmin, emax, nmax),
        "transform": transform,
        "ground_z_p05_m": ground,
        "shape_hw": (height, width),
    }


def robust_u8(arr: np.ndarray, valid: np.ndarray, lo_hi: tuple[float, float] = (2, 98)) -> np.ndarray:
    out = np.zeros(arr.shape, np.uint8)
    valid = valid & np.isfinite(arr)
    if not np.any(valid):
        return out
    lo, hi = np.percentile(arr[valid], lo_hi)
    if hi <= lo:
        hi = lo + 1.0
    out[valid] = np.clip((arr[valid] - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return out


def orthoproject_rgb(
    dsm: dict[str, Any],
    odom: list[dict[str, float]],
    rgb_stamps: np.ndarray,
    rgb_images: list[np.ndarray],
    calib: dict[str, np.ndarray],
    scale: float,
    rot2: np.ndarray,
    trans2: np.ndarray,
    max_odom_dt_ms: float,
    rgb_roi: tuple[int, int, int, int] | None,
    min_cam_z: float,
    invert_t_cam_lidar: bool,
    reject_bright_low_sat: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    valid = dsm["valid_filled"]
    rows, cols = np.where(valid)
    if len(rows) == 0:
        raise RuntimeError("DSM has no valid cells")
    transform = dsm["transform"]
    east = transform.c + (cols.astype(np.float64) + 0.5) * transform.a
    north = transform.f + (rows.astype(np.float64) + 0.5) * transform.e
    xy_map = inverse_transform_xy(np.column_stack([east, north]), scale, rot2, trans2)
    z_map = dsm["dsm_filled"][rows, cols].astype(np.float64)
    points_map = np.column_stack([xy_map, z_map])
    out = np.zeros((*valid.shape, 3), np.uint8)
    score = np.full(valid.shape, -np.inf, np.float32)
    used = np.zeros(valid.shape, bool)
    odom_stamps = np.asarray([r["stamp_ns"] for r in odom], dtype=np.int64)
    T_cam_lidar = np.linalg.inv(calib["T_cam_lidar"]) if invert_t_cam_lidar else calib["T_cam_lidar"]
    K, dist = calib["K"], calib["dist"]
    stats = {"rgb_frames": len(rgb_images), "rgb_frames_used": 0, "projected_cells": 0, "painted_cells": 0}
    for stamp, img in zip(rgb_stamps, rgb_images):
        oi = nearest_index(odom_stamps, int(stamp), max_odom_dt_ms)
        if oi is None:
            continue
        T_map_lidar = pose_matrix(odom[oi])
        T_lidar_map = np.linalg.inv(T_map_lidar)
        pts_lidar = (T_lidar_map[:3, :3] @ points_map.T).T + T_lidar_map[:3, 3]
        pts_cam = (T_cam_lidar[:3, :3] @ pts_lidar.T).T + T_cam_lidar[:3, 3]
        front = pts_cam[:, 2] > min_cam_z
        if not np.any(front):
            continue
        uv, _ = cv2.projectPoints(pts_cam[front], np.zeros(3), np.zeros(3), K, dist)
        uv = uv.reshape(-1, 2)
        finite = np.isfinite(uv).all(axis=1) & (np.abs(uv).max(axis=1) < 1e7)
        if not np.any(finite):
            continue
        idx_all = np.flatnonzero(front)[finite]
        uv = uv[finite]
        h, w = img.shape[:2]
        u = np.round(uv[:, 0]).astype(np.int32)
        v = np.round(uv[:, 1]).astype(np.int32)
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if rgb_roi is not None:
            x0, y0, x1, y1 = rgb_roi
            inside &= (u >= x0) & (u < x1) & (v >= y0) & (v < y1)
            cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
            sx, sy = max((x1 - x0) * 0.35, 1.0), max((y1 - y0) * 0.35, 1.0)
        else:
            cx, cy = w * 0.5, h * 0.5
            sx, sy = w * 0.35, h * 0.35
        if not np.any(inside):
            continue
        idx = idx_all[inside]
        uu, vv = u[inside], v[inside]
        if reject_bright_low_sat:
            pix_bgr = img[vv, uu]
            hsv = cv2.cvtColor(pix_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
            keep_color = ~((hsv[:, 2] > 230) & (hsv[:, 1] < 55))
            if not np.any(keep_color):
                continue
            idx, uu, vv = idx[keep_color], uu[keep_color], vv[keep_color]
        # Prefer image-center observations; they have less lens distortion and
        # less chance of seeing rig/panel edges.
        sc = -(((uu - cx) / sx) ** 2 + ((vv - cy) / sy) ** 2).astype(np.float32)
        rr, cc = rows[idx], cols[idx]
        update = sc > score[rr, cc]
        if np.any(update):
            rr_u, cc_u = rr[update], cc[update]
            out[rr_u, cc_u] = img[vv[update], uu[update]][:, ::-1]
            score[rr_u, cc_u] = sc[update]
            used[rr_u, cc_u] = True
        stats["rgb_frames_used"] += 1
        stats["projected_cells"] += int(len(idx))
    stats["painted_cells"] = int(used.sum())
    return out, used, stats


def write_geotiff(path: Path, arr: np.ndarray, transform: Any, crs: str, dtype: str, nodata: float | int | None = None) -> str:
    count = 3 if arr.ndim == 3 else 1
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
    ) as dst:
        if arr.ndim == 3:
            for k in range(3):
                dst.write(arr[:, :, k].astype(dtype), k + 1)
        else:
            dst.write(arr.astype(dtype), 1)
    return str(path)


def write_outputs(out: Path, dsm: dict[str, Any], rgb_ortho: np.ndarray, rgb_used: np.ndarray, epsg: int) -> dict[str, Any]:
    qgis = out / "qgis"
    qgis.mkdir(parents=True, exist_ok=True)
    crs = f"EPSG:{epsg}"
    tr = dsm["transform"]
    valid = dsm["valid_filled"]
    height_color = cv2.cvtColor(cv2.applyColorMap(robust_u8(dsm["height_rel"], valid), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    intensity_color = cv2.cvtColor(cv2.applyColorMap(robust_u8(dsm["intensity"], valid), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    outputs: dict[str, Any] = {}
    outputs["dsm_local_z_m"] = write_geotiff(qgis / "fastlio_dsm_local_z_m.tif", dsm["dsm_filled"], tr, crs, "float32", np.nan)
    outputs["height_rel_m"] = write_geotiff(qgis / "fastlio_height_rel_m.tif", dsm["height_rel"], tr, crs, "float32", np.nan)
    outputs["intensity"] = write_geotiff(qgis / "fastlio_intensity.tif", dsm["intensity"], tr, crs, "float32", np.nan)
    outputs["density"] = write_geotiff(qgis / "fastlio_density.tif", dsm["density"], tr, crs, "uint16", 0)
    outputs["dsm_valid_mask"] = write_geotiff(qgis / "fastlio_dsm_valid_mask.tif", valid.astype(np.uint8) * 255, tr, crs, "uint8", 0)
    outputs["rgb_ortho"] = write_geotiff(qgis / "fastlio_rgb_orthoproject.tif", rgb_ortho, tr, crs, "uint8", 0)
    outputs["rgb_valid_mask"] = write_geotiff(qgis / "fastlio_rgb_orthoproject_valid_mask.tif", rgb_used.astype(np.uint8) * 255, tr, crs, "uint8", 0)
    outputs["height_color"] = write_geotiff(qgis / "fastlio_height_color.tif", height_color, tr, crs, "uint8", 0)
    outputs["intensity_color"] = write_geotiff(qgis / "fastlio_intensity_color.tif", intensity_color, tr, crs, "uint8", 0)
    cv2.imwrite(str(out / "fastlio_rgb_orthoproject_preview.jpg"), cv2.cvtColor(rgb_ortho, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 96])
    cv2.imwrite(str(out / "fastlio_height_preview.jpg"), cv2.cvtColor(height_color, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 96])
    h_rgb = max(1, int(700 * rgb_ortho.shape[0] / max(rgb_ortho.shape[1], 1)))
    overview = np.vstack([
        cv2.resize(cv2.cvtColor(rgb_ortho, cv2.COLOR_RGB2BGR), (700, h_rgb), interpolation=cv2.INTER_NEAREST),
        cv2.resize(cv2.cvtColor(height_color, cv2.COLOR_RGB2BGR), (700, h_rgb), interpolation=cv2.INTER_NEAREST),
    ])
    cv2.imwrite(str(out / "fastlio_dsm_orthoproject_overview.jpg"), overview, [cv2.IMWRITE_JPEG_QUALITY, 95])
    outputs["overview"] = str(out / "fastlio_dsm_orthoproject_overview.jpg")
    outputs["rgb_preview"] = str(out / "fastlio_rgb_orthoproject_preview.jpg")
    outputs["height_preview"] = str(out / "fastlio_height_preview.jpg")
    return outputs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--odometry-csv", type=Path, required=True)
    ap.add_argument("--fastlio-bag", type=Path, required=True)
    ap.add_argument("--original-bag", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, default=Path("data/calibration/new_session/20260623/calibration_20260623_final_candidate.json"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--align-mode", choices=["rigid", "similarity"], default="rigid")
    ap.add_argument("--max-gps-dt-ms", type=float, default=700.0)
    ap.add_argument("--cloud-every", type=int, default=2)
    ap.add_argument("--max-points-total", type=int, default=1500000)
    ap.add_argument("--resolution-m", type=float, default=0.03)
    ap.add_argument("--fill-radius-px", type=int, default=4)
    ap.add_argument("--surface-mode", choices=["max", "p95"], default="p95")
    ap.add_argument("--z-clip-pct", type=float, nargs=2, default=(1.0, 99.5))
    ap.add_argument("--strip-half-width-m", type=float, default=1.2)
    ap.add_argument("--strip-end-pad-m", type=float, default=0.4)
    ap.add_argument("--rgb-every", type=int, default=1)
    ap.add_argument("--max-odom-dt-ms", type=float, default=160.0)
    ap.add_argument("--min-cam-z", type=float, default=0.05)
    ap.add_argument("--invert-t-cam-lidar", action="store_true")
    ap.add_argument("--rgb-roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), default=(649, 1176, 1794, 1543))
    ap.add_argument("--no-rgb-roi", action="store_true", help="Diagnostic: project using the full RGB image instead of the crop ROI.")
    ap.add_argument("--keep-bright-low-sat", action="store_true", help="Keep white low-saturation pixels such as panels/robot parts.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    odom = load_odometry(args.odometry_csv)
    gps = load_gps(args.original_bag)
    epsg = utm_epsg(gps[0]["lon"], gps[0]["lat"])
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    pairs = associate_gps(odom, gps, args.max_gps_dt_ms)
    if len(pairs) < 3:
        raise SystemExit(f"not enough FAST-LIO/GPS associations: {len(pairs)}")
    for _idx, fix in pairs:
        fix["easting"], fix["northing"] = to_utm.transform(fix["lon"], fix["lat"])
    src = np.asarray([[odom[i]["x"], odom[i]["y"]] for i, _fix in pairs], dtype=np.float64)
    dst = np.asarray([[fix["easting"], fix["northing"]] for _i, fix in pairs], dtype=np.float64)
    scale, rot2, trans2 = rigid_2d(src, dst) if args.align_mode == "rigid" else similarity_2d(src, dst)
    sim_scale, sim_rot, sim_trans = similarity_2d(src, dst)
    align_res = alignment_residuals(src, dst, scale, rot2, trans2)
    sim_res = alignment_residuals(src, dst, sim_scale, sim_rot, sim_trans)
    traj_xy = transform_xy(np.asarray([[r["x"], r["y"]] for r in odom], dtype=np.float64), scale, rot2, trans2)
    xyz_utm, intensity = collect_map_points(
        args.fastlio_bag,
        scale,
        rot2,
        trans2,
        args.cloud_every,
        args.max_points_total,
        tuple(args.z_clip_pct),
    )
    xyz_utm, intensity, corridor_stats = filter_to_trajectory_corridor(
        xyz_utm,
        intensity,
        traj_xy,
        args.strip_half_width_m,
        args.strip_end_pad_m,
    )
    if len(xyz_utm) == 0:
        raise SystemExit(f"trajectory corridor removed all LiDAR points: {corridor_stats}")
    dsm = build_dsm(xyz_utm, intensity, args.resolution_m, args.fill_radius_px, args.surface_mode)
    t0, t1 = min(r["stamp_ns"] for r in odom), max(r["stamp_ns"] for r in odom)
    rgb_stamps, rgb_images = load_rgb_frames(args.original_bag, int(t0 - 1_000_000_000), int(t1 + 1_000_000_000), args.rgb_every)
    calib = load_calibration(args.calibration)
    rgb_ortho, rgb_used, rgb_stats = orthoproject_rgb(
        dsm,
        odom,
        rgb_stamps,
        rgb_images,
        calib,
        scale,
        rot2,
        trans2,
        args.max_odom_dt_ms,
        None if args.no_rgb_roi else (tuple(args.rgb_roi) if args.rgb_roi else None),
        args.min_cam_z,
        args.invert_t_cam_lidar,
        not args.keep_bright_low_sat,
    )
    outputs = write_outputs(args.out, dsm, rgb_ortho, rgb_used, epsg)
    to_wgs = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    traj = gpd.GeoDataFrame(
        {"kind": ["fastlio_pose_rigid" if args.align_mode == "rigid" else "fastlio_pose_similarity"]},
        geometry=[LineString([Point(*to_wgs.transform(float(e), float(n))) for e, n in traj_xy])],
        crs="EPSG:4326",
    )
    traj_path = args.out / "fastlio_dsm_trajectory_wgs84.geojson"
    traj.to_file(traj_path, driver="GeoJSON")
    summary = {
        "odometry_csv": str(args.odometry_csv),
        "fastlio_bag": str(args.fastlio_bag),
        "original_bag": str(args.original_bag),
        "calibration": str(args.calibration),
        "crs": f"EPSG:{epsg}",
        "alignment": {
            "mode": args.align_mode,
            "scale_used": scale,
            "rotation_2d": rot2.tolist(),
            "translation_utm_m": trans2.tolist(),
            "residual_used": align_res,
            "diagnostic_similarity_scale": sim_scale,
            "diagnostic_similarity_residual": sim_res,
        },
        "dsm": {
            "source_points": int(len(xyz_utm)),
            "trajectory_corridor_filter": corridor_stats,
            "resolution_m": args.resolution_m,
            "fill_radius_px": args.fill_radius_px,
            "surface_mode": args.surface_mode,
            "shape_hw": list(map(int, dsm["shape_hw"])),
            "valid_cells_raw": int(dsm["valid"].sum()),
            "valid_cells_filled": int(dsm["valid_filled"].sum()),
            "ground_z_p05_m": dsm["ground_z_p05_m"],
        },
        "rgb_orthoproject": {
            **rgb_stats,
            "rgb_frames_loaded": len(rgb_images),
            "painted_fraction_of_filled_dsm": float(rgb_used.sum() / max(dsm["valid_filled"].sum(), 1)),
            "rgb_roi": None if args.no_rgb_roi else (list(args.rgb_roi) if args.rgb_roi else None),
            "invert_t_cam_lidar": bool(args.invert_t_cam_lidar),
            "reject_bright_low_sat": not args.keep_bright_low_sat,
        },
        "outputs": outputs | {"trajectory_wgs84": str(traj_path)},
        "notes": [
            "DSM/height/intensity geometry comes from FAST-LIO LiDAR map.",
            "RGB is orthoprojected onto the DSM with FAST-LIO pose(t), not image homography stitching.",
            "Default alignment is rigid SE(2), preserving FAST-LIO metric scale.",
        ],
    }
    (args.out / "fastlio_dsm_orthoproject_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
