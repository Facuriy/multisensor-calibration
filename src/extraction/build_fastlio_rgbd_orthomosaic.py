#!/usr/bin/env python3
"""Build a first RGB-D orthomosaic from FAST-LIO/GNSS pose and RGB projection.

This is an inspection product, not the final photogrammetric pipeline.  It uses:

* FAST-LIO /Odometry exported to CSV;
* FAST-LIO /cloud_registered points in the local FAST-LIO map frame;
* GNSS fixes from the original bag to align FAST-LIO XY to UTM;
* the calibrated Ouster->RGB transform to color each 3D point from the nearest
  RGB frame.

The output is a top-down raster in UTM: RGB color, relative height, intensity
and density.  This avoids the perspective fan introduced by image homographies;
geometry comes from the LiDAR map and pose.
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
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation as R
from shapely.geometry import LineString, Point
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction.export_fastlio_gps_qgis import (  # noqa: E402
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
CLOUD_TOPIC = "/cloud_registered"


def load_calibration(path: Path) -> dict[str, np.ndarray]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    rgb = doc["intrinsics"]["rgb"]
    return {
        "K": np.asarray(rgb["K"], dtype=np.float64),
        "dist": np.asarray(rgb.get("dist_coeffs", [0, 0, 0, 0, 0]), dtype=np.float64).reshape(-1, 1),
        "T_cam_lidar": np.asarray(doc["ouster_rgb"]["T_cam_lidar"]["matrix"], dtype=np.float64),
    }


def pose_matrix(row: dict[str, float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat([row["qx"], row["qy"], row["qz"], row["qw"]]).as_matrix()
    T[:3, 3] = [row["x"], row["y"], row["z"]]
    return T


def nearest_index(stamps: np.ndarray, stamp_ns: int, max_dt_ms: float) -> int | None:
    if len(stamps) == 0:
        return None
    idx = int(np.argmin(np.abs(stamps - int(stamp_ns))))
    if abs(int(stamps[idx]) - int(stamp_ns)) / 1e6 > max_dt_ms:
        return None
    return idx


def load_rgb_frames(original_bag: Path, start_ns: int, stop_ns: int, max_frames: int = 0) -> tuple[np.ndarray, list[np.ndarray]]:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    stamps: list[int] = []
    imgs: list[np.ndarray] = []
    with Reader(original_bag) as reader:
        conns = [c for c in reader.connections if c.topic == RGB_TOPIC]
        for conn, ts, raw in reader.messages(connections=conns):
            if int(ts) < start_ns or int(ts) > stop_ns:
                continue
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            img = decode_ros_image(msg)
            if img is None or img.ndim != 3:
                continue
            stamps.append(msg_stamp_ns(msg, int(ts)))
            imgs.append(img)
            if max_frames > 0 and len(imgs) >= max_frames:
                break
    return np.asarray(stamps, dtype=np.int64), imgs


def colorize_clouds(
    fastlio_bag: Path,
    odom: list[dict[str, float]],
    rgb_stamps: np.ndarray,
    rgb_imgs: list[np.ndarray],
    calib: dict[str, np.ndarray],
    scale: float,
    rot2: np.ndarray,
    trans2: np.ndarray,
    cloud_every: int,
    max_points_total: int,
    max_odom_dt_ms: float,
    max_rgb_dt_ms: float,
    min_cam_z: float,
    invert_t: bool,
    rgb_roi: tuple[int, int, int, int] | None,
) -> dict[str, np.ndarray]:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    odom_stamps = np.asarray([r["stamp_ns"] for r in odom], dtype=np.int64)
    T_cam_lidar = np.linalg.inv(calib["T_cam_lidar"]) if invert_t else calib["T_cam_lidar"]
    K, dist = calib["K"], calib["dist"]
    xyz_utm_all: list[np.ndarray] = []
    rgb_all: list[np.ndarray] = []
    intensity_all: list[np.ndarray] = []
    depth_all: list[np.ndarray] = []
    stats = {
        "clouds_seen": 0,
        "clouds_used": 0,
        "raw_points": 0,
        "projected_points": 0,
        "colored_points": 0,
    }
    with Reader(fastlio_bag) as reader:
        conns = [c for c in reader.connections if c.topic == CLOUD_TOPIC]
        for conn, ts, raw in reader.messages(connections=conns):
            ci = stats["clouds_seen"]
            stats["clouds_seen"] += 1
            if ci % max(1, cloud_every) != 0:
                continue
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            stamp = msg_stamp_ns(msg, int(ts))
            oi = nearest_index(odom_stamps, stamp, max_odom_dt_ms)
            ri = nearest_index(rgb_stamps, stamp, max_rgb_dt_ms)
            if oi is None or ri is None:
                continue
            xyz_local, intensity = pointcloud_xyz_i(msg)
            if len(xyz_local) == 0:
                continue
            stats["raw_points"] += int(len(xyz_local))
            T_world_lidar = pose_matrix(odom[oi])
            T_lidar_world = np.linalg.inv(T_world_lidar)
            pts_lidar = (T_lidar_world[:3, :3] @ xyz_local.T).T + T_lidar_world[:3, 3]
            pts_cam = (T_cam_lidar[:3, :3] @ pts_lidar.T).T + T_cam_lidar[:3, 3]
            front = pts_cam[:, 2] > min_cam_z
            if not np.any(front):
                continue
            pts_cam_f = pts_cam[front]
            xyz_local_f = xyz_local[front]
            intensity_f = intensity[front]
            uv, _ = cv2.projectPoints(pts_cam_f, np.zeros(3), np.zeros(3), K, dist)
            uv = uv.reshape(-1, 2)
            finite_uv = np.isfinite(uv).all(axis=1)
            if not np.any(finite_uv):
                continue
            uv = uv[finite_uv]
            pts_cam_f = pts_cam_f[finite_uv]
            xyz_local_f = xyz_local_f[finite_uv]
            intensity_f = intensity_f[finite_uv]
            img = rgb_imgs[ri]
            h, w = img.shape[:2]
            u = np.round(uv[:, 0]).astype(np.int32)
            v = np.round(uv[:, 1]).astype(np.int32)
            inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            if rgb_roi is not None:
                x0, y0, x1, y1 = rgb_roi
                inside &= (u >= x0) & (u < x1) & (v >= y0) & (v < y1)
            stats["projected_points"] += int(len(uv))
            if not np.any(inside):
                continue
            xyz_sel = xyz_local_f[inside]
            xy_utm = transform_xy(xyz_sel[:, :2].astype(np.float64), scale, rot2, trans2)
            xyz_utm = np.column_stack([xy_utm, xyz_sel[:, 2]]).astype(np.float32)
            # decode_ros_image returns BGR for RGB camera; GeoTIFF/JPG preview wants RGB order.
            color_rgb = img[v[inside], u[inside]][:, ::-1].astype(np.uint8)
            xyz_utm_all.append(xyz_utm)
            rgb_all.append(color_rgb)
            intensity_all.append(intensity_f[inside].astype(np.float32))
            depth_all.append(pts_cam_f[inside, 2].astype(np.float32))
            stats["clouds_used"] += 1
    if not xyz_utm_all:
        raise RuntimeError(f"no RGB-colored FAST-LIO points produced; stats={stats}")
    xyz = np.vstack(xyz_utm_all)
    rgb = np.vstack(rgb_all)
    intensity = np.concatenate(intensity_all)
    depth = np.concatenate(depth_all)
    if max_points_total > 0 and len(xyz) > max_points_total:
        rng = np.random.default_rng(17)
        take = rng.choice(len(xyz), int(max_points_total), replace=False)
        xyz, rgb, intensity, depth = xyz[take], rgb[take], intensity[take], depth[take]
    stats["colored_points"] = int(len(xyz))
    return {"xyz": xyz, "rgb": rgb, "intensity": intensity, "depth": depth, "stats": stats}


def robust_u8(arr: np.ndarray, valid: np.ndarray | None = None, lo_hi: tuple[float, float] = (2, 98)) -> np.ndarray:
    valid = np.isfinite(arr) if valid is None else (valid & np.isfinite(arr))
    out = np.zeros(arr.shape, np.uint8)
    if not np.any(valid):
        return out
    lo, hi = np.percentile(arr[valid], lo_hi)
    if hi <= lo:
        hi = lo + 1.0
    out[valid] = np.clip((arr[valid] - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return out


def nearest_fill(arr: np.ndarray, valid: np.ndarray, radius_px: int) -> tuple[np.ndarray, np.ndarray]:
    if radius_px <= 0 or not np.any(valid):
        return arr, valid
    dist, indices = distance_transform_edt(~valid, return_indices=True)
    fill = (~valid) & (dist <= radius_px)
    if arr.ndim == 3:
        out = arr.copy()
        out[fill] = arr[indices[0][fill], indices[1][fill]]
    else:
        out = arr.copy()
        out[fill] = arr[indices[0][fill], indices[1][fill]]
    return out, valid | fill


def rasterize_rgbd(points: dict[str, np.ndarray], out: Path, epsg: int, res: float, fill_radius_px: int) -> dict[str, str]:
    xyz = points["xyz"]
    rgb = points["rgb"]
    intensity = points["intensity"]
    depth = points["depth"]
    e, n, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    pad = res * 2
    emin, emax = float(e.min() - pad), float(e.max() + pad)
    nmin, nmax = float(n.min() - pad), float(n.max() + pad)
    width = max(1, int(math.ceil((emax - emin) / res)))
    height = max(1, int(math.ceil((nmax - nmin) / res)))
    col = np.clip(((e - emin) / res).astype(np.int32), 0, width - 1)
    row = np.clip(((nmax - n) / res).astype(np.int32), 0, height - 1)
    flat = row * width + col
    order = np.lexsort((z, flat))
    last = np.r_[flat[order][1:] != flat[order][:-1], True]
    keep = order[last]
    rgb_r = np.zeros((height * width, 3), np.uint8)
    z_r = np.full(height * width, np.nan, np.float32)
    depth_r = np.full(height * width, np.nan, np.float32)
    int_r = np.full(height * width, np.nan, np.float32)
    dens = np.zeros(height * width, np.uint16)
    np.add.at(dens, flat, 1)
    rgb_r[flat[keep]] = rgb[keep]
    z_r[flat[keep]] = z[keep]
    depth_r[flat[keep]] = depth[keep]
    int_r[flat[keep]] = intensity[keep]
    rgb_img = rgb_r.reshape(height, width, 3)
    z_img = z_r.reshape(height, width)
    depth_img = depth_r.reshape(height, width)
    int_img = int_r.reshape(height, width)
    dens_img = dens.reshape(height, width)
    valid = np.isfinite(z_img)
    ground = float(np.nanpercentile(z_img, 5)) if np.any(valid) else 0.0
    height_rel = (z_img - ground).astype(np.float32)
    if fill_radius_px > 0:
        rgb_img, valid_filled = nearest_fill(rgb_img, valid, fill_radius_px)
        height_rel, _ = nearest_fill(height_rel, valid, fill_radius_px)
        depth_img, _ = nearest_fill(depth_img, valid, fill_radius_px)
        int_img, _ = nearest_fill(int_img, valid, fill_radius_px)
        valid = valid_filled
    transform = from_origin(emin, nmax, res, res)
    qgis = out / "qgis"
    qgis.mkdir(parents=True, exist_ok=True)
    crs = f"EPSG:{epsg}"
    outputs: dict[str, str] = {}

    def write_tif(name: str, arr: np.ndarray, dtype: str, nodata: float | int | None = None) -> str:
        path = qgis / name
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

    outputs["rgb_ortho"] = write_tif("fastlio_rgbd_rgb.tif", rgb_img, "uint8", 0)
    outputs["height_rel_m"] = write_tif("fastlio_rgbd_height_rel_m.tif", height_rel, "float32", np.nan)
    outputs["depth_m"] = write_tif("fastlio_rgbd_depth_m.tif", depth_img, "float32", np.nan)
    outputs["intensity"] = write_tif("fastlio_rgbd_intensity.tif", int_img, "float32", np.nan)
    outputs["density"] = write_tif("fastlio_rgbd_density.tif", dens_img, "uint16", 0)
    height_color = cv2.cvtColor(cv2.applyColorMap(robust_u8(height_rel, valid), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    int_color = cv2.cvtColor(cv2.applyColorMap(robust_u8(int_img, np.isfinite(int_img)), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    outputs["height_color"] = write_tif("fastlio_rgbd_height_color.tif", height_color, "uint8", 0)
    outputs["intensity_color"] = write_tif("fastlio_rgbd_intensity_color.tif", int_color, "uint8", 0)
    cv2.imwrite(str(out / "fastlio_rgbd_rgb_preview.jpg"), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 96])
    cv2.imwrite(str(out / "fastlio_rgbd_height_preview.jpg"), cv2.cvtColor(height_color, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 96])
    overview = np.vstack([
        cv2.resize(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR), (700, max(1, int(700 * rgb_img.shape[0] / rgb_img.shape[1]))), interpolation=cv2.INTER_AREA),
        cv2.resize(cv2.cvtColor(height_color, cv2.COLOR_RGB2BGR), (700, max(1, int(700 * height_color.shape[0] / height_color.shape[1]))), interpolation=cv2.INTER_AREA),
    ])
    cv2.imwrite(str(out / "fastlio_rgbd_overview.jpg"), overview, [cv2.IMWRITE_JPEG_QUALITY, 95])
    outputs["overview"] = str(out / "fastlio_rgbd_overview.jpg")
    outputs["rgb_preview"] = str(out / "fastlio_rgbd_rgb_preview.jpg")
    outputs["height_preview"] = str(out / "fastlio_rgbd_height_preview.jpg")
    outputs["raster_shape_hw"] = [int(height), int(width)]
    outputs["ground_z_p05_m"] = ground
    outputs["fill_radius_px"] = int(fill_radius_px)
    return outputs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--odometry-csv", type=Path, required=True)
    ap.add_argument("--fastlio-bag", type=Path, required=True)
    ap.add_argument("--original-bag", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, default=Path("data/calibration/new_session/20260623/calibration_20260623_final_candidate.json"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-gps-dt-ms", type=float, default=700.0)
    ap.add_argument("--max-rgb-dt-ms", type=float, default=250.0)
    ap.add_argument("--max-odom-dt-ms", type=float, default=120.0)
    ap.add_argument("--cloud-every", type=int, default=4)
    ap.add_argument("--max-points-total", type=int, default=650000)
    ap.add_argument("--resolution-m", type=float, default=0.015)
    ap.add_argument("--fill-radius-px", type=int, default=0)
    ap.add_argument("--min-cam-z", type=float, default=0.05)
    ap.add_argument("--invert-t-cam-lidar", action="store_true")
    ap.add_argument("--rgb-roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), help="Optional RGB pixel ROI for color sampling.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    odom = load_odometry(args.odometry_csv)
    gps = load_gps(args.original_bag)
    if len(odom) < 3:
        raise SystemExit("not enough odometry rows")
    if len(gps) < 3:
        raise SystemExit("not enough GPS fixes")
    epsg = utm_epsg(gps[0]["lon"], gps[0]["lat"])
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    pairs = associate_gps(odom, gps, args.max_gps_dt_ms)
    for _idx, fix in pairs:
        fix["easting"], fix["northing"] = to_utm.transform(fix["lon"], fix["lat"])
    src = np.asarray([[odom[i]["x"], odom[i]["y"]] for i, _fix in pairs], dtype=np.float64)
    dst = np.asarray([[fix["easting"], fix["northing"]] for _i, fix in pairs], dtype=np.float64)
    scale, rot2, trans2 = similarity_2d(src, dst)
    t0, t1 = min(r["stamp_ns"] for r in odom), max(r["stamp_ns"] for r in odom)
    rgb_stamps, rgb_imgs = load_rgb_frames(args.original_bag, int(t0 - 1_000_000_000), int(t1 + 1_000_000_000))
    if len(rgb_imgs) == 0:
        raise SystemExit("no RGB frames found around FAST-LIO interval")
    calib = load_calibration(args.calibration)
    pts = colorize_clouds(
        args.fastlio_bag,
        odom,
        rgb_stamps,
        rgb_imgs,
        calib,
        scale,
        rot2,
        trans2,
        args.cloud_every,
        args.max_points_total,
        args.max_odom_dt_ms,
        args.max_rgb_dt_ms,
        args.min_cam_z,
        args.invert_t_cam_lidar,
        tuple(args.rgb_roi) if args.rgb_roi else None,
    )
    outputs = rasterize_rgbd(pts, args.out, epsg, args.resolution_m, args.fill_radius_px)
    trajectory = gpd.GeoDataFrame(
        {"kind": ["fastlio_rgbd_track"]},
        geometry=[
            LineString(
                [
                    Point(*Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform(e, n))
                    for e, n in transform_xy(np.asarray([[r["x"], r["y"]] for r in odom], dtype=np.float64), scale, rot2, trans2)
                ]
            )
        ],
        crs="EPSG:4326",
    )
    traj_path = args.out / "fastlio_rgbd_trajectory_wgs84.geojson"
    trajectory.to_file(traj_path, driver="GeoJSON")
    summary = {
        "odometry_csv": str(args.odometry_csv),
        "fastlio_bag": str(args.fastlio_bag),
        "original_bag": str(args.original_bag),
        "calibration": str(args.calibration),
        "crs": f"EPSG:{epsg}",
        "rgb_frames_loaded": len(rgb_imgs),
        "gps_associations": len(pairs),
        "similarity": {"scale": scale, "rotation_2d": rot2.tolist(), "translation_utm_m": trans2.tolist()},
        "color_projection_stats": pts["stats"],
        "raster_outputs": outputs,
        "trajectory_wgs84": str(traj_path),
        "notes": [
            "RGB-D orthomosaic: geometry from FAST-LIO cloud/pose; RGB sampled by Ouster->RGB projection.",
            "FAST-LIO XY is aligned to GNSS by 2D similarity, not yet full GPS-factor graph.",
            "Use as diagnostic before production pose-graph orthorectification.",
        ],
    }
    (args.out / "fastlio_rgbd_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
