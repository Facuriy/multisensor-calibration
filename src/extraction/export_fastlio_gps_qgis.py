#!/usr/bin/env python3
"""Georeference FAST-LIO output with GNSS and export QGIS layers.

Inputs:
  - FAST-LIO odometry.csv exported from /Odometry
  - FAST-LIO output bag containing /cloud_registered
  - original field bag containing /ssf/gnss/fix

The output is an inspection product: FAST-LIO local XY is aligned to GNSS UTM
with a 2D similarity transform. This gives QGIS-ready trajectory and LiDAR
rasters, but it is not a full GPS-factor pose graph.
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
from rasterio.transform import from_bounds
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from shapely.geometry import LineString, Point

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


GPS_TOPIC = "/ssf/gnss/fix"
CLOUD_TOPIC = "/cloud_registered"

POINT_DTYPES = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}


def msg_stamp_ns(msg: Any, fallback: int) -> int:
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None:
        return int(fallback)
    sec = int(getattr(stamp, "sec", 0))
    nsec = int(getattr(stamp, "nanosec", getattr(stamp, "nsec", 0)))
    out = sec * 1_000_000_000 + nsec
    return int(out if out > 0 else fallback)


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180.0) // 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


def load_odometry(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "stamp_ns": int(float(row["field.header.stamp"])),
                    "x": float(row["field.pose.pose.position.x"]),
                    "y": float(row["field.pose.pose.position.y"]),
                    "z": float(row["field.pose.pose.position.z"]),
                    "qx": float(row["field.pose.pose.orientation.x"]),
                    "qy": float(row["field.pose.pose.orientation.y"]),
                    "qz": float(row["field.pose.pose.orientation.z"]),
                    "qw": float(row["field.pose.pose.orientation.w"]),
                }
            )
    return rows


def load_gps(original_bag: Path) -> list[dict[str, float]]:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    fixes = []
    with Reader(original_bag) as reader:
        conns = [c for c in reader.connections if c.topic == GPS_TOPIC]
        for conn, ts, raw in reader.messages(connections=conns):
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            status = getattr(getattr(msg, "status", None), "status", 0)
            if int(status) < 0:
                continue
            lat = float(getattr(msg, "latitude"))
            lon = float(getattr(msg, "longitude"))
            alt = float(getattr(msg, "altitude", 0.0))
            if not np.isfinite([lat, lon, alt]).all():
                continue
            fixes.append({"stamp_ns": msg_stamp_ns(msg, int(ts)), "lat": lat, "lon": lon, "alt": alt})
    return fixes


def associate_gps(odom: list[dict[str, float]], gps: list[dict[str, float]], max_dt_ms: float) -> list[tuple[int, dict[str, float]]]:
    out = []
    gps_stamps = np.asarray([g["stamp_ns"] for g in gps], dtype=np.int64)
    for i, row in enumerate(odom):
        if len(gps_stamps) == 0:
            break
        j = int(np.argmin(np.abs(gps_stamps - int(row["stamp_ns"]))))
        dt_ms = abs(int(gps_stamps[j]) - int(row["stamp_ns"])) / 1e6
        if dt_ms <= max_dt_ms:
            out.append((i, gps[j] | {"dt_ms": float(dt_ms)}))
    return out


def similarity_2d(src_xy: np.ndarray, dst_xy: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src_c = src_xy.mean(axis=0)
    dst_c = dst_xy.mean(axis=0)
    xs = src_xy - src_c
    xd = dst_xy - dst_c
    norm = float((xs * xs).sum())
    if norm <= 1e-12:
        return 1.0, np.eye(2), dst_c - src_c
    u, s, vt = np.linalg.svd(xs.T @ xd)
    r = u @ vt
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = u @ vt
    scale = float(s.sum() / norm)
    t = dst_c - scale * (src_c @ r)
    return scale, r, t


def transform_xy(xy: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return scale * (xy @ rot) + trans


def pointcloud_xyz_i(msg: Any) -> tuple[np.ndarray, np.ndarray]:
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    step = int(msg.point_step)
    count = int(msg.width) * int(msg.height)
    offsets = {str(f.name): int(f.offset) for f in msg.fields}
    types = {str(f.name): int(f.datatype) for f in msg.fields}
    cols = {}
    for name in ("x", "y", "z", "intensity"):
        if name not in offsets:
            continue
        dtype = np.dtype(POINT_DTYPES.get(types[name], np.float32))
        cols[name] = np.ndarray((count,), dtype=dtype, buffer=raw, offset=offsets[name], strides=(step,)).copy()
    if not all(k in cols for k in ("x", "y", "z")):
        return np.empty((0, 3), np.float32), np.empty((0,), np.float32)
    xyz = np.column_stack([cols["x"], cols["y"], cols["z"]]).astype(np.float32)
    intensity = np.asarray(cols.get("intensity", np.zeros(count, np.float32)), dtype=np.float32)
    ok = np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) > 0.2)
    return xyz[ok], intensity[ok]


def collect_fastlio_clouds(
    fastlio_bag: Path,
    scale: float,
    rot: np.ndarray,
    trans: np.ndarray,
    every: int,
    max_points_total: int,
) -> tuple[np.ndarray, np.ndarray]:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    all_xyz = []
    all_int = []
    idx = 0
    with Reader(fastlio_bag) as reader:
        conns = [c for c in reader.connections if c.topic == CLOUD_TOPIC]
        for conn, _ts, raw in reader.messages(connections=conns):
            if idx % max(1, every) != 0:
                idx += 1
                continue
            idx += 1
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            xyz, intensity = pointcloud_xyz_i(msg)
            if len(xyz) == 0:
                continue
            xy_utm = transform_xy(xyz[:, :2].astype(np.float64), scale, rot, trans)
            all_xyz.append(np.column_stack([xy_utm, xyz[:, 2]]).astype(np.float32))
            all_int.append(intensity.astype(np.float32))
    if not all_xyz:
        return np.empty((0, 3), np.float32), np.empty((0,), np.float32)
    xyz = np.vstack(all_xyz)
    intensity = np.concatenate(all_int)
    if max_points_total > 0 and len(xyz) > max_points_total:
        rng = np.random.default_rng(11)
        take = rng.choice(len(xyz), int(max_points_total), replace=False)
        xyz = xyz[take]
        intensity = intensity[take]
    return xyz, intensity


def write_pose_outputs(
    odom: list[dict[str, float]],
    pairs: list[tuple[int, dict[str, float]]],
    epsg: int,
    scale: float,
    rot: np.ndarray,
    trans: np.ndarray,
    out: Path,
) -> dict[str, Any]:
    to_wgs = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    local_xy = np.asarray([[r["x"], r["y"]] for r in odom], dtype=np.float64)
    utm_xy = transform_xy(local_xy, scale, rot, trans)
    paired = {idx: fix for idx, fix in pairs}
    rows = []
    csv_path = out / "fastlio_pose_utm.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "index",
            "stamp_ns",
            "easting_m",
            "northing_m",
            "z_fastlio_m",
            "lon",
            "lat",
            "qx_local",
            "qy_local",
            "qz_local",
            "qw_local",
            "gps_lon",
            "gps_lat",
            "gps_dt_ms",
            "gps_residual_m",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        gps_residuals = []
        for i, (row, xy) in enumerate(zip(odom, utm_xy)):
            lon, lat = to_wgs.transform(float(xy[0]), float(xy[1]))
            rec = {
                "index": i,
                "stamp_ns": int(row["stamp_ns"]),
                "easting_m": float(xy[0]),
                "northing_m": float(xy[1]),
                "z_fastlio_m": float(row["z"]),
                "lon": float(lon),
                "lat": float(lat),
                "qx_local": row["qx"],
                "qy_local": row["qy"],
                "qz_local": row["qz"],
                "qw_local": row["qw"],
                "gps_lon": "",
                "gps_lat": "",
                "gps_dt_ms": "",
                "gps_residual_m": "",
            }
            if i in paired:
                fix = paired[i]
                gps_xy = np.asarray([[fix["easting"], fix["northing"]]], dtype=np.float64)[0]
                res = float(np.linalg.norm(xy - gps_xy))
                gps_residuals.append(res)
                rec.update({"gps_lon": fix["lon"], "gps_lat": fix["lat"], "gps_dt_ms": fix["dt_ms"], "gps_residual_m": res})
            writer.writerow(rec)
            rows.append(rec)

    line = LineString([(r["lon"], r["lat"]) for r in rows])
    points = [Point(r["lon"], r["lat"]) for r in rows]
    line_path = out / "fastlio_trajectory_wgs84.geojson"
    points_path = out / "fastlio_pose_points_wgs84.geojson"
    gpd.GeoDataFrame([{"kind": "fastlio_gps_aligned_trajectory", "geometry": line}], crs="EPSG:4326").to_file(line_path, driver="GeoJSON")
    gpd.GeoDataFrame([{**r, "geometry": p} for r, p in zip(rows, points)], crs="EPSG:4326").to_file(points_path, driver="GeoJSON")

    residuals = np.asarray(gps_residuals, dtype=np.float64)
    return {
        "pose_csv": str(csv_path),
        "trajectory_wgs84_geojson": str(line_path),
        "pose_points_wgs84_geojson": str(points_path),
        "gps_residual_mean_m": float(residuals.mean()) if len(residuals) else None,
        "gps_residual_median_m": float(np.median(residuals)) if len(residuals) else None,
        "gps_residual_max_m": float(residuals.max()) if len(residuals) else None,
    }


def colorize_u8(u8: np.ndarray, cmap: int) -> np.ndarray:
    mask = u8 > 0
    color = cv2.applyColorMap(u8, cmap)
    color[~mask] = 0
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def rasterize_points(points_utm: np.ndarray, intensity: np.ndarray, out: Path, epsg: int, resolution: float) -> dict[str, str]:
    qgis = out / "qgis"
    qgis.mkdir(parents=True, exist_ok=True)
    if len(points_utm) == 0:
        return {}
    x, y, z = points_utm[:, 0], points_utm[:, 1], points_utm[:, 2]
    minx, maxx = float(np.nanmin(x) - 0.5), float(np.nanmax(x) + 0.5)
    miny, maxy = float(np.nanmin(y) - 0.5), float(np.nanmax(y) + 0.5)
    w = max(1, int(math.ceil((maxx - minx) / resolution)))
    h = max(1, int(math.ceil((maxy - miny) / resolution)))
    col = np.clip(((x - minx) / resolution).astype(np.int64), 0, w - 1)
    row = np.clip(((maxy - y) / resolution).astype(np.int64), 0, h - 1)
    idx = row * w + col

    order = np.argsort(z)
    flat_h = np.full(w * h, np.nan, dtype=np.float32)
    flat_i = np.full(w * h, np.nan, dtype=np.float32)
    flat_h[idx[order]] = z[order].astype(np.float32)
    flat_i[idx[order]] = intensity[order].astype(np.float32)
    density = np.bincount(idx, minlength=w * h).astype(np.float32).reshape(h, w)
    height = flat_h.reshape(h, w)
    inten = flat_i.reshape(h, w)

    transform = from_bounds(minx, miny, maxx, maxy, w, h)
    outputs = {}
    for name, arr in (("fastlio_height_m", height), ("fastlio_intensity", inten), ("fastlio_density", density)):
        tif = qgis / f"{name}.tif"
        nodata = -9999.0
        data = np.where(np.isfinite(arr), arr, nodata).astype(np.float32)
        with rasterio.open(
            tif,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype="float32",
            crs=f"EPSG:{epsg}",
            transform=transform,
            nodata=nodata,
            compress="deflate",
        ) as ds:
            ds.write(data, 1)
            ds.update_tags(layer=name, georef_quality="FAST_LIO_2D_similarity_to_GNSS")
        outputs[name] = str(tif)

    def robust_u8(arr: np.ndarray) -> np.ndarray:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros(arr.shape, np.uint8)
        lo, hi = np.percentile(finite, [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        return np.where(np.isfinite(arr), np.clip((arr - lo) * 255 / (hi - lo), 0, 255), 0).astype(np.uint8)

    previews = {
        "fastlio_height_color": colorize_u8(robust_u8(height), cv2.COLORMAP_TURBO),
        "fastlio_intensity_color": colorize_u8(robust_u8(inten), cv2.COLORMAP_INFERNO),
        "fastlio_density_color": colorize_u8(robust_u8(density), cv2.COLORMAP_VIRIDIS),
    }
    for name, img in previews.items():
        jpg = out / f"{name}.jpg"
        cv2.imwrite(str(jpg), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        tif = qgis / f"{name}.tif"
        with rasterio.open(
            tif,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=3,
            dtype="uint8",
            crs=f"EPSG:{epsg}",
            transform=transform,
            compress="deflate",
            photometric="RGB",
        ) as ds:
            ds.write(img.transpose(2, 0, 1))
            ds.update_tags(layer=name, georef_quality="FAST_LIO_2D_similarity_to_GNSS")
        outputs[name] = str(tif)
    return outputs


def write_overview(out: Path) -> str | None:
    imgs = []
    for name, title in [
        ("fastlio_intensity_color.jpg", "FAST-LIO intensity"),
        ("fastlio_height_color.jpg", "FAST-LIO height"),
        ("fastlio_density_color.jpg", "FAST-LIO density"),
    ]:
        img = cv2.imread(str(out / name), cv2.IMREAD_COLOR)
        if img is None:
            continue
        if img.shape[1] > 700:
            scale = 700 / img.shape[1]
            img = cv2.resize(img, (700, max(1, int(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (20, 20, 20), -1)
        cv2.putText(img, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        imgs.append(img)
    if not imgs:
        return None
    w = max(i.shape[1] for i in imgs)
    padded = [cv2.copyMakeBorder(i, 0, 0, 0, w - i.shape[1], cv2.BORDER_CONSTANT, value=(245, 245, 245)) for i in imgs]
    path = out / "fastlio_qgis_overview.jpg"
    cv2.imwrite(str(path), np.vstack(padded), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return str(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--odometry-csv", type=Path, required=True)
    ap.add_argument("--fastlio-bag", type=Path, required=True)
    ap.add_argument("--original-bag", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-gps-dt-ms", type=float, default=600.0)
    ap.add_argument("--cloud-every", type=int, default=4)
    ap.add_argument("--max-points-total", type=int, default=700000)
    ap.add_argument("--resolution-m", type=float, default=0.025)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    odom = load_odometry(args.odometry_csv)
    gps = load_gps(args.original_bag)
    if len(odom) < 2:
        raise SystemExit("not enough odometry rows")
    if len(gps) < 3:
        raise SystemExit("not enough GPS fixes")
    epsg = utm_epsg(gps[0]["lon"], gps[0]["lat"])
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    pairs = associate_gps(odom, gps, args.max_gps_dt_ms)
    if len(pairs) < 3:
        raise SystemExit(f"not enough odom/GPS associations: {len(pairs)}")
    for _idx, fix in pairs:
        fix["easting"], fix["northing"] = to_utm.transform(fix["lon"], fix["lat"])
    src = np.asarray([[odom[i]["x"], odom[i]["y"]] for i, _fix in pairs], dtype=np.float64)
    dst = np.asarray([[fix["easting"], fix["northing"]] for _i, fix in pairs], dtype=np.float64)
    scale, rot, trans = similarity_2d(src, dst)
    pose_outputs = write_pose_outputs(odom, pairs, epsg, scale, rot, trans, args.out)
    points, intensity = collect_fastlio_clouds(args.fastlio_bag, scale, rot, trans, args.cloud_every, args.max_points_total)
    raster_outputs = rasterize_points(points, intensity, args.out, epsg, args.resolution_m)
    overview = write_overview(args.out)
    summary = {
        "odometry_csv": str(args.odometry_csv),
        "fastlio_bag": str(args.fastlio_bag),
        "original_bag": str(args.original_bag),
        "crs": f"EPSG:{epsg}",
        "odom_count": len(odom),
        "gps_count": len(gps),
        "associations": len(pairs),
        "similarity": {
            "scale": scale,
            "rotation_2d": rot.tolist(),
            "translation_utm_m": trans.tolist(),
        },
        "pose_outputs": pose_outputs,
        "raster_points_used": int(len(points)),
        "raster_outputs": raster_outputs,
        "overview": overview,
        "notes": [
            "FAST-LIO local XY aligned to GNSS by 2D similarity.",
            "This is a QGIS inspection product, not a GPS-factor graph optimization.",
            "Use pose CSV/GeoJSON as pose(t) baseline for the next orthomosaic integration step.",
        ],
    }
    (args.out / "fastlio_gps_qgis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
