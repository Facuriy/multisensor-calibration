#!/usr/bin/env python3
"""Local Ouster ICP-SLAM prototype for one plot window.

The goal is to put all selected Ouster clouds into one local coordinate system,
estimate one global ground plane, and render top-down intensity/height maps.
This fixes the "per-frame relative height" problem from image-space fusion.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_bounds
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
from shapely.geometry import LineString, Point
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.extract_ouster_panel_from_manual_intensity_roi import cloud_to_organized, stamp_ns  # noqa: E402
from src.extraction.decode_ouster_imu_packets import decode_packet  # noqa: E402

CLOUD_TOPIC = "/ssf/os1_cloud_node/points"
GPS_TOPIC = "/ssf/gnss/fix"
VEL_TOPIC = "/ssf/gnss/vel"
IMU_TOPIC = "/ssf/os1_node/imu_packets"
PACKET_TYPE = "ouster_ros/msg/PacketMsg"


@dataclass
class CloudFrame:
    stamp_ns: int
    xyz: np.ndarray
    intensity: np.ndarray
    speed_mps: float | None = None
    lat: float | None = None
    lon: float | None = None


@dataclass
class ImuSample:
    stamp_ns: int
    accel_g: np.ndarray
    gyro_radps: np.ndarray


def typestore_with_ouster_packet():
    typestore = get_typestore(Stores.ROS1_NOETIC)
    try:
        typestore.register(get_types_from_msg("uint8[] buf\n", PACKET_TYPE))
    except Exception:
        # Some environments may already know the custom type.
        pass
    return typestore


def velocity_norm(msg: object) -> float | None:
    tw = getattr(msg, "twist", None)
    lin = getattr(tw, "linear", None) if tw is not None else None
    if lin is None:
        return None
    x = float(getattr(lin, "x", 0.0))
    y = float(getattr(lin, "y", 0.0))
    z = float(getattr(lin, "z", 0.0))
    return float(math.sqrt(x * x + y * y + z * z))


def nearest(items: list[tuple[int, object]], stamp_ns: int, max_dt_ms: float) -> object | None:
    if not items:
        return None
    best = min(items, key=lambda x: abs(x[0] - stamp_ns))
    if abs(best[0] - stamp_ns) > max_dt_ms * 1_000_000:
        return None
    return best[1]


def voxel_downsample_with_intensity(xyz: np.ndarray, intensity: np.ndarray, voxel: float, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if len(xyz) == 0:
        return xyz, intensity
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    xyz = xyz[idx]
    intensity = intensity[idx]
    if len(xyz) > max_points:
        rng = np.random.default_rng(42)
        take = rng.choice(len(xyz), max_points, replace=False)
        xyz = xyz[take]
        intensity = intensity[take]
    return xyz.astype(np.float64), intensity.astype(np.float32)


def msg_to_points(msg, voxel: float, max_points: int, max_range_m: float) -> tuple[np.ndarray, np.ndarray]:
    cloud = cloud_to_organized(msg)
    xyz = np.dstack([cloud["x"], cloud["y"], cloud["z"]]).reshape(-1, 3).astype(np.float64)
    intensity = np.asarray(cloud.get("intensity", np.zeros(len(xyz)))).reshape(-1).astype(np.float32)
    rng = np.linalg.norm(xyz, axis=1)
    ok = np.isfinite(xyz).all(axis=1) & (rng > 0.5) & (rng < max_range_m)
    xyz = xyz[ok]
    intensity = intensity[ok]
    # Keep mostly the ground/plant swath around the robot, not far background.
    swath = (np.abs(xyz[:, 1]) < 2.2) & (xyz[:, 0] > -1.5) & (xyz[:, 0] < max_range_m)
    xyz = xyz[swath]
    intensity = intensity[swath]
    return voxel_downsample_with_intensity(xyz, intensity, voxel, max_points)


def collect_cloud_frames(
    bag: Path,
    center_ns: int,
    window_ms: float,
    max_sync_ms: float,
    every: int,
    voxel: float,
    max_points: int,
    max_range_m: float,
) -> tuple[list[CloudFrame], list[ImuSample]]:
    typestore = typestore_with_ouster_packet()
    half = int(window_ms * 1_000_000)
    start = int(center_ns - half)
    stop = int(center_ns + half)
    cloud_msgs: list[tuple[int, object]] = []
    gps_msgs: list[tuple[int, object]] = []
    vel_msgs: list[tuple[int, object]] = []
    imu_samples: list[ImuSample] = []
    with Reader(bag) as reader:
        conns = [c for c in reader.connections if c.topic in {CLOUD_TOPIC, GPS_TOPIC, VEL_TOPIC, IMU_TOPIC}]
        for conn, ts, raw in reader.messages(connections=conns, start=start, stop=stop):
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            if conn.topic == CLOUD_TOPIC:
                cloud_msgs.append((stamp_ns(msg, ts), msg))
            elif conn.topic == GPS_TOPIC:
                gps_msgs.append((int(ts), msg))
            elif conn.topic == VEL_TOPIC:
                vel_msgs.append((int(ts), msg))
            elif conn.topic == IMU_TOPIC:
                decoded = decode_packet(bytes(msg.buf))
                imu_samples.append(ImuSample(int(ts), decoded["accel_g"], decoded["gyro_radps"]))

    selected = cloud_msgs[::max(1, every)]
    frames: list[CloudFrame] = []
    for t, msg in selected:
        xyz, intensity = msg_to_points(msg, voxel=voxel, max_points=max_points, max_range_m=max_range_m)
        if len(xyz) < 200:
            continue
        gps = nearest(gps_msgs, t, max_sync_ms)
        vel = nearest(vel_msgs, t, max_sync_ms)
        frames.append(
            CloudFrame(
                stamp_ns=t,
                xyz=xyz,
                intensity=intensity,
                speed_mps=velocity_norm(vel) if vel is not None else None,
                lat=float(getattr(gps, "latitude", math.nan)) if gps is not None else None,
                lon=float(getattr(gps, "longitude", math.nan)) if gps is not None else None,
            )
        )
    imu_samples.sort(key=lambda s: s.stamp_ns)
    return frames, imu_samples


def rigid_fit(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    cs = src.mean(axis=0)
    cd = dst.mean(axis=0)
    xs = src - cs
    xd = dst - cd
    h = xs.T @ xd
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    t = cd - r @ cs
    out = np.eye(4)
    out[:3, :3] = r
    out[:3, 3] = t
    return out


def transform_points(xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (xyz @ transform[:3, :3].T) + transform[:3, 3]


def icp_point_to_point(
    src: np.ndarray,
    dst: np.ndarray,
    max_corr: float,
    iterations: int,
    init: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    tree = cKDTree(dst)
    transform = np.eye(4) if init is None else init.copy()
    prev_rmse = None
    info = {"pairs": 0, "rmse": None, "iterations": 0}
    for it in range(iterations):
        src_t = transform_points(src, transform)
        dist, idx = tree.query(src_t, k=1, workers=-1)
        ok = dist < max_corr
        if int(ok.sum()) < 80:
            break
        delta = rigid_fit(src_t[ok], dst[idx[ok]])
        transform = delta @ transform
        rmse = float(np.sqrt(np.mean(dist[ok] ** 2)))
        info = {"pairs": int(ok.sum()), "rmse": rmse, "iterations": it + 1}
        if prev_rmse is not None and abs(prev_rmse - rmse) < 1e-5:
            break
        prev_rmse = rmse
    return transform, info


def rotation_angle_deg(rot: np.ndarray) -> float:
    cosang = float(np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.rad2deg(math.acos(cosang)))


def clamp_transform(transform: np.ndarray, max_translation_m: float, max_rotation_deg: float) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    t = transform[:3, 3].astype(np.float64)
    t_norm = float(np.linalg.norm(t))
    if t_norm > max_translation_m > 0:
        t = t * (max_translation_m / t_norm)

    rvec, _ = cv2.Rodrigues(transform[:3, :3].astype(np.float64))
    angle = float(np.linalg.norm(rvec))
    max_angle = math.radians(max_rotation_deg)
    if angle > max_angle > 0:
        rvec = rvec * (max_angle / angle)
    rot, _ = cv2.Rodrigues(rvec)
    out[:3, :3] = rot
    out[:3, 3] = t
    return out


def imu_interval_delta(imu: list[ImuSample], t0_ns: int, t1_ns: int) -> tuple[np.ndarray, dict]:
    """Return a small rotation from integrated gyro between two bag timestamps.

    The Ouster IMU packet timestamps are sensor-local in this dataset, so bag
    time is the sync clock. We integrate raw gyro without removing the full
    interval mean; subtracting it would also remove real yaw during motion.
    """
    if not imu or t1_ns <= t0_ns:
        return np.eye(3, dtype=np.float64), {"imu_samples": 0, "gyro_mean_dps": [0.0, 0.0, 0.0]}
    stamps = np.asarray([s.stamp_ns for s in imu], dtype=np.int64)
    lo = int(np.searchsorted(stamps, t0_ns, side="left"))
    hi = int(np.searchsorted(stamps, t1_ns, side="right"))
    seg = imu[max(0, lo - 1):min(len(imu), hi + 1)]
    if len(seg) < 2:
        return np.eye(3, dtype=np.float64), {"imu_samples": len(seg), "gyro_mean_dps": [0.0, 0.0, 0.0]}
    ts = np.asarray([s.stamp_ns for s in seg], dtype=np.float64) / 1e9
    gyro = np.vstack([s.gyro_radps for s in seg]).astype(np.float64)
    ts = np.clip(ts, t0_ns / 1e9, t1_ns / 1e9)
    order = np.argsort(ts)
    ts = ts[order]
    gyro = gyro[order]
    keep = np.r_[True, np.diff(ts) > 1e-7]
    ts = ts[keep]
    gyro = gyro[keep]
    if len(ts) < 2:
        return np.eye(3, dtype=np.float64), {"imu_samples": len(seg), "gyro_mean_dps": np.rad2deg(gyro.mean(axis=0)).tolist()}
    rotvec = np.trapezoid(gyro, ts, axis=0)
    rot, _ = cv2.Rodrigues(rotvec.reshape(3, 1))
    return rot.astype(np.float64), {
        "imu_samples": int(len(seg)),
        "gyro_mean_dps": np.rad2deg(gyro.mean(axis=0)).tolist(),
        "gyro_integral_deg": np.rad2deg(rotvec).tolist(),
    }


def initial_motion_candidates(distance_m: float) -> list[tuple[str, np.ndarray]]:
    candidates = [("identity", np.eye(4))]
    for name, vec in [
        ("x_minus", [-distance_m, 0.0, 0.0]),
        ("x_plus", [distance_m, 0.0, 0.0]),
        ("y_minus", [0.0, -distance_m, 0.0]),
        ("y_plus", [0.0, distance_m, 0.0]),
    ]:
        t = np.eye(4)
        t[:3, 3] = np.asarray(vec, dtype=np.float64)
        candidates.append((name, t))
    return candidates


def build_trajectory(frames: list[CloudFrame], max_corr: float, iterations: int) -> tuple[list[np.ndarray], list[dict]]:
    poses = [np.eye(4)]
    diagnostics = []
    for i in range(1, len(frames)):
        dt_s = max(0.0, (frames[i].stamp_ns - frames[i - 1].stamp_ns) / 1e9)
        speeds = [v for v in (frames[i - 1].speed_mps, frames[i].speed_mps) if v is not None and np.isfinite(v)]
        distance_m = float(np.mean(speeds) * dt_s) if speeds else 0.0
        distance_m = float(np.clip(distance_m, 0.0, 1.2))
        best = None
        for init_name, init in initial_motion_candidates(distance_m):
            candidate_t, candidate_info = icp_point_to_point(
                frames[i].xyz,
                frames[i - 1].xyz,
                max_corr=max_corr,
                iterations=iterations,
                init=init,
            )
            score = (candidate_info.get("rmse") if candidate_info.get("rmse") is not None else 999.0) - 0.000001 * candidate_info.get("pairs", 0)
            if best is None or score < best[0]:
                best = (score, init_name, candidate_t, candidate_info)
        assert best is not None
        _score, init_name, t_cur_to_prev, info = best
        poses.append(poses[-1] @ t_cur_to_prev)
        info["pair"] = [i - 1, i]
        info["dt_s"] = dt_s
        info["velocity_prior_distance_m"] = distance_m
        info["selected_initial_guess"] = init_name
        info["translation_m"] = t_cur_to_prev[:3, 3].tolist()
        diagnostics.append(info)
    return poses, diagnostics


def build_velocity_trajectory(frames: list[CloudFrame], axis: str) -> tuple[list[np.ndarray], list[dict]]:
    vecs = {
        "x_plus": np.array([1.0, 0.0, 0.0]),
        "x_minus": np.array([-1.0, 0.0, 0.0]),
        "y_plus": np.array([0.0, 1.0, 0.0]),
        "y_minus": np.array([0.0, -1.0, 0.0]),
    }
    direction = vecs[axis]
    poses = [np.eye(4)]
    diagnostics = []
    for i in range(1, len(frames)):
        dt_s = max(0.0, (frames[i].stamp_ns - frames[i - 1].stamp_ns) / 1e9)
        speeds = [v for v in (frames[i - 1].speed_mps, frames[i].speed_mps) if v is not None and np.isfinite(v)]
        distance_m = float(np.mean(speeds) * dt_s) if speeds else 0.0
        step = np.eye(4)
        step[:3, 3] = direction * distance_m
        poses.append(poses[-1] @ step)
        diagnostics.append({
            "pair": [i - 1, i],
            "method": f"velocity_dead_reckoning_{axis}",
            "dt_s": dt_s,
            "velocity_prior_distance_m": distance_m,
            "translation_m": step[:3, 3].tolist(),
        })
    return poses, diagnostics


def build_imu_gps_constrained_trajectory(
    frames: list[CloudFrame],
    imu_samples: list[ImuSample],
    axis: str,
    max_corr: float,
    iterations: int,
    max_icp_correction_m: float,
    max_icp_correction_deg: float,
    use_imu_rotation: bool,
) -> tuple[list[np.ndarray], list[dict]]:
    vecs = {
        "x_plus": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "x_minus": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        "y_plus": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "y_minus": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    }
    direction = vecs[axis]
    poses = [np.eye(4)]
    diagnostics = []
    for i in range(1, len(frames)):
        dt_s = max(0.0, (frames[i].stamp_ns - frames[i - 1].stamp_ns) / 1e9)
        speeds = [v for v in (frames[i - 1].speed_mps, frames[i].speed_mps) if v is not None and np.isfinite(v)]
        distance_m = float(np.mean(speeds) * dt_s) if speeds else 0.0
        distance_m = float(np.clip(distance_m, 0.0, 1.2))
        imu_rot_prev_to_cur, imu_info = imu_interval_delta(imu_samples, frames[i - 1].stamp_ns, frames[i].stamp_ns)

        init = np.eye(4, dtype=np.float64)
        if use_imu_rotation:
            init[:3, :3] = imu_rot_prev_to_cur.T
        init[:3, 3] = direction * distance_m

        raw_t, info = icp_point_to_point(
            frames[i].xyz,
            frames[i - 1].xyz,
            max_corr=max_corr,
            iterations=iterations,
            init=init,
        )
        correction = raw_t @ np.linalg.inv(init)
        correction_clamped = clamp_transform(correction, max_icp_correction_m, max_icp_correction_deg)
        t_cur_to_prev = correction_clamped @ init
        poses.append(poses[-1] @ t_cur_to_prev)
        diagnostics.append({
            "pair": [i - 1, i],
            "method": "imu_gps_constrained_icp",
            "axis": axis,
            "dt_s": dt_s,
            "velocity_prior_distance_m": distance_m,
            "imu": imu_info,
            "icp_pairs": info.get("pairs"),
            "icp_rmse": info.get("rmse"),
            "icp_iterations": info.get("iterations"),
            "raw_icp_translation_m": raw_t[:3, 3].tolist(),
            "prior_translation_m": init[:3, 3].tolist(),
            "applied_translation_m": t_cur_to_prev[:3, 3].tolist(),
            "raw_correction_translation_m": correction[:3, 3].tolist(),
            "applied_correction_translation_m": correction_clamped[:3, 3].tolist(),
            "raw_correction_rotation_deg": rotation_angle_deg(correction[:3, :3]),
            "applied_correction_rotation_deg": rotation_angle_deg(correction_clamped[:3, :3]),
        })
    return poses, diagnostics


def trajectory_score(poses: list[np.ndarray], diagnostics: list[dict]) -> dict:
    centers = np.asarray([p[:3, 3] for p in poses], dtype=np.float64)
    if len(centers) < 2:
        return {"score": -1e9, "path_length_m": 0.0, "footprint_area_m2": 0.0}
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    path_len = float(steps.sum())
    span = np.ptp(centers[:, :2], axis=0)
    area = float(max(span[0], 1e-6) * max(span[1], 1e-6))
    rmse_vals = [d.get("icp_rmse") for d in diagnostics if d.get("icp_rmse") is not None]
    rmse = float(np.mean(rmse_vals)) if rmse_vals else 0.3
    corr_vals = [np.linalg.norm(d.get("applied_correction_translation_m", [0, 0, 0])) for d in diagnostics]
    corr = float(np.mean(corr_vals)) if corr_vals else 0.0
    score = area + 0.08 * path_len - 0.8 * rmse - 0.5 * corr
    return {
        "score": float(score),
        "path_length_m": path_len,
        "footprint_area_m2": area,
        "mean_icp_rmse_m": rmse,
        "mean_applied_correction_m": corr,
    }


def fit_global_ground_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    pts = points[np.isfinite(points).all(axis=1)]
    if len(pts) > 60000:
        rng = np.random.default_rng(123)
        pts = pts[rng.choice(len(pts), 60000, replace=False)]
    # Candidate ground: lower part of the vertical coordinate in the local map.
    low = pts[:, 2] < np.percentile(pts[:, 2], 45)
    ground = pts[low] if int(low.sum()) > 100 else pts
    c = ground.mean(axis=0)
    _, _, vh = np.linalg.svd(ground - c, full_matrices=False)
    n = vh[-1]
    n = n / np.linalg.norm(n)
    if n[2] < 0:
        n = -n
    d = -float(n @ c)
    return n.astype(np.float64), d


def plane_basis(normal: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = points - points.mean(axis=0)
    projected = centered - (centered @ normal)[:, None] * normal[None, :]
    _, _, vh = np.linalg.svd(projected, full_matrices=False)
    e1 = vh[0]
    e1 = e1 - normal * float(e1 @ normal)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    e2 = e2 / np.linalg.norm(e2)
    return e1, e2


def render_topdown(points: np.ndarray, intensity: np.ndarray, poses: list[np.ndarray], out_dir: Path, resolution_m: float) -> dict:
    normal, d = fit_global_ground_plane(points)
    e1, e2 = plane_basis(normal, points)
    origin = points.mean(axis=0)
    uv = np.column_stack([(points - origin) @ e1, (points - origin) @ e2])
    height = points @ normal + d
    height = height - np.percentile(height, 3)
    height = np.clip(height, 0, None)

    min_uv = uv.min(axis=0)
    max_uv = uv.max(axis=0)
    pad = 0.25
    w = int(math.ceil((max_uv[0] - min_uv[0] + 2 * pad) / resolution_m))
    h = int(math.ceil((max_uv[1] - min_uv[1] + 2 * pad) / resolution_m))
    x = np.floor((uv[:, 0] - min_uv[0] + pad) / resolution_m).astype(int)
    y = np.floor((uv[:, 1] - min_uv[1] + pad) / resolution_m).astype(int)
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    x, y = x[ok], y[ok]
    inten = intensity[ok]
    hei = height[ok]

    intensity_img = np.zeros((h, w), np.float32)
    height_img = np.zeros((h, w), np.float32)
    count = np.zeros((h, w), np.uint16)
    np.maximum.at(intensity_img, (y, x), inten)
    np.maximum.at(height_img, (y, x), hei)
    np.add.at(count, (y, x), 1)
    support = count > 0

    if support.any():
        fill_high = float(np.percentile(height_img[support], 90))
        low_envelope = np.where(support, height_img, fill_high).astype(np.float32)
        # A ~0.6 m lower-envelope filter removes slow ground/sensor tilt while
        # keeping row-scale plant relief. This is global-map detrending, not
        # the old per-frame normalization.
        k = max(7, int(round(0.60 / resolution_m)) | 1)
        low_envelope = cv2.erode(low_envelope, np.ones((k, k), np.uint8))
        low_envelope = cv2.GaussianBlur(low_envelope, (0, 0), max(1.5, k / 8.0))
        canopy_img = np.clip(height_img - low_envelope, 0.0, None)
    else:
        canopy_img = np.zeros_like(height_img)

    kernel = np.ones((5, 5), np.uint8)
    intensity_img = cv2.dilate(intensity_img, kernel)
    height_img = cv2.dilate(height_img, kernel)
    canopy_img = cv2.dilate(canopy_img, kernel)
    support = cv2.dilate(support.astype(np.uint8), kernel).astype(bool)

    def colorize(values, cmap, lohi=(2, 98), gamma=0.8):
        out = np.zeros(values.shape, np.float32)
        if support.any():
            lo, hi = np.percentile(values[support], lohi)
            if hi <= lo:
                hi = lo + 1
            out[support] = np.clip((values[support] - lo) / (hi - lo), 0, 1) ** gamma
        color = cv2.applyColorMap(np.clip(out * 255, 0, 255).astype(np.uint8), cmap)
        color[~support] = 245
        return color

    intensity_color = colorize(intensity_img, cv2.COLORMAP_INFERNO, (5, 98), 0.75)
    height_color = colorize(height_img, cv2.COLORMAP_VIRIDIS, (2, 99), 0.65)
    canopy_color = colorize(canopy_img, cv2.COLORMAP_TURBO, (20, 99.5), 0.55)

    traj = np.full((h, w, 3), 245, np.uint8)
    centers = np.asarray([p[:3, 3] for p in poses])
    cuv = np.column_stack([(centers - origin) @ e1, (centers - origin) @ e2])
    cx = np.floor((cuv[:, 0] - min_uv[0] + pad) / resolution_m).astype(int)
    cy = np.floor((cuv[:, 1] - min_uv[1] + pad) / resolution_m).astype(int)
    for i in range(1, len(cx)):
        cv2.line(traj, (int(cx[i - 1]), int(cy[i - 1])), (int(cx[i]), int(cy[i])), (30, 30, 220), 2, cv2.LINE_AA)
    for i, (xx, yy) in enumerate(zip(cx, cy)):
        if 0 <= xx < w and 0 <= yy < h:
            cv2.circle(traj, (int(xx), int(yy)), 4, (20, 140, 20), -1)
            cv2.putText(traj, str(i), (int(xx) + 4, int(yy) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

    out_int = out_dir / "slam_topdown_intensity.jpg"
    out_h = out_dir / "slam_topdown_global_height.jpg"
    out_canopy = out_dir / "slam_topdown_canopy_height.jpg"
    out_traj = out_dir / "slam_trajectory_topdown.jpg"
    cv2.imwrite(str(out_int), intensity_color, [cv2.IMWRITE_JPEG_QUALITY, 96])
    cv2.imwrite(str(out_h), height_color, [cv2.IMWRITE_JPEG_QUALITY, 96])
    cv2.imwrite(str(out_canopy), canopy_color, [cv2.IMWRITE_JPEG_QUALITY, 96])
    cv2.imwrite(str(out_traj), traj, [cv2.IMWRITE_JPEG_QUALITY, 96])
    return {
        "intensity": str(out_int),
        "height": str(out_h),
        "canopy_height": str(out_canopy),
        "trajectory": str(out_traj),
        "plane_normal": normal.tolist(),
        "plane_d": d,
        "resolution_m": resolution_m,
        "size_px": [w, h],
    }


def utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    zone = int((lon + 180.0) // 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


def pose_yaw_deg(pose: np.ndarray) -> float:
    return float(math.degrees(math.atan2(pose[1, 0], pose[0, 0])))


def save_local_poses(frames: list[CloudFrame], poses: list[np.ndarray], out_dir: Path) -> dict:
    csv_path = out_dir / "slam_poses_local.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["index", "stamp_ns", "x_m", "y_m", "z_m", "yaw_deg", "speed_mps", "lat", "lon"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, (frame, pose) in enumerate(zip(frames, poses)):
            writer.writerow({
                "index": i,
                "stamp_ns": int(frame.stamp_ns),
                "x_m": float(pose[0, 3]),
                "y_m": float(pose[1, 3]),
                "z_m": float(pose[2, 3]),
                "yaw_deg": pose_yaw_deg(pose),
                "speed_mps": frame.speed_mps,
                "lat": frame.lat,
                "lon": frame.lon,
            })
    return {"poses_local_csv": str(csv_path)}


def similarity_2d(src_xy: np.ndarray, dst_xy: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return scale, 2x2 rotation, translation mapping src -> dst."""
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


def georeference_poses(frames: list[CloudFrame], poses: list[np.ndarray], out_dir: Path) -> dict:
    gps_idx = [i for i, f in enumerate(frames) if f.lat is not None and f.lon is not None and np.isfinite([f.lat, f.lon]).all()]
    if len(gps_idx) < 3:
        return {"georef_available": False, "reason": "fewer than 3 GPS-matched cloud poses"}
    lat0 = float(frames[gps_idx[0]].lat)
    lon0 = float(frames[gps_idx[0]].lon)
    epsg = utm_epsg_for_lonlat(lon0, lat0)
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    to_wgs = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    src = np.asarray([[poses[i][0, 3], poses[i][1, 3]] for i in gps_idx], dtype=np.float64)
    dst = np.asarray([to_utm.transform(float(frames[i].lon), float(frames[i].lat)) for i in gps_idx], dtype=np.float64)
    scale, rot, trans = similarity_2d(src, dst)
    aligned = np.asarray([scale * (np.array([p[0, 3], p[1, 3]]) @ rot) + trans for p in poses], dtype=np.float64)
    residuals = np.linalg.norm(aligned[gps_idx] - dst, axis=1)

    csv_path = out_dir / "slam_poses_georef.csv"
    rows = []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "index", "stamp_ns", "easting_m", "northing_m", "z_local_m",
            "lon", "lat", "gps_lon", "gps_lat", "gps_residual_m", "speed_mps",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, (frame, xy) in enumerate(zip(frames, aligned)):
            lon, lat = to_wgs.transform(float(xy[0]), float(xy[1]))
            gps_res = ""
            if i in gps_idx:
                gps_res = float(residuals[gps_idx.index(i)])
            row = {
                "index": i,
                "stamp_ns": int(frame.stamp_ns),
                "easting_m": float(xy[0]),
                "northing_m": float(xy[1]),
                "z_local_m": float(poses[i][2, 3]),
                "lon": float(lon),
                "lat": float(lat),
                "gps_lon": frame.lon,
                "gps_lat": frame.lat,
                "gps_residual_m": gps_res,
                "speed_mps": frame.speed_mps,
            }
            writer.writerow(row)
            rows.append(row)

    line = LineString([(r["lon"], r["lat"]) for r in rows])
    points = [Point(r["lon"], r["lat"]) for r in rows]
    gdf_line = gpd.GeoDataFrame([{"kind": "slam_trajectory", "plot": "", "geometry": line}], crs="EPSG:4326")
    gdf_points = gpd.GeoDataFrame([{**r, "geometry": pt} for r, pt in zip(rows, points)], crs="EPSG:4326")
    line_path = out_dir / "slam_trajectory_wgs84.geojson"
    points_path = out_dir / "slam_poses_wgs84.geojson"
    gdf_line.to_file(line_path, driver="GeoJSON")
    gdf_points.to_file(points_path, driver="GeoJSON")
    return {
        "georef_available": True,
        "crs": f"EPSG:{epsg}",
        "scale_local_to_utm": scale,
        "rotation_2d": rot.tolist(),
        "translation_utm_m": trans.tolist(),
        "gps_residual_m_mean": float(residuals.mean()),
        "gps_residual_m_median": float(np.median(residuals)),
        "gps_residual_m_max": float(residuals.max()),
        "poses_georef_csv": str(csv_path),
        "trajectory_wgs84_geojson": str(line_path),
        "poses_wgs84_geojson": str(points_path),
    }


def write_topdown_geotiffs(outputs: dict, out_dir: Path, georef: dict) -> dict:
    if not georef.get("georef_available"):
        return {}
    # The rendered SLAM rasters are in local top-down map coordinates. We can
    # put them in a metric CRS using the georeferenced trajectory bounds as an
    # approximate local-map placement. This is for GIS inspection, not final
    # orthophoto production.
    csv_path = Path(georef["poses_georef_csv"])
    coords = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            coords.append([float(row["easting_m"]), float(row["northing_m"])])
    coords_np = np.asarray(coords, dtype=np.float64)
    minx, miny = coords_np.min(axis=0) - 1.5
    maxx, maxy = coords_np.max(axis=0) + 1.5
    qgis = out_dir / "qgis"
    qgis.mkdir(parents=True, exist_ok=True)
    out = {}
    for key in ["intensity", "height", "canopy_height", "trajectory"]:
        img = cv2.imread(outputs[key], cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        transform = from_bounds(float(minx), float(miny), float(maxx), float(maxy), w, h)
        path = qgis / f"slam_{key}.tif"
        data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=3,
            dtype=data.dtype,
            crs=georef["crs"],
            transform=transform,
            compress="deflate",
            photometric="RGB",
        ) as ds:
            ds.write(data)
            ds.update_tags(layer=f"slam_{key}", georef_quality="trajectory_gnss_similarity_approx")
        out[f"{key}_geotiff"] = str(path)
    return out


def write_overview(paths: dict, out: Path) -> None:
    items = [
        ("intensity", "SLAM Ouster intensity"),
        ("height", "Global relative height"),
        ("canopy_height", "Canopy height detrended"),
        ("trajectory", "Estimated trajectory"),
    ]
    panels = []
    for key, label in items:
        img = cv2.imread(paths[key], cv2.IMREAD_COLOR)
        if img is None:
            continue
        max_w = 620
        if img.shape[1] > max_w:
            scale = max_w / img.shape[1]
            img = cv2.resize(img, (max_w, max(1, int(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (0, 0), (img.shape[1], 46), (0, 0, 0), -1)
        cv2.putText(img, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(img)
    h = max(p.shape[0] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245)) for p in panels]
    cv2.imwrite(str(out), cv2.hconcat(panels), [cv2.IMWRITE_JPEG_QUALITY, 96])


def write_candidate_comparison(candidate_overviews: list[tuple[str, Path]], out: Path) -> None:
    imgs = []
    for name, path in candidate_overviews:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        max_w = 920
        if img.shape[1] > max_w:
            scale = max_w / img.shape[1]
            img = cv2.resize(img, (max_w, max(1, int(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (0, 0), (img.shape[1], 42), (20, 20, 20), -1)
        cv2.putText(img, name, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
        imgs.append(img)
    if not imgs:
        return
    w = max(i.shape[1] for i in imgs)
    padded = [cv2.copyMakeBorder(i, 0, 0, 0, w - i.shape[1], cv2.BORDER_CONSTANT, value=(245, 245, 245)) for i in imgs]
    cv2.imwrite(str(out), np.vstack(padded), [cv2.IMWRITE_JPEG_QUALITY, 96])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bag", type=Path, required=True)
    ap.add_argument("--center-ns", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window-ms", type=float, default=8000)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--max-sync-ms", type=float, default=1500)
    ap.add_argument("--voxel", type=float, default=0.045)
    ap.add_argument("--max-points", type=int, default=9000)
    ap.add_argument("--max-range-m", type=float, default=8.0)
    ap.add_argument("--max-corr-m", type=float, default=0.22)
    ap.add_argument("--icp-iters", type=int, default=25)
    ap.add_argument("--resolution-m", type=float, default=0.015)
    ap.add_argument(
        "--pose-mode",
        choices=[
            "icp",
            "velocity_x_plus",
            "velocity_x_minus",
            "velocity_y_plus",
            "velocity_y_minus",
            "imu_gps_constrained",
            "imu_gps_auto",
        ],
        default="imu_gps_auto",
    )
    ap.add_argument("--motion-axis", choices=["x_plus", "x_minus", "y_plus", "y_minus"], default="y_plus")
    ap.add_argument("--max-icp-correction-m", type=float, default=0.07)
    ap.add_argument("--max-icp-correction-deg", type=float, default=2.0)
    ap.add_argument("--no-imu-rotation", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    frames, imu_samples = collect_cloud_frames(
        args.bag,
        args.center_ns,
        args.window_ms,
        args.max_sync_ms,
        args.every,
        args.voxel,
        args.max_points,
        args.max_range_m,
    )
    if len(frames) < 2:
        raise RuntimeError(f"Need at least 2 cloud frames, got {len(frames)}")
    candidate_summaries = {}
    candidate_overviews = []
    selected_name = args.pose_mode
    if args.pose_mode == "icp":
        poses, diagnostics = build_trajectory(frames, args.max_corr_m, args.icp_iters)
    elif args.pose_mode.startswith("velocity_"):
        axis = args.pose_mode.replace("velocity_", "")
        poses, diagnostics = build_velocity_trajectory(frames, axis)
    elif args.pose_mode == "imu_gps_constrained":
        selected_name = f"imu_gps_{args.motion_axis}"
        poses, diagnostics = build_imu_gps_constrained_trajectory(
            frames,
            imu_samples,
            args.motion_axis,
            args.max_corr_m,
            args.icp_iters,
            args.max_icp_correction_m,
            args.max_icp_correction_deg,
            not args.no_imu_rotation,
        )
    else:
        best = None
        for axis in ["x_plus", "x_minus", "y_plus", "y_minus"]:
            cand_dir = args.out / f"candidate_{axis}"
            cand_dir.mkdir(parents=True, exist_ok=True)
            cand_poses, cand_diag = build_imu_gps_constrained_trajectory(
                frames,
                imu_samples,
                axis,
                args.max_corr_m,
                args.icp_iters,
                args.max_icp_correction_m,
                args.max_icp_correction_deg,
                not args.no_imu_rotation,
            )
            cand_points = []
            cand_int = []
            for frame, pose in zip(frames, cand_poses):
                cand_points.append(transform_points(frame.xyz, pose))
                cand_int.append(frame.intensity)
            cand_points_np = np.vstack(cand_points)
            cand_int_np = np.concatenate(cand_int)
            if len(cand_points_np) > 350000:
                rng = np.random.default_rng(7)
                take = rng.choice(len(cand_points_np), 350000, replace=False)
                cand_points_np = cand_points_np[take]
                cand_int_np = cand_int_np[take]
            cand_outputs = render_topdown(cand_points_np, cand_int_np, cand_poses, cand_dir, args.resolution_m)
            cand_overview = cand_dir / "slam_lidar_overview.jpg"
            write_overview(cand_outputs, cand_overview)
            cand_score = trajectory_score(cand_poses, cand_diag)
            candidate_summaries[axis] = cand_score | {
                "outputs": cand_outputs | {"overview": str(cand_overview)},
                "diagnostics": cand_diag,
            }
            candidate_overviews.append((f"imu_gps_constrained {axis} score={cand_score['score']:.3f}", cand_overview))
            if best is None or cand_score["score"] > best[0]["score"]:
                best = (cand_score, axis, cand_poses, cand_diag)
        assert best is not None
        _score, selected_axis, poses, diagnostics = best
        selected_name = f"imu_gps_auto_selected_{selected_axis}"
        write_candidate_comparison(candidate_overviews, args.out / "candidate_comparison_overview.jpg")

    map_points = []
    map_intensity = []
    for frame, pose in zip(frames, poses):
        map_points.append(transform_points(frame.xyz, pose))
        map_intensity.append(frame.intensity)
    points = np.vstack(map_points)
    intensity = np.concatenate(map_intensity)
    if len(points) > 500000:
        rng = np.random.default_rng(7)
        take = rng.choice(len(points), 500000, replace=False)
        points = points[take]
        intensity = intensity[take]
    outputs = render_topdown(points, intensity, poses, args.out, args.resolution_m)
    overview = args.out / "slam_lidar_overview.jpg"
    write_overview(outputs, overview)
    pose_outputs = save_local_poses(frames, poses, args.out)
    georef = georeference_poses(frames, poses, args.out)
    geotiff_outputs = write_topdown_geotiffs(outputs, args.out, georef)
    meta = {
        "bag": str(args.bag),
        "center_ns": args.center_ns,
        "window_ms": args.window_ms,
        "cloud_frames": len(frames),
        "imu_samples": len(imu_samples),
        "every": args.every,
        "pose_mode": args.pose_mode,
        "selected_pose": selected_name,
        "mean_speed_mps": float(np.nanmean([f.speed_mps for f in frames if f.speed_mps is not None])) if any(f.speed_mps is not None for f in frames) else None,
        "icp": diagnostics,
        "candidate_summaries": candidate_summaries,
        "georef": georef,
        "outputs": outputs | {"overview": str(overview)} | pose_outputs | geotiff_outputs,
        "notes": "LiDAR local map using Ouster PointCloud2, GNSS velocity prior, decoded Ouster IMU gyro prior, and bounded ICP refinement. Internal Ouster IMU timestamps are not used for sync; bag timestamps are used.",
    }
    (args.out / "slam_lidar_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
