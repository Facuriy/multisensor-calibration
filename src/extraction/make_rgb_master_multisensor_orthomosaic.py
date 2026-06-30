#!/usr/bin/env python3
"""RGB-master multisensor plot orthomosaic prototype.

This script reads a ROS1 bag directly, selects synchronized RGB/VIS/NIR/Thermal
frames inside one plot, registers every camera into RGB, crops the common valid
camera intersection, projects Ouster into RGB, then builds local visual mosaics
using one RGB-estimated image trajectory shared by all layers.

It is a rigorous prototype, not a global photogrammetry replacement:

* dense camera layers are cropped to a no-hole common intersection;
* Ouster depth/height/intensity remain masked sparse measurements;
* temporal alignment is visual homography-based over a short plot segment;
* georeferencing is metadata-only for now, not a metric GeoTIFF transform.
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
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from shapely.geometry import Point

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.extract_ouster_panel_from_manual_intensity_roi import cloud_to_organized, stamp_ns  # noqa: E402
from src.extraction.extract_all_bag_images import decode_ros_image, robust_preview  # noqa: E402
from src.extraction.make_plot_presentation_gallery import soil_cover, to_bgr_for_display  # noqa: E402
from src.extraction.make_rgb_plot_orthomosaic import estimate_homography  # noqa: E402
from src.registration.coregister_rgb_master import largest_true_rectangle, safe_erode, warp_image_and_mask  # noqa: E402


TOPICS = {
    "rgb": "/ssf/BFS_usb_0/image_raw",
    "vis": "/ssf/photonfocus_camera_vis_node/image_raw",
    "nir": "/ssf/photonfocus_camera_nir_node/image_raw",
    "thermal": "/ssf/thermalgrabber_ros/image_deg_celsius",
    "cloud": "/ssf/os1_cloud_node/points",
    "gps": "/ssf/gnss/fix",
    "vel": "/ssf/gnss/vel",
}
DEFAULT_CALIBRATION = Path("data/calibration/new_session/20260623/calibration_20260623_final_candidate.json")
GEOTIFF_KEYS = {
    "rgb",
    "vis",
    "nir",
    "soil_cover",
    "thermal_celsius_color",
    "depth_mm_color",
    "height_m_color",
    "intensity_color",
    "depth_mm",
    "height_m_float",
    "thermal_celsius_float",
    "camera_valid_mask",
    "ouster_valid_mask",
}


@dataclass
class Ref:
    stamp_ns: int
    msg: object


@dataclass
class Frame:
    stamp_ns: int
    bag_plot_id: str
    lat: float | None
    lon: float | None
    speed_mps: float | None
    rgb: np.ndarray
    vis: np.ndarray
    nir: np.ndarray
    thermal_c: np.ndarray
    thermal_mask: np.ndarray
    soil: np.ndarray
    depth_mm: np.ndarray
    depth_mask: np.ndarray
    height_m: np.ndarray
    height_mask: np.ndarray
    intensity: np.ndarray
    intensity_mask: np.ndarray
    common_roi_rgb_xyxy: tuple[int, int, int, int]


def load_final_calibration(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    regs = doc["target_plane_registration_to_rgb"]
    rgb_intr = doc["intrinsics"]["rgb"]
    return {
        "H": {
            "vis": np.asarray(regs["vis_to_rgb"]["H_sensor_to_rgb"], dtype=np.float64),
            "nir": np.asarray(regs["nir_to_rgb"]["H_sensor_to_rgb"], dtype=np.float64),
            "thermal": np.asarray(regs["thermal_to_rgb"]["H_sensor_to_rgb"], dtype=np.float64),
        },
        "K_rgb": np.asarray(rgb_intr["K"], dtype=np.float64),
        "dist_rgb": np.asarray(rgb_intr.get("dist_coeffs", [0, 0, 0, 0, 0]), dtype=np.float64).reshape(-1, 1),
        "T_cam_lidar": np.asarray(doc["ouster_rgb"]["T_cam_lidar"]["matrix"], dtype=np.float64),
    }


def load_plots(gpkg: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(gpkg)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf.copy()
    gdf["plot_id"] = [str(i + 1) for i in range(len(gdf))]
    return gdf


def plot_for_gps(gdf: gpd.GeoDataFrame, lon: float | None, lat: float | None) -> str | None:
    if lon is None or lat is None or not np.isfinite([lon, lat]).all():
        return None
    hits = gdf[gdf.geometry.contains(Point(float(lon), float(lat)))]
    if hits.empty:
        return None
    return str(hits.iloc[0]["plot_id"])


def nearest(items: list[Ref], stamp: int, max_dt_ms: float) -> Ref | None:
    if not items:
        return None
    best = min(items, key=lambda r: abs(r.stamp_ns - stamp))
    if abs(best.stamp_ns - stamp) > max_dt_ms * 1_000_000:
        return None
    return best


def velocity_norm(msg: object) -> float | None:
    tw = getattr(msg, "twist", None)
    lin = getattr(tw, "linear", None) if tw is not None else None
    if lin is None:
        return None
    x = float(getattr(lin, "x", 0.0))
    y = float(getattr(lin, "y", 0.0))
    z = float(getattr(lin, "z", 0.0))
    return float(math.sqrt(x * x + y * y + z * z))


def collect_refs(
    bag: Path,
    center_ns: int | None,
    window_ms: float | None,
    include_cloud: bool,
) -> dict[str, list[Ref]]:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    start = stop = None
    if center_ns is not None and window_ms is not None and window_ms > 0:
        half = int(window_ms * 1_000_000)
        start = int(center_ns - half)
        stop = int(center_ns + half)
    wanted = set(TOPICS.values())
    if not include_cloud:
        wanted.remove(TOPICS["cloud"])
    refs = {k: [] for k in TOPICS}
    with Reader(bag) as reader:
        conns = [c for c in reader.connections if c.topic in wanted]
        kwargs = {}
        if start is not None and stop is not None:
            kwargs = {"start": start, "stop": stop}
        for conn, ts, raw in reader.messages(connections=conns, **kwargs):
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            key = next(k for k, topic in TOPICS.items() if topic == conn.topic)
            msg_stamp = stamp_ns(msg, ts) if key == "cloud" else int(ts)
            refs[key].append(Ref(int(msg_stamp), msg))
    return refs


def scan_plot_counts(bag: Path, gpkg: Path, center_ns: int | None, window_ms: float | None, max_sync_ms: float) -> dict[str, Any]:
    plots = load_plots(gpkg)
    refs = collect_refs(bag, center_ns, window_ms, include_cloud=False)
    counts: dict[str, int] = {}
    examples: dict[str, int] = {}
    for rgb in refs["rgb"]:
        gps = nearest(refs["gps"], rgb.stamp_ns, max_sync_ms)
        if gps is None:
            continue
        lat = float(getattr(gps.msg, "latitude", math.nan))
        lon = float(getattr(gps.msg, "longitude", math.nan))
        pid = plot_for_gps(plots, lon, lat)
        if pid is None:
            continue
        counts[pid] = counts.get(pid, 0) + 1
        examples.setdefault(pid, rgb.stamp_ns)
    return {"bag": str(bag), "plot_frame_counts": dict(sorted(counts.items(), key=lambda kv: int(kv[0]))), "example_center_ns": examples}


def thermal_float(raw: np.ndarray) -> np.ndarray:
    if raw.ndim == 3:
        return cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return raw.astype(np.float32)


def common_camera_crops(
    rgb_raw: np.ndarray,
    vis_raw: np.ndarray,
    nir_raw: np.ndarray,
    thermal_raw: np.ndarray,
    calib: dict[str, Any],
    margin_px: int,
    trim_bottom_px: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], tuple[int, int, int, int]] | None:
    rgb = to_bgr_for_display(rgb_raw, "rgb")
    # These previews reproduce the 512x256 geometry used by the 20260623
    # homographies and validation images.
    vis = robust_preview(vis_raw, "vis")
    nir = robust_preview(nir_raw, "nir")
    thermal = thermal_float(thermal_raw)
    rgb_shape = rgb.shape[:2]
    registered: dict[str, np.ndarray] = {"rgb": rgb}
    masks: dict[str, np.ndarray] = {"rgb": np.ones(rgb_shape, dtype=bool)}
    for sensor, img in (("vis", vis), ("nir", nir), ("thermal", thermal)):
        warped, mask = warp_image_and_mask(img, calib["H"][sensor], rgb_shape, cv2.INTER_LINEAR)
        registered[sensor] = warped
        masks[sensor] = mask
    common = np.ones(rgb_shape, dtype=bool)
    for mask in masks.values():
        common &= mask
    rect = largest_true_rectangle(safe_erode(common, margin_px))
    if rect is None:
        return None
    x0, y0, x1, y1 = rect
    if trim_bottom_px > 0:
        y1 = max(y0 + 32, y1 - int(trim_bottom_px))
    crops = {k: v[y0:y1, x0:x1].copy() for k, v in registered.items()}
    crop_masks = {k: v[y0:y1, x0:x1].copy() for k, v in masks.items()}
    return crops, crop_masks, (x0, y0, x1, y1)


def project_ouster(
    cloud_msg: object | None,
    calib: dict[str, Any],
    full_rgb_shape: tuple[int, int],
    roi: tuple[int, int, int, int],
    max_range_m: float,
    splat_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = roi
    h_crop, w_crop = y1 - y0, x1 - x0
    depth = np.zeros((h_crop, w_crop), dtype=np.float32)
    height = np.zeros((h_crop, w_crop), dtype=np.float32)
    intensity = np.zeros((h_crop, w_crop), dtype=np.float32)
    mask = np.zeros((h_crop, w_crop), dtype=bool)
    if cloud_msg is None:
        return depth, height, intensity, mask, mask.copy()
    cloud = cloud_to_organized(cloud_msg)
    xyz = np.dstack([cloud["x"], cloud["y"], cloud["z"]]).reshape(-1, 3).astype(np.float64)
    inten = np.asarray(cloud.get("intensity", np.zeros(xyz.shape[0])), dtype=np.float32).reshape(-1)
    rng = np.linalg.norm(xyz, axis=1)
    ok = np.isfinite(xyz).all(axis=1) & (rng > 0.35) & (rng < max_range_m)
    xyz = xyz[ok]
    inten = inten[ok]
    if len(xyz) == 0:
        return depth, height, intensity, mask, mask.copy()
    T = calib["T_cam_lidar"]
    cam = (T[:3, :3] @ xyz.T).T + T[:3, 3]
    front = cam[:, 2] > 0.05
    cam, xyz, inten = cam[front], xyz[front], inten[front]
    if len(cam) == 0:
        return depth, height, intensity, mask, mask.copy()
    uv, _ = cv2.projectPoints(cam, np.zeros(3), np.zeros(3), calib["K_rgb"], calib["dist_rgb"])
    uv = uv.reshape(-1, 2)
    fh, fw = full_rgb_shape
    inside = (uv[:, 0] >= x0) & (uv[:, 0] < x1) & (uv[:, 1] >= y0) & (uv[:, 1] < y1) & (uv[:, 0] >= 0) & (uv[:, 0] < fw) & (uv[:, 1] >= 0) & (uv[:, 1] < fh)
    uv, cam, xyz, inten = uv[inside], cam[inside], xyz[inside], inten[inside]
    if len(cam) == 0:
        return depth, height, intensity, mask, mask.copy()
    px = np.round(uv[:, 0] - x0).astype(np.int32)
    py = np.round(uv[:, 1] - y0).astype(np.int32)
    pix = (px >= 0) & (px < w_crop) & (py >= 0) & (py < h_crop)
    px, py, cam, xyz, inten = px[pix], py[pix], cam[pix], xyz[pix], inten[pix]
    if len(cam) == 0:
        return depth, height, intensity, mask, mask.copy()
    z_mm = cam[:, 2].astype(np.float32) * 1000.0
    rel_height = relative_height(xyz)
    lin = py * w_crop + px
    order = np.lexsort((z_mm, lin))
    first = np.r_[True, lin[order][1:] != lin[order][:-1]]
    keep = order[first]
    depth[py[keep], px[keep]] = z_mm[keep]
    height[py[keep], px[keep]] = rel_height[keep]
    intensity[py[keep], px[keep]] = inten[keep]
    mask[py[keep], px[keep]] = True
    sparse_mask = mask.copy()
    if splat_radius > 1:
        kernel = np.ones((splat_radius, splat_radius), np.uint8)
        depth = cv2.dilate(depth, kernel)
        height = cv2.dilate(height, kernel)
        intensity = cv2.dilate(intensity, kernel)
        mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    return depth, height, intensity, mask, sparse_mask


def relative_height(xyz_lidar: np.ndarray) -> np.ndarray:
    if len(xyz_lidar) < 20:
        return np.zeros(len(xyz_lidar), dtype=np.float32)
    # In this rig, negative lidar z is visually the crop/plant height direction
    # used in the previous Ouster diagnostics.
    vals = -xyz_lidar[:, 2].astype(np.float64)
    xy = xyz_lidar[:, :2].astype(np.float64)
    low = vals <= np.percentile(vals, 65)
    if int(low.sum()) < 12:
        low = np.ones(len(vals), dtype=bool)
    A = np.column_stack([xy[low, 0], xy[low, 1], np.ones(int(low.sum()))])
    try:
        coeff, *_ = np.linalg.lstsq(A, vals[low], rcond=None)
        base = coeff[0] * xy[:, 0] + coeff[1] * xy[:, 1] + coeff[2]
        rel = vals - base
    except np.linalg.LinAlgError:
        rel = vals - np.percentile(vals, 5)
    rel -= np.percentile(rel, 5)
    return np.clip(rel, 0, None).astype(np.float32)


def build_frames(
    bag: Path,
    gpkg: Path,
    plot_id: str,
    calib: dict[str, Any],
    center_ns: int | None,
    window_ms: float | None,
    max_sync_ms: float,
    max_range_m: float,
    splat_radius: int,
    margin_px: int,
    trim_bottom_px: int,
    include_lidar: bool,
) -> list[Frame]:
    plots = load_plots(gpkg)
    refs = collect_refs(bag, center_ns, window_ms, include_cloud=include_lidar)
    frames: list[Frame] = []
    for rgb_ref in refs["rgb"]:
        gps = nearest(refs["gps"], rgb_ref.stamp_ns, max_sync_ms)
        if gps is None:
            continue
        lat = float(getattr(gps.msg, "latitude", math.nan))
        lon = float(getattr(gps.msg, "longitude", math.nan))
        pid = plot_for_gps(plots, lon, lat)
        if pid != str(plot_id):
            continue
        vis = nearest(refs["vis"], rgb_ref.stamp_ns, max_sync_ms)
        nir = nearest(refs["nir"], rgb_ref.stamp_ns, max_sync_ms)
        thermal = nearest(refs["thermal"], rgb_ref.stamp_ns, max_sync_ms)
        if vis is None or nir is None or thermal is None:
            continue
        rgb_raw = decode_ros_image(rgb_ref.msg)
        vis_raw = decode_ros_image(vis.msg)
        nir_raw = decode_ros_image(nir.msg)
        thermal_raw = decode_ros_image(thermal.msg)
        if rgb_raw is None or vis_raw is None or nir_raw is None or thermal_raw is None:
            continue
        coreg = common_camera_crops(rgb_raw, vis_raw, nir_raw, thermal_raw, calib, margin_px, trim_bottom_px)
        if coreg is None:
            continue
        crops, _masks, roi = coreg
        cloud = nearest(refs["cloud"], rgb_ref.stamp_ns, max_sync_ms) if include_lidar else None
        depth, height, intensity, lidar_mask, _sparse_mask = project_ouster(
            cloud.msg if cloud else None,
            calib,
            rgb_raw.shape[:2],
            roi,
            max_range_m,
            splat_radius,
        )
        soil, _soil_mask, _gli, _metrics = soil_cover(crops["rgb"])
        vel = nearest(refs["vel"], rgb_ref.stamp_ns, max_sync_ms)
        frames.append(
            Frame(
                stamp_ns=rgb_ref.stamp_ns,
                bag_plot_id=pid,
                lat=lat,
                lon=lon,
                speed_mps=velocity_norm(vel.msg) if vel else None,
                rgb=crops["rgb"],
                vis=crops["vis"],
                nir=crops["nir"],
                thermal_c=crops["thermal"].astype(np.float32),
                thermal_mask=np.isfinite(crops["thermal"]),
                soil=soil,
                depth_mm=depth,
                depth_mask=lidar_mask,
                height_m=height,
                height_mask=lidar_mask,
                intensity=intensity,
                intensity_mask=lidar_mask,
                common_roi_rgb_xyxy=roi,
            )
        )
    return frames


def select_frames(frames: list[Frame], count: int) -> list[Frame]:
    if count <= 0 or len(frames) <= count:
        return frames
    idx = np.linspace(0, len(frames) - 1, count, dtype=int)
    return [frames[int(i)] for i in idx]


def prep_alignment_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.astype(np.float32)
        if gray.dtype != np.uint8:
            lo, hi = np.percentile(gray[np.isfinite(gray)], [2, 98]) if np.isfinite(gray).any() else (0, 1)
            gray = np.clip((gray - lo) * 255.0 / max(hi - lo, 1e-6), 0, 255).astype(np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)


def estimate_translation_phase(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray | None, float]:
    g1 = prep_alignment_gray(src).astype(np.float32)
    g2 = prep_alignment_gray(dst).astype(np.float32)
    if g1.shape != g2.shape or min(g1.shape) < 32:
        return None, 0.0
    win = cv2.createHanningWindow((g1.shape[1], g1.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(g1, g2, win)
    if not np.isfinite([dx, dy, response]).all() or response < 0.08:
        return None, float(response) if np.isfinite(response) else 0.0
    H = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return H, float(response)


def estimate_ecc_affine(src: np.ndarray, dst: np.ndarray, initial: np.ndarray | None = None) -> tuple[np.ndarray | None, float]:
    g1 = prep_alignment_gray(src)
    g2 = prep_alignment_gray(dst)
    scale = 0.5 if max(g1.shape) > 900 else 1.0
    if scale != 1.0:
        g1s = cv2.resize(g1, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        g2s = cv2.resize(g2, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        g1s, g2s = g1, g2
    warp = np.eye(2, 3, dtype=np.float32)
    if initial is not None:
        warp[:, :] = initial[:2, :3].astype(np.float32)
        warp[:, 2] *= scale
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 120, 1e-6)
    try:
        cc, warp = cv2.findTransformECC(g2s, g1s, warp, cv2.MOTION_AFFINE, criteria, None, 5)
    except cv2.error:
        return None, 0.0
    H = np.eye(3, dtype=np.float64)
    H[:2, :3] = warp.astype(np.float64)
    H[:2, 2] /= scale
    if not np.isfinite(H).all() or cc < 0.25:
        return None, float(cc) if np.isfinite(cc) else 0.0
    return H, float(cc)


def translation_from_homography(H: np.ndarray, shape_hw: tuple[int, int], vertical_only: bool) -> np.ndarray:
    h, w = shape_hw
    pts = np.float32([[w * 0.25, h * 0.25], [w * 0.75, h * 0.25], [w * 0.75, h * 0.75], [w * 0.25, h * 0.75]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    delta = np.median(warped - pts.reshape(-1, 2), axis=0)
    dx = 0.0 if vertical_only else float(delta[0])
    dy = float(delta[1])
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float64)


def constrain_pair_transform(H: np.ndarray, shape_hw: tuple[int, int], mode: str) -> np.ndarray:
    if mode == "translation":
        return translation_from_homography(H, shape_hw, vertical_only=False)
    if mode == "vertical_strip":
        return translation_from_homography(H, shape_hw, vertical_only=True)
    return H


def estimate_pair_transform(src: Frame, dst: Frame, trajectory_mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    H, inliers = estimate_homography(src.rgb, dst.rgb)
    if H is not None and inliers >= 25:
        H_out = constrain_pair_transform(H, src.rgb.shape[:2], trajectory_mode)
        return H_out, {"method": f"rgb_feature_{trajectory_mode}", "score": int(inliers), "inliers": int(inliers)}

    H_soil, soil_inliers = estimate_homography(src.soil, dst.soil)
    if H_soil is not None and soil_inliers >= 18:
        H_out = constrain_pair_transform(H_soil, src.rgb.shape[:2], trajectory_mode)
        return H_out, {"method": f"soil_feature_{trajectory_mode}", "score": int(soil_inliers), "inliers": int(soil_inliers), "rgb_inliers": int(inliers)}

    H_phase, response = estimate_translation_phase(src.rgb, dst.rgb)
    H_ecc, cc = estimate_ecc_affine(src.rgb, dst.rgb, H_phase)
    if H_ecc is not None:
        H_out = constrain_pair_transform(H_ecc, src.rgb.shape[:2], trajectory_mode)
        return H_out, {
            "method": f"rgb_ecc_{trajectory_mode}",
            "score": round(cc, 4),
            "ecc_cc": round(cc, 4),
            "phase_response": round(response, 4),
            "rgb_inliers": int(inliers),
            "soil_inliers": int(soil_inliers),
        }
    if H_phase is not None:
        H_out = constrain_pair_transform(H_phase, src.rgb.shape[:2], trajectory_mode)
        return H_out, {
            "method": f"rgb_phase_{trajectory_mode}",
            "score": round(response, 4),
            "phase_response": round(response, 4),
            "rgb_inliers": int(inliers),
            "soil_inliers": int(soil_inliers),
        }

    fallback = -0.28 * src.rgb.shape[0]
    H_fallback = np.array([[1, 0, 0], [0, 1, fallback], [0, 0, 1]], dtype=np.float64)
    return H_fallback, {
        "method": "fallback_translation",
        "score": 0,
        "rgb_inliers": int(inliers),
        "soil_inliers": int(soil_inliers),
    }


def temporal_transforms(frames: list[Frame], trajectory_mode: str) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    transforms = [np.eye(3, dtype=np.float64)]
    diagnostics = []
    for i in range(1, len(frames)):
        H, diag = estimate_pair_transform(frames[i], frames[i - 1], trajectory_mode)
        diag = {"pair": [i - 1, i], **diag}
        diagnostics.append(diag)
        transforms.append(transforms[-1] @ H)
    return transforms, diagnostics


def trajectory_quality(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    total = max(len(diagnostics), 1)
    reliable = sum(
        d["method"].startswith(("rgb_feature_", "soil_feature_", "rgb_ecc_", "rgb_phase_"))
        for d in diagnostics
    )
    fallback = sum(d["method"] == "fallback_translation" for d in diagnostics)
    return {
        "pairs": len(diagnostics),
        "reliable_pairs": reliable,
        "fallback_pairs": fallback,
        "reliable_fraction": round(reliable / total, 3),
        "recommended_use": "production_candidate" if fallback == 0 and reliable / total >= 0.85 else "visual_review_only",
    }



def canvas_geometry(shape_hw: tuple[int, int], transforms: list[np.ndarray]) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = shape_hw
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    pts = np.vstack([cv2.perspectiveTransform(corners, H) for H in transforms]).reshape(-1, 2)
    min_xy = np.floor(pts.min(axis=0)).astype(int)
    max_xy = np.ceil(pts.max(axis=0)).astype(int)
    pad = 120
    translate = np.array([[1, 0, -min_xy[0] + pad], [0, 1, -min_xy[1] + pad], [0, 0, 1]], dtype=np.float64)
    size = (int(max_xy[0] - min_xy[0] + 2 * pad), int(max_xy[1] - min_xy[1] + 2 * pad))
    return translate, size


def balance_color_images(images: list[np.ndarray]) -> list[np.ndarray]:
    if len(images) < 2:
        return images
    ref = images[len(images) // 2].astype(np.float32)
    ref_mean = ref.reshape(-1, 3).mean(axis=0)
    ref_std = ref.reshape(-1, 3).std(axis=0)
    balanced: list[np.ndarray] = []
    for img in images:
        src = img.astype(np.float32)
        pix = src.reshape(-1, 3)
        mean = pix.mean(axis=0)
        std = pix.std(axis=0)
        gain = np.clip(ref_std / np.maximum(std, 1.0), 0.75, 1.35)
        out = (src - mean[None, None, :]) * gain[None, None, :] + ref_mean[None, None, :]
        balanced.append(np.clip(out, 0, 255).astype(np.uint8))
    return balanced


def stitch_color(
    images: list[np.ndarray],
    transforms: list[np.ndarray],
    translate: np.ndarray,
    size: tuple[int, int],
    color_balance: bool = True,
    blend_mode: str = "feather",
) -> tuple[np.ndarray, np.ndarray]:
    out_w, out_h = size
    acc = np.zeros((out_h, out_w, 3), np.float32)
    weight = np.zeros((out_h, out_w), np.float32)
    if color_balance:
        images = balance_color_images(images)
    for img, H in zip(images, transforms):
        M = translate @ H
        warped = cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full(img.shape[:2], 255, np.uint8), M, (out_w, out_h), flags=cv2.INTER_NEAREST) > 0
        if blend_mode == "centerline":
            h = img.shape[0]
            yy = np.arange(h, dtype=np.float32)[:, None]
            local = np.exp(-0.5 * ((yy - h * 0.5) / max(h * 0.18, 1.0)) ** 2)
            local = np.repeat(local, img.shape[1], axis=1).astype(np.float32)
            feather = cv2.warpPerspective(local, M, (out_w, out_h), flags=cv2.INTER_LINEAR)
            feather *= mask.astype(np.float32)
        else:
            feather = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 11.0)
        acc += warped.astype(np.float32) * feather[:, :, None]
        weight += feather
    return np.divide(acc, np.maximum(weight[:, :, None], 1e-3)).astype(np.uint8), weight > 0.05


def stitch_scalar(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    transforms: list[np.ndarray],
    translate: np.ndarray,
    size: tuple[int, int],
    reducer: str = "mean",
    blend_mode: str = "feather",
) -> tuple[np.ndarray, np.ndarray]:
    out_w, out_h = size
    if reducer == "max":
        out = np.zeros((out_h, out_w), np.float32)
        valid_out = np.zeros((out_h, out_w), bool)
        for img, mask, H in zip(images, masks, transforms):
            M = translate @ H
            warped = cv2.warpPerspective(np.nan_to_num(img.astype(np.float32), nan=0), M, (out_w, out_h), flags=cv2.INTER_LINEAR)
            valid = cv2.warpPerspective(mask.astype(np.uint8) * 255, M, (out_w, out_h), flags=cv2.INTER_NEAREST) > 0
            upd = valid & (~valid_out | (warped > out))
            out[upd] = warped[upd]
            valid_out |= valid
        return out, valid_out
    acc = np.zeros((out_h, out_w), np.float32)
    weight = np.zeros((out_h, out_w), np.float32)
    for img, mask, H in zip(images, masks, transforms):
        M = translate @ H
        warped = cv2.warpPerspective(np.nan_to_num(img.astype(np.float32), nan=0), M, (out_w, out_h), flags=cv2.INTER_LINEAR)
        valid = cv2.warpPerspective(mask.astype(np.uint8) * 255, M, (out_w, out_h), flags=cv2.INTER_NEAREST) > 0
        if blend_mode == "centerline":
            h = img.shape[0]
            yy = np.arange(h, dtype=np.float32)[:, None]
            local = np.exp(-0.5 * ((yy - h * 0.5) / max(h * 0.18, 1.0)) ** 2)
            local = np.repeat(local, img.shape[1], axis=1).astype(np.float32)
            feather = cv2.warpPerspective(local, M, (out_w, out_h), flags=cv2.INTER_LINEAR)
            feather *= valid.astype(np.float32)
        else:
            feather = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), 5.0)
        acc += warped * feather
        weight += feather
    return acc / np.maximum(weight, 1e-6), weight > 0.05


def crop_to_valid(arrays: dict[str, np.ndarray], valid: np.ndarray) -> dict[str, np.ndarray]:
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return arrays
    x0, x1 = int(xs.min()), int(xs.max() + 1)
    y0, y1 = int(ys.min()), int(ys.max() + 1)
    return {k: v[y0:y1, x0:x1].copy() for k, v in arrays.items()}


def utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    zone = int((lon + 180.0) // 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


def plot_geometry_for_id(gpkg: Path, plot_id: str) -> gpd.GeoDataFrame:
    plots = load_plots(gpkg)
    row = plots[plots["plot_id"] == str(plot_id)]
    if row.empty:
        raise ValueError(f"plot_id {plot_id} not found in {gpkg}")
    return row


def write_array_geotiff(path: Path, arr: np.ndarray, transform: Any, crs: str, key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 3:
        # OpenCV arrays are BGR; GeoTIFF previews should open as RGB.
        data = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        profile = {
            "driver": "GTiff",
            "height": arr.shape[0],
            "width": arr.shape[1],
            "count": 3,
            "dtype": data.dtype,
            "crs": crs,
            "transform": transform,
            "compress": "deflate",
            "predictor": 2,
            "photometric": "RGB",
        }
    else:
        data = arr[None, :, :]
        profile = {
            "driver": "GTiff",
            "height": arr.shape[0],
            "width": arr.shape[1],
            "count": 1,
            "dtype": arr.dtype,
            "crs": crs,
            "transform": transform,
            "compress": "deflate",
        }
        if np.issubdtype(arr.dtype, np.integer):
            profile["nodata"] = 0
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        dst.update_tags(layer=key, georef_quality="approx_plot_bounds")


def write_qgis_exports(
    arrays: dict[str, np.ndarray],
    out_dir: Path,
    gpkg: Path,
    plot_id: str,
    trajectory_info: dict[str, Any],
) -> dict[str, str]:
    plot_wgs84 = plot_geometry_for_id(gpkg, plot_id)
    centroid = plot_wgs84.geometry.iloc[0].centroid
    epsg = utm_epsg_for_lonlat(float(centroid.x), float(centroid.y))
    crs = f"EPSG:{epsg}"
    plot_metric = plot_wgs84.to_crs(crs)
    minx, miny, maxx, maxy = plot_metric.total_bounds
    if maxx <= minx or maxy <= miny:
        raise ValueError(f"Invalid metric bounds for plot {plot_id}")

    qgis_dir = out_dir / "qgis"
    qgis_dir.mkdir(parents=True, exist_ok=True)
    plot_wgs84.to_file(qgis_dir / "plot_footprint_wgs84.geojson", driver="GeoJSON")
    plot_metric.to_file(qgis_dir / f"plot_footprint_{epsg}.geojson", driver="GeoJSON")

    outputs: dict[str, str] = {}
    for key in sorted(GEOTIFF_KEYS):
        arr = arrays.get(key)
        if arr is None or arr.ndim not in (2, 3):
            continue
        h, w = arr.shape[:2]
        transform = from_bounds(minx, miny, maxx, maxy, w, h)
        tif_path = qgis_dir / f"{key}.tif"
        write_array_geotiff(tif_path, arr, transform, crs, key)
        outputs[f"{key}_geotiff"] = str(tif_path)

    spatial_meta = {
        "georef_method": "approx_plot_bounds",
        "georef_crs": crs,
        "plot_id": str(plot_id),
        "metric_bounds_xy": [float(minx), float(miny), float(maxx), float(maxy)],
        "warning": (
            "These GeoTIFFs are QGIS-ready approximate plot-footprint exports. "
            "They share one spatial extent per layer, but are not yet SLAM/bundle-adjusted "
            "metric orthophotos."
        ),
        "trajectory_quality": trajectory_info,
        "files": outputs,
    }
    meta_path = qgis_dir / "qgis_export_metadata.json"
    meta_path.write_text(json.dumps(spatial_meta, indent=2), encoding="utf-8")
    outputs["qgis_export_metadata"] = str(meta_path)
    outputs["plot_footprint_wgs84"] = str(qgis_dir / "plot_footprint_wgs84.geojson")
    outputs["plot_footprint_metric"] = str(qgis_dir / f"plot_footprint_{epsg}.geojson")
    return outputs


def colorize_scalar(values: np.ndarray, mask: np.ndarray, cmap: int, unit: str, label: str, lohi: tuple[float, float] | None = None) -> np.ndarray:
    valid = mask.astype(bool) & np.isfinite(values)
    norm = np.zeros(values.shape, dtype=np.uint8)
    if int(valid.sum()) > 10:
        if lohi is None:
            lo, hi = np.percentile(values[valid], [2, 98])
        else:
            lo, hi = lohi
        if hi <= lo:
            hi = lo + 1.0
        norm[valid] = np.clip((values[valid] - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cmap)
    color[~valid] = (0, 0, 0)
    cv2.putText(color, f"{label} ({unit})", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return color


def write_outputs(
    frames: list[Frame],
    out_dir: Path,
    title: str,
    gpkg: Path | None = None,
    plot_id: str | None = None,
    write_geotiff: bool = True,
    trajectory_mode: str = "homography",
    blend_mode: str = "feather",
) -> dict[str, str]:
    transforms, diagnostics = temporal_transforms(frames, trajectory_mode)
    translate, size = canvas_geometry(frames[0].rgb.shape[:2], transforms)
    rgb_m, valid = stitch_color([f.rgb for f in frames], transforms, translate, size, blend_mode=blend_mode)
    vis_m, _ = stitch_color([f.vis for f in frames], transforms, translate, size, blend_mode=blend_mode)
    nir_m, _ = stitch_color([f.nir for f in frames], transforms, translate, size, blend_mode=blend_mode)
    soil_m, _ = stitch_color([f.soil for f in frames], transforms, translate, size, color_balance=False, blend_mode=blend_mode)
    thermal_m, thermal_valid = stitch_scalar([f.thermal_c for f in frames], [f.thermal_mask for f in frames], transforms, translate, size, blend_mode=blend_mode)
    depth_m, depth_valid = stitch_scalar([f.depth_mm for f in frames], [f.depth_mask for f in frames], transforms, translate, size, reducer="mean", blend_mode=blend_mode)
    height_m, height_valid = stitch_scalar([f.height_m for f in frames], [f.height_mask for f in frames], transforms, translate, size, reducer="max")
    intensity_m, intensity_valid = stitch_scalar([f.intensity for f in frames], [f.intensity_mask for f in frames], transforms, translate, size, reducer="max")

    arrays = crop_to_valid(
        {
            "rgb": rgb_m,
            "vis": vis_m,
            "nir": nir_m,
            "soil_cover": soil_m,
            "thermal_celsius_color": colorize_scalar(thermal_m, thermal_valid, cv2.COLORMAP_INFERNO, "C", "Thermal"),
            "depth_mm_color": colorize_scalar(depth_m, depth_valid, cv2.COLORMAP_TURBO, "mm", "Ouster depth"),
            "height_m_color": colorize_scalar(height_m, height_valid, cv2.COLORMAP_VIRIDIS, "m", "Ouster height"),
            "intensity_color": colorize_scalar(intensity_m, intensity_valid, cv2.COLORMAP_MAGMA, "a.u.", "Ouster intensity"),
            "depth_mm": np.clip(depth_m, 0, 65535).astype(np.uint16),
            "height_m_float": height_m.astype(np.float32),
            "thermal_celsius_float": thermal_m.astype(np.float32),
            "camera_valid_mask": valid.astype(np.uint8) * 255,
            "ouster_valid_mask": depth_valid.astype(np.uint8) * 255,
        },
        valid,
    )
    quality = trajectory_quality(diagnostics)
    outputs: dict[str, str] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, img in arrays.items():
        if img.dtype == np.float32:
            path = out_dir / f"{key}.npy"
            np.save(path, img)
        else:
            ext = ".png" if img.dtype == np.uint16 or key.endswith("mask") else ".jpg"
            path = out_dir / f"{key}{ext}"
            cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        outputs[key] = str(path)
    sheet = make_sheet(arrays, title)
    sheet_path = out_dir / "multisensor_orthomosaic_sheet.jpg"
    cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])
    outputs["sheet"] = str(sheet_path)
    if write_geotiff and gpkg is not None and plot_id is not None:
        outputs.update(write_qgis_exports(arrays, out_dir, gpkg, plot_id, quality))
    meta = {
        "trajectory_diagnostics": diagnostics,
        "trajectory_quality": quality,
        "canvas_size_wh": list(size),
        "outputs": outputs,
    }
    (out_dir / "mosaic_internal_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return outputs | {"_diagnostics": json.dumps(diagnostics)}


def make_sheet(arrays: dict[str, np.ndarray], title: str) -> np.ndarray:
    has_ouster = bool(np.any(arrays.get("ouster_valid_mask", np.zeros((1, 1), dtype=np.uint8)) > 0))
    specs = [
        ("rgb", "RGB"),
        ("vis", "VIS"),
        ("nir", "NIR"),
        ("thermal_celsius_color", "Thermal C"),
        ("soil_cover", "Soil cover"),
    ]
    if has_ouster:
        specs.extend(
            [
                ("depth_mm_color", "Depth mm"),
                ("height_m_color", "Height"),
                ("intensity_color", "Intensity"),
            ]
        )
    tiles = []
    for key, label in specs:
        img = arrays.get(key)
        if img is None or img.ndim != 3:
            continue
        tiles.append(label_tile(img, label, (440, 300)))
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 245))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    width = rows[0].shape[1]
    header = np.full((78, width, 3), 15, dtype=np.uint8)
    cv2.putText(header, title[:100], (22, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(header, "RGB-master common crop + shared RGB visual trajectory", (22, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)
    return np.vstack([header] + rows)


def label_tile(img: np.ndarray, label: str, size: tuple[int, int]) -> np.ndarray:
    tw, th = size
    h, w = img.shape[:2]
    scale = min(tw / max(w, 1), (th - 34) / max(h, 1))
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((th, tw, 3), 245, dtype=np.uint8)
    x = (tw - small.shape[1]) // 2
    y = 34 + (th - 34 - small.shape[0]) // 2
    canvas[y:y + small.shape[0], x:x + small.shape[1]] = small
    cv2.putText(canvas, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bag", type=Path, required=True)
    ap.add_argument("--gpkg", type=Path, required=True)
    ap.add_argument("--plot-id", default="auto")
    ap.add_argument("--center-ns", type=int)
    ap.add_argument("--window-ms", type=float, default=12000.0)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    ap.add_argument("--max-sync-ms", type=float, default=900.0)
    ap.add_argument("--max-range-m", type=float, default=8.0)
    ap.add_argument("--splat-radius", type=int, default=11)
    ap.add_argument("--margin-px", type=int, default=2)
    ap.add_argument("--trim-bottom-px", type=int, default=0, help="Remove this many RGB-master crop pixels from the bottom after common sensor intersection.")
    ap.add_argument(
        "--trajectory-mode",
        choices=["homography", "translation", "vertical_strip"],
        default="homography",
        help="Temporal mosaic geometry. vertical_strip keeps plot width constant and only accumulates forward motion.",
    )
    ap.add_argument("--blend-mode", choices=["feather", "centerline"], default="feather")
    ap.add_argument("--skip-lidar", action="store_true")
    ap.add_argument("--no-geotiff", action="store_true", help="Do not write approximate QGIS GeoTIFF exports.")
    ap.add_argument("--scan-plots", action="store_true")
    args = ap.parse_args()

    if args.scan_plots:
        print(json.dumps(scan_plot_counts(args.bag, args.gpkg, args.center_ns, args.window_ms, args.max_sync_ms), indent=2))
        return 0

    calib = load_final_calibration(args.calibration)
    plot_id = args.plot_id
    if plot_id == "auto":
        scan = scan_plot_counts(args.bag, args.gpkg, args.center_ns, args.window_ms, args.max_sync_ms)
        counts = scan["plot_frame_counts"]
        if not counts:
            raise SystemExit("No plot GPS hits found. Run with --scan-plots or provide a different window.")
        plot_id = max(counts.items(), key=lambda kv: kv[1])[0]
        if args.center_ns is None:
            args.center_ns = scan["example_center_ns"][plot_id]
    frames_all = build_frames(
        args.bag,
        args.gpkg,
        str(plot_id),
        calib,
        args.center_ns,
        args.window_ms,
        args.max_sync_ms,
        args.max_range_m,
        args.splat_radius,
        args.margin_px,
        args.trim_bottom_px,
        include_lidar=not args.skip_lidar,
    )
    frames = select_frames(frames_all, args.frames)
    if len(frames) < 2:
        raise SystemExit(f"Need at least 2 synchronized frames for plot {plot_id}; found {len(frames)}")
    title = f"Plot {plot_id} | {args.bag.name} | {len(frames)} frames"
    outputs = write_outputs(
        frames,
        args.out,
        title,
        gpkg=args.gpkg,
        plot_id=str(plot_id),
        write_geotiff=not args.no_geotiff,
        trajectory_mode=args.trajectory_mode,
        blend_mode=args.blend_mode,
    )
    summary = {
        "bag": str(args.bag),
        "plot_id": str(plot_id),
        "candidate_frames": len(frames_all),
        "used_frames": len(frames),
        "timestamps_ns": [int(f.stamp_ns) for f in frames],
        "lat_lon": [[f.lat, f.lon] for f in frames],
        "speed_mps": [f.speed_mps for f in frames],
        "common_roi_rgb_xyxy_per_frame": [list(f.common_roi_rgb_xyxy) for f in frames],
        "trim_bottom_px": int(args.trim_bottom_px),
        "trajectory_mode": args.trajectory_mode,
        "blend_mode": args.blend_mode,
        "outputs": {k: v for k, v in outputs.items() if not k.startswith("_")},
        "notes": [
            "Camera layers are dense inside the RGB-master common crop.",
            "Ouster layers are sparse/masked measurements; depth_mm is metric where ouster_valid_mask is nonzero.",
            "This is local visual orthomosaic, not yet global SLAM/bundle-adjusted georectification.",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rgb_master_multisensor_orthomosaic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
