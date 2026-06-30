#!/usr/bin/env python3
"""Extract presentation-ready multisensor plot images from ROS1 bags.

The script samples RGB frames, assigns each sample to the closest field plot
using GNSS + a GPKG layout, synchronizes the other camera topics and Ouster,
then writes cropped camera images, soil-cover masks, and depth/height maps.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.interpolate import griddata
from shapely.geometry import Point

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.extract_ouster_panel_from_manual_intensity_roi import cloud_to_organized, stamp_ns  # noqa: E402
from src.calibration.render_full_cloud_rgb_overlay import K_RGB  # noqa: E402
from src.depth.projected_depth_experiment import transform_points  # noqa: E402


TOPICS = {
    "rgb": "/ssf/BFS_usb_0/image_raw",
    "vis": "/ssf/photonfocus_camera_vis_node/image_raw",
    "nir": "/ssf/photonfocus_camera_nir_node/image_raw",
    "thermal": "/ssf/thermalgrabber_ros/image_deg_celsius",
    "thermal_raw": "/ssf/thermalgrabber_ros/image_mono16",
    "ouster_intensity": "/ssf/img_node/intensity_image",
    "ouster_range": "/ssf/img_node/range_image",
    "cloud": "/ssf/os1_cloud_node/points",
    "gps": "/ssf/gnss/fix",
}

DEFAULT_HOMOGRAPHIES = Path("data/matrices/mixed_vis_nir_thermal_homographies.json")


@dataclass
class MessageRef:
    topic: str
    stamp_ns: int
    msg: object


def decode_ros_image(msg) -> np.ndarray | None:
    enc = str(msg.encoding).lower().strip()
    h, w = int(msg.height), int(msg.width)
    raw = bytes(msg.data)
    if enc in ("rgb8",):
        return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)[:, :, ::-1].copy()
    if enc in ("bgr8",):
        return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
    if enc.startswith("bayer"):
        bayer = np.frombuffer(raw, dtype=np.uint8).reshape(h, w)
        code_map = {
            "bayer_rggb8": cv2.COLOR_BAYER_BG2BGR,
            "bayer_bggr8": cv2.COLOR_BAYER_RG2BGR,
            "bayer_gbrg8": cv2.COLOR_BAYER_GR2BGR,
            "bayer_grbg8": cv2.COLOR_BAYER_GB2BGR,
        }
        return cv2.cvtColor(bayer, code_map.get(enc, cv2.COLOR_BAYER_BG2BGR))
    if enc in ("mono8", "8uc1"):
        return np.frombuffer(raw, dtype=np.uint8).reshape(h, w).copy()
    if enc in ("mono16", "16uc1", "16sc1"):
        dtype = np.int16 if enc == "16sc1" else np.uint16
        return np.frombuffer(raw, dtype=dtype).reshape(h, w).copy()
    if enc in ("32fc1",):
        return np.frombuffer(raw, dtype=np.float32).reshape(h, w).copy()
    if enc in ("64fc1",):
        return np.frombuffer(raw, dtype=np.float64).reshape(h, w).astype(np.float32)
    return None


def photonfocus_preview(image: np.ndarray, pattern: int = 4) -> np.ndarray:
    """Convert raw Photonfocus spectral mosaic into a display/registration preview."""
    if image.ndim != 2:
        return image
    image = image[:1024, :]
    usable_h = (image.shape[0] // pattern) * pattern
    usable_w = (image.shape[1] // pattern) * pattern
    image = image[:usable_h, :usable_w]
    bands = []
    for row_offset in range(pattern):
        for col_offset in range(pattern):
            band = image[row_offset::pattern, col_offset::pattern]
            band = cv2.medianBlur(band, 3)
            bands.append(band.astype(np.float32))
    preview = np.mean(np.stack(bands, axis=-1), axis=-1)
    return np.clip(preview, 0, np.iinfo(image.dtype).max).astype(image.dtype)


def sensor_display_image(img: np.ndarray, key: str) -> np.ndarray:
    """Return a BGR display image in the geometry expected by stored homographies."""
    if key == "vis" and img.ndim == 2 and img.shape[0] > 300:
        img = photonfocus_preview(img, pattern=4)
    elif key == "nir" and img.ndim == 2 and img.shape[0] > 300:
        # The available NIR->VIS homography was built on the historical preview
        # extractor, which used the same 4x4 preview geometry.
        img = photonfocus_preview(img, pattern=4)
    return to_bgr_for_display(img, key)


def robust_colorize(img: np.ndarray, cmap: int, invert: bool = False, valid_mask: np.ndarray | None = None) -> np.ndarray:
    src = img.astype(np.float32)
    finite = np.isfinite(src) if valid_mask is None else (np.isfinite(src) & valid_mask.astype(bool))
    out = np.zeros(src.shape, dtype=np.uint8)
    if finite.any():
        lo, hi = np.percentile(src[finite], [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        norm = np.clip((src - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
        out = 255 - norm if invert else norm
    return cv2.applyColorMap(out, cmap)


def colorize_with_scale(
    img: np.ndarray,
    cmap: int,
    label: str,
    unit: str,
    valid_mask: np.ndarray | None = None,
    invert: bool = False,
) -> np.ndarray:
    src = img.astype(np.float32)
    valid = np.isfinite(src) if valid_mask is None else (np.isfinite(src) & valid_mask.astype(bool))
    color = robust_colorize(src, cmap, invert=invert, valid_mask=valid)
    if valid.any():
        lo, hi = np.percentile(src[valid], [2, 98])
    else:
        lo, hi = 0.0, 1.0
    bar_w = 76
    h, w = color.shape[:2]
    canvas = np.full((h, w + bar_w, 3), 18, dtype=np.uint8)
    canvas[:, :w] = color
    grad = np.linspace(255, 0, h, dtype=np.uint8).reshape(h, 1)
    if invert:
        grad = 255 - grad
    bar = cv2.applyColorMap(np.repeat(grad, 22, axis=1), cmap)
    canvas[:, w + 10:w + 32] = bar
    cv2.putText(canvas, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"{hi:.2f} {unit}".rstrip(), (w + 4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{lo:.2f} {unit}".rstrip(), (w + 4, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def enhance_height_contrast(values: np.ndarray, support: np.ndarray) -> np.ndarray:
    valid = support.astype(bool) & np.isfinite(values)
    out = np.zeros(values.shape, dtype=np.float32)
    if int(valid.sum()) < 10:
        return out
    lo, hi = np.percentile(values[valid], [8, 99.5])
    if hi <= lo:
        hi = lo + 0.05
    rel = np.clip(values.astype(np.float32) - float(lo), 0.0, float(hi - lo))
    norm = np.clip(rel / float(hi - lo), 0.0, 1.0)
    # Gamma expansion makes small crop-height differences visible without inventing
    # continuous surfaces between Ouster scan lines.
    out[valid] = (np.power(norm[valid], 0.65) * float(hi - lo)).astype(np.float32)
    return out


def crop_fraction(img: np.ndarray, frac: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = int(round(w * frac[0]))
    y0 = int(round(h * frac[1]))
    x1 = int(round(w * frac[2]))
    y1 = int(round(h * frac[3]))
    return img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)].copy()


def crop_box(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    h, w = img.shape[:2]
    return img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)].copy()


def load_homographies(path: Path) -> dict[str, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    h = data["homographies"]
    return {
        "rgb": np.asarray(h["rgb_to_vis"], dtype=np.float64),
        "vis": np.eye(3, dtype=np.float64),
        "nir": np.asarray(h["nir_to_vis"], dtype=np.float64),
        "thermal": np.asarray(h["thermal_deg_to_vis"], dtype=np.float64),
    }


def warp_to_vis(img: np.ndarray, H: np.ndarray, vis_shape: tuple[int, int], interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    h, w = vis_shape
    return cv2.warpPerspective(img, H, (w, h), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def common_valid_box(masks: list[np.ndarray], margin: int = 3, bottom_shave_frac: float = 0.0) -> tuple[int, int, int, int] | None:
    if not masks:
        return None
    common = np.ones(masks[0].shape[:2], dtype=bool)
    for m in masks:
        common &= m.astype(bool)
    if bottom_shave_frac > 0:
        h = common.shape[0]
        common[int(h * (1.0 - bottom_shave_frac)):, :] = False
    ys, xs = np.where(common)
    if len(xs) == 0:
        return None
    x0 = max(0, int(xs.min()) + margin)
    y0 = max(0, int(ys.min()) + margin)
    x1 = min(common.shape[1], int(xs.max()) - margin + 1)
    y1 = min(common.shape[0], int(ys.max()) - margin + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def to_bgr_for_display(img: np.ndarray, kind: str) -> np.ndarray:
    if img.ndim == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lo, hi = np.percentile(l, [1, 99])
        if hi <= lo:
            hi = lo + 1
        l2 = np.clip((l.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
        l2 = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(l2)
        return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
    cmap = cv2.COLORMAP_INFERNO if kind.startswith("thermal") else cv2.COLORMAP_VIRIDIS
    return robust_colorize(img, cmap)


def soil_cover(rgb_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    denom = (2.0 * g + r + b)
    gli = np.divide(2.0 * g - r - b, denom, out=np.zeros_like(g), where=denom > 1e-6)
    gli_u8 = np.clip((gli + 1.0) * 127.5, 0, 255).astype(np.uint8)
    _thr, veg_u8 = cv2.threshold(gli_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    green_hue = (hsv[:, :, 0] > 25) & (hsv[:, :, 0] < 100) & (hsv[:, :, 1] > 25)
    veg = (veg_u8 > 0) & green_hue
    veg = cv2.morphologyEx(veg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
    veg = cv2.morphologyEx(veg.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool)
    overlay = rgb_bgr.copy()
    veg_color = np.array([30, 255, 80], dtype=np.float32)
    soil_color = np.array([28, 28, 28], dtype=np.float32)
    overlay_f = overlay.astype(np.float32)
    overlay_f[veg] = overlay_f[veg] * 0.20 + veg_color * 0.80
    overlay_f[~veg] = overlay_f[~veg] * 0.35 + soil_color * 0.65
    overlay = np.clip(overlay_f, 0, 255).astype(np.uint8)
    mask_vis = np.zeros_like(rgb_bgr)
    mask_vis[veg] = (30, 255, 80)
    gli_vis = colorize_with_scale(gli, cv2.COLORMAP_VIRIDIS, "GLI", "", valid_mask=np.ones_like(gli, dtype=bool))
    metrics = {
        "vegetation_cover_pct": round(float(veg.mean() * 100.0), 2),
        "soil_visible_pct": round(float((1.0 - veg.mean()) * 100.0), 2),
        "gli_otsu_threshold_u8": int(_thr),
        "method": "GLI auto-threshold (Otsu) constrained by green hue",
    }
    return overlay, mask_vis, gli_vis, metrics


def load_plots(gpkg: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(gpkg)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")
    gdf["plot_id"] = [str(i + 1) for i in range(len(gdf))]
    metric = gdf.to_crs("EPSG:32632")
    centers = metric.geometry.centroid.to_crs("EPSG:4326")
    gdf["centroid_lon"] = centers.x
    gdf["centroid_lat"] = centers.y
    return gdf


def assign_plot(gdf: gpd.GeoDataFrame, lon: float | None, lat: float | None) -> dict:
    if lon is None or lat is None or not np.isfinite([lon, lat]).all():
        return {"plot_id": None, "assignment": "no_gps"}
    pt = Point(lon, lat)
    contains = gdf[gdf.geometry.contains(pt)]
    if not contains.empty:
        row = contains.iloc[0]
        dist_m = 0.0
        assignment = "inside_polygon"
    else:
        metric = gdf.to_crs("EPSG:32632")
        pt_m = gpd.GeoSeries([pt], crs="EPSG:4326").to_crs("EPSG:32632").iloc[0]
        dists = metric.geometry.distance(pt_m)
        row = gdf.iloc[int(dists.idxmin())]
        dist_m = float(dists.min())
        assignment = "nearest_polygon"
    return {
        "plot_id": str(row["plot_id"]),
        "assignment": assignment,
        "distance_m": round(dist_m, 2),
        "sorte": str(row.get("Sorte ", "")),
        "variante": str(row.get("Variante ", "")),
        "area_m2": float(row.get("Area", math.nan)),
        "centroid_lon": float(row["centroid_lon"]),
        "centroid_lat": float(row["centroid_lat"]),
    }


def nearest_by_time(items: list[MessageRef], target_ns: int, max_dt_ms: float) -> MessageRef | None:
    if not items:
        return None
    best = min(items, key=lambda x: abs(x.stamp_ns - target_ns))
    if abs(best.stamp_ns - target_ns) > max_dt_ms * 1_000_000:
        return None
    return best


def collect_refs(bag: Path) -> dict[str, list[MessageRef]]:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    wanted = set(TOPICS.values())
    refs = {k: [] for k in TOPICS}
    with Reader(bag) as reader:
        conns = [c for c in reader.connections if c.topic in wanted]
        for conn, ts, raw in reader.messages(connections=conns):
            key = next(k for k, v in TOPICS.items() if v == conn.topic)
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            msg_stamp = stamp_ns(msg, ts) if key == "cloud" else int(ts)
            refs[key].append(MessageRef(conn.topic, msg_stamp, msg))
    for values in refs.values():
        values.sort(key=lambda x: x.stamp_ns)
    return refs


def collect_rgb_times(bag: Path) -> list[int]:
    times: list[int] = []
    with Reader(bag) as reader:
        conns = [c for c in reader.connections if c.topic == TOPICS["rgb"]]
        for _, ts, _ in reader.messages(connections=conns):
            times.append(int(ts))
    return times


def bag_time_bounds(bag: Path) -> tuple[int, int]:
    reader = Reader(bag)
    reader.open()
    try:
        return int(reader.start_time), int(reader.end_time)
    finally:
        reader.close()


def choose_target_times_from_bounds(start_ns: int, end_ns: int, count: int) -> list[int]:
    if count <= 1:
        return [int((start_ns + end_ns) // 2)]
    return [int(v) for v in np.linspace(start_ns, end_ns, count + 2, dtype=np.int64)[1:-1]]


def choose_target_times(rgb_times: list[int], count: int) -> list[int]:
    if len(rgb_times) <= count:
        return rgb_times
    idx = np.linspace(0, len(rgb_times) - 1, count + 2, dtype=int)[1:-1]
    return [rgb_times[int(i)] for i in idx]


def collect_nearest_refs_for_targets(bag: Path, target_times: list[int], max_sync_ms: float) -> list[dict[str, MessageRef | None]]:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    wanted = set(TOPICS.values())
    max_dt_ns = int(max_sync_ms * 1_000_000)
    target_refs: list[dict[str, MessageRef | None]] = [{k: None for k in TOPICS} for _ in target_times]
    target_dts: list[dict[str, int]] = [{k: 10**30 for k in TOPICS} for _ in target_times]

    start = min(target_times) - max(int(max_sync_ms * 1_000_000), int(2500.0 * 1_000_000))
    stop = max(target_times) + max(int(max_sync_ms * 1_000_000), int(2500.0 * 1_000_000))
    with Reader(bag) as reader:
        conns = [c for c in reader.connections if c.topic in wanted]
        for conn, ts, raw in reader.messages(connections=conns, start=start, stop=stop):
            key = next(k for k, v in TOPICS.items() if v == conn.topic)
            msg_time = int(ts)
            nearest_i = int(np.argmin([abs(msg_time - t) for t in target_times]))
            dt = abs(msg_time - target_times[nearest_i])
            allowed = max_dt_ns
            if key == "gps":
                allowed = max(allowed, int(2500.0 * 1_000_000))
            if dt > allowed or dt >= target_dts[nearest_i][key]:
                continue
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            msg_stamp = stamp_ns(msg, ts) if key == "cloud" else msg_time
            target_refs[nearest_i][key] = MessageRef(conn.topic, msg_stamp, msg)
            target_dts[nearest_i][key] = dt
    return target_refs


def collect_cloud_refs_near(bag: Path, target_ns: int, window_ms: float) -> list[MessageRef]:
    if window_ms <= 0:
        return []
    typestore = get_typestore(Stores.ROS1_NOETIC)
    half_window_ns = int(window_ms * 1_000_000)
    start = int(target_ns - half_window_ns)
    stop = int(target_ns + half_window_ns)
    refs: list[MessageRef] = []
    with Reader(bag) as reader:
        conns = [c for c in reader.connections if c.topic == TOPICS["cloud"]]
        for conn, ts, raw in reader.messages(connections=conns, start=start, stop=stop):
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            refs.append(MessageRef(conn.topic, stamp_ns(msg, ts), msg))
    refs.sort(key=lambda x: abs(x.stamp_ns - target_ns))
    return refs


def choose_targets(rgb_refs: list[MessageRef], count: int) -> list[MessageRef]:
    if len(rgb_refs) <= count:
        return rgb_refs
    idx = np.linspace(0, len(rgb_refs) - 1, count + 2, dtype=int)[1:-1]
    return [rgb_refs[int(i)] for i in idx]


def ground_relative_height(xyz: np.ndarray) -> np.ndarray:
    if len(xyz) < 20:
        return np.zeros(len(xyz), dtype=np.float32)
    pts = xyz.astype(np.float64)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    out = np.zeros(len(xyz), dtype=np.float32)
    if len(pts) < 20:
        return out

    z_limit = np.percentile(pts[:, 2], 65)
    low = pts[:, 2] <= z_limit
    if int(low.sum()) < 12:
        low = np.ones(len(pts), dtype=bool)
    a = np.column_stack([pts[low, 0], pts[low, 1], np.ones(int(low.sum()))])
    try:
        coeff, *_ = np.linalg.lstsq(a, pts[low, 2], rcond=None)
        plane_z = coeff[0] * pts[:, 0] + coeff[1] * pts[:, 1] + coeff[2]
        rel = pts[:, 2] - plane_z
    except np.linalg.LinAlgError:
        rel = pts[:, 2] - np.percentile(pts[:, 2], 8)
    rel -= np.percentile(rel, 5)
    rel = np.clip(rel, 0.0, None)
    out[finite] = rel.astype(np.float32)
    return out


def modulate_brightness_by_intensity(color: np.ndarray, intensity: np.ndarray, support: np.ndarray) -> np.ndarray:
    valid = support.astype(bool) & np.isfinite(intensity)
    if not valid.any():
        return color
    lo, hi = np.percentile(intensity[valid], [5, 99])
    if hi <= lo:
        hi = lo + 1.0
    gain = np.zeros(intensity.shape, dtype=np.float32)
    gain[valid] = np.clip((intensity[valid] - lo) / (hi - lo), 0.0, 1.0)
    gain = cv2.GaussianBlur(gain, (0, 0), 0.6)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (0.85 + 0.35 * gain), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (0.82 + 0.75 * gain), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def shift_image_x(img: np.ndarray, shift_px: int) -> np.ndarray:
    if shift_px == 0:
        return img
    h, w = img.shape[:2]
    matrix = np.array([[1.0, 0.0, float(shift_px)], [0.0, 1.0, 0.0]], dtype=np.float32)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def shift_image_xy(img: np.ndarray, dx: float, dy: float, nearest: bool = False) -> np.ndarray:
    h, w = img.shape[:2]
    matrix = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.warpAffine(img, matrix, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def normalize_score(values: np.ndarray, support: np.ndarray, lo_pct: float, hi_pct: float, gamma: float) -> np.ndarray:
    valid = support.astype(bool) & np.isfinite(values)
    out = np.zeros(values.shape, dtype=np.float32)
    if int(valid.sum()) < 10:
        return out
    lo, hi = np.percentile(values[valid], [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    out[valid] = np.clip((values[valid] - lo) / (hi - lo), 0.0, 1.0)
    out[valid] = np.power(out[valid], gamma)
    return out


def render_ouster_intensity_soft_veg(
    cloud_refs: list[MessageRef],
    transform: np.ndarray,
    rgb_shape: tuple[int, int],
    homographies: dict[str, np.ndarray],
    vis_shape: tuple[int, int],
    box: tuple[int, int, int, int],
    bottom_shave_frac: float,
    veg_mask_vis: np.ndarray,
    max_range_m: float,
    splat_radius: int = 21,
    interlace_span_px: float = 10.0,
) -> np.ndarray | None:
    if not cloud_refs:
        return None
    n = len(cloud_refs)
    intensity_acc: np.ndarray | None = None
    support_acc: np.ndarray | None = None
    offsets_y = np.linspace(-interlace_span_px, interlace_span_px, n, dtype=np.float32) if n > 1 else np.zeros(n, dtype=np.float32)
    for idx, ref in enumerate(cloud_refs):
        _depth, _height, intensity, support = projected_depth_height_rgb_full(ref.msg, transform, rgb_shape, max_range_m)
        intensity_s, support_s = splat_for_display(intensity, support, radius=splat_radius)
        intensity_w = warp_to_vis(intensity_s, homographies["rgb"], vis_shape)
        support_w = warp_to_vis((support_s.astype(np.uint8) * 255), homographies["rgb"], vis_shape, cv2.INTER_NEAREST) > 0
        intensity_c = crop_box(intensity_w, box)
        support_c = crop_box(support_w.astype(np.uint8), box).astype(bool)
        if bottom_shave_frac > 0:
            keep_h = max(1, int(intensity_c.shape[0] * (1.0 - bottom_shave_frac)))
            intensity_c = intensity_c[:keep_h, :]
            support_c = support_c[:keep_h, :]
        dx = float(((idx % 3) - 1) * 0.45)
        dy = float(offsets_y[idx])
        intensity_c = shift_image_xy(intensity_c, dx, dy)
        support_c = shift_image_xy((support_c.astype(np.uint8) * 255), dx, dy, nearest=True) > 0
        if intensity_acc is None:
            intensity_acc = np.zeros_like(intensity_c, dtype=np.float32)
            support_acc = np.zeros_like(support_c, dtype=bool)
        valid = support_c.astype(bool)
        intensity_acc[valid] = np.maximum(intensity_acc[valid], intensity_c[valid])
        support_acc |= valid
    if intensity_acc is None or support_acc is None or not support_acc.any():
        return None

    veg_crop = crop_box(veg_mask_vis.astype(np.uint8), box).astype(bool)
    if bottom_shave_frac > 0:
        veg_crop = veg_crop[:intensity_acc.shape[0], :]
    soft_veg = cv2.GaussianBlur(veg_crop.astype(np.float32), (0, 0), 3.0)
    score = normalize_score(intensity_acc, support_acc, 8, 97, gamma=0.75)
    score = score * (0.18 + 0.82 * soft_veg)
    color = cv2.applyColorMap(np.clip(score * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    out = np.zeros_like(color)
    out[support_acc] = color[support_acc]
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return out


def colorize_height_panel(height: np.ndarray, intensity: np.ndarray, support: np.ndarray) -> np.ndarray:
    valid = support.astype(bool) & np.isfinite(height)
    out = np.zeros((*height.shape, 3), dtype=np.uint8)
    if int(valid.sum()) < 10:
        return out
    lo, hi = np.percentile(height[valid], [3, 99])
    if hi <= lo:
        hi = lo + 0.03
    norm = np.zeros(height.shape, dtype=np.float32)
    norm[valid] = np.clip((height[valid] - lo) / (hi - lo), 0.0, 1.0)
    norm[valid] = np.power(norm[valid], 0.55)
    gray = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
    color = modulate_brightness_by_intensity(color, intensity, valid)
    out[valid] = color[valid]
    out = cv2.dilate(out, np.ones((3, 3), np.uint8))
    return out


def projected_depth_height_rgb(
    cloud_msg,
    transform: np.ndarray,
    rgb_shape: tuple[int, int],
    crop: tuple[float, float, float, float],
    max_range_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cloud = cloud_to_organized(cloud_msg)
    xyz = np.dstack([cloud["x"], cloud["y"], cloud["z"]]).reshape(-1, 3).astype(np.float64)
    rng = np.linalg.norm(xyz, axis=1)
    ok = np.isfinite(xyz).all(axis=1) & (rng > 0.4) & (rng < max_range_m)
    xyz = xyz[ok]
    pts_cam = transform_points(xyz, transform)
    z = pts_cam[:, 2]
    uvw = (K_RGB @ pts_cam.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    h, w = rgb_shape
    inside = (z > 0.05) & np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv = uv[inside]
    z = z[inside]
    xyz_inside = xyz[inside]
    if len(z) == 0:
        empty = crop_fraction(np.zeros((h, w), np.float32), crop)
        return empty, empty.copy(), np.zeros_like(empty, dtype=bool)
    px = np.round(uv[:, 0]).astype(np.int32)
    py = np.round(uv[:, 1]).astype(np.int32)
    pix_ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    px = px[pix_ok]
    py = py[pix_ok]
    z = z[pix_ok]
    xyz_inside = xyz_inside[pix_ok]
    height = ground_relative_height(xyz_inside)
    if len(z) == 0:
        empty = crop_fraction(np.zeros((h, w), np.float32), crop)
        return empty, empty.copy(), np.zeros_like(empty, dtype=bool)
    lin = py * w + px
    order = np.lexsort((z, lin))
    first = np.r_[True, lin[order][1:] != lin[order][:-1]]
    keep = order[first]
    depth = np.zeros((h, w), dtype=np.float32)
    height_img = np.zeros((h, w), dtype=np.float32)
    support = np.zeros((h, w), dtype=np.uint8)
    depth[py[keep], px[keep]] = z[keep].astype(np.float32)
    base = float(np.percentile(height, 5))
    height_img[py[keep], px[keep]] = (height[keep] - base).astype(np.float32)
    support[py[keep], px[keep]] = 255
    return crop_fraction(depth, crop), crop_fraction(height_img, crop), crop_fraction(support, crop).astype(bool)


def projected_depth_height_rgb_full(
    cloud_msg,
    transform: np.ndarray,
    rgb_shape: tuple[int, int],
    max_range_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cloud = cloud_to_organized(cloud_msg)
    xyz = np.dstack([cloud["x"], cloud["y"], cloud["z"]]).reshape(-1, 3).astype(np.float64)
    intensity_all = cloud.get("intensity")
    intensity = np.zeros(len(xyz), dtype=np.float32)
    if intensity_all is not None:
        intensity = np.asarray(intensity_all).reshape(-1).astype(np.float32)
    rng = np.linalg.norm(xyz, axis=1)
    ok = np.isfinite(xyz).all(axis=1) & (rng > 0.4) & (rng < max_range_m)
    xyz = xyz[ok]
    intensity = intensity[ok]
    h, w = rgb_shape
    if len(xyz) == 0:
        empty = np.zeros((h, w), np.float32)
        return empty, empty.copy(), empty.copy(), np.zeros((h, w), bool)
    pts_cam = transform_points(xyz, transform)
    z = pts_cam[:, 2]
    uvw = (K_RGB @ pts_cam.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    inside = (z > 0.05) & np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv = uv[inside]
    z = z[inside]
    xyz_inside = xyz[inside]
    intensity = intensity[inside]
    if len(z) == 0:
        empty = np.zeros((h, w), np.float32)
        return empty, empty.copy(), empty.copy(), np.zeros((h, w), bool)
    px = np.round(uv[:, 0]).astype(np.int32)
    py = np.round(uv[:, 1]).astype(np.int32)
    pix_ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    px, py, z, xyz_inside, intensity = px[pix_ok], py[pix_ok], z[pix_ok], xyz_inside[pix_ok], intensity[pix_ok]
    height = ground_relative_height(xyz_inside)
    if len(z) == 0:
        empty = np.zeros((h, w), np.float32)
        return empty, empty.copy(), empty.copy(), np.zeros((h, w), bool)
    lin = py * w + px
    order = np.lexsort((z, lin))
    first = np.r_[True, lin[order][1:] != lin[order][:-1]]
    keep = order[first]
    depth = np.zeros((h, w), dtype=np.float32)
    height_img = np.zeros((h, w), dtype=np.float32)
    intensity_img = np.zeros((h, w), dtype=np.float32)
    support = np.zeros((h, w), dtype=bool)
    depth[py[keep], px[keep]] = z[keep].astype(np.float32)
    base = float(np.percentile(height, 5))
    height_img[py[keep], px[keep]] = (height[keep] - base).astype(np.float32)
    intensity_img[py[keep], px[keep]] = intensity[keep].astype(np.float32)
    support[py[keep], px[keep]] = True
    return depth, height_img, intensity_img, support


def accumulate_projected_height_rgb_full(
    cloud_refs: list[MessageRef],
    transform: np.ndarray,
    rgb_shape: tuple[int, int],
    max_range_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = rgb_shape
    depth_acc = np.zeros((h, w), dtype=np.float32)
    height_acc = np.zeros((h, w), dtype=np.float32)
    intensity_acc = np.zeros((h, w), dtype=np.float32)
    support_acc = np.zeros((h, w), dtype=bool)
    for ref in cloud_refs:
        depth, height, intensity, support = projected_depth_height_rgb_full(ref.msg, transform, rgb_shape, max_range_m)
        valid = support.astype(bool)
        if not valid.any():
            continue
        replace_depth = valid & (~support_acc | (depth < depth_acc) | (depth_acc <= 0))
        depth_acc[replace_depth] = depth[replace_depth]
        height_acc[valid] = np.maximum(height_acc[valid], height[valid])
        intensity_acc[valid] = np.maximum(intensity_acc[valid], intensity[valid])
        support_acc |= valid
    return depth_acc, height_acc, intensity_acc, support_acc


def overlay_sparse(rgb_crop: np.ndarray, values: np.ndarray, cmap: int, alpha: float = 0.70, support: np.ndarray | None = None) -> np.ndarray:
    valid = values > 0 if support is None else support.astype(bool)
    color = robust_colorize(values, cmap, valid_mask=valid)
    out = rgb_crop.copy()
    if valid.any():
        dilated = cv2.dilate(valid.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
        color = cv2.dilate(color, np.ones((17, 17), np.uint8))
        out[dilated] = cv2.addWeighted(out[dilated], 1.0 - alpha, color[dilated], alpha, 0)
    return out


def splat_for_display(values: np.ndarray, support: np.ndarray, radius: int = 9) -> tuple[np.ndarray, np.ndarray]:
    kernel = np.ones((radius, radius), np.uint8)
    valid = support.astype(bool)
    if not valid.any():
        return values, valid
    # Max-filtered splat is only for presentation; it makes Ouster scanlines readable.
    splat_values = cv2.dilate(values.astype(np.float32), kernel)
    splat_support = cv2.dilate(valid.astype(np.uint8), kernel).astype(bool)
    return splat_values, splat_support


def interpolate_for_display(values: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = support.astype(bool) & np.isfinite(values) & (values > 0)
    if int(valid.sum()) < 20:
        return splat_for_display(values, support, radius=17)

    h, w = values.shape[:2]
    yy, xx = np.where(valid)
    pts = np.column_stack([xx, yy]).astype(np.float32)
    vals = values[yy, xx].astype(np.float32)
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    try:
        dense = griddata(pts, vals, (grid_x, grid_y), method="cubic")
    except Exception:
        dense = griddata(pts, vals, (grid_x, grid_y), method="linear")

    if dense is None or not np.isfinite(dense).any():
        dense = griddata(pts, vals, (grid_x, grid_y), method="linear")
    nearest = griddata(pts, vals, (grid_x, grid_y), method="nearest")
    dense = np.where(np.isfinite(dense), dense, nearest)

    hull = np.zeros((h, w), dtype=np.uint8)
    if len(pts) >= 3:
        hull_pts = cv2.convexHull(pts.astype(np.int32))
        cv2.fillConvexPoly(hull, hull_pts, 255)
    support_d = cv2.dilate(valid.astype(np.uint8), np.ones((35, 35), np.uint8))
    mask = (hull > 0) & (support_d > 0)
    dense = cv2.GaussianBlur(dense.astype(np.float32), (0, 0), 1.2)
    dense[~mask] = 0.0
    return dense.astype(np.float32), mask


def intersection_quicklook(rgb_crop: np.ndarray, soil_mask: np.ndarray, lidar_support: np.ndarray) -> np.ndarray:
    support = cv2.dilate(lidar_support.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
    plants = soil_mask.any(axis=2)
    both = support & plants
    out = cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2GRAY)
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    out[support] = out[support] * 0.45 + np.array([255, 150, 30], dtype=np.float32) * 0.55
    out[plants] = out[plants] * 0.25 + np.array([30, 255, 80], dtype=np.float32) * 0.75
    out[both] = np.array([255, 255, 255], dtype=np.uint8)
    cv2.putText(out, "white = vegetation + LiDAR support", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, "white = vegetation + LiDAR support", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    return np.clip(out, 0, 255).astype(np.uint8)


def write_contact_sheet(paths: list[Path], labels: list[str], out: Path, tile_w: int = 420) -> None:
    tiles = []
    for p, label in zip(paths, labels):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        scale = tile_w / img.shape[1]
        tile = cv2.resize(img, (tile_w, max(1, int(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 44), (0, 0, 0), -1)
        cv2.putText(tile, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return
    rows = []
    for i in range(0, len(tiles), 3):
        row = tiles[i:i + 3]
        h = max(t.shape[0] for t in row)
        padded = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245)) for t in row]
        rows.append(np.hstack(padded))
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(245, 245, 245)) for r in rows]
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 96])


def write_registered_vis_common(
    bag: Path,
    sample_dir: Path,
    rgb: np.ndarray,
    refs: dict[str, list[MessageRef]],
    target_stamp_ns: int,
    transform: np.ndarray,
    max_sync_ms: float,
    max_range_m: float,
    homographies: dict[str, np.ndarray],
    bottom_shave_frac: float,
    vis_shift_x_px: int,
    ouster_accum_ms: float,
) -> dict[str, str]:
    vis_ref = nearest_by_time(refs["vis"], target_stamp_ns, max_sync_ms)
    if vis_ref is None:
        return {}
    vis_img = decode_ros_image(vis_ref.msg)
    if vis_img is None:
        return {}
    vis = sensor_display_image(vis_img, "vis")
    vis_shape = vis.shape[:2]

    warped: dict[str, np.ndarray] = {"vis": shift_image_x(vis, vis_shift_x_px)}
    cloud_refs: list[MessageRef] = []
    veg_mask_vis: np.ndarray | None = None
    masks: list[np.ndarray] = [np.ones(vis_shape, dtype=np.uint8)]

    rgb_w = warp_to_vis(to_bgr_for_display(rgb, "rgb"), homographies["rgb"], vis_shape)
    rgb_mask = warp_to_vis(np.full(rgb.shape[:2], 255, np.uint8), homographies["rgb"], vis_shape, cv2.INTER_NEAREST) > 0
    warped["rgb"] = rgb_w
    masks.append(rgb_mask.astype(np.uint8))

    nir_ref = nearest_by_time(refs["nir"], target_stamp_ns, max_sync_ms)
    if nir_ref is not None:
        nir_img = decode_ros_image(nir_ref.msg)
        if nir_img is not None:
            nir = sensor_display_image(nir_img, "nir")
            warped["nir"] = warp_to_vis(nir, homographies["nir"], vis_shape)
            masks.append(warp_to_vis(np.full(nir.shape[:2], 255, np.uint8), homographies["nir"], vis_shape, cv2.INTER_NEAREST) > 0)

    thermal_ref = nearest_by_time(refs["thermal"], target_stamp_ns, max_sync_ms)
    if thermal_ref is not None:
        thermal_img = decode_ros_image(thermal_ref.msg)
        if thermal_img is not None:
            thermal = colorize_with_scale(thermal_img.astype(np.float32), cv2.COLORMAP_INFERNO, "Thermal C", "C")
            # Remove scale bar before geometric warp; keep clean image for common crop.
            thermal_clean = thermal[:, :thermal_img.shape[1]]
            warped["thermal_c"] = warp_to_vis(thermal_clean, homographies["thermal"], vis_shape)
            masks.append(warp_to_vis(np.full(thermal_img.shape[:2], 255, np.uint8), homographies["thermal"], vis_shape, cv2.INTER_NEAREST) > 0)

    cover, soil_mask, _gli_vis, _cover_metrics = soil_cover(rgb)
    warped["gli_plants"] = warp_to_vis(soil_mask, homographies["rgb"], vis_shape, cv2.INTER_NEAREST)
    veg_mask_vis = warped["gli_plants"].any(axis=2)

    cloud_refs = collect_cloud_refs_near(bag, target_stamp_ns, ouster_accum_ms) if ouster_accum_ms > 0 else []
    if not cloud_refs:
        cloud_ref = nearest_by_time(refs["cloud"], target_stamp_ns, max_sync_ms)
        cloud_refs = [cloud_ref] if cloud_ref is not None else []

    box = common_valid_box([np.asarray(m, dtype=np.uint8) for m in masks], bottom_shave_frac=bottom_shave_frac)
    if box is None:
        return {}

    out_dir = sample_dir / "registered_vis_common"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for name, img in warped.items():
        cropped = crop_box(img, box)
        if bottom_shave_frac > 0:
            cropped = cropped[:max(1, int(cropped.shape[0] * (1.0 - bottom_shave_frac))), :]
        path = out_dir / f"{name}.jpg"
        cv2.imwrite(str(path), cropped, [cv2.IMWRITE_JPEG_QUALITY, 96])
        outputs[name] = str(path)

    if cloud_refs and veg_mask_vis is not None:
        ouster_panel = render_ouster_intensity_soft_veg(
            cloud_refs=cloud_refs,
            transform=transform,
            rgb_shape=rgb.shape[:2],
            homographies=homographies,
            vis_shape=vis_shape,
            box=box,
            bottom_shave_frac=bottom_shave_frac,
            veg_mask_vis=veg_mask_vis,
            max_range_m=max_range_m,
        )
        path = out_dir / "ouster_intensity.jpg"
        if ouster_panel is not None:
            cv2.imwrite(str(path), ouster_panel, [cv2.IMWRITE_JPEG_QUALITY, 96])
            outputs["ouster_intensity"] = str(path)
        outputs["ouster_clouds_accumulated"] = str(len(cloud_refs))

        legacy_path = out_dir / "ouster_height.jpg"
        if ouster_panel is not None:
            cv2.imwrite(str(legacy_path), ouster_panel, [cv2.IMWRITE_JPEG_QUALITY, 96])
            outputs["ouster_height"] = str(legacy_path)

    preferred = [
        ("rgb", "RGB"),
        ("vis", "VIS"),
        ("nir", "NIR"),
        ("thermal_c", "Thermal"),
        ("gli_plants", "Soil cover"),
        ("ouster_intensity", "Ouster intensity"),
    ]
    sheet_paths = [Path(outputs[k]) for k, _ in preferred if k in outputs]
    labels = [label for k, label in preferred if k in outputs]
    write_contact_sheet(sheet_paths, labels, out_dir / "registered_vis_common_sheet.jpg", tile_w=420)
    outputs["sheet"] = str(out_dir / "registered_vis_common_sheet.jpg")
    outputs["crop_box_vis_xyxy"] = json.dumps([int(v) for v in box])
    return outputs


def process_sample(
    bag: Path,
    refs: dict[str, list[MessageRef]],
    target: MessageRef,
    plots: gpd.GeoDataFrame,
    transform: np.ndarray,
    out_dir: Path,
    crop: tuple[float, float, float, float],
    max_sync_ms: float,
    max_range_m: float,
    register_to_vis: bool,
    homographies: dict[str, np.ndarray] | None,
    vis_bottom_shave_frac: float,
    vis_shift_x_px: int,
    ouster_accum_ms: float,
) -> dict:
    sample_dir = out_dir
    sample_dir.mkdir(parents=True, exist_ok=True)
    rgb = decode_ros_image(target.msg)
    if rgb is None:
        raise RuntimeError("Could not decode RGB sample")
    rgb_crop = crop_fraction(rgb, crop)
    cv2.imwrite(str(sample_dir / "rgb_crop.jpg"), rgb_crop, [cv2.IMWRITE_JPEG_QUALITY, 96])

    gps_ref = nearest_by_time(refs["gps"], target.stamp_ns, 2500.0)
    lon = getattr(gps_ref.msg, "longitude", None) if gps_ref else None
    lat = getattr(gps_ref.msg, "latitude", None) if gps_ref else None
    plot = assign_plot(plots, lon, lat)

    outputs = {"rgb": "rgb_crop.jpg"}
    for key in ["vis", "nir", "thermal", "thermal_raw", "ouster_intensity", "ouster_range"]:
        ref = nearest_by_time(refs[key], target.stamp_ns, max_sync_ms)
        if ref is None:
            continue
        img = decode_ros_image(ref.msg)
        if img is None:
            continue
        if key.startswith("thermal") and img.ndim == 2:
            cropped_num = crop_fraction(img.astype(np.float32), crop)
            unit = "C" if key == "thermal" else "raw"
            label = "Thermal C" if key == "thermal" else "Thermal Raw"
            cropped = colorize_with_scale(cropped_num, cv2.COLORMAP_INFERNO, label, unit)
        else:
            bgr = sensor_display_image(img, key) if key in ("vis", "nir") else to_bgr_for_display(img, key)
            cropped = crop_fraction(bgr, crop)
        name = f"{key}_crop.jpg"
        cv2.imwrite(str(sample_dir / name), cropped, [cv2.IMWRITE_JPEG_QUALITY, 96])
        outputs[key] = name

    cover, soil_mask, gli_vis, cover_metrics = soil_cover(rgb_crop)
    cv2.imwrite(str(sample_dir / "soil_cover_quicklook.jpg"), cover, [cv2.IMWRITE_JPEG_QUALITY, 96])
    cv2.imwrite(str(sample_dir / "soil_cover_binary_mask.jpg"), soil_mask, [cv2.IMWRITE_JPEG_QUALITY, 96])
    cv2.imwrite(str(sample_dir / "gli_index_scaled.jpg"), gli_vis, [cv2.IMWRITE_JPEG_QUALITY, 96])
    outputs["soil_cover"] = "soil_cover_quicklook.jpg"
    outputs["soil_mask"] = "soil_cover_binary_mask.jpg"
    outputs["gli"] = "gli_index_scaled.jpg"

    cloud_ref = nearest_by_time(refs["cloud"], target.stamp_ns, max_sync_ms)
    if cloud_ref is not None:
        depth, height, support = projected_depth_height_rgb(cloud_ref.msg, transform, rgb.shape[:2], crop, max_range_m)
        depth_vis, support_vis = splat_for_display(depth, support, radius=15)
        height_vis, _ = splat_for_display(height, support, radius=15)
        depth_interp, depth_interp_support = interpolate_for_display(depth, support)
        height_interp, height_interp_support = interpolate_for_display(height, support)
        cv2.imwrite(str(sample_dir / "depth_rgb_crop.jpg"), colorize_with_scale(depth_vis, cv2.COLORMAP_TURBO, "Depth", "m", support_vis), [cv2.IMWRITE_JPEG_QUALITY, 96])
        cv2.imwrite(str(sample_dir / "height_rgb_crop.jpg"), colorize_with_scale(height_vis, cv2.COLORMAP_MAGMA, "Height", "m", support_vis), [cv2.IMWRITE_JPEG_QUALITY, 96])
        cv2.imwrite(str(sample_dir / "depth_interpolated_crop.jpg"), colorize_with_scale(depth_interp, cv2.COLORMAP_TURBO, "Depth Interp.", "m", depth_interp_support), [cv2.IMWRITE_JPEG_QUALITY, 96])
        cv2.imwrite(str(sample_dir / "height_interpolated_crop.jpg"), colorize_with_scale(height_interp, cv2.COLORMAP_MAGMA, "Height Interp.", "m", height_interp_support), [cv2.IMWRITE_JPEG_QUALITY, 96])
        cv2.imwrite(str(sample_dir / "depth_overlay_crop.jpg"), overlay_sparse(rgb_crop, depth, cv2.COLORMAP_TURBO, support=support), [cv2.IMWRITE_JPEG_QUALITY, 96])
        cv2.imwrite(str(sample_dir / "height_overlay_crop.jpg"), overlay_sparse(rgb_crop, height, cv2.COLORMAP_MAGMA, support=support), [cv2.IMWRITE_JPEG_QUALITY, 96])
        cv2.imwrite(str(sample_dir / "intersection_rgb_soil_lidar.jpg"), intersection_quicklook(rgb_crop, soil_mask, support_vis), [cv2.IMWRITE_JPEG_QUALITY, 96])
        outputs.update({
            "depth_rgb": "depth_rgb_crop.jpg",
            "height_rgb": "height_rgb_crop.jpg",
            "depth_interp": "depth_interpolated_crop.jpg",
            "height_interp": "height_interpolated_crop.jpg",
            "depth_overlay": "depth_overlay_crop.jpg",
            "height_overlay": "height_overlay_crop.jpg",
            "intersection": "intersection_rgb_soil_lidar.jpg",
        })

    registered_outputs = {}
    if register_to_vis and homographies is not None:
        registered_outputs = write_registered_vis_common(
            bag=bag,
            sample_dir=sample_dir,
            rgb=rgb,
            refs=refs,
            target_stamp_ns=target.stamp_ns,
            transform=transform,
            max_sync_ms=max_sync_ms,
            max_range_m=max_range_m,
            homographies=homographies,
            bottom_shave_frac=vis_bottom_shave_frac,
            vis_shift_x_px=vis_shift_x_px,
            ouster_accum_ms=ouster_accum_ms,
        )

    sheet_paths = [sample_dir / outputs[k] for k in outputs if k in {
        "rgb", "vis", "nir", "thermal",
        "soil_cover", "soil_mask", "depth_interp", "height_interp", "height_overlay", "intersection",
    }]
    labels = [p.stem.replace("_crop", "").replace("_", " ").title() for p in sheet_paths]
    write_contact_sheet(sheet_paths, labels, sample_dir / "contact_sheet.jpg")

    return {
        "bag": str(bag),
        "target_stamp_ns": int(target.stamp_ns),
        "gps": {"latitude": lat, "longitude": lon},
        "plot": plot,
        "soil_cover": cover_metrics,
        "outputs": {k: str(sample_dir / v) for k, v in outputs.items()},
        "registered_vis_common": registered_outputs,
        "contact_sheet": str(sample_dir / "contact_sheet.jpg"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bags", nargs="+", type=Path, required=True)
    ap.add_argument("--gpkg", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--transform", type=Path, default=Path("runs/calibration/multipose_6dof_refined_20260528/T_cam_lidar_multipose_6dof.npy"))
    ap.add_argument("--samples-per-bag", type=int, default=2)
    ap.add_argument("--max-sync-ms", type=float, default=700.0)
    ap.add_argument("--max-range-m", type=float, default=8.0)
    ap.add_argument("--crop", nargs=4, type=float, default=[0.16, 0.24, 0.90, 0.84])
    ap.add_argument("--sample-from-rgb-index", action="store_true",
                    help="Scan RGB timestamps before sampling. Slower on very large bags.")
    ap.add_argument("--register-to-vis", action="store_true",
                    help="Warp RGB/NIR/Thermal and RGB-derived maps to VIS, then crop common valid intersection.")
    ap.add_argument("--homographies", type=Path, default=DEFAULT_HOMOGRAPHIES)
    ap.add_argument("--vis-bottom-shave-frac", type=float, default=0.10,
                    help="Drop this fraction from the bottom of the VIS common crop to hide robot panel/reflection.")
    ap.add_argument("--vis-shift-x-px", type=int, default=0,
                    help="Presentation-only horizontal shift for the VIS panel after preview extraction.")
    ap.add_argument("--ouster-accum-ms", type=float, default=0.0,
                    help="For registered VIS sheets, accumulate PointCloud2 frames within +/- this time window.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    plots = load_plots(args.gpkg)
    transform = np.load(args.transform) if args.transform.suffix.lower() == ".npy" else np.loadtxt(args.transform)
    homographies = load_homographies(args.homographies) if args.register_to_vis else None
    all_summaries = []
    overview_paths = []
    overview_labels = []

    for bag in args.bags:
        if args.sample_from_rgb_index:
            rgb_times = collect_rgb_times(bag)
            target_times = choose_target_times(rgb_times, args.samples_per_bag)
        else:
            start_ns, end_ns = bag_time_bounds(bag)
            target_times = choose_target_times_from_bounds(start_ns, end_ns, args.samples_per_bag)
        target_ref_sets = collect_nearest_refs_for_targets(bag, target_times, args.max_sync_ms)
        for i, refs_by_key in enumerate(target_ref_sets):
            target = refs_by_key.get("rgb")
            if target is None:
                continue
            refs = {k: ([v] if v is not None else []) for k, v in refs_by_key.items()}
            sample_name = f"{bag.stem}_s{i:02d}"
            summary = process_sample(
                bag=bag,
                refs=refs,
                target=target,
                plots=plots,
                transform=transform,
                out_dir=args.out / sample_name,
                crop=tuple(args.crop),
                max_sync_ms=args.max_sync_ms,
                max_range_m=args.max_range_m,
                register_to_vis=args.register_to_vis,
                homographies=homographies,
                vis_bottom_shave_frac=args.vis_bottom_shave_frac,
                vis_shift_x_px=args.vis_shift_x_px,
                ouster_accum_ms=args.ouster_accum_ms,
            )
            all_summaries.append(summary)
            overview_paths.append(Path(summary["contact_sheet"]))
            plot_id = summary["plot"].get("plot_id")
            variant = summary["plot"].get("variante", "")
            overview_labels.append(f"{bag.stem} plot {plot_id} {variant}")

    write_contact_sheet(overview_paths, overview_labels, args.out / "presentation_overview.jpg", tile_w=520)
    (args.out / "presentation_gallery_summary.json").write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "samples": len(all_summaries), "overview": str(args.out / "presentation_overview.jpg")}, indent=2))


if __name__ == "__main__":
    main()
