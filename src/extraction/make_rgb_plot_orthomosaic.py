#!/usr/bin/env python3
"""Prototype RGB plot orthomosaic from a ROS1 bag using GNSS selection + visual stitching."""

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
from shapely.geometry import Point

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction.make_plot_presentation_gallery import decode_ros_image, to_bgr_for_display  # noqa: E402

RGB_TOPIC = "/ssf/BFS_usb_0/image_raw"
GPS_TOPIC = "/ssf/gnss/fix"
VEL_TOPIC = "/ssf/gnss/vel"


@dataclass
class Ref:
    stamp_ns: int
    msg: object


@dataclass
class Frame:
    stamp_ns: int
    image: np.ndarray
    lat: float | None
    lon: float | None
    speed_mps: float | None
    plot_id: str | None


def load_plots(gpkg: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(gpkg)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")
    gdf["plot_id"] = [str(i + 1) for i in range(len(gdf))]
    return gdf


def plot_for_gps(gdf: gpd.GeoDataFrame, lon: float | None, lat: float | None) -> str | None:
    if lon is None or lat is None or not np.isfinite([lon, lat]).all():
        return None
    pt = Point(float(lon), float(lat))
    hits = gdf[gdf.geometry.contains(pt)]
    if not hits.empty:
        return str(hits.iloc[0]["plot_id"])
    return None


def nearest(items: list[Ref], stamp_ns: int, max_dt_ms: float) -> Ref | None:
    if not items:
        return None
    best = min(items, key=lambda r: abs(r.stamp_ns - stamp_ns))
    if abs(best.stamp_ns - stamp_ns) > max_dt_ms * 1_000_000:
        return None
    return best


def collect_refs(bag: Path, center_ns: int | None = None, window_ms: float | None = None) -> tuple[list[Ref], list[Ref], list[Ref]]:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    rgb: list[Ref] = []
    gps: list[Ref] = []
    vel: list[Ref] = []
    start = stop = None
    if center_ns is not None and window_ms is not None and window_ms > 0:
        half = int(window_ms * 1_000_000)
        start = int(center_ns - half)
        stop = int(center_ns + half)
    with Reader(bag) as reader:
        wanted = {RGB_TOPIC, GPS_TOPIC, VEL_TOPIC}
        conns = [c for c in reader.connections if c.topic in wanted]
        kwargs = {}
        if start is not None and stop is not None:
            kwargs = {"start": start, "stop": stop}
        for conn, ts, raw in reader.messages(connections=conns, **kwargs):
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            if conn.topic == RGB_TOPIC:
                rgb.append(Ref(int(ts), msg))
            elif conn.topic == GPS_TOPIC:
                gps.append(Ref(int(ts), msg))
            elif conn.topic == VEL_TOPIC:
                vel.append(Ref(int(ts), msg))
    return rgb, gps, vel


def velocity_norm(msg: object) -> float | None:
    tw = getattr(msg, "twist", None)
    lin = getattr(tw, "linear", None) if tw is not None else None
    if lin is None:
        return None
    x = float(getattr(lin, "x", 0.0))
    y = float(getattr(lin, "y", 0.0))
    z = float(getattr(lin, "z", 0.0))
    return float(math.sqrt(x * x + y * y + z * z))


def crop_fraction(img: np.ndarray, crop: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = int(w * crop[0]), int(h * crop[1]), int(w * crop[2]), int(h * crop[3])
    return img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)].copy()


def collect_plot_frames(
    bag: Path,
    plots: gpd.GeoDataFrame,
    plot_id: str,
    crop: tuple[float, float, float, float],
    max_sync_ms: float,
    center_ns: int | None,
    window_ms: float | None,
) -> list[Frame]:
    rgb_refs, gps_refs, vel_refs = collect_refs(bag, center_ns=center_ns, window_ms=window_ms)
    frames: list[Frame] = []
    for ref in rgb_refs:
        gps = nearest(gps_refs, ref.stamp_ns, max_sync_ms)
        vel = nearest(vel_refs, ref.stamp_ns, max_sync_ms)
        lat = float(getattr(gps.msg, "latitude", math.nan)) if gps else None
        lon = float(getattr(gps.msg, "longitude", math.nan)) if gps else None
        pid = plot_for_gps(plots, lon, lat)
        if pid != plot_id:
            continue
        img = decode_ros_image(ref.msg)
        if img is None:
            continue
        img = crop_fraction(to_bgr_for_display(img, "rgb"), crop)
        frames.append(Frame(ref.stamp_ns, img, lat, lon, velocity_norm(vel.msg) if vel else None, pid))
    return frames


def select_frames(frames: list[Frame], count: int, stride: int) -> list[Frame]:
    if not frames:
        return []
    if stride > 1:
        mid = len(frames) // 2
        half = (count // 2) * stride
        start = max(0, mid - half)
        idx = [start + i * stride for i in range(count) if start + i * stride < len(frames)]
        return [frames[i] for i in idx]
    if len(frames) <= count:
        return frames
    # Choose a compact window around the middle to preserve overlap.
    start = max(0, len(frames) // 2 - count // 2)
    return frames[start:start + count]


def prep_gray(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return gray


def estimate_homography(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray | None, int]:
    g1 = prep_gray(src)
    g2 = prep_gray(dst)
    try:
        detector = cv2.SIFT_create(nfeatures=3500)
        norm = cv2.NORM_L2
    except Exception:
        detector = cv2.ORB_create(nfeatures=5000)
        norm = cv2.NORM_HAMMING
    k1, d1 = detector.detectAndCompute(g1, None)
    k2, d2 = detector.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12:
        return None, 0
    matcher = cv2.BFMatcher(norm)
    pairs = matcher.knnMatch(d1, d2, k=2)
    good = []
    for a, b in pairs:
        if a.distance < 0.72 * b.distance:
            good.append(a)
    if len(good) < 10:
        return None, len(good)
    src_pts = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    h, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
    inliers = int(mask.sum()) if mask is not None else 0
    if h is None or inliers < 8:
        return None, inliers
    return h.astype(np.float64), inliers


def stitch(frames: list[Frame]) -> tuple[np.ndarray, list[dict]]:
    if not frames:
        raise ValueError("No frames to stitch")
    h0, w0 = frames[0].image.shape[:2]
    transforms = [np.eye(3, dtype=np.float64)]
    diagnostics = []
    for i in range(1, len(frames)):
        h_prev, inliers = estimate_homography(frames[i].image, frames[i - 1].image)
        if h_prev is None:
            # Fallback: small vertical advance in image coordinates.
            h_prev = np.array([[1, 0, 0], [0, 1, -0.18 * h0], [0, 0, 1]], dtype=np.float64)
            diagnostics.append({"pair": [i - 1, i], "method": "fallback_translation", "inliers": inliers})
        else:
            diagnostics.append({"pair": [i - 1, i], "method": "features_homography", "inliers": inliers})
        transforms.append(transforms[-1] @ h_prev)

    corners = np.float32([[0, 0], [w0, 0], [w0, h0], [0, h0]]).reshape(-1, 1, 2)
    all_corners = [cv2.perspectiveTransform(corners, h) for h in transforms]
    pts = np.vstack(all_corners).reshape(-1, 2)
    min_xy = np.floor(pts.min(axis=0)).astype(int)
    max_xy = np.ceil(pts.max(axis=0)).astype(int)
    pad = 80
    tx, ty = -min_xy[0] + pad, -min_xy[1] + pad
    out_w = int(max_xy[0] - min_xy[0] + 2 * pad)
    out_h = int(max_xy[1] - min_xy[1] + 2 * pad)
    translate = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)

    acc = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weight = np.zeros((out_h, out_w, 1), dtype=np.float32)
    for frame, h in zip(frames, transforms):
        warp_h = translate @ h
        warped = cv2.warpPerspective(frame.image, warp_h, (out_w, out_h), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.ones(frame.image.shape[:2], dtype=np.uint8) * 255, warp_h, (out_w, out_h), flags=cv2.INTER_NEAREST) > 0
        feather = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 7.0)[:, :, None]
        acc += warped.astype(np.float32) * feather
        weight += feather
    mosaic = np.divide(acc, np.maximum(weight, 1e-3)).astype(np.uint8)
    valid = weight[:, :, 0] > 0.05
    ys, xs = np.where(valid)
    if len(xs):
        mosaic = mosaic[max(0, ys.min()):ys.max() + 1, max(0, xs.min()):xs.max() + 1]
    return mosaic, diagnostics


def write_contact(frames: list[Frame], mosaic: np.ndarray, out: Path) -> None:
    thumbs = []
    for i, frame in enumerate(frames):
        img = frame.image.copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 52), (0, 0, 0), -1)
        cv2.putText(img, f"RGB frame {i}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        thumbs.append(cv2.resize(img, (360, max(1, int(img.shape[0] * 360 / img.shape[1]))), interpolation=cv2.INTER_AREA))
    mos = mosaic.copy()
    cv2.rectangle(mos, (0, 0), (mos.shape[1], 58), (0, 0, 0), -1)
    cv2.putText(mos, "RGB local orthomosaic prototype", (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    mos = cv2.resize(mos, (min(1440, mos.shape[1]), max(1, int(mos.shape[0] * min(1440, mos.shape[1]) / mos.shape[1]))), interpolation=cv2.INTER_AREA)
    row_h = max(t.shape[0] for t in thumbs)
    thumbs = [cv2.copyMakeBorder(t, 0, row_h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245)) for t in thumbs]
    top = np.hstack(thumbs)
    width = max(top.shape[1], mos.shape[1])
    top = cv2.copyMakeBorder(top, 0, 0, 0, width - top.shape[1], cv2.BORDER_CONSTANT, value=(245, 245, 245))
    mos = cv2.copyMakeBorder(mos, 0, 0, 0, width - mos.shape[1], cv2.BORDER_CONSTANT, value=(245, 245, 245))
    cv2.imwrite(str(out), np.vstack([top, np.full((24, width, 3), 245, np.uint8), mos]), [cv2.IMWRITE_JPEG_QUALITY, 96])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bag", type=Path, required=True)
    ap.add_argument("--gpkg", type=Path, required=True)
    ap.add_argument("--plot-id", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-sync-ms", type=float, default=1500.0)
    ap.add_argument("--center-ns", type=int, default=None,
                    help="Optional center timestamp; when set, only this time window is read.")
    ap.add_argument("--window-ms", type=float, default=None,
                    help="Half-window around --center-ns to read from the bag.")
    ap.add_argument("--crop", nargs=4, type=float, default=[0.16, 0.24, 0.90, 0.84])
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    plots = load_plots(args.gpkg)
    candidates = collect_plot_frames(
        args.bag,
        plots,
        str(args.plot_id),
        tuple(args.crop),
        args.max_sync_ms,
        args.center_ns,
        args.window_ms,
    )
    selected = select_frames(candidates, args.frames, args.stride)
    if len(selected) < 2:
        raise RuntimeError(f"Need at least 2 frames in plot {args.plot_id}; found {len(selected)}")
    for i, frame in enumerate(selected):
        cv2.imwrite(str(args.out / f"rgb_frame_{i:02d}.jpg"), frame.image, [cv2.IMWRITE_JPEG_QUALITY, 96])
    mosaic, diagnostics = stitch(selected)
    cv2.imwrite(str(args.out / "rgb_orthomosaic.jpg"), mosaic, [cv2.IMWRITE_JPEG_QUALITY, 96])
    write_contact(selected, mosaic, args.out / "rgb_orthomosaic_contact.jpg")
    meta = {
        "bag": str(args.bag),
        "plot_id": str(args.plot_id),
        "candidate_frames_in_plot": len(candidates),
        "selected_frames": len(selected),
        "timestamps_ns": [int(f.stamp_ns) for f in selected],
        "speed_mps": [f.speed_mps for f in selected],
        "lat_lon": [[f.lat, f.lon] for f in selected],
        "diagnostics": diagnostics,
        "outputs": {
            "mosaic": str(args.out / "rgb_orthomosaic.jpg"),
            "contact": str(args.out / "rgb_orthomosaic_contact.jpg"),
        },
    }
    (args.out / "rgb_orthomosaic_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
