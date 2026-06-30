#!/usr/bin/env python3
"""Extract organized multisensor images and metadata from ROS1 bags.

This is the broad preprocessing entry point for field data. It writes previews,
optional raw arrays, a frame CSV, and optional GeoJSON points with nearest GNSS.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

try:
    import geopandas as gpd
    from shapely.geometry import Point
except Exception:  # pragma: no cover - optional for dry/light extraction.
    gpd = None
    Point = None


TOPICS = {
    "rgb": "/ssf/BFS_usb_0/image_raw",
    "vis": "/ssf/photonfocus_camera_vis_node/image_raw",
    "nir": "/ssf/photonfocus_camera_nir_node/image_raw",
    "thermal_c": "/ssf/thermalgrabber_ros/image_deg_celsius",
    "thermal_raw": "/ssf/thermalgrabber_ros/image_mono16",
    "ouster_intensity": "/ssf/img_node/intensity_image",
    "ouster_range": "/ssf/img_node/range_image",
    "ouster_noise": "/ssf/img_node/noise_image",
}

GPS_TOPICS = {
    "gps_fix": "/ssf/gnss/fix",
}


@dataclass
class GpsFix:
    stamp_ns: int
    lat: float
    lon: float
    alt: float | None


def stamp_to_iso(stamp_ns: int) -> str:
    return datetime.fromtimestamp(stamp_ns / 1e9, tz=timezone.utc).isoformat()


def safe_topic_name(topic: str) -> str:
    return topic.strip("/").replace("/", "__") or "topic"


def find_bags(paths: list[Path]) -> list[Path]:
    bags: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".bag":
            bags.append(path)
        elif path.is_dir():
            bags.extend(sorted(path.rglob("*.bag")))
    return sorted(dict.fromkeys(bags))


def decode_ros_image(msg: Any) -> np.ndarray | None:
    enc = str(msg.encoding).lower().strip()
    h, w = int(msg.height), int(msg.width)
    raw = bytes(msg.data)
    if enc == "rgb8":
        return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)[:, :, ::-1].copy()
    if enc == "bgr8":
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
    if enc == "32fc1":
        return np.frombuffer(raw, dtype=np.float32).reshape(h, w).copy()
    if enc == "64fc1":
        return np.frombuffer(raw, dtype=np.float64).reshape(h, w).astype(np.float32)
    return None


def photonfocus_preview(image: np.ndarray, pattern: int = 4) -> np.ndarray:
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
            bands.append(cv2.medianBlur(band, 3).astype(np.float32))
    return np.mean(np.stack(bands, axis=-1), axis=-1).astype(image.dtype)


def robust_preview(img: np.ndarray, key: str) -> np.ndarray:
    if key in ("vis", "nir") and img.ndim == 2 and img.shape[0] > 300:
        img = photonfocus_preview(img)
    if img.ndim == 3:
        return img
    src = img.astype(np.float32)
    finite = np.isfinite(src)
    if not finite.any():
        norm = np.zeros(src.shape, dtype=np.uint8)
    else:
        lo, hi = np.percentile(src[finite], [1, 99])
        if hi <= lo:
            hi = lo + 1.0
        norm = np.clip((src - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    cmap = cv2.COLORMAP_INFERNO if key.startswith("thermal") else cv2.COLORMAP_VIRIDIS
    if key in ("ouster_range",):
        norm = 255 - norm
    return cv2.applyColorMap(norm, cmap)


def nearest_gps(gps: list[GpsFix], stamp_ns: int, max_dt_ms: float) -> tuple[GpsFix | None, float | None]:
    if not gps:
        return None, None
    best = min(gps, key=lambda g: abs(g.stamp_ns - stamp_ns))
    dt_ms = abs(best.stamp_ns - stamp_ns) / 1e6
    if dt_ms > max_dt_ms:
        return None, dt_ms
    return best, dt_ms


def load_plots(gpkg: Path | None):
    if gpkg is None:
        return None
    if gpd is None or Point is None:
        raise RuntimeError("geopandas/shapely are required when --gpkg is used")
    plots = gpd.read_file(gpkg)
    if plots.crs is None:
        plots = plots.set_crs("EPSG:4326")
    return plots.to_crs("EPSG:4326")


def assign_plot(plots, lon: float | None, lat: float | None) -> dict[str, str]:
    if plots is None or lon is None or lat is None or Point is None:
        return {}
    hits = plots[plots.geometry.contains(Point(float(lon), float(lat)))]
    if hits.empty:
        return {}
    row = hits.iloc[0]
    out = {"plot_index": str(row.name)}
    for key in ("plot_id", "Plot", "plot", "treatment", "variant", "Variante", "Sorte"):
        if key in row and row[key] is not None:
            out[key] = str(row[key])
    return out


def collect_gps(reader: Reader, typestore, gps_topic: str) -> list[GpsFix]:
    conns = [c for c in reader.connections if c.topic == gps_topic]
    fixes: list[GpsFix] = []
    for conn, ts, raw in reader.messages(connections=conns):
        msg = typestore.deserialize_ros1(raw, conn.msgtype)
        lat = float(getattr(msg, "latitude", np.nan))
        lon = float(getattr(msg, "longitude", np.nan))
        alt = float(getattr(msg, "altitude", np.nan))
        if np.isfinite(lat) and np.isfinite(lon):
            fixes.append(GpsFix(int(ts), lat, lon, alt if np.isfinite(alt) else None))
    return fixes


def write_geojson(records: list[dict[str, Any]], path: Path) -> None:
    features = []
    for rec in records:
        lat = rec.get("gps_lat")
        lon = rec.get("gps_lon")
        if lat in ("", None) or lon in ("", None):
            continue
        props = {k: v for k, v in rec.items() if k not in ("gps_lat", "gps_lon")}
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                "properties": props,
            }
        )
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8")


def process_bag(
    bag: Path,
    out: Path,
    topics: dict[str, str],
    plots,
    every_n: int,
    max_gps_dt_ms: float,
    write_raw_npy: bool,
    dry_run: bool,
) -> list[dict[str, Any]]:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    records: list[dict[str, Any]] = []
    bag_id = bag.stem
    with Reader(bag) as reader:
        available = {c.topic: c for c in reader.connections}
        selected = [available[t] for t in topics.values() if t in available]
        gps = collect_gps(reader, typestore, GPS_TOPICS["gps_fix"]) if GPS_TOPICS["gps_fix"] in available else []
        if dry_run:
            print(f"{bag}: {len(selected)} image topics, {len(gps)} gps fixes")
            return []

        counters = {key: 0 for key in topics}
        topic_to_key = {topic: key for key, topic in topics.items()}
        for conn, ts, raw in reader.messages(connections=selected):
            key = topic_to_key[conn.topic]
            idx = counters[key]
            counters[key] += 1
            if every_n > 1 and idx % every_n != 0:
                continue
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            img = decode_ros_image(msg)
            if img is None:
                continue

            gps_fix, gps_dt_ms = nearest_gps(gps, int(ts), max_gps_dt_ms)
            plot_meta = assign_plot(plots, gps_fix.lon if gps_fix else None, gps_fix.lat if gps_fix else None)
            rel_base = Path("images") / bag_id / key / f"{idx:06d}_{int(ts)}.png"
            image_path = out / rel_base
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), robust_preview(img, key))

            raw_rel = ""
            if write_raw_npy:
                raw_rel_path = Path("raw") / bag_id / key / f"{idx:06d}_{int(ts)}.npy"
                raw_path = out / raw_rel_path
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(raw_path, img)
                raw_rel = raw_rel_path.as_posix()

            record: dict[str, Any] = {
                "bag": str(bag),
                "bag_id": bag_id,
                "sensor": key,
                "topic": conn.topic,
                "frame_index": idx,
                "stamp_ns": int(ts),
                "iso_utc": stamp_to_iso(int(ts)),
                "encoding": str(getattr(msg, "encoding", "")),
                "width": int(getattr(msg, "width", img.shape[1])),
                "height": int(getattr(msg, "height", img.shape[0])),
                "image_path": rel_base.as_posix(),
                "raw_npy_path": raw_rel,
                "gps_lat": gps_fix.lat if gps_fix else "",
                "gps_lon": gps_fix.lon if gps_fix else "",
                "gps_alt": gps_fix.alt if gps_fix and gps_fix.alt is not None else "",
                "gps_dt_ms": gps_dt_ms if gps_dt_ms is not None else "",
            }
            record.update(plot_meta)
            records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag-root", action="append", type=Path, required=True, help="Bag file or directory. Repeatable.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpkg", type=Path)
    parser.add_argument("--topics", nargs="*", choices=sorted(TOPICS), default=sorted(TOPICS))
    parser.add_argument("--limit-bags", type=int, default=0, help="Process only the first N bags. Useful for smoke tests.")
    parser.add_argument("--every-n", type=int, default=1)
    parser.add_argument("--max-gps-dt-ms", type=float, default=1000.0)
    parser.add_argument("--write-raw-npy", action="store_true")
    parser.add_argument("--write-geojson", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bags = find_bags(args.bag_root)
    if args.limit_bags > 0:
        bags = bags[: args.limit_bags]
    selected_topics = {key: TOPICS[key] for key in args.topics}
    print(f"Found {len(bags)} bag(s). Sensors: {', '.join(selected_topics)}")
    plots = load_plots(args.gpkg)
    all_records: list[dict[str, Any]] = []
    for bag in bags:
        all_records.extend(
            process_bag(
                bag=bag,
                out=args.out,
                topics=selected_topics,
                plots=plots,
                every_n=max(1, args.every_n),
                max_gps_dt_ms=args.max_gps_dt_ms,
                write_raw_npy=args.write_raw_npy,
                dry_run=args.dry_run,
            )
        )
    if args.dry_run:
        return

    metadata_dir = args.out / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metadata_dir / "frames.csv"
    fieldnames = sorted({k for rec in all_records for k in rec.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    if args.write_geojson:
        write_geojson(all_records, metadata_dir / "frames.geojson")
    print(f"Wrote {len(all_records)} image records to {csv_path}")


if __name__ == "__main__":
    main()
