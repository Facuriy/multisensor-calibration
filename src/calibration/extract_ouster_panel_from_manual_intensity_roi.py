#!/usr/bin/env python3
"""Extract Ouster panel point candidates from manual intensity search windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

CLOUD_TOPIC = "/ssf/os1_cloud_node/points"


def stamp_ns(msg, fallback: int) -> int:
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None:
        return int(fallback)
    sec = getattr(stamp, "sec", None)
    nsec = getattr(stamp, "nanosec", getattr(stamp, "nsec", None))
    return int(fallback) if sec is None or nsec is None else int(sec) * 1_000_000_000 + int(nsec)


def cloud_to_organized(msg) -> dict[str, np.ndarray]:
    offsets = {field.name: int(field.offset) for field in msg.fields}
    step = int(msg.point_step)
    h = int(msg.height)
    w = int(msg.width)
    count = h * w
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    dtypes = {
        "x": "<f4",
        "y": "<f4",
        "z": "<f4",
        "intensity": "<f4",
        "range": "<u4",
        "reflectivity": "<u2",
        "noise": "<u2",
    }
    out = {}
    for name, dtype in dtypes.items():
        if name in offsets:
            arr = np.ndarray((count,), dtype=np.dtype(dtype), buffer=raw, offset=offsets[name], strides=(step,)).copy()
            # Ouster ROS stores this cloud with azimuth-major linear order:
            # each contiguous block contains all 64 rings for one column.
            # PointCloud2 still reports height=64,width=2048, so a normal
            # row-major reshape mixes all rings into every row.
            out[name] = arr.reshape(w, h).T
    return out


def normalize_u8(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        hi = lo + 1
    return np.clip((image.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def fit_plane(points: np.ndarray, threshold: float = 0.035, iterations: int = 1200) -> dict:
    finite = np.isfinite(points).all(axis=1)
    finite &= np.linalg.norm(points, axis=1) > 0.4
    finite &= np.linalg.norm(points, axis=1) < 5.0
    pts = points[finite]
    if len(pts) < 25:
        return {"detected": False, "reason": "too_few_points", "points": int(len(pts))}
    rng = np.random.default_rng(20260521)
    best = None
    for _ in range(iterations):
        tri = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = np.linalg.norm(n)
        if norm < 1e-8:
            continue
        n = n / norm
        d = -float(n @ tri[0])
        dist = np.abs(pts @ n + d)
        inliers = dist < threshold
        score = int(inliers.sum())
        if best is None or score > best[0]:
            best = (score, inliers)
    if best is None or best[0] < 25:
        return {"detected": False, "reason": "ransac_failed", "points": int(len(pts))}
    plane = pts[best[1]]
    c = plane.mean(axis=0)
    _, _, vh = np.linalg.svd(plane - c, full_matrices=False)
    n = vh[-1]
    n = n / np.linalg.norm(n)
    d = -float(n @ c)
    return {
        "detected": True,
        "points": int(len(plane)),
        "normal_xyz": n.tolist(),
        "d": d,
        "centroid_xyz": c.tolist(),
        "extent_p05_p95_xyz": (np.percentile(plane, 95, axis=0) - np.percentile(plane, 5, axis=0)).tolist(),
        "plane_points": plane,
    }


def save_views(points: np.ndarray, outdir: Path, tag: str, title: str) -> None:
    if len(points) == 0:
        return
    for suffix, a, b, xlabel, ylabel in [
        ("xy", points[:, 0], points[:, 1], "x [m]", "y [m]"),
        ("xz", points[:, 0], points[:, 2], "x [m]", "z [m]"),
        ("yz", points[:, 1], points[:, 2], "y [m]", "z [m]"),
    ]:
        plt.figure(figsize=(6, 5))
        plt.scatter(a, b, s=5)
        plt.axis("equal")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{tag} {title} {suffix}")
        plt.tight_layout()
        plt.savefig(outdir / f"{tag}_{title}_{suffix}.png", dpi=160)
        plt.close()


def nearest_clouds(bag: Path, targets: list[int]) -> list[tuple[int, object]]:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    best: list[tuple[int, object] | None] = [None] * len(targets)
    best_dt = [None] * len(targets)
    with Reader(bag) as reader:
        for conn, bag_time, raw in reader.messages():
            if conn.topic != CLOUD_TOPIC:
                continue
            idx = int(np.argmin(np.abs(np.asarray(targets, dtype=np.int64) - int(bag_time))))
            dt = abs(int(targets[idx]) - int(bag_time))
            if best_dt[idx] is not None and dt >= best_dt[idx]:
                continue
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            best[idx] = (stamp_ns(msg, bag_time), msg)
            best_dt[idx] = dt
    return [item for item in best if item is not None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--sequence-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--x-margin", type=int, default=180)
    parser.add_argument("--y-margin", type=int, default=10)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    windows = json.loads(args.windows.read_text(encoding="utf-8"))["windows"]
    seq = json.loads(args.sequence_summary.read_text(encoding="utf-8"))
    targets = [int(v) for v in seq["target_times_ns"]]
    clouds = nearest_clouds(args.bag, targets)

    out = {"bag": str(args.bag), "source_windows": str(args.windows), "samples": []}
    for i, (cloud_stamp, msg) in enumerate(clouds):
        sample = f"s{i:02d}"
        entry = next((v for v in windows.values() if v["sample"] == sample and v["sensor"] == "ouster_intensity"), None)
        if entry is None or entry["search_roi_xyxy"] is None:
            out["samples"].append({"sample": sample, "detected": False, "reason": "missing_intensity_window"})
            continue
        cloud = cloud_to_organized(msg)
        h, w = cloud["x"].shape
        x0, y0, x1, y1 = entry["search_roi_xyxy"]
        x0 = max(0, x0 - args.x_margin)
        x1 = min(w, x1 + args.x_margin)
        y0 = max(0, y0 - args.y_margin)
        y1 = min(h, y1 + args.y_margin)
        xyz = np.dstack([cloud["x"][y0:y1, x0:x1], cloud["y"][y0:y1, x0:x1], cloud["z"][y0:y1, x0:x1]]).reshape(-1, 3)
        intensity_crop = normalize_u8(cloud.get("intensity", cloud["x"])[y0:y1, x0:x1])
        cv2.imwrite(str(args.out / f"{sample}_manual_intensity_crop.png"), cv2.resize(intensity_crop, (intensity_crop.shape[1] * 8, intensity_crop.shape[0] * 8), interpolation=cv2.INTER_NEAREST))
        finite = np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) > 0.4) & (np.linalg.norm(xyz, axis=1) < 5.0)
        crop_pts = xyz[finite]
        np.save(args.out / f"{sample}_manual_ouster_crop_xyz.npy", crop_pts)
        save_views(crop_pts, args.out, sample, "crop")
        plane = fit_plane(crop_pts)
        serial_plane = {k: v for k, v in plane.items() if k != "plane_points"}
        if plane.get("detected"):
            np.save(args.out / f"{sample}_manual_ouster_plane_xyz.npy", plane["plane_points"])
            save_views(plane["plane_points"], args.out, sample, "plane")
        out["samples"].append(
            {
                "sample": sample,
                "cloud_stamp_ns": int(cloud_stamp),
                "expanded_roi_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                "crop_points": int(len(crop_pts)),
                "plane": serial_plane,
            }
        )
    (args.out / "manual_ouster_roi_extraction_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "planes": sum(1 for s in out["samples"] if s.get("plane", {}).get("detected"))}, indent=2))


if __name__ == "__main__":
    main()
