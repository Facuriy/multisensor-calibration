#!/usr/bin/env python3
"""Scan 2026-06-23 calibration bags for camera checkerboards and LiDAR panel ROIs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.extract_ouster_panel_from_manual_intensity_roi import CLOUD_TOPIC, cloud_to_organized  # noqa: E402
from src.extraction.extract_all_bag_images import decode_ros_image, robust_preview  # noqa: E402


RGB_TOPIC = "/ssf/BFS_usb_0/image_raw"
VIS_TOPIC = "/ssf/photonfocus_camera_vis_node/image_raw"
NIR_TOPIC = "/ssf/photonfocus_camera_nir_node/image_raw"
THERMAL_TOPIC = "/ssf/thermalgrabber_ros/image_deg_celsius"

CAMERA_TOPICS = {
    "rgb": RGB_TOPIC,
    "vis": VIS_TOPIC,
    "nir": NIR_TOPIC,
    "thermal_c": THERMAL_TOPIC,
}

PATTERN = (9, 6)
SQUARE_M = 0.04
OBJ_PTS = np.array(
    [[c * SQUARE_M, r * SQUARE_M, 0.0] for r in range(PATTERN[1]) for c in range(PATTERN[0])],
    dtype=np.float32,
)

# Historical RGB intrinsics only guide the LiDAR search ROI. Final intrinsics
# must be recomputed from this dataset.
K_RGB_APPROX = np.array(
    [[3623.188, 0.0, 1224.0], [0.0, 3623.188, 1024.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
DIST_ZERO = np.zeros(5, dtype=np.float64)


def normalize_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.float32)
    finite = np.isfinite(gray)
    if not finite.any():
        return np.zeros(gray.shape, dtype=np.uint8)
    lo, hi = np.percentile(gray[finite], [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((gray - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return cv2.equalizeHist(out)


def detect_checker(image: np.ndarray) -> np.ndarray | None:
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    h, w = image.shape[:2]

    def try_img(src: np.ndarray, scale: float, off_x: int = 0, off_y: int = 0) -> np.ndarray | None:
        sh, sw = src.shape[:2]
        small = cv2.resize(
            src,
            (max(1, int(sw * scale)), max(1, int(sh * scale))),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )
        gray = normalize_gray(small)
        variants = [gray, cv2.bitwise_not(gray)]
        for v in variants:
            ok, corners = cv2.findChessboardCornersSB(v, PATTERN, flags=flags)
            if ok and corners is not None:
                pts = corners.reshape(-1, 2).astype(np.float32) / scale
                pts[:, 0] += off_x
                pts[:, 1] += off_y
                return pts
        return None

    for scale in (0.25, 0.5, 1.0):
        pts = try_img(image, scale)
        if pts is not None:
            return pts

    # Some high/edge poses show only part of the board in the large frame. Try
    # broad crops, but still require a complete internal 9x6 detection.
    crops = [
        (0, 0, w // 2, h),
        (w // 2, 0, w, h),
        (0, 0, w, h // 2),
        (0, h // 2, w, h),
        (0, 0, int(w * 0.65), int(h * 0.65)),
        (int(w * 0.35), 0, w, int(h * 0.65)),
        (0, int(h * 0.35), int(w * 0.65), h),
        (int(w * 0.35), int(h * 0.35), w, h),
    ]
    for x0, y0, x1, y1 in crops:
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        for scale in (0.5, 0.25, 1.0):
            pts = try_img(crop, scale, x0, y0)
            if pts is not None:
                return pts
    return None


def solve_rgb_pose(corners: np.ndarray) -> dict | None:
    ok, rvec, tvec = cv2.solvePnP(
        OBJ_PTS,
        corners.reshape(-1, 1, 2),
        K_RGB_APPROX,
        DIST_ZERO,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    pts_cam = OBJ_PTS @ R.T + tvec.reshape(1, 3)
    return {
        "rvec": rvec.reshape(-1).tolist(),
        "tvec": tvec.reshape(-1).tolist(),
        "centroid_cam": pts_cam.mean(axis=0).tolist(),
        "normal_cam": R[:, 2].reshape(-1).tolist(),
        "distance_m": float(np.linalg.norm(pts_cam.mean(axis=0))),
    }


def collect_messages(bag: Path, topic: str, typestore, max_messages: int = 0) -> list[tuple[int, object]]:
    out: list[tuple[int, object]] = []
    with Reader(bag) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            return out
        for conn, ts, raw in reader.messages(connections=conns):
            out.append((int(ts), typestore.deserialize_ros1(raw, conn.msgtype)))
            if max_messages and len(out) >= max_messages:
                break
    return out


def nearest(items: list[tuple[int, object]], stamp_ns: int) -> tuple[int, object] | None:
    if not items:
        return None
    return min(items, key=lambda item: abs(item[0] - stamp_ns))


def candidate_frame_indices(n: int, max_frames: int) -> list[int]:
    if n <= max_frames:
        return list(range(n))
    # Dense enough to catch a stable frame, including beginning/middle/end.
    return sorted(set(np.linspace(0, n - 1, max_frames, dtype=int).tolist()))


def draw_corners_preview(img: np.ndarray, corners: np.ndarray | None, label: str) -> np.ndarray:
    prev = robust_preview(img, "rgb" if img.ndim == 3 else "vis")
    h, w = prev.shape[:2]
    scale = min(720 / max(w, 1), 520 / max(h, 1), 1.0)
    small = cv2.resize(prev, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    if corners is not None:
        pts = (corners * scale).astype(int)
        for p in pts:
            cv2.circle(small, tuple(p), 4, (0, 255, 255), -1, cv2.LINE_AA)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        cv2.rectangle(small, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.putText(small, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return small


def ransac_plane(points: np.ndarray, threshold: float = 0.035, iterations: int = 700) -> dict:
    if len(points) < 12:
        return {"detected": False, "reason": "too_few_points", "points": int(len(points))}
    rng = np.random.default_rng(20260623)
    best_mask = None
    best_score = 0
    for _ in range(iterations):
        tri = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        d = -float(n @ tri[0])
        dist = np.abs(points @ n + d)
        mask = dist < threshold
        score = int(mask.sum())
        if score > best_score:
            best_score = score
            best_mask = mask
    if best_mask is None or best_score < 12:
        return {"detected": False, "reason": "ransac_failed", "points": int(len(points))}
    plane = points[best_mask]
    c = plane.mean(axis=0)
    _, _, vh = np.linalg.svd(plane - c, full_matrices=False)
    n = vh[-1]
    n = n / np.linalg.norm(n)
    d = -float(n @ c)
    rms = float(np.sqrt(np.mean((plane @ n + d) ** 2)))
    ext = np.percentile(plane, 95, axis=0) - np.percentile(plane, 5, axis=0)
    return {
        "detected": True,
        "points": int(len(plane)),
        "normal_xyz": n.tolist(),
        "d": d,
        "centroid_xyz": c.tolist(),
        "rms_m": rms,
        "extent_p05_p95_xyz": ext.tolist(),
        "_mask": best_mask,
    }


def lidar_panel_from_rgb_pose(cloud_msg, pose: dict, T_cam_lidar: np.ndarray, out_path: Path, tag: str) -> dict:
    cloud = cloud_to_organized(cloud_msg)
    xyz_img = np.dstack([cloud["x"], cloud["y"], cloud["z"]]).astype(np.float64)
    intensity = cloud.get("intensity", cloud["x"]).astype(np.float32)
    h, w = intensity.shape

    centroid_cam = np.asarray(pose["centroid_cam"], dtype=np.float64)
    normal_cam = np.asarray(pose["normal_cam"], dtype=np.float64)
    T_lidar_cam = np.linalg.inv(T_cam_lidar)
    centroid_lid = T_lidar_cam[:3, :3] @ centroid_cam + T_lidar_cam[:3, 3]
    normal_lid = T_lidar_cam[:3, :3] @ normal_cam
    normal_lid = normal_lid / max(np.linalg.norm(normal_lid), 1e-9)

    xyz = xyz_img.reshape(-1, 3)
    finite = np.isfinite(xyz).all(axis=1)
    finite &= np.linalg.norm(xyz, axis=1) > 0.05
    finite &= np.linalg.norm(xyz, axis=1) < 5.0
    dv = xyz - centroid_lid.reshape(1, 3)
    along = dv @ normal_lid
    lateral = np.linalg.norm(dv - along[:, None] * normal_lid.reshape(1, 3), axis=1)
    broad = finite & (np.abs(along) < 0.20) & (lateral < 0.62)
    tight = finite & (np.abs(along) < 0.10) & (lateral < 0.42)

    rows, cols = np.where(broad.reshape(h, w))
    if len(rows):
        y0 = max(0, int(rows.min()) - 4)
        y1 = min(h, int(rows.max()) + 5)
        x0 = max(0, int(cols.min()) - 20)
        x1 = min(w, int(cols.max()) + 21)
    else:
        # Fallback: use only valid columns around finite target; never scan whole
        # 2048x64 image.
        valid_rows, valid_cols = np.where(finite.reshape(h, w))
        if len(valid_rows) == 0:
            return {"detected": False, "reason": "no_finite_lidar_points"}
        y0, y1 = int(np.percentile(valid_rows, 5)), int(np.percentile(valid_rows, 95)) + 1
        x0, x1 = int(np.percentile(valid_cols, 40)), int(np.percentile(valid_cols, 70)) + 1

    roi_mask = np.zeros((h, w), dtype=bool)
    roi_mask[y0:y1, x0:x1] = True
    pts = xyz[(tight.reshape(h, w) & roi_mask).reshape(-1)]
    if len(pts) < 12:
        pts = xyz[(broad.reshape(h, w) & roi_mask).reshape(-1)]
    plane = ransac_plane(pts)

    crop = intensity[y0:y1, x0:x1].copy()
    finite_crop = np.isfinite(crop)
    if finite_crop.any():
        lo, hi = np.percentile(crop[finite_crop], [1, 99])
        if hi <= lo:
            hi = lo + 1
        norm = np.clip((crop - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    else:
        norm = np.zeros(crop.shape, dtype=np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    # Draw candidate pixels selected by the geometry-guided ROI.
    mask_crop = (tight.reshape(h, w) & roi_mask)[y0:y1, x0:x1]
    color[mask_crop] = (0, 255, 255)
    scale_x = 5
    scale_y = 8
    view = cv2.resize(color, (max(1, color.shape[1] * scale_x), max(1, color.shape[0] * scale_y)), interpolation=cv2.INTER_NEAREST)
    cv2.putText(view, tag, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), view)

    serial_plane = {k: v for k, v in plane.items() if not k.startswith("_")}
    return {
        "roi_xyxy_ouster": [int(x0), int(y0), int(x1), int(y1)],
        "candidate_points": int(len(pts)),
        "plane": serial_plane,
        "predicted_centroid_lidar": centroid_lid.tolist(),
        "predicted_normal_lidar": normal_lid.tolist(),
        "preview": str(out_path),
    }


def process_bag(row: dict[str, str], args, typestore) -> dict:
    bag = Path(row["bag_path"])
    bag_out = args.out / "bags" / bag.stem
    bag_out.mkdir(parents=True, exist_ok=True)
    result = {**row, "camera_detections": {}, "best_rgb_stamp_ns": None, "lidar": None}

    with Reader(bag) as reader:
        conns_by_topic = {c.topic: c for c in reader.connections}
    rgb_msgs = collect_messages(bag, RGB_TOPIC, typestore)
    cloud_msgs = collect_messages(bag, CLOUD_TOPIC, typestore) if args.lidar else []

    best_rgb = None
    for idx in candidate_frame_indices(len(rgb_msgs), args.max_rgb_frames):
        ts, msg = rgb_msgs[idx]
        img = decode_ros_image(msg)
        if img is None:
            continue
        corners = detect_checker(img)
        if corners is None:
            continue
        pose = solve_rgb_pose(corners)
        if pose is None:
            continue
        best_rgb = (idx, ts, img, corners, pose)
        break

    if best_rgb is not None:
        idx, ts, img, corners, pose = best_rgb
        result["best_rgb_stamp_ns"] = int(ts)
        result["best_rgb_frame_index"] = int(idx)
        result["rgb_pose_approx"] = pose
        result["camera_detections"]["rgb"] = {
            "detected": True,
            "frame_index": int(idx),
            "stamp_ns": int(ts),
            "corners_px": corners.tolist(),
            "bbox_xyxy": [
                float(corners[:, 0].min()),
                float(corners[:, 1].min()),
                float(corners[:, 0].max()),
                float(corners[:, 1].max()),
            ],
        }
        cv2.imwrite(str(bag_out / "rgb_checker.jpg"), draw_corners_preview(img, corners, "rgb checker"))
    else:
        result["camera_detections"]["rgb"] = {"detected": False}

    # Check synchronized non-RGB sensors near the RGB detection, or the middle
    # frame if RGB failed. These detections are for quality control and future
    # intrinsics, not for LiDAR ROI guidance.
    ref_ts = best_rgb[1] if best_rgb is not None else None
    for key, topic in CAMERA_TOPICS.items():
        if key == "rgb":
            continue
        msgs = collect_messages(bag, topic, typestore)
        if not msgs:
            result["camera_detections"][key] = {"detected": False, "reason": "missing_topic"}
            continue
        item = nearest(msgs, ref_ts) if ref_ts is not None else msgs[len(msgs) // 2]
        assert item is not None
        ts, msg = item
        img = decode_ros_image(msg)
        if img is None:
            result["camera_detections"][key] = {"detected": False, "reason": "decode_failed"}
            continue
        corners = detect_checker(robust_preview(img, key))
        det = {"detected": corners is not None, "stamp_ns": int(ts)}
        if corners is not None:
            det["corners_px"] = corners.tolist()
            det["bbox_xyxy"] = [
                float(corners[:, 0].min()),
                float(corners[:, 1].min()),
                float(corners[:, 0].max()),
                float(corners[:, 1].max()),
            ]
        cv2.imwrite(str(bag_out / f"{key}_checker.jpg"), draw_corners_preview(robust_preview(img, key), corners, f"{key} checker"))
        result["camera_detections"][key] = det

    if args.lidar and best_rgb is not None and cloud_msgs:
        cloud_item = nearest(cloud_msgs, best_rgb[1])
        assert cloud_item is not None
        dt_ms = abs(cloud_item[0] - best_rgb[1]) / 1e6
        if dt_ms <= args.max_cloud_dt_ms:
            result["lidar"] = lidar_panel_from_rgb_pose(
                cloud_item[1],
                best_rgb[4],
                args.T_cam_lidar,
                bag_out / "ouster_intensity_guided_roi.jpg",
                f"lidar roi | dt={dt_ms:.1f} ms",
            )
            result["lidar"]["cloud_stamp_ns"] = int(cloud_item[0])
            result["lidar"]["cloud_dt_ms"] = float(dt_ms)
        else:
            result["lidar"] = {"detected": False, "reason": "nearest_cloud_too_far", "cloud_dt_ms": float(dt_ms)}
    elif args.lidar:
        result["lidar"] = {"detected": False, "reason": "no_rgb_detection_or_no_cloud"}

    return result


def make_summary_pages(results: list[dict], out: Path) -> None:
    thumbs = []
    for res in results:
        bag_dir = out / "bags" / Path(res["bag_path"]).stem
        cells = []
        for name in ("rgb_checker.jpg", "vis_checker.jpg", "nir_checker.jpg", "thermal_c_checker.jpg", "ouster_intensity_guided_roi.jpg"):
            path = bag_dir / name
            if path.exists():
                img = cv2.imread(str(path))
                img = cv2.resize(img, (260, 170), interpolation=cv2.INTER_AREA)
            else:
                img = np.full((170, 260, 3), 35, dtype=np.uint8)
                cv2.putText(img, "missing", (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 255), 2, cv2.LINE_AA)
            cells.append(img)
        row_img = np.hstack(cells)
        label = f"{res.get('label_norm')} | RGB={res['camera_detections'].get('rgb',{}).get('detected')} | LIDAR={res.get('lidar',{}).get('plane',{}).get('detected')}"
        label_img = np.full((36, row_img.shape[1], 3), 18, dtype=np.uint8)
        cv2.putText(label_img, label[:150], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
        thumbs.append(np.vstack([label_img, row_img]))
    for page, start in enumerate(range(0, len(thumbs), 6), start=1):
        cv2.imwrite(str(out / f"detection_review_page_{page:02d}.jpg"), np.vstack(thumbs[start:start + 6]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--T-cam-lidar", type=Path, required=True)
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--lidar", action="store_true")
    parser.add_argument("--only-label-contains", default="")
    parser.add_argument("--max-rgb-frames", type=int, default=18)
    parser.add_argument("--max-cloud-dt-ms", type=float, default=180.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.T_cam_lidar = np.load(args.T_cam_lidar)

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    if not args.include_test:
        rows = [r for r in rows if r.get("height_level") != "test"]
    if args.only_label_contains:
        rows = [r for r in rows if args.only_label_contains.lower() in r.get("label_norm", "").lower()]
    typestore = get_typestore(Stores.ROS1_NOETIC)
    results = []
    for i, row in enumerate(rows, start=1):
        print(f"[{i}/{len(rows)}] {row['label_norm']} {Path(row['bag_path']).name}")
        try:
            results.append(process_bag(row, args, typestore))
        except Exception as exc:
            results.append({**row, "error": repr(exc), "camera_detections": {}, "lidar": {"detected": False, "reason": "exception"}})
            print(f"  ERROR: {exc!r}")

    summary = {
        "pattern_internal_corners": list(PATTERN),
        "square_size_m": SQUARE_M,
        "n_bags": len(results),
        "rgb_detected": sum(1 for r in results if r.get("camera_detections", {}).get("rgb", {}).get("detected")),
        "vis_detected": sum(1 for r in results if r.get("camera_detections", {}).get("vis", {}).get("detected")),
        "nir_detected": sum(1 for r in results if r.get("camera_detections", {}).get("nir", {}).get("detected")),
        "thermal_detected": sum(1 for r in results if r.get("camera_detections", {}).get("thermal_c", {}).get("detected")),
        "lidar_planes": sum(1 for r in results if r.get("lidar", {}).get("plane", {}).get("detected")),
        "results": results,
    }
    (args.out / "scan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    flat_rows = []
    for r in results:
        flat_rows.append(
            {
                "bag": Path(r["bag_path"]).name,
                "label_norm": r.get("label_norm", ""),
                "rgb": r.get("camera_detections", {}).get("rgb", {}).get("detected", False),
                "vis": r.get("camera_detections", {}).get("vis", {}).get("detected", False),
                "nir": r.get("camera_detections", {}).get("nir", {}).get("detected", False),
                "thermal_c": r.get("camera_detections", {}).get("thermal_c", {}).get("detected", False),
                "lidar_plane": r.get("lidar", {}).get("plane", {}).get("detected", False),
                "lidar_points": r.get("lidar", {}).get("plane", {}).get("points", ""),
                "lidar_reason": r.get("lidar", {}).get("reason", r.get("lidar", {}).get("plane", {}).get("reason", "")),
                "error": r.get("error", ""),
            }
        )
    with (args.out / "scan_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)
    make_summary_pages(results, args.out)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
