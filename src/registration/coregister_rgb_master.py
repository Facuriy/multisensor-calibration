#!/usr/bin/env python3
"""Coregister VIS/NIR/Thermal into RGB and crop the common valid intersection.

This is the production-oriented RGB-master coregistration step for extraction.
It intentionally separates dense camera products from sparse Ouster products:

* RGB, VIS, NIR and Thermal are warped/cropped so every pixel in the output crop
  has valid camera data.
* Ouster should be projected later as a sparse/depth layer with its own mask.

Supported inputs:

1. Calibration review layout:

   runs/calibration_20260623_full_review/per_bag/<bag_id>/{rgb,vis,nir,thermal_c}.jpg

2. Extraction layout from ``src/extraction/extract_all_bag_images.py``:

   <root>/metadata/frames.csv
   <root>/images/<bag_id>/<sensor>/<frame>_<stamp>.png

For extraction layout, RGB frames are used as reference and nearest VIS/NIR/
Thermal frames are selected by timestamp.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_CALIBRATION = Path("data/calibration/new_session/20260623/calibration_20260623_final_candidate.json")
DEFAULT_OUT = Path("runs/rgb_master_common_crop")
CAMERA_SENSORS = ("rgb", "vis", "nir", "thermal")
EXTRACT_SENSOR_MAP = {
    "rgb": "rgb",
    "vis": "vis",
    "nir": "nir",
    "thermal_c": "thermal",
    "thermal_raw": "thermal",
}


@dataclass
class SensorImage:
    sensor: str
    path: Path
    stamp_ns: int | None


@dataclass
class FrameSet:
    frame_id: str
    bag_id: str
    images: dict[str, SensorImage]
    source: str


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_homographies(path: Path) -> dict[str, np.ndarray]:
    doc = load_json(path)
    regs = doc["target_plane_registration_to_rgb"]
    return {
        "vis": np.asarray(regs["vis_to_rgb"]["H_sensor_to_rgb"], dtype=np.float64),
        "nir": np.asarray(regs["nir_to_rgb"]["H_sensor_to_rgb"], dtype=np.float64),
        "thermal": np.asarray(regs["thermal_to_rgb"]["H_sensor_to_rgb"], dtype=np.float64),
    }


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError(f"Could not read image: {path}")
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return img[:, :, :3]
    return img


def to_bgr_for_preview(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        if img.dtype == np.uint8:
            return img
        return robust_stretch_color(img)
    gray = robust_stretch(img)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def robust_stretch(img: np.ndarray) -> np.ndarray:
    src = img.astype(np.float32)
    finite = np.isfinite(src)
    if not finite.any():
        return np.zeros(src.shape, dtype=np.uint8)
    lo, hi = np.percentile(src[finite], [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((src - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def robust_stretch_color(img: np.ndarray) -> np.ndarray:
    channels = [robust_stretch(img[:, :, c]) for c in range(img.shape[2])]
    return np.dstack(channels)


def warp_image_and_mask(
    img: np.ndarray,
    h: np.ndarray | None,
    rgb_shape_hw: tuple[int, int],
    interpolation: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = rgb_shape_hw
    if h is None:
        mask = np.ones((height, width), dtype=bool)
        return img.copy(), mask
    warped = cv2.warpPerspective(
        img,
        h,
        (width, height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    src_mask = np.full(img.shape[:2], 255, dtype=np.uint8)
    mask = cv2.warpPerspective(
        src_mask,
        h,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped, mask > 0


def largest_true_rectangle(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return x0,y0,x1,y1 of largest all-True axis-aligned rectangle."""
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    h, w = mask.shape
    heights = np.zeros(w, dtype=np.int32)
    best_area = 0
    best: tuple[int, int, int, int] | None = None
    for y in range(h):
        heights = np.where(mask[y], heights + 1, 0)
        stack: list[int] = []
        for x in range(w + 1):
            current = heights[x] if x < w else 0
            while stack and current < heights[stack[-1]]:
                top = stack.pop()
                height = int(heights[top])
                left = stack[-1] + 1 if stack else 0
                width = x - left
                area = width * height
                if area > best_area:
                    best_area = area
                    best = (left, y - height + 1, x, y + 1)
            stack.append(x)
    return best


def safe_erode(mask: np.ndarray, margin_px: int) -> np.ndarray:
    if margin_px <= 0:
        return mask.astype(bool)
    k = 2 * int(margin_px) + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8) * 255, kernel, iterations=1)
    return eroded > 0


def parse_stamp_from_name(path: Path) -> int | None:
    parts = path.stem.split("_")
    for part in reversed(parts):
        try:
            value = int(part)
        except ValueError:
            continue
        if value > 1_000_000_000_000:
            return value
    return None


def discover_review_frames(input_dir: Path, labels: list[str] | None = None) -> list[FrameSet]:
    out: list[FrameSet] = []
    label_set = set(labels or [])
    for folder in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        if label_set and folder.name not in label_set:
            continue
        paths = {
            "rgb": folder / "rgb.jpg",
            "vis": folder / "vis.jpg",
            "nir": folder / "nir.jpg",
            "thermal": folder / "thermal_c.jpg",
        }
        if not all(p.exists() for p in paths.values()):
            continue
        images = {
            sensor: SensorImage(sensor=sensor, path=path, stamp_ns=None)
            for sensor, path in paths.items()
        }
        out.append(FrameSet(frame_id=folder.name, bag_id=folder.name, images=images, source="review_layout"))
    return out


def discover_extracted_frames(
    input_dir: Path,
    max_sync_ms: float,
    labels: list[str] | None = None,
) -> list[FrameSet]:
    csv_path = input_dir / "metadata" / "frames.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing extraction metadata: {csv_path}")
    label_set = set(labels or [])
    rows: list[dict[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_bag: dict[str, dict[str, list[dict[str, str]]]] = {}
    for row in rows:
        bag_id = row.get("bag_id") or Path(row.get("bag", "")).stem
        if label_set and bag_id not in label_set:
            continue
        sensor_raw = row.get("sensor", "")
        sensor = EXTRACT_SENSOR_MAP.get(sensor_raw)
        if sensor is None:
            continue
        path = input_dir / str(row.get("image_path", ""))
        if not path.exists():
            continue
        by_bag.setdefault(bag_id, {}).setdefault(sensor, []).append(row)
    out: list[FrameSet] = []
    max_dt_ns = int(max_sync_ms * 1_000_000)
    for bag_id, sensors in sorted(by_bag.items()):
        if not all(k in sensors for k in CAMERA_SENSORS):
            continue
        for sensor_rows in sensors.values():
            sensor_rows.sort(key=lambda r: int(r.get("stamp_ns") or parse_stamp_from_name(input_dir / str(r.get("image_path", ""))) or 0))
        for rgb_row in sensors["rgb"]:
            rgb_ts = int(rgb_row.get("stamp_ns") or parse_stamp_from_name(input_dir / str(rgb_row.get("image_path", ""))) or 0)
            images: dict[str, SensorImage] = {
                "rgb": SensorImage("rgb", input_dir / str(rgb_row["image_path"]), rgb_ts)
            }
            ok = True
            for sensor in ("vis", "nir", "thermal"):
                best = min(
                    sensors[sensor],
                    key=lambda r: abs(int(r.get("stamp_ns") or parse_stamp_from_name(input_dir / str(r.get("image_path", ""))) or 0) - rgb_ts),
                )
                ts = int(best.get("stamp_ns") or parse_stamp_from_name(input_dir / str(best.get("image_path", ""))) or 0)
                if abs(ts - rgb_ts) > max_dt_ns:
                    ok = False
                    break
                images[sensor] = SensorImage(sensor, input_dir / str(best["image_path"]), ts)
            if ok:
                frame_index = str(rgb_row.get("frame_index") or "0").zfill(6)
                out.append(FrameSet(frame_id=f"{bag_id}_{frame_index}_{rgb_ts}", bag_id=bag_id, images=images, source="extracted_layout"))
    return out


def discover_frames(input_dir: Path, layout: str, max_sync_ms: float, labels: list[str] | None) -> list[FrameSet]:
    if layout == "review":
        return discover_review_frames(input_dir, labels)
    if layout == "extracted":
        return discover_extracted_frames(input_dir, max_sync_ms, labels)
    if (input_dir / "metadata" / "frames.csv").exists():
        return discover_extracted_frames(input_dir, max_sync_ms, labels)
    return discover_review_frames(input_dir, labels)


def write_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if img.dtype == np.bool_:
        img = img.astype(np.uint8) * 255
    if not cv2.imwrite(str(path), img):
        raise OSError(f"Could not write {path}")


def process_frame(
    frame: FrameSet,
    homographies: dict[str, np.ndarray],
    out_dir: Path,
    margin_px: int,
    min_width: int,
    min_height: int,
    write_full: bool,
) -> dict[str, Any]:
    raw = {sensor: read_image(info.path) for sensor, info in frame.images.items()}
    rgb = raw["rgb"]
    rgb_shape = rgb.shape[:2]
    registered: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for sensor in CAMERA_SENSORS:
        h = None if sensor == "rgb" else homographies[sensor]
        interpolation = cv2.INTER_LINEAR
        registered[sensor], masks[sensor] = warp_image_and_mask(raw[sensor], h, rgb_shape, interpolation)

    common = np.ones(rgb_shape, dtype=bool)
    for mask in masks.values():
        common &= mask
    common_safe = safe_erode(common, margin_px)
    rect = largest_true_rectangle(common_safe)
    if rect is None:
        return {"frame_id": frame.frame_id, "bag_id": frame.bag_id, "processed": False, "reason": "no_common_intersection"}
    x0, y0, x1, y1 = rect
    if (x1 - x0) < min_width or (y1 - y0) < min_height:
        return {
            "frame_id": frame.frame_id,
            "bag_id": frame.bag_id,
            "processed": False,
            "reason": "common_intersection_too_small",
            "common_roi_rgb_xyxy": [x0, y0, x1, y1],
        }

    stem = frame.frame_id.replace(":", "_").replace("\\", "_").replace("/", "_")
    frame_dir = out_dir / "frames" / stem
    for sensor, img in registered.items():
        crop = img[y0:y1, x0:x1]
        write_image(frame_dir / "common_crop" / f"{sensor}.png", crop)
        write_image(frame_dir / "common_crop_preview" / f"{sensor}.jpg", to_bgr_for_preview(crop))
        write_image(frame_dir / "masks" / f"{sensor}_valid_mask.png", masks[sensor][y0:y1, x0:x1])
        if write_full:
            write_image(frame_dir / "full_registered" / f"{sensor}.png", img)
    write_crop_qa(frame_dir, {sensor: registered[sensor][y0:y1, x0:x1] for sensor in CAMERA_SENSORS})
    write_image(frame_dir / "masks" / "common_valid_mask_full.png", common)
    write_image(frame_dir / "masks" / "common_valid_mask_safe_full.png", common_safe)

    invalid_after_crop = {
        sensor: int((~masks[sensor][y0:y1, x0:x1]).sum())
        for sensor in CAMERA_SENSORS
    }
    result = {
        "frame_id": frame.frame_id,
        "bag_id": frame.bag_id,
        "source": frame.source,
        "processed": True,
        "frame_dir": str(frame_dir),
        "common_roi_rgb_xyxy": [x0, y0, x1, y1],
        "common_crop_width": int(x1 - x0),
        "common_crop_height": int(y1 - y0),
        "common_crop_area_px": int((x1 - x0) * (y1 - y0)),
        "rgb_frame_area_px": int(rgb_shape[0] * rgb_shape[1]),
        "common_crop_fraction_of_rgb": float(((x1 - x0) * (y1 - y0)) / (rgb_shape[0] * rgb_shape[1])),
        "mask_margin_px": int(margin_px),
        "invalid_pixels_after_crop": invalid_after_crop,
        "inputs": {
            sensor: {
                "path": str(info.path),
                "stamp_ns": info.stamp_ns,
            }
            for sensor, info in frame.images.items()
        },
    }
    (frame_dir / "coregistration_metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def gray_u8(img: np.ndarray) -> np.ndarray:
    bgr = to_bgr_for_preview(img)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def false_color_pair(rgb_crop: np.ndarray, other_crop: np.ndarray) -> np.ndarray:
    rgb_g = gray_u8(rgb_crop)
    other_g = gray_u8(other_crop)
    out = np.zeros((*rgb_g.shape, 3), dtype=np.uint8)
    out[..., 1] = rgb_g
    out[..., 0] = other_g
    out[..., 2] = other_g
    return out


def edge_pair(rgb_crop: np.ndarray, other_crop: np.ndarray) -> np.ndarray:
    rgb_e = cv2.Canny(gray_u8(rgb_crop), 60, 160)
    other_e = cv2.Canny(gray_u8(other_crop), 50, 150)
    out = np.zeros((*rgb_e.shape, 3), dtype=np.uint8)
    out[..., 1] = rgb_e
    out[..., 2] = other_e
    return out


def write_crop_qa(frame_dir: Path, crops: dict[str, np.ndarray]) -> None:
    qa_dir = frame_dir / "qa_overlay"
    qa_dir.mkdir(parents=True, exist_ok=True)
    rgb_bgr = to_bgr_for_preview(crops["rgb"])
    for sensor in ("vis", "nir", "thermal"):
        other_bgr = to_bgr_for_preview(crops[sensor])
        alpha = cv2.addWeighted(rgb_bgr, 0.58, other_bgr, 0.42, 0)
        false_color = false_color_pair(crops["rgb"], crops[sensor])
        edges = edge_pair(crops["rgb"], crops[sensor])
        write_image(qa_dir / f"rgb_plus_{sensor}.jpg", alpha)
        write_image(qa_dir / f"rgb_green_{sensor}_magenta.jpg", false_color)
        write_image(qa_dir / f"rgb_edges_green_{sensor}_edges_red.jpg", edges)


def make_contactsheet(results: list[dict[str, Any]], out_dir: Path, max_frames: int = 20) -> list[str]:
    pages = []
    tiles = []
    for result in results:
        if not result.get("processed"):
            continue
        frame_dir = Path(result["frame_dir"])
        row_tiles = []
        for sensor in CAMERA_SENSORS:
            img = cv2.imread(str(frame_dir / "common_crop_preview" / f"{sensor}.jpg"), cv2.IMREAD_COLOR)
            if img is None:
                continue
            row_tiles.append(fit_tile(img, (310, 210), sensor.upper()))
        if row_tiles:
            title = np.full((38, 310 * len(row_tiles), 3), 245, dtype=np.uint8)
            cv2.putText(title, str(result["frame_id"])[:80], (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
            tiles.append(np.vstack([title, np.hstack(row_tiles)]))
        if len(tiles) >= max_frames:
            break
    for i in range(0, len(tiles), 4):
        page = np.vstack(tiles[i : i + 4])
        path = out_dir / f"common_crop_contactsheet_{len(pages)+1:02d}.jpg"
        write_image(path, page)
        pages.append(str(path))
    return pages


def make_overlay_contactsheet(results: list[dict[str, Any]], out_dir: Path, max_frames: int = 20) -> list[str]:
    pages = []
    rows = []
    for result in results:
        if not result.get("processed"):
            continue
        frame_dir = Path(result["frame_dir"])
        row_tiles = []
        for sensor in ("vis", "nir", "thermal"):
            img = cv2.imread(str(frame_dir / "qa_overlay" / f"rgb_green_{sensor}_magenta.jpg"), cv2.IMREAD_COLOR)
            if img is not None:
                row_tiles.append(fit_tile(img, (360, 240), f"RGB green / {sensor.upper()} magenta"))
        if row_tiles:
            title = np.full((38, 360 * len(row_tiles), 3), 245, dtype=np.uint8)
            cv2.putText(title, str(result["frame_id"])[:80], (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
            rows.append(np.vstack([title, np.hstack(row_tiles)]))
        if len(rows) >= max_frames:
            break
    for i in range(0, len(rows), 4):
        page = np.vstack(rows[i : i + 4])
        path = out_dir / f"common_crop_overlay_contactsheet_{len(pages)+1:02d}.jpg"
        write_image(path, page)
        pages.append(str(path))
    return pages


def fit_tile(img: np.ndarray, size: tuple[int, int], label: str) -> np.ndarray:
    width, height = size
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    h, w = img.shape[:2]
    scale = min(width / max(w, 1), (height - 28) / max(h, 1))
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    x = (width - small.shape[1]) // 2
    y = 28 + (height - 28 - small.shape[0]) // 2
    canvas[y : y + small.shape[0], x : x + small.shape[1]] = small
    cv2.putText(canvas, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (25, 25, 25), 2, cv2.LINE_AA)
    return canvas


def write_summary(results: list[dict[str, Any]], out_dir: Path, calibration: Path, input_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    processed = [r for r in results if r.get("processed")]
    csv_path = out_dir / "coregistered_frames.csv"
    fieldnames = sorted({k for r in results for k in r.keys() if k not in ("inputs", "invalid_pixels_after_crop")})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    contacts = make_contactsheet(results, out_dir)
    overlay_contacts = make_overlay_contactsheet(results, out_dir)
    summary = {
        "input_dir": str(input_dir),
        "calibration": str(calibration),
        "n_frames": len(results),
        "n_processed": len(processed),
        "n_failed": len(results) - len(processed),
        "contactsheets": contacts,
        "overlay_contactsheets": overlay_contacts,
        "results": results,
    }
    if processed:
        summary["common_crop_fraction_of_rgb_median"] = float(np.median([r["common_crop_fraction_of_rgb"] for r in processed]))
        summary["common_crop_width_median"] = float(np.median([r["common_crop_width"] for r in processed]))
        summary["common_crop_height_median"] = float(np.median([r["common_crop_height"] for r in processed]))
    (out_dir / "coregistration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--layout", choices=["auto", "review", "extracted"], default="auto")
    parser.add_argument("--max-sync-ms", type=float, default=80.0)
    parser.add_argument("--margin-px", type=int, default=2, help="Erode common mask before selecting the crop.")
    parser.add_argument("--min-width", type=int, default=64)
    parser.add_argument("--min-height", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--labels", nargs="*")
    parser.add_argument("--write-full", action="store_true")
    args = parser.parse_args()

    layout = args.layout
    if layout == "auto":
        layout = "extracted" if (args.input_dir / "metadata" / "frames.csv").exists() else "review"
    frames = discover_frames(args.input_dir, layout, args.max_sync_ms, args.labels)
    if args.limit > 0:
        frames = frames[: args.limit]
    if not frames:
        raise SystemExit(f"No synchronized RGB/VIS/NIR/Thermal frames found in {args.input_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    homographies = load_homographies(args.calibration)
    results = []
    for frame in frames:
        result = process_frame(
            frame,
            homographies,
            args.out_dir,
            margin_px=args.margin_px,
            min_width=args.min_width,
            min_height=args.min_height,
            write_full=args.write_full,
        )
        print(json.dumps({k: v for k, v in result.items() if k not in ("inputs",)}, ensure_ascii=False))
        results.append(result)
    write_summary(results, args.out_dir, args.calibration, args.input_dir)
    print(f"Wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
