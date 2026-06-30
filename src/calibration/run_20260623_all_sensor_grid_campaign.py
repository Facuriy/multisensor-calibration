#!/usr/bin/env python3
"""Fast all-frame grid recovery campaign for the 2026-06-23 calibration bags.

This script integrates the thermal grid-recovery idea into a bounded production
pass.  It scans every frame for the selected cameras, uses cheap template scores
to find visible checker grids, and writes best candidates per bag/sensor.

The outputs are model-based candidates.  They are useful for review and for
building calibration candidates, but should not silently replace subpixel
OpenCV detections without visual QA.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIB_DIR = Path(__file__).resolve().parent
for p in (PROJECT_ROOT, CALIB_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.extraction.extract_all_bag_images import decode_ros_image  # noqa: E402
from src.calibration.thermal_checker_raw import flat_field_raw, robust01  # noqa: E402


MANIFEST = Path("data/calibration/new_session/20260623/bag_manifest_20260623.csv")
BAG_CACHE = Path("runs/calibration_20260623_raw_bag_cache")
OUT = Path("runs/calibration_20260623_all_sensor_grid_campaign")

TOPICS = {
    "rgb": "/ssf/BFS_usb_0/image_raw",
    "vis": "/ssf/photonfocus_camera_vis_node/image_raw",
    "nir": "/ssf/photonfocus_camera_nir_node/image_raw",
    "thermal_raw": "/ssf/thermalgrabber_ros/image_mono16",
    "thermal_c": "/ssf/thermalgrabber_ros/image_deg_celsius",
}

GRID_SQUARES = (9, 6)


@dataclass
class Template:
    grid: tuple[int, int]
    width: int
    height: int
    image: np.ndarray


def load_manifest(path: Path, include_nondefault: bool) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    out = []
    for row in rows:
        if row.get("label_norm") == "test":
            continue
        if not include_nondefault and row.get("include_default") != "yes":
            continue
        out.append(row)
    return out


def checker_template(nx: int, ny: int, width: int, height: int) -> np.ndarray:
    small = np.zeros((ny, nx), dtype=np.uint8)
    for y in range(ny):
        for x in range(nx):
            small[y, x] = 255 if ((x + y) % 2) else 0
    templ = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    return cv2.GaussianBlur(templ, (3, 3), 0)


def build_templates(widths: range, height_scales: tuple[float, ...]) -> list[Template]:
    templates: list[Template] = []
    nx, ny = GRID_SQUARES
    for width in widths:
        base_h = width * ny / nx
        for hs in height_scales:
            height = int(round(base_h * hs))
            if height < 24:
                continue
            templates.append(Template(GRID_SQUARES, int(width), height, checker_template(nx, ny, int(width), height)))
    return templates


def normalize_u8(img: np.ndarray, lo: float = 1, hi: float = 99) -> np.ndarray:
    src = img.astype(np.float32)
    finite = np.isfinite(src)
    if not finite.any():
        return np.zeros(src.shape[:2], dtype=np.uint8)
    a, b = np.percentile(src[finite], (lo, hi))
    if b <= a:
        b = a + 1.0
    return np.clip((src - a) * 255.0 / (b - a), 0, 255).astype(np.uint8)


def local_detail(img: np.ndarray, sigma: float = 13.0) -> np.ndarray:
    src = img.astype(np.float32)
    bg = cv2.GaussianBlur(src, (0, 0), sigma)
    return normalize_u8(src - bg, 1, 99)


def photonfocus_preview(raw: np.ndarray, pattern: int) -> np.ndarray:
    h = (raw.shape[0] // pattern) * pattern
    w = (raw.shape[1] // pattern) * pattern
    work = raw[:h, :w].astype(np.float32)
    bands = [work[ro::pattern, co::pattern] for ro in range(pattern) for co in range(pattern)]
    return np.mean(np.stack(bands, axis=-1), axis=-1).astype(np.float32)


def sensor_variants(sensor: str, image: np.ndarray) -> list[tuple[str, np.ndarray, float]]:
    """Return (variant_name, u8_image, coordinate_scale_to_sensor).

    coordinate_scale_to_sensor converts candidate coordinates in variant image
    coordinates back to the image coordinate system used by that sensor's
    preview/calibration output.
    """
    if sensor == "rgb":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else normalize_u8(image)
        small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        return [("rgb_gray_s025_clahe", cv2.createCLAHE(2.0, (8, 8)).apply(small), 4.0)]

    if sensor in ("vis", "nir"):
        pat = 4 if sensor == "vis" else 5
        mean = photonfocus_preview(image, pat)
        u8 = normalize_u8(mean, 1, 99)
        detail = local_detail(u8, 7)
        return [
            (f"{sensor}_mean{pat}_clahe", cv2.createCLAHE(2.2, (6, 6)).apply(u8), 1.0),
            (f"{sensor}_mean{pat}_detail", cv2.bilateralFilter(detail, 5, 25, 5), 1.0),
        ]

    # thermal_raw and thermal_c are both scalar images in native thermal coords.
    work = image.astype(np.float32)
    resid = flat_field_raw(work)
    base = (robust01(resid, 1, 99) * 255).astype(np.uint8)
    bil = cv2.bilateralFilter(base, 5, 30, 5)
    detail = local_detail(work, 17)
    return [
        (f"{sensor}_ff_bilateral_clahe", cv2.createCLAHE(2.0, (8, 8)).apply(bil), 1.0),
        (f"{sensor}_detail17_bilateral", cv2.bilateralFilter(detail, 5, 25, 5), 1.0),
    ]


def score_cells(u8: np.ndarray, box: tuple[int, int, int, int]) -> float:
    nx, ny = GRID_SQUARES
    x0, y0, x1, y1 = box
    crop = u8[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0
    h, w = crop.shape[:2]
    means = np.zeros((ny, nx), dtype=np.float32)
    for y in range(ny):
        for x in range(nx):
            cx0 = int(round(x * w / nx + 0.18 * w / nx))
            cx1 = int(round((x + 1) * w / nx - 0.18 * w / nx))
            cy0 = int(round(y * h / ny + 0.18 * h / ny))
            cy1 = int(round((y + 1) * h / ny - 0.18 * h / ny))
            means[y, x] = float(np.mean(crop[cy0:max(cy0 + 1, cy1), cx0:max(cx0 + 1, cx1)]))
    parity = np.fromfunction(lambda yy, xx: ((xx + yy) % 2) * 2 - 1, means.shape).astype(np.float32)
    centered = means - float(np.mean(means))
    denom = float(np.sqrt(np.sum(centered * centered) * np.sum(parity * parity))) + 1e-6
    corr = abs(float(np.sum(centered * parity) / denom))
    diff = abs(float(np.mean(means[parity > 0]) - np.mean(means[parity < 0])))
    return 100.0 * corr + 0.25 * diff


def fast_template_search(u8: np.ndarray, templates: list[Template]) -> dict[str, Any] | None:
    img = cv2.GaussianBlur(u8, (3, 3), 0)
    h, w = img.shape[:2]
    best: dict[str, Any] | None = None
    for templ in templates:
        if templ.width >= w or templ.height >= h:
            continue
        res = cv2.matchTemplate(img, templ.image, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if abs(min_val) > abs(max_val):
            loc = min_loc
            corr = -float(min_val)
            polarity = "inv"
        else:
            loc = max_loc
            corr = float(max_val)
            polarity = "pos"
        x0, y0 = int(loc[0]), int(loc[1])
        box = (x0, y0, x0 + templ.width, y0 + templ.height)
        cell = score_cells(img, box)
        score = 100.0 * corr + cell
        if best is None or score > float(best["score"]):
            best = {
                "grid_squares": list(templ.grid),
                "pattern_internal_corners": [templ.grid[0] - 1, templ.grid[1] - 1],
                "x0": box[0],
                "y0": box[1],
                "x1": box[2],
                "y1": box[3],
                "corr": corr,
                "polarity": polarity,
                "cell_score": cell,
                "score": score,
            }
    return best


def corners_from_candidate(cand: dict[str, Any], scale_to_sensor: float) -> np.ndarray:
    nx, ny = [int(v) for v in cand["grid_squares"]]
    xs = np.linspace(cand["x0"], cand["x1"], nx + 1, dtype=np.float32)[1:-1]
    ys = np.linspace(cand["y0"], cand["y1"], ny + 1, dtype=np.float32)[1:-1]
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    pts *= float(scale_to_sensor)
    return pts


def confidence(sensor: str, cand: dict[str, Any]) -> str:
    corr = float(cand.get("corr", 0.0))
    score = float(cand.get("score", 0.0))
    if sensor == "rgb":
        return "strong" if corr >= 0.45 and score >= 125 else "weak_reject"
    if sensor in ("vis", "nir"):
        if corr >= 0.50 and score >= 140:
            return "strong"
        if corr >= 0.38 and score >= 118:
            return "partial_visual_check"
        return "weak_reject"
    if corr >= 0.55 and score >= 150:
        return "strong"
    if corr >= 0.38 and score >= 118:
        return "partial_visual_check"
    return "weak_reject"


def draw_review(u8: np.ndarray, cand: dict[str, Any], out_path: Path, title: str) -> None:
    color = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    x0, y0, x1, y1 = [int(cand[k]) for k in ("x0", "y0", "x1", "y1")]
    nx, ny = [int(v) for v in cand["grid_squares"]]
    cv2.rectangle(color, (x0, y0), (x1, y1), (0, 210, 255), 2)
    for x in np.linspace(x0, x1, nx + 1):
        cv2.line(color, (int(round(x)), y0), (int(round(x)), y1), (0, 120, 255), 1)
    for y in np.linspace(y0, y1, ny + 1):
        cv2.line(color, (x0, int(round(y))), (x1, int(round(y))), (0, 120, 255), 1)
    cv2.putText(color, title[:95], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
    h, w = color.shape[:2]
    scale = min(560 / max(w, 1), 420 / max(h, 1), 3.0)
    view = cv2.resize(color, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), view)


def process_bag(
    row: dict[str, str],
    bag_path: Path,
    sensors: list[str],
    templates_by_sensor: dict[str, list[Template]],
    out_dir: Path,
) -> dict[str, Any]:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    label = row["label_norm"]
    best_by_sensor: dict[str, dict[str, Any]] = {}
    with Reader(bag_path) as reader:
        available = {c.topic: c for c in reader.connections}
        selected = []
        topic_to_sensor = {}
        for sensor in sensors:
            topic = TOPICS[sensor]
            if topic in available:
                selected.append(available[topic])
                topic_to_sensor[topic] = sensor
        counters = {s: 0 for s in sensors}
        for conn, ts, raw in reader.messages(connections=selected):
            sensor = topic_to_sensor[conn.topic]
            frame_index = counters[sensor]
            counters[sensor] += 1
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            img = decode_ros_image(msg)
            if img is None:
                continue
            for variant_name, u8, scale_to_sensor in sensor_variants(sensor, img):
                cand = fast_template_search(u8, templates_by_sensor[sensor])
                if cand is None:
                    continue
                conf = confidence(sensor, cand)
                if conf == "weak_reject":
                    continue
                score = float(cand["score"])
                current = best_by_sensor.get(sensor)
                if current is not None and score <= float(current["score"]):
                    continue
                corners = corners_from_candidate(cand, scale_to_sensor)
                review_path = out_dir / "bags" / label / f"{sensor}_best_grid.jpg"
                draw_review(
                    u8,
                    cand,
                    review_path,
                    f"{label} {sensor} f{frame_index} {conf} corr={cand['corr']:.2f} score={score:.1f}",
                )
                best_by_sensor[sensor] = {
                    "checker_detected": True,
                    "source": "fast_grid_campaign",
                    "label_norm": label,
                    "bag": row["bag"],
                    "sensor": sensor,
                    "frame_index": int(frame_index),
                    "stamp_ns": int(ts),
                    "variant": variant_name,
                    "confidence": conf,
                    "usable_for_calibration": conf == "strong",
                    "method": f"{variant_name}_fast_template_grid",
                    "score": score,
                    "corr": float(cand["corr"]),
                    "cell_score": float(cand["cell_score"]),
                    "grid_squares": cand["grid_squares"],
                    "pattern_internal_corners": cand["pattern_internal_corners"],
                    "box_xyxy_variant": [cand["x0"], cand["y0"], cand["x1"], cand["y1"]],
                    "corners_px": corners.tolist(),
                    "review_path": str(review_path),
                    "warning": "model-based grid candidate; visually review before calibration use",
                }
    return {
        **row,
        "bag_path_local": str(bag_path),
        "processed": True,
        "detections": best_by_sensor,
    }


def make_contact_sheets(results: list[dict[str, Any]], sensors: list[str], out_dir: Path) -> None:
    strips = []
    for row in results:
        cells = []
        for sensor in sensors:
            det = row.get("detections", {}).get(sensor, {})
            path = det.get("review_path")
            img = cv2.imread(path) if path else None
            if img is None:
                img = np.full((210, 280, 3), 245, dtype=np.uint8)
                cv2.putText(img, f"{sensor}: none", (18, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2)
            img = cv2.resize(img, (280, 210), interpolation=cv2.INTER_AREA)
            cells.append(img)
        strip = np.hstack(cells)
        canvas = np.full((240, strip.shape[1], 3), 250, dtype=np.uint8)
        canvas[30:, :] = strip
        label = row.get("label_norm", "")
        status = " ".join(
            f"{s}:{row.get('detections', {}).get(s, {}).get('confidence', '-')[:1]}" for s in sensors
        )
        cv2.putText(canvas, f"{label} | {status}"[:120], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
        strips.append(canvas)
    for page_i, start in enumerate(range(0, len(strips), 4), 1):
        batch = strips[start : start + 4]
        if not batch:
            continue
        while len(batch) < 4:
            batch.append(np.full_like(batch[0], 250))
        cv2.imwrite(str(out_dir / f"all_sensor_grid_campaign_page_{page_i:02d}.jpg"), np.vstack(batch))


def write_outputs(results: list[dict[str, Any]], sensors: list[str], out_dir: Path, args: argparse.Namespace) -> None:
    counts = {
        s: {
            "strong": sum(1 for r in results if r.get("detections", {}).get(s, {}).get("confidence") == "strong"),
            "partial": sum(1 for r in results if r.get("detections", {}).get(s, {}).get("confidence") == "partial_visual_check"),
        }
        for s in sensors
    }
    summary = {
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "grid_squares_searched": list(GRID_SQUARES),
        "n_bags": len(results),
        "counts": counts,
        "results": results,
    }
    (out_dir / "all_sensor_grid_campaign_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "all_sensor_grid_campaign_table.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["label_norm", "bag"]
        for s in sensors:
            fields += [f"{s}_confidence", f"{s}_frame", f"{s}_corr", f"{s}_score", f"{s}_review"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            rec = {"label_norm": row.get("label_norm"), "bag": row.get("bag")}
            for s in sensors:
                det = row.get("detections", {}).get(s, {})
                rec[f"{s}_confidence"] = det.get("confidence", "")
                rec[f"{s}_frame"] = det.get("frame_index", "")
                rec[f"{s}_corr"] = det.get("corr", "")
                rec[f"{s}_score"] = det.get("score", "")
                rec[f"{s}_review"] = det.get("review_path", "")
            writer.writerow(rec)
    make_contact_sheets(results, sensors, out_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--bag-cache", type=Path, default=BAG_CACHE)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--sensors", nargs="+", choices=sorted(TOPICS), default=["vis", "nir", "thermal_raw", "thermal_c"])
    parser.add_argument("--include-nondefault", action="store_true")
    parser.add_argument("--only-label", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(args.manifest, args.include_nondefault)
    if args.only_label:
        rows = [r for r in rows if r.get("label_norm") == args.only_label or r.get("bag") == args.only_label]

    templates_by_sensor: dict[str, list[Template]] = {}
    for sensor in args.sensors:
        if sensor == "rgb":
            templates_by_sensor[sensor] = build_templates(range(25, 140, 8), (0.72, 0.86, 1.0))
        elif sensor in ("vis", "nir"):
            templates_by_sensor[sensor] = build_templates(range(70, 300, 12), (0.70, 0.85, 1.0))
        else:
            templates_by_sensor[sensor] = build_templates(range(110, 390, 14), (0.68, 0.84, 1.0))

    results = []
    for i, row in enumerate(rows, 1):
        label = row["label_norm"]
        checkpoint = args.out_dir / "checkpoints" / f"{label}.json"
        if args.resume and checkpoint.exists():
            print(f"[{i}/{len(rows)}] {label}: checkpoint")
            results.append(json.loads(checkpoint.read_text(encoding="utf-8")))
            continue
        bag_path = args.bag_cache / row["bag"]
        if not bag_path.exists():
            rec = {**row, "processed": False, "reason": f"missing_local_bag:{bag_path}", "detections": {}}
            results.append(rec)
            continue
        print(f"[{i}/{len(rows)}] {label}: scanning {', '.join(args.sensors)}")
        rec = process_bag(row, bag_path, args.sensors, templates_by_sensor, args.out_dir)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        results.append(rec)
        write_outputs(results, args.sensors, args.out_dir, args)
    write_outputs(results, args.sensors, args.out_dir, args)
    print(json.dumps({"n_bags": len(results), "counts": {
        s: {
            "strong": sum(1 for r in results if r.get("detections", {}).get(s, {}).get("confidence") == "strong"),
            "partial": sum(1 for r in results if r.get("detections", {}).get(s, {}).get("confidence") == "partial_visual_check"),
        }
        for s in args.sensors
    }, "out": str(args.out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
