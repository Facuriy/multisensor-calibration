#!/usr/bin/env python3
"""Small RAW thermal checker diagnostic for extracted 20260623 frames."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from thermal_checker import detect_sb
from thermal_checker_raw import load_thermal_raw, preprocess_raw_variants, robust01


DEFAULT_GUIDED = Path("runs/calibration_20260623_thermal_guided_panel/thermal_guided_detection_summary.json")
DEFAULT_FRAMES = Path("runs/calibration_20260623_thermal_raw_smoke_extract/metadata/frames.csv")
DEFAULT_OUT = Path("runs/calibration_20260623_thermal_raw_checker_test")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_guided_row(guided: dict[str, Any], bag_id: str) -> dict[str, Any]:
    for row in guided.get("results", []):
        if Path(str(row.get("bag", ""))).stem == bag_id:
            return row
    raise KeyError(f"no guided row for {bag_id}")


def expand_box(box: list[float], shape_hw: tuple[int, int], frac: float = 0.16, pad: int = 18) -> tuple[int, int, int, int]:
    h, w = shape_hw
    x0, y0, x1, y1 = [float(v) for v in box]
    px = max(pad, int((x1 - x0) * frac))
    py = max(pad, int((y1 - y0) * frac))
    return (
        max(0, int(np.floor(x0)) - px),
        max(0, int(np.floor(y0)) - py),
        min(w, int(np.ceil(x1)) + px),
        min(h, int(np.ceil(y1)) + py),
    )


def u8_preview(img: np.ndarray) -> np.ndarray:
    return (robust01(img, 1, 99) * 255.0).astype(np.uint8)


def quick_raw_detect(raw_crop: np.ndarray) -> dict[str, Any]:
    patterns = [(9, 5), (9, 6), (8, 5)]
    best: dict[str, Any] | None = None
    for variant_name, variant in preprocess_raw_variants(raw_crop)[:3]:
        for pattern in patterns:
            corners, sweep = detect_sb(
                variant,
                pattern,
                scales=(1.0, 2.0),
                rotations=(0, 8, -8),
                try_invert=True,
            )
            if corners is None:
                continue
            pts = corners.reshape(-1, 2).astype(np.float32)
            area = float(cv2.contourArea(cv2.convexHull(pts)))
            score = area + 1500.0 * pattern[0] * pattern[1]
            if best is None or score > best["score"]:
                best = {
                    "detected": True,
                    "pattern_internal_corners": list(pattern),
                    "variant": variant_name,
                    "sweep": sweep,
                    "score": score,
                    "corners_px_crop": pts.tolist(),
                }
    return best if best is not None else {"detected": False}


def tile(img: np.ndarray, title: str, size: tuple[int, int] = (260, 230)) -> np.ndarray:
    tw, th = size
    canvas = np.full((th, tw, 3), 245, dtype=np.uint8)
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        bgr = img
    h, w = bgr.shape[:2]
    scale = min(tw / max(1, w), (th - 28) / max(1, h))
    resized = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    x = (tw - resized.shape[1]) // 2
    y = 28 + (th - 28 - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    cv2.putText(canvas, title[:30], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 2, cv2.LINE_AA)
    return canvas


def draw_case(raw: np.ndarray, crop_box: tuple[int, int, int, int], det: dict[str, Any], out_path: Path, label: str) -> None:
    x0, y0, x1, y1 = crop_box
    crop = raw[y0:y1, x0:x1]
    variants = preprocess_raw_variants(crop)
    base = u8_preview(crop)
    ff = variants[0][1]
    overlay = cv2.cvtColor(ff, cv2.COLOR_GRAY2BGR)
    if det.get("detected"):
        pts = np.asarray(det["corners_px_crop"], dtype=np.int32)
        for p in pts:
            cv2.circle(overlay, tuple(p), 2, (0, 255, 0), -1, cv2.LINE_AA)
        hull = cv2.convexHull(pts.astype(np.float32)).astype(np.int32)
        cv2.polylines(overlay, [hull], True, (0, 255, 0), 2, cv2.LINE_AA)
    page = np.hstack(
        [
            tile(base, f"{label} raw stretch"),
            tile(ff, "raw flatfield+clahe16"),
            tile(overlay, f"detect={det.get('detected')}"),
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), page)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-csv", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--guided-summary", type=Path, default=DEFAULT_GUIDED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sensor", choices=["thermal_raw", "thermal_c"], default="thermal_raw")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    guided = load_json(args.guided_summary)
    rows = list(csv.DictReader(args.frames_csv.open(encoding="utf-8")))
    raw_rows = [r for r in rows if r.get("sensor") == args.sensor]
    results = []
    pages = []
    for row in raw_rows:
        bag_id = row["bag_id"]
        guide = find_guided_row(guided, bag_id)
        raw_path = args.frames_csv.parents[1] / row["raw_npy_path"]
        raw = load_thermal_raw(raw_path)
        box = guide.get("guide_box_xyxy") or guide.get("panel_box_xyxy")
        crop_box = expand_box([float(v) for v in box], raw.shape[:2])
        det = quick_raw_detect(raw[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]])
        out_case = args.out_dir / "cases" / f"{bag_id}_{int(row['frame_index']):06d}.jpg"
        draw_case(raw, crop_box, det, out_case, f"{bag_id} f{row['frame_index']}")
        rec = {
            "sensor": row.get("sensor"),
            "bag_id": bag_id,
            "frame_index": int(row["frame_index"]),
            "stamp_ns": int(row["stamp_ns"]),
            "raw_path": str(raw_path),
            "raw_min": float(np.nanmin(raw)),
            "raw_max": float(np.nanmax(raw)),
            "raw_mean": float(np.nanmean(raw)),
            "roi_xyxy": list(crop_box),
            **{k: v for k, v in det.items() if k != "corners_px_crop"},
            "review_path": str(out_case),
        }
        results.append(rec)
        img = cv2.imread(str(out_case), cv2.IMREAD_COLOR)
        if img is not None:
            pages.append(img)

    if pages:
        page = np.vstack(pages)
        page_path = args.out_dir / "thermal_raw_checker_page_01.jpg"
        cv2.imwrite(str(page_path), page)
        page_paths = [str(page_path)]
    else:
        page_paths = []
    summary = {
        "frames_csv": str(args.frames_csv),
        "guided_summary": str(args.guided_summary),
        "sensor": args.sensor,
        "n_cases": len(results),
        "detected": sum(1 for r in results if r.get("detected")),
        "pages": page_paths,
        "results": results,
    }
    with (args.out_dir / "thermal_raw_checker_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (args.out_dir / "thermal_raw_checker_table.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["sensor", "bag_id", "frame_index", "detected", "pattern_internal_corners", "variant", "sweep", "score", "raw_min", "raw_max", "raw_mean", "review_path"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fields})
    print(f"Cases: {summary['n_cases']} detected: {summary['detected']}")
    print(f"Wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
