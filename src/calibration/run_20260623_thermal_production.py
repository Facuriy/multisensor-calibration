#!/usr/bin/env python3
"""Production thermal checker detection for the 20260623 calibration bags.

This script works on local bag files visible to Python. On this Windows setup,
PowerShell can see mapped drive ``X:`` while Python sometimes cannot, so use a
local cache directory for bag files when processing network-drive bags.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.thermal_checker_raw import preprocess_raw_variants, robust01  # noqa: E402


MANIFEST = Path("data/calibration/new_session/20260623/bag_manifest_20260623.csv")
GUIDED = Path("runs/calibration_20260623_thermal_guided_panel/thermal_guided_detection_summary.json")
OUT = Path("runs/calibration_20260623_thermal_production")

TOPICS = {
    "thermal_raw": "/ssf/thermalgrabber_ros/image_mono16",
    "thermal_c": "/ssf/thermalgrabber_ros/image_deg_celsius",
}


@dataclass
class FrameCandidate:
    sensor: str
    frame_index: int
    stamp_ns: int
    encoding: str
    image: np.ndarray
    crop: np.ndarray
    roi_xyxy: tuple[int, int, int, int]
    quality: float


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_manifest(path: Path, include_nondefault: bool = False) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    out = []
    for row in rows:
        if row.get("label_norm") == "test":
            continue
        if not include_nondefault and row.get("include_default") != "yes":
            continue
        out.append(row)
    return out


def guided_by_label(path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(path)
    return {str(r.get("label_norm")): r for r in data.get("results", [])}


def decode_ros_image(msg: Any) -> np.ndarray | None:
    enc = str(msg.encoding).lower().strip()
    h, w = int(msg.height), int(msg.width)
    raw = bytes(msg.data)
    if enc in ("mono16", "16uc1", "16sc1"):
        dtype = np.int16 if enc == "16sc1" else np.uint16
        return np.frombuffer(raw, dtype=dtype).reshape(h, w).copy()
    if enc == "32fc1":
        return np.frombuffer(raw, dtype=np.float32).reshape(h, w).copy()
    if enc == "64fc1":
        return np.frombuffer(raw, dtype=np.float64).reshape(h, w).astype(np.float32)
    return None


def expand_box(box: list[float], shape_hw: tuple[int, int], frac: float = 0.16, pad: int = 18) -> tuple[int, int, int, int]:
    h, w = shape_hw
    x0, y0, x1, y1 = [float(v) for v in box]
    px = max(pad, int((x1 - x0) * frac))
    py = max(pad, int((y1 - y0) * frac))
    return (
        max(0, int(math.floor(x0)) - px),
        max(0, int(math.floor(y0)) - py),
        min(w, int(math.ceil(x1)) + px),
        min(h, int(math.ceil(y1)) + py),
    )


def frame_quality(raw_crop: np.ndarray) -> float:
    v = (robust01(raw_crop, 1, 99) * 255.0).astype(np.uint8)
    gx = cv2.Sobel(v, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3)
    grad = float(np.mean(np.abs(gx) + np.abs(gy)))
    contrast = float(np.std(v))
    # Penalize nearly blank or almost fully saturated crops.
    sat = float(np.mean((v <= 2) | (v >= 253)))
    quality = contrast + 0.35 * grad - 35.0 * sat
    return quality


FAST_SB_FLAGS = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY


def score_points(pts: np.ndarray, pattern: tuple[int, int]) -> float:
    area = float(cv2.contourArea(cv2.convexHull(pts.astype(np.float32))))
    full_bonus = 20000.0 if pattern in ((9, 6), (9, 5)) else 0.0
    return area + full_bonus + 1500.0 * pattern[0] * pattern[1]


def remap_scaled(corners: np.ndarray, scale: float) -> np.ndarray:
    pts = corners.reshape(-1, 2).astype(np.float32)
    if scale != 1.0:
        pts /= scale
    return pts


def detect_sb_fast(u8: np.ndarray, pattern: tuple[int, int]) -> tuple[np.ndarray | None, str | None]:
    for scale in (1.0, 2.0):
        work = cv2.resize(u8, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1.0 else u8
        for inverted, img in ((False, work), (True, cv2.bitwise_not(work))):
            ok, corners = cv2.findChessboardCornersSB(img, pattern, flags=FAST_SB_FLAGS)
            if ok and corners is not None:
                return remap_scaled(corners, scale), f"fast_s{scale:g}_{'inv' if inverted else 'pos'}"
    return None, None


def detect_classic_fast(u8: np.ndarray, pattern: tuple[int, int]) -> tuple[np.ndarray | None, str | None]:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    for inverted, img in ((False, u8), (True, cv2.bitwise_not(u8))):
        ok, corners = cv2.findChessboardCorners(img, pattern, flags=flags)
        if not ok or corners is None:
            continue
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-4)
        cv2.cornerSubPix(img, corners, (9, 9), (-1, -1), term)
        return corners.reshape(-1, 2).astype(np.float32), f"classic_{'inv' if inverted else 'pos'}"
    return None, None


def detect_in_crop(raw_crop: np.ndarray) -> dict[str, Any]:
    patterns = [(9, 5), (9, 6), (8, 5)]
    best: dict[str, Any] | None = None
    for variant_name, variant in preprocess_raw_variants(raw_crop)[:3]:
        for pattern in patterns:
            for detector_name, fn in (("SB", detect_sb_fast), ("classic", detect_classic_fast)):
                pts, sweep = fn(variant, pattern)
                if pts is None:
                    continue
                score = score_points(pts, pattern)
                if best is None or score > float(best["score"]):
                    best = {
                        "detected": True,
                        "pattern_internal_corners": list(pattern),
                        "variant": variant_name,
                        "sweep": f"{detector_name}_{sweep}",
                        "score": score,
                        "corners_px_crop": pts.tolist(),
                    }
                if pattern in ((9, 5), (9, 6)):
                    return best
    return best if best is not None else {"detected": False, "score": 0.0}


def collect_candidates(
    bag_path: Path,
    label: str,
    guided_row: dict[str, Any],
    max_quality_frames: int,
    sensors: list[str],
    frame_step: int,
) -> list[FrameCandidate]:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    box = guided_row.get("guide_box_xyxy") or guided_row.get("panel_box_xyxy")
    if not box:
        raise RuntimeError(f"{label}: no guided thermal ROI")

    top_frames: dict[str, list[FrameCandidate]] = {sensor: [] for sensor in sensors}
    with Reader(bag_path) as reader:
        available = {c.topic: c for c in reader.connections}
        wanted_topics = {sensor: TOPICS[sensor] for sensor in sensors}
        selected = [available[t] for t in wanted_topics.values() if t in available]
        topic_to_sensor = {topic: sensor for sensor, topic in wanted_topics.items()}
        counters = {sensor: 0 for sensor in sensors}
        for conn, ts, raw in reader.messages(connections=selected):
            sensor = topic_to_sensor[conn.topic]
            idx = counters[sensor]
            counters[sensor] += 1
            if frame_step > 1 and idx % frame_step != 0:
                continue
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            img = decode_ros_image(msg)
            if img is None:
                continue
            roi = expand_box([float(v) for v in box], img.shape[:2])
            x0, y0, x1, y1 = roi
            crop = img[y0:y1, x0:x1].astype(np.float32)
            if crop.size == 0:
                continue
            quality = frame_quality(crop)
            candidate = FrameCandidate(
                sensor=sensor,
                frame_index=idx,
                stamp_ns=int(ts),
                encoding=str(getattr(msg, "encoding", "")),
                image=img.astype(np.float32),
                crop=crop,
                roi_xyxy=roi,
                quality=quality,
            )
            bucket = top_frames[sensor]
            bucket.append(candidate)
            bucket.sort(key=lambda f: f.quality, reverse=True)
            del bucket[max_quality_frames:]

    selected_frames: list[FrameCandidate] = []
    for sensor in sensors:
        selected_frames.extend(top_frames.get(sensor, []))
    return selected_frames


def draw_candidate(candidate: FrameCandidate, det: dict[str, Any], out_path: Path, title: str) -> None:
    raw_u8 = (robust01(candidate.crop, 1, 99) * 255.0).astype(np.uint8)
    ff = preprocess_raw_variants(candidate.crop)[0][1]
    overlay = cv2.cvtColor(ff, cv2.COLOR_GRAY2BGR)
    if det.get("detected"):
        pts = np.asarray(det["corners_px_crop"], dtype=np.int32)
        for p in pts:
            cv2.circle(overlay, tuple(p), 2, (0, 255, 0), -1, cv2.LINE_AA)
        hull = cv2.convexHull(pts.astype(np.float32)).astype(np.int32)
        cv2.polylines(overlay, [hull], True, (0, 255, 0), 2, cv2.LINE_AA)
    parts = [
        fit_tile(raw_u8, (300, 230), f"{title} raw"),
        fit_tile(ff, (300, 230), "flatfield+clahe"),
        fit_tile(overlay, (300, 230), f"detect={det.get('detected')}"),
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.hstack(parts))


def fit_tile(img: np.ndarray, size: tuple[int, int], label: str) -> np.ndarray:
    width, height = size
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    h, w = bgr.shape[:2]
    scale = min(width / max(w, 1), (height - 28) / max(h, 1))
    small = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    x = (width - small.shape[1]) // 2
    y = 28 + (height - 28 - small.shape[0]) // 2
    canvas[y : y + small.shape[0], x : x + small.shape[1]] = small
    cv2.putText(canvas, label[:42], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 20, 20), 2, cv2.LINE_AA)
    return canvas


def process_bag(
    row: dict[str, str],
    bag_path: Path,
    guided_rows: dict[str, dict[str, Any]],
    out_dir: Path,
    max_quality_frames: int,
    sensors: list[str],
    frame_step: int,
) -> dict[str, Any]:
    label = row["label_norm"]
    if label not in guided_rows:
        return {"label_norm": label, "bag": row["bag"], "processed": False, "reason": "missing_guided_roi"}
    candidates = collect_candidates(bag_path, label, guided_rows[label], max_quality_frames, sensors, frame_step)
    detections = []
    for cand in candidates:
        det = detect_in_crop(cand.crop)
        rec = {
            "sensor": cand.sensor,
            "frame_index": cand.frame_index,
            "stamp_ns": cand.stamp_ns,
            "encoding": cand.encoding,
            "quality": cand.quality,
            "roi_xyxy": list(cand.roi_xyxy),
            "raw_min": float(np.nanmin(cand.image)),
            "raw_max": float(np.nanmax(cand.image)),
            "raw_mean": float(np.nanmean(cand.image)),
            **{k: v for k, v in det.items() if k != "corners_px_crop"},
        }
        if det.get("detected"):
            pts = np.asarray(det["corners_px_crop"], dtype=np.float64)
            x0, y0, _x1, _y1 = cand.roi_xyxy
            pts[:, 0] += x0
            pts[:, 1] += y0
            rec["corners_px"] = pts.tolist()
        case_path = out_dir / "bags" / label / f"{cand.sensor}_f{cand.frame_index:04d}.jpg"
        draw_candidate(cand, det, case_path, f"{label} {cand.sensor} f{cand.frame_index}")
        rec["review_path"] = str(case_path)
        detections.append(rec)

    detected = [d for d in detections if d.get("detected")]
    detected.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
    best = detected[0] if detected else None
    return {
        "label_norm": label,
        "bag": row["bag"],
        "bag_path": str(bag_path),
        "processed": True,
        "n_candidates": len(candidates),
        "detected": bool(best),
        "best": best,
        "detections": detections,
        "reason": "" if best else "no_checker_detected_in_top_quality_frames",
    }


def make_contact_sheet(results: list[dict[str, Any]], out_dir: Path) -> str | None:
    imgs = []
    for row in results:
        best = row.get("best") or {}
        path = best.get("review_path")
        if not path:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        imgs.append(cv2.resize(img, (900, 230), interpolation=cv2.INTER_AREA))
    if not imgs:
        return None
    pages = []
    for start in range(0, len(imgs), 4):
        batch = imgs[start : start + 4]
        while len(batch) < 4:
            batch.append(np.full_like(batch[0], 245))
        pages.append(np.vstack(batch))
    page_paths = []
    for i, page in enumerate(pages, 1):
        p = out_dir / f"thermal_production_best_page_{i:02d}.jpg"
        cv2.imwrite(str(p), page)
        page_paths.append(str(p))
    return page_paths[0] if page_paths else None


def write_outputs(results: list[dict[str, Any]], out_dir: Path, manifest: Path, guided: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    page = make_contact_sheet(results, out_dir)
    summary = {
        "manifest": str(manifest),
        "guided_summary": str(guided),
        "n_rows": len(results),
        "processed": sum(1 for r in results if r.get("processed")),
        "detected": sum(1 for r in results if r.get("detected")),
        "contact_sheet": page,
        "results": results,
    }
    with (out_dir / "thermal_production_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fields = [
        "label_norm",
        "bag",
        "processed",
        "detected",
        "reason",
        "sensor",
        "frame_index",
        "pattern",
        "variant",
        "sweep",
        "score",
        "quality",
        "review_path",
    ]
    with (out_dir / "thermal_production_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            best = row.get("best") or {}
            writer.writerow(
                {
                    "label_norm": row.get("label_norm"),
                    "bag": row.get("bag"),
                    "processed": row.get("processed"),
                    "detected": row.get("detected"),
                    "reason": row.get("reason"),
                    "sensor": best.get("sensor", ""),
                    "frame_index": best.get("frame_index", ""),
                    "pattern": "x".join(map(str, best.get("pattern_internal_corners", []))) if best else "",
                    "variant": best.get("variant", ""),
                    "sweep": best.get("sweep", ""),
                    "score": best.get("score", ""),
                    "quality": best.get("quality", ""),
                    "review_path": best.get("review_path", ""),
                }
            )

    compatible = {"results": []}
    for row in results:
        best = row.get("best") or {}
        compatible["results"].append(
            {
                "label_norm": row.get("label_norm"),
                "bag": row.get("bag"),
                "checker": {
                    "detected": bool(row.get("detected")),
                    "pattern_internal_corners": best.get("pattern_internal_corners", []),
                    "corners_px": best.get("corners_px", []),
                    "method": f"{best.get('sensor', '')}_{best.get('variant', '')}_{best.get('sweep', '')}",
                    "score": best.get("score", 0.0),
                    "roi_xyxy": best.get("roi_xyxy", []),
                    "frame_index": best.get("frame_index", ""),
                    "stamp_ns": best.get("stamp_ns", ""),
                },
            }
        )
    compatible["checker_detected"] = sum(1 for r in compatible["results"] if r["checker"]["detected"])
    compatible["source"] = str(out_dir / "thermal_production_summary.json")
    with (out_dir / "thermal_production_candidates_for_to_vis.json").open("w", encoding="utf-8") as f:
        json.dump(compatible, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--guided-summary", type=Path, default=GUIDED)
    parser.add_argument("--bag-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--include-nondefault", action="store_true")
    parser.add_argument("--only-label", default="")
    parser.add_argument("--max-quality-frames", type=int, default=4)
    parser.add_argument("--sensors", nargs="+", choices=sorted(TOPICS), default=["thermal_raw", "thermal_c"])
    parser.add_argument("--frame-step", type=int, default=1)
    args = parser.parse_args()

    rows = read_manifest(args.manifest, include_nondefault=args.include_nondefault)
    if args.only_label:
        rows = [r for r in rows if r.get("label_norm") == args.only_label or r.get("bag") == args.only_label]
    guided_rows = guided_by_label(args.guided_summary)
    results = []
    for row in rows:
        bag_path = args.bag_cache / row["bag"]
        if not bag_path.exists():
            results.append(
                {
                    "label_norm": row.get("label_norm"),
                    "bag": row.get("bag"),
                    "processed": False,
                    "detected": False,
                    "reason": f"missing_local_bag:{bag_path}",
                }
            )
            continue
        print(f"Processing {row['label_norm']} {bag_path}")
        results.append(
            process_bag(
                row,
                bag_path,
                guided_rows,
                args.out_dir,
                args.max_quality_frames,
                args.sensors,
                max(1, args.frame_step),
            )
        )

    write_outputs(results, args.out_dir, args.manifest, args.guided_summary)
    print(f"Processed {sum(1 for r in results if r.get('processed'))}/{len(results)}")
    print(f"Detected {sum(1 for r in results if r.get('detected'))}/{len(results)}")
    print(f"Wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
