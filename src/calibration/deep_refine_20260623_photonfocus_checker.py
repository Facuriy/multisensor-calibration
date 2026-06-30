#!/usr/bin/env python3
"""Deep VIS/NIR checker refinement for 2026-06-23 calibration bags."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.refine_20260623_multisensor_detection import (  # noqa: E402
    TOPICS,
    Detection,
    canonicalize_and_score,
    detection_to_json,
    detect_panel_by_contrast,
    draw_detection,
    gray_candidates,
    panel_from_checker,
    read_sampled_messages,
)
from src.extraction.extract_all_bag_images import decode_ros_image  # noqa: E402


PATTERNS = [(9, 6), (9, 5), (8, 5)]


def canonicalize_pattern(corners: np.ndarray, pattern: tuple[int, int], shape: tuple[int, int]) -> tuple[np.ndarray, float, float, bool]:
    cols, rows = pattern
    pts = corners.reshape(rows, cols, 2).astype(np.float32)
    row_vec = pts[-1, 0] - pts[0, 0]
    col_vec = pts[0, -1] - pts[0, 0]
    cross = float(col_vec[0] * row_vec[1] - col_vec[1] * row_vec[0])
    if cross < 0:
        pts = pts[::-1, :, :]
        row_vec = pts[-1, 0] - pts[0, 0]
        col_vec = pts[0, -1] - pts[0, 0]
        cross = float(col_vec[0] * row_vec[1] - col_vec[1] * row_vec[0])
    angle = float(np.degrees(np.arctan2(float(col_vec[1]), float(col_vec[0]))))
    diffs_x = np.linalg.norm(np.diff(pts, axis=1), axis=2)
    diffs_y = np.linalg.norm(np.diff(pts, axis=0), axis=2)
    spacing = float(np.median(np.r_[diffs_x.reshape(-1), diffs_y.reshape(-1)]))
    bbox = [pts[:, :, 0].min(), pts[:, :, 1].min(), pts[:, :, 0].max(), pts[:, :, 1].max()]
    bbox_area = max(1.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
    quality = spacing + 80.0 * min(1.0, bbox_area / max(1.0, shape[0] * shape[1]))
    return pts.reshape(-1, 2), angle, quality, bool(cross > 0)


def deep_try(gray: np.ndarray, projected_box: list[float] | None) -> tuple[np.ndarray | None, tuple[int, int] | None, str]:
    sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS
    h, w = gray.shape[:2]
    crops: list[tuple[int, int, int, int, str]] = []
    if projected_box is not None:
        x0, y0, x1, y1 = projected_box
        pad_x = max(25, int((x1 - x0) * 0.70))
        pad_y = max(25, int((y1 - y0) * 0.70))
        crops.append((max(0, int(x0) - pad_x), max(0, int(y0) - pad_y), min(w, int(x1) + pad_x), min(h, int(y1) + pad_y), "guided"))
    crops.append((0, 0, w, h, "full"))
    seen = set()
    uniq = []
    for crop in crops:
        if crop[:4] not in seen and crop[2] - crop[0] > 50 and crop[3] - crop[1] > 35:
            uniq.append(crop)
            seen.add(crop[:4])
    for x0, y0, x1, y1, crop_name in uniq:
        crop = gray[y0:y1, x0:x1]
        for scale in (1.0, 1.25):
            resized = cv2.resize(
                crop,
                (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
            )
            for inv in (False, True):
                src = 255 - resized if inv else resized
                for pattern in PATTERNS:
                    # Try classic first: often works on Photonfocus and is faster.
                    ok, corners = cv2.findChessboardCorners(src, pattern, flags=classic_flags)
                    alg = "classic"
                    if not ok:
                        ok, corners = cv2.findChessboardCornersSB(src, pattern, flags=sb_flags)
                        alg = "sb_exh"
                    if ok and corners is not None:
                        pts = corners.reshape(-1, 2).astype(np.float32) / scale
                        pts[:, 0] += x0
                        pts[:, 1] += y0
                        return pts, pattern, f"{alg}_p{pattern[0]}x{pattern[1]}_{crop_name}_s{scale:g}_{'inv' if inv else 'norm'}"
    return None, None, ""


def deep_detect(sensor: str, messages, projected_box: list[float] | None, max_frames: int) -> Detection:
    if not messages:
        return Detection(False, False, "missing_topic", None, None, None, None, None, False, 0.0, None, "missing_topic")
    best: Detection | None = None
    for idx, (stamp, msg) in enumerate(messages[:max_frames]):
        img = decode_ros_image(msg)
        if img is None:
            continue
        for variant_name, gray in gray_candidates(sensor, img)[:6]:
            corners, pattern, method = deep_try(gray, projected_box)
            if corners is None:
                continue
            assert pattern is not None
            corners, angle, quality, rotation_ok = canonicalize_pattern(corners, pattern, gray.shape[:2])
            det = Detection(
                checker_detected=True,
                panel_detected=True,
                method=f"{variant_name}_{method}",
                frame_index=idx,
                stamp_ns=stamp,
                corners_px=corners,
                panel_box_xyxy=panel_from_checker(corners),
                orientation_deg=angle,
                rotation_ok=rotation_ok,
                quality=quality,
                source_shape=gray.shape[:2],
                processed_image=gray,
            )
            det_json_pattern = pattern
            if best is None or det.quality > best.quality:
                best = det
                best.pattern_internal_corners = det_json_pattern  # type: ignore[attr-defined]
                if rotation_ok:
                    return best
    if best is not None:
        return best
    img = decode_ros_image(messages[0][1])
    gray = gray_candidates(sensor, img)[0][1] if img is not None else None
    if gray is not None:
        panel_box, method, score = detect_panel_by_contrast(gray, projected_box, sensor)
        if panel_box is not None:
            return Detection(False, True, method, 0, messages[0][0], None, panel_box, None, False, score, gray.shape[:2], "panel_only", gray)
    return Detection(False, False, "not_found", None, None, None, None, None, False, 0.0, None, "not_found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest_rows = {r["label_norm"]: r for r in csv.DictReader(args.manifest.open(encoding="utf-8"))}
    base = json.loads(args.base_summary.read_text(encoding="utf-8"))
    results = []
    for i, res in enumerate(base["results"], start=1):
        label = res["label_norm"]
        row = manifest_rows[label]
        bag = Path(row["bag_path"])
        print(f"[{i}/{len(base['results'])}] {label}")
        messages = read_sampled_messages(bag, {"vis": TOPICS["vis"], "nir": TOPICS["nir"]}, row, args.max_frames)
        out_bag = args.out / "bags" / bag.stem
        out_bag.mkdir(parents=True, exist_ok=True)
        refined = {}
        for sensor in ("vis", "nir"):
            projected = res.get("projected_boxes_from_rgb", {}).get(sensor)
            # If the fast pass already found a checker, still re-run deep and
            # keep the better-quality result. This normalizes methods.
            det = deep_detect(sensor, messages[sensor], projected, args.max_frames)
            det_json = detection_to_json(det)
            det_json["pattern_internal_corners"] = list(getattr(det, "pattern_internal_corners", (9, 6))) if det.checker_detected else None
            refined[sensor] = det_json
            draw_detection(det, sensor, out_bag / f"{sensor}_deep_detection.jpg")
        results.append({**row, "detections": refined})

    summary = {
        "n_bags": len(results),
        "checker_counts": {
            "vis": sum(1 for r in results if r["detections"]["vis"]["checker_detected"]),
            "nir": sum(1 for r in results if r["detections"]["nir"]["checker_detected"]),
        },
        "panel_counts": {
            "vis": sum(1 for r in results if r["detections"]["vis"]["panel_detected"]),
            "nir": sum(1 for r in results if r["detections"]["nir"]["panel_detected"]),
        },
        "rotation_warnings": {
            "vis": [r["label_norm"] for r in results if r["detections"]["vis"]["checker_detected"] and not r["detections"]["vis"]["rotation_ok"]],
            "nir": [r["label_norm"] for r in results if r["detections"]["nir"]["checker_detected"] and not r["detections"]["nir"]["rotation_ok"]],
        },
        "results": results,
    }
    (args.out / "deep_photonfocus_detection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    flat = []
    for r in results:
        rec = {"bag": Path(r["bag_path"]).name, "label_norm": r["label_norm"]}
        for sensor in ("vis", "nir"):
            d = r["detections"][sensor]
            rec[f"{sensor}_checker"] = d["checker_detected"]
            rec[f"{sensor}_panel"] = d["panel_detected"]
            rec[f"{sensor}_method"] = d["method"]
            rec[f"{sensor}_rot_ok"] = d["rotation_ok"]
            rec[f"{sensor}_quality"] = d["quality"]
        flat.append(rec)
    with (args.out / "deep_photonfocus_detection_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
