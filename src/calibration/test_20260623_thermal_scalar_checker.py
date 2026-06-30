#!/usr/bin/env python3
"""Evaluate colormap inversion for 20260623 thermal checker candidates.

This is a bounded diagnostic, not the production detector. It tests Claude's
suggested scalar recovery on guided thermal ROIs and writes visual before/after
pages plus a small JSON/CSV summary.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from thermal_checker import detect_sb, preprocess_variants, recover_scalar, to_u8


DEFAULT_GUIDED = Path("runs/calibration_20260623_thermal_guided_panel/thermal_guided_detection_summary.json")
DEFAULT_OUT = Path("runs/calibration_20260623_thermal_scalar_checker_test")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def choose_checker_box(row: dict[str, Any]) -> list[float]:
    # The projected guide is usually tighter and less biased by thermal threshold
    # leakage into the background than the yellow panel component.
    guide = row.get("guide_box_xyxy")
    if guide:
        return [float(v) for v in guide]
    return [float(v) for v in row["panel_box_xyxy"]]


def quick_scalar_detect(scalar_crop: np.ndarray) -> dict[str, Any]:
    patterns = [(9, 6), (9, 5), (8, 5)]
    best: dict[str, Any] | None = None
    for variant_name, variant in preprocess_variants(scalar_crop)[:3]:
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


def tile(img: np.ndarray, title: str, size: tuple[int, int] = (300, 250)) -> np.ndarray:
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
    cv2.putText(canvas, title[:38], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 2, cv2.LINE_AA)
    return canvas


def draw_case(row: dict[str, Any], out_path: Path) -> dict[str, Any]:
    img = cv2.imread(row["thermal_path"], cv2.IMREAD_COLOR)
    if img is None:
        return {"label_norm": row["label_norm"], "detected": False, "reason": "missing_image"}
    scalar, scalar_info = recover_scalar(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap="auto")
    x0, y0, x1, y1 = expand_box(choose_checker_box(row), img.shape[:2])
    crop = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scalar_crop = scalar[y0:y1, x0:x1]
    variants = preprocess_variants(scalar_crop)
    best_u8 = variants[0][1]
    det = quick_scalar_detect(scalar_crop)

    overlay = cv2.cvtColor(best_u8, cv2.COLOR_GRAY2BGR)
    if det.get("detected"):
        pts = np.asarray(det["corners_px_crop"], dtype=np.int32)
        for p in pts:
            cv2.circle(overlay, tuple(p), 2, (0, 255, 0), -1, cv2.LINE_AA)
        hull = cv2.convexHull(pts.astype(np.float32)).astype(np.int32)
        cv2.polylines(overlay, [hull], True, (0, 255, 0), 2, cv2.LINE_AA)

    page = np.hstack(
        [
            tile(crop, f"{row['label_norm']} thermal"),
            tile(to_u8(gray), "naive gray"),
            tile(to_u8(scalar_crop), f"recovered {scalar_info['cmap']}"),
            tile(best_u8, "flatfield+clahe"),
            tile(overlay, f"detect={det.get('detected')}"),
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), page)
    result = {
        "label_norm": row["label_norm"],
        "bag": row["bag"],
        "cmap": scalar_info["cmap"],
        "cmap_residual": scalar_info["residual"],
        "roi_xyxy": [x0, y0, x1, y1],
        **{k: v for k, v in det.items() if k != "corners_px_crop"},
        "review_path": str(out_path),
    }
    if det.get("detected"):
        pts_abs = np.asarray(det["corners_px_crop"], dtype=np.float64)
        pts_abs[:, 0] += x0
        pts_abs[:, 1] += y0
        result["corners_px"] = pts_abs.tolist()
    return result


def make_pages(results: list[dict[str, Any]], out_dir: Path) -> list[str]:
    imgs = []
    pages = []
    for r in results:
        p = r.get("review_path")
        if not p:
            continue
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is not None:
            imgs.append(cv2.resize(img, (1000, 200), interpolation=cv2.INTER_AREA))
    for start in range(0, len(imgs), 4):
        batch = imgs[start : start + 4]
        if not batch:
            continue
        while len(batch) < 4:
            batch.append(np.full_like(batch[0], 245))
        page = np.vstack(batch)
        out = out_dir / f"thermal_scalar_checker_page_{len(pages)+1:02d}.jpg"
        cv2.imwrite(str(out), page)
        pages.append(str(out))
    return pages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guided-summary", type=Path, default=DEFAULT_GUIDED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--mode",
        choices=["old-checkers", "all"],
        default="old-checkers",
        help="old-checkers tests the 8 previous thermal checker candidates only.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(args.guided_summary)
    rows = data["results"]
    if args.mode == "old-checkers":
        rows = [r for r in rows if (r.get("checker") or {}).get("detected")]

    results = []
    for row in rows:
        out = args.out_dir / "cases" / f"{row['label_norm']}.jpg"
        results.append(draw_case(row, out))

    pages = make_pages(results, args.out_dir)
    summary = {
        "source": str(args.guided_summary),
        "mode": args.mode,
        "n_cases": len(results),
        "detected": sum(1 for r in results if r.get("detected")),
        "pages": pages,
        "results": results,
    }
    with (args.out_dir / "thermal_scalar_checker_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    compatible = load_json(args.guided_summary)
    by_label = {r["label_norm"]: r for r in results if r.get("detected")}
    for row in compatible.get("results", []):
        hit = by_label.get(row.get("label_norm"))
        if not hit:
            continue
        row["checker"] = {
            "detected": True,
            "pattern_internal_corners": hit["pattern_internal_corners"],
            "corners_px": hit["corners_px"],
            "method": f"scalar_{hit.get('variant')}_{hit.get('sweep')}",
            "score": hit.get("score"),
            "roi_xyxy": hit.get("roi_xyxy"),
            "cmap": hit.get("cmap"),
            "cmap_residual": hit.get("cmap_residual"),
        }
    compatible["checker_detected"] = sum(1 for r in compatible.get("results", []) if (r.get("checker") or {}).get("detected"))
    compatible["scalar_checker_source"] = str(args.out_dir / "thermal_scalar_checker_summary.json")
    with (args.out_dir / "thermal_guided_detection_summary_scalar_candidates.json").open("w", encoding="utf-8") as f:
        json.dump(compatible, f, indent=2)

    with (args.out_dir / "thermal_scalar_checker_table.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["label_norm", "bag", "cmap", "cmap_residual", "detected", "pattern_internal_corners", "variant", "sweep", "score", "review_path"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fields})
    print(f"Cases: {summary['n_cases']} detected: {summary['detected']}")
    print(f"Wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
