#!/usr/bin/env python3
"""Build final/candidate multisensor calibration package for 2026-06-23.

This merges:
  * RGB checker detections and intrinsics.
  * VIS/NIR direct detections.
  * NIR/thermal model-based grid recoveries from all-frame campaign.
  * Ouster->RGB multipose physical extrinsic.

Outputs are explicit about confidence.  Model-based grid observations are
useful, but lower trust than subpixel OpenCV detections.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.calibrate_rgb_intrinsics_from_detections import (  # noqa: E402
    ViewDetection,
    choose_one_per_label,
    collect_refined,
    collect_scan,
    reprojection_errors,
)
from src.registration.refine_20260623_vis_homographies import (  # noqa: E402
    best_subgrid_pair,
    fit_homography,
    reprojection_metrics,
)


RGB_PATTERN = (9, 6)
SQUARE_SIZE_M = 0.04

DEFAULT_OUT = Path("runs/calibration_20260623_final_multisensor_calibration")
SESSION_DIR = Path("data/calibration/new_session/20260623")


@dataclass
class Candidate:
    label: str
    bag: str
    sensor: str
    pattern: tuple[int, int]
    corners: np.ndarray
    method: str
    source: str
    confidence: str
    frame_index: int | None = None
    stamp_ns: int | None = None
    review_path: str = ""


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    return value


def object_points(pattern: tuple[int, int], square_size_m: float = SQUARE_SIZE_M) -> np.ndarray:
    cols, rows = pattern
    obj = np.zeros((cols * rows, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj[:, :2] = grid * float(square_size_m)
    return obj


def object_points_with_offset(
    pattern: tuple[int, int],
    col0: int,
    row0: int,
    square_size_m: float = SQUARE_SIZE_M,
) -> np.ndarray:
    """Like object_points but shifts the origin to (col0, row0) on the board.

    When a subgrid is extracted from a full checkerboard pattern (e.g. the
    common_pattern starting at source_offset_col_row = [col0, row0]), the
    physical 3-D coordinates must start at (col0*sq, row0*sq, 0) rather
    than (0, 0, 0).  Using the wrong origin causes cv2.stereoCalibrate to
    absorb the board offset into the extrinsic translation, corrupting R and T.
    """
    cols, rows = pattern
    obj = np.zeros((cols * rows, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj[:, 0] = (grid[:, 0] + col0) * float(square_size_m)
    obj[:, 1] = (grid[:, 1] + row0) * float(square_size_m)
    return obj


def normalize_candidate(raw: Any, pattern: tuple[int, int]) -> np.ndarray | None:
    pts = np.asarray(raw, dtype=np.float32)
    if pts.shape != (pattern[0] * pattern[1], 2):
        return None
    if not np.isfinite(pts).all():
        return None
    return canonicalize_corners(pts, pattern)


def canonicalize_corners(pts: np.ndarray, pattern: tuple[int, int]) -> np.ndarray:
    """Return row-major corners with rows roughly left->right and top->bottom."""
    cols, rows = pattern
    grid = pts.reshape(rows, cols, 2).astype(np.float32)
    variants = [
        grid,
        grid[::-1, :, :],
        grid[:, ::-1, :],
        grid[::-1, ::-1, :],
    ]
    best = None
    for g in variants:
        row_vec = np.mean(g[:, -1, :] - g[:, 0, :], axis=0)
        col_vec = np.mean(g[-1, :, :] - g[0, :, :], axis=0)
        row_norm = float(np.linalg.norm(row_vec)) + 1e-6
        col_norm = float(np.linalg.norm(col_vec)) + 1e-6
        # In these calibration frames the board is not upside-down. Prefer
        # image-left to image-right rows and image-top to image-bottom columns.
        score = float(row_vec[0] / row_norm + col_vec[1] / col_norm)
        # Keep a right-handed image grid when possible.
        cross = float(row_vec[0] * col_vec[1] - row_vec[1] * col_vec[0])
        if cross > 0:
            score += 0.25
        if best is None or score > best[0]:
            best = (score, g)
    assert best is not None
    return best[1].reshape(-1, 2).astype(np.float32)


def load_rgb_by_label(scan: Path, refined: Path) -> dict[str, Candidate]:
    detections = choose_one_per_label(collect_scan(scan, RGB_PATTERN) + collect_refined(refined, RGB_PATTERN))
    out: dict[str, Candidate] = {}
    for d in detections:
        out[d.label] = Candidate(
            label=d.label,
            bag=d.bag,
            sensor="rgb",
            pattern=RGB_PATTERN,
            corners=d.corners.reshape(-1, 2).astype(np.float32),
            method=d.method,
            source=d.source,
            confidence="subpixel",
            frame_index=d.frame_index,
            stamp_ns=d.stamp_ns,
        )
    return out


def load_deep_candidates(path: Path, sensor: str) -> list[Candidate]:
    data = load_json(path)
    out: list[Candidate] = []
    for row in data.get("results", []):
        label = str(row.get("label_norm") or row.get("label_raw") or row.get("bag"))
        det = row.get("detections", {}).get(sensor, {})
        if not det.get("checker_detected"):
            continue
        pattern_raw = det.get("pattern_internal_corners")
        if not pattern_raw:
            continue
        pattern = (int(pattern_raw[0]), int(pattern_raw[1]))
        corners = normalize_candidate(det.get("corners_px"), pattern)
        if corners is None:
            continue
        out.append(
            Candidate(
                label=label,
                bag=str(row.get("bag") or ""),
                sensor=sensor,
                pattern=pattern,
                corners=corners,
                method=str(det.get("method") or ""),
                source="deep_photonfocus",
                confidence="subpixel",
                frame_index=det.get("frame_index"),
                stamp_ns=det.get("stamp_ns"),
            )
        )
    return out


def load_grid_campaign(path: Path, sensor: str) -> list[Candidate]:
    data = load_json(path)
    out: list[Candidate] = []
    for row in data.get("results", []):
        det = row.get("detections", {}).get(sensor)
        if not det or det.get("confidence") != "strong":
            continue
        pattern_raw = det.get("pattern_internal_corners")
        if not pattern_raw:
            continue
        pattern = (int(pattern_raw[0]), int(pattern_raw[1]))
        corners = normalize_candidate(det.get("corners_px"), pattern)
        if corners is None:
            continue
        out.append(
            Candidate(
                label=str(row.get("label_norm")),
                bag=str(row.get("bag") or det.get("bag") or ""),
                sensor=sensor,
                pattern=pattern,
                corners=corners,
                method=str(det.get("method") or ""),
                source="all_frame_grid_model",
                confidence="model_strong",
                frame_index=det.get("frame_index"),
                stamp_ns=det.get("stamp_ns"),
                review_path=str(det.get("review_path") or ""),
            )
        )
    return out


def load_photonfocus_partial(path: Path, sensor: str) -> list[Candidate]:
    data = load_json(path)
    out: list[Candidate] = []
    for row in data.get("results", []):
        best = (row.get("detections", {}).get(sensor, {}) or {}).get("best")
        if not best:
            continue
        confidence = str(best.get("confidence") or "")
        if not confidence.startswith("subpixel"):
            continue
        pattern_raw = best.get("pattern_internal_corners")
        if not pattern_raw:
            continue
        pattern = (int(pattern_raw[0]), int(pattern_raw[1]))
        corners = normalize_candidate(best.get("corners_px"), pattern)
        if corners is None:
            continue
        out.append(
            Candidate(
                label=str(row.get("label_norm")),
                bag=str(row.get("bag") or ""),
                sensor=sensor,
                pattern=pattern,
                corners=corners,
                method=str(best.get("method") or ""),
                source="photonfocus_partial_subpixel",
                confidence="subpixel",
                frame_index=best.get("frame_index"),
                stamp_ns=best.get("stamp_ns"),
                review_path=str(best.get("review_path") or ""),
            )
        )
    return out


def load_thermal_merged(path: Path) -> list[Candidate]:
    data = load_json(path)
    out: list[Candidate] = []
    for row in data.get("results", []):
        chk = row.get("checker", {})
        if not chk.get("detected"):
            continue
        pattern_raw = chk.get("pattern_internal_corners")
        if not pattern_raw:
            continue
        pattern = (int(pattern_raw[0]), int(pattern_raw[1]))
        corners = normalize_candidate(chk.get("corners_px"), pattern)
        if corners is None:
            continue
        out.append(
            Candidate(
                label=str(row.get("label_norm")),
                bag=str(row.get("bag") or ""),
                sensor="thermal",
                pattern=pattern,
                corners=corners,
                method=str(chk.get("method") or ""),
                source=str(row.get("source_priority") or "thermal_merged"),
                confidence="subpixel_or_scalar",
                frame_index=chk.get("frame_index") if chk.get("frame_index") != "" else None,
                stamp_ns=chk.get("stamp_ns") if chk.get("stamp_ns") != "" else None,
            )
        )
    return out


def choose_candidates(candidates: list[Candidate]) -> list[Candidate]:
    priority = {
        "subpixel": 0,
        "subpixel_or_scalar": 1,
        "model_strong": 2,
    }
    grouped: dict[str, list[Candidate]] = {}
    for c in candidates:
        grouped.setdefault(c.label, []).append(c)
    chosen = []
    for label, items in sorted(grouped.items()):
        items = sorted(
            items,
            key=lambda c: (
                priority.get(c.confidence, 9),
                -float(cv2.contourArea(cv2.convexHull(c.corners.astype(np.float32)))),
                c.frame_index if c.frame_index is not None else 10**9,
            ),
        )
        chosen.append(items[0])
    return chosen


def filter_by_pattern(
    candidates: list[Candidate],
    min_cols: int,
    min_rows: int,
) -> list[Candidate]:
    """Keep detections with enough board geometry for stable calibration."""
    return [c for c in candidates if c.pattern[0] >= min_cols and c.pattern[1] >= min_rows]


def selection_summary(candidates: list[Candidate]) -> dict[str, Any]:
    patterns: dict[str, int] = {}
    sources: dict[str, int] = {}
    for c in candidates:
        patterns[f"{c.pattern[0]}x{c.pattern[1]}"] = patterns.get(f"{c.pattern[0]}x{c.pattern[1]}", 0) + 1
        sources[c.source] = sources.get(c.source, 0) + 1
    return {
        "n": len(candidates),
        "patterns": dict(sorted(patterns.items())),
        "sources": dict(sorted(sources.items())),
        "labels": [c.label for c in candidates],
    }


def initial_k(intrinsics: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(intrinsics[key]["K_initial"], dtype=np.float64)


def calibrate_intrinsics(
    sensor: str,
    candidates: list[Candidate],
    image_size_wh: tuple[int, int],
    k0: np.ndarray,
    min_views: int,
) -> dict[str, Any]:
    if len(candidates) < min_views:
        return {
            "sensor": sensor,
            "status": "insufficient_views",
            "n_views": len(candidates),
            "min_views": min_views,
        }
    objpoints = [object_points(c.pattern) for c in candidates]
    imgpoints = [c.corners.reshape(-1, 1, 2).astype(np.float32) for c in candidates]
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K3
    )
    d0 = np.zeros((5, 1), dtype=np.float64)
    rms, k, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size_wh, k0.copy(), d0.copy(), flags=flags
    )
    errors = reprojection_errors(objpoints, imgpoints, rvecs, tvecs, k, dist)
    mean_err = np.asarray([e["mean_px"] for e in errors], dtype=np.float64)
    max_err = np.asarray([e["max_px"] for e in errors], dtype=np.float64)
    med = float(np.median(mean_err))
    mad = float(np.median(np.abs(mean_err - med)))
    sigma = 1.4826 * mad if mad > 1e-9 else float(np.std(mean_err))
    mean_ok = mean_err <= max(3.0, 2.8 * med, med + 3.0 * sigma)
    max_ok = max_err <= max(10.0, 5.0 * med)  # Fix A3: reject views with outlier corners
    keep = mean_ok & max_ok
    if int(keep.sum()) < min_views:
        order = np.argsort(mean_err)
        keep[:] = False
        keep[order[:min_views]] = True
    kept = [c for c, use in zip(candidates, keep) if use]
    obj2 = [object_points(c.pattern) for c in kept]
    img2 = [c.corners.reshape(-1, 1, 2).astype(np.float32) for c in kept]
    rms2, k2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(
        obj2, img2, image_size_wh, k0.copy(), d0.copy(), flags=flags
    )
    # Fix C2: if k2 magnitude is physically implausible, retry with k2 fixed at 0.
    # Thresholds per sensor: thermal cameras have near-zero k2; VIS/NIR with k2<-8 are overfitting.
    _k2_thresh = 2.0 if sensor == "thermal" else 8.0
    if abs(float(dist2.reshape(-1)[1])) > _k2_thresh and sensor in ("vis", "nir", "thermal"):
        flags_k2fixed = flags | cv2.CALIB_FIX_K2
        rms2_k2, k2_k2, dist2_k2, rvecs2_k2, tvecs2_k2 = cv2.calibrateCamera(
            obj2, img2, image_size_wh, k0.copy(), d0.copy(), flags=flags_k2fixed
        )
        if float(rms2_k2) <= float(rms2) * 1.15:
            rms2, k2, dist2, rvecs2, tvecs2 = rms2_k2, k2_k2, dist2_k2, rvecs2_k2, tvecs2_k2
    errors2 = reprojection_errors(obj2, img2, rvecs2, tvecs2, k2, dist2)
    status = "candidate"
    if sensor in ("vis", "nir"):
        if float(rms2) <= 3.5 and float(np.median([e["mean_px"] for e in errors2])) <= 1.5:
            status = "candidate_partial_subpixel"
        else:
            status = "weak_candidate_high_reprojection_error"
    if sensor == "thermal":
        status = "candidate_model_assisted" if float(rms2) <= 2.0 else "weak_candidate_model_assisted"
    return {
        "sensor": sensor,
        "status": status,
        "image_size_wh": list(image_size_wh),
        "n_views_input": len(candidates),
        "n_views_used": len(kept),
        "outlier_labels": [c.label for c, use in zip(candidates, keep) if not use],
        "rms_px": float(rms2),
        "mean_view_error_px": float(np.mean([e["mean_px"] for e in errors2])),
        "median_view_error_px": float(np.median([e["mean_px"] for e in errors2])),
        "max_view_error_px": float(np.max([e["mean_px"] for e in errors2])),
        "K": k2,
        "dist_coeffs": dist2.reshape(-1),
        "views": [
            {
                "label": c.label,
                "bag": c.bag,
                "pattern": list(c.pattern),
                "source": c.source,
                "confidence": c.confidence,
                "method": c.method,
                "frame_index": c.frame_index,
                **err,
            }
            for c, err in zip(kept, errors2)
        ],
    }


def transform_points(points: np.ndarray, h: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float64), h).reshape(-1, 2)


def fit_sensor_to_rgb(
    sensor: str,
    candidates: list[Candidate],
    rgb_by_label: dict[str, Candidate],
    h_hint: np.ndarray,
    max_hint_px: float,
    min_common: tuple[int, int],
) -> dict[str, Any]:
    option_records = []
    rejected = []
    stereo_objects = []
    stereo_src = []
    stereo_rgb = []
    for c in candidates:
        rgb = rgb_by_label.get(c.label)
        if rgb is None:
            continue
        common = (min(c.pattern[0], rgb.pattern[0]), min(c.pattern[1], rgb.pattern[1]))
        if common[0] < min_common[0] or common[1] < min_common[1]:
            continue
        for s_row in range(c.pattern[1] - common[1] + 1):
            for s_col in range(c.pattern[0] - common[0] + 1):
                src_sub = subgrid_points(c.corners, c.pattern, common, s_col, s_row)
                for r_row in range(rgb.pattern[1] - common[1] + 1):
                    for r_col in range(rgb.pattern[0] - common[0] + 1):
                        dst_sub = subgrid_points(rgb.corners, rgb.pattern, common, r_col, r_row)
                        option_records.append(
                            {
                                "label": c.label,
                                "source": c.source,
                                "confidence": c.confidence,
                                "source_pattern": list(c.pattern),
                                "target_pattern": list(rgb.pattern),
                                "common_pattern": list(common),
                                "source_offset_col_row": [int(s_col), int(s_row)],
                                "target_offset_col_row": [int(r_col), int(r_row)],
                                "_src": src_sub,
                                "_dst": dst_sub,
                            }
                        )
    if not option_records:
        return {"sensor": sensor, "status": "no_pairs", "n_pairs": 0, "pairs": [], "rejected_pairs": rejected}

    src_options = np.vstack([o["_src"] for o in option_records])
    dst_options = np.vstack([o["_dst"] for o in option_records])
    seed_thr = 12.0 if sensor == "thermal" else 8.0
    h_seed, seed_mask = cv2.findHomography(
        src_options.astype(np.float64),
        dst_options.astype(np.float64),
        cv2.RANSAC,
        seed_thr,
    )
    if h_seed is None:
        h_seed, seed_mask = cv2.findHomography(src_options.astype(np.float64), dst_options.astype(np.float64), 0)
    if h_seed is None:
        return {
            "sensor": sensor,
            "status": "seed_homography_failed",
            "n_pairs": 0,
            "pairs": [],
            "rejected_pairs": rejected,
        }
    h_seed = h_seed / h_seed[2, 2]

    pairs = []
    for label in sorted(set(o["label"] for o in option_records)):
        best = None
        for opt in [o for o in option_records if o["label"] == label]:
            pred = transform_points(opt["_src"], h_seed)
            err = np.linalg.norm(pred - opt["_dst"], axis=1)
            median_err = float(np.median(err))
            mean_err = float(np.mean(err))
            if best is None or median_err < best[0]:
                best = (median_err, mean_err, opt)
        if best is None:
            continue
        median_err, mean_err, opt = best
        if median_err > max_hint_px:
            rejected.append({**{k: v for k, v in opt.items() if not k.startswith("_")}, "reason": "robust_pair_error_too_high", "seed_median_error_px": median_err})
            continue
        pairs.append(
            {
                **{k: v for k, v in opt.items() if not k.startswith("_")},
                "seed_median_error_px": median_err,
                "seed_mean_error_px": mean_err,
            }
        )
        # The subgrid image corners start at (src_col0, src_row0) in the full
        # board.  The 3D object points must reflect that physical offset so the
        # stereo solver sees the correct geometry.  Previously this always used
        # (0,0) which corrupted T_rgb_sensor for pairs with non-zero offsets.
        _src_col0, _src_row0 = opt["source_offset_col_row"]
        stereo_objects.append(
            object_points_with_offset(
                tuple(opt["common_pattern"]),
                col0=int(_src_col0),
                row0=int(_src_row0),
            )
        )
        stereo_src.append(opt["_src"].reshape(-1, 1, 2).astype(np.float32))
        stereo_rgb.append(opt["_dst"].reshape(-1, 1, 2).astype(np.float32))

    if len(pairs) < 2:
        return {
            "sensor": sensor,
            "status": "insufficient_robust_pairs",
            "n_pairs": len(pairs),
            "pairs": pairs,
            "rejected_pairs": rejected,
        }

    src = np.vstack([p_src.reshape(-1, 2) for p_src in stereo_src])
    dst = np.vstack([p_dst.reshape(-1, 2) for p_dst in stereo_rgb])
    final_thr = 10.0 if sensor == "thermal" else 6.0
    h, mask = cv2.findHomography(src.astype(np.float64), dst.astype(np.float64), cv2.RANSAC, final_thr)
    if h is None:
        h, mask = cv2.findHomography(src.astype(np.float64), dst.astype(np.float64), 0)
    if h is None:
        return {
            "sensor": sensor,
            "status": "final_homography_failed",
            "n_pairs": len(pairs),
            "pairs": pairs,
            "rejected_pairs": rejected,
        }
    h = h / h[2, 2]
    inliers = np.ones((len(src),), dtype=bool) if mask is None else mask.reshape(-1).astype(bool)
    direct_metrics = reprojection_metrics(src, dst, h)
    direct_median = float(direct_metrics.get("median_px", np.inf))
    good_thr = 18.0 if sensor in ("vis", "nir") else 15.0
    weak_thr = 35.0 if sensor in ("vis", "nir") else 30.0
    if direct_median <= good_thr:
        status = "target_plane_homography_candidate"
    elif direct_median <= weak_thr:
        status = "target_plane_homography_weak_high_error"
    else:
        status = "target_plane_homography_rejected_high_error"
    return {
        "sensor": sensor,
        "status": status,
        "H_sensor_to_rgb": h,
        "n_pairs": len(pairs),
        "n_points": int(len(src)),
        "inliers": int(inliers.sum()),
        "hint_metrics_px": reprojection_metrics(src, dst, h_hint),
        "direct_metrics_px": direct_metrics,
        "selection": "all_subgrid_options_ransac_then_best_per_label",
        "pairs": pairs,
        "rejected_pairs": rejected,
        "_stereo_objects": stereo_objects,
        "_stereo_src": stereo_src,
        "_stereo_rgb": stereo_rgb,
    }


def subgrid_points(
    points: np.ndarray,
    full_pattern: tuple[int, int],
    common_pattern: tuple[int, int],
    col0: int,
    row0: int,
) -> np.ndarray:
    full_cols, _full_rows = full_pattern
    sub_cols, sub_rows = common_pattern
    idx = []
    for row in range(sub_rows):
        for col in range(sub_cols):
            idx.append((row0 + row) * full_cols + (col0 + col))
    return points[np.asarray(idx, dtype=int)]


def stereo_extrinsic_from_pairs(
    sensor: str,
    pair_doc: dict[str, Any],
    sensor_intr: dict[str, Any],
    rgb_intr: dict[str, Any],
    image_size_wh: tuple[int, int],
) -> dict[str, Any]:
    if pair_doc.get("n_pairs", 0) < 5:
        return {"sensor": sensor, "status": "insufficient_pairs", "n_pairs": pair_doc.get("n_pairs", 0)}
    if "K" not in sensor_intr:
        return {"sensor": sensor, "status": "missing_sensor_intrinsics"}
    try:
        flags = cv2.CALIB_FIX_INTRINSIC
        _rms, _k1, _d1, _k2, _d2, R, T, E, F = cv2.stereoCalibrate(
            pair_doc["_stereo_objects"],
            pair_doc["_stereo_src"],
            pair_doc["_stereo_rgb"],
            np.asarray(sensor_intr["K"], dtype=np.float64),
            np.asarray(sensor_intr["dist_coeffs"], dtype=np.float64).reshape(-1, 1),
            np.asarray(rgb_intr["K"], dtype=np.float64),
            np.asarray(rgb_intr["dist_coeffs"], dtype=np.float64).reshape(-1, 1),
            image_size_wh,
            flags=flags,
        )
    except Exception as exc:
        return {"sensor": sensor, "status": f"stereo_failed:{exc}", "n_pairs": pair_doc.get("n_pairs", 0)}
    T44 = np.eye(4, dtype=np.float64)
    T44[:3, :3] = R
    T44[:3, 3] = T.reshape(3)
    rms = float(_rms)
    if rms <= 10.0:
        status = "physical_extrinsic_candidate"
    elif rms <= 25.0:
        status = "physical_extrinsic_weak_high_stereo_rms"
    else:
        status = "physical_extrinsic_rejected_high_stereo_rms"
    return {
        "sensor": sensor,
        "status": status,
        "convention": f"X_rgb = T_rgb_{sensor} @ X_{sensor}_homogeneous",
        "stereo_rms_px": rms,
        "n_pairs": pair_doc.get("n_pairs", 0),
        "T_rgb_sensor": T44,
        "R": R,
        "t_m": T.reshape(3),
    }


def strip_private(doc: Any) -> Any:
    if isinstance(doc, dict):
        return {k: strip_private(v) for k, v in doc.items() if not k.startswith("_")}
    if isinstance(doc, list):
        return [strip_private(v) for v in doc]
    return doc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--scan-summary", type=Path, default=Path("runs/calibration_20260623_checker_lidar_scan/scan_summary.json"))
    parser.add_argument("--refined-summary", type=Path, default=Path("runs/calibration_20260623_refined_multisensor_detection/refined_multisensor_detection_summary.json"))
    parser.add_argument("--deep-summary", type=Path, default=Path("runs/calibration_20260623_deep_photonfocus_detection/deep_photonfocus_detection_summary.json"))
    parser.add_argument("--grid-summary", type=Path, default=Path("runs/calibration_20260623_all_sensor_grid_campaign/all_sensor_grid_campaign_summary.json"))
    parser.add_argument("--photonfocus-partial-summary", type=Path, default=Path("runs/calibration_20260623_photonfocus_partial_grid_campaign/photonfocus_partial_grid_summary.json"))
    parser.add_argument("--thermal-merged", type=Path, default=Path("runs/calibration_20260623_thermal_candidates_merged/thermal_candidates_merged_summary.json"))
    parser.add_argument("--rgb-master-h", type=Path, default=Path("data/calibration/new_session/20260623/homographies_20260623_to_rgb.json"))
    parser.add_argument("--initial-intrinsics", type=Path, default=Path("data/matrices/initial_camera_intrinsics_from_report.json"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    initial = load_json(args.initial_intrinsics)
    rgb_intr = load_json(SESSION_DIR / "rgb_intrinsics_20260623.json")
    rgb_by_label = load_rgb_by_label(args.scan_summary, args.refined_summary)

    vis_candidates_all = choose_candidates(
        load_deep_candidates(args.deep_summary, "vis")
        + load_photonfocus_partial(args.photonfocus_partial_summary, "vis")
    )
    nir_candidates_all = choose_candidates(
        load_deep_candidates(args.deep_summary, "nir")
        + load_grid_campaign(args.grid_summary, "nir")
        + load_photonfocus_partial(args.photonfocus_partial_summary, "nir")
    )
    thermal_candidates = choose_candidates(
        load_thermal_merged(args.thermal_merged)
        + load_grid_campaign(args.grid_summary, "thermal_raw")
        + load_grid_campaign(args.grid_summary, "thermal_c")
    )
    # VIS/NIR are often cropped.  Very small partial grids are excellent for
    # detection review, but they make the camera model under-constrained.
    vis_intr_candidates = filter_by_pattern(vis_candidates_all, min_cols=7, min_rows=4)
    nir_intr_candidates = filter_by_pattern(nir_candidates_all, min_cols=7, min_rows=4)
    vis_reg_candidates = filter_by_pattern(vis_candidates_all, min_cols=7, min_rows=5)
    nir_reg_candidates = filter_by_pattern(nir_candidates_all, min_cols=7, min_rows=5)

    intrinsics = {
        "rgb": rgb_intr,
        "vis": calibrate_intrinsics(
            "vis",
            vis_intr_candidates,
            (512, 256),
            initial_k(initial, "vis_photonfocus_hs03_single_band_preview"),
            min_views=6,
        ),
        "nir": calibrate_intrinsics(
            "nir",
            nir_intr_candidates,
            (409, 217),
            initial_k(initial, "nir_photonfocus_hs02_single_band_preview"),
            min_views=7,
        ),
        "thermal": calibrate_intrinsics(
            "thermal",
            thermal_candidates,
            (640, 512),
            initial_k(initial, "thermal_tau2_640_19mm"),
            min_views=8,
        ),
    }

    hdoc = load_json(args.rgb_master_h)
    hs = hdoc.get("homographies", {})
    vis_hint = np.asarray(hs.get("vis_to_rgb"), dtype=np.float64)
    nir_hint = np.asarray(hs.get("nir_to_rgb"), dtype=np.float64)
    thermal_hint = np.asarray(hs.get("thermal_mono16_to_rgb", hs.get("thermal_deg_to_rgb")), dtype=np.float64)

    registration = {
        "vis_to_rgb": fit_sensor_to_rgb("vis", vis_reg_candidates, rgb_by_label, vis_hint, 55.0, (7, 5)),
        "nir_to_rgb": fit_sensor_to_rgb("nir", nir_reg_candidates, rgb_by_label, nir_hint, 70.0, (7, 5)),
        "thermal_to_rgb": fit_sensor_to_rgb("thermal", thermal_candidates, rgb_by_label, thermal_hint, 110.0, (6, 4)),
    }

    physical_extrinsics = {
        "T_rgb_vis": stereo_extrinsic_from_pairs("vis", registration["vis_to_rgb"], intrinsics["vis"], rgb_intr, (512, 256)),
        "T_rgb_nir": stereo_extrinsic_from_pairs("nir", registration["nir_to_rgb"], intrinsics["nir"], rgb_intr, (409, 217)),
        "T_rgb_thermal": stereo_extrinsic_from_pairs("thermal", registration["thermal_to_rgb"], intrinsics["thermal"], rgb_intr, (640, 512)),
    }

    ouster_summary = load_json(Path("runs/calibration_20260623_ouster_rgb_multipose_6dof/ouster_rgb_multipose_6dof_summary.json"))
    active = load_json(Path("data/calibration/active_calibration.json"))
    final_doc = {
        "version": "20260623_final_candidate_v2_grid_assisted",
        "status": "final_candidate_with_model_based_multisensor_observations",
        "important_notes": [
            "RGB intrinsics and Ouster->RGB are the strongest calibrated components.",
            "VIS/NIR partial-board subpixel detections were recovered in all 36 bags; calibration uses only larger partial grids.",
            "Thermal includes model-based/scalar grid observations from the thermal recovery campaign.",
            "VIS/NIR/Thermal physical extrinsics are candidates and must be validated visually/geometrically before field use.",
            "For pragmatic pixel registration, prefer the target-plane homographies to RGB.",
        ],
        "observation_counts": {
            "rgb": len(rgb_by_label),
            "vis_total": len(vis_candidates_all),
            "vis_intrinsics_used_input": len(vis_intr_candidates),
            "vis_rgb_registration_used_input": len(vis_reg_candidates),
            "nir_total": len(nir_candidates_all),
            "nir_intrinsics_used_input": len(nir_intr_candidates),
            "nir_rgb_registration_used_input": len(nir_reg_candidates),
            "thermal": len(thermal_candidates),
        },
        "candidate_selection": {
            "vis_all": selection_summary(vis_candidates_all),
            "vis_intrinsics": selection_summary(vis_intr_candidates),
            "vis_rgb_registration": selection_summary(vis_reg_candidates),
            "nir_all": selection_summary(nir_candidates_all),
            "nir_intrinsics": selection_summary(nir_intr_candidates),
            "nir_rgb_registration": selection_summary(nir_reg_candidates),
            "thermal": selection_summary(thermal_candidates),
        },
        "intrinsics": intrinsics,
        "target_plane_registration_to_rgb": registration,
        "physical_extrinsics_candidates": physical_extrinsics,
        "ouster_rgb": {
            "status": "physical_extrinsic_final_candidate",
            "source": "runs/calibration_20260623_ouster_rgb_multipose_6dof",
            "T_cam_lidar": active.get("extrinsics", {}).get("T_cam_lidar"),
            "summary": ouster_summary,
        },
        "source_files": {
            "deep_summary": str(args.deep_summary),
            "grid_summary": str(args.grid_summary),
            "photonfocus_partial_summary": str(args.photonfocus_partial_summary),
            "thermal_merged": str(args.thermal_merged),
            "rgb_master_homographies": str(args.rgb_master_h),
        },
    }
    final_public = to_jsonable(strip_private(final_doc))
    out_path = args.out_dir / "final_multisensor_calibration_20260623.json"
    out_path.write_text(json.dumps(final_public, indent=2), encoding="utf-8")
    (SESSION_DIR / "calibration_20260623_final_candidate.json").write_text(json.dumps(final_public, indent=2), encoding="utf-8")

    # Write compact CSV summaries.
    with (args.out_dir / "final_calibration_compact.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["block", "sensor", "status", "n_views_or_pairs", "rms_or_median_px", "notes"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for sensor, intr in intrinsics.items():
            writer.writerow(
                {
                    "block": "intrinsics",
                    "sensor": sensor,
                    "status": intr.get("status"),
                    "n_views_or_pairs": intr.get("n_views", intr.get("n_views_used", "")),
                    "rms_or_median_px": intr.get("rms_px", ""),
                    "notes": intr.get("selected_model", ""),
                }
            )
        for name, reg in registration.items():
            med = (reg.get("direct_metrics_px") or {}).get("median_px", "")
            writer.writerow(
                {
                    "block": "homography_to_rgb",
                    "sensor": name,
                    "status": reg.get("status"),
                    "n_views_or_pairs": reg.get("n_pairs", ""),
                    "rms_or_median_px": med,
                    "notes": f"points={reg.get('n_points','')}",
                }
            )
        for name, ext in physical_extrinsics.items():
            writer.writerow(
                {
                    "block": "physical_extrinsic_candidate",
                    "sensor": name,
                    "status": ext.get("status"),
                    "n_views_or_pairs": ext.get("n_pairs", ""),
                    "rms_or_median_px": ext.get("stereo_rms_px", ""),
                    "notes": ext.get("convention", ""),
                }
            )
    print(json.dumps({
        "out": str(out_path),
        "observation_counts": final_doc["observation_counts"],
        "intrinsics_status": {k: v.get("status") for k, v in intrinsics.items()},
        "registration_pairs": {k: v.get("n_pairs") for k, v in registration.items()},
        "physical_extrinsics_status": {k: v.get("status") for k, v in physical_extrinsics.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
