#!/usr/bin/env python3
"""Batch runner for RGB-master multisensor plot orthomosaics.

The runner scans short time windows in a ROS1 bag, finds plot IDs from GNSS/GPKG,
then calls make_rgb_master_multisensor_orthomosaic.py for the best window per
plot. It keeps the heavy LiDAR reads out of the discovery pass.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from rosbags.rosbag1 import Reader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction.make_rgb_master_multisensor_orthomosaic import scan_plot_counts  # noqa: E402


def bag_time_bounds(bag: Path) -> tuple[int, int]:
    with Reader(bag) as reader:
        return int(reader.start_time), int(reader.end_time)


def discover_plot_windows(
    bag: Path,
    gpkg: Path,
    sample_windows: int,
    window_ms: float,
    max_sync_ms: float,
    min_frames: int,
) -> dict[str, dict[str, Any]]:
    start_ns, end_ns = bag_time_bounds(bag)
    if sample_windows <= 1:
        centers = [(start_ns + end_ns) // 2]
    else:
        span = end_ns - start_ns
        margin = int(window_ms * 1_000_000)
        lo, hi = start_ns + margin, end_ns - margin
        centers = [int(lo + (hi - lo) * i / (sample_windows - 1)) for i in range(sample_windows)]
    best: dict[str, dict[str, Any]] = {}
    for center in centers:
        scan = scan_plot_counts(bag, gpkg, center, window_ms, max_sync_ms)
        for plot_id, count in scan["plot_frame_counts"].items():
            if int(count) < min_frames:
                continue
            prev = best.get(plot_id)
            if prev is None or int(count) > int(prev["frames_in_window"]):
                best[plot_id] = {
                    "plot_id": plot_id,
                    "center_ns": int(center),
                    "frames_in_window": int(count),
                    "window_ms": float(window_ms),
                }
    return dict(sorted(best.items(), key=lambda kv: int(kv[0])))


def run_one(
    script: Path,
    bag: Path,
    gpkg: Path,
    out_root: Path,
    entry: dict[str, Any],
    frames: int,
    splat_radius: int,
    max_sync_ms: float,
    max_range_m: float,
    margin_px: int,
    trim_bottom_px: int,
    calibration: Path,
    skip_lidar: bool,
    no_geotiff: bool,
    child_timeout_sec: int,
) -> dict[str, Any]:
    plot_id = str(entry["plot_id"])
    out_dir = out_root / f"plot_{int(plot_id):02d}"
    cmd = [
        sys.executable,
        str(script),
        "--bag",
        str(bag),
        "--gpkg",
        str(gpkg),
        "--plot-id",
        plot_id,
        "--center-ns",
        str(int(entry["center_ns"])),
        "--window-ms",
        str(float(entry["window_ms"])),
        "--frames",
        str(int(frames)),
        "--out",
        str(out_dir),
        "--splat-radius",
        str(int(splat_radius)),
        "--max-sync-ms",
        str(float(max_sync_ms)),
        "--max-range-m",
        str(float(max_range_m)),
        "--margin-px",
        str(int(margin_px)),
        "--trim-bottom-px",
        str(int(trim_bottom_px)),
        "--calibration",
        str(calibration),
    ]
    if skip_lidar:
        cmd.append("--skip-lidar")
    if no_geotiff:
        cmd.append("--no-geotiff")
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=child_timeout_sec if child_timeout_sec > 0 else None,
    )
    summary_path = out_dir / "mosaic_internal_summary.json"
    quality = None
    if summary_path.exists():
        quality = json.loads(summary_path.read_text(encoding="utf-8")).get("trajectory_quality")
    return {
        **entry,
        "out_dir": str(out_dir),
        "returncode": result.returncode,
        "quality": quality,
        "cmd": cmd,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bag", type=Path, required=True)
    ap.add_argument("--gpkg", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--sample-windows", type=int, default=12)
    ap.add_argument("--window-ms", type=float, default=7000.0)
    ap.add_argument("--max-sync-ms", type=float, default=900.0)
    ap.add_argument("--min-frames", type=int, default=12)
    ap.add_argument("--frames", type=int, default=0, help="0 means use all frames in the selected window.")
    ap.add_argument("--splat-radius", type=int, default=11)
    ap.add_argument("--max-range-m", type=float, default=8.0)
    ap.add_argument("--margin-px", type=int, default=2)
    ap.add_argument("--trim-bottom-px", type=int, default=0)
    ap.add_argument("--calibration", type=Path, default=PROJECT_ROOT / "data/calibration/new_session/20260623/calibration_20260623_final_candidate.json")
    ap.add_argument("--max-plots", type=int, default=0)
    ap.add_argument("--plots", nargs="*", help="Optional explicit plot IDs to process after discovery.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-lidar", action="store_true", help="Build camera-only mosaics. Much lighter; no Ouster depth/height/intensity layers.")
    ap.add_argument("--no-geotiff", action="store_true")
    ap.add_argument("--child-timeout-sec", type=int, default=0, help="Optional timeout for each plot subprocess. 0 means no timeout.")
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    windows = discover_plot_windows(args.bag, args.gpkg, args.sample_windows, args.window_ms, args.max_sync_ms, args.min_frames)
    if args.plots:
        wanted = {str(p) for p in args.plots}
        windows = {k: v for k, v in windows.items() if k in wanted}
    entries = list(windows.values())
    if args.max_plots > 0:
        entries = entries[: args.max_plots]

    script = PROJECT_ROOT / "src" / "extraction" / "make_rgb_master_multisensor_orthomosaic.py"
    results = []
    if not args.dry_run:
        for entry in entries:
            try:
                results.append(
                    run_one(
                        script,
                        args.bag,
                        args.gpkg,
                        args.out_root,
                        entry,
                        args.frames,
                        args.splat_radius,
                        args.max_sync_ms,
                        args.max_range_m,
                        args.margin_px,
                        args.trim_bottom_px,
                        args.calibration,
                        args.skip_lidar,
                        args.no_geotiff,
                        args.child_timeout_sec,
                    )
                )
            except subprocess.TimeoutExpired as exc:
                results.append(
                    {
                        **entry,
                        "out_dir": str(args.out_root / f"plot_{int(entry['plot_id']):02d}"),
                        "returncode": "timeout",
                        "quality": None,
                        "cmd": exc.cmd,
                        "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
                        "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
                    }
                )
    summary = {
        "bag": str(args.bag),
        "gpkg": str(args.gpkg),
        "calibration": str(args.calibration),
        "discovered_windows": windows,
        "selected_entries": entries,
        "dry_run": bool(args.dry_run),
        "low_resource_options": {
            "frames": args.frames,
            "skip_lidar": bool(args.skip_lidar),
            "max_range_m": args.max_range_m,
            "splat_radius": args.splat_radius,
            "trim_bottom_px": args.trim_bottom_px,
            "window_ms": args.window_ms,
            "sample_windows": args.sample_windows,
            "no_geotiff": bool(args.no_geotiff),
            "child_timeout_sec": args.child_timeout_sec,
        },
        "results": results,
    }
    out = args.out_root / "batch_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
