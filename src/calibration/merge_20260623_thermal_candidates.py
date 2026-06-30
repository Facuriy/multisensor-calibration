#!/usr/bin/env python3
"""Merge production RAW thermal detections with vetted scalar fallback detections."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


RAW_DEFAULT = Path("runs/calibration_20260623_thermal_production_lowload/thermal_production_candidates_for_to_vis.json")
SCALAR_DEFAULT = Path(
    "runs/calibration_20260623_thermal_scalar_checker_test_v2/"
    "thermal_guided_detection_summary_scalar_candidates.json"
)
OUT_DEFAULT = Path("runs/calibration_20260623_thermal_candidates_merged")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rows_by_label(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(r.get("label_norm")): r for r in doc.get("results", [])}


def checker_detected(row: dict[str, Any] | None) -> bool:
    return bool(row and (row.get("checker") or {}).get("detected"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-candidates", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--scalar-candidates", type=Path, default=SCALAR_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_json(args.raw_candidates)
    scalar = load_json(args.scalar_candidates)
    raw_by = rows_by_label(raw)
    scalar_by = rows_by_label(scalar)

    labels = sorted(set(raw_by) | set(scalar_by))
    merged = []
    table_rows = []
    for label in labels:
        raw_row = raw_by.get(label)
        scalar_row = scalar_by.get(label)
        if checker_detected(raw_row):
            out = raw_row
            source = "raw_production"
        elif checker_detected(scalar_row):
            out = {
                "label_norm": scalar_row.get("label_norm"),
                "bag": scalar_row.get("bag"),
                "checker": scalar_row.get("checker"),
            }
            source = "scalar_fallback"
        else:
            out = raw_row or scalar_row or {"label_norm": label, "checker": {"detected": False}}
            source = "none"
            out.setdefault("checker", {"detected": False})
        out = json.loads(json.dumps(out))
        out["source_priority"] = source
        merged.append(out)
        chk = out.get("checker") or {}
        table_rows.append(
            {
                "label_norm": label,
                "bag": out.get("bag", ""),
                "detected": chk.get("detected", False),
                "source_priority": source,
                "pattern": "x".join(map(str, chk.get("pattern_internal_corners", []))) if chk.get("detected") else "",
                "method": chk.get("method", ""),
                "score": chk.get("score", ""),
                "frame_index": chk.get("frame_index", ""),
            }
        )

    summary = {
        "raw_candidates": str(args.raw_candidates),
        "scalar_candidates": str(args.scalar_candidates),
        "n_rows": len(merged),
        "checker_detected": sum(1 for r in merged if (r.get("checker") or {}).get("detected")),
        "raw_detected": sum(1 for r in merged if r.get("source_priority") == "raw_production"),
        "scalar_fallback_detected": sum(1 for r in merged if r.get("source_priority") == "scalar_fallback"),
        "results": merged,
    }
    with (args.out_dir / "thermal_candidates_merged_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (args.out_dir / "thermal_candidates_merged_table.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["label_norm", "bag", "detected", "source_priority", "pattern", "method", "score", "frame_index"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table_rows)
    print(
        f"Merged detections: {summary['checker_detected']}/{summary['n_rows']} "
        f"(raw={summary['raw_detected']}, scalar_fallback={summary['scalar_fallback_detected']})"
    )
    print(f"Wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

