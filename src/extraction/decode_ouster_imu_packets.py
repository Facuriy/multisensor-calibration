#!/usr/bin/env python3
"""Decode Ouster raw IMU PacketMsg messages from ROS1 bags.

Ouster IMU UDP payload layout is 48 useful bytes, little-endian:
  uint64 diagnostic/system timestamp [ns]
  uint64 accelerometer timestamp [ns]
  uint64 gyroscope timestamp [ns]
  float32 accel_x, accel_y, accel_z [g]
  float32 gyro_x, gyro_y, gyro_z [deg/s]

The ROS ouster_ros/PacketMsg stores this payload in uint8[] buf. In our bags the
buffer length is 49 bytes; the first 48 bytes are the documented IMU packet.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

IMU_TOPIC = "/ssf/os1_node/imu_packets"
PACKET_TYPE = "ouster_ros/msg/PacketMsg"


def register_packet_type():
    typestore = get_typestore(Stores.ROS1_NOETIC)
    typestore.register(get_types_from_msg("uint8[] buf\n", PACKET_TYPE))
    return typestore


def decode_packet(buf: bytes) -> dict:
    if len(buf) < 48:
        raise ValueError(f"IMU packet too short: {len(buf)} bytes")
    payload = buf[:48]
    diag_ts, accel_ts, gyro_ts = np.frombuffer(payload[:24], dtype="<u8", count=3)
    accel = np.frombuffer(payload[24:36], dtype="<f4", count=3).astype(np.float64)
    gyro = np.frombuffer(payload[36:48], dtype="<f4", count=3).astype(np.float64)
    return {
        "diag_ts_ns": int(diag_ts),
        "accel_ts_ns": int(accel_ts),
        "gyro_ts_ns": int(gyro_ts),
        "accel_g": accel,
        "accel_mps2": accel * 9.80665,
        "gyro_dps": gyro,
        "gyro_radps": np.deg2rad(gyro),
    }


def read_imu(bag: Path, topic: str, start_ns: int | None, stop_ns: int | None, limit: int | None) -> list[dict]:
    typestore = register_packet_type()
    rows = []
    with Reader(bag) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            raise RuntimeError(f"Topic not found: {topic}")
        kwargs = {}
        if start_ns is not None:
            kwargs["start"] = int(start_ns)
        if stop_ns is not None:
            kwargs["stop"] = int(stop_ns)
        for conn, ts, raw in reader.messages(connections=conns, **kwargs):
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            buf = bytes(msg.buf)
            decoded = decode_packet(buf)
            decoded["bag_ts_ns"] = int(ts)
            decoded["buf_len"] = len(buf)
            rows.append(decoded)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "bag_ts_ns",
        "diag_ts_ns",
        "accel_ts_ns",
        "gyro_ts_ns",
        "buf_len",
        "accel_x_g",
        "accel_y_g",
        "accel_z_g",
        "accel_x_mps2",
        "accel_y_mps2",
        "accel_z_mps2",
        "gyro_x_dps",
        "gyro_y_dps",
        "gyro_z_dps",
        "gyro_x_radps",
        "gyro_y_radps",
        "gyro_z_radps",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            out = {
                "bag_ts_ns": r["bag_ts_ns"],
                "diag_ts_ns": r["diag_ts_ns"],
                "accel_ts_ns": r["accel_ts_ns"],
                "gyro_ts_ns": r["gyro_ts_ns"],
                "buf_len": r["buf_len"],
            }
            for i, axis in enumerate("xyz"):
                out[f"accel_{axis}_g"] = float(r["accel_g"][i])
                out[f"accel_{axis}_mps2"] = float(r["accel_mps2"][i])
                out[f"gyro_{axis}_dps"] = float(r["gyro_dps"][i])
                out[f"gyro_{axis}_radps"] = float(r["gyro_radps"][i])
            writer.writerow(out)


def stats(rows: list[dict]) -> dict:
    bag_ts = np.asarray([r["bag_ts_ns"] for r in rows], dtype=np.float64)
    acc_ts = np.asarray([r["accel_ts_ns"] for r in rows], dtype=np.float64)
    gyr_ts = np.asarray([r["gyro_ts_ns"] for r in rows], dtype=np.float64)
    accel = np.vstack([r["accel_g"] for r in rows])
    gyro = np.vstack([r["gyro_dps"] for r in rows])
    acc_norm = np.linalg.norm(accel, axis=1)
    gyro_norm = np.linalg.norm(gyro, axis=1)
    dt = np.diff(bag_ts) / 1e9 if len(rows) > 1 else np.asarray([])
    duration = (bag_ts[-1] - bag_ts[0]) / 1e9 if len(rows) > 1 else 0.0
    return {
        "samples": len(rows),
        "duration_s": float(duration),
        "rate_hz_mean": float(1.0 / np.mean(dt)) if len(dt) and np.mean(dt) > 0 else None,
        "rate_hz_median": float(1.0 / np.median(dt)) if len(dt) and np.median(dt) > 0 else None,
        "buf_len_unique": sorted({int(r["buf_len"]) for r in rows}),
        "accel_g_mean_xyz": accel.mean(axis=0).tolist(),
        "accel_g_std_xyz": accel.std(axis=0).tolist(),
        "accel_norm_g_mean": float(acc_norm.mean()),
        "accel_norm_g_std": float(acc_norm.std()),
        "gyro_dps_mean_xyz": gyro.mean(axis=0).tolist(),
        "gyro_dps_std_xyz": gyro.std(axis=0).tolist(),
        "gyro_norm_dps_mean": float(gyro_norm.mean()),
        "gyro_norm_dps_std": float(gyro_norm.std()),
        "sensor_accel_gyro_ts_delta_ms_mean": float(np.mean((gyr_ts - acc_ts) / 1e6)),
        "bag_to_accel_ts_offset_s_median": float(np.median((bag_ts - acc_ts) / 1e9)),
        "bag_to_gyro_ts_offset_s_median": float(np.median((bag_ts - gyr_ts) / 1e9)),
    }


def draw_timeseries(rows: list[dict], out: Path) -> None:
    if len(rows) < 2:
        return
    t = (np.asarray([r["bag_ts_ns"] for r in rows], dtype=np.float64) - rows[0]["bag_ts_ns"]) / 1e9
    accel = np.vstack([r["accel_g"] for r in rows])
    gyro = np.vstack([r["gyro_dps"] for r in rows])
    h, w = 760, 1400
    canvas = np.full((h, w, 3), 250, np.uint8)
    cv2.rectangle(canvas, (0, 0), (w, 52), (8, 10, 10), -1)
    cv2.putText(canvas, "Decoded Ouster IMU packets", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    panels = [
        ("accel [g]", accel, (-1.4, 1.4), 86, 330),
        ("gyro [deg/s]", gyro, (-30.0, 30.0), 410, 655),
    ]
    colors = [(40, 40, 220), (40, 150, 40), (220, 80, 40)]
    labels = ["x", "y", "z"]
    for title, arr, ylim, y0, y1 in panels:
        cv2.rectangle(canvas, (70, y0), (w - 40, y1), (255, 255, 255), -1)
        cv2.rectangle(canvas, (70, y0), (w - 40, y1), (80, 80, 80), 1)
        cv2.putText(canvas, title, (78, y0 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2, cv2.LINE_AA)
        for value in [ylim[0], 0.0, ylim[1]]:
            yy = int(y1 - (value - ylim[0]) / (ylim[1] - ylim[0]) * (y1 - y0))
            cv2.line(canvas, (70, yy), (w - 40, yy), (220, 220, 220), 1)
            cv2.putText(canvas, f"{value:g}", (12, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
        for j in range(3):
            pts = []
            for tt, val in zip(t, arr[:, j]):
                x = int(70 + tt / max(t[-1], 1e-6) * (w - 110))
                y = int(y1 - (float(val) - ylim[0]) / (ylim[1] - ylim[0]) * (y1 - y0))
                pts.append([x, np.clip(y, y0, y1)])
            pts_np = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts_np], False, colors[j], 2, cv2.LINE_AA)
            cv2.putText(canvas, labels[j], (w - 145 + j * 35, y0 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[j], 2, cv2.LINE_AA)
    cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 96])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bag", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--topic", default=IMU_TOPIC)
    ap.add_argument("--center-ns", type=int, default=None)
    ap.add_argument("--window-ms", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    start_ns = stop_ns = None
    if args.center_ns is not None and args.window_ms is not None:
        half = int(args.window_ms * 1_000_000)
        start_ns = args.center_ns - half
        stop_ns = args.center_ns + half
    rows = read_imu(args.bag, args.topic, start_ns, stop_ns, args.limit)
    if not rows:
        raise RuntimeError("No IMU packets decoded")
    csv_path = args.out / "ouster_imu_decoded.csv"
    summary_path = args.out / "ouster_imu_summary.json"
    plot_path = args.out / "ouster_imu_timeseries.jpg"
    write_csv(rows, csv_path)
    summary = stats(rows)
    summary["bag"] = str(args.bag)
    summary["topic"] = args.topic
    summary["csv"] = str(csv_path)
    summary["plot"] = str(plot_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    draw_timeseries(rows, plot_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
