#!/usr/bin/env python3
"""Export a clean SLAM dataset from the SWISS ROS1 bags.

The source bags contain Ouster IMU packets as raw ouster_ros/PacketMsg buffers.
This exporter decodes them and writes a standard sensor_msgs/Imu stream while
copying the original PointCloud2 messages, GNSS fixes and velocity messages.

Outputs:
  out/
    slam_input.bag              optional ROS1 bag with remapped topics
    clouds_npz/cloud_*.npz      organized Ouster fields for offline pipelines
    imu.csv
    gps.csv
    velocity.csv
    clouds.csv
    manifest.json
    lio_sam_params_seed.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rosbags.rosbag1 import Reader, Writer
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.extract_ouster_panel_from_manual_intensity_roi import stamp_ns  # noqa: E402
from src.extraction.decode_ouster_imu_packets import decode_packet  # noqa: E402

CLOUD_TOPIC = "/ssf/os1_cloud_node/points"
IMU_PACKET_TOPIC = "/ssf/os1_node/imu_packets"
GPS_TOPIC = "/ssf/gnss/fix"
VEL_TOPIC = "/ssf/gnss/vel"
PACKET_TYPE = "ouster_ros/msg/PacketMsg"

OUT_CLOUD_TOPIC = "/points_raw"
OUT_ORIGINAL_CLOUD_TOPIC = "/points_ouster"
OUT_IMU_TOPIC = "/imu/data"
OUT_GPS_TOPIC = "/gps/fix"
OUT_VEL_TOPIC = "/gps/vel"

OUSTER_LIDAR_TO_SENSOR_MM = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 38.195],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
OUSTER_IMU_TO_SENSOR_MM = np.array(
    [
        [1.0, 0.0, 0.0, 6.253],
        [0.0, 1.0, 0.0, -11.775],
        [0.0, 0.0, 1.0, 7.645],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass
class ExportCounts:
    clouds: int = 0
    imu: int = 0
    gps: int = 0
    velocity: int = 0


def typestore_with_packet():
    typestore = get_typestore(Stores.ROS1_NOETIC)
    try:
        typestore.register(get_types_from_msg("uint8[] buf\n", PACKET_TYPE))
    except Exception:
        pass
    return typestore


def ros_time(typestore, stamp: int):
    cls = typestore.types["builtin_interfaces/msg/Time"]
    sec = int(stamp // 1_000_000_000)
    nanosec = int(stamp % 1_000_000_000)
    return cls(sec=sec, nanosec=nanosec)


def header(typestore, stamp: int, frame_id: str, seq: int = 0):
    cls = typestore.types["std_msgs/msg/Header"]
    return cls(seq=int(seq), stamp=ros_time(typestore, stamp), frame_id=frame_id)


def vector3(typestore, values: np.ndarray):
    cls = typestore.types["geometry_msgs/msg/Vector3"]
    return cls(x=float(values[0]), y=float(values[1]), z=float(values[2]))


def quaternion_identity(typestore):
    cls = typestore.types["geometry_msgs/msg/Quaternion"]
    return cls(x=0.0, y=0.0, z=0.0, w=1.0)


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float64,
    )


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def quat_inverse(q: np.ndarray) -> np.ndarray:
    denom = float(np.dot(q, q))
    if denom <= 0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64) / denom


def quat_msg(typestore, q_xyzw: np.ndarray):
    q = q_xyzw.astype(np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= 0:
        return quaternion_identity(typestore)
    q /= norm
    cls = typestore.types["geometry_msgs/msg/Quaternion"]
    return cls(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))


def accel_tilt_quaternion(accel: np.ndarray, extrinsic_rot: np.ndarray) -> np.ndarray:
    """Estimate roll/pitch from acceleration for LIO-SAM's required IMU attitude.

    LIO-SAM rotates the incoming IMU orientation by `extrinsicRPY` internally.
    This function returns the input quaternion that will become the desired
    lidar-frame tilt after that internal multiplication. Yaw is unobservable
    from an accelerometer, so it is fixed at zero.
    """
    accel_lidar = extrinsic_rot @ accel.astype(np.float64)
    ax, ay, az = [float(v) for v in accel_lidar]
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
    q_desired_lidar = quat_from_rpy(roll, pitch, 0.0)

    # extQRPY is represented by the same rotation matrix used in the YAML.
    # For the Ouster design seed this is a 180-degree rotation around Z.
    if np.allclose(extrinsic_rot, np.diag([-1.0, -1.0, 1.0]), atol=1e-9):
        q_ext = quat_from_rpy(0.0, 0.0, math.pi)
    elif np.allclose(extrinsic_rot, np.eye(3), atol=1e-9):
        q_ext = quat_from_rpy(0.0, 0.0, 0.0)
    else:
        # Generic conversion from rotation matrix to quaternion.
        m = extrinsic_rot.astype(np.float64)
        tr = float(np.trace(m))
        if tr > 0:
            s = math.sqrt(tr + 1.0) * 2.0
            q_ext = np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s])
        else:
            idx = int(np.argmax(np.diag(m)))
            if idx == 0:
                s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
                q_ext = np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
            elif idx == 1:
                s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
                q_ext = np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
            else:
                s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
                q_ext = np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s])
    q_in = quat_multiply(q_desired_lidar, quat_inverse(q_ext))
    return q_in / (np.linalg.norm(q_in) + 1e-12)


def imu_msg_from_packet(
    typestore,
    packet_msg,
    stamp: int,
    frame_id: str,
    seq: int,
    orientation_mode: str,
    extrinsic_rot: np.ndarray,
    pretransform_imu_to_lidar: bool,
):
    decoded = decode_packet(bytes(packet_msg.buf))
    accel = decoded["accel_mps2"].astype(np.float64)
    gyro = decoded["gyro_radps"].astype(np.float64)
    orientation_rot = extrinsic_rot
    if pretransform_imu_to_lidar:
        accel = extrinsic_rot @ accel
        gyro = extrinsic_rot @ gyro
        orientation_rot = np.eye(3, dtype=np.float64)

    imu_cls = typestore.types["sensor_msgs/msg/Imu"]
    orientation_cov = np.zeros(9, dtype=np.float64)
    if orientation_mode == "identity":
        orientation = quaternion_identity(typestore)
        orientation_cov[0] = -1.0
    elif orientation_mode == "accel_tilt":
        orientation = quat_msg(typestore, accel_tilt_quaternion(accel, orientation_rot))
        orientation_cov[:] = np.diag([0.05, 0.05, 10.0]).reshape(-1)
    else:
        raise ValueError(f"Unknown IMU orientation mode: {orientation_mode}")
    angular_cov = np.diag([0.02, 0.02, 0.02]).reshape(-1).astype(np.float64)
    linear_cov = np.diag([0.08, 0.08, 0.08]).reshape(-1).astype(np.float64)
    return imu_cls(
        header=header(typestore, stamp, frame_id, seq),
        orientation=orientation,
        orientation_covariance=orientation_cov,
        angular_velocity=vector3(typestore, gyro),
        angular_velocity_covariance=angular_cov,
        linear_acceleration=vector3(typestore, accel),
        linear_acceleration_covariance=linear_cov,
    ), decoded


POINT_DATATYPES = {
    1: ("i1", np.int8),
    2: ("u1", np.uint8),
    3: ("<i2", np.int16),
    4: ("<u2", np.uint16),
    5: ("<i4", np.int32),
    6: ("<u4", np.uint32),
    7: ("<f4", np.float32),
    8: ("<f8", np.float64),
}


def pointcloud_fields(msg) -> dict[str, np.ndarray]:
    """Read common Ouster PointCloud2 fields as organized [ring, column] arrays."""
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    step = int(msg.point_step)
    h = int(msg.height)
    w = int(msg.width)
    count = h * w
    out: dict[str, np.ndarray] = {}
    wanted = {"x", "y", "z", "intensity", "t", "reflectivity", "ring", "noise", "range"}
    for field in msg.fields:
        name = str(field.name)
        if name not in wanted:
            continue
        dtype_entry = POINT_DATATYPES.get(int(field.datatype))
        if dtype_entry is None:
            continue
        dtype = np.dtype(dtype_entry[0])
        arr = np.ndarray((count,), dtype=dtype, buffer=raw, offset=int(field.offset), strides=(step,)).copy()
        # Ouster cloud is azimuth-major in this dataset: width blocks, each with
        # all rings. The reported PointCloud2 shape is height=64,width=2048.
        out[name] = arr.reshape(w, h).T
    return out


def make_liosam_cloud(typestore, msg, stamp: int, frame_id: str):
    """Convert Ouster cloud fields to a compact PointXYZIRT cloud.

    LIO-SAM style pipelines commonly expect per-point relative time under the
    field name `time` in seconds. The Ouster ROS cloud stores this as `t` in
    nanoseconds, so this creates a companion cloud without modifying the
    original copied cloud.
    """
    fields_in = pointcloud_fields(msg)
    xyz = np.dstack([fields_in["x"], fields_in["y"], fields_in["z"]]).reshape(-1, 3).astype("<f4")
    intensity = fields_in.get("intensity", np.zeros(fields_in["x"].shape, np.float32)).reshape(-1).astype("<f4")
    ring = fields_in.get("ring", np.zeros(fields_in["x"].shape, np.uint8)).reshape(-1).astype("<u2")
    if "t" in fields_in:
        time_s = (fields_in["t"].reshape(-1).astype(np.float32) * 1e-9).astype("<f4")
    else:
        time_s = np.zeros(len(xyz), dtype="<f4")
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    intensity = intensity[finite]
    ring = ring[finite]
    time_s = time_s[finite]

    point_step = 24
    data = np.zeros((len(xyz), point_step), dtype=np.uint8)
    data[:, 0:4] = np.ascontiguousarray(xyz[:, 0]).view(np.uint8).reshape(-1, 4)
    data[:, 4:8] = np.ascontiguousarray(xyz[:, 1]).view(np.uint8).reshape(-1, 4)
    data[:, 8:12] = np.ascontiguousarray(xyz[:, 2]).view(np.uint8).reshape(-1, 4)
    data[:, 12:16] = np.ascontiguousarray(intensity).view(np.uint8).reshape(-1, 4)
    data[:, 16:18] = np.ascontiguousarray(ring).view(np.uint8).reshape(-1, 2)
    data[:, 20:24] = np.ascontiguousarray(time_s).view(np.uint8).reshape(-1, 4)

    pf_cls = typestore.types["sensor_msgs/msg/PointField"]
    pc_cls = typestore.types["sensor_msgs/msg/PointCloud2"]
    return pc_cls(
        header=header(typestore, stamp, frame_id),
        height=1,
        width=int(len(xyz)),
        fields=[
            pf_cls(name="x", offset=0, datatype=7, count=1),
            pf_cls(name="y", offset=4, datatype=7, count=1),
            pf_cls(name="z", offset=8, datatype=7, count=1),
            pf_cls(name="intensity", offset=12, datatype=7, count=1),
            pf_cls(name="ring", offset=16, datatype=4, count=1),
            pf_cls(name="time", offset=20, datatype=7, count=1),
        ],
        is_bigendian=False,
        point_step=point_step,
        row_step=int(point_step * len(xyz)),
        data=data.reshape(-1),
        is_dense=True,
    )


def write_cloud_npz(msg, path: Path) -> dict:
    fields = pointcloud_fields(msg)
    xyz = np.dstack([fields["x"], fields["y"], fields["z"]]).astype(np.float32)
    finite = np.isfinite(xyz).all(axis=2)
    range_m = np.linalg.norm(xyz, axis=2)
    valid = finite & (range_m > 0.1)
    payload = {"xyz": xyz}
    for name in ["intensity", "t", "reflectivity", "ring", "noise", "range"]:
        if name in fields:
            payload[name] = fields[name]
    np.savez_compressed(path, **payload)
    return {
        "height": int(msg.height),
        "width": int(msg.width),
        "point_step": int(msg.point_step),
        "valid_points": int(valid.sum()),
        "fields": sorted(payload.keys()),
        "npz": str(path),
    }


def velocity_norm(msg: object) -> float:
    tw = getattr(msg, "twist", None)
    lin = getattr(tw, "linear", None) if tw is not None else None
    if lin is None:
        return float("nan")
    vals = [float(getattr(lin, axis, 0.0)) for axis in "xyz"]
    return float(math.sqrt(sum(v * v for v in vals)))


def append_csv(path: Path, fieldnames: list[str], row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def ouster_design_extrinsic(mode: str) -> dict:
    """Return the Ouster design LiDAR/IMU extrinsic in meters.

    LIO-SAM labels this block as T_lb (lidar -> imu). The raw bags inspected on
    2026-06-27 publish `/ssf/os1_cloud_node/points` in `/os1_lidar`, so the
    default should be `lidar_frame`.
    """
    mode = mode.lower()
    if mode == "identity":
        t_lidar_to_imu_mm = np.eye(4, dtype=np.float64)
        source = "identity_debug_only"
        note = "Identity is only useful as a launch sanity check, not as an Ouster calibration."
    elif mode == "sensor_frame":
        t_lidar_to_imu_mm = OUSTER_IMU_TO_SENSOR_MM.copy()
        source = "ouster_design_imu_to_sensor"
        note = "Use only if the PointCloud2 XYZ is already expressed in the Ouster sensor frame."
    elif mode == "lidar_frame":
        t_lidar_to_imu_mm = np.linalg.inv(OUSTER_IMU_TO_SENSOR_MM) @ OUSTER_LIDAR_TO_SENSOR_MM
        source = "ouster_design_inv_imu_to_sensor_times_lidar_to_sensor"
        note = "Use when PointCloud2 XYZ is expressed in the Ouster lidar frame."
    else:
        raise ValueError(f"Unknown Ouster extrinsic mode: {mode}")

    t_lidar_to_imu_m = t_lidar_to_imu_mm.copy()
    t_lidar_to_imu_m[:3, 3] *= 0.001
    return {
        "mode": mode,
        "source": source,
        "note": note,
        "T_lidar_to_imu_m": t_lidar_to_imu_m.tolist(),
        "translation_m": t_lidar_to_imu_m[:3, 3].tolist(),
        "rotation": t_lidar_to_imu_m[:3, :3].tolist(),
        "ouster_design_inputs_mm": {
            "lidar_to_sensor_transform": OUSTER_LIDAR_TO_SENSOR_MM.tolist(),
            "imu_to_sensor_transform": OUSTER_IMU_TO_SENSOR_MM.tolist(),
        },
    }


def yaml_matrix(mat: np.ndarray, indent: str = "               ") -> str:
    rows = []
    for i, row in enumerate(mat):
        prefix = "" if i == 0 else indent
        rows.append(prefix + ", ".join(f"{float(v):.9g}" for v in row))
    return ",\n".join(rows)


def make_lio_sam_seed(path: Path, imu_frame: str, lidar_frame: str, extrinsic: dict) -> None:
    rot = np.array(extrinsic["rotation"], dtype=np.float64)
    trans = np.array(extrinsic["translation_m"], dtype=np.float64)
    path.write_text(
        f"""# Seed parameters for LIO-SAM trials.
# Ouster design extrinsic mode: {extrinsic["mode"]}
# Source: {extrinsic["source"]}
# Note: {extrinsic["note"]}
pointCloudTopic: "{OUT_CLOUD_TOPIC}"
imuTopic: "{OUT_IMU_TOPIC}"
gpsTopic: "{OUT_GPS_TOPIC}"

lidarFrame: "{lidar_frame}"
baselinkFrame: "base_link"
odometryFrame: "odom"
mapFrame: "map"

sensor: ouster
N_SCAN: 64
Horizon_SCAN: 2048
downsampleRate: 1
lidarMinRange: 0.5
lidarMaxRange: 80.0

# Ouster IMU packet has been decoded to SI units:
# angular_velocity [rad/s], linear_acceleration [m/s^2].
imu_frame: "{imu_frame}"

# LIO-SAM convention: T_lb / lidar -> imu.
extrinsicTrans: [{trans[0]:.9g}, {trans[1]:.9g}, {trans[2]:.9g}]
extrinsicRot: [{yaml_matrix(rot)}]
extrinsicRPY: [{yaml_matrix(rot)}]
""",
        encoding="utf-8",
    )


def export_dataset(args: argparse.Namespace) -> dict:
    args.out.mkdir(parents=True, exist_ok=True)
    clouds_dir = args.out / "clouds_npz"
    clouds_dir.mkdir(parents=True, exist_ok=True)

    for csv_name in ["imu.csv", "gps.csv", "velocity.csv", "clouds.csv"]:
        remove_if_exists(args.out / csv_name)
    if args.write_bag:
        remove_if_exists(args.out / "slam_input.bag")

    typestore = typestore_with_packet()
    half = int(args.window_ms * 1_000_000) if args.center_ns and args.window_ms else None
    start = int(args.center_ns - half) if half is not None else None
    stop = int(args.center_ns + half) if half is not None else None

    counts = ExportCounts()
    cloud_entries: list[dict] = []
    imu_accel = []
    imu_gyro = []
    source_cloud_frames: dict[str, int] = {}
    bag_writer = None
    extrinsic = ouster_design_extrinsic(args.ouster_extrinsic_mode)
    extrinsic_rot = np.array(extrinsic["rotation"], dtype=np.float64)

    with Reader(args.bag) as reader:
        wanted = {CLOUD_TOPIC, IMU_PACKET_TOPIC, GPS_TOPIC, VEL_TOPIC}
        conns = [c for c in reader.connections if c.topic in wanted]
        if not conns:
            raise RuntimeError("No requested SLAM topics found")

        if args.write_bag:
            bag_writer = Writer(args.out / "slam_input.bag")
            bag_writer.open()
            out_cloud_conn = bag_writer.add_connection(OUT_CLOUD_TOPIC, "sensor_msgs/msg/PointCloud2", typestore=typestore)
            out_original_cloud_conn = bag_writer.add_connection(OUT_ORIGINAL_CLOUD_TOPIC, "sensor_msgs/msg/PointCloud2", typestore=typestore)
            out_imu_conn = bag_writer.add_connection(OUT_IMU_TOPIC, "sensor_msgs/msg/Imu", typestore=typestore)
            out_gps_conn = bag_writer.add_connection(OUT_GPS_TOPIC, "sensor_msgs/msg/NavSatFix", typestore=typestore)
            out_vel_conn = bag_writer.add_connection(OUT_VEL_TOPIC, "geometry_msgs/msg/TwistStamped", typestore=typestore)
        else:
            out_cloud_conn = out_original_cloud_conn = out_imu_conn = out_gps_conn = out_vel_conn = None

        kwargs = {}
        if start is not None:
            kwargs["start"] = start
        if stop is not None:
            kwargs["stop"] = stop

        try:
            for conn, ts, raw in reader.messages(connections=conns, **kwargs):
                msg = typestore.deserialize_ros1(raw, conn.msgtype)
                if conn.topic == CLOUD_TOPIC:
                    stamp = stamp_ns(msg, ts)
                    source_frame = str(getattr(getattr(msg, "header", None), "frame_id", ""))
                    source_cloud_frames[source_frame] = source_cloud_frames.get(source_frame, 0) + 1
                    if counts.clouds % max(1, args.npz_every) == 0:
                        npz_path = clouds_dir / f"cloud_{counts.clouds:06d}_{stamp}.npz"
                        cloud_meta = write_cloud_npz(msg, npz_path)
                    else:
                        cloud_meta = {}
                    row = {
                        "index": counts.clouds,
                        "bag_ts_ns": int(ts),
                        "stamp_ns": int(stamp),
                        "height": int(msg.height),
                        "width": int(msg.width),
                        "point_step": int(msg.point_step),
                        "source_frame_id": source_frame,
                        "npz": cloud_meta.get("npz", ""),
                        "valid_points": cloud_meta.get("valid_points", ""),
                    }
                    append_csv(args.out / "clouds.csv", list(row.keys()), row)
                    cloud_entries.append(row)
                    if bag_writer is not None:
                        liosam_cloud = make_liosam_cloud(typestore, msg, stamp, args.lidar_frame)
                        bag_writer.write(out_cloud_conn, int(ts), typestore.serialize_ros1(liosam_cloud, "sensor_msgs/msg/PointCloud2"))
                        bag_writer.write(out_original_cloud_conn, int(ts), raw)
                    counts.clouds += 1

                elif conn.topic == IMU_PACKET_TOPIC:
                    imu_msg, decoded = imu_msg_from_packet(
                        typestore,
                        msg,
                        int(ts),
                        args.imu_frame,
                        counts.imu,
                        args.imu_orientation_mode,
                        extrinsic_rot,
                        args.pretransform_imu_to_lidar,
                    )
                    serialized = typestore.serialize_ros1(imu_msg, "sensor_msgs/msg/Imu")
                    if bag_writer is not None:
                        bag_writer.write(out_imu_conn, int(ts), serialized)
                    accel = decoded["accel_mps2"]
                    gyro = decoded["gyro_radps"]
                    imu_accel.append(accel)
                    imu_gyro.append(gyro)
                    row = {
                        "index": counts.imu,
                        "bag_ts_ns": int(ts),
                        "accel_x_mps2": float(accel[0]),
                        "accel_y_mps2": float(accel[1]),
                        "accel_z_mps2": float(accel[2]),
                        "gyro_x_radps": float(gyro[0]),
                        "gyro_y_radps": float(gyro[1]),
                        "gyro_z_radps": float(gyro[2]),
                        "accel_ts_ns": int(decoded["accel_ts_ns"]),
                        "gyro_ts_ns": int(decoded["gyro_ts_ns"]),
                    }
                    append_csv(args.out / "imu.csv", list(row.keys()), row)
                    counts.imu += 1

                elif conn.topic == GPS_TOPIC:
                    if bag_writer is not None:
                        bag_writer.write(out_gps_conn, int(ts), raw)
                    row = {
                        "index": counts.gps,
                        "bag_ts_ns": int(ts),
                        "stamp_ns": int(stamp_ns(msg, ts)),
                        "latitude": float(getattr(msg, "latitude", float("nan"))),
                        "longitude": float(getattr(msg, "longitude", float("nan"))),
                        "altitude": float(getattr(msg, "altitude", float("nan"))),
                        "position_covariance_type": int(getattr(msg, "position_covariance_type", 0)),
                    }
                    append_csv(args.out / "gps.csv", list(row.keys()), row)
                    counts.gps += 1

                elif conn.topic == VEL_TOPIC:
                    if bag_writer is not None:
                        bag_writer.write(out_vel_conn, int(ts), raw)
                    lin = getattr(getattr(msg, "twist", None), "linear", None)
                    ang = getattr(getattr(msg, "twist", None), "angular", None)
                    row = {
                        "index": counts.velocity,
                        "bag_ts_ns": int(ts),
                        "stamp_ns": int(stamp_ns(msg, ts)),
                        "linear_x": float(getattr(lin, "x", float("nan"))),
                        "linear_y": float(getattr(lin, "y", float("nan"))),
                        "linear_z": float(getattr(lin, "z", float("nan"))),
                        "angular_x": float(getattr(ang, "x", float("nan"))),
                        "angular_y": float(getattr(ang, "y", float("nan"))),
                        "angular_z": float(getattr(ang, "z", float("nan"))),
                        "speed_mps": velocity_norm(msg),
                    }
                    append_csv(args.out / "velocity.csv", list(row.keys()), row)
                    counts.velocity += 1
        finally:
            if bag_writer is not None:
                bag_writer.close()

    imu_accel_np = np.vstack(imu_accel) if imu_accel else np.zeros((0, 3))
    imu_gyro_np = np.vstack(imu_gyro) if imu_gyro else np.zeros((0, 3))
    make_lio_sam_seed(args.out / "lio_sam_params_seed.yaml", args.imu_frame, args.lidar_frame, extrinsic)
    (args.out / "ouster_design_extrinsics.json").write_text(json.dumps(extrinsic, indent=2), encoding="utf-8")

    manifest = {
        "bag": str(args.bag),
        "out": str(args.out),
        "window": {"center_ns": args.center_ns, "window_ms": args.window_ms, "start_ns": start, "stop_ns": stop},
        "source_topics": {
            "cloud": CLOUD_TOPIC,
            "imu_packet": IMU_PACKET_TOPIC,
            "gps": GPS_TOPIC,
            "velocity": VEL_TOPIC,
        },
        "output_topics": {
            "cloud": OUT_CLOUD_TOPIC,
            "original_cloud": OUT_ORIGINAL_CLOUD_TOPIC,
            "imu": OUT_IMU_TOPIC,
            "gps": OUT_GPS_TOPIC,
            "velocity": OUT_VEL_TOPIC,
        },
        "counts": counts.__dict__,
        "source_cloud_frames": source_cloud_frames,
        "cloud_npz_every": args.npz_every,
        "ouster_design_extrinsic": extrinsic,
        "imu_orientation_mode": args.imu_orientation_mode,
        "pretransform_imu_to_lidar": bool(args.pretransform_imu_to_lidar),
        "imu_stats": {
            "accel_norm_mps2_mean": float(np.linalg.norm(imu_accel_np, axis=1).mean()) if len(imu_accel_np) else None,
            "gyro_norm_radps_mean": float(np.linalg.norm(imu_gyro_np, axis=1).mean()) if len(imu_gyro_np) else None,
            "accel_mean_xyz_mps2": imu_accel_np.mean(axis=0).tolist() if len(imu_accel_np) else None,
            "gyro_mean_xyz_radps": imu_gyro_np.mean(axis=0).tolist() if len(imu_gyro_np) else None,
        },
        "outputs": {
            "rosbag": str(args.out / "slam_input.bag") if args.write_bag else None,
            "clouds_csv": str(args.out / "clouds.csv"),
            "imu_csv": str(args.out / "imu.csv"),
            "gps_csv": str(args.out / "gps.csv"),
            "velocity_csv": str(args.out / "velocity.csv"),
            "clouds_npz": str(clouds_dir),
            "lio_sam_seed": str(args.out / "lio_sam_params_seed.yaml"),
            "ouster_design_extrinsics": str(args.out / "ouster_design_extrinsics.json"),
        },
        "notes": [
            "Generated IMU uses bag timestamps for sync.",
            "Original Ouster internal packet timestamps are preserved in imu.csv only.",
            "PointCloud2 messages are copied without modifying header/frame/content.",
            "The LIO-SAM yaml now uses the Ouster design LiDAR/IMU extrinsic as a seed.",
            "IMU orientation mode identity means no attitude is available; accel_tilt estimates roll/pitch from acceleration and fixes yaw to zero.",
            "If pretransform_imu_to_lidar is true, /imu/data acceleration and gyro are already rotated into the cloud frame and LIO-SAM extrinsics should be identity.",
            "If the cloud source frame changes from /os1_lidar to an Ouster sensor frame, regenerate with --ouster-extrinsic-mode sensor_frame.",
        ],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bag", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--center-ns", type=int, default=None)
    ap.add_argument("--window-ms", type=float, default=None)
    ap.add_argument("--npz-every", type=int, default=1, help="Save every Nth cloud as NPZ.")
    ap.add_argument("--write-bag", action="store_true", help="Write remapped ROS1 bag with standard IMU.")
    ap.add_argument("--imu-frame", default="os_imu")
    ap.add_argument("--lidar-frame", default="os1_lidar")
    ap.add_argument(
        "--imu-orientation-mode",
        choices=["identity", "accel_tilt"],
        default="identity",
        help="Orientation written to /imu/data. LIO-SAM needs a valid attitude; use accel_tilt for Ouster packets without orientation.",
    )
    ap.add_argument(
        "--pretransform-imu-to-lidar",
        action="store_true",
        help="Rotate decoded IMU accel/gyro into the selected Ouster cloud frame before writing /imu/data.",
    )
    ap.add_argument(
        "--ouster-extrinsic-mode",
        choices=["lidar_frame", "sensor_frame", "identity"],
        default="lidar_frame",
        help=(
            "Ouster design LiDAR/IMU extrinsic convention for the LIO seed. "
            "Use lidar_frame for this dataset because PointCloud2 frame_id is /os1_lidar."
        ),
    )
    args = ap.parse_args()
    manifest = export_dataset(args)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
