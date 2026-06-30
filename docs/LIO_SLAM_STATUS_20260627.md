# LIO / SLAM Status - 2026-06-27

This note documents the first professional SLAM/LIO attempt for the SWISS UGV
multisensor pipeline. The goal is to estimate `pose(t)` for RGB/VIS/NIR/Thermal
orthomosaic generation using the Ouster LiDAR, IMU, GPS, velocity and timestamps.

## Environment

Processing was set up in WSL2 because the serious ROS LiDAR-inertial packages
are Linux/ROS-first.

- WSL distro: `Ubuntu-20.04-LIO`
- WSL path: `C:\DATA\WSL\Ubuntu-20.04-LIO`
- ROS: Noetic
- LIO package: LIO-SAM
- GTSAM: built from source, version 4.0.3
- LIO-SAM workspace: `/root/lio_ws`

LIO-SAM builds successfully and can run headless on exported SWISS bags.

## Source Data

Test bag:

```text
X:\PhenoRob_UAVClimate\Projects\MSP_im_Mais\UGV\BAGS\20260601\2026-06-01-14-59-48.bag
```

Exported dataset:

```text
runs\slam_dataset_plot17_20260601_lio_imu_lidar_frame
```

Window:

```text
center_ns: 1780318960714678272
window_ms: 7000
duration: about 14 s
```

Counts:

```text
clouds:   140
imu:     1400
gps:       35
velocity:  35
```

Ouster cloud topic:

```text
/ssf/os1_cloud_node/points
```

Exported LIO-SAM topics:

```text
/points_raw
/imu/data
/gps/fix
/gps/vel
```

The Ouster cloud frame in the bag is:

```text
/os1_lidar
```

The PointCloud2 fields are:

```text
x, y, z, intensity, t, reflectivity, ring, noise, range
```

The Ouster `t` field is in nanoseconds within each scan. LIO-SAM converts it
with:

```cpp
dst.time = src.t * 1e-9f;
```

So the per-point LiDAR time scaling is correct.

## Ouster LiDAR / IMU Extrinsic Seed

No per-unit Ouster metadata was found inside the ROS bag. This is acceptable for
the first LIO attempt because the published PointCloud2 already contains XYZ
coordinates from the Ouster driver.

The Ouster OS1 design transforms were used as the initial mechanical seed:

```text
lidar_to_sensor_transform =
[-1, 0, 0, 0,
  0,-1, 0, 0,
  0, 0, 1, 38.195,
  0, 0, 0, 1] mm

imu_to_sensor_transform =
[1, 0, 0, 6.253,
 0, 1, 0,-11.775,
 0, 0, 1, 7.645,
 0, 0, 0, 1] mm
```

Because the cloud frame is `/os1_lidar`, the seed transform is:

```text
T_lidar_to_imu = inv(imu_to_sensor) * lidar_to_sensor
translation_m = [-0.006253, 0.011775, 0.03055]
rotation      = diag(-1, -1, 1)
```

For the latest test, IMU acceleration and gyro were pre-rotated into the LiDAR
frame before writing `/imu/data`, and LIO-SAM was run with identity extrinsics.

## LIO-SAM Results

### First working setup

Output:

```text
runs\lio_sam_plot17_accel_tilt_first_test
```

Short 8 s playback:

```text
poses:       39
delta:       [6.78, 1.83, -0.94] m
path length: 10.56 m
```

This was much better than the original invalid-orientation attempt, but still
not proven correct.

### Full window, IMU pre-rotated into LiDAR frame

Output:

```text
runs\lio_sam_plot17_imu_lidar_full
```

LIO-SAM published:

```text
/lio_sam/mapping/cloud_registered    69 msgs
/lio_sam/mapping/odometry            69 msgs
/lio_sam/mapping/path                69 msgs
/odometry/imu                      1350 msgs
/odometry/imu_incremental          1350 msgs
```

Metrics:

```text
LIO poses:                  69
LIO delta XYZ:              [31.25, 15.29, 5.80] m
LIO horizontal path length: 36.01 m
LIO 3D path length:         40.33 m

GPS count:                  35
GPS delta ENU:              [-1.50, 1.15, 1.00] m
GPS horizontal path length: 3.39 m
GPS 3D path length:         3.88 m
```

Velocity topic sanity check:

```text
mean speed:     0.152 m/s
median speed:   0.154 m/s
approx distance over 14 s: 2.13 m
```

Conclusion: this LIO-SAM trajectory is not valid yet. It overestimates movement
by roughly one order of magnitude relative to GPS and GNSS velocity.

Trajectory plot:

```text
runs\lio_sam_plot17_imu_lidar_full\trajectory_lio_vs_gps.png
```

## KISS-ICP LiDAR-Only Check

KISS-ICP was installed natively in Windows with:

```text
python -m pip install kiss-icp
```

This was used as a geometry-only sanity check, without IMU and without GPS.

Output:

```text
runs\kiss_icp_plot17_full_no_deskew
```

Metrics:

```text
poses:                  140
mean points per scan:   61631
delta XYZ:              [-0.075, 0.017, -0.005] m
horizontal path length: 0.472 m
3D path length:         0.481 m
```

Conclusion: LiDAR-only registration is also not reliable on this segment. It
almost sticks in place while GPS/velocity indicate about 2-4 m of movement.

This suggests the downward-looking Ouster view over plants/ground does not give
enough stable 3D structure for pure ICP in this short plot segment, or that the
map is dominated by repetitive/near-planar geometry.

Trajectory plot:

```text
runs\kiss_icp_plot17_full_no_deskew\trajectory_kiss_icp_vs_gps.png
```

## Current Diagnosis

The SLAM software stack is installed and operational, but the trajectory is not
yet usable for production orthomosaics.

Known-good facts:

- ROS Noetic and LIO-SAM run in WSL2.
- LIO-SAM receives clouds, IMU and publishes odometry/path.
- Ouster PointCloud2 per-point time field is present and correctly interpreted.
- IMU packets are decoded to SI units.
- Input IMU quaternions are normalized.
- GPS and velocity provide plausible low-speed motion for the selected segment.

Remaining problem:

```text
LIO-SAM is dominated by IMU/gravity/attitude/convention errors, while LiDAR-only
ICP is under-constrained by the downward-looking agricultural scene.
```

The latest exported IMU in LiDAR frame has mean acceleration approximately:

```text
[-0.33, -9.62, -0.47] m/s^2
```

This means gravity is mostly along the LiDAR `-Y` axis in the current convention.
That can be valid only if the IMU orientation is handled consistently. LIO-SAM is
very sensitive to this because it expects a usable attitude estimate.

## Decision

Do not use the current LIO-SAM output as final `pose(t)` for orthomosaic
generation.

The immediate production-safe fallback is:

```text
GNSS velocity / GPS / timestamps -> smooth 1D forward trajectory
```

for plot-level orthomosaics, combined with the already calibrated
RGB-master multisensor registration.

The research/professional SLAM path remains open, but it needs a controlled IMU
convention campaign.

## Recommended Next Experiments

1. Build a small automatic convention sweep for the Ouster IMU.
   Test axis/sign transforms and gravity sign hypotheses, then score each
   trajectory against GPS and GNSS velocity.

2. Add a GPS odometry topic for LIO-SAM.
   LIO-SAM expects `nav_msgs/Odometry`, not raw `NavSatFix`, so `/gps/fix` must
   be converted to `/odometry/gps` or passed through `robot_localization`.

3. Test a longer, straighter bag segment with more geometric structure.
   Short downward-looking plant rows are difficult for LiDAR-only ICP.

4. Use GNSS/velocity as the first orthomosaic pose source.
   This is likely more stable for the immediate extraction pipeline than the
   current LIO output.

5. If high-precision SLAM remains required, run LI-Init or a dedicated
   LiDAR-IMU calibration motion with strong 6DoF excitation.

## FAST-LIO Update - 2026-06-28

FAST-LIO was installed and built successfully in the same WSL2 ROS Noetic
environment.

Workspace:

```text
/root/fastlio_ws
```

Installed/built components:

```text
Livox-SDK
livox_ros_driver
FAST_LIO
```

Two small patches were required:

```text
FAST_LIO/CMakeLists.txt
  - add devel/include to include dirs
  - add explicit generated-message dependency for fastlio_mapping

FAST_LIO/src/preprocess.h
  - use Ouster field "noise" instead of "ambient"
```

The important data-interface discovery:

```text
/points_raw
  LIO-SAM-style converted cloud
  fields: x, y, z, intensity, ring, time
  NOT suitable for FAST-LIO Ouster mode

/points_ouster
  original Ouster cloud copied from the bag
  fields: x, y, z, intensity, t, reflectivity, ring, noise, range
  suitable for FAST-LIO Ouster mode
```

FAST-LIO must use:

```yaml
common:
  lid_topic: "/points_ouster"
  imu_topic: "/imu/data"

preprocess:
  lidar_type: 3
  scan_line: 64
  timestamp_unit: 3
  blind: 0.35
```

The first wrong FAST-LIO test used `/points_raw` with `lidar_type: 3`, which
caused repeated warnings:

```text
Failed to find match for field 't'
Failed to find match for field 'reflectivity'
Failed to find match for field 'ring'
```

After switching to `/points_ouster`, those field warnings disappeared.

### FAST-LIO Result On Plot 17

Dataset:

```text
runs\slam_dataset_plot17_20260601_lio_imu_lidar_frame\slam_input.bag
```

FAST-LIO output:

```text
runs\fastlio_plot17_ouster_full
```

Published:

```text
/Odometry          135 msgs
/cloud_registered 135 msgs
/path              13 msgs
```

Metrics:

```text
FAST-LIO horizontal delta:       2.04 m
FAST-LIO horizontal path length: 3.27 m
FAST-LIO 3D path length:         3.85 m
FAST-LIO vertical delta:        -0.43 m

GPS horizontal delta:            1.89 m
GPS horizontal path length:      3.39 m
GPS 3D path length:              3.88 m

GNSS velocity integral approx:   2.13 m
```

This is the first useful SLAM/LIO result: the horizontal scale is now close to
GPS and GNSS velocity. FAST-LIO is therefore a much better fit than LIO-SAM for
the Ouster internal 6-axis IMU.

The trajectory is not yet georeferenced or heading-aligned to GPS/ENU. The next
step is to fuse or align FAST-LIO with GPS:

```text
FAST-LIO relative odometry + GPS/velocity absolute constraint
-> robot_localization EKF or offline similarity/SE2 alignment
-> pose(t) for RGB-master orthomosaic generation
```

Generated checks:

```text
runs\fastlio_plot17_ouster_full\trajectory_fastlio_vs_gps.png
runs\fastlio_plot17_ouster_full\fastlio_registered_cloud_preview.png
runs\fastlio_plot17_ouster_full\fastlio_metrics.json
```
