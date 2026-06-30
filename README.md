# Multisensor Calibration

This is a standalone calibration working repo extracted from
`C:\DATA\ENV\SWISS`.

It contains the scripts, manifests, calibration packages, detection summaries
and documentation needed to improve the current intrinsic/extrinsic calibration
from the existing 2026-06-23 checkerboard bags. The original ROS bags are not
copied here because they are large; their absolute paths are preserved in
`bag_links/bag_manifest_20260623_with_original_paths.csv`.

## Current Goal

Improve the existing calibration without acquiring new data.

The weak parts are not RGB. They are:

```text
VIS:     many partial checkerboards, only a few full 9x6 views
NIR:     many partial checkerboards, no strong full 9x6 set in final selection
Thermal: model-assisted/partial checker recovery, useful but needs caution
```

The strong parts are:

```text
RGB intrinsics: 32 views, RMS about 1.75 px
Ouster -> RGB: multipose 6DoF candidate, visually usable
RGB-master homographies: useful for target-plane registration/crop
```

## Checkerboard

```text
total squares:      10 x 7
internal corners:   9 x 6
square size:        0.04 m
outer border:       slightly irregular; use internal corners only
```

Config:

```text
data/calibration/new_session/20260623/checkerboard_config.json
```

## Original Bag Location

Main bag folder:

```text
X:\PhenoRob_UAVClimate\Projects\MSP_im_Mais\UGV\BAGS\20260623
```

Manifest with exact bag paths:

```text
bag_links/bag_manifest_20260623_with_original_paths.csv
data/calibration/new_session/20260623/bag_manifest_20260623.csv
```

Important topics:

```text
RGB:         /ssf/BFS_usb_0/image_raw
VIS:         /ssf/photonfocus_camera_vis_node/image_raw
NIR:         /ssf/photonfocus_camera_nir_node/image_raw
Thermal C:   /ssf/thermalgrabber_ros/image_deg_celsius
Thermal RAW: /ssf/thermalgrabber_ros/image_mono16
Ouster:      /ssf/os1_cloud_node/points
```

## Environment

Current working environment on the original machine:

```text
Python executable: C:\Python312\python.exe
Python version:    3.12.6
Virtual env:       none detected; system Python312 is being used
```

Install a clean equivalent environment with:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements\requirements-calibration.txt
```

Some scripts require ROS1 bag support through `rosbags`, not a live ROS install.
FAST-LIO/LIO work was done separately in WSL and is documented in
`docs/LIO_SLAM_STATUS_20260627.md`.

## Most Important Files

Final candidate package:

```text
data/calibration/new_session/20260623/calibration_20260623_final_candidate.json
runs_summaries/calibration_20260623_final_multisensor_calibration/final_multisensor_calibration_20260623.json
runs_summaries/calibration_20260623_final_multisensor_calibration/final_calibration_compact.csv
```

Main intrinsic detector with rescue mode:

```text
src/calibration/calibrate_intrinsics_from_checkerboard_bags.py
```

Historical and specialized detectors:

```text
src/calibration/refine_20260623_multisensor_detection.py
src/calibration/deep_refine_20260623_photonfocus_checker.py
src/calibration/run_20260623_photonfocus_partial_grid_campaign.py
src/calibration/run_20260623_all_sensor_grid_campaign.py
src/calibration/run_20260623_thermal_production.py
src/calibration/lab_20260623_thermal_grid_recovery.py
src/calibration/thermal_checker.py
src/calibration/thermal_checker_raw.py
```

Ouster/RGB:

```text
src/calibration/scan_20260623_checker_lidar.py
src/calibration/refine_20260623_ouster_rgb_multipose_6dof.py
src/calibration/render_20260623_ouster_rgb_overlay_validation.py
```

RGB-master registration:

```text
src/registration/build_20260623_rgb_master_homographies.py
src/registration/coregister_rgb_master.py
src/registration/render_20260623_final_rgb_validation.py
```

Ortomosaic/extraction previews:

```text
src/extraction/extract_all_bag_images.py
src/extraction/make_rgb_master_multisensor_orthomosaic.py
src/extraction/run_rgb_master_orthomosaic_batch.py
```

## Current Calibration Metrics

From `final_calibration_compact.csv`:

```text
RGB intrinsics:      RMS 1.746 px, 32 views
VIS intrinsics:      RMS 2.693 px, partial subpixel candidate
NIR intrinsics:      RMS 2.157 px, partial subpixel candidate
Thermal intrinsics:  RMS 0.638 px, model-assisted candidate

VIS -> RGB homography:      8 pairs,  median 15.76 px
NIR -> RGB homography:      6 pairs,  median 10.80 px
Thermal -> RGB homography: 17 pairs, median 9.87 px

T_rgb_vis physical candidate:      RMS 7.24 px
T_rgb_nir physical candidate:      RMS 4.25 px
T_rgb_thermal physical candidate:  RMS 4.60 px
```

Interpretation:

```text
Use current calibration for previews and pipeline development.
Do not claim final millimetric pixel-perfect calibration yet.
Main improvement opportunity: recover more reliable full 9x6 or strong large
partial checker detections in VIS/NIR/Thermal from the existing bags.
```

## Recommended First Commands

Smoke test one sensor:

```powershell
python src\calibration\calibrate_intrinsics_from_checkerboard_bags.py `
  --manifest data\calibration\new_session\20260623\bag_manifest_20260623.csv `
  --checker-config data\calibration\new_session\20260623\checkerboard_config.json `
  --initial-intrinsics data\matrices\initial_camera_intrinsics_from_report.json `
  --sensors nir `
  --limit-bags 2 `
  --max-frames-per-bag 1 `
  --rescue `
  --rescue-always `
  --rescue-max-frames-per-bag 1 `
  --rescue-max-variants 8 `
  --out runs\calibration_intrinsics_test_nir
```

Full careful detector run:

```powershell
python src\calibration\calibrate_intrinsics_from_checkerboard_bags.py `
  --manifest data\calibration\new_session\20260623\bag_manifest_20260623.csv `
  --checker-config data\calibration\new_session\20260623\checkerboard_config.json `
  --initial-intrinsics data\matrices\initial_camera_intrinsics_from_report.json `
  --sensors rgb,vis,nir,thermal_c,thermal_raw `
  --max-frames-per-bag 2 `
  --min-views 8 `
  --rescue `
  --out runs\calibration_intrinsics_checkerboard_bags_20260623
```

Stronger but slower rescue for a problem sensor:

```powershell
python src\calibration\calibrate_intrinsics_from_checkerboard_bags.py `
  --manifest data\calibration\new_session\20260623\bag_manifest_20260623.csv `
  --checker-config data\calibration\new_session\20260623\checkerboard_config.json `
  --initial-intrinsics data\matrices\initial_camera_intrinsics_from_report.json `
  --sensors thermal_c `
  --max-frames-per-bag 2 `
  --rescue `
  --deep `
  --crops `
  --rescue-exhaustive `
  --out runs\calibration_intrinsics_thermal_rescue_deep
```

## Rule For Using Detections

For intrinsics:

```text
Safe:      full 9x6 subpixel detections in the calibrated image plane
Cautious:  large partial subpixel detections if no full board exists
Avoid:     model-recovered or alternate-plane detections unless manually reviewed
```

For extrinsics/registration:

```text
Partial grids can be useful.
Model-assisted thermal grids can be useful.
Always keep confidence labels and review contact sheets.
```

## Guidance For Another LLM

Read these first:

```text
docs/CALIBRATION_20260623_PROTOCOL.md
docs/CALIBRATION_20260623_BAGS_REVIEW.md
LLM_HANDOFF.md
```

Then inspect:

```text
runs_summaries/calibration_20260623_refined_multisensor_detection/
runs_summaries/calibration_20260623_deep_photonfocus_detection/
runs_summaries/calibration_20260623_thermal_candidates_merged/
```

The likely best improvement is not rewriting the whole calibration. It is:

```text
1. improve VIS/NIR/Thermal checker detection on existing frames;
2. preserve confidence labels;
3. rerun intrinsics/extrinsics comparison;
4. compare against current final candidate visually and numerically.
```
