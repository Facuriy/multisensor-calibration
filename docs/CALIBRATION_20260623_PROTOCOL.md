# Calibration 20260623 Protocol

Status: `final_candidate_validated_visually`

This document is the operational source of truth for the 2026-06-23
multisensor calibration session. It supersedes the older May/June exploratory
notes for this dataset.

## Goal

Create a practical RGB-master calibration chain for:

```text
Ouster 3D -> RGB
VIS -> RGB
NIR -> RGB
Thermal -> RGB
```

The chain supports registered previews, plot extraction, orthomosaic
preprocessing and later hyperspectral/hypercube products. RGB is the master
camera because it is sharpest and has the strongest physical link to Ouster.

## Board

```text
total squares:     10 x 7
internal corners:   9 x 6
square size:        0.04 m
```

Only internal corners are used for metric calibration. The outer border row is
slightly irregular.

## Data

Session folder:

```text
data/calibration/new_session/20260623/
```

Important files:

```text
bag_manifest_20260623.csv
bag_manifest_20260623.json
checkerboard_config.json
rgb_intrinsics_20260623.json
homographies_20260623_to_rgb.json
homographies_20260623_to_vis.json
calibration_20260623_final_candidate.json
README.md
```

Original bags are recorded in the manifest under:

```text
X:\PhenoRob_UAVClimate\Projects\MSP_im_Mais\UGV\BAGS\20260623
```

Local bag caches under `runs/` are disposable and can be rebuilt from the
manifest.

## Final Calibration Products

Main package:

```text
data/calibration/new_session/20260623/calibration_20260623_final_candidate.json
runs/calibration_20260623_final_multisensor_calibration/final_multisensor_calibration_20260623.json
```

Compact table:

```text
runs/calibration_20260623_final_multisensor_calibration/final_calibration_compact.csv
```

Current metrics:

```text
RGB intrinsics:     RMS 1.746 px, 32 views
VIS intrinsics:     RMS 2.693 px, 12 used views
NIR intrinsics:     RMS 2.157 px, 16 used views
Thermal intrinsics: RMS 0.638 px, 17 used views

VIS -> RGB homography:      8 pairs,  median 15.76 px
NIR -> RGB homography:      6 pairs,  median 10.80 px
Thermal -> RGB homography: 17 pairs, median 9.87 px

VIS -> RGB physical candidate:      stereo RMS 7.24 px
NIR -> RGB physical candidate:      stereo RMS 4.25 px
Thermal -> RGB physical candidate:  stereo RMS 4.60 px
Ouster -> RGB physical candidate:   strongest physical link, plane median 5.8 mm
```

Interpretation:

```text
VIS/NIR/Thermal -> RGB homographies are good target-plane registration tools.
They are not full-scene 3D registration.

Ouster -> RGB is usable for visual/depth products, but a residual right-side
bias remains visible in final overlays. Treat it as strong candidate, not
millimetric pixel-perfect registration.
```

## Active Scripts

### Extraction and Future Products

`src/extraction/extract_all_bag_images.py`

General ROS1 bag extractor. Use this as the base for future full-experiment
preprocessing. It writes previews, optional raw arrays, frame metadata and GPS
sidecars.

`src/extraction/make_registered_multisensor_orthomosaic.py`

Existing multisensor orthomosaic builder. Next phase should update it to use
the RGB-master calibration package and explicitly handle displacement/motion.

`src/extraction/make_rgb_lidar_fusion_mosaic.py`

Existing RGB/Ouster fusion mosaic builder. Useful for rapid previews, but not
yet a metric SLAM/orthorectification solution.

### RGB and Ouster Calibration

`src/calibration/calibrate_rgb_intrinsics_from_detections.py`

Builds RGB intrinsics from detected checkerboard views.

`src/calibration/calibrate_intrinsics_from_checkerboard_bags.py`

Production camera-intrinsics calibrator for a new checkerboard capture set. It
reads the bag manifest directly, detects the 9x6 internal checkerboard for
RGB/VIS/NIR/Thermal, keeps one best static-frame detection per bag/sensor,
calibrates K/distortion and writes contact-sheet QA plus per-view reprojection
errors. Use this entry point for future intrinsic recalibration before rebuilding
the RGB-master calibration package.

It also has a bounded `--rescue` mode for difficult VIS/NIR/Thermal frames.
This mode reuses the historical RGB-guided panel boxes, tries Photonfocus
band/index combinations, thermal RAW flat-field variants, CLAHE/detail filters
and partial checker patterns. Rescue candidates are written to
`rescue_candidates.csv`, `rescue_candidates.json` and
`rescue_candidates_contactsheet.jpg`. Only candidates marked as full 9x6
subpixel detections in the calibrated image plane are promoted into intrinsic
calibration. Partial grids are intentionally kept as review/registration
evidence, not as K-calibration views.

`src/calibration/scan_20260623_checker_lidar.py`

Scans the 20260623 calibration bags, extracts RGB checker detections and LiDAR
board-plane evidence.

`src/calibration/refine_20260623_ouster_rgb_multipose_6dof.py`

Optimizes the 6DoF Ouster -> RGB transform over multiple checkerboard poses.

`src/calibration/render_20260623_ouster_rgb_overlay_validation.py`

Renders visual Ouster -> RGB overlay checks. It needs either visible original
bags or a local bag cache.

### VIS/NIR/Thermal Detection

`src/calibration/refine_20260623_multisensor_detection.py`

Original RGB/VIS/NIR/Thermal guided detection pass. Kept because later scripts
use its RGB-guided approximate ROIs.

`src/calibration/run_20260623_photonfocus_partial_grid_campaign.py`

Production VIS/NIR detector for this session. It uses RGB-guided ROIs, band
combinations, normalization, enhancement variants and partial checker grids.
This fixed the VIS/NIR failure mode: many boards were cropped.

`src/calibration/run_20260623_all_sensor_grid_campaign.py`

Model-based grid campaign for VIS/NIR/Thermal. It is useful mainly as an
additional NIR/Thermal source and as diagnostic context.

`src/calibration/lab_20260623_thermal_grid_recovery.py`

Thermal workbench for hard cases: flat-fielding, scalar recovery, morphology,
watershed diagnostics and model-based grid recovery.

`src/calibration/thermal_checker.py`

Thermal checker utilities for colormapped JPG/preview images. It recovers a
scalar field from color maps such as inferno.

`src/calibration/thermal_checker_raw.py`

Thermal checker utilities for raw `mono16` or Celsius arrays. Preferred when
raw scalar thermal data is available.

`src/calibration/run_20260623_thermal_production.py`

Low-load production thermal checker detector for the 20260623 bags.

`src/calibration/merge_20260623_thermal_candidates.py`

Merges thermal candidates from RAW/Celsius/scalar/model-assisted sources.

### Registration and Packaging

`src/registration/refine_20260623_vis_homographies.py`

Historical VIS-master target-plane homography refinement.

`src/registration/build_20260623_rgb_master_homographies.py`

Builds RGB-master target-plane homographies.

`src/calibration/build_20260623_final_multisensor_calibration.py`

Builds the final package. It selects filtered VIS/NIR partial grids for
intrinsics/registration and merges RGB, Ouster, VIS, NIR and Thermal.

`src/registration/render_20260623_final_rgb_validation.py`

Final visual validation renderer. It warps VIS/NIR/Thermal to RGB from the
final calibration package and creates normal, false-color and edge overlays.

`src/registration/coregister_rgb_master.py`

Production RGB-master coregistration/crop module. It warps VIS/NIR/Thermal into
RGB, computes the largest axis-aligned rectangle fully inside the common valid
camera mask, writes same-size crops, masks and QA overlays.

## Rebuild Order

Use this order when recreating the calibration from scratch:

```powershell
python src\calibration\calibrate_rgb_intrinsics_from_detections.py
python src\calibration\scan_20260623_checker_lidar.py
python src\calibration\refine_20260623_ouster_rgb_multipose_6dof.py
python src\calibration\refine_20260623_multisensor_detection.py
python src\calibration\run_20260623_photonfocus_partial_grid_campaign.py --resume
python src\calibration\run_20260623_all_sensor_grid_campaign.py --resume
python src\calibration\merge_20260623_thermal_candidates.py
python src\registration\build_20260623_rgb_master_homographies.py
python src\calibration\build_20260623_final_multisensor_calibration.py
python src\registration\render_20260623_final_rgb_validation.py
python src\calibration\render_20260623_ouster_rgb_overlay_validation.py --bag-cache runs\calibration_20260623_raw_bag_cache --out-dir runs\calibration_20260623_final_ouster_rgb_validation
```

Some steps require a local bag cache. If it was deleted to save disk, rebuild it
from `bag_manifest_20260623.csv` or point scripts directly to the original
network bags if Python can access the mapped `X:` drive.

## Final Validation Outputs

```text
runs/calibration_20260623_final_rgb_validation/
runs/calibration_20260623_final_ouster_rgb_validation/
```

The final RGB validation shows VIS/NIR/Thermal aligned well enough on the
checkerboard plane. Ouster validation shows useful alignment but residual bias.

## Next Phase

The next problem is motion/displacement during real plot extraction:

```text
sequential frames + GPS/IMU/Ouster/RGB motion -> stable plot products
```

For orthomosaics and hypercubes, do not assume a static rig over a whole plot.
The next implementation should explicitly estimate or compensate motion using
timestamps, GPS/IMU, image overlap and/or LiDAR odometry.

Before mass extraction, two additional pipeline blocks are required:

## Pre-Extraction Block 1: Radiometric Panel Calibration

The real field campaign includes a radiometric/reflectance panel visible in the
images. It must be detected or annotated so that per-frame/per-sensor
radiometric normalization can be applied.

This is intentionally separate from geometric calibration:

```text
geometric calibration: where pixels correspond
radiometric calibration: how pixel values should be normalized
```

Implementation plan:

```text
1. Try automatic panel detection from a known/expected rig position.
2. Reuse a stable fixed ROI when the panel position is consistent.
3. Provide a manual annotation fallback window for difficult bags.
4. Store panel ROI, method and quality flags in frame metadata.
5. Compute per-sensor/band panel statistics and normalization coefficients.
```

Expected metadata:

```text
radiometric_panel_roi_px
radiometric_panel_detection_method: auto | fixed_roi | manual | missing
radiometric_panel_quality
radiometric_panel_stats_by_sensor
radiometric_normalization_coefficients
```

This can be implemented after the geometric extraction skeleton, but it is
required before publishing final quantitative products.

## Pre-Extraction Block 2: Common Intersection Crop

Implemented by:

```text
src/registration/coregister_rgb_master.py
```

The final extraction should not use the full RGB frame when VIS/NIR/Thermal
cover only a smaller warped region. Every synchronized frame should produce a
common valid area in RGB coordinates:

```text
common_valid_mask = RGB_crop
                  & VIS_to_RGB_valid_mask
                  & NIR_to_RGB_valid_mask
                  & Thermal_to_RGB_valid_mask
```

Ouster should be handled as an additional sparse/depth validity layer:

```text
ouster_valid_mask_rgb
ouster_depth_rgb
ouster_intensity_rgb
```

Recommended outputs:

```text
common_roi_rgb_xyxy
common_valid_mask.png
sensor_valid_masks/
registered_cropped_rgb/
registered_cropped_vis/
registered_cropped_nir/
registered_cropped_thermal/
registered_ouster_depth_or_points/
metadata_valid_area_fraction
```

This common crop/mask is mandatory for orthomosaics and hypercubes, because it
prevents black borders, extrapolated pixels and false pixel-to-pixel
comparisons outside the shared sensor footprint.

Validation on the 20260623 calibration review frames:

```powershell
python src\registration\coregister_rgb_master.py `
  --input-dir runs\calibration_20260623_full_review\per_bag `
  --out-dir runs\calibration_20260623_rgb_master_common_crop_validation `
  --layout review
```

Result:

```text
frames processed: 37 / 37
common ROI in RGB: [649, 1176, 1794, 1743]
crop size: 1145 x 567 px
crop fraction of RGB frame: 12.95%
invalid camera pixels after crop: 0 for RGB/VIS/NIR/Thermal
```

Important limitation:

```text
This guarantees dense camera coverage inside the crop. It does not make Ouster
dense; Ouster remains a sparse/depth layer with its own valid mask.
Pixel agreement is limited by the current calibration accuracy and by the fact
that VIS/NIR/Thermal -> RGB are target-plane homographies.
```
