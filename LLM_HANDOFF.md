# Handoff For Codex / Claude

## User Intent

The user will not acquire new checkerboard captures. Treat the current
2026-06-23 bags as the best available calibration dataset.

Your task is to improve detection/calibration from the existing data, especially
VIS, NIR and Thermal.

Do not start from zero. The current calibration package is already useful.

## Main Problem

RGB is strong. VIS/NIR/Thermal are weaker because many checkerboards are
partial or low-contrast.

Known current state:

```text
RGB intrinsics:      32 views, RMS ~1.75 px
VIS intrinsics:      partial subpixel candidate, RMS ~2.69 px
NIR intrinsics:      partial subpixel candidate, RMS ~2.16 px
Thermal intrinsics:  model-assisted candidate, RMS ~0.64 px
```

The issue is not only detector recall. It is also confidence:

```text
full 9x6 subpixel detection      -> good for intrinsics
large partial subpixel detection -> useful but handle carefully
model-assisted thermal detection -> useful, but review before using for K
panel-only detection             -> useful for ROI/registration, not for K
```

## Dataset

Manifest:

```text
data/calibration/new_session/20260623/bag_manifest_20260623.csv
bag_links/bag_manifest_20260623_with_original_paths.csv
```

Original bags:

```text
X:\PhenoRob_UAVClimate\Projects\MSP_im_Mais\UGV\BAGS\20260623
```

Checkerboard:

```text
10 x 7 total squares
9 x 6 internal corners
square size 0.04 m
outer border slightly irregular; use internal corners only
```

## Important Scripts

Use this first:

```text
src/calibration/calibrate_intrinsics_from_checkerboard_bags.py
```

It has the newest `--rescue` mode:

```text
RGB-guided ROI
Photonfocus VIS/NIR band variants
NIR/VIS index-like band combinations
thermal RAW/flat-field/CLAHE/detail variants
partial-grid candidate reporting
promotion to intrinsics only for full 9x6 subpixel detections
```

Older/specialized scripts worth mining:

```text
src/calibration/run_20260623_photonfocus_partial_grid_campaign.py
src/calibration/run_20260623_all_sensor_grid_campaign.py
src/calibration/run_20260623_thermal_production.py
src/calibration/lab_20260623_thermal_grid_recovery.py
src/calibration/thermal_checker.py
src/calibration/thermal_checker_raw.py
src/calibration/deep_refine_20260623_photonfocus_checker.py
```

## Outputs To Inspect Before Coding

```text
runs_summaries/calibration_20260623_final_multisensor_calibration/final_calibration_compact.csv
runs_summaries/calibration_20260623_refined_multisensor_detection/refined_multisensor_detection_table.csv
runs_summaries/calibration_20260623_refined_multisensor_detection/refined_multisensor_detection_summary.json
runs_summaries/calibration_20260623_deep_photonfocus_detection/deep_photonfocus_detection_table.csv
runs_summaries/calibration_20260623_thermal_candidates_merged/thermal_candidates_merged_table.csv
```

Review images:

```text
runs_summaries/calibration_20260623_refined_multisensor_detection/refined_detection_page_*.jpg
runs_summaries/calibration_20260623_checker_lidar_scan/detection_review_page_*.jpg
```

## Suggested Improvement Plan

1. Run current detector on a small subset and reproduce current behavior.
2. Focus on failed/partial VIS, NIR and Thermal cases.
3. Try to recover more full 9x6 detections, but do not accept false positives.
4. If only partial grids are possible, keep them as partial with confidence.
5. Recompute intrinsics/extrinsics and compare against the current candidate.
6. Do visual validation with RGB-master overlays.

## Things Not To Break

Do not overwrite the current final candidate unless explicitly requested.

Current final candidate:

```text
data/calibration/new_session/20260623/calibration_20260623_final_candidate.json
```

Write new attempts under:

```text
runs/
```

Suggested output names:

```text
runs/calibration_intrinsics_checkerboard_bags_20260623_experiment_NAME
runs/calibration_20260623_final_candidate_experiment_NAME
```

## What A Better Result Should Show

For intrinsics:

```text
more full 9x6 views, or more large reliable subpixel partials
lower or comparable reprojection RMS
fewer outlier removals
contact sheets visually plausible
```

For extrinsics:

```text
lower RGB registration residuals
better visual RGB/VIS/NIR/Thermal overlay
no degradation on held-out poses
```

For Ouster/RGB:

```text
do not solve this via 2D homography
use plane evidence + RGB checker pose + LiDAR intensity/depth validation
```

## Known Caveat

Homographies VIS/NIR/Thermal -> RGB are useful for target-plane visual products
and common crops. They are not full physical 3D calibration for arbitrary depth.
