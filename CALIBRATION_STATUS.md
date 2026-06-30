# Calibration Status

## What Works

```text
RGB intrinsics:
  32 views
  RMS about 1.75 px
  strongest camera component

Ouster -> RGB:
  multipose 6DoF candidate
  usable for depth/intensity visualization
  strongest physical 3D link

RGB-master target-plane registration:
  VIS/NIR/Thermal can be warped to RGB for planar/crop/preview products
  useful for orthomosaic prototypes and common valid crop
```

## What Is Weak

VIS:

```text
36 total recovered detections
only 2 full 9x6 detections in candidate selection
intrinsics use partial subpixel detections
RMS about 2.69 px
weakest optical camera link
```

NIR:

```text
36 total recovered detections
intrinsics use partial subpixel detections
final selected intrinsic set has no strong full 9x6 group
RMS about 2.16 px
better than VIS but still partial-driven
```

Thermal:

```text
17 observations
8x5 and 9x5 partial/model-assisted observations
RMS about 0.64 px
numeric RMS is good, but confidence is lower because recovery is model-assisted
```

## Why The Current Calibration Is Not Bad

The current calibration is good enough for:

```text
visual previews
registered RGB/VIS/NIR/Thermal crops
plot-level orthomosaic prototypes
QGIS preliminary layers
debugging Ouster/RGB projection
```

It is not yet strong enough to claim:

```text
millimetric pixel-perfect multisensor alignment
full-scene 3D physical camera registration
final quantitative hyperspectral/thermal/depth fusion
```

## Best Improvement Without New Captures

1. Improve checkerboard detection on existing VIS/NIR/Thermal bags.
2. Recover more full 9x6 detections if possible.
3. If only partial detections are possible, keep confidence labels explicit.
4. Recompute intrinsics and extrinsics as experiments, not overwriting the current final candidate.
5. Compare against the current final candidate using:

```text
reprojection RMS
view outlier count
registration residuals to RGB
visual overlay validation
held-out pose validation
```

## Do Not Mix Confidence Levels

Recommended use:

```text
full 9x6 subpixel detections:
  use for intrinsics

large partial subpixel detections:
  use carefully; better for extrinsics/registration than for K

model-assisted thermal grids:
  useful but must be reviewed

panel-only / contrast-only:
  ROI evidence only, not calibration corners
```
