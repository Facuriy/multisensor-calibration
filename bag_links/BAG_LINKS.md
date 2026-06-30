# Bag Links

The original ROS1 bags are not copied into this repo. They are large and remain
on the network/project drive.

Main calibration bag folder:

```text
X:\PhenoRob_UAVClimate\Projects\MSP_im_Mais\UGV\BAGS\20260623
```

Manifest files in this repo:

```text
bag_manifest_20260623_with_original_paths.csv
bag_manifest_20260623_with_original_paths.json
```

The same manifests are also available under:

```text
data/calibration/new_session/20260623/bag_manifest_20260623.csv
data/calibration/new_session/20260623/bag_manifest_20260623.json
```

The manifest contains:

```text
bag
bag_path
label_norm
height_level
pose
include_default
rgb_msgs
vis_msgs
nir_msgs
thermal_c_msgs
thermal_raw_msgs
ouster_msgs
```

Only lightweight extracted products are included here:

```text
runs_summaries/*/*.csv
runs_summaries/*/*.json
runs_summaries/*/*.jpg
runs_summaries/*/*.txt
```

Full frame extraction is intentionally not included. If needed, use:

```powershell
python src\extraction\extract_all_bag_images.py `
  --bag-root X:\PhenoRob_UAVClimate\Projects\MSP_im_Mais\UGV\BAGS\20260623 `
  --out runs\extracted_20260623 `
  --topics rgb vis nir thermal_c thermal_raw `
  --limit-bags 2
```

For calibration work, prefer reading directly from bags via the manifest unless
you explicitly need extracted images for manual review.
