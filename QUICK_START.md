# Quick Start

Open this folder as the project root:

```text
C:\DATA\ENV\SWISS\multisensor-calibration
```

Create a clean environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements\requirements-calibration.txt
```

Check imports and script syntax:

```powershell
python -m py_compile `
  src\calibration\calibrate_intrinsics_from_checkerboard_bags.py `
  src\calibration\thermal_checker.py `
  src\calibration\thermal_checker_raw.py `
  src\registration\coregister_rgb_master.py
```

Run a lightweight NIR detector test:

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

Inspect:

```text
runs/calibration_intrinsics_test_nir/detections.csv
runs/calibration_intrinsics_test_nir/*contactsheet.jpg
runs/calibration_intrinsics_test_nir/rescue_candidates.csv
```

For context before modifying code:

```text
README.md
LLM_HANDOFF.md
docs/CALIBRATION_20260623_PROTOCOL.md
docs/CALIBRATION_20260623_BAGS_REVIEW.md
```
