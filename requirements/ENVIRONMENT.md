# Environment

Original machine environment used during development:

```text
OS: Windows
workspace: C:\DATA\ENV\SWISS
Python executable: C:\Python312\python.exe
Python version: 3.12.6
virtual environment: none detected
```

Important installed packages observed:

```text
opencv-python / cv2: 4.11.0
numpy: 2.3.2
scipy: 1.15.2
pandas: 2.2.3
rosbags: 0.11.1
geopandas: 1.0.1
rasterio: 1.4.3
shapely: 2.0.7
pyproj: 3.7.1
```

Recommended clean setup:

```powershell
cd C:\DATA\ENV\SWISS\multisensor-calibration
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements\requirements-calibration.txt
```

No live ROS installation is required for the Python bag-processing scripts.
They read ROS1 bags through `rosbags`.

FAST-LIO / LIO-SLAM experiments are documented separately in:

```text
docs/LIO_SLAM_STATUS_20260627.md
```

Those were run in WSL/ROS and are not required for checkerboard detection.
