# AI-Sports-Coach

FastAPI backend for pose-based sports analysis. The project currently supports
football, athletics, basketball, and badminton analysis endpoints.

## Setup on Windows PowerShell

Run these commands from the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the same PowerShell window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Start the API

```powershell
python -m uvicorn backend.main:app --reload
```

Open <http://127.0.0.1:8000/docs> for the interactive API documentation.

The first analysis that uses YOLO downloads the `yolo11n.pt` model. The
MediaPipe pose model is already included at
`backend/models/trained_models/pose_landmarker_full.task`.

## Run tests

```powershell
python -m pytest -q
```

The video integration tests require their input videos under `uploads/`.
Feature and geometry tests can run without a video.

## Angle summary script

Place a video in `uploads/`, update `VIDEO_PATH` in
`test_angle_summary.py`, then run:

```powershell
python test_angle_summary.py
```