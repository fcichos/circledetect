# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CircleDetection is a Python package for detecting and tracking microparticles in dark field microscopy images and videos. The project focuses on analyzing gold nanoparticles (AuNP) where microparticles (~48 pixels diameter) need to be distinguished from smaller nanoparticles.

## Directory Structure

```
CircleDetection/
├── circledetection/       # Main Python package
│   ├── cli.py            # CLI commands (circledetect)
│   ├── detection.py      # Circle detection algorithms
│   ├── tracking.py       # Kalman filtering and trajectory extraction
│   ├── smoothing.py      # Trajectory smoothing (savgol, spline, moving avg)
│   ├── visualization.py  # Plotting functions
│   └── utils.py          # File I/O utilities
├── scripts/              # Legacy standalone scripts
├── data/                 # Input data (images, videos)
├── results/              # Output files
│   ├── csv/             # Trajectory data
│   ├── videos/          # Annotated videos
│   └── plots/           # Analysis plots
├── setup.py             # Package installation
└── requirements.txt     # Python dependencies
```

## Setup

```bash
# Install package in development mode (creates 'circledetect' command)
pip install -e .

# Or install dependencies only (for using scripts directly)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### CLI Commands (Recommended)

After installing with `pip install -e .`, use the `circledetect` command:

#### Full automatic pipeline (easiest)
```bash
# Process video start to finish with recommended settings
circledetect auto video.mp4 --no-rotation

# Custom smoothing window
circledetect auto video.mp4 --smooth-window 21 --no-rotation
```

#### Extract trajectory
```bash
# With Kalman filtering (default)
circledetect process video.mp4

# With immediate smoothing
circledetect process video.mp4 --smooth savgol --smooth-window 15

# Custom particle size
circledetect process video.mp4 --diameter 50
```

#### Smooth existing trajectory
```bash
# Savitzky-Golay (recommended - preserves features)
circledetect smooth trajectory.csv --method savgol --window 15

# Spline (very smooth)
circledetect smooth trajectory.csv --method spline

# Moving average (fast)
circledetect smooth trajectory.csv --method moving --window 7
```

#### Transform to particle reference frame
```bash
# Center only (no rotation - avoids jitter from Brownian motion)
circledetect transform video.mp4 trajectory.csv --no-rotation

# With rotation to align velocity
circledetect transform video.mp4 trajectory.csv

# With annotation (shows axes and velocity)
circledetect transform video.mp4 trajectory.csv --no-rotation --annotate
```

### Python API

Use CircleDetection as a library in your own scripts:

```python
from circledetection import extract_trajectory, smooth_trajectory
from circledetection.utils import save_trajectory_csv

# Extract trajectory
raw_traj, filtered_traj = extract_trajectory(
    'video.mp4',
    expected_diameter=48,
    use_kalman=True
)

# Convert to dict format
trajectory = [
    {'frame': f, 'x': x, 'y': y, 'radius': r}
    for f, x, y, r in filtered_traj
]

# Smooth trajectory
smoothed = smooth_trajectory(
    trajectory,
    method='savgol',
    window_length=15
)

# Save to CSV
save_trajectory_csv(smoothed, 'trajectory_smooth.csv')
```

### Legacy Scripts

Original scripts in `scripts/` directory still work:

```bash
python scripts/extract_trajectory_stabilized.py data/video.mp4
python scripts/smooth_trajectory.py results/csv/trajectory.csv --method savgol --window 15
python scripts/transform_to_particle_frame.py data/video.mp4 results/csv/trajectory.csv --no-rotation
python scripts/plot_trajectory.py results/csv/trajectory.csv
```

## Architecture

### Code Organization

The package is organized into modular components:

- **`detection.py`**: Core circle detection using Hough Circle Transform
  - `detect_circle_robust()`: Main detection function with subpixel refinement
  - Handles prediction-based search regions for tracking
  - Robust centroid calculation that removes outlier bright pixels (nanoparticles)

- **`tracking.py`**: Trajectory extraction with Kalman filtering
  - `KalmanTracker`: Implements constant velocity Kalman filter (state: x, y, vx, vy)
  - `extract_trajectory()`: Main video processing function
  - Returns both raw and filtered trajectories for comparison

- **`smoothing.py`**: Post-processing trajectory smoothing
  - Three methods: Savitzky-Golay (preserves features), spline (smooth curves), moving average (fast)
  - `calculate_smoothness()`: Computes RMS of second derivative for quantitative comparison

- **`cli.py`**: Click-based command-line interface
  - Four main commands: `auto`, `process`, `smooth`, `transform`
  - Handles file I/O, progress reporting, output directory management

- **`utils.py`**: CSV file I/O utilities
  - `load_trajectory_csv()`, `save_trajectory_csv()`
  - Standard format: frame, x, y, radius

- **`visualization.py`**: Plotting functions for trajectory analysis

### Detection Algorithm

Uses OpenCV's Hough Circle Transform with parameters tuned for ~48 pixel diameter circles:
- Gaussian blur preprocessing (5x5 kernel) for noise reduction
- Radius range: 40-70% of expected diameter (19-33 pixels for 48px diameter)
- Subpixel refinement: Intensity-weighted centroid within circular mask (0.1-0.9 pixel precision)
- Robust centroid: Clips outlier bright pixels (nanoparticles) using median + 2σ threshold

### Kalman Filtering (2x stability improvement)

Constant velocity model with 4D state [x, y, vx, vy]:
- Process noise: 0.03 (trusts model moderately)
- Measurement noise: 0.5 (accounts for detection uncertainty)
- Motion prediction limits search region to ±50 pixels from predicted position
- Correlation with raw detection: >0.91 (maintains accuracy while reducing noise)

### Smoothing Methods (up to 5x improvement)

Post-processing smoothing applied to trajectory CSV:
- **Savitzky-Golay** (default): Polynomial fitting in sliding window, preserves peaks
- **Spline**: Smooth curve interpolation for gradual motion
- **Moving average**: Simple uniform kernel, fastest

Typical smoothness scores (lower = better):
- Raw detection: ~1.5
- + Kalman: ~0.7 (2x improvement)
- + Savgol (window=15): ~0.4 (3.5x improvement)
- + Savgol (window=21): ~0.3 (5x improvement)

### Reference Frame Transformation

Centers microparticle and optionally aligns velocity:
- Translation: Particle centered at each frame (appears stationary)
- Rotation (optional): Aligns velocity vector horizontally (enables flow field tracking)
- Velocity smoothing (moving average) reduces rotation jitter from Brownian motion
- Use `--no-rotation` to avoid jitter when motion is primarily linear
- Black padding for rotated frame boundaries
- Saves transformation parameters: tx, ty, rotation_angle_deg, vx, vy

### Output Files

Standard output structure in `results/`:
- **CSV trajectories**: `results/csv/` - frame, x, y, radius with subpixel precision
- **Annotated videos**: `results/videos/` - overlays circles and tracking info
- **Transformed videos**: `results/videos/` - particle-centered reference frame
- **Plots**: `results/plots/` - 2D trajectory, position vs time, displacement analysis
