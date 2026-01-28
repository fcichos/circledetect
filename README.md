# CircleDetection

Microparticle tracking for dark field microscopy. Detect, track, and analyze gold nanoparticle (AuNP) motion in video with advanced stabilization and reference frame transformation.

## Features

- **Particle Detection**: Hough Circle Transform with subpixel accuracy (0.1-0.9 pixel precision)
- **Kalman Filtering**: Smooth tracking with 2x improvement in stability
- **Trajectory Smoothing**: Post-processing with Savitzky-Golay, spline, or moving average filters (up to 5x improvement)
- **Reference Frame Transform**: Center particle and optionally align velocity vector
- **CLI Commands**: Easy-to-use command-line interface
- **Python API**: Use as a library in your own scripts

## Installation

```bash
# Clone/navigate to repository
cd CircleDetection

# Install package (creates 'circledetect' command)
pip install -e .
```

## Quick Start

### One-Command Processing

Process an MP4 video with one command:

```bash
circledetect auto video.mp4 --no-rotation
```

This automatically:
1. Extracts trajectory with Kalman filtering
2. Applies Savitzky-Golay smoothing
3. Transforms video to particle reference frame
4. Saves all outputs to `./results/`

### Output Files

- `results/csv/` - Trajectory CSVs
- `results/videos/` - Tracking and transformed videos

## CLI Commands

### `circledetect auto` - Full Pipeline (Recommended)

Process video start to finish:

```bash
# Default (Savitzky-Golay smoothing, no rotation)
circledetect auto video.mp4

# No rotation (avoids jitter from Brownian motion)
circledetect auto video.mp4 --no-rotation

# Custom smoothing window
circledetect auto video.mp4 --smooth-window 21 --no-rotation

# No smoothing
circledetect auto video.mp4 --smooth none
```

### `circledetect process` - Extract Trajectory

Extract particle trajectory from video:

```bash
# Basic (with Kalman filtering)
circledetect process video.mp4

# With immediate smoothing
circledetect process video.mp4 --smooth savgol --smooth-window 15

# Custom particle size
circledetect process video.mp4 --diameter 50

# Custom output directory
circledetect process video.mp4 --output-dir ./my_results
```

### `circledetect smooth` - Smooth Trajectory

Apply post-processing smoothing to existing trajectory:

```bash
# Savitzky-Golay filter (recommended)
circledetect smooth trajectory.csv --method savgol --window 15

# Spline smoothing (very smooth curves)
circledetect smooth trajectory.csv --method spline

# Moving average (simple, fast)
circledetect smooth trajectory.csv --method moving --window 7

# Custom output path
circledetect smooth trajectory.csv --method savgol --output smooth_traj.csv
```

### `circledetect transform` - Transform Video

Transform video to particle reference frame:

```bash
# Center only (no rotation - avoids jitter)
circledetect transform video.mp4 trajectory.csv --no-rotation

# Center + rotate to align velocity
circledetect transform video.mp4 trajectory.csv

# With annotation (shows axes and velocity)
circledetect transform video.mp4 trajectory.csv --no-rotation --annotate

# Custom velocity smoothing
circledetect transform video.mp4 trajectory.csv --smooth-window 7
```

## Usage Examples

### Example 1: Quick Processing

```bash
circledetect auto video.mp4 --no-rotation
```

**Output:**
- `results/csv/video_trajectory_kalman.csv`
- `results/csv/video_trajectory_smoothed.csv`
- `results/videos/video_tracking.mp4`
- `results/videos/video_particle_frame.mp4`
- `results/videos/video_particle_frame_annotated.mp4`

### Example 2: Maximum Stability

For very jittery tracking:

```bash
# Step 1: Extract with Kalman
circledetect process video.mp4

# Step 2: Aggressive smoothing
circledetect smooth results/csv/video_trajectory_kalman.csv \
    --method savgol --window 21

# Step 3: Transform (no rotation)
circledetect transform video.mp4 \
    results/csv/video_trajectory_kalman_smoothed_savgol.csv \
    --no-rotation --annotate
```

### Example 3: Custom Pipeline

```bash
# Extract without Kalman
circledetect process video.mp4 --no-kalman

# Apply spline smoothing
circledetect smooth results/csv/video_trajectory.csv --method spline

# Transform with rotation
circledetect transform video.mp4 results/csv/video_trajectory_smoothed_spline.csv
```

## Python API

Use CircleDetection as a library:

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

## Advanced Options

### Smoothing Methods

- **savgol** (default): Savitzky-Golay filter - preserves peaks, best for varying motion
- **spline**: Spline interpolation - very smooth curves for gradual motion
- **moving**: Moving average - simple and fast

### Smoothing Window Sizes

- `--window 7`: Light smoothing (preserves rapid changes)
- `--window 15`: **Balanced (recommended)** - good for most cases
- `--window 21-31`: Heavy smoothing (removes most jitter)

### Stability Metrics

Lower smoothness score = more stable tracking:

| Method | Smoothness | Improvement |
|--------|------------|-------------|
| Raw detection | ~1.5 | 1.0x |
| + Kalman filtering | ~0.7 | 2.0x ✓ |
| + Savgol (window=15) | ~0.4 | 3.5x ✓✓ |
| + Savgol (window=21) | ~0.3 | 5.0x ✓✓✓ |

### When to Use `--no-rotation`

Use `--no-rotation` when:
- Particle motion is primarily linear/vertical
- Brownian motion causes excessive rotation jitter
- You want smaller output file size (~36% reduction)
- You don't need velocity alignment

## Legacy Scripts

Original Python scripts still available in `scripts/`:

```bash
# Extract trajectory
python scripts/extract_trajectory_stabilized.py data/video.mp4

# Smooth trajectory
python scripts/smooth_trajectory.py results/csv/trajectory.csv --method savgol --window 15

# Transform video
python scripts/transform_to_particle_frame.py data/video.mp4 results/csv/trajectory.csv --no-rotation

# Plot trajectory
python scripts/plot_trajectory.py results/csv/trajectory.csv
```

## Project Structure

```
CircleDetection/
├── circledetection/        # Python package
│   ├── __init__.py
│   ├── cli.py             # CLI commands ⭐
│   ├── detection.py       # Circle detection
│   ├── tracking.py        # Kalman filtering
│   ├── smoothing.py       # Trajectory smoothing
│   ├── visualization.py   # Plotting
│   └── utils.py           # File I/O
├── scripts/               # Legacy scripts
├── data/                  # Input videos
├── results/               # Output files
│   ├── csv/              # Trajectory CSVs
│   ├── videos/           # Videos
│   └── plots/            # Plots
├── setup.py              # Package installer
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Requirements

- Python >= 3.8
- OpenCV >= 4.5.0
- NumPy >= 1.20.0
- Matplotlib >= 3.5.0
- SciPy >= 1.7.0
- Click >= 8.0.0

## Algorithm Details

### Detection
- Hough Circle Transform with radius constraints (40-70% of expected diameter)
- Gaussian blur preprocessing for noise reduction
- Subpixel refinement via intensity-weighted centroid

### Stabilization
- **Robust centroid**: Clips outlier bright pixels (nanoparticles) using median + 2σ threshold
- **Kalman filtering**: Constant velocity model with motion prediction
- **Constrained search**: Limits detection region to predicted position
- **Performance**: 2x smoother, 0.91-0.99 correlation with raw detection

### Smoothing
- **Savitzky-Golay**: Polynomial fitting in sliding window - preserves features
- **Spline**: Smooth curve interpolation - very smooth output
- **Moving Average**: Simple uniform kernel - fast computation

### Reference Frame Transform
- Centers microparticle at each frame (particle appears stationary)
- Optional rotation to align velocity vector horizontally
- Black padding for rotated frame boundaries
- Saves transformation parameters for inverse mapping

## Troubleshooting

### Jittery Tracking

Try larger smoothing window:
```bash
circledetect smooth trajectory.csv --method savgol --window 21
```

### Particle Not Detected

Adjust expected diameter:
```bash
circledetect process video.mp4 --diameter 50
```

### Rotation Jitter

Use `--no-rotation`:
```bash
circledetect transform video.mp4 trajectory.csv --no-rotation
```

## Help

Get help for any command:

```bash
circledetect --help
circledetect auto --help
circledetect process --help
circledetect smooth --help
circledetect transform --help
```

## License

MIT License

## Citation

If you use this software in your research, please cite:

```bibtex
@software{circledetection2024,
  title = {CircleDetection: Microparticle Tracking for Dark Field Microscopy},
  year = {2024},
}
```

## Support

- Issues: GitHub Issues
- Documentation: See `CLAUDE.md` for detailed technical notes
- Scripts: Original scripts in `scripts/` directory
