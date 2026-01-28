#!/usr/bin/env python3
"""
Post-process trajectory CSV to apply additional smoothing.
Useful for removing residual jitter from nanoparticle interference.
"""

import csv
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
import sys
import os
from config import get_csv_path
import argparse


def load_trajectory(csv_path):
    """Load trajectory from CSV file."""
    trajectory = []

    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            trajectory.append({
                'frame': int(row['frame']),
                'x': float(row['x']),
                'y': float(row['y']),
                'radius': float(row['radius'])
            })

    return trajectory


def smooth_savgol(trajectory, window_length=11, polyorder=3):
    """
    Apply Savitzky-Golay filter for smoothing.

    Args:
        trajectory: List of trajectory points
        window_length: Length of filter window (must be odd, larger = smoother)
        polyorder: Order of polynomial (2-3 typical)

    Returns:
        Smoothed trajectory
    """
    if len(trajectory) < window_length:
        print(f"Warning: Trajectory too short ({len(trajectory)}) for window {window_length}")
        return trajectory

    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1

    # Extract coordinates
    frames = np.array([t['frame'] for t in trajectory])
    x = np.array([t['x'] for t in trajectory])
    y = np.array([t['y'] for t in trajectory])
    radii = np.array([t['radius'] for t in trajectory])

    # Apply filter
    x_smooth = savgol_filter(x, window_length, polyorder)
    y_smooth = savgol_filter(y, window_length, polyorder)
    radii_smooth = savgol_filter(radii, window_length, polyorder)

    # Reconstruct trajectory
    smoothed = []
    for i, frame in enumerate(frames):
        smoothed.append({
            'frame': frame,
            'x': x_smooth[i],
            'y': y_smooth[i],
            'radius': radii_smooth[i]
        })

    return smoothed


def smooth_spline(trajectory, smoothing_factor=None):
    """
    Apply spline smoothing.

    Args:
        trajectory: List of trajectory points
        smoothing_factor: Smoothing parameter (None=auto, higher=smoother)

    Returns:
        Smoothed trajectory
    """
    if len(trajectory) < 4:
        print("Warning: Trajectory too short for spline smoothing")
        return trajectory

    # Extract coordinates
    frames = np.array([t['frame'] for t in trajectory])
    x = np.array([t['x'] for t in trajectory])
    y = np.array([t['y'] for t in trajectory])
    radii = np.array([t['radius'] for t in trajectory])

    # Auto-calculate smoothing factor if not provided
    if smoothing_factor is None:
        # Use variance-based estimate
        smoothing_factor = len(frames) * np.var(np.diff(x))

    # Fit splines
    spline_x = UnivariateSpline(frames, x, s=smoothing_factor)
    spline_y = UnivariateSpline(frames, y, s=smoothing_factor)
    spline_r = UnivariateSpline(frames, radii, s=smoothing_factor * 0.1)  # Less smoothing for radius

    # Evaluate at frame positions
    x_smooth = spline_x(frames)
    y_smooth = spline_y(frames)
    radii_smooth = spline_r(frames)

    # Reconstruct trajectory
    smoothed = []
    for i, frame in enumerate(frames):
        smoothed.append({
            'frame': frame,
            'x': x_smooth[i],
            'y': y_smooth[i],
            'radius': radii_smooth[i]
        })

    return smoothed


def smooth_moving_average(trajectory, window_length=5):
    """
    Apply simple moving average smoothing.

    Args:
        trajectory: List of trajectory points
        window_length: Window size for averaging

    Returns:
        Smoothed trajectory
    """
    if len(trajectory) < window_length:
        return trajectory

    # Extract coordinates
    frames = np.array([t['frame'] for t in trajectory])
    x = np.array([t['x'] for t in trajectory])
    y = np.array([t['y'] for t in trajectory])
    radii = np.array([t['radius'] for t in trajectory])

    # Apply moving average using convolution
    kernel = np.ones(window_length) / window_length
    x_smooth = np.convolve(x, kernel, mode='same')
    y_smooth = np.convolve(y, kernel, mode='same')
    radii_smooth = np.convolve(radii, kernel, mode='same')

    # Fix edges (use original values)
    half_window = window_length // 2
    x_smooth[:half_window] = x[:half_window]
    x_smooth[-half_window:] = x[-half_window:]
    y_smooth[:half_window] = y[:half_window]
    y_smooth[-half_window:] = y[-half_window:]
    radii_smooth[:half_window] = radii[:half_window]
    radii_smooth[-half_window:] = radii[-half_window:]

    # Reconstruct trajectory
    smoothed = []
    for i, frame in enumerate(frames):
        smoothed.append({
            'frame': frame,
            'x': x_smooth[i],
            'y': y_smooth[i],
            'radius': radii_smooth[i]
        })

    return smoothed


def save_trajectory(trajectory, output_path):
    """Save trajectory to CSV file."""
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame', 'x', 'y', 'radius'])
        for point in trajectory:
            writer.writerow([
                point['frame'],
                f"{point['x']:.6f}",
                f"{point['y']:.6f}",
                f"{point['radius']:.6f}"
            ])


def calculate_smoothness(trajectory):
    """Calculate trajectory smoothness metric (lower is smoother)."""
    if len(trajectory) < 2:
        return 0.0

    x = np.array([t['x'] for t in trajectory])
    y = np.array([t['y'] for t in trajectory])

    # Step sizes between consecutive points
    steps = np.sqrt(np.diff(x)**2 + np.diff(y)**2)

    # Standard deviation of step sizes (lower = more uniform motion)
    return np.std(steps)


def main():
    parser = argparse.ArgumentParser(
        description='Post-process trajectory to apply additional smoothing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Smoothing Methods:
  savgol    - Savitzky-Golay filter (preserves peaks, good for non-uniform motion)
  spline    - Spline smoothing (very smooth, good for gradual curves)
  moving    - Moving average (simple, fast)

Examples:
  # Savitzky-Golay with default window (11 frames)
  python scripts/smooth_trajectory.py results/csv/trajectory.csv --method savgol

  # Stronger smoothing (larger window)
  python scripts/smooth_trajectory.py results/csv/trajectory.csv --method savgol --window 21

  # Spline smoothing with auto smoothing factor
  python scripts/smooth_trajectory.py results/csv/trajectory.csv --method spline

  # Moving average
  python scripts/smooth_trajectory.py results/csv/trajectory.csv --method moving --window 7
        """
    )

    parser.add_argument('input_csv', help='Input trajectory CSV file')
    parser.add_argument('--method', default='savgol',
                       choices=['savgol', 'spline', 'moving'],
                       help='Smoothing method (default: savgol)')
    parser.add_argument('--window', type=int, default=11,
                       help='Window length for savgol/moving average (default: 11, must be odd)')
    parser.add_argument('--polyorder', type=int, default=3,
                       help='Polynomial order for savgol (default: 3)')
    parser.add_argument('--smoothing-factor', type=float, default=None,
                       help='Smoothing factor for spline (default: auto)')
    parser.add_argument('--output', help='Output CSV path (default: input_smoothed.csv)')

    args = parser.parse_args()

    # Load trajectory
    print(f"Loading trajectory from: {args.input_csv}")
    trajectory = load_trajectory(args.input_csv)
    print(f"  Loaded {len(trajectory)} points")

    # Calculate original smoothness
    smoothness_before = calculate_smoothness(trajectory)
    print(f"  Original smoothness: {smoothness_before:.3f} (std of step sizes)")

    # Apply smoothing
    print(f"\nApplying {args.method} smoothing...")

    if args.method == 'savgol':
        smoothed = smooth_savgol(trajectory, args.window, args.polyorder)
        print(f"  Window: {args.window}, Polynomial order: {args.polyorder}")
    elif args.method == 'spline':
        smoothed = smooth_spline(trajectory, args.smoothing_factor)
        print(f"  Smoothing factor: {args.smoothing_factor if args.smoothing_factor else 'auto'}")
    elif args.method == 'moving':
        smoothed = smooth_moving_average(trajectory, args.window)
        print(f"  Window: {args.window}")

    # Calculate smoothness after
    smoothness_after = calculate_smoothness(smoothed)
    improvement = smoothness_before / smoothness_after if smoothness_after > 0 else 1.0
    print(f"  Smoothed trajectory: {smoothness_after:.3f}")
    print(f"  Improvement: {improvement:.2f}x smoother")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base = os.path.basename(args.input_csv).replace('.csv', '')
        output_path = get_csv_path(f"{base}_smoothed_{args.method}.csv")

    # Save
    save_trajectory(smoothed, output_path)
    print(f"\nSmoothed trajectory saved to: {output_path}")


if __name__ == "__main__":
    main()
