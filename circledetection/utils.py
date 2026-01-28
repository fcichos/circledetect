"""Utility functions for file I/O and data handling."""

import csv
import os


def load_trajectory_csv(csv_path):
    """
    Load trajectory from CSV file.

    Args:
        csv_path: Path to CSV file with columns: frame, x, y, radius

    Returns:
        List of dicts with keys: frame, x, y, radius
    """
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


def save_trajectory_csv(trajectory, output_path):
    """
    Save trajectory to CSV file.

    Args:
        trajectory: List of dicts with keys: frame, x, y, radius
        output_path: Path for output CSV file
    """
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


def ensure_output_dirs(base_dir='results'):
    """
    Create output directory structure.

    Args:
        base_dir: Base directory for outputs

    Returns:
        tuple: (csv_dir, video_dir, plots_dir)
    """
    csv_dir = os.path.join(base_dir, 'csv')
    video_dir = os.path.join(base_dir, 'videos')
    plots_dir = os.path.join(base_dir, 'plots')

    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    return csv_dir, video_dir, plots_dir
