#!/usr/bin/env python3
"""
Plot the trajectory of a microparticle from CSV data.
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from config import get_csv_path, get_plot_path


def load_trajectory(csv_path):
    """
    Load trajectory data from CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        tuple: (frames, x_coords, y_coords, radii)
    """
    frames = []
    x_coords = []
    y_coords = []
    radii = []

    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            frames.append(int(row['frame']))
            x_coords.append(int(row['x']))
            y_coords.append(int(row['y']))
            radii.append(int(row['radius']))

    return np.array(frames), np.array(x_coords), np.array(y_coords), np.array(radii)


def plot_trajectory(csv_path):
    """
    Create a comprehensive trajectory plot.

    Args:
        csv_path: Path to CSV file containing trajectory data
    """
    frames, x, y, radii = load_trajectory(csv_path)

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: 2D trajectory
    ax1 = axes[0, 0]
    scatter = ax1.scatter(x, y, c=frames, cmap='viridis', s=50, alpha=0.6)
    ax1.plot(x, y, 'b-', alpha=0.3, linewidth=1)
    ax1.plot(x[0], y[0], 'go', markersize=10, label='Start')
    ax1.plot(x[-1], y[-1], 'ro', markersize=10, label='End')
    ax1.set_xlabel('X position (pixels)')
    ax1.set_ylabel('Y position (pixels)')
    ax1.set_title('2D Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.invert_yaxis()  # Invert Y axis to match image coordinates
    plt.colorbar(scatter, ax=ax1, label='Frame number')

    # Plot 2: X position vs time
    ax2 = axes[0, 1]
    ax2.plot(frames, x, 'b-', linewidth=2)
    ax2.set_xlabel('Frame number')
    ax2.set_ylabel('X position (pixels)')
    ax2.set_title('X Position over Time')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Y position vs time
    ax3 = axes[1, 0]
    ax3.plot(frames, y, 'r-', linewidth=2)
    ax3.set_xlabel('Frame number')
    ax3.set_ylabel('Y position (pixels)')
    ax3.set_title('Y Position over Time')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Displacement from start
    ax4 = axes[1, 1]
    displacement = np.sqrt((x - x[0])**2 + (y - y[0])**2)
    ax4.plot(frames, displacement, 'g-', linewidth=2)
    ax4.set_xlabel('Frame number')
    ax4.set_ylabel('Displacement (pixels)')
    ax4.set_title('Displacement from Start Position')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = get_plot_path(os.path.basename(csv_path).replace('.csv', '_plot.png'))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Trajectory plot saved to: {output_path}")

    # Calculate and print statistics
    print(f"\nTrajectory Statistics:")
    print(f"  Total frames: {len(frames)}")
    print(f"  X range: {x.min()} to {x.max()} (Δ = {x.max() - x.min()} pixels)")
    print(f"  Y range: {y.min()} to {y.max()} (Δ = {y.max() - y.min()} pixels)")
    print(f"  Average X: {x.mean():.1f} ± {x.std():.1f} pixels")
    print(f"  Average Y: {y.mean():.1f} ± {y.std():.1f} pixels")
    print(f"  Total path length: {np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)):.1f} pixels")
    print(f"  Maximum displacement: {displacement.max():.1f} pixels")
    print(f"  Average radius: {radii.mean():.1f} ± {radii.std():.1f} pixels")

    # Calculate velocity (assuming 30 fps)
    fps = 30.0
    if len(frames) > 1:
        dt = 1.0 / fps  # time between frames in seconds
        dx = np.diff(x)
        dy = np.diff(y)
        velocities = np.sqrt(dx**2 + dy**2) / dt  # pixels per second
        print(f"\nVelocity Statistics:")
        print(f"  Average velocity: {velocities.mean():.1f} ± {velocities.std():.1f} pixels/s")
        print(f"  Maximum velocity: {velocities.max():.1f} pixels/s")

    # plt.show()  # Commented out for non-interactive mode


def main():
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = get_csv_path("Ms_4_4_AuNP_moving_n0_laser_2_40_038_trajectory.csv")

    plot_trajectory(csv_path)


if __name__ == "__main__":
    main()
