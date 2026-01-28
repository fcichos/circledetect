#!/usr/bin/env python3
"""
Compare integer pixel and subpixel trajectory tracking.
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from config import get_csv_path, get_plot_path


def load_trajectory(csv_path):
    """Load trajectory data from CSV file."""
    frames = []
    x_coords = []
    y_coords = []

    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            frames.append(int(row['frame']))
            x_coords.append(float(row['x']))
            y_coords.append(float(row['y']))

    return np.array(frames), np.array(x_coords), np.array(y_coords)


def compare_trajectories(csv_integer, csv_subpixel):
    """
    Compare integer and subpixel trajectories.
    """
    # Load both trajectories
    frames_int, x_int, y_int = load_trajectory(csv_integer)
    frames_sub, x_sub, y_sub = load_trajectory(csv_subpixel)

    # Create comparison figure
    fig = plt.figure(figsize=(16, 10))

    # Plot 1: 2D trajectories comparison
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(x_int, y_int, 'b.-', alpha=0.5, linewidth=1, markersize=4, label='Integer pixel')
    ax1.plot(x_sub, y_sub, 'r.-', alpha=0.5, linewidth=1, markersize=4, label='Subpixel')
    ax1.plot(x_int[0], y_int[0], 'go', markersize=10, label='Start')
    ax1.plot(x_int[-1], y_int[-1], 'mo', markersize=10, label='End')
    ax1.set_xlabel('X position (pixels)')
    ax1.set_ylabel('Y position (pixels)')
    ax1.set_title('2D Trajectory Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()

    # Plot 2: X position comparison
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(frames_int, x_int, 'b.-', alpha=0.6, linewidth=1, markersize=3, label='Integer')
    ax2.plot(frames_sub, x_sub, 'r.-', alpha=0.6, linewidth=1, markersize=3, label='Subpixel')
    ax2.set_xlabel('Frame number')
    ax2.set_ylabel('X position (pixels)')
    ax2.set_title('X Position Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Y position comparison
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(frames_int, y_int, 'b.-', alpha=0.6, linewidth=1, markersize=3, label='Integer')
    ax3.plot(frames_sub, y_sub, 'r.-', alpha=0.6, linewidth=1, markersize=3, label='Subpixel')
    ax3.set_xlabel('Frame number')
    ax3.set_ylabel('Y position (pixels)')
    ax3.set_title('Y Position Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Step size distribution
    ax4 = plt.subplot(2, 3, 4)
    steps_int = np.sqrt(np.diff(x_int)**2 + np.diff(y_int)**2)
    steps_sub = np.sqrt(np.diff(x_sub)**2 + np.diff(y_sub)**2)
    ax4.hist(steps_int, bins=20, alpha=0.5, label=f'Integer (σ={steps_int.std():.3f})', color='blue')
    ax4.hist(steps_sub, bins=20, alpha=0.5, label=f'Subpixel (σ={steps_sub.std():.3f})', color='red')
    ax4.set_xlabel('Step size (pixels)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Step Size Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Difference between methods
    ax5 = plt.subplot(2, 3, 5)
    diff_x = x_sub - x_int
    diff_y = y_sub - y_int
    diff_total = np.sqrt(diff_x**2 + diff_y**2)
    ax5.plot(frames_int, diff_total, 'g.-', linewidth=1, markersize=3)
    ax5.axhline(y=diff_total.mean(), color='r', linestyle='--',
                label=f'Mean: {diff_total.mean():.3f} px')
    ax5.set_xlabel('Frame number')
    ax5.set_ylabel('Position difference (pixels)')
    ax5.set_title('Distance Between Integer and Subpixel')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Zoomed in section of trajectory
    ax6 = plt.subplot(2, 3, 6)
    zoom_frames = 20  # Show first 20 frames
    ax6.plot(x_int[:zoom_frames], y_int[:zoom_frames], 'b.-',
             linewidth=2, markersize=8, label='Integer pixel')
    ax6.plot(x_sub[:zoom_frames], y_sub[:zoom_frames], 'r.-',
             linewidth=2, markersize=8, label='Subpixel')
    for i in range(min(5, zoom_frames)):
        ax6.annotate(f'{i}', (x_int[i], y_int[i]), fontsize=8,
                    xytext=(5, 5), textcoords='offset points')
    ax6.set_xlabel('X position (pixels)')
    ax6.set_ylabel('Y position (pixels)')
    ax6.set_title(f'Zoomed View (First {zoom_frames} Frames)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.invert_yaxis()

    plt.tight_layout()

    # Save figure
    output_path = get_plot_path(os.path.basename(csv_subpixel).replace('.csv', '_comparison.png'))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")

    # Print statistics
    print(f"\n=== COMPARISON STATISTICS ===")
    print(f"\nInteger pixel tracking:")
    print(f"  X std dev: {x_int.std():.3f} pixels")
    print(f"  Y std dev: {y_int.std():.3f} pixels")
    print(f"  Mean step size: {steps_int.mean():.3f} ± {steps_int.std():.3f} pixels")
    print(f"  Total path length: {steps_int.sum():.3f} pixels")

    print(f"\nSubpixel tracking:")
    print(f"  X std dev: {x_sub.std():.3f} pixels")
    print(f"  Y std dev: {y_sub.std():.3f} pixels")
    print(f"  Mean step size: {steps_sub.mean():.3f} ± {steps_sub.std():.3f} pixels")
    print(f"  Total path length: {steps_sub.sum():.3f} pixels")

    print(f"\nDifference between methods:")
    print(f"  Mean position difference: {diff_total.mean():.3f} ± {diff_total.std():.3f} pixels")
    print(f"  Max position difference: {diff_total.max():.3f} pixels")
    print(f"  RMS difference: {np.sqrt(np.mean(diff_total**2)):.3f} pixels")

    # Calculate trajectory smoothness improvement
    # Lower variance in step sizes indicates smoother trajectory
    smoothness_ratio = steps_int.std() / steps_sub.std()
    print(f"\nSmoothness improvement:")
    print(f"  Step size variance ratio: {smoothness_ratio:.2f}x")
    if smoothness_ratio > 1:
        print(f"  → Subpixel tracking is {smoothness_ratio:.2f}x smoother")
    else:
        print(f"  → Integer tracking is {1/smoothness_ratio:.2f}x smoother")

    # plt.show()


def main():
    csv_integer = get_csv_path("Ms_4_4_AuNP_moving_n0_laser_2_40_038_trajectory.csv")
    csv_subpixel = get_csv_path("Ms_4_4_AuNP_moving_n0_laser_2_40_038_trajectory_subpixel.csv")

    if len(sys.argv) > 2:
        csv_integer = sys.argv[1]
        csv_subpixel = sys.argv[2]

    compare_trajectories(csv_integer, csv_subpixel)


if __name__ == "__main__":
    main()
