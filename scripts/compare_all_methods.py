#!/usr/bin/env python3
"""
Compare all trajectory tracking methods: integer, subpixel, and stabilized.
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


def compare_all_methods():
    """Compare all trajectory tracking methods."""

    # Load all trajectories
    print("Loading trajectories...")
    frames_int, x_int, y_int = load_trajectory(
        "Ms_4_4_AuNP_moving_n0_laser_2_40_038_trajectory.csv")
    frames_sub, x_sub, y_sub = load_trajectory(
        "Ms_4_4_AuNP_moving_n0_laser_2_40_038_trajectory_subpixel.csv")
    frames_raw, x_raw, y_raw = load_trajectory(
        "Ms_4_4_AuNP_moving_n0_laser_2_40_038_trajectory_stabilized_raw.csv")
    frames_filt, x_filt, y_filt = load_trajectory(
        "Ms_4_4_AuNP_moving_n0_laser_2_40_038_trajectory_stabilized_filtered.csv")

    # Create comprehensive comparison figure
    fig = plt.figure(figsize=(18, 12))

    # Plot 1: 2D trajectories - all methods
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(x_int, y_int, '.-', alpha=0.4, linewidth=1, markersize=3,
             label='Integer', color='blue')
    ax1.plot(x_sub, y_sub, '.-', alpha=0.4, linewidth=1, markersize=3,
             label='Subpixel', color='green')
    ax1.plot(x_raw, y_raw, '.-', alpha=0.3, linewidth=1, markersize=2,
             label='Raw detection', color='gray')
    ax1.plot(x_filt, y_filt, '-', alpha=0.8, linewidth=2,
             label='Kalman filtered', color='red')
    ax1.plot(x_filt[0], y_filt[0], 'go', markersize=10, label='Start')
    ax1.plot(x_filt[-1], y_filt[-1], 'mo', markersize=10, label='End')
    ax1.set_xlabel('X position (pixels)')
    ax1.set_ylabel('Y position (pixels)')
    ax1.set_title('2D Trajectory - All Methods')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()

    # Plot 2: Step size comparison
    ax2 = plt.subplot(2, 3, 2)
    steps_int = np.sqrt(np.diff(x_int)**2 + np.diff(y_int)**2)
    steps_sub = np.sqrt(np.diff(x_sub)**2 + np.diff(y_sub)**2)
    steps_raw = np.sqrt(np.diff(x_raw)**2 + np.diff(y_raw)**2)
    steps_filt = np.sqrt(np.diff(x_filt)**2 + np.diff(y_filt)**2)

    frames_for_steps = frames_int[1:]
    ax2.plot(frames_for_steps, steps_int, '.-', alpha=0.5, linewidth=1,
             markersize=3, label=f'Integer (σ={steps_int.std():.3f})')
    ax2.plot(frames_for_steps, steps_sub, '.-', alpha=0.5, linewidth=1,
             markersize=3, label=f'Subpixel (σ={steps_sub.std():.3f})')
    ax2.plot(frames_for_steps, steps_raw, '.-', alpha=0.3, linewidth=1,
             markersize=2, label=f'Raw (σ={steps_raw.std():.3f})')
    ax2.plot(frames_for_steps, steps_filt, '-', alpha=0.8, linewidth=2,
             label=f'Kalman (σ={steps_filt.std():.3f})')
    ax2.set_xlabel('Frame number')
    ax2.set_ylabel('Step size (pixels)')
    ax2.set_title('Step Size per Frame')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Step size histogram
    ax3 = plt.subplot(2, 3, 3)
    bins = np.linspace(0, max(steps_raw.max(), steps_filt.max()), 30)
    ax3.hist(steps_raw, bins=bins, alpha=0.4, label='Raw', color='gray')
    ax3.hist(steps_filt, bins=bins, alpha=0.6, label='Kalman filtered', color='red')
    ax3.axvline(steps_raw.mean(), color='gray', linestyle='--', linewidth=2,
                label=f'Raw mean: {steps_raw.mean():.3f}')
    ax3.axvline(steps_filt.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Filtered mean: {steps_filt.mean():.3f}')
    ax3.set_xlabel('Step size (pixels)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Step Size Distribution')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: X position over time
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(frames_raw, x_raw, '.-', alpha=0.3, linewidth=1, markersize=2,
             label='Raw', color='gray')
    ax4.plot(frames_filt, x_filt, '-', alpha=0.8, linewidth=2,
             label='Kalman filtered', color='red')
    ax4.set_xlabel('Frame number')
    ax4.set_ylabel('X position (pixels)')
    ax4.set_title('X Position - Raw vs Filtered')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Y position over time
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(frames_raw, y_raw, '.-', alpha=0.3, linewidth=1, markersize=2,
             label='Raw', color='gray')
    ax5.plot(frames_filt, y_filt, '-', alpha=0.8, linewidth=2,
             label='Kalman filtered', color='red')
    ax5.set_xlabel('Frame number')
    ax5.set_ylabel('Y position (pixels)')
    ax5.set_title('Y Position - Raw vs Filtered')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Zoomed trajectory showing stabilization
    ax6 = plt.subplot(2, 3, 6)
    zoom_frames = 30
    ax6.plot(x_raw[:zoom_frames], y_raw[:zoom_frames], '.-',
             linewidth=1, markersize=6, alpha=0.5, label='Raw detection', color='gray')
    ax6.plot(x_filt[:zoom_frames], y_filt[:zoom_frames], '.-',
             linewidth=2, markersize=8, label='Kalman filtered', color='red')
    for i in [0, 10, 20]:
        if i < zoom_frames:
            ax6.annotate(f'{i}', (x_filt[i], y_filt[i]), fontsize=9,
                        xytext=(5, 5), textcoords='offset points')
    ax6.set_xlabel('X position (pixels)')
    ax6.set_ylabel('Y position (pixels)')
    ax6.set_title(f'Zoomed View - First {zoom_frames} Frames')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.invert_yaxis()

    plt.tight_layout()

    # Save figure
    output_path = get_plot_path("trajectory_comparison_all_methods.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {output_path}")

    # Print comprehensive statistics
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE COMPARISON OF ALL METHODS")
    print(f"{'='*60}")

    methods = [
        ('Integer pixel', x_int, y_int, steps_int),
        ('Subpixel', x_sub, y_sub, steps_sub),
        ('Raw detection', x_raw, y_raw, steps_raw),
        ('Kalman filtered', x_filt, y_filt, steps_filt)
    ]

    for name, x, y, steps in methods:
        print(f"\n{name}:")
        print(f"  Mean step: {steps.mean():.3f} pixels")
        print(f"  Step std dev: {steps.std():.3f} pixels (lower = smoother)")
        print(f"  Total path length: {steps.sum():.3f} pixels")
        print(f"  Position std dev: X={x.std():.3f}, Y={y.std():.3f}")

    print(f"\n{'='*60}")
    print(f"SMOOTHNESS IMPROVEMENT")
    print(f"{'='*60}")

    baseline_std = steps_raw.std()
    improvement = baseline_std / steps_filt.std()
    print(f"Kalman filter vs Raw detection: {improvement:.2f}x smoother")

    # Calculate correlation between raw and filtered
    corr_x = np.corrcoef(x_raw, x_filt)[0, 1]
    corr_y = np.corrcoef(y_raw, y_filt)[0, 1]
    print(f"\nCorrelation with raw detection:")
    print(f"  X: {corr_x:.4f}")
    print(f"  Y: {corr_y:.4f}")

    # Calculate mean deviation
    deviation = np.sqrt((x_raw - x_filt)**2 + (y_raw - y_filt)**2)
    print(f"\nMean deviation from raw: {deviation.mean():.3f} ± {deviation.std():.3f} pixels")

    # plt.show()


def main():
    compare_all_methods()


if __name__ == "__main__":
    main()
