"""Visualization functions for trajectory plotting."""

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(trajectory, output_path=None, show=False):
    """
    Create a comprehensive trajectory plot.

    Args:
        trajectory: List of dicts with keys: frame, x, y, radius
        output_path: Path to save plot (if None, doesn't save)
        show: If True, display plot interactively
    """
    if len(trajectory) == 0:
        raise ValueError("Empty trajectory")

    # Extract data
    frames = np.array([t['frame'] for t in trajectory])
    x = np.array([t['x'] for t in trajectory])
    y = np.array([t['y'] for t in trajectory])
    radii = np.array([t['radius'] for t in trajectory])

    # Create figure with subplots
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
    ax1.invert_yaxis()
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

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig
