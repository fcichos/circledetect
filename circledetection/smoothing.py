"""Trajectory smoothing algorithms."""

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline


def smooth_trajectory(trajectory, method='savgol', window_length=11, polyorder=3,
                     smoothing_factor=None):
    """
    Apply smoothing to trajectory data.

    Args:
        trajectory: List of dicts with keys: frame, x, y, radius
        method: Smoothing method ('savgol', 'spline', 'moving')
        window_length: Window size for savgol/moving average (must be odd)
        polyorder: Polynomial order for savgol (default: 3)
        smoothing_factor: Smoothing parameter for spline (None=auto)

    Returns:
        Smoothed trajectory (same format as input)
    """
    if method == 'savgol':
        return _smooth_savgol(trajectory, window_length, polyorder)
    elif method == 'spline':
        return _smooth_spline(trajectory, smoothing_factor)
    elif method == 'moving':
        return _smooth_moving_average(trajectory, window_length)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def _smooth_savgol(trajectory, window_length=11, polyorder=3):
    """Apply Savitzky-Golay filter."""
    if len(trajectory) < window_length:
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


def _smooth_spline(trajectory, smoothing_factor=None):
    """Apply spline smoothing."""
    if len(trajectory) < 4:
        return trajectory

    # Extract coordinates
    frames = np.array([t['frame'] for t in trajectory])
    x = np.array([t['x'] for t in trajectory])
    y = np.array([t['y'] for t in trajectory])
    radii = np.array([t['radius'] for t in trajectory])

    # Auto-calculate smoothing factor if not provided
    if smoothing_factor is None:
        smoothing_factor = len(frames) * np.var(np.diff(x))

    # Fit splines
    spline_x = UnivariateSpline(frames, x, s=smoothing_factor)
    spline_y = UnivariateSpline(frames, y, s=smoothing_factor)
    spline_r = UnivariateSpline(frames, radii, s=smoothing_factor * 0.1)

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


def _smooth_moving_average(trajectory, window_length=5):
    """Apply moving average smoothing."""
    if len(trajectory) < window_length:
        return trajectory

    # Extract coordinates
    frames = np.array([t['frame'] for t in trajectory])
    x = np.array([t['x'] for t in trajectory])
    y = np.array([t['y'] for t in trajectory])
    radii = np.array([t['radius'] for t in trajectory])

    # Apply moving average
    kernel = np.ones(window_length) / window_length
    x_smooth = np.convolve(x, kernel, mode='same')
    y_smooth = np.convolve(y, kernel, mode='same')
    radii_smooth = np.convolve(radii, kernel, mode='same')

    # Fix edges
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


def calculate_smoothness(trajectory):
    """
    Calculate trajectory smoothness metric (lower is smoother).

    Args:
        trajectory: List of dicts with keys: frame, x, y, radius

    Returns:
        float: Standard deviation of step sizes (lower = smoother)
    """
    if len(trajectory) < 2:
        return 0.0

    x = np.array([t['x'] for t in trajectory])
    y = np.array([t['y'] for t in trajectory])

    # Step sizes between consecutive points
    steps = np.sqrt(np.diff(x)**2 + np.diff(y)**2)

    # Standard deviation of step sizes
    return np.std(steps)
