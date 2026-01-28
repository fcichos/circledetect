"""
CircleDetection - Microparticle tracking for dark field microscopy.

A Python package for detecting and tracking microparticles in dark field
microscopy videos, with support for stabilization, smoothing, and reference
frame transformation.
"""

__version__ = "1.0.0"
__author__ = "CircleDetection Team"

from .detection import detect_circle_robust
from .tracking import extract_trajectory, KalmanTracker
from .smoothing import smooth_trajectory
from .visualization import plot_trajectory
from .transform import transform_video_to_particle_frame

__all__ = [
    'detect_circle_robust',
    'extract_trajectory',
    'KalmanTracker',
    'smooth_trajectory',
    'plot_trajectory',
    'transform_video_to_particle_frame',
]
