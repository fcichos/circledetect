"""
Configuration and path utilities for CircleDetection project.
"""

import os

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
CSV_DIR = os.path.join(RESULTS_DIR, 'csv')
VIDEO_DIR = os.path.join(RESULTS_DIR, 'videos')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')

# Ensure output directories exist
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def get_data_path(filename):
    """Get path to a file in the data directory."""
    return os.path.join(DATA_DIR, filename)


def get_csv_path(filename):
    """Get path to a file in the CSV results directory."""
    return os.path.join(CSV_DIR, filename)


def get_video_path(filename):
    """Get path to a file in the video results directory."""
    return os.path.join(VIDEO_DIR, filename)


def get_plot_path(filename):
    """Get path to a file in the plots results directory."""
    return os.path.join(PLOTS_DIR, filename)
