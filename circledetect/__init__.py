"""
CircleDetect - Python pipeline for tracking circular particles.

This package provides tools to detect and track circular particles in images
with tracer particle backgrounds, and to transform images according to the
particle position.
"""

__version__ = "0.1.0"

from .detector import CircleDetector
from .tracker import ParticleTracker
from .transformer import ImageTransformer
from .pipeline import ParticleTrackingPipeline

__all__ = [
    "CircleDetector",
    "ParticleTracker",
    "ImageTransformer",
    "ParticleTrackingPipeline",
]
