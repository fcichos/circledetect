"""Unit tests for ParticleTrackingPipeline."""

import numpy as np
import cv2
import pytest
from circledetect.pipeline import ParticleTrackingPipeline
from circledetect.detector import CircleDetector
from circledetect.tracker import ParticleTracker
from circledetect.transformer import ImageTransformer


def create_test_frame(size=(200, 200), particle_pos=(100, 100), radius=20):
    """Create a test frame with a particle."""
    frame = np.zeros(size + (3,), dtype=np.uint8)
    cv2.circle(frame, particle_pos, radius, (255, 255, 255), -1)
    return frame


class TestParticleTrackingPipeline:
    """Test cases for ParticleTrackingPipeline class."""
    
    def test_initialization(self):
        """Test pipeline initialization."""
        pipeline = ParticleTrackingPipeline()
        assert pipeline.detector is not None
        assert pipeline.tracker is not None
        assert pipeline.transformer is not None
        
        # Test with custom components
        detector = CircleDetector(min_radius=5)
        tracker = ParticleTracker(max_history=10)
        transformer = ImageTransformer()
        
        pipeline = ParticleTrackingPipeline(detector, tracker, transformer)
        assert pipeline.detector.min_radius == 5
        assert pipeline.tracker.max_history == 10
    
    def test_process_frame(self):
        """Test processing a single frame."""
        pipeline = ParticleTrackingPipeline()
        frame = create_test_frame()
        
        processed, position = pipeline.process_frame(frame, transform=False)
        
        assert processed.shape == frame.shape
        # May or may not detect depending on parameters
    
    def test_process_frame_with_transform(self):
        """Test processing frame with transformation."""
        detector = CircleDetector(min_radius=10, max_radius=30, param2=15)
        pipeline = ParticleTrackingPipeline(detector=detector)
        frame = create_test_frame()
        
        processed, position = pipeline.process_frame(frame, transform=True)
        
        assert processed.shape == frame.shape
    
    def test_process_image_sequence(self):
        """Test processing a sequence of images."""
        pipeline = ParticleTrackingPipeline()
        
        # Create a sequence of frames
        frames = [
            create_test_frame(particle_pos=(100, 100)),
            create_test_frame(particle_pos=(105, 105)),
            create_test_frame(particle_pos=(110, 110)),
        ]
        
        processed, trajectory = pipeline.process_image_sequence(
            frames, transform=False, visualize=False
        )
        
        assert len(processed) == 3
        assert all(p.shape == frames[0].shape for p in processed)
    
    def test_get_trajectory(self):
        """Test trajectory retrieval."""
        detector = CircleDetector(min_radius=10, max_radius=30, param2=15)
        pipeline = ParticleTrackingPipeline(detector=detector)
        
        # Process some frames
        frames = [
            create_test_frame(particle_pos=(100, 100)),
            create_test_frame(particle_pos=(105, 105)),
        ]
        
        for frame in frames:
            pipeline.process_frame(frame, transform=False)
        
        trajectory = pipeline.get_trajectory()
        assert isinstance(trajectory, list)
    
    def test_reset(self):
        """Test pipeline reset."""
        detector = CircleDetector(min_radius=10, max_radius=30, param2=15)
        pipeline = ParticleTrackingPipeline(detector=detector)
        
        # Process a frame
        frame = create_test_frame()
        pipeline.process_frame(frame)
        
        # Reset
        pipeline.reset()
        
        # Tracker should be reset
        assert len(pipeline.tracker.history) == 0
        assert not pipeline.tracker.initialized
