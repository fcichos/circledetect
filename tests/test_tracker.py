"""Unit tests for ParticleTracker."""

import numpy as np
import pytest
from circledetect.tracker import ParticleTracker


class TestParticleTracker:
    """Test cases for ParticleTracker class."""
    
    def test_initialization(self):
        """Test tracker initialization."""
        tracker = ParticleTracker()
        assert tracker.max_history == 30
        assert tracker.max_distance == 50.0
        assert not tracker.initialized
        
        tracker = ParticleTracker(max_history=10, max_distance=20.0)
        assert tracker.max_history == 10
        assert tracker.max_distance == 20.0
    
    def test_first_update(self):
        """Test first particle detection update."""
        tracker = ParticleTracker()
        position = (100, 100, 20)
        
        result = tracker.update(position)
        assert result == position
        assert tracker.initialized
        assert len(tracker.history) == 1
    
    def test_update_with_none(self):
        """Test update with no detection."""
        tracker = ParticleTracker()
        
        # First update with None should return None
        result = tracker.update(None)
        assert result is None
        assert not tracker.initialized
    
    def test_valid_position_update(self):
        """Test update with valid position."""
        tracker = ParticleTracker(max_distance=50.0)
        
        # Initialize
        tracker.update((100, 100, 20))
        
        # Update with nearby position
        result = tracker.update((110, 105, 20))
        assert result == (110, 105, 20)
        assert len(tracker.history) == 2
    
    def test_invalid_position_rejection(self):
        """Test rejection of position too far from previous."""
        tracker = ParticleTracker(max_distance=10.0)
        
        # Initialize
        tracker.update((100, 100, 20))
        
        # Update with far position (should be rejected)
        result = tracker.update((200, 200, 20))
        # Should use prediction instead
        assert result is not None
        assert result != (200, 200, 20)
    
    def test_prediction(self):
        """Test position prediction."""
        tracker = ParticleTracker()
        
        # Create a moving particle
        tracker.update((100, 100, 20))
        tracker.update((110, 110, 20))
        
        # Update with None to trigger prediction
        result = tracker.update(None)
        assert result is not None
        # Should predict continued motion
        x, y, r = result
        assert x >= 110  # Should continue in same direction
        assert y >= 110
    
    def test_get_trajectory(self):
        """Test trajectory retrieval."""
        tracker = ParticleTracker()
        
        positions = [(100, 100, 20), (110, 105, 20), (120, 110, 20)]
        for pos in positions:
            tracker.update(pos)
        
        trajectory = tracker.get_trajectory()
        assert len(trajectory) == 3
        assert trajectory == positions
    
    def test_get_current_position(self):
        """Test current position retrieval."""
        tracker = ParticleTracker()
        
        assert tracker.get_current_position() is None
        
        position = (100, 100, 20)
        tracker.update(position)
        assert tracker.get_current_position() == position
    
    def test_reset(self):
        """Test tracker reset."""
        tracker = ParticleTracker()
        
        tracker.update((100, 100, 20))
        tracker.update((110, 105, 20))
        
        assert tracker.initialized
        assert len(tracker.history) > 0
        
        tracker.reset()
        assert not tracker.initialized
        assert len(tracker.history) == 0
