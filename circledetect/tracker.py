"""
Particle tracking module for maintaining particle identity across frames.

This module provides functionality to track a circular particle across
multiple frames in a video sequence.
"""

import numpy as np
from typing import Optional, List, Tuple, Deque
from collections import deque


class ParticleTracker:
    """
    Track a circular particle across multiple frames.
    
    Maintains particle history and predicts position in case of detection failures.
    
    Attributes:
        max_history: Maximum number of historical positions to store
        max_distance: Maximum distance a particle can move between frames
    """
    
    def __init__(self, max_history: int = 30, max_distance: float = 50.0):
        """
        Initialize the ParticleTracker.
        
        Args:
            max_history: Maximum number of frames to keep in history (default: 30)
            max_distance: Maximum pixel distance between frames (default: 50.0)
        """
        self.max_history = max_history
        self.max_distance = max_distance
        self.history: Deque[Tuple[int, int, int]] = deque(maxlen=max_history)
        self.initialized = False
    
    def update(
        self, detected_particle: Optional[Tuple[int, int, int]]
    ) -> Optional[Tuple[int, int, int]]:
        """
        Update tracker with newly detected particle position.
        
        Args:
            detected_particle: Detected particle as (x, y, radius) or None
            
        Returns:
            Tracked particle position (x, y, radius) or None
        """
        # First detection
        if not self.initialized:
            if detected_particle is not None:
                self.history.append(detected_particle)
                self.initialized = True
                return detected_particle
            return None
        
        # No detection - try to predict
        if detected_particle is None:
            return self._predict_position()
        
        # Validate detection is close to expected position
        if self._is_valid_position(detected_particle):
            self.history.append(detected_particle)
            return detected_particle
        else:
            # Detection seems wrong, use prediction
            predicted = self._predict_position()
            if predicted is not None:
                self.history.append(predicted)
            return predicted
    
    def _is_valid_position(self, position: Tuple[int, int, int]) -> bool:
        """
        Check if detected position is valid based on tracking history.
        
        Args:
            position: Detected position as (x, y, radius)
            
        Returns:
            True if position is valid, False otherwise
        """
        if len(self.history) == 0:
            return True
        
        last_pos = self.history[-1]
        distance = np.sqrt(
            (position[0] - last_pos[0]) ** 2 + (position[1] - last_pos[1]) ** 2
        )
        
        return distance <= self.max_distance
    
    def _predict_position(self) -> Optional[Tuple[int, int, int]]:
        """
        Predict particle position based on tracking history.
        
        Returns:
            Predicted position (x, y, radius) or None
        """
        if len(self.history) == 0:
            return None
        
        if len(self.history) == 1:
            return self.history[-1]
        
        # Simple linear prediction based on last two positions
        if len(self.history) >= 2:
            last = np.array(self.history[-1])
            second_last = np.array(self.history[-2])
            velocity = last - second_last
            
            # Predict next position
            predicted = last + velocity
            return tuple(predicted.astype(int))
        
        return self.history[-1]
    
    def get_trajectory(self) -> List[Tuple[int, int, int]]:
        """
        Get the complete trajectory of tracked particle.
        
        Returns:
            List of (x, y, radius) positions
        """
        return list(self.history)
    
    def get_current_position(self) -> Optional[Tuple[int, int, int]]:
        """
        Get the current tracked position.
        
        Returns:
            Current position (x, y, radius) or None
        """
        if len(self.history) > 0:
            return self.history[-1]
        return None
    
    def reset(self):
        """Reset the tracker, clearing all history."""
        self.history.clear()
        self.initialized = False
