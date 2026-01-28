"""
Image transformation module for centering and transforming images.

This module provides functionality to transform images based on particle
position, typically to center the particle or maintain a stable reference frame.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class ImageTransformer:
    """
    Transform images based on particle position.
    
    Supports translation, rotation, and cropping operations to maintain
    the particle in a fixed reference frame.
    """
    
    def __init__(self, center_position: Optional[Tuple[int, int]] = None):
        """
        Initialize the ImageTransformer.
        
        Args:
            center_position: Target center position (x, y) for transformations.
                           If None, uses image center.
        """
        self.center_position = center_position
    
    def center_on_particle(
        self, image: np.ndarray, particle_pos: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Translate image to center the particle.
        
        Args:
            image: Input image
            particle_pos: Particle position as (x, y, radius)
            
        Returns:
            Transformed image with particle centered
        """
        height, width = image.shape[:2]
        
        # Determine target center
        if self.center_position is not None:
            target_x, target_y = self.center_position
        else:
            target_x, target_y = width // 2, height // 2
        
        # Calculate translation
        particle_x, particle_y = particle_pos[0], particle_pos[1]
        tx = target_x - particle_x
        ty = target_y - particle_y
        
        # Create translation matrix
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        
        # Apply translation
        transformed = cv2.warpAffine(image, translation_matrix, (width, height))
        
        return transformed
    
    def stabilize_frame(
        self,
        image: np.ndarray,
        particle_pos: Tuple[int, int, int],
        reference_pos: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """
        Stabilize frame by compensating for particle movement.
        
        Args:
            image: Input image
            particle_pos: Current particle position (x, y, radius)
            reference_pos: Reference position to stabilize to. If None, uses center.
            
        Returns:
            Stabilized image
        """
        if reference_pos is None:
            # Use image center as reference
            height, width = image.shape[:2]
            reference_pos = (width // 2, height // 2, particle_pos[2])
        
        # Calculate displacement
        dx = reference_pos[0] - particle_pos[0]
        dy = reference_pos[1] - particle_pos[1]
        
        # Create translation matrix
        height, width = image.shape[:2]
        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        
        # Apply translation
        stabilized = cv2.warpAffine(image, translation_matrix, (width, height))
        
        return stabilized
    
    def crop_around_particle(
        self,
        image: np.ndarray,
        particle_pos: Tuple[int, int, int],
        crop_size: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """
        Crop image around the particle position.
        
        Args:
            image: Input image
            particle_pos: Particle position (x, y, radius)
            crop_size: Size of crop as (width, height)
            
        Returns:
            Cropped image or None if crop is out of bounds
        """
        height, width = image.shape[:2]
        particle_x, particle_y = particle_pos[0], particle_pos[1]
        crop_width, crop_height = crop_size
        
        # Calculate crop boundaries
        x1 = max(0, particle_x - crop_width // 2)
        y1 = max(0, particle_y - crop_height // 2)
        x2 = min(width, particle_x + crop_width // 2)
        y2 = min(height, particle_y + crop_height // 2)
        
        # Check if crop is valid
        if x2 - x1 < crop_width // 2 or y2 - y1 < crop_height // 2:
            return None
        
        # Crop image
        cropped = image[y1:y2, x1:x2]
        
        return cropped
    
    def transform_coordinates(
        self,
        points: np.ndarray,
        particle_pos: Tuple[int, int, int],
        reference_pos: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        Transform coordinates from image frame to particle frame.
        
        Args:
            points: Array of points with shape (N, 2) as [[x, y], ...]
            particle_pos: Current particle position (x, y, radius)
            reference_pos: Reference position (x, y, radius)
            
        Returns:
            Transformed points in particle reference frame
        """
        # Calculate translation
        dx = reference_pos[0] - particle_pos[0]
        dy = reference_pos[1] - particle_pos[1]
        
        # Apply translation
        transformed = points.copy()
        transformed[:, 0] += dx
        transformed[:, 1] += dy
        
        return transformed
