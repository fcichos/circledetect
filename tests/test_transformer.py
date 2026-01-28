"""Unit tests for ImageTransformer."""

import numpy as np
import cv2
import pytest
from circledetect.transformer import ImageTransformer


def create_test_image(size=(200, 200)):
    """Create a simple test image."""
    image = np.zeros(size, dtype=np.uint8)
    # Add a marker at specific position
    cv2.circle(image, (50, 50), 10, 255, -1)
    return image


class TestImageTransformer:
    """Test cases for ImageTransformer class."""
    
    def test_initialization(self):
        """Test transformer initialization."""
        transformer = ImageTransformer()
        assert transformer.center_position is None
        
        transformer = ImageTransformer(center_position=(100, 100))
        assert transformer.center_position == (100, 100)
    
    def test_center_on_particle(self):
        """Test centering image on particle."""
        image = create_test_image(size=(200, 200))
        transformer = ImageTransformer()
        
        # Particle at (50, 50)
        particle_pos = (50, 50, 10)
        
        # Center the image
        centered = transformer.center_on_particle(image, particle_pos)
        
        assert centered.shape == image.shape
        # Image should be translated
        assert not np.array_equal(centered, image)
    
    def test_stabilize_frame(self):
        """Test frame stabilization."""
        image = create_test_image(size=(200, 200))
        transformer = ImageTransformer()
        
        particle_pos = (50, 50, 10)
        reference_pos = (100, 100, 10)
        
        stabilized = transformer.stabilize_frame(image, particle_pos, reference_pos)
        
        assert stabilized.shape == image.shape
        assert not np.array_equal(stabilized, image)
    
    def test_crop_around_particle(self):
        """Test cropping around particle."""
        image = create_test_image(size=(200, 200))
        transformer = ImageTransformer()
        
        particle_pos = (100, 100, 10)
        crop_size = (50, 50)
        
        cropped = transformer.crop_around_particle(image, particle_pos, crop_size)
        
        assert cropped is not None
        assert cropped.shape == (50, 50)
    
    def test_crop_out_of_bounds(self):
        """Test cropping when particle is near edge."""
        image = create_test_image(size=(200, 200))
        transformer = ImageTransformer()
        
        # Particle very close to edge
        particle_pos = (10, 10, 10)
        crop_size = (100, 100)
        
        cropped = transformer.crop_around_particle(image, particle_pos, crop_size)
        
        # Should return None or handle edge case
        # (depending on implementation, might return None or smaller crop)
        assert cropped is None or cropped.shape[0] < 100
    
    def test_transform_coordinates(self):
        """Test coordinate transformation."""
        transformer = ImageTransformer()
        
        points = np.array([[10, 10], [20, 20], [30, 30]])
        particle_pos = (50, 50, 10)
        reference_pos = (100, 100, 10)
        
        transformed = transformer.transform_coordinates(
            points, particle_pos, reference_pos
        )
        
        assert transformed.shape == points.shape
        # Points should be translated by (50, 50)
        expected = points + np.array([50, 50])
        np.testing.assert_array_equal(transformed, expected)
