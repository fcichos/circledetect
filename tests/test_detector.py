"""Unit tests for CircleDetector."""

import numpy as np
import cv2
import pytest
from circledetect.detector import CircleDetector


def create_test_image_with_circle(size=(200, 200), circle_pos=(100, 100), radius=30):
    """Create a test image with a single circle."""
    image = np.zeros(size, dtype=np.uint8)
    cv2.circle(image, circle_pos, radius, 255, -1)
    return image


def create_test_image_with_multiple_circles(size=(300, 300)):
    """Create a test image with multiple circles (one large, several small)."""
    image = np.zeros(size, dtype=np.uint8)
    # Large circle (main particle)
    cv2.circle(image, (150, 150), 40, 255, -1)
    # Small circles (tracer particles)
    cv2.circle(image, (50, 50), 10, 200, -1)
    cv2.circle(image, (250, 50), 10, 200, -1)
    cv2.circle(image, (50, 250), 10, 200, -1)
    return image


class TestCircleDetector:
    """Test cases for CircleDetector class."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = CircleDetector()
        assert detector.min_radius == 10
        assert detector.max_radius == 100
        
        detector = CircleDetector(min_radius=5, max_radius=50)
        assert detector.min_radius == 5
        assert detector.max_radius == 50
    
    def test_detect_single_circle(self):
        """Test detection of a single circle."""
        image = create_test_image_with_circle()
        detector = CircleDetector(min_radius=20, max_radius=40, param2=20)
        
        circles = detector.detect(image)
        assert circles is not None
        assert len(circles) > 0
    
    def test_detect_no_circles(self):
        """Test detection when no circles present."""
        image = np.zeros((200, 200), dtype=np.uint8)
        detector = CircleDetector()
        
        circles = detector.detect(image)
        assert circles is None
    
    def test_detect_main_particle(self):
        """Test detection of main particle among multiple circles."""
        image = create_test_image_with_multiple_circles()
        detector = CircleDetector(min_radius=5, max_radius=50, param2=15)
        
        main_particle = detector.detect_main_particle(image)
        assert main_particle is not None
        x, y, r = main_particle
        # Main particle should be near center with larger radius
        assert r > 30  # Should detect the large circle
    
    def test_detect_all_particles(self):
        """Test detection of all particles."""
        image = create_test_image_with_multiple_circles()
        detector = CircleDetector(min_radius=5, max_radius=50, param2=15)
        
        main, tracers = detector.detect_all_particles(image)
        assert main is not None
        assert isinstance(tracers, list)
