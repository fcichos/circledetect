"""
Circle detection module using Hough Circle Transform.

This module provides functionality to detect circular particles in images,
distinguishing the main particle from tracer particles in the background.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


class CircleDetector:
    """
    Detect circular particles in images using the Hough Circle Transform.
    
    Attributes:
        min_radius: Minimum radius of circles to detect
        max_radius: Maximum radius of circles to detect
        param1: First method-specific parameter for Canny edge detection
        param2: Accumulator threshold for circle centers
        min_dist: Minimum distance between detected circle centers
    """
    
    def __init__(
        self,
        min_radius: int = 10,
        max_radius: int = 100,
        param1: int = 100,
        param2: int = 30,
        min_dist: int = 50,
    ):
        """
        Initialize the CircleDetector.
        
        Args:
            min_radius: Minimum radius of circles to detect (default: 10)
            max_radius: Maximum radius of circles to detect (default: 100)
            param1: Canny edge detection threshold (default: 100)
            param2: Accumulator threshold for circle centers (default: 30)
            min_dist: Minimum distance between circle centers (default: 50)
        """
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.param1 = param1
        self.param2 = param2
        self.min_dist = min_dist
    
    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect circles in an image.
        
        Args:
            image: Input image (grayscale or color)
            
        Returns:
            Array of detected circles in format [[x, y, radius], ...] or None
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            return circles
        
        return None
    
    def detect_main_particle(
        self, image: np.ndarray, size_threshold: float = 1.5
    ) -> Optional[Tuple[int, int, int]]:
        """
        Detect the main circular particle (typically larger than tracer particles).
        
        Args:
            image: Input image
            size_threshold: Ratio threshold for main particle vs tracer particles
            
        Returns:
            Tuple of (x, y, radius) for the main particle or None
        """
        circles = self.detect(image)
        
        if circles is None or len(circles) == 0:
            return None
        
        # Sort circles by radius (descending)
        circles_sorted = circles[circles[:, 2].argsort()[::-1]]
        
        # If only one circle, return it
        if len(circles_sorted) == 1:
            return tuple(circles_sorted[0])
        
        # Check if largest circle is significantly larger than others
        largest_radius = circles_sorted[0, 2]
        second_largest_radius = circles_sorted[1, 2]
        
        if largest_radius >= size_threshold * second_largest_radius:
            return tuple(circles_sorted[0])
        
        # If no clear main particle, return the largest one
        return tuple(circles_sorted[0])
    
    def detect_all_particles(
        self, image: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
        """
        Detect both main particle and tracer particles.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (main_particle, tracer_particles) where main_particle is
            (x, y, radius) or None, and tracer_particles is a list of (x, y, radius)
        """
        circles = self.detect(image)
        
        if circles is None or len(circles) == 0:
            return None, []
        
        # Sort by radius
        circles_sorted = circles[circles[:, 2].argsort()[::-1]]
        
        # Assume largest is main particle
        main_particle = tuple(circles_sorted[0])
        tracer_particles = [tuple(c) for c in circles_sorted[1:]]
        
        return main_particle, tracer_particles
