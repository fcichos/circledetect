"""
Main pipeline for particle tracking and image transformation.

This module integrates detection, tracking, and transformation into a
complete pipeline for processing video sequences.
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple, Callable
from pathlib import Path

from .detector import CircleDetector
from .tracker import ParticleTracker
from .transformer import ImageTransformer


class ParticleTrackingPipeline:
    """
    Complete pipeline for tracking circular particles and transforming images.
    
    Integrates circle detection, particle tracking, and image transformation
    for processing video sequences or image series.
    """
    
    def __init__(
        self,
        detector: Optional[CircleDetector] = None,
        tracker: Optional[ParticleTracker] = None,
        transformer: Optional[ImageTransformer] = None,
    ):
        """
        Initialize the tracking pipeline.
        
        Args:
            detector: CircleDetector instance (default: creates new with defaults)
            tracker: ParticleTracker instance (default: creates new with defaults)
            transformer: ImageTransformer instance (default: creates new)
        """
        self.detector = detector if detector is not None else CircleDetector()
        self.tracker = tracker if tracker is not None else ParticleTracker()
        self.transformer = transformer if transformer is not None else ImageTransformer()
    
    def process_frame(
        self, frame: np.ndarray, transform: bool = True
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int, int]]]:
        """
        Process a single frame.
        
        Args:
            frame: Input frame
            transform: Whether to apply transformation (default: True)
            
        Returns:
            Tuple of (processed_frame, particle_position)
        """
        # Detect main particle
        detected = self.detector.detect_main_particle(frame)
        
        # Update tracker
        tracked_pos = self.tracker.update(detected)
        
        # Transform frame if requested
        if transform and tracked_pos is not None:
            transformed = self.transformer.center_on_particle(frame, tracked_pos)
        else:
            transformed = frame.copy()
        
        return transformed, tracked_pos
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        transform: bool = True,
        visualize: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Tuple[int, int, int]]:
        """
        Process a complete video file.
        
        Args:
            video_path: Path to input video
            output_path: Path for output video (optional)
            transform: Whether to transform frames (default: True)
            visualize: Whether to draw detection/tracking visualizations (default: True)
            progress_callback: Optional callback function(frame_num, total_frames)
            
        Returns:
            List of tracked particle positions
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output video writer if requested
        writer = None
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        trajectory = []
        frame_num = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed, position = self.process_frame(frame, transform=transform)
                
                # Store trajectory
                if position is not None:
                    trajectory.append(position)
                
                # Visualize if requested
                if visualize and position is not None:
                    self._draw_particle(processed, position)
                
                # Write output
                if writer is not None:
                    writer.write(processed)
                
                # Progress callback
                frame_num += 1
                if progress_callback is not None:
                    progress_callback(frame_num, total_frames)
        
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        
        return trajectory
    
    def process_image_sequence(
        self,
        images: List[np.ndarray],
        transform: bool = True,
        visualize: bool = True,
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
        """
        Process a sequence of images.
        
        Args:
            images: List of input images
            transform: Whether to transform images (default: True)
            visualize: Whether to draw visualizations (default: True)
            
        Returns:
            Tuple of (processed_images, trajectory)
        """
        processed_images = []
        trajectory = []
        
        for image in images:
            processed, position = self.process_frame(image, transform=transform)
            
            if visualize and position is not None:
                self._draw_particle(processed, position)
            
            processed_images.append(processed)
            if position is not None:
                trajectory.append(position)
        
        return processed_images, trajectory
    
    def _draw_particle(
        self, image: np.ndarray, position: Tuple[int, int, int], color: Tuple[int, int, int] = (0, 255, 0)
    ):
        """
        Draw particle detection on image (modifies image in-place).
        
        Args:
            image: Image to draw on
            position: Particle position (x, y, radius)
            color: Circle color in BGR (default: green)
        """
        x, y, r = position
        cv2.circle(image, (x, y), r, color, 2)
        cv2.circle(image, (x, y), 2, color, 3)
    
    def get_trajectory(self) -> List[Tuple[int, int, int]]:
        """
        Get the tracked particle trajectory.
        
        Returns:
            List of (x, y, radius) positions
        """
        return self.tracker.get_trajectory()
    
    def reset(self):
        """Reset the pipeline, clearing all tracking history."""
        self.tracker.reset()
