"""
Example: Basic usage of the circle detection and tracking pipeline.

This example demonstrates how to:
1. Create synthetic test images with particles
2. Detect and track the main particle
3. Transform images to center the particle
"""

import numpy as np
import cv2
from circledetect import CircleDetector, ParticleTracker, ImageTransformer, ParticleTrackingPipeline


def create_synthetic_frame(frame_num, image_size=(400, 400)):
    """Create a synthetic frame with a moving particle and background tracers."""
    # Create blank image
    frame = np.zeros(image_size + (3,), dtype=np.uint8)
    
    # Add some background noise
    noise = np.random.randint(0, 30, image_size + (3,), dtype=np.uint8)
    frame = cv2.add(frame, noise)
    
    # Add main particle (moving in a circle)
    center_x = 200 + int(80 * np.cos(frame_num * 0.1))
    center_y = 200 + int(80 * np.sin(frame_num * 0.1))
    cv2.circle(frame, (center_x, center_y), 25, (255, 255, 255), -1)
    
    # Add tracer particles (smaller, stationary)
    tracer_positions = [(100, 100), (300, 100), (100, 300), (300, 300), (200, 50)]
    for pos in tracer_positions:
        cv2.circle(frame, pos, 8, (180, 180, 180), -1)
    
    return frame


def main():
    """Run the basic example."""
    print("CircleDetect - Basic Usage Example")
    print("=" * 50)
    
    # Create pipeline with custom parameters
    detector = CircleDetector(
        min_radius=15,
        max_radius=35,
        param1=100,
        param2=20,
        min_dist=30
    )
    tracker = ParticleTracker(max_history=50, max_distance=30.0)
    transformer = ImageTransformer()
    
    pipeline = ParticleTrackingPipeline(detector, tracker, transformer)
    
    # Generate and process synthetic frames
    num_frames = 30
    print(f"\nProcessing {num_frames} synthetic frames...")
    
    for i in range(num_frames):
        # Create synthetic frame
        frame = create_synthetic_frame(i)
        
        # Process frame
        processed, position = pipeline.process_frame(frame, transform=True)
        
        if position is not None:
            x, y, r = position
            print(f"Frame {i:3d}: Particle at ({x:3d}, {y:3d}), radius={r:2d}")
        else:
            print(f"Frame {i:3d}: No particle detected")
    
    # Get complete trajectory
    trajectory = pipeline.get_trajectory()
    print(f"\n✓ Tracked particle across {len(trajectory)} frames")
    
    # Show trajectory statistics
    if len(trajectory) > 0:
        x_coords = [pos[0] for pos in trajectory]
        y_coords = [pos[1] for pos in trajectory]
        print(f"\nTrajectory Statistics:")
        print(f"  X range: {min(x_coords)} - {max(x_coords)}")
        print(f"  Y range: {min(y_coords)} - {max(y_coords)}")
        print(f"  Mean position: ({np.mean(x_coords):.1f}, {np.mean(y_coords):.1f})")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")


if __name__ == "__main__":
    main()
