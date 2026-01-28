"""
Example: Advanced usage with custom image sequence.

This example demonstrates:
1. Processing custom images
2. Visualizing tracking results
3. Exporting transformed image sequence
"""

import numpy as np
import cv2
from pathlib import Path
from circledetect import CircleDetector, ParticleTracker, ImageTransformer


def create_particle_sequence(num_frames=20, size=(300, 300)):
    """Create a sequence of images with a moving particle."""
    frames = []
    
    for i in range(num_frames):
        # Create frame
        frame = np.ones(size + (3,), dtype=np.uint8) * 50
        
        # Main particle moving diagonally
        x = 50 + i * 10
        y = 50 + i * 8
        cv2.circle(frame, (x, y), 20, (255, 255, 255), -1)
        
        # Add some tracer particles
        for j in range(5):
            tx = 50 + j * 50
            ty = 250 - j * 30
            cv2.circle(frame, (tx, ty), 6, (200, 200, 200), -1)
        
        frames.append(frame)
    
    return frames


def main():
    """Run the advanced example."""
    print("CircleDetect - Advanced Usage Example")
    print("=" * 50)
    
    # Create image sequence
    print("\nGenerating synthetic image sequence...")
    frames = create_particle_sequence(num_frames=15)
    print(f"✓ Generated {len(frames)} frames")
    
    # Initialize components
    detector = CircleDetector(min_radius=12, max_radius=30, param2=15)
    tracker = ParticleTracker(max_history=30, max_distance=20.0)
    transformer = ImageTransformer()
    
    # Process each frame
    print("\nProcessing frames...")
    transformed_frames = []
    
    for i, frame in enumerate(frames):
        # Detect particle
        particle = detector.detect_main_particle(frame)
        
        # Update tracker
        tracked_pos = tracker.update(particle)
        
        if tracked_pos is not None:
            # Transform frame to center particle
            transformed = transformer.center_on_particle(frame, tracked_pos)
            
            # Draw visualization
            x, y, r = tracked_pos
            cv2.circle(transformed, (x, y), r, (0, 255, 0), 2)
            cv2.circle(transformed, (x, y), 2, (0, 0, 255), 3)
            
            # Add frame number
            cv2.putText(
                transformed,
                f"Frame {i}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            transformed_frames.append(transformed)
            print(f"  Frame {i:2d}: Tracked at ({x:3d}, {y:3d})")
        else:
            transformed_frames.append(frame)
            print(f"  Frame {i:2d}: Not tracked")
    
    # Analyze trajectory
    trajectory = tracker.get_trajectory()
    print(f"\n✓ Successfully tracked {len(trajectory)} positions")
    
    if len(trajectory) > 1:
        # Calculate movement statistics
        movements = []
        for i in range(1, len(trajectory)):
            prev = np.array(trajectory[i-1][:2])
            curr = np.array(trajectory[i][:2])
            dist = np.linalg.norm(curr - prev)
            movements.append(dist)
        
        print(f"\nMovement Statistics:")
        print(f"  Average displacement: {np.mean(movements):.2f} pixels")
        print(f"  Max displacement: {np.max(movements):.2f} pixels")
        print(f"  Total path length: {np.sum(movements):.2f} pixels")
    
    print("\n" + "=" * 50)
    print("Advanced example completed!")
    print(f"Processed {len(transformed_frames)} frames successfully")


if __name__ == "__main__":
    main()
