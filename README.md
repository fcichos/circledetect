# CircleDetect

Python pipeline to track a circular particle in a tracer particle background and to transform the images according to the particle position.

## Features

- **Circle Detection**: Detect circular particles in images using the Hough Circle Transform
- **Particle Tracking**: Track particles across frames with motion prediction
- **Image Transformation**: Transform images to center or stabilize based on particle position
- **Complete Pipeline**: Integrated pipeline for processing video sequences and image series

## Installation

### From source

```bash
git clone https://github.com/fcichos/circledetect.git
cd circledetect
pip install -e .
```

### Requirements

- Python >= 3.7
- numpy >= 1.20.0
- opencv-python >= 4.5.0
- scipy >= 1.7.0
- matplotlib >= 3.3.0

## Quick Start

### Basic Usage

```python
from circledetect import ParticleTrackingPipeline
import cv2

# Create pipeline
pipeline = ParticleTrackingPipeline()

# Process a video file
trajectory = pipeline.process_video(
    "input_video.mp4",
    output_path="output_video.mp4",
    transform=True,
    visualize=True
)

print(f"Tracked particle across {len(trajectory)} frames")
```

### Custom Configuration

```python
from circledetect import (
    CircleDetector,
    ParticleTracker,
    ImageTransformer,
    ParticleTrackingPipeline
)

# Configure components
detector = CircleDetector(
    min_radius=15,
    max_radius=50,
    param1=100,
    param2=30,
    min_dist=40
)

tracker = ParticleTracker(
    max_history=50,
    max_distance=30.0
)

transformer = ImageTransformer()

# Create pipeline with custom components
pipeline = ParticleTrackingPipeline(detector, tracker, transformer)

# Process a single frame
import cv2
frame = cv2.imread("image.png")
transformed_frame, position = pipeline.process_frame(frame, transform=True)

if position:
    x, y, radius = position
    print(f"Particle detected at ({x}, {y}) with radius {radius}")
```

### Processing Image Sequences

```python
import numpy as np
from circledetect import ParticleTrackingPipeline

# Load your image sequence
images = [...]  # List of numpy arrays

# Create and run pipeline
pipeline = ParticleTrackingPipeline()
processed_images, trajectory = pipeline.process_image_sequence(
    images,
    transform=True,
    visualize=True
)

# Access trajectory data
for i, (x, y, radius) in enumerate(trajectory):
    print(f"Frame {i}: position=({x}, {y}), radius={radius}")
```

## API Reference

### CircleDetector

Detects circular particles using the Hough Circle Transform.

```python
detector = CircleDetector(
    min_radius=10,      # Minimum circle radius to detect
    max_radius=100,     # Maximum circle radius to detect
    param1=100,         # Canny edge detection threshold
    param2=30,          # Circle detection threshold
    min_dist=50         # Minimum distance between circle centers
)

# Detect all circles
circles = detector.detect(image)

# Detect main particle (largest)
main_particle = detector.detect_main_particle(image)

# Detect main particle and tracers
main, tracers = detector.detect_all_particles(image)
```

### ParticleTracker

Tracks a particle across multiple frames with motion prediction.

```python
tracker = ParticleTracker(
    max_history=30,      # Number of frames to keep in history
    max_distance=50.0    # Maximum movement between frames (pixels)
)

# Update with new detection
tracked_position = tracker.update(detected_position)

# Get trajectory
trajectory = tracker.get_trajectory()

# Get current position
current = tracker.get_current_position()

# Reset tracker
tracker.reset()
```

### ImageTransformer

Transforms images based on particle position.

```python
transformer = ImageTransformer(center_position=(200, 200))

# Center image on particle
centered = transformer.center_on_particle(image, particle_position)

# Stabilize frame relative to reference
stabilized = transformer.stabilize_frame(
    image,
    particle_position,
    reference_position
)

# Crop around particle
cropped = transformer.crop_around_particle(
    image,
    particle_position,
    crop_size=(100, 100)
)

# Transform coordinates
transformed_coords = transformer.transform_coordinates(
    points,
    particle_position,
    reference_position
)
```

### ParticleTrackingPipeline

Complete pipeline integrating detection, tracking, and transformation.

```python
pipeline = ParticleTrackingPipeline(detector, tracker, transformer)

# Process single frame
processed_frame, position = pipeline.process_frame(frame, transform=True)

# Process video
trajectory = pipeline.process_video(
    video_path="input.mp4",
    output_path="output.mp4",
    transform=True,
    visualize=True
)

# Process image sequence
processed_images, trajectory = pipeline.process_image_sequence(
    images,
    transform=True,
    visualize=True
)

# Get trajectory
trajectory = pipeline.get_trajectory()

# Reset pipeline
pipeline.reset()
```

## Examples

See the `examples/` directory for complete examples:

- `basic_usage.py`: Simple example with synthetic data
- `advanced_usage.py`: Advanced features and trajectory analysis

Run examples:

```bash
python examples/basic_usage.py
python examples/advanced_usage.py
```

## Testing

Run tests with pytest:

```bash
pip install pytest
pytest tests/
```

## Use Cases

- Track microparticles in microscopy videos
- Analyze particle motion in fluid dynamics experiments
- Stabilize video based on tracked object position
- Extract particle trajectories for further analysis

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

fcichos