"""
Transform video to particle reference frame where the microparticle appears
centered and stationary, with velocity vector aligned to horizontal axis.
"""

import cv2
import numpy as np
import csv
from pathlib import Path


def load_trajectory(csv_path):
    """
    Load trajectory data from CSV file.

    Args:
        csv_path: Path to CSV file with columns: frame, x, y, radius

    Returns:
        dict: {frame_number: (x, y, radius)}
    """
    trajectory = {}

    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            frame = int(row['frame'])
            x = float(row['x'])
            y = float(row['y'])
            radius = float(row['radius'])
            trajectory[frame] = (x, y, radius)

    return trajectory


def compute_velocities(trajectory, fps, smoothing_window=5):
    """
    Compute velocity vectors with smoothing.

    Args:
        trajectory: dict {frame: (x, y, radius)}
        fps: Frames per second
        smoothing_window: Window size for moving average smoothing (odd number)

    Returns:
        dict: {frame: (vx, vy)} in pixels/second
    """
    # Get sorted frame numbers
    frames = sorted(trajectory.keys())

    if len(frames) < 2:
        return {}

    # Compute raw velocities using centered finite differences
    raw_velocities = {}

    for i, frame in enumerate(frames):
        if i == 0:
            # Forward difference for first frame
            x0, y0, _ = trajectory[frames[i]]
            x1, y1, _ = trajectory[frames[i + 1]]
            dt = (frames[i + 1] - frames[i]) / fps
            vx = (x1 - x0) / dt
            vy = (y1 - y0) / dt
        elif i == len(frames) - 1:
            # Backward difference for last frame
            x0, y0, _ = trajectory[frames[i - 1]]
            x1, y1, _ = trajectory[frames[i]]
            dt = (frames[i] - frames[i - 1]) / fps
            vx = (x1 - x0) / dt
            vy = (y1 - y0) / dt
        else:
            # Centered difference for middle frames
            x_prev, y_prev, _ = trajectory[frames[i - 1]]
            x_next, y_next, _ = trajectory[frames[i + 1]]
            dt = (frames[i + 1] - frames[i - 1]) / fps
            vx = (x_next - x_prev) / dt
            vy = (y_next - y_prev) / dt

        raw_velocities[frame] = (vx, vy)

    # Apply moving average smoothing if window > 1
    if smoothing_window > 1:
        smoothed_velocities = {}
        half_window = smoothing_window // 2

        for i, frame in enumerate(frames):
            # Determine window bounds
            start_idx = max(0, i - half_window)
            end_idx = min(len(frames), i + half_window + 1)

            # Calculate average velocity in window
            vx_sum = 0.0
            vy_sum = 0.0
            count = 0

            for j in range(start_idx, end_idx):
                vx, vy = raw_velocities[frames[j]]
                vx_sum += vx
                vy_sum += vy
                count += 1

            smoothed_velocities[frame] = (vx_sum / count, vy_sum / count)

        return smoothed_velocities
    else:
        return raw_velocities


def compute_output_size(video_width, video_height, trajectory, apply_rotation=True):
    """
    Calculate square canvas size to fit all frames.

    Args:
        video_width: Original video width
        video_height: Original video height
        trajectory: dict {frame: (x, y, radius)}
        apply_rotation: If True, use diagonal for rotation; if False, use original size

    Returns:
        tuple: (output_width, output_height) - square canvas
    """
    if apply_rotation:
        # Use diagonal of original frame as base (to fit rotated frames)
        base_size = int(np.sqrt(video_width**2 + video_height**2))
    else:
        # No rotation - use original dimensions
        base_size = max(video_width, video_height)

    # Add margin for particle motion range
    if len(trajectory) > 0:
        positions = list(trajectory.values())
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]

        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        max_range = max(x_range, y_range)

        # Add 50% margin for safety
        margin = int(max_range * 0.5)
    else:
        margin = 100

    output_size = base_size + margin

    # Make it even for video encoding
    if output_size % 2 != 0:
        output_size += 1

    return (output_size, output_size)


def transform_frame(frame, px, py, vx, vy, output_width, output_height, apply_rotation=True):
    """
    Apply affine transformation to center particle and optionally align velocity.

    Args:
        frame: Input frame (BGR image)
        px, py: Particle position in frame
        vx, vy: Velocity vector (pixels/second)
        output_width, output_height: Output canvas size
        apply_rotation: If True, rotate to align velocity; if False, only translate

    Returns:
        tuple: (transformed_frame, rotation_angle_deg)
    """
    # Calculate output center
    cx = output_width / 2.0
    cy = output_height / 2.0

    if apply_rotation:
        # Compute rotation angle to align velocity with horizontal axis (pointing right)
        # Negative angle for counter-clockwise rotation
        if vx == 0 and vy == 0:
            theta_rad = 0.0
        else:
            theta_rad = -np.arctan2(vy, vx)

        theta_deg = np.degrees(theta_rad)

        # Create combined affine transformation matrix:
        # 1. Rotate around particle position
        # 2. Translate to center of output canvas
        M = cv2.getRotationMatrix2D((px, py), theta_deg, 1.0)
        M[0, 2] += (cx - px)
        M[1, 2] += (cy - py)
    else:
        # No rotation - just translate to center
        theta_deg = 0.0
        M = np.array([
            [1.0, 0.0, cx - px],
            [0.0, 1.0, cy - py]
        ], dtype=np.float32)

    # Apply transformation
    frame_transformed = cv2.warpAffine(
        frame, M, (output_width, output_height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)  # Black padding
    )

    return frame_transformed, theta_deg


def interpolate_missing_frames(trajectory, total_frames):
    """
    Interpolate positions for missing frames.

    Args:
        trajectory: dict {frame: (x, y, radius)}
        total_frames: Total number of frames in video

    Returns:
        dict: Complete trajectory with interpolated values
    """
    if len(trajectory) == 0:
        return {}

    frames = sorted(trajectory.keys())
    complete_trajectory = trajectory.copy()

    # Interpolate missing frames between detected frames
    for i in range(len(frames) - 1):
        frame1 = frames[i]
        frame2 = frames[i + 1]

        if frame2 - frame1 > 1:
            # Linear interpolation for missing frames
            x1, y1, r1 = trajectory[frame1]
            x2, y2, r2 = trajectory[frame2]

            for frame in range(frame1 + 1, frame2):
                alpha = (frame - frame1) / (frame2 - frame1)
                x = x1 + alpha * (x2 - x1)
                y = y1 + alpha * (y2 - y1)
                r = r1 + alpha * (r2 - r1)
                complete_trajectory[frame] = (x, y, r)

    return complete_trajectory


def handle_zero_velocity(velocities, frame):
    """
    Get velocity for frame, using previous non-zero velocity if current is zero.

    Args:
        velocities: dict {frame: (vx, vy)}
        frame: Current frame number

    Returns:
        tuple: (vx, vy)
    """
    frames = sorted(velocities.keys())

    if frame not in velocities:
        # Use nearest frame
        idx = min(range(len(frames)), key=lambda i: abs(frames[i] - frame))
        frame = frames[idx]

    vx, vy = velocities[frame]

    # If velocity is very small (near zero), look for previous non-zero velocity
    if abs(vx) < 0.1 and abs(vy) < 0.1:
        frame_idx = frames.index(frame)
        for i in range(frame_idx - 1, -1, -1):
            prev_vx, prev_vy = velocities[frames[i]]
            if abs(prev_vx) >= 0.1 or abs(prev_vy) >= 0.1:
                return prev_vx, prev_vy

        # If no previous non-zero velocity, look forward
        for i in range(frame_idx + 1, len(frames)):
            next_vx, next_vy = velocities[frames[i]]
            if abs(next_vx) >= 0.1 or abs(next_vy) >= 0.1:
                return next_vx, next_vy

        # If all velocities are zero, return (1, 0) as default
        return 1.0, 0.0

    return vx, vy


def transform_video_to_particle_frame(
    video_path,
    trajectory_csv_path,
    output_dir='./results',
    velocity_smoothing_window=5,
    annotate=False,
    apply_rotation=True,
    progress_callback=None
):
    """
    Transform video to particle reference frame.

    Args:
        video_path: Path to input video
        trajectory_csv_path: Path to trajectory CSV
        output_dir: Output directory for results
        velocity_smoothing_window: Window size for velocity smoothing
        annotate: If True, create annotated video with axes and info
        apply_rotation: If True, rotate frames to align velocity; if False, only translate
        progress_callback: Optional callback function(frame_num, total_frames)

    Returns:
        tuple: (output_video_path, params_csv_path)
    """
    # Setup output directories
    output_dir = Path(output_dir)
    csv_dir = output_dir / 'csv'
    video_dir = output_dir / 'videos'
    csv_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading trajectory from: {trajectory_csv_path}")
    trajectory = load_trajectory(trajectory_csv_path)

    if len(trajectory) == 0:
        print("Error: Empty trajectory file")
        return None, None

    print(f"Loaded {len(trajectory)} trajectory points")

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None, None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")

    # Interpolate missing frames
    print(f"\nInterpolating missing frames...")
    trajectory = interpolate_missing_frames(trajectory, total_frames)
    print(f"Complete trajectory: {len(trajectory)} points")

    # Compute velocities
    print(f"\nComputing velocities (smoothing window: {velocity_smoothing_window})...")
    velocities = compute_velocities(trajectory, fps, velocity_smoothing_window)
    print(f"Computed velocities for {len(velocities)} frames")

    # Calculate output size
    output_width, output_height = compute_output_size(width, height, trajectory, apply_rotation)
    print(f"\nOutput canvas size: {output_width}x{output_height}")
    if not apply_rotation:
        print(f"  (rotation disabled - using smaller canvas)")

    # Prepare output paths
    basename = Path(video_path).stem
    output_video_path = str(video_dir / f"{basename}_particle_frame.mp4")
    params_csv_path = str(csv_dir / f"{basename}_particle_frame_params.csv")

    if annotate:
        annotated_video_path = str(video_dir / f"{basename}_particle_frame_annotated.mp4")

    # Prepare video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

    if annotate:
        out_annotated = cv2.VideoWriter(annotated_video_path, fourcc, fps, (output_width, output_height))

    # Prepare params CSV
    params_file = open(params_csv_path, 'w', newline='')
    params_writer = csv.writer(params_file)
    params_writer.writerow(['frame', 'tx', 'ty', 'rotation_angle_deg', 'vx', 'vy'])

    print(f"\nTransforming video...")
    frame_number = 0
    processed_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_number in trajectory:
            px, py, radius = trajectory[frame_number]
            vx, vy = handle_zero_velocity(velocities, frame_number)

            # Transform frame
            frame_transformed, rotation_angle = transform_frame(
                frame, px, py, vx, vy, output_width, output_height, apply_rotation
            )

            # Calculate translation (particle position to canvas center)
            tx = output_width / 2.0 - px
            ty = output_height / 2.0 - py

            # Write transformed frame
            out.write(frame_transformed)

            # Write transformation parameters
            params_writer.writerow([
                frame_number,
                f'{tx:.6f}',
                f'{ty:.6f}',
                f'{rotation_angle:.6f}',
                f'{vx:.6f}',
                f'{vy:.6f}'
            ])

            # Create annotated version if requested
            if annotate:
                frame_annotated = frame_transformed.copy()

                # Draw coordinate axes
                center = (output_width // 2, output_height // 2)
                axis_length = 100

                # Horizontal axis (red) - velocity direction
                cv2.arrowedLine(frame_annotated, center,
                               (center[0] + axis_length, center[1]),
                               (0, 0, 255), 2, tipLength=0.3)
                cv2.putText(frame_annotated, "V", (center[0] + axis_length + 10, center[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Vertical axis (green)
                cv2.arrowedLine(frame_annotated, center,
                               (center[0], center[1] - axis_length),
                               (0, 255, 0), 2, tipLength=0.3)

                # Draw particle circle
                cv2.circle(frame_annotated, center, int(radius), (255, 255, 0), 2)

                # Add info text
                cv2.putText(frame_annotated, f"Frame: {frame_number}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame_annotated, f"V: ({vx:.1f}, {vy:.1f}) px/s", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame_annotated, f"Angle: {rotation_angle:.1f} deg", (10, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                out_annotated.write(frame_annotated)

            processed_count += 1
        else:
            # Frame not in trajectory - write black frame
            black_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            out.write(black_frame)

            if annotate:
                out_annotated.write(black_frame)

        frame_number += 1

        # Progress callback
        if progress_callback:
            progress_callback(frame_number, total_frames)
        elif frame_number % 30 == 0:
            print(f"  Processed {frame_number}/{total_frames} frames ({processed_count} transformed)")

    # Cleanup
    cap.release()
    out.release()
    params_file.close()

    if annotate:
        out_annotated.release()

    print(f"\nTransformation complete!")
    print(f"  Total frames processed: {frame_number}")
    print(f"  Frames transformed: {processed_count}")
    print(f"\nOutputs:")
    print(f"  Transformed video: {output_video_path}")
    print(f"  Parameters CSV: {params_csv_path}")

    if annotate:
        print(f"  Annotated video: {annotated_video_path}")

    return output_video_path, params_csv_path
