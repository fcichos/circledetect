#!/usr/bin/env python3
"""
Extract the trajectory of a microparticle with subpixel resolution.
Uses Hough Circle Transform for initial detection, then refines to subpixel accuracy
using intensity-weighted centroid calculation.
"""

import cv2
import numpy as np
import sys
import os
from config import get_data_path, get_csv_path, get_video_path
import csv


def detect_circle_subpixel(frame, expected_diameter=48):
    """
    Detect a circular microparticle with subpixel accuracy.

    Args:
        frame: Input frame (BGR image)
        expected_diameter: Expected diameter of the microparticle in pixels

    Returns:
        tuple: (center_x, center_y, radius) with subpixel coordinates, or None if not found
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Hough Circle Transform for initial detection
    min_radius = int(expected_diameter * 0.4)
    max_radius = int(expected_diameter * 0.7)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype("int")
    if len(circles) == 0:
        return None

    # Use first detected circle as initial estimate
    x_init, y_init, r_init = circles[0]

    # Refine to subpixel accuracy using circular mask and intensity-weighted centroid
    # Extract region around the detected circle (just slightly larger than the circle)
    roi_size = int(r_init * 1.5)
    x1 = max(0, x_init - roi_size)
    y1 = max(0, y_init - roi_size)
    x2 = min(gray.shape[1], x_init + roi_size)
    y2 = min(gray.shape[0], y_init + roi_size)

    roi = gray[y1:y2, x1:x2].astype(float)

    # Create circular mask centered on the detected circle
    # This ensures we only use pixels within the particle
    mask = np.zeros(roi.shape, dtype=np.uint8)
    center_in_roi = (x_init - x1, y_init - y1)
    cv2.circle(mask, center_in_roi, r_init, 255, -1)

    # Apply mask to ROI
    roi_masked = roi * (mask.astype(float) / 255.0)

    # Calculate intensity-weighted centroid within the circular mask
    total_intensity = np.sum(roi_masked)

    if total_intensity > 0:
        y_indices, x_indices = np.indices(roi_masked.shape)
        cx_roi = np.sum(x_indices * roi_masked) / total_intensity
        cy_roi = np.sum(y_indices * roi_masked) / total_intensity

        # Convert back to full image coordinates
        cx_subpixel = x1 + cx_roi
        cy_subpixel = y1 + cy_roi
    else:
        # Fallback to integer coordinates if calculation fails
        cx_subpixel = float(x_init)
        cy_subpixel = float(y_init)

    return (cx_subpixel, cy_subpixel, float(r_init))


def extract_trajectory_subpixel(video_path, expected_diameter=48, save_video=True):
    """
    Extract the trajectory of a microparticle with subpixel resolution.

    Args:
        video_path: Path to the input video file
        expected_diameter: Expected diameter of the microparticle in pixels
        save_video: If True, save an annotated video showing the trajectory

    Returns:
        list: List of tuples (frame_number, x, y, radius) with subpixel coordinates
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"\nProcessing frames with SUBPIXEL accuracy...")

    trajectory = []
    frame_number = 0

    # Prepare video writer if saving annotated video
    if save_video:
        output_path = get_video_path(os.path.basename(video_path).replace('.mp4', '_trajectory_subpixel.mp4'))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Detect circle with subpixel accuracy
        result = detect_circle_subpixel(frame, expected_diameter)

        if result is not None:
            x, y, r = result
            trajectory.append((frame_number, x, y, r))

            if save_video:
                # Draw the detected circle (use integer coords for drawing)
                ix, iy = int(round(x)), int(round(y))
                cv2.circle(frame, (ix, iy), int(round(r)), (0, 255, 0), 2)
                cv2.circle(frame, (ix, iy), 3, (0, 0, 255), -1)

                # Draw trajectory path with subpixel precision
                if len(trajectory) > 1:
                    for i in range(1, len(trajectory)):
                        pt1 = (int(round(trajectory[i-1][1])), int(round(trajectory[i-1][2])))
                        pt2 = (int(round(trajectory[i][1])), int(round(trajectory[i][2])))
                        cv2.line(frame, pt1, pt2, (255, 0, 0), 1)

                # Add frame number and coordinates with subpixel precision
                cv2.putText(frame, f"Frame: {frame_number}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Pos: ({x:.2f}, {y:.2f})", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                out.write(frame)

        frame_number += 1

        # Print progress
        if frame_number % 10 == 0:
            print(f"  Processed {frame_number}/{total_frames} frames, detected {len(trajectory)} positions")

    cap.release()

    if save_video:
        out.release()
        print(f"\nAnnotated video saved to: {output_path}")

    print(f"\nTotal frames processed: {frame_number}")
    print(f"Particle detected in {len(trajectory)} frames ({100*len(trajectory)/frame_number:.1f}%)")

    return trajectory


def save_trajectory_to_csv(trajectory, output_path):
    """
    Save trajectory data to a CSV file with subpixel precision.

    Args:
        trajectory: List of tuples (frame_number, x, y, radius)
        output_path: Path to output CSV file
    """
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame', 'x', 'y', 'radius'])
        for frame, x, y, r in trajectory:
            writer.writerow([frame, f'{x:.6f}', f'{y:.6f}', f'{r:.6f}'])

    print(f"Trajectory data saved to: {output_path}")


def main():
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = get_data_path("Ms_4_4_AuNP_moving_n0_laser_2_40_038.mp4")

    # Extract trajectory with subpixel accuracy
    trajectory = extract_trajectory_subpixel(video_path, expected_diameter=48, save_video=True)

    if trajectory:
        print(f"\nTrajectory summary:")
        print(f"  Number of detected positions: {len(trajectory)}")

        # Calculate statistics
        x_coords = np.array([x for _, x, _, _ in trajectory])
        y_coords = np.array([y for _, y, _, _ in trajectory])

        print(f"  X range: {x_coords.min():.3f} to {x_coords.max():.3f} pixels")
        print(f"  Y range: {y_coords.min():.3f} to {y_coords.max():.3f} pixels")

        # Calculate displacement
        if len(trajectory) > 1:
            start_x, start_y = trajectory[0][1], trajectory[0][2]
            end_x, end_y = trajectory[-1][1], trajectory[-1][2]
            displacement = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            print(f"  Total displacement: {displacement:.3f} pixels")
            print(f"  Start position: ({start_x:.3f}, {start_y:.3f})")
            print(f"  End position: ({end_x:.3f}, {end_y:.3f})")

            # Calculate frame-to-frame distances for smoothness comparison
            distances = []
            for i in range(1, len(trajectory)):
                dx = trajectory[i][1] - trajectory[i-1][1]
                dy = trajectory[i][2] - trajectory[i-1][2]
                distances.append(np.sqrt(dx**2 + dy**2))

            distances = np.array(distances)
            print(f"\nTrajectory smoothness:")
            print(f"  Mean step size: {distances.mean():.3f} ± {distances.std():.3f} pixels")
            print(f"  Min/Max step: {distances.min():.3f} / {distances.max():.3f} pixels")

        # Save to CSV with subpixel precision
        csv_path = get_csv_path(os.path.basename(video_path).replace('.mp4', '_trajectory_subpixel.csv'))
        save_trajectory_to_csv(trajectory, csv_path)

        print(f"\nFirst 10 positions (subpixel):")
        for i, (frame, x, y, r) in enumerate(trajectory[:10]):
            print(f"  Frame {frame}: ({x:.3f}, {y:.3f}), radius={r:.1f}")
    else:
        print("\nFailed to extract trajectory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
