#!/usr/bin/env python3
"""
Extract the trajectory of a microparticle from a video file.
The microparticle has a diameter of approximately 48 pixels.
"""

import cv2
import numpy as np
import sys
import os
from config import get_data_path, get_csv_path, get_video_path
import csv


def detect_circle_in_frame(frame, expected_diameter=48):
    """
    Detect a circular microparticle in a single frame.

    Args:
        frame: Input frame (BGR image)
        expected_diameter: Expected diameter of the microparticle in pixels

    Returns:
        tuple: (center_x, center_y, radius) or None if not found
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Hough Circle Transform to detect circles
    min_radius = int(expected_diameter * 0.4)  # 19 pixels
    max_radius = int(expected_diameter * 0.7)  # 33 pixels

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

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        if len(circles) > 0:
            x, y, r = circles[0]
            return (x, y, r)

    return None


def extract_trajectory(video_path, expected_diameter=48, save_video=True):
    """
    Extract the trajectory of a microparticle from a video file.

    Args:
        video_path: Path to the input video file
        expected_diameter: Expected diameter of the microparticle in pixels
        save_video: If True, save an annotated video showing the trajectory

    Returns:
        list: List of tuples (frame_number, x, y, radius)
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
    print(f"\nProcessing frames...")

    trajectory = []
    frame_number = 0

    # Prepare video writer if saving annotated video
    if save_video:
        output_path = get_video_path(os.path.basename(video_path).replace('.mp4', '_trajectory.mp4'))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Detect circle in current frame
        result = detect_circle_in_frame(frame, expected_diameter)

        if result is not None:
            x, y, r = result
            trajectory.append((frame_number, x, y, r))

            if save_video:
                # Draw the detected circle
                cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

                # Draw trajectory path (connect previous positions)
                if len(trajectory) > 1:
                    for i in range(1, len(trajectory)):
                        pt1 = (trajectory[i-1][1], trajectory[i-1][2])
                        pt2 = (trajectory[i][1], trajectory[i][2])
                        cv2.line(frame, pt1, pt2, (255, 0, 0), 1)

                # Add frame number
                cv2.putText(frame, f"Frame: {frame_number}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
    Save trajectory data to a CSV file.

    Args:
        trajectory: List of tuples (frame_number, x, y, radius)
        output_path: Path to output CSV file
    """
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame', 'x', 'y', 'radius'])
        for frame, x, y, r in trajectory:
            writer.writerow([frame, x, y, r])

    print(f"Trajectory data saved to: {output_path}")


def main():
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = get_data_path("Ms_4_4_AuNP_moving_n0_laser_2_40_038.mp4")

    # Extract trajectory
    trajectory = extract_trajectory(video_path, expected_diameter=48, save_video=True)

    if trajectory:
        print(f"\nTrajectory summary:")
        print(f"  Number of detected positions: {len(trajectory)}")

        # Calculate some statistics
        x_coords = [x for _, x, _, _ in trajectory]
        y_coords = [y for _, y, _, _ in trajectory]

        print(f"  X range: {min(x_coords)} to {max(x_coords)} pixels")
        print(f"  Y range: {min(y_coords)} to {max(y_coords)} pixels")

        # Calculate displacement
        if len(trajectory) > 1:
            start_x, start_y = trajectory[0][1], trajectory[0][2]
            end_x, end_y = trajectory[-1][1], trajectory[-1][2]
            displacement = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            print(f"  Total displacement: {displacement:.1f} pixels")
            print(f"  Start position: ({start_x}, {start_y})")
            print(f"  End position: ({end_x}, {end_y})")

        # Save to CSV
        csv_path = get_csv_path(os.path.basename(video_path).replace('.mp4', '_trajectory.csv'))
        save_trajectory_to_csv(trajectory, csv_path)

        print(f"\nFirst 10 positions:")
        for i, (frame, x, y, r) in enumerate(trajectory[:10]):
            print(f"  Frame {frame}: ({x}, {y}), radius={r}")
    else:
        print("\nFailed to extract trajectory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
