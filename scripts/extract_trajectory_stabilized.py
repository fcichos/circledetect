#!/usr/bin/env python3
"""
Extract the trajectory of a microparticle with stabilization using Kalman filtering.
Handles noise from small nanoparticles crossing the large particle.
"""

import cv2
import numpy as np
import sys
import os
from config import get_data_path, get_csv_path, get_video_path
import csv


class KalmanTracker:
    """Kalman filter for smooth particle tracking."""

    def __init__(self):
        """Initialize Kalman filter for 2D position tracking with velocity."""
        # State: [x, y, vx, vy] - position and velocity
        self.kalman = cv2.KalmanFilter(4, 2)

        # Transition matrix (constant velocity model)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ], dtype=np.float32)

        # Measurement matrix (we only measure position)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # Process noise covariance (how much we trust the model)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        # Measurement noise covariance (how much we trust measurements)
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

        # Error covariance
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)

        self.initialized = False

    def initialize(self, x, y):
        """Initialize the filter with first measurement."""
        self.kalman.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.kalman.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.initialized = True

    def predict(self):
        """Predict next position."""
        prediction = self.kalman.predict()
        return prediction[0, 0], prediction[1, 0]

    def update(self, x, y):
        """Update filter with new measurement."""
        measurement = np.array([[x], [y]], dtype=np.float32)
        self.kalman.correct(measurement)

        # Get corrected state
        state = self.kalman.statePost
        return state[0, 0], state[1, 0]

    def get_state(self):
        """Get current state estimate."""
        state = self.kalman.statePost
        return state[0, 0], state[1, 0], state[2, 0], state[3, 0]


def detect_circle_robust(frame, expected_diameter=48, prediction=None):
    """
    Detect a circular microparticle with robust handling of nanoparticles.

    Args:
        frame: Input frame (BGR image)
        expected_diameter: Expected diameter of the microparticle in pixels
        prediction: Optional (x, y) prediction from Kalman filter

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

    # If we have a prediction, search in a limited region
    if prediction is not None:
        pred_x, pred_y = prediction
        search_radius = 50  # Search within 50 pixels of prediction

        # Create a mask to limit search region
        mask = np.zeros(blurred.shape, dtype=np.uint8)
        cv2.circle(mask, (int(pred_x), int(pred_y)), search_radius, 255, -1)
        blurred_masked = cv2.bitwise_and(blurred, blurred, mask=mask)
    else:
        blurred_masked = blurred

    circles = cv2.HoughCircles(
        blurred_masked,
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

    # If we have prediction, choose circle closest to prediction
    if prediction is not None:
        pred_x, pred_y = prediction
        distances = [np.sqrt((c[0] - pred_x)**2 + (c[1] - pred_y)**2) for c in circles]
        best_idx = np.argmin(distances)
        x_init, y_init, r_init = circles[best_idx]
    else:
        x_init, y_init, r_init = circles[0]

    # Refine to subpixel accuracy with robust centroid calculation
    roi_size = int(r_init * 1.5)
    x1 = max(0, x_init - roi_size)
    y1 = max(0, y_init - roi_size)
    x2 = min(gray.shape[1], x_init + roi_size)
    y2 = min(gray.shape[0], y_init + roi_size)

    roi = gray[y1:y2, x1:x2].astype(float)

    # Create circular mask
    mask = np.zeros(roi.shape, dtype=np.uint8)
    center_in_roi = (x_init - x1, y_init - y1)
    cv2.circle(mask, center_in_roi, r_init, 255, -1)

    # Apply mask to ROI
    roi_masked = roi * (mask.astype(float) / 255.0)

    # Robust centroid calculation: Remove outlier bright pixels (nanoparticles)
    # Calculate intensity threshold to exclude very bright spots
    masked_values = roi_masked[roi_masked > 0]
    if len(masked_values) > 0:
        # Use median-based threshold to remove outliers
        median_intensity = np.median(masked_values)
        std_intensity = np.std(masked_values)

        # Clip intensities above median + 2*std (removes bright nanoparticles)
        roi_clipped = roi_masked.copy()
        threshold = median_intensity + 2 * std_intensity
        roi_clipped[roi_clipped > threshold] = median_intensity

        # Calculate intensity-weighted centroid
        total_intensity = np.sum(roi_clipped)

        if total_intensity > 0:
            y_indices, x_indices = np.indices(roi_clipped.shape)
            cx_roi = np.sum(x_indices * roi_clipped) / total_intensity
            cy_roi = np.sum(y_indices * roi_clipped) / total_intensity

            # Convert back to full image coordinates
            cx_subpixel = x1 + cx_roi
            cy_subpixel = y1 + cy_roi
        else:
            cx_subpixel = float(x_init)
            cy_subpixel = float(y_init)
    else:
        cx_subpixel = float(x_init)
        cy_subpixel = float(y_init)

    return (cx_subpixel, cy_subpixel, float(r_init))


def extract_trajectory_stabilized(video_path, expected_diameter=48, save_video=True):
    """
    Extract trajectory with Kalman filtering for stability.

    Args:
        video_path: Path to the input video file
        expected_diameter: Expected diameter of the microparticle in pixels
        save_video: If True, save an annotated video

    Returns:
        tuple: (raw_trajectory, filtered_trajectory)
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None, None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"\nProcessing with Kalman filter stabilization...")

    raw_trajectory = []
    filtered_trajectory = []
    frame_number = 0

    # Initialize Kalman tracker
    kalman = KalmanTracker()

    # Prepare video writer if saving annotated video
    if save_video:
        output_path = get_video_path(os.path.basename(video_path).replace('.mp4', '_trajectory_stabilized.mp4'))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Get prediction from Kalman filter if initialized
        prediction = None
        if kalman.initialized:
            prediction = kalman.predict()

        # Detect circle with prediction
        result = detect_circle_robust(frame, expected_diameter, prediction)

        if result is not None:
            x_raw, y_raw, r = result
            raw_trajectory.append((frame_number, x_raw, y_raw, r))

            # Initialize or update Kalman filter
            if not kalman.initialized:
                kalman.initialize(x_raw, y_raw)
                x_filtered, y_filtered = x_raw, y_raw
            else:
                x_filtered, y_filtered = kalman.update(x_raw, y_raw)

            filtered_trajectory.append((frame_number, x_filtered, y_filtered, r))

            if save_video:
                # Draw raw detection (green)
                ix_raw, iy_raw = int(round(x_raw)), int(round(y_raw))
                cv2.circle(frame, (ix_raw, iy_raw), int(round(r)), (0, 255, 0), 1)
                cv2.circle(frame, (ix_raw, iy_raw), 2, (0, 255, 0), -1)

                # Draw filtered position (red - more prominent)
                ix_filt, iy_filt = int(round(x_filtered)), int(round(y_filtered))
                cv2.circle(frame, (ix_filt, iy_filt), int(round(r)), (0, 0, 255), 2)
                cv2.circle(frame, (ix_filt, iy_filt), 3, (0, 0, 255), -1)

                # Draw filtered trajectory path
                if len(filtered_trajectory) > 1:
                    for i in range(1, len(filtered_trajectory)):
                        pt1 = (int(round(filtered_trajectory[i-1][1])),
                               int(round(filtered_trajectory[i-1][2])))
                        pt2 = (int(round(filtered_trajectory[i][1])),
                               int(round(filtered_trajectory[i][2])))
                        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

                # Add frame info
                cv2.putText(frame, f"Frame: {frame_number}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Filtered: ({x_filtered:.2f}, {y_filtered:.2f})",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                cv2.putText(frame, f"Raw: ({x_raw:.2f}, {y_raw:.2f})",
                           (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

                out.write(frame)

        frame_number += 1

        # Print progress
        if frame_number % 10 == 0:
            print(f"  Processed {frame_number}/{total_frames} frames, detected {len(filtered_trajectory)} positions")

    cap.release()

    if save_video:
        out.release()
        print(f"\nAnnotated video saved to: {output_path}")

    print(f"\nTotal frames processed: {frame_number}")
    print(f"Particle detected in {len(filtered_trajectory)} frames ({100*len(filtered_trajectory)/frame_number:.1f}%)")

    return raw_trajectory, filtered_trajectory


def save_trajectory_to_csv(trajectory, output_path):
    """Save trajectory data to CSV file."""
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

    # Extract trajectory with stabilization
    raw_trajectory, filtered_trajectory = extract_trajectory_stabilized(
        video_path, expected_diameter=48, save_video=True
    )

    if filtered_trajectory:
        print(f"\n=== TRAJECTORY SUMMARY ===")
        print(f"Number of detected positions: {len(filtered_trajectory)}")

        # Calculate statistics for both trajectories
        x_raw = np.array([x for _, x, _, _ in raw_trajectory])
        y_raw = np.array([y for _, y, _, _ in raw_trajectory])
        x_filt = np.array([x for _, x, _, _ in filtered_trajectory])
        y_filt = np.array([y for _, y, _, _ in filtered_trajectory])

        print(f"\nFiltered trajectory:")
        print(f"  X range: {x_filt.min():.3f} to {x_filt.max():.3f} pixels")
        print(f"  Y range: {y_filt.min():.3f} to {y_filt.max():.3f} pixels")
        print(f"  Start: ({x_filt[0]:.3f}, {y_filt[0]:.3f})")
        print(f"  End: ({x_filt[-1]:.3f}, {y_filt[-1]:.3f})")

        # Calculate smoothness
        if len(filtered_trajectory) > 1:
            steps_raw = np.sqrt(np.diff(x_raw)**2 + np.diff(y_raw)**2)
            steps_filt = np.sqrt(np.diff(x_filt)**2 + np.diff(y_filt)**2)

            print(f"\nSmoothness comparison:")
            print(f"  Raw detection: {steps_raw.mean():.3f} ± {steps_raw.std():.3f} pixels/frame")
            print(f"  Kalman filtered: {steps_filt.mean():.3f} ± {steps_filt.std():.3f} pixels/frame")
            print(f"  Smoothness improvement: {steps_raw.std() / steps_filt.std():.2f}x")

        # Save both trajectories
        csv_path_raw = get_csv_path(os.path.basename(video_path).replace('.mp4', '_trajectory_stabilized_raw.csv'))
        csv_path_filtered = get_csv_path(os.path.basename(video_path).replace('.mp4', '_trajectory_stabilized_filtered.csv'))

        save_trajectory_to_csv(raw_trajectory, csv_path_raw)
        save_trajectory_to_csv(filtered_trajectory, csv_path_filtered)

        print(f"\nFirst 10 filtered positions:")
        for i, (frame, x, y, r) in enumerate(filtered_trajectory[:10]):
            print(f"  Frame {frame}: ({x:.3f}, {y:.3f})")
    else:
        print("\nFailed to extract trajectory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
