"""Trajectory extraction and Kalman filtering for particle tracking."""

import cv2
import numpy as np
from .detection import detect_circle_robust


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


def extract_trajectory(video_path, expected_diameter=48, use_kalman=True, save_video=False,
                      output_video_path=None, progress_callback=None):
    """
    Extract trajectory from video with optional Kalman filtering.

    Args:
        video_path: Path to the input video file
        expected_diameter: Expected diameter of the microparticle in pixels
        use_kalman: If True, apply Kalman filtering for smoothing
        save_video: If True, save an annotated video
        output_video_path: Path for annotated video (required if save_video=True)
        progress_callback: Optional callback function(frame_num, total_frames)

    Returns:
        tuple: (raw_trajectory, filtered_trajectory) if use_kalman=True
               or (trajectory, trajectory) if use_kalman=False

        Each trajectory is a list of tuples: (frame, x, y, radius)
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    raw_trajectory = []
    filtered_trajectory = []
    frame_number = 0

    # Initialize Kalman tracker if requested
    kalman = KalmanTracker() if use_kalman else None

    # Prepare video writer if saving annotated video
    if save_video:
        if output_video_path is None:
            raise ValueError("output_video_path required when save_video=True")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    else:
        out = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Get prediction from Kalman filter if initialized
        prediction = None
        if use_kalman and kalman.initialized:
            prediction = kalman.predict()

        # Detect circle
        result = detect_circle_robust(frame, expected_diameter, prediction)

        if result is not None:
            x_raw, y_raw, r = result
            raw_trajectory.append((frame_number, x_raw, y_raw, r))

            if use_kalman:
                # Initialize or update Kalman filter
                if not kalman.initialized:
                    kalman.initialize(x_raw, y_raw)
                    x_filtered, y_filtered = x_raw, y_raw
                else:
                    x_filtered, y_filtered = kalman.update(x_raw, y_raw)

                filtered_trajectory.append((frame_number, x_filtered, y_filtered, r))
            else:
                filtered_trajectory.append((frame_number, x_raw, y_raw, r))

            if save_video and out is not None:
                # Draw detection
                ix, iy = int(round(x_raw)), int(round(y_raw))
                cv2.circle(frame, (ix, iy), int(round(r)), (0, 255, 0), 2)
                cv2.circle(frame, (ix, iy), 3, (0, 255, 0), -1)

                if use_kalman:
                    # Draw filtered position
                    ix_filt, iy_filt = int(round(x_filtered)), int(round(y_filtered))
                    cv2.circle(frame, (ix_filt, iy_filt), int(round(r)), (0, 0, 255), 2)
                    cv2.circle(frame, (ix_filt, iy_filt), 3, (0, 0, 255), -1)

                # Add frame info
                cv2.putText(frame, f"Frame: {frame_number}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                out.write(frame)

        frame_number += 1

        # Call progress callback
        if progress_callback:
            progress_callback(frame_number, total_frames)

    cap.release()
    if out is not None:
        out.release()

    return raw_trajectory, filtered_trajectory
