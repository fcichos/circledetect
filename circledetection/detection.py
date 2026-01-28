"""Circle detection functions for microparticle tracking."""

import cv2
import numpy as np


def detect_circle_robust(frame, expected_diameter=48, prediction=None):
    """
    Detect a circular microparticle with robust handling of nanoparticles.

    Args:
        frame: Input frame (BGR image)
        expected_diameter: Expected diameter of the microparticle in pixels
        prediction: Optional (x, y) prediction for search region

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
