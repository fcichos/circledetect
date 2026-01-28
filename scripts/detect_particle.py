#!/usr/bin/env python3
"""
Detect microparticle in darkfield microscopy image and find its center coordinates.
The microparticle has a diameter of approximately 48 pixels.
"""

import cv2
import numpy as np
import sys
import os


def detect_microparticle(image_path, expected_diameter=48, visualize=False):
    """
    Detect a circular microparticle in a darkfield microscopy image.

    Args:
        image_path: Path to the input image
        expected_diameter: Expected diameter of the microparticle in pixels
        visualize: If True, display the result with detected circle

    Returns:
        tuple: (center_x, center_y, radius) or None if not found
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Hough Circle Transform to detect circles
    # Parameters tuned for detecting a ~48 pixel diameter circle
    min_radius = int(expected_diameter * 0.4)  # 19 pixels
    max_radius = int(expected_diameter * 0.7)  # 33 pixels

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,  # Minimum distance between circle centers
        param1=50,    # Canny edge detection threshold
        param2=30,    # Accumulator threshold for circle detection
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        # If multiple circles detected, take the one with strongest response (first one)
        if len(circles) > 0:
            x, y, r = circles[0]
            print(f"Microparticle detected:")
            print(f"  Center coordinates: ({x}, {y})")
            print(f"  Radius: {r} pixels")
            print(f"  Diameter: {2*r} pixels")

            # Create visualization
            output = img.copy()

            # Draw the circle outline
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)

            # Draw the center point
            cv2.circle(output, (x, y), 5, (0, 0, 255), -1)

            # Add text with coordinates
            text = f"Center: ({x}, {y})"
            cv2.putText(output, text, (x - 50, y - r - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Save the result
            basename = os.path.basename(image_path).replace('.png', '_detected.png')
            output_path = os.path.join('results', 'plots', basename)
            os.makedirs('results/plots', exist_ok=True)
            cv2.imwrite(output_path, output)
            print(f"Result saved to: {output_path}")

            if visualize:
                # Display the result
                cv2.imshow("Detected Microparticle", output)
                print("\nPress any key to close the window...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            return (x, y, r)

    print("No microparticle detected. Try adjusting the detection parameters.")
    return None


def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "data/Ms_4_4_AuNP_moving_n0_laser_2_40_001_001.png"

    result = detect_microparticle(image_path, expected_diameter=48, visualize=False)

    if result:
        x, y, r = result
        print(f"\nSummary: Microparticle found at pixel coordinates ({x}, {y})")
    else:
        print("\nFailed to detect microparticle.")
        sys.exit(1)


if __name__ == "__main__":
    main()
