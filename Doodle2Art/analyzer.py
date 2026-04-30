# analyzer.py
import cv2
import numpy as np


def analyze_doodle(image_path, save_path="processed_doodle.png"):
    """
    Preprocess user-drawn or photographed pencil doodles.
    Output: clean black lines on white background (512x512).
    """

    # Load and normalize size
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (512, 512))

    # Adaptive threshold to remove paper texture & lighting
    thresh = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        4
    )

    # Remove small noise and notebook lines
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Strengthen strokes
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)

    # Smooth jagged edges slightly
    cleaned = cv2.GaussianBlur(cleaned, (3, 3), 0)

    # White background, black lines
    final = cv2.bitwise_not(cleaned)

    cv2.imwrite(save_path, final)
    return save_path


if __name__ == "__main__":
    analyze_doodle("input_doodle.png")