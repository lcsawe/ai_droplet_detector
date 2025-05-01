import cv2
import numpy as np

def detect_wraps(frame):
    """
    Detect outer wraps by applying CLAHE + Canny + Hough Circle Transform,
    then filter circles to only those inside the desired detection window.
    """

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (contrast enhancement)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply Canny Edge Detection
    edges = cv2.Canny(enhanced, threshold1=50, threshold2=150)

    # Apply Hough Circle Transform
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=50,
        param2=55,  # <- stricter
        minRadius=20,
        maxRadius=50
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))

        # Now filter circles inside the fixed window (yellow lines area)
        filtered_circles = []
        window_x_start = 410
        window_x_end = 500
        window_y_start = 0
        window_y_end = frame.shape[0]

        for circle in circles[0, :]:
            x, y, radius = circle
            if window_x_start <= x <= window_x_end and window_y_start <= y <= window_y_end:
                filtered_circles.append((x, y, radius))

        if len(filtered_circles) > 0:
            filtered_circles = np.array([filtered_circles])  # match expected shape
        else:
            filtered_circles = None

        return filtered_circles

    return None
