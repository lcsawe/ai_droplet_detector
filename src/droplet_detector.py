
import cv2
import numpy as np

def find_droplets(mask, min_area=300):
    """
    Finds blobs (droplets) in the binary mask, applies cleaning and area filtering.
    Args:
        mask (ndarray): Binary mask image.
        min_area (int): Minimum area to consider a blob as a droplet.
    Returns:
        contours (list): List of valid contours.
    """
    # Morphological closing to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter small contours
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            valid_contours.append(cnt)

    return valid_contours

def calculate_centers(contours):
    """
    Calculates center of mass for each contour.
    Args:
        contours (list): List of contours.
    Returns:
        centers (list of tuples): List of (x, y) center coordinates.
    """
    centers = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))
    return centers
