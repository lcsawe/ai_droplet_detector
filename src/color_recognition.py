import cv2
import numpy as np


def convert_to_hsv(frame):
    """
    Converts a BGR frame to HSV color space.
    Args:
        frame (ndarray): BGR image.
    Returns:
        hsv (ndarray): HSV image.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return hsv


def create_droplet_mask(hsv_frame, droplet_hue_range, background_hue_range):
    """
    Creates a binary mask separating droplets from background based on hue.

    Args:
        hsv_frame (ndarray): HSV image.
        droplet_hue_range (tuple): (lower, upper) bounds for droplet hue.
        background_hue_range (tuple): (lower, upper) bounds for background hue.

    Returns:
        mask (ndarray): Binary mask (droplet = 255, background = 0)
    """
    # Extract hue channel
    hue = hsv_frame[:, :, 0]

    # Create background mask
    background_mask = cv2.inRange(hue, background_hue_range[0], background_hue_range[1])

    # Create droplet mask
    droplet_mask = cv2.inRange(hue, droplet_hue_range[0], droplet_hue_range[1])

    # Invert background mask (background becomes 0, droplet becomes 255)
    mask = droplet_mask

    return mask
