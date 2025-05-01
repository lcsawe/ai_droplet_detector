import numpy as np

class DropletTracker:
    def __init__(self, distance_threshold=20):
        self.tracked_centers = []
        self.distance_threshold = distance_threshold

    def update(self, detected_centers):
        for center in detected_centers:
            if not self.is_already_tracked(center):
                self.tracked_centers.append(center)

    def is_already_tracked(self, center):
        for tracked_center in self.tracked_centers:
            distance = np.linalg.norm(np.array(center) - np.array(tracked_center))
            if distance < self.distance_threshold:
                return True
        return False

    def get_total_count(self):
        return len(self.tracked_centers)

class OuterDropletStateTracker:
    def __init__(self, window_x_start=410, window_x_end=500, max_missed_frames=5):
        self.window_x_start = window_x_start
        self.window_x_end = window_x_end
        self.successful_droplets = 0
        self.unsuccessful_droplets = 0
        self.total_droplets = 0
        self.waiting_for_next = False
        self.missed_frames = 0
        self.max_missed_frames = max_missed_frames

    def update(self, outer_circles, inner_centers):
        """Update tracker based on outer wraps + inner centers."""

        found_valid_circle = False
        found_inner_match = False

        if outer_circles is not None:
            for circle in outer_circles[0, :]:
                x, y, radius = circle

                # Strict: Only process circles FULLY inside window
                if (x - radius >= self.window_x_start) and (x + radius <= self.window_x_end):
                    found_valid_circle = True

                    for inner_center in inner_centers:
                        distance = np.linalg.norm(np.array((x, y)) - np.array(inner_center))
                        if distance <= radius:
                            found_inner_match = True
                            break
                    break  # Only consider the first fully inside one

        if not self.waiting_for_next:
            if found_valid_circle:
                # New droplet detected -> Count immediately
                self.total_droplets += 1
                if found_inner_match:
                    self.successful_droplets += 1
                else:
                    self.unsuccessful_droplets += 1

                self.waiting_for_next = True  # Now wait before counting again
                self.missed_frames = 0

        else:
            if found_valid_circle:
                # Still seeing the same droplet -> Reset missed frames
                self.missed_frames = 0
            else:
                self.missed_frames += 1
                if self.missed_frames >= self.max_missed_frames:
                    # Enough empty frames passed -> allow next droplet counting
                    self.waiting_for_next = False
                    self.missed_frames = 0

        return self.successful_droplets, self.total_droplets, self.unsuccessful_droplets


def count_successful_droplets(inner_centers, circles, tolerance=5):
    """
    Counts successfully formed droplets:
    Each successful droplet must have an inner droplet center inside a detected outer wrap (circle).
    """
    if circles is None or len(circles) == 0:
        return 0

    successful_count = 0

    # Check each outer circle
    for circle in circles[0, :]:
        circle_x, circle_y, radius = circle
        matched_inner = False

        # Check if any inner droplet center falls inside this outer wrap
        for center in inner_centers:
            x, y = center
            distance = np.sqrt((x - circle_x)**2 + (y - circle_y)**2)

            if distance <= (radius - tolerance):  # Allow a small margin
                matched_inner = True
                break  # One inner droplet inside is enough

        if matched_inner:
            successful_count += 1

    return successful_count

