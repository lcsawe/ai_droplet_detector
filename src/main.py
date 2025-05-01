import cv2
from video_loader import load_video, read_frames
from color_recognition import convert_to_hsv, create_droplet_mask
from droplet_detector import find_droplets, calculate_centers
from wrap_detector import detect_wraps
from utils import DropletTracker, OuterDropletStateTracker
import config
# The video on slide 49 Lecture 1 was loaded to initiate the identification
def main():
    video_path = config.VIDEO_PATH
    cap = load_video(video_path)

    if cap is None:
        return
    # Colour range detection and isolation of background
    droplet_hue_range = config.DROPLET_HUE_RANGE
    background_hue_range = config.BACKGROUND_HUE_RANGE

    # Track inner droplet movement between frames
    tracker = DropletTracker(distance_threshold=config.TRACKER_DISTANCE_THRESHOLD)

    # track droplets passing through the window and if successfully formed outer wrap
    outer_state_tracker = OuterDropletStateTracker(window_x_start=410, window_x_end=500, max_missed_frames=5)
    # Iteration
    for idx, frame in enumerate(read_frames(cap)):
        # Convert to hsv
        hsv_frame = convert_to_hsv(frame)

    # part 1:Inner droplet detection
        # Colour Thresholding, contours and centre computation
        mask = create_droplet_mask(hsv_frame, droplet_hue_range, background_hue_range)
        contours = find_droplets(mask, min_area=config.MIN_DROPLET_AREA)
        centers = calculate_centers(contours)
        # Part 4: Mass/Blob Detection

        # Define detection window
        window_x_start = 410
        window_x_end = 500
        window_y_start = 0
        window_y_end = frame.shape[0]

        # Filter inner centers inside window
        filtered_centers = []
        for (x, y) in centers:
            if window_x_start <= x <= window_x_end and window_y_start <= y <= window_y_end:
                filtered_centers.append((x, y))

        # Update tracker with centers
        tracker.update(filtered_centers)

        inner_droplet_info = []  # list of (center, radius) tuples

        for cnt in contours:
            if cv2.contourArea(cnt) >= config.MIN_DROPLET_AREA:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)
                inner_droplet_info.append((center, radius))

        for center, radius in inner_droplet_info:
            # Optional: scale radius slightly bigger for better visibility
            display_radius = int(radius * 1.2)

            cv2.circle(frame, center, radius=display_radius, color=(0, 0, 255), thickness=2)
            cv2.circle(frame, center, 2, (255, 0, 0), 3)  # small blue center dot

        # part 2: Outer wrap detection
        circles = detect_wraps(frame)

        # Draw outer wraps
        if circles is not None:
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]
                cv2.circle(frame, center, radius, (0, 255, 0), 2)
                cv2.circle(frame, center, 2, (0, 0, 255), 3)



        circles = detect_wraps(frame)


        # Draw detection window
        cv2.rectangle(
            frame,
            (window_x_start, window_y_start),
            (window_x_end, window_y_end),
            color=(0, 255, 255),  # Yellow
            thickness=2
        )
        # Part 3: Count Successfully Formed Droplets
        successful_droplet_count, total_droplet_count, unsuccessful_droplet_count = outer_state_tracker.update(circles,                                                                                                                                                                                                 filtered_centers)

        cv2.putText(frame, f"Successful: {successful_droplet_count}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Total: {total_droplet_count}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Unsuccessful: {unsuccessful_droplet_count}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        # Show frames
        cv2.imshow("Original Frame with Detections", frame)
        cv2.imshow("Droplet Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("=" * 60)
    print("Final Summary:")
    print(f"Total Droplets Detected: {outer_state_tracker.total_droplets}")
    print(f"Successfully Formed Droplets: {outer_state_tracker.successful_droplets}")
    print(f"Unsuccessfully Formed Droplets: {outer_state_tracker.unsuccessful_droplets}")
    print("=" * 60)


if __name__ == "__main__":
    main()
