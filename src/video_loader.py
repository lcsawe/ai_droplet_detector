import cv2

def load_video(video_path):
    """
    Opens a video file for reading frames.
    Args:
        video_path (str): Path to the video file.
    Returns:
        cv2.VideoCapture object if successful, else None.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    return cap

def read_frames(cap):
    """
    Generator function that yields frames from a VideoCapture object.
    Args:
        cap: OpenCV VideoCapture object.
    Yields:
        frame (ndarray): Next frame from the video.
    """
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()
