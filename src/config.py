# config.py

# Path to video
VIDEO_PATH = "../videos/Media1.mp4"

# Hue thresholds
DROPLET_HUE_RANGE = (20, 40)
BACKGROUND_HUE_RANGE = (0, 10)

# Blob detection
MIN_DROPLET_AREA = 300           # minimum blob area in pixels

# Tracker settings
TRACKER_DISTANCE_THRESHOLD = 20  # how much a droplet can move between frames
