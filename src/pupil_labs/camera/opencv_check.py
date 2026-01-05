try:
    import cv2  # noqa: F401
except ImportError as e:
    raise ImportError(
        "OpenCV (cv2) is required but not installed.\n"
        "Install one of:\n"
        "  - opencv-python\n"
        "  - opencv-python-headless\n"
        "  - system OpenCV (e.g. apt install python3-opencv)"
    ) from e
