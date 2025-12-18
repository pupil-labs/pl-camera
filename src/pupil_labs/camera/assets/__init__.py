from importlib.resources import files

import cv2

chessboard_image = cv2.imread(
    str(files("pupil_labs.camera") / "assets" / "chessboard.jpeg")
)
