# Pupil Labs Camera

[![ci](https://github.com/pupil-labs/pl-camera/actions/workflows/main.yml/badge.svg)](https://github.com/pupil-labs/pl-camera/actions/workflows/main.yml)
[![documentation](https://img.shields.io/badge/docs-mkdocs-708FCC.svg?style=flat)](https://pupil-labs.github.io/pl-camera/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre_commit-black?logo=pre-commit&logoColor=FAB041)](https://github.com/pre-commit/pre-commit)
[![pypi version](https://img.shields.io/pypi/v/pupil-labs-camera.svg)](https://pypi.org/project/pupil-labs-camera/)
[![python version](https://img.shields.io/pypi/pyversions/pupil-labs-camera)](https://pypi.org/project/pupil-labs-camera/)

This repo contains functionality around the usage of camera intrinsics for undistorting data, projecting and unprojecting points.

It is mostly a wrapper around OpenCV's functionality, providing type hints, input validation, a more intuitive interface, and some changes to improve computational performance.

## Installation

```
pip install pupil-labs-camera
```

or

```bash
pip install -e git+https://github.com/pupil-labs/pl-camera.git
```

## Quick Start

The following code demonstrates how to use the library in the context of a Neon recording to undistort an image and 2D gaze points.

```python
import cv2

from pupil_labs import neon_recording as nr
from pupil_labs.camera import Camera

recording = nr.NeonRecording(
    "/path/to/recording"
)
camera = Camera(
    pixel_width=1600,
    pixel_height=1200,
    camera_matrix=recording.calibration.scene_camera_matrix,
    distortion_coefficients=recording.calibration.scene_distortion_coefficients,
)

data = zip(recording.scene, recording.gaze.sample(recording.scene.time))

for scene_frame, gaze_sample in data:
    distorted_image = scene_frame.bgr
    undistorted_image = camera.undistort_image(distorted_image)

    distorted_gaze = gaze_sample.point
    undistorted_gaze = camera.undistort_points(distorted_gaze)

    cv2.circle(
        distorted_image,
        tuple(map(int, distorted_gaze)),
        radius=25,
        color=(0, 0, 255),
        thickness=5,
    )

    cv2.circle(
        undistorted_image,
        tuple(map(int, undistorted_gaze)),
        radius=25,
        color=(0, 255, 0),
        thickness=5,
    )

    cv2.imshow("Distorted Image", distorted_image)
    cv2.imshow("Undistorted Image", undistorted_image)
    cv2.waitKey(30)
```
