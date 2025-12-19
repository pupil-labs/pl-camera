"""Top-level entry-point for the pupil_labs.camera package"""

import contextlib
from importlib.metadata import PackageNotFoundError, version

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("pupil_labs.camera")

from .radial import Camera
from .utils import get_perspective_transform, perspective_transform

__all__ = [
    "Camera",
    "__version__",
    "get_perspective_transform",
    "perspective_transform",
]
