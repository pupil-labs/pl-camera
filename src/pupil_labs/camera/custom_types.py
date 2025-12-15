from collections.abc import Sequence

from numpy import float64, floating, integer, uint8
from numpy.typing import NDArray

CameraMatrixLike = NDArray[floating] | Sequence[Sequence[float]]
DistortionCoefficientsLike = (NDArray[floating] | Sequence[float]) | None
CameraMatrix = NDArray[float64]
DistortionCoefficients = NDArray[float64]
Image = NDArray[uint8]
Points2D = NDArray[float64]
Points3D = NDArray[float64]
NumberLike = int | float | integer | floating
Point2DLike = tuple[NumberLike, NumberLike]
Points2DLike = (
    NDArray[floating] | Sequence[Sequence[NumberLike]] | list[Point2DLike] | Point2DLike
)
Point3DLike = tuple[NumberLike, NumberLike, NumberLike]
Points3DLike = (
    NDArray[floating] | Sequence[Sequence[NumberLike]] | list[Point3DLike] | Point3DLike
)
UndistortRectifyMap = NDArray[float64]
