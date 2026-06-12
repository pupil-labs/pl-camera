from collections.abc import Sequence
from typing import TypeAlias

from numpy import float64, floating, integer, uint8
from numpy.typing import NDArray

CameraMatrixLike: TypeAlias = NDArray[floating] | Sequence[Sequence[float]]
DistortionCoefficientsLike: TypeAlias = NDArray[floating] | Sequence[float]
AffineMatrixLike: TypeAlias = NDArray[floating] | Sequence[Sequence[float]]
CameraMatrix: TypeAlias = NDArray[float64]
DistortionCoefficients: TypeAlias = NDArray[float64]
Image: TypeAlias = NDArray[uint8]
Points2D: TypeAlias = NDArray[float64]
Points3D: TypeAlias = NDArray[float64]
NumberLike: TypeAlias = int | float | integer | floating
Point2DLike: TypeAlias = tuple[NumberLike, NumberLike]
Points2DLike: TypeAlias = (
    NDArray[floating] | Sequence[Sequence[NumberLike]] | list[Point2DLike] | Point2DLike
)
Point3DLike: TypeAlias = tuple[NumberLike, NumberLike, NumberLike]
Points3DLike: TypeAlias = (
    NDArray[floating] | Sequence[Sequence[NumberLike]] | list[Point3DLike] | Point3DLike
)
RectifyMap: TypeAlias = tuple[NDArray[float64], NDArray[float64]]
