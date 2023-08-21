import enum
from typing import Optional, Sequence

from numpy import float32, float64, uint8
from numpy.typing import NDArray

CameraMatrixLike = NDArray[float32 | float64] | Sequence[Sequence[float]]
DistortionCoefficientsLike = Optional[NDArray[float32 | float64] | Sequence[float]]
CameraMatrix = NDArray[float64]
DistortionCoefficients = NDArray[float64]
Image = NDArray[uint8]
Points2D = NDArray[float64]
Points3D = NDArray[float64]
Points2DLike = (
    NDArray[float32 | float64] | list[tuple] | Sequence[Sequence[float | int]]
)
Points3DLike = (
    NDArray[float32 | float64] | list[tuple] | Sequence[Sequence[float | int]]
)
UndistortRectifyMap = NDArray[float64]
