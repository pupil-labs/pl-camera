import numpy as np
import numpy.typing as npt
from numpy.lib.recfunctions import structured_to_unstructured

import pupil_labs.camera.custom_types as CT


def apply_distortion_model(
    point: tuple[float, float], dist_coeffs: CT.DistortionCoefficients
) -> npt.NDArray[np.float64]:
    x, y = point
    r = np.linalg.norm([x, y])

    k1, k2, p1, p2, k3, k4, k5, k6 = dist_coeffs

    scale = 1 + k1 * r**2 + k2 * r**4 + k3 * r**6
    scale /= 1 + k4 * r**2 + k5 * r**4 + k6 * r**6

    x_dist = scale * x + 2 * p1 * x * y + p2 * (r**2 + 2 * x**2)
    y_dist = scale * y + p1 * (r**2 + 2 * y**2) + 2 * p2 * x * y

    return np.asarray([x_dist, y_dist])


def to_np_point_array(
    points: CT.Points2DLike | CT.Points3DLike,
    n_coords: int = 2,
) -> npt.NDArray[np.float64]:
    """Convert a python, numpy or structured array of points into unstructured

    Examples:
        >>> to_np_point_array([1, 10])
        array([[ 1., 10.]])
        >>> to_np_point_array([(1, 10), (2, 20)])
        array([[ 1., 10.],
               [ 2., 20.]])
        >>> to_np_point_array([(1, 10, 100), (2, 20, 200)])
        array([[ 1., 10.],
               [ 2., 20.]])
        >>> to_np_point_array([(1, 10, 100), (2, 20, 200)], n_coords=3)
        array([[  1.,  10., 100.],
               [  2.,  20., 200.]])
        >>> to_np_point_array([1, 10])
        array([[ 1., 10.]])
        >>> to_np_point_array(
        ...     np.array([(1, 10), (2, 20)],
        ...     dtype=[("x", np.int32), ("y", np.int32)])
        ... )
        array([[ 1., 10.],
               [ 2., 20.]])

    """
    if not len(points):
        return np.array([], dtype=np.float64).reshape((-1, n_coords))

    if hasattr(points, "dtype") and points.dtype.names is not None:
        if n_coords > len(points.dtype.names):
            raise ValueError(
                f"can not convert {len(points.dtype.names)}D points to {n_coords}D"
            )
        np_points = structured_to_unstructured(points, dtype=np.float64)[:, :n_coords]  # type: ignore
    else:
        np_points = np.asarray(points, dtype=np.float64)
        data_n_coords = (
            np_points.shape[0] if np_points.ndim == 1 else np_points.shape[1]
        )
        if n_coords > data_n_coords:
            raise ValueError(f"can not convert {data_n_coords}D points to {n_coords}D")
        if np_points.ndim == 1:
            np_points = np_points.reshape((-1, len(np_points)))
        np_points = np_points[:, :n_coords]

    return np_points
