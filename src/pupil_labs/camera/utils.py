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
    coords: CT.Points2DLike, n_coords: int
) -> npt.NDArray[np.float64]:
    """Convert/validate python/numpy/structured array of coordinates into unstructured

    Args:
        coords: list of coordinates
        n_coords: number of expected coordinates

    Returns:
        (np.ndarray, bool)
        - The numpy array containing coordinates (always unstructured)
        - A boolean indicating whether the argument was a single coordinate

    Examples:
        >>> to_np_point_array([1, 10], n_coords=2)
        array([[ 1., 10.]])
        >>> to_np_point_array([(1, 10), (2, 20)], n_coords=2)
        array([[ 1., 10.],
               [ 2., 20.]])
        >>> to_np_point_array([(1, 10, 100), (2, 20, 200)], n_coords=2)
        array([[ 1., 10.],
               [ 2., 20.]])
        >>> to_np_point_array([(1, 10, 100), (2, 20, 200)], n_coords=3)
        array([[  1.,  10., 100.],
               [  2.,  20., 200.]])
        >>> to_np_point_array([1, 10], n_coords=2)
        array([[ 1., 10.]])
        >>> to_np_point_array(
        ...     np.array([(1, 10), (2, 20)], dtype=[("x", np.int32), ("y", np.int32)]),
        ...     n_coords=2
        ... )
        array([[ 1., 10.],
               [ 2., 20.]])

    """
    arr = np.array(coords)

    if arr.dtype.names is not None:
        try:
            arr = structured_to_unstructured(arr)
        except Exception as e:
            raise ValueError(f"Failed to convert structured array: {e}") from e

    # is_single_coord = arr.ndim == 1
    arr = np.asarray(arr).astype(np.float64)
    arr = np.squeeze(arr)

    if arr.ndim == 1:
        if len(arr) != n_coords:
            raise ValueError(
                f"Expected {n_coords} coordinate values but got {len(arr)}."
            )
        return arr

    elif arr.ndim == 2:
        if arr.shape[1] != n_coords:
            raise ValueError(
                f"Expected {n_coords} coordinate values but array shape is {arr.shape}."
            )
        return arr

    else:
        raise ValueError(
            f"Invalid coordinate shape after squeezing: {arr.shape}. "
            f"Expected shape ({n_coords},) or (N, {n_coords})."
        )
