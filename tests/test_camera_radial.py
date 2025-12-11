import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from pupil_labs.camera import CameraRadial
from pupil_labs.camera import custom_types as CT


@pytest.fixture
def camera_radial():
    image_width = 1088
    image_height = 1080
    camera_matrix = [
        [766.3037717610379, 0.0, 559.7158729463123],
        [0.0, 765.4514012936911, 537.2187571096966],
        [0.0, 0.0, 1.0],
    ]
    distortion_coefficients = [
        -0.12571787111434657,
        0.1009174721106796,
        0.0004064475713640723,
        -0.0001776950802199194,
        0.017309286074375808,
        0.20449589859897552,
        0.008640898256976831,
        0.06428433887310138,
    ]
    return CameraRadial(
        image_width, image_height, camera_matrix, distortion_coefficients
    )


@pytest.mark.parametrize(
    "points",
    [
        np.array([100, 200]),  # unstructured ints
        np.array([100, 200], dtype=np.int32),  # unstructured ints
        [100, 200],  # list
        (100, 200),  # tuple
    ],
)
def test_unproject_point(camera_radial: CameraRadial, points):
    expected = np.array([-0.75240, -0.55311, 1.0])
    undistorted = camera_radial.unproject_points(points)
    assert_almost_equal(undistorted, np.asarray(expected), decimal=3)


@pytest.mark.parametrize(
    "points",
    [
        np.array([(100, 200), (800, 600)]),  # unstructured ints
        np.array(
            [(100, 200), (800, 600)],
            dtype=[("x", np.int32), ("y", np.int32)],
        ),  # structured ints
        np.array([(100, 200), (800, 600)], dtype=np.int32),  # unstructured ints
        np.array(
            [(100.0, 200.0), (800.0, 600.0)],
            dtype=[("x", np.float32), ("y", np.float32)],
        ),  # structured floats
        np.array(
            [(100.0, 200.0), (800.0, 600.0)],
            dtype=[("x", np.float32), ("y", np.float32)],
        ),  # structured floats
        [(100, 200), (800, 600)],  # list of tuples
        ([100, 200], [800, 600]),  # tuple of lists
        [[100, 200], [800, 600]],  # list of lists
        ((100, 200), (800, 600)),  # tuple of tuples
    ],
)
def test_unproject_points(camera_radial: CameraRadial, points):
    expected = np.array([[-0.75240, -0.55311, 1.0], [0.32508, 0.08498, 1.0]])
    undistorted = camera_radial.unproject_points(points)
    assert_almost_equal(undistorted, np.asarray(expected), decimal=3)


def test_unproject_point_without_distortion(camera_radial: CameraRadial):
    points = np.array([100.3349, 200.2458])
    undistorted = camera_radial.unproject_points(points, use_distortion=False)
    expected = np.array([-0.59947633, -0.44022776, 1.0])
    assert_almost_equal(undistorted, expected, decimal=4)


def test_unproject_points_without_distortion(camera_radial: CameraRadial):
    points = np.array([(100.3349, 200.2458), (799.9932, 599.9996)])
    undistorted = camera_radial.unproject_points(points, use_distortion=False)
    expected = np.array([
        [-0.59947633, -0.44022776, 1.0],
        [0.31355363, 0.08201807, 1.0],
    ])
    assert_almost_equal(undistorted, expected, decimal=4)


@pytest.mark.parametrize(
    "points",
    [
        np.array([-0.75170, -0.55260, 1.0]),  # unstructured
        [-0.75170, -0.55260, 1.0],  # list
        (-0.75170, -0.55260, 1.0),  # tuple
    ],
)
def test_project_point(camera_radial: CameraRadial, points: CT.Points2DLike):
    expected = np.array([100.3349, 200.2458])
    assert_almost_equal(
        camera_radial.project_points(points),
        expected,
        decimal=4,
    )


@pytest.mark.parametrize(
    "points",
    [
        np.array([(-0.75170, -0.55260, 1.0), (0.32508, 0.08498, 1.0)]),  # unstructured
        np.array(
            [(-0.75170, -0.55260, 1.0), (0.32508, 0.08498, 1.0)],
            dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32)],
        ),  # structured
        [(-0.75170, -0.55260, 1.0), (0.32508, 0.08498, 1.0)],  # list of tuples
        ([-0.75170, -0.55260, 1.0], [0.32508, 0.08498, 1.0]),  # tuple of lists
        [[-0.75170, -0.55260, 1.0], [0.32508, 0.08498, 1.0]],  # list of lists
        ((-0.75170, -0.55260, 1.0), (0.32508, 0.08498, 1.0)),  # tuple of tuples
    ],
)
def test_project_points(camera_radial: CameraRadial, points: CT.Points2DLike):
    expected = np.array([(100.3349, 200.2458), (799.9932, 599.9996)])
    assert_almost_equal(
        camera_radial.project_points(points),
        expected,
        decimal=4,
    )


def test_project_point_without_distortion(camera_radial: CameraRadial):
    points = np.array([-0.59947633, -0.44022776, 1.0])
    projected = camera_radial.project_points(points, use_distortion=False)
    expected = np.array([100.3349, 200.2458])
    assert_almost_equal(projected, expected, decimal=4)


def test_project_points_without_distortion(camera_radial: CameraRadial):
    points = np.array([[-0.59947633, -0.44022776, 1.0], [0.31355363, 0.08201807, 1.0]])
    projected = camera_radial.project_points(points, use_distortion=False)
    expected = np.array([(100.3349, 200.2458), (799.9932, 599.9996)])
    assert_almost_equal(projected, expected, decimal=4)


def test_undistort_point(camera_radial: CameraRadial):
    points = np.array([100.3349, 200.2458])
    undistorted = camera_radial.undistort_points(points)
    expected = np.array([-16.314667, 114.23032316])
    assert_almost_equal(undistorted, expected, decimal=4)


def test_undistort_points(camera_radial: CameraRadial):
    points = np.array([(100.3349, 200.2458), (799.9932, 599.9996)])
    undistorted = camera_radial.undistort_points(points)
    expected = np.array([[-16.314667, 114.23032316], [808.82587932, 602.26686841]])
    assert_almost_equal(undistorted, expected, decimal=4)


@pytest.mark.parametrize("width", [-1, 0])
def test_invalid_width(camera_radial: CameraRadial, width: int):
    with pytest.raises(ValueError):
        CameraRadial(width, 1000, [[1, 2, 3], [1, 2, 3], [1, 2, 3]], [1, 2, 3, 4])
    with pytest.raises(ValueError):
        camera_radial.pixel_width = width


@pytest.mark.parametrize("height", [-1, 0])
def test_invalid_height(camera_radial: CameraRadial, height: int):
    with pytest.raises(ValueError):
        CameraRadial(1000, height, [[1, 2, 3], [1, 2, 3], [1, 2, 3]], [1, 2, 3, 4])
    with pytest.raises(ValueError):
        camera_radial.pixel_height = height


@pytest.mark.parametrize(
    "camera_matrix",
    [
        None,
        [],
        [[1, 2, 3], [1, 2, 3]],
        [[1, 2], [1, 2], [1, 2]],
    ],
)
def test_invalid_camera_matrix(
    camera_radial: CameraRadial, camera_matrix: CT.CameraMatrixLike
):
    with pytest.raises(ValueError):
        CameraRadial(1000, 1000, camera_matrix, [1, 2, 3, 4])
    with pytest.raises(ValueError):
        camera_radial.camera_matrix = camera_matrix


@pytest.mark.parametrize(
    "distortion_coefficients",
    [
        [[1, 2, 3, 4]],
        [],
        [1],
        [1, 2],
        [1, 2, 3],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    ],
)
def test_invalid_distortion_coefficients(
    camera_radial: CameraRadial, distortion_coefficients: CT.DistortionCoefficientsLike
):
    with pytest.raises(ValueError):
        CameraRadial(
            1000, 1000, [[1, 2, 3], [1, 2, 3], [1, 2, 3]], distortion_coefficients
        )
    with pytest.raises(ValueError):
        camera_radial.distortion_coefficients = distortion_coefficients


@pytest.mark.parametrize(
    "distortion_coefficients",
    [
        None,
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    ],
)
def test_valid_distortion_coefficients(
    camera_radial: CameraRadial, distortion_coefficients: CT.DistortionCoefficientsLike
):
    CameraRadial(1000, 1000, [[1, 2, 3], [1, 2, 3], [1, 2, 3]], distortion_coefficients)
    camera_radial.distortion_coefficients = distortion_coefficients


def test_unprojection_and_reprojection(camera_radial: CameraRadial):
    original = [(0, 0), (0, 1080), (1088, 0), (1088, 1080)]

    unprojected = camera_radial.unproject_points(original)
    reprojected = camera_radial.project_points(unprojected)
    assert_almost_equal(original, reprojected, decimal=4)
