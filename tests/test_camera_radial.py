import cv2
import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from pupil_labs.camera import Camera
from pupil_labs.camera import custom_types as CT

CAMERA_MATRIX = [
    [891.61897098, 0.0, 816.30726443],
    [0.0, 890.94104777, 614.49661859],
    [0.0, 0.0, 1.0],
]
DISTORTION_COEFFICIENTS = [
    -0.13057592,
    0.10888688,
    0.00038934,
    -0.00046976,
    -0.00072779,
    0.17010936,
    0.05234352,
    0.02383326,
]


@pytest.fixture
def camera_radial():
    image_width = 1600
    image_height = 1200
    camera_matrix = CAMERA_MATRIX
    distortion_coefficients = DISTORTION_COEFFICIENTS
    return Camera(image_width, image_height, camera_matrix, distortion_coefficients)


@pytest.mark.parametrize(
    "distortion_coefficients",
    [
        None,
        DISTORTION_COEFFICIENTS,
    ],
)
@pytest.mark.parametrize("use_distortion", [True, False])
@pytest.mark.parametrize("use_optimal_camera_matrix", [True, False])
def test_various_configurations(
    distortion_coefficients, use_optimal_camera_matrix, use_distortion
):
    camera = Camera(1600, 1200, CAMERA_MATRIX, distortion_coefficients)
    camera.unproject_points(
        (200.0, 200.0),
        use_optimal_camera_matrix=use_optimal_camera_matrix,
        use_distortion=use_distortion,
    )
    camera.undistort_image(
        np.zeros((1600, 1200), dtype=np.uint8),
        use_optimal_camera_matrix=use_optimal_camera_matrix,
    )
    camera.undistort_points(
        (200.0, 200.0),
        use_optimal_camera_matrix=use_optimal_camera_matrix,
    )
    camera.project_points(
        (0.5, 0.5, 1),
        use_optimal_camera_matrix=use_optimal_camera_matrix,
        use_distortion=use_distortion,
    )


@pytest.mark.parametrize(
    "point",
    [
        np.array([100, 200]),  # unstructured ints
        np.array([100, 200], dtype=np.int32),  # unstructured ints
        [100, 200],  # list
        (100, 200),  # tuple
        (100, 200),  # tuple
    ],
)
def test_unproject_point(camera_radial: Camera, point):
    expected = np.array([-1.15573178, -0.67095352, 1.0])
    unprojected = camera_radial.unproject_points(point)
    assert_almost_equal(unprojected, np.asarray(expected), decimal=3)


def test_unproject_point_optimal(camera_radial: Camera):
    point = np.array([100.3349, 200.2458])
    expected = np.array([-3.37628209, -1.66810606, 1.0])
    unprojected = camera_radial.unproject_points(point, use_optimal_camera_matrix=True)
    assert_almost_equal(unprojected, np.asarray(expected), decimal=3)


@pytest.mark.parametrize(
    "points",
    [
        # unstructured ints
        np.array([(100, 200)]),
        np.array([(100, 200), (800, 600)]),
        # unstructured np ints
        np.array([(100, 200)], dtype=np.int32),
        np.array([(100, 200), (800, 600)], dtype=np.int32),
        # structured floats
        np.array(
            [(100.0, 200.0)],
            dtype=[("x", np.float32), ("y", np.float32)],
        ),
        np.array(
            [(100, 200), (800, 600)],
            dtype=[("x", np.int32), ("y", np.int32)],
        ),
        # structured floats
        np.array(
            [(100.0, 200.0)],
            dtype=[("x", np.float32), ("y", np.float32)],
        ),
        np.array(
            [(100.0, 200.0), (800.0, 600.0)],
            dtype=[("x", np.float32), ("y", np.float32)],
        ),
        # list of tuples
        [(100, 200)],
        [(100, 200), (800, 600)],
        # tuple of lists
        ([100, 200],),
        ([100, 200], [800, 600]),
        # tuple of lists
        ([100, 200],),
        ([100, 200], [800, 600]),
        # list of lists
        [[100, 200]],
        [[100, 200], [800, 600]],
        # tuple of tuples
        ((100, 200),),
        ((100, 200), (800, 600)),
        # unsqueezed list of lists
        [[[100, 200]]],
        [[[100, 200], [800, 600]]],
    ],
)
def test_unproject_points(camera_radial: Camera, points):
    expected = np.array([
        [-1.15573178, -0.67095352, 1.0],
        [-0.01829243, -0.01627422, 1.0],
    ])
    unprojected = camera_radial.unproject_points(points)

    np_points = np.array(points)
    n_expected_points = len(np_points[0]) if np_points.ndim > 2 else len(np_points)

    assert_almost_equal(
        unprojected, np.asarray(expected[:n_expected_points]), decimal=3
    )


def test_unproject_many_points(camera_radial: Camera):
    points = [(x, y) for x in range(0, 1600, 10) for y in range(0, 1200, 10)]
    output = camera_radial.unproject_points(points)
    assert len(output) == len(points)


def test_unproject_point_without_distortion(camera_radial: Camera):
    point = np.array([100.3349, 200.2458])
    unprojected = camera_radial.unproject_points(point, use_distortion=False)
    expected = np.array([-0.80300261, -0.46495873, 1.0])
    assert_almost_equal(unprojected, expected, decimal=4)


def test_unproject_point_without_distortion_optimal(camera_radial: Camera):
    point = np.array([100.3349, 200.2458])
    unprojected = camera_radial.unproject_points(
        point, use_distortion=False, use_optimal_camera_matrix=True
    )
    expected = np.array([-1.14129205, -0.55826439, 1.0])
    assert_almost_equal(unprojected, expected, decimal=4)


def test_unproject_points_without_distortion(camera_radial: Camera):
    points = np.array([(100.3349, 200.2458), (799.9932, 599.9996)])
    unprojected = camera_radial.unproject_points(points, use_distortion=False)
    expected = np.array([
        [-0.80300261, -0.46495873, 1.0],
        [-0.01829713, -0.01627158, 1.0],
    ])
    assert_almost_equal(unprojected, expected, decimal=4)


@pytest.mark.parametrize(
    "point",
    [
        np.array([-0.75170, -0.55260, 1.0]),  # unstructured
        [-0.75170, -0.55260, 1.0],  # list
        (-0.75170, -0.55260, 1.0),  # tuple
    ],
)
def test_project_point(camera_radial: Camera, point: CT.Points3DLike):
    expected = np.array([276.45064393, 218.50131053])
    projected = camera_radial.project_points(point)
    assert_almost_equal(projected, expected, decimal=4)


def test_project_point_from_cv2_homogenous(camera_radial: Camera):
    cv2_output = cv2.convertPointsToHomogeneous(
        cv2.undistortPoints(
            np.array([(600.0, 600)], dtype=np.float32),
            camera_radial.camera_matrix,
            camera_radial.distortion_coefficients,
        )
    )
    expected = np.array([[600.0, 600]])
    projected = camera_radial.project_points(cv2_output)  # type: ignore
    assert_almost_equal(projected, expected, decimal=4)


def test_project_point_optimal(camera_radial: Camera):
    point = np.array([-0.59947633, -0.44022776, 1.0])
    undistorted = camera_radial.project_points(point, use_optimal_camera_matrix=True)
    expected = np.array([497.45664004, 335.03763505])
    assert_almost_equal(undistorted, expected, decimal=4)


@pytest.mark.parametrize(
    "points",
    [
        # unstructured
        np.array([(-0.75170, -0.55260, 1.0)]),
        np.array([(-0.75170, -0.55260, 1.0), (0.32508, 0.08498, 1.0)]),
        # unsqueezed unstructured
        np.array([[[-0.75170, -0.55260, 1.0]]]),
        np.array([[[-0.75170, -0.55260, 1.0], [0.32508, 0.08498, 1.0]]]),
        # structured
        np.array(
            [(-0.75170, -0.55260, 1.0)],
            dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32)],
        ),
        np.array(
            [(-0.75170, -0.55260, 1.0), (0.32508, 0.08498, 1.0)],
            dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32)],
        ),
        # list of tuples
        [(-0.75170, -0.55260, 1.0)],
        [(-0.75170, -0.55260, 1.0), (0.32508, 0.08498, 1.0)],
        # tuple of lists
        ([-0.75170, -0.55260, 1.0],),
        ([-0.75170, -0.55260, 1.0], [0.32508, 0.08498, 1.0]),
        # list of lists
        [[-0.75170, -0.55260, 1.0]],
        [[-0.75170, -0.55260, 1.0], [0.32508, 0.08498, 1.0]],
        # unsqueezed list of lists
        [[[-0.75170, -0.55260, 1.0]]],
        [[[-0.75170, -0.55260, 1.0], [0.32508, 0.08498, 1.0]]],
        # tuple of tuples
        ((-0.75170, -0.55260, 1.0),),
        ((-0.75170, -0.55260, 1.0), (0.32508, 0.08498, 1.0)),
    ],
)
def test_project_points(camera_radial: Camera, points: CT.Points3DLike):
    expected = np.array([[276.45064393, 218.50131053], [1096.58550912, 687.76068265]])
    projected = camera_radial.project_points(points)

    np_points = np.array(points)
    n_expected_points = len(np_points[0]) if np_points.ndim > 2 else len(np_points)

    assert_almost_equal(projected, expected[:n_expected_points], decimal=4)


def test_project_many_points(camera_radial: Camera):
    points = [(x, y, 1) for x in np.arange(0, 1, 0.01) for y in np.arange(0, 1, 0.01)]
    output = camera_radial.project_points(points)
    assert len(output) == len(points)


def test_project_point_without_distortion(camera_radial: Camera):
    point = np.array([-0.59947633, -0.44022776, 1.0])
    projected = camera_radial.project_points(point, use_distortion=False)
    expected = np.array([281.80279595, 222.27963684])
    assert_almost_equal(projected, expected, decimal=4)


def test_project_point_without_distortion_optimal(camera_radial: Camera):
    point = np.array([-0.59947633, -0.44022776, 1.0])
    undistorted = camera_radial.project_points(
        point, use_distortion=False, use_optimal_camera_matrix=True
    )
    expected = np.array([445.23836289, 289.28855095])
    assert_almost_equal(undistorted, expected, decimal=4)


def test_undistort_point(camera_radial: Camera):
    point = np.array([10, 10])
    undistorted = camera_radial.undistort_points(point)
    expected = np.array([-688.71195284, -518.85317532])
    assert_almost_equal(undistorted, expected, decimal=4)


def test_undistort_point_optimal(camera_radial: Camera):
    point = np.array([500, 500])
    undistorted = camera_radial.undistort_points(point, use_optimal_camera_matrix=True)
    expected = np.array([466.53389311, 487.41463334])
    assert_almost_equal(undistorted, expected, decimal=4)


@pytest.mark.parametrize(
    "points",
    [
        # unstructured
        np.array([(10, 10)]),
        np.array([(10, 10), (50, 50), (100, 100), (600, 600)]),
        # unsqueezed unstructured
        np.array([[(10, 10)]]),
        np.array([[(10, 10), (50, 50), (100, 100), (600, 600)]]),
        # structured
        np.array(
            [(10, 10)],
            dtype=[("x", np.float32), ("y", np.float32)],
        ),
        np.array(
            [(10, 10), (50, 50), (100, 100), (600, 600)],
            dtype=[("x", np.float32), ("y", np.float32)],
        ),
        # list of tuples
        [(10, 10)],
        [(10, 10), (50, 50), (100, 100), (600, 600)],
        # tuple of lists
        ([10, 10],),
        ([10, 10], [50, 50], [100, 100], [600, 600]),
        # list of lists
        [[10, 10]],
        [[10, 10], [50, 50], [100, 100], [600, 600]],
        # unsqueezed list of lists
        [[[10, 10]]],
        [[[10, 10], [50, 50], [100, 100], [600, 600]]],
        # tuple of tuples
        ((10, 10),),
        ((10, 10), (50, 50), (100, 100), (600, 600)),
    ],
)
def test_undistort_points(camera_radial: Camera, points):
    undistorted = camera_radial.undistort_points(points)
    expected = np.array([
        [-688.71195284, -518.85317532],
        [-473.65891857, -339.15262622],
        [-280.38559214, -175.43871054],
        [596.10485541, 599.71556346],
    ])

    np_points = np.array(points)
    n_expected_points = len(np_points[0]) if np_points.ndim > 2 else len(np_points)
    assert_almost_equal(undistorted, expected[:n_expected_points], decimal=4)


def test_undistort_many_points(camera_radial: Camera):
    points = [(x, y) for x in range(0, 1600, 10) for y in range(0, 1200, 10)]
    output = camera_radial.undistort_points(points)
    assert len(output) == len(points)


@pytest.mark.parametrize("width", [-1, 0])
def test_invalid_width(camera_radial: Camera, width: int):
    with pytest.raises(ValueError):
        Camera(width, 1000, [[1, 2, 3], [1, 2, 3], [1, 2, 3]], [1, 2, 3, 4])
    with pytest.raises(ValueError):
        camera_radial.pixel_width = width


@pytest.mark.parametrize("height", [-1, 0])
def test_invalid_height(camera_radial: Camera, height: int):
    with pytest.raises(ValueError):
        Camera(1000, height, [[1, 2, 3], [1, 2, 3], [1, 2, 3]], [1, 2, 3, 4])
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
    camera_radial: Camera, camera_matrix: CT.CameraMatrixLike
):
    with pytest.raises(ValueError):
        Camera(1000, 1000, camera_matrix, [1, 2, 3, 4])
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
    camera_radial: Camera, distortion_coefficients: CT.DistortionCoefficientsLike
):
    with pytest.raises(ValueError):
        Camera(1000, 1000, [[1, 2, 3], [1, 2, 3], [1, 2, 3]], distortion_coefficients)
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
    camera_radial: Camera, distortion_coefficients: CT.DistortionCoefficientsLike
):
    Camera(1000, 1000, [[1, 2, 3], [1, 2, 3], [1, 2, 3]], distortion_coefficients)
    camera_radial.distortion_coefficients = distortion_coefficients


# NOTE(dan): this test is disabled because unproject/project is not completely accurate
# using opencv funcs, when a suitable perfomant replacement is found it can be enabled
# def test_unprojection_and_reprojection(camera_radial: CameraRadial):
#     original = [
#         (0, 0),
#         (0, camera_radial.pixel_height),
#         (camera_radial.pixel_width, 0),
#         (camera_radial.pixel_width, camera_radial.pixel_height),
#     ]

#     unprojected = camera_radial.unproject_points(original)
#     reprojected = camera_radial.project_points(unprojected)
#     assert_almost_equal(original, reprojected, decimal=4)


@pytest.mark.parametrize(
    "points",
    [
        [10, 10, 1],
        [[10, 10, 1]],
        [[10, 10, 1], [10, 10, 1]],
    ],
)
def test_invalid_unproject_points_shapes(camera_radial: Camera, points):
    with pytest.raises(ValueError, match="2 coordinate"):
        camera_radial.unproject_points(points)


@pytest.mark.parametrize(
    "points",
    [
        [10, 10],
        [[10, 10]],
        [[10, 10], [10, 10]],
    ],
)
def test_invalid_project_points_shapes(camera_radial: Camera, points):
    with pytest.raises(ValueError, match="3 coordinate"):
        camera_radial.project_points(points)
