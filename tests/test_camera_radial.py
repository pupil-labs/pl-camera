from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from pupil_labs.camera import Camera
from pupil_labs.camera import custom_types as CT
from pupil_labs.camera.assets import chessboard_image

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


def show(img: np.ndarray):
    """Quickly show an image in pytest debugger"""
    import PIL.Image

    PIL.Image.fromarray(img[:, :, ::-1]).show()


def make_point_variants(points: list[tuple[float | int, ...]]):
    single = points[0]
    dtype = [("x", np.float32), ("y", np.float32), ("z", np.float32)][: len(single)]
    return [
        # unstructured numpy
        np.array([single]),
        np.array(points),
        # unsqueezed numpy
        np.array([[single]]),
        np.array([list(points)]),
        # structured numpy
        np.array([single], dtype=dtype),
        np.array(points, dtype=dtype),
        # list of tuples
        [single],
        list(points),
        # tuple of lists
        (single,),
        tuple([list(p) for p in points]),
        # list of lists
        [[single]],
        [list(p) for p in points],
        # unsqueezed list of lists
        [[single]],
        [[list(p) for p in points]],
        # tuple of tuples
        (single,),
        points,
    ]


@pytest.fixture  # (scope="session")
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
    width = 1600
    height = 1200
    camera = Camera(1600, 1200, CAMERA_MATRIX, distortion_coefficients)
    camera.unproject_points(
        (200.0, 200.0),
        use_optimal_camera_matrix=use_optimal_camera_matrix,
        use_distortion=use_distortion,
    )
    camera.undistort_image(
        np.zeros((height, width), dtype=np.uint8),
        use_optimal_camera_matrix=use_optimal_camera_matrix,
    )
    camera.distort_image(
        np.zeros((height, width), dtype=np.uint8),
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
    camera.distort_points(
        (200.0, 200.0),
        use_optimal_camera_matrix=use_optimal_camera_matrix,
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


@pytest.mark.parametrize("points", make_point_variants([(100, 200), (800, 600)]))
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
    "points", make_point_variants([(-0.75170, -0.55260, 1.0), (0.32508, 0.08498, 1.0)])
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
    point = np.array([10, 10])
    undistorted = camera_radial.undistort_points(point, use_optimal_camera_matrix=True)
    expected = np.array([-247.65881137, -338.23329978])
    assert_almost_equal(undistorted, expected, decimal=4)


@pytest.mark.parametrize(
    "points", make_point_variants([(10, 10), (50, 50), (100, 100), (600, 600)])
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


def test_unprojection_and_reprojection_edge_cases(camera_radial: Camera):
    original = [
        [0, 0],  # top-left
        [camera_radial.pixel_width, 0],  # top-right
        [0, camera_radial.pixel_height],  # bottom-left
        [camera_radial.pixel_width, camera_radial.pixel_height],  # bottom-right
    ]

    # Due to distortion, edge points may not map perfectly, but should be close
    # This test primarily ensures no crashes occur at boundaries
    unprojected = camera_radial.unproject_points(original)
    reprojected = camera_radial.project_points(unprojected)
    assert (np.absolute(original - reprojected)).max() < 50


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


def test_distort_point(camera_radial: Camera):
    """Test distorting a single point."""
    point = np.array([400, 400])
    distorted = camera_radial.distort_points(point)
    expected = np.array([431.1344, 416.1969])
    assert_almost_equal(distorted, expected, decimal=4)


def test_distort_point_optimal(camera_radial: Camera):
    """Test distorting a single point with optimal camera matrix."""
    point = np.array([400, 400])
    distorted = camera_radial.distort_points(point, use_optimal_camera_matrix=True)
    expected = np.array([456.8362, 429.7182])
    assert_almost_equal(distorted, expected, decimal=4)


@pytest.mark.parametrize(
    "points",
    [
        # unstructured
        np.array([(800, 800)]),
        np.array([
            (800, 800),
            (400, 400),
            (100, 100),
            (10, 10),
        ]),
        # unsqueezed unstructured
        np.array([[(800, 800)]]),
        np.array([
            [
                (800, 800),
                (400, 400),
                (100, 100),
                (10, 10),
            ]
        ]),
        # structured
        np.array(
            [(800, 800)],
            dtype=[("x", np.float32), ("y", np.float32)],
        ),
        np.array(
            [
                (800, 800),
                (400, 400),
                (100, 100),
                (10, 10),
            ],
            dtype=[("x", np.float32), ("y", np.float32)],
        ),
        # list of tuples
        [(800, 800)],
        [
            (800, 800),
            (400, 400),
            (100, 100),
            (10, 10),
        ],
        # tuple of lists
        ([800, 800],),
        (
            [800, 800],
            [400, 400],
            [100, 100],
            [10, 10],
        ),
        # list of lists
        [[800, 800]],
        [
            [800, 800],
            [400, 400],
            [100, 100],
            [10, 10],
        ],
        # unsqueezed list of lists
        [[[800, 800]]],
        [
            [
                [800, 800],
                [400, 400],
                [100, 100],
                [10, 10],
            ]
        ],
        # tuple of tuples
        ((800, 800),),
        (
            (800, 800),
            (400, 400),
            (100, 100),
            (10, 10),
        ),
    ],
)
def test_distort_points(camera_radial: Camera, points):
    """Test distorting multiple points with various input formats."""
    distorted = camera_radial.distort_points(points)
    expected = np.array([
        [800.1897, 797.6494],
        [431.1344, 416.1969],
        [251.5298, 209.4722],
        [213.7963, 163.6328],
    ])

    np_points = np.array(points)
    n_expected_points = len(np_points[0]) if np_points.ndim > 2 else len(np_points)
    assert_almost_equal(distorted, expected[:n_expected_points], decimal=4)


def test_distort_many_points(camera_radial: Camera):
    """Test distorting a large number of points."""
    points = [(x, y) for x in range(-800, 800, 10) for y in range(-600, 600, 10)]
    output = camera_radial.distort_points(points)
    assert len(output) == len(points)


@pytest.mark.parametrize("delta", [0, 100, 300])
def test_distort_undistort_roundtrip(camera_radial: Camera, delta: int):
    """Test distort and undistort are inverse operations."""
    width, height = camera_radial.pixel_width, camera_radial.pixel_height
    original = np.array([
        [width // 2, height // 2],
        [width // 2 + delta, height // 2 + delta],
        [width // 2 - delta, height // 2 - delta],
        [width // 2 + delta, height // 2 - delta],
        [width // 2 - delta, height // 2 + delta],
    ])

    undistorted = camera_radial.undistort_points(original)
    redistorted = camera_radial.distort_points(undistorted)
    assert_almost_equal(redistorted, original, decimal=2)


@pytest.mark.parametrize("delta", [0, 100, 300])
def test_undistort_distort_roundtrip(camera_radial: Camera, delta: int):
    """Test undistort and distort for points near center are inverse operations"""
    width, height = camera_radial.pixel_width, camera_radial.pixel_height
    original = np.array([
        [width // 2, height // 2],
        [width // 2 + delta, height // 2 + delta],
        [width // 2 - delta, height // 2 - delta],
        [width // 2 + delta, height // 2 - delta],
        [width // 2 - delta, height // 2 + delta],
    ])

    distorted = camera_radial.distort_points(original)
    reundistorted = camera_radial.undistort_points(distorted)
    assert_almost_equal(reundistorted, original, decimal=2)


@pytest.mark.parametrize(
    "points",
    [
        [10, 10, 1],
        [[10, 10, 1]],
        [[10, 10, 1], [10, 10, 1]],
    ],
)
def test_invalid_distort_points_shapes(camera_radial: Camera, points):
    """Test that distort_points raises ValueError for invalid input shapes."""
    with pytest.raises(ValueError, match="2 coordinate"):
        camera_radial.distort_points(points)


def test_distort_points_edge_cases(camera_radial: Camera):
    """Test distorting points at image edges."""
    original = np.array(
        [
            [0, 0],  # top-left
            [camera_radial.pixel_width, 0],  # top-right
            [0, camera_radial.pixel_height],  # bottom-left
            [camera_radial.pixel_width, camera_radial.pixel_height],  # bottom-right
        ],
        dtype=np.float32,
    )

    undistorted = camera_radial.undistort_points(original)
    redistorted = camera_radial.distort_points(undistorted)

    # Due to distortion, edge points may not map perfectly, but should be close
    # This test primarily ensures no crashes occur at boundaries
    assert redistorted.shape == original.shape
    assert (np.absolute(original - redistorted)).max() < 50


@pytest.fixture  # (scope="session")
def distorted_image(camera_radial: Camera, test_data_path: Path):
    return chessboard_image


@pytest.fixture
def undistorted_image(camera_radial: Camera, distorted_image: np.ndarray):
    return cv2.undistort(
        distorted_image,
        camera_radial.camera_matrix,
        camera_radial.distortion_coefficients,
    )


@pytest.fixture
def undistorted_image_optimal(
    camera_radial: Camera, distorted_image, undistorted_image
):
    return cv2.undistort(
        distorted_image,
        camera_radial.camera_matrix,
        camera_radial.distortion_coefficients,
        newCameraMatrix=camera_radial.optimal_camera_matrix,
    )


def test_distort_image(camera_radial: Camera, distorted_image, undistorted_image):
    distorted = camera_radial.distort_image(undistorted_image)
    assert distorted.shape == undistorted_image.shape
    assert distorted.mean() == 83.77018263888888


def test_distort_image_optimal(
    camera_radial: Camera, distorted_image, undistorted_image_optimal
):
    distorted = camera_radial.distort_image(
        undistorted_image_optimal, use_optimal_camera_matrix=True
    )
    assert distorted.shape == distorted_image.shape
    assert distorted.mean() == 117.12533385416667


def test_undistort_image(camera_radial: Camera, distorted_image, undistorted_image):
    undistorted = camera_radial.undistort_image(distorted_image)
    assert undistorted.shape == distorted_image.shape
    assert undistorted.mean() == 120.6645625


def test_undistort_image_optimal(
    camera_radial: Camera, distorted_image, undistorted_image
):
    undistorted = camera_radial.undistort_image(
        distorted_image, use_optimal_camera_matrix=True
    )
    assert undistorted.shape == undistorted.shape
    assert undistorted.mean() == 132.47595399305555
