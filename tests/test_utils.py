import numpy as np
from numpy.testing import assert_almost_equal

from pupil_labs.camera.utils import get_perspective_transform, perspective_transform


class TestGetPerspectiveTransform:
    """Smoke tests for get_perspective_transform."""

    def test_identity_transform(self):
        """Same source and destination points should yield identity-like transform."""
        points = [(0, 0), (100, 0), (100, 100), (0, 100)]
        transform = get_perspective_transform(points, points)
        assert transform.shape == (3, 3)
        assert_almost_equal(transform, np.eye(3), decimal=5)

    def test_translation_transform(self):
        """Shifted destination points should yield a translation transform."""
        src = [(0, 0), (100, 0), (100, 100), (0, 100)]
        dst = [(10, 20), (110, 20), (110, 120), (10, 120)]
        transform = get_perspective_transform(src, dst)
        assert transform.shape == (3, 3)

    def test_scale_transform(self):
        """Scaled destination points should yield a scaling transform."""
        src = [(0, 0), (100, 0), (100, 100), (0, 100)]
        dst = [(0, 0), (200, 0), (200, 200), (0, 200)]
        transform = get_perspective_transform(src, dst)
        assert transform.shape == (3, 3)

    def test_accepts_numpy_arrays(self):
        """Function should accept numpy arrays as input."""
        src = np.array([(0, 0), (100, 0), (100, 100), (0, 100)], dtype=np.float32)
        dst = np.array([(10, 10), (110, 10), (110, 110), (10, 110)], dtype=np.float32)
        transform = get_perspective_transform(src, dst)
        assert transform.shape == (3, 3)

    def test_accepts_lists(self):
        """Function should accept lists as input."""
        src = [[0, 0], [100, 0], [100, 100], [0, 100]]
        dst = [[10, 10], [110, 10], [110, 110], [10, 110]]
        transform = get_perspective_transform(src, dst)
        assert transform.shape == (3, 3)


class TestPerspectiveTransform:
    """Smoke tests for perspective_transform."""

    def test_identity_transform_single_point(self):
        """Identity matrix should not change the point."""
        point = (50, 50)
        identity = np.eye(3, dtype=np.float64)
        result = perspective_transform(point, identity)
        assert result.shape == (1, 2)
        assert_almost_equal(result[0], point, decimal=5)

    def test_identity_transform_multiple_points(self):
        """Identity matrix should not change multiple points."""
        points = [(0, 0), (100, 0), (100, 100), (0, 100)]
        identity = np.eye(3, dtype=np.float64)
        result = perspective_transform(points, identity)
        assert result.shape == (4, 2)
        assert_almost_equal(result, points, decimal=5)

    def test_translation_transform(self):
        """Translation matrix should shift points."""
        points = [(0, 0), (100, 100)]
        translation = np.array(
            [
                [1, 0, 10],
                [0, 1, 20],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        result = perspective_transform(points, translation)
        expected = [(10, 20), (110, 120)]
        assert_almost_equal(result, expected, decimal=5)

    def test_scale_transform(self):
        """Scaling matrix should scale points."""
        points = [(10, 10), (50, 50)]
        scale = np.array(
            [
                [2, 0, 0],
                [0, 2, 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        result = perspective_transform(points, scale)
        expected = [(20, 20), (100, 100)]
        assert_almost_equal(result, expected, decimal=5)

    def test_accepts_numpy_array_points(self):
        """Function should accept numpy array points."""
        points = np.array([(10, 20), (30, 40)], dtype=np.float32)
        identity = np.eye(3, dtype=np.float64)
        result = perspective_transform(points, identity)
        assert result.shape == (2, 2)
        assert_almost_equal(result, points, decimal=5)

    def test_accepts_list_points(self):
        """Function should accept a list of points."""
        points = [[10, 20], [30, 40]]
        identity = np.eye(3, dtype=np.float64)
        result = perspective_transform(points, identity)
        assert result.shape == (2, 2)
        assert_almost_equal(result, points, decimal=5)


class TestPerspectiveRoundtrip:
    """Test that get_perspective_transform and perspective_transform work together."""

    def test_roundtrip_simple(self):
        """Transform computed from correspondences should map src to dst."""
        src = [(0, 0), (100, 0), (100, 100), (0, 100)]
        dst = [(10, 10), (90, 15), (85, 95), (5, 90)]
        transform = get_perspective_transform(src, dst)
        result = perspective_transform(src, transform)
        assert_almost_equal(result, dst, decimal=4)

    def test_roundtrip_with_inverse(self):
        """Inverse transform should map dst back to src."""
        src = [(0, 0), (100, 0), (100, 100), (0, 100)]
        dst = [(10, 10), (110, 10), (110, 110), (10, 110)]
        transform = get_perspective_transform(src, dst)
        inverse = np.linalg.inv(transform)
        result = perspective_transform(dst, inverse)
        assert_almost_equal(result, src, decimal=4)
