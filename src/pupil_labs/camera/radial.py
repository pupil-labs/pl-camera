from functools import cached_property
from typing import cast

import cv2
import numpy as np

from pupil_labs.camera import custom_types as CT
from pupil_labs.camera.utils import to_np_point_array


class CameraRadial:
    _distortion_coefficients: CT.DistortionCoefficients | None

    def __init__(
        self,
        pixel_width: int,
        pixel_height: int,
        camera_matrix: CT.CameraMatrixLike,
        distortion_coefficients: CT.DistortionCoefficientsLike | None = None,
        use_optimal_camera_matrix: bool = False,
    ):
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients
        self.use_optimal_camera_matrix = use_optimal_camera_matrix

    @property
    def pixel_width(self) -> int:
        return self._pixel_width

    @pixel_width.setter
    def pixel_width(self, value: int) -> None:
        if value <= 0:
            raise ValueError(f"pixel_width must be positive, got {value}")
        self._pixel_width = value

    @property
    def pixel_height(self) -> int:
        return self._pixel_height

    @pixel_height.setter
    def pixel_height(self, value: int) -> None:
        if value <= 0:
            raise ValueError(f"pixel_height must be positive, got {value}")
        self._pixel_height = value

    @property
    def camera_matrix(self) -> CT.CameraMatrix:
        return self._camera_matrix

    @camera_matrix.setter
    def camera_matrix(self, value: CT.CameraMatrixLike) -> None:
        camera_matrix = np.asarray(value, dtype=np.float64)
        if camera_matrix.shape != (3, 3):
            raise ValueError(
                f"camera_matrix should have 3x3 shape, got {'x'.join(map(str, camera_matrix.shape))}"  # noqa: E501
            )
        self._camera_matrix = camera_matrix

    @property
    def distortion_coefficients(self) -> CT.DistortionCoefficients | None:
        return self._distortion_coefficients

    @distortion_coefficients.setter
    def distortion_coefficients(
        self, value: CT.DistortionCoefficientsLike | None
    ) -> None:
        if value is None:
            self._distortion_coefficients = None
        else:
            distortion_coefficients = np.asarray(value, dtype=np.float64)
            if distortion_coefficients.ndim != 1:
                raise ValueError(
                    f"distortion_coefficients should be a 1-dim array: {distortion_coefficients.shape}"  # noqa: E501
                )

            valid_lengths = [4, 5, 8, 12, 14]
            if distortion_coefficients.shape[0] not in valid_lengths:
                raise ValueError(
                    f"distortion_coefficients should be None or have a size of {valid_lengths}"  # noqa: E501
                )
            self._distortion_coefficients = distortion_coefficients

    @property
    def use_optimal_camera_matrix(self):
        return self._use_optimal_camera_matrix

    @use_optimal_camera_matrix.setter
    def use_optimal_camera_matrix(self, value: bool):
        self._use_optimal_camera_matrix = bool(value)

    @cached_property
    def optimal_camera_matrix(self) -> CT.CameraMatrix:
        """The "optimal" camera matrix for undistorting images.

        This method uses OpenCV's `getOptimalNewCameraMatrix` to calculate a new camera
        matrix that maximizes the retirval of sensible pixels in the undistortion
        process, while avoiding "virtual" black pixels stemming from outside the
        captured distorted image.
        """
        optimal_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            cameraMatrix=np.asarray(self.camera_matrix),
            distCoeffs=np.asarray(self.distortion_coefficients),
            imageSize=(self.pixel_width, self.pixel_height),
            newImgSize=(self.pixel_width, self.pixel_height),
            alpha=0,
            centerPrincipalPoint=False,  # TODO(dan): test that gaze on center is correct here  # noqa: E501
        )
        return np.array(optimal_camera_matrix, dtype=np.float64)

    @cached_property
    def _undistort_rectify_map(
        self,
    ) -> tuple[CT.UndistortRectifyMap, CT.UndistortRectifyMap]:
        return self._make_undistort_rectify_map(self.camera_matrix)

    @cached_property
    def _optimal_undistort_rectify_map(
        self,
    ) -> tuple[CT.UndistortRectifyMap, CT.UndistortRectifyMap]:
        return self._make_undistort_rectify_map(self.optimal_camera_matrix)

    def _make_undistort_rectify_map(
        self, camera_matrix: CT.CameraMatrixLike
    ) -> tuple[CT.UndistortRectifyMap, CT.UndistortRectifyMap]:
        return cast(
            tuple[CT.UndistortRectifyMap, CT.UndistortRectifyMap],
            cv2.initUndistortRectifyMap(
                self.camera_matrix,
                self.distortion_coefficients,
                None,
                (camera_matrix),
                (self.pixel_width, self.pixel_height),
                cv2.CV_32FC1,
            ),
        )

    def undistort_image(self, image: CT.Image) -> CT.Image:
        """Return an undistorted image

        This implementation uses cv2.remap with a precomputed map, instead of
        cv2.undistort. This is significantly faster when undistorting multiple images
        because the undistortion maps are computed only once.

        Args:
            image: Image array

        """
        map1, map2 = self._undistort_rectify_map
        remapped: CT.Image = cv2.remap(
            image,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderValue=0,
        )
        return remapped

    def unproject_points(
        self, points_2d: CT.Points2DLike, use_distortion: bool = True
    ) -> CT.Points3D:
        """Unprojects 2D image points to 3D space using the camera's intrinsics.

        Args:
            points_2d: Array-like of 2D point(s) to be unprojected.
            use_distortion: If True, applies distortion correction using the camera's
                distortion coefficients. If False, ignores distortion correction.

        """
        points_2d = to_np_point_array(points_2d, 2)

        if not (
            (points_2d.ndim == 2 and points_2d.shape[1] == 2)
            or (points_2d.ndim == 1 and points_2d.shape[0] == 2)
        ):
            raise ValueError(
                f"points_2d should have shape `(N, 2)` or `(2,)`, got {points_2d.shape}"
            )

        input_dim = points_2d.ndim
        if input_dim == 1:
            points_2d = points_2d[np.newaxis, :]

        camera_matrix = self.camera_matrix
        if self.use_optimal_camera_matrix:
            camera_matrix = self.optimal_camera_matrix

        distortion_coefficients = None
        if use_distortion:
            distortion_coefficients = self.distortion_coefficients

        points_3d = cv2.undistortPoints(
            src=points_2d,
            cameraMatrix=camera_matrix,
            distCoeffs=distortion_coefficients,
        )
        points_3d = cv2.convertPointsToHomogeneous(np.array(points_3d))

        # Remove unnecessary dimension if input was a single point
        if input_dim == 1:
            points_3d = points_3d[0]

        return points_3d.squeeze()

    def project_points(
        self, points_3d: CT.Points3DLike, use_distortion: bool = True
    ) -> CT.Points2D:
        """Projects 3D points onto the 2D image plane using the camera's intrinsics.

        Args:
            points_3d: Array of 3D point(s) to be projected.
            use_distortion: If True, applies distortion using the camera's distortion
                coefficients. If False, ignores distortion.

        """
        points_3d = to_np_point_array(points_3d, 3)

        if not (
            (points_3d.ndim == 2 and points_3d.shape[1] == 3)
            or (points_3d.ndim == 1 and points_3d.shape[0] == 3)
        ):
            raise ValueError(
                f"points_3d should have shape `(N, 3)` or `(3,)`, got {points_3d.shape}"
            )

        input_dim = points_3d.ndim
        if input_dim == 1:
            points_3d = points_3d[np.newaxis, :]

        distortion_coefficients = None
        if use_distortion:
            distortion_coefficients = self.distortion_coefficients

        camera_matrix = self.camera_matrix
        if self.use_optimal_camera_matrix:
            camera_matrix = self.optimal_camera_matrix

        rvec = tvec = np.zeros((1, 1, 3))

        projected, _ = cv2.projectPoints(
            objectPoints=points_3d,
            rvec=rvec,
            tvec=tvec,
            cameraMatrix=camera_matrix,
            distCoeffs=distortion_coefficients,
        )

        return np.array(projected).astype(np.float64).squeeze()

    def undistort_points(self, points_2d: CT.Points2DLike) -> CT.Points2D:
        """Undistorts 2D image points using the camera's intrinsics.

        Args:
            points_2d: Array-like of 2D point(s) to be undistorted.

        """
        points_2d = to_np_point_array(points_2d, 2)

        if not (
            (points_2d.ndim == 2 and points_2d.shape[1] == 2)
            or (points_2d.ndim == 1 and points_2d.shape[0] == 2)
        ):
            raise ValueError(
                f"points_2d should have shape `(N, 2)` or `(2,)`, got {points_2d.shape}"
            )

        input_dim = points_2d.ndim
        if input_dim == 1:
            points_2d = points_2d[np.newaxis, :]

        camera_matrix = self.camera_matrix
        if self.use_optimal_camera_matrix:
            camera_matrix = self.optimal_camera_matrix

        undistorted_2d = cv2.undistortPoints(
            src=points_2d,
            cameraMatrix=camera_matrix,
            distCoeffs=self.distortion_coefficients,
            R=None,
            P=camera_matrix,
        )
        return undistorted_2d.squeeze()
