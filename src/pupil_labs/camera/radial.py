from functools import cached_property

import cv2
import numpy as np
import numpy.typing as npt
from numpy.lib.recfunctions import structured_to_unstructured

from pupil_labs.camera import custom_types as CT
from pupil_labs.camera import opencv_funcs


class CameraRadial:
    _distortion_coefficients: CT.DistortionCoefficients | None

    def __init__(
        self,
        pixel_width: int,
        pixel_height: int,
        camera_matrix: CT.CameraMatrix,
        distortion_coefficients: CT.DistortionCoefficients | None = None,
    ):
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients

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
        if value is not None:
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
        else:
            self._distortion_coefficients = None

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
    def _optimal_undistort_rectify_map(
        self,
    ) -> tuple[CT.UndistortRectifyMap, CT.UndistortRectifyMap]:
        return opencv_funcs.undistort_rectify_map(
            self.camera_matrix,
            self.pixel_width,
            self.pixel_height,
            self.distortion_coefficients,
            self.optimal_camera_matrix,
        )

    @cached_property
    def _undistort_rectify_map(
        self,
    ) -> tuple[CT.UndistortRectifyMap, CT.UndistortRectifyMap]:
        return opencv_funcs.undistort_rectify_map(
            self.camera_matrix,
            self.pixel_width,
            self.pixel_height,
            self.distortion_coefficients,
            self.camera_matrix,
        )

    def undistort_image(
        self, image: CT.Image, use_optimal_camera_matrix: bool = False
    ) -> CT.Image:
        if use_optimal_camera_matrix:
            map1, map2 = self._optimal_undistort_rectify_map
        else:
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
        self,
        points_2d: CT.Points2DLike,
        use_distortion: bool = True,
        use_optimal_camera_matrix: bool = False,
    ) -> CT.Points3D:
        """Unprojects 2D image points to 3D space using the camera's intrinsics.

        Args:
            points_2d: Array-like of 2D point(s) to be unprojected.
            use_distortion: If True, applies distortion correction using the camera's
                distortion coefficients. If False, ignores distortion correction.
            use_optimal_camera_matrix: If True, uses the optimal camera matrix for
                unprojection.

        """
        points_2d = self._to_np_array(points_2d)

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

        if use_optimal_camera_matrix:
            camera_matrix = self.optimal_camera_matrix
        else:
            camera_matrix = self.camera_matrix

        if use_distortion:
            distortion_coefficients = self.distortion_coefficients
        else:
            distortion_coefficients = None

        points_3d = opencv_funcs.undistort_points(
            points_2d, camera_matrix, distortion_coefficients, None
        )

        # Remove unnecessary dimension if input was a single point
        if input_dim == 1:
            points_3d = points_3d[0]

        return points_3d

    def project_points(
        self,
        points_3d: CT.Points3DLike,
        use_distortion: bool = True,
        use_optimal_camera_matrix: bool = False,
    ) -> CT.Points2D:
        """Projects 3D points onto the 2D image plane using the camera's intrinsics.

        Args:
            points_3d: Array of 3D point(s) to be projected.
            use_distortion: If True, applies distortion using the camera's distortion
                coefficients. If False, ignores distortion.
            use_optimal_camera_matrix: If True, uses the optimal camera matrix for
                projection instead of the regular camera matrix.

        """
        points_3d = self._to_np_array(points_3d)

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

        if use_distortion:
            distortion_coefficients = self.distortion_coefficients
        else:
            distortion_coefficients = None

        if use_optimal_camera_matrix:
            camera_matrix = self.optimal_camera_matrix
        else:
            camera_matrix = self.camera_matrix

        points_2d = opencv_funcs.project_points(
            points_3d, camera_matrix, distortion_coefficients
        ).reshape(-1, 2)

        if input_dim == 1:
            points_2d = points_2d[0]

        return points_2d

    def undistort_points(
        self,
        points_2d: CT.Points2DLike,
        use_optimal_camera_matrix: bool = False,
    ) -> CT.Points2D:
        """Undistorts 2D image points using the camera's intrinsics.

        Args:
            points_2d: Array-like of 2D point(s) to be undistorted.
            use_optimal_camera_matrix: If True, uses the *regular* camera matrix for
                unprojection, and the optimal camera matrix for reprojection. If False,
                uses the regular camera matrix for both.

        """
        points_2d = self._to_np_array(points_2d)

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

        if use_optimal_camera_matrix:
            camera_matrix = self.optimal_camera_matrix
        else:
            camera_matrix = self.camera_matrix

        points_undistorted = opencv_funcs.undistort_points(
            points_2d, camera_matrix, self.distortion_coefficients, camera_matrix
        ).reshape(-1, 3)

        points_undistorted = (
            points_undistorted[:, :2] / points_undistorted[:, 2, np.newaxis]
        )

        if input_dim == 1:
            points_undistorted = points_undistorted[0]

        return points_undistorted

    @staticmethod
    def _to_np_array(
        points: CT.Points2DLike | CT.Points3DLike,
    ) -> npt.NDArray[np.float64]:
        """Convert a list, array, or record array of points into an unstructured array."""  # noqa: E501
        if not len(points):
            raise ValueError("points array is empty")

        if hasattr(points, "dtype") and points.dtype.names is not None:  # type: ignore
            np_points = structured_to_unstructured(points, dtype=np.float64)  # type: ignore
        else:
            np_points = np.asarray(points, dtype=np.float64)

        return np_points
