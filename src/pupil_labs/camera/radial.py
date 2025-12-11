from functools import cached_property

import cv2
import numpy as np

from pupil_labs.camera import custom_types as CT
from pupil_labs.camera import opencv_funcs


class CameraRadial:
    _distortion_coefficients: CT.DistortionCoefficients | None

    def __init__(
        self,
        pixel_width: int,
        pixel_height: int,
        camera_matrix: CT.CameraMatrixLike,
        distortion_coefficients: CT.DistortionCoefficientsLike | None = None,
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

    def undistort_points(
        self,
        points_2d: CT.Points2DLike,
        use_distortion: bool = True,
        reproject_to_image: bool = False,
        use_optimal_camera_matrix: bool = False,
    ) -> CT.Points3D:
        """Undistorts 2D image points using the camera's intrinsics.

        Args:
            points_2d: Array-like of 2D points to be undistorted.
            use_distortion: If True, applies distortion correction using the camera's
                distortion coefficients. If False, ignores distortion correction.
            reproject_to_image: If True, reprojects undistorted points back to the image
                plane using the camera matrix.
            use_optimal_camera_matrix: If True, uses the optimal camera matrix for
                reprojection.

        """
        if use_optimal_camera_matrix and not reproject_to_image:
            raise ValueError(
                "use_optimal_camera_matrix can only be True when reproject_to_image is True"  # noqa: E501
            )

        if use_distortion:
            distortion_coefficients = self.distortion_coefficients
        else:
            distortion_coefficients = None

        if reproject_to_image:
            new_camera_matrix = self.camera_matrix
            if use_optimal_camera_matrix:
                new_camera_matrix = self.optimal_camera_matrix
        else:
            new_camera_matrix = None

        return opencv_funcs.undistort_points(
            points_2d, self.camera_matrix, distortion_coefficients, new_camera_matrix
        )

    def project_points(
        self,
        points_3d: CT.Points3DLike,
        use_distortion: bool = True,
        use_optimal_camera_matrix: bool = False,
    ) -> CT.Points2D:
        """Projects 3D points onto the 2D image plane using the camera's intrinsics.

        Args:
            points_3d: Array of 3D points to be projected.
            use_distortion: If True, applies distortion using the camera's distortion
                coefficients. If False, ignores distortion.
            use_optimal_camera_matrix: If True, uses the optimal camera matrix for
                projection instead of the regular camera matrix.

        """
        if use_distortion:
            distortion_coefficients = self.distortion_coefficients
        else:
            distortion_coefficients = None

        if use_optimal_camera_matrix:
            camera_matrix = self.optimal_camera_matrix
        else:
            camera_matrix = self.camera_matrix

        return opencv_funcs.project_points(
            points_3d, camera_matrix, distortion_coefficients
        )
