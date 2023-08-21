from typing import Annotated, Optional

import cv2
import numpy as np
from cv2 import remap
from pydantic import AfterValidator, ConfigDict, Field
from pydantic.dataclasses import dataclass

from pupil_labs.camera import opencv_funcs
from pupil_labs.camera import types as CT
from pupil_labs.camera.base import CameraBase


class CameraRadialBase(CameraBase):
    pass


def check_camera_matrix(camera_matrix: CT.CameraMatrixLike) -> CT.CameraMatrix:
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    assert camera_matrix.shape == (
        3,
        3,
    ), f"camera_matrix should have 3x3 shape, got {'x'.join(map(str, camera_matrix.shape))}"
    return camera_matrix


@staticmethod
def check_distortion_coefficients(
    distortion_coefficients: CT.DistortionCoefficientsLike,
) -> CT.DistortionCoefficients:
    distortion_coefficients = np.asarray(distortion_coefficients, dtype=np.float64)
    assert (
        distortion_coefficients.ndim == 1
    ), f"distortion_coefficients should be a 1-dim array: {distortion_coefficients.shape}"

    valid_lengths = [4, 5, 8, 12, 14]
    assert (
        distortion_coefficients.shape[0] in valid_lengths
    ), f"distortion_coefficients should be None or have a size of {valid_lengths}"

    return distortion_coefficients


CameraMatrix = Annotated[CT.CameraMatrixLike, AfterValidator(check_camera_matrix)]
DistortionCoefficients = Annotated[
    CT.DistortionCoefficientsLike, AfterValidator(check_distortion_coefficients)
]
CameraImageDimension = Annotated[int, Field(gt=0)]


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, validate_assignment=True))
class CameraRadial(CameraRadialBase):
    pixel_width: CameraImageDimension
    pixel_height: CameraImageDimension
    camera_matrix: CameraMatrix
    distortion_coefficients: Optional[DistortionCoefficients] = None

    _undistort_rectify_map = None
    _optimal_undistorted_camera_matrix = None

    @property
    def optimal_undistorted_camera_matrix(self):
        """
        Camera matrix that undistorts the image so that there are no curved edges
        """
        if self._optimal_undistorted_camera_matrix is None:
            self._optimal_undistorted_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                cameraMatrix=np.asarray(self.camera_matrix),
                distCoeffs=np.asarray(self.distortion_coefficients),
                imageSize=(self.pixel_width, self.pixel_height),
                newImgSize=(self.pixel_width, self.pixel_height),
                alpha=0,
                centerPrincipalPoint=False,  # TODO(dan): test that gaze on center is correct here
            )
        return self._optimal_undistorted_camera_matrix

    @property
    def undistort_rectify_map(
        self,
    ) -> tuple[CT.UndistortRectifyMap, CT.UndistortRectifyMap]:
        if self._undistort_rectify_map is None:
            self._undistort_rectify_map = opencv_funcs.undistort_rectify_map(
                self.camera_matrix,
                self.pixel_width,
                self.pixel_height,
                self.distortion_coefficients,
                self.optimal_undistorted_camera_matrix,
            )
        return self._undistort_rectify_map

    def undistort_image(self, image: CT.Image) -> CT.Image:
        map1, map2 = self.undistort_rectify_map
        remapped = remap(
            image,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            # borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return remapped

    def undistort_points(
        self, points_2d: CT.Points2DLike, use_distortion: bool = True
    ) -> CT.Points3D:
        distortion_coefficients = self.distortion_coefficients
        if not use_distortion:
            distortion_coefficients = None
        return opencv_funcs.undistort_points(
            points_2d, self.camera_matrix, distortion_coefficients
        )

    def project_points(
        self, points_3d: CT.Points3DLike, use_distortion: bool = True
    ) -> CT.Points2D:
        distortion_coefficients = self.distortion_coefficients
        if not use_distortion:
            distortion_coefficients = None
        return opencv_funcs.project_points(
            points_3d, self.camera_matrix, distortion_coefficients
        )
