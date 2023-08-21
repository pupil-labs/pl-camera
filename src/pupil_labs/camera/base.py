import abc
import typing as T
import warnings
from pathlib import Path

import numpy as np

from . import types as CT


class CameraBase(abc.ABC):
    def __init__(
        self,
        pixel_width: int,
        pixel_height: int,
        camera_matrix: CT.CameraMatrix,
        distortion_coefficents: T.Optional[CT.DistortionCoefficientsLike] = None,
    ):
        if distortion_coefficents is None:
            distortion_coefficents = [0.0, 0.0, 0.0, 0.0, 0.0]

        camera_matrix = np.asarray(camera_matrix, dtype=np.float32)
        distortion_coefficents = np.asarray(distortion_coefficents, dtype=np.float32)

        distortion_coefficents = np.squeeze(distortion_coefficents)
        camera_matrix = np.squeeze(camera_matrix)

        if pixel_width <= 0:
            raise ValueError(
                f"pixel_width should be a positive non-zero integer: {pixel_width}"
            )
        if pixel_height <= 0:
            raise ValueError(
                f"pixel_width should be a positive non-zero integer: {pixel_height}"
            )
        if camera_matrix.shape != (3, 3):
            raise ValueError(
                f"camera_matrix should have 3x3 shape: {camera_matrix.shape}"
            )
        if len(distortion_coefficents.shape) != 1:
            raise ValueError(
                f"dist_coeffs should be a 1-dim array: {distortion_coefficents.shape}"
            )
        if distortion_coefficents.shape[0] < 5:
            # TODO: Not sure about which lengths for dist_coeffs are valid
            raise ValueError(
                f"distortion_coefficents should have at least 5 elements: {distortion_coefficents.shape}"
            )

        self.camera_matrix = camera_matrix
        self.dist_coeffs = distortion_coefficents
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height

    @property
    def focal_length(self) -> float:
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        return (fx + fy) / 2

    @abc.abstractmethod
    def undistort_image(self, image: CT.Image) -> CT.Image:
        raise NotImplementedError()

    @abc.abstractmethod
    def undistort_points(
        self, points_2d: CT.Points3DLike, use_distortion: bool = True
    ) -> CT.Points3D:
        raise NotImplementedError()

    @abc.abstractmethod
    def project_points(
        self, points_3d: CT.Points3DLike, use_distortion: bool = True
    ) -> CT.Points2D:
        raise NotImplementedError()

    def undistort_points_on_image_plane(
        self, points_2d: CT.Points2DLike
    ) -> CT.Points2D:
        points_3d = self.undistort_points(points_2d, use_distortion=True)
        points_2d = self.project_points(points_3d, use_distortion=False)
        return points_2d

    def distort_points_on_image_plane(self, points_2d: CT.Points2DLike) -> CT.Points2D:
        points_3d = self.undistort_points(points_2d, use_distortion=False)
        points_2d = self.project_points(points_3d, use_distortion=True)
        return points_2d
