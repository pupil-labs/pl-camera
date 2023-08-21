from typing import Optional

import cv2
import numpy as np

from pupil_labs.camera.utils import convert_points_to_unstructured_numpy

from . import types as CT


def undistort_image(
    image: CT.Image,
    camera_matrix: CT.CameraMatrixLike,
    distortion_coefficients: CT.DistortionCoefficientsLike,
) -> CT.Image:
    return cv2.undistort(image, camera_matrix, distortion_coefficients)


def undistort_points(
    points_2d: CT.Points2DLike,
    camera_matrix: CT.CameraMatrixLike,
    distortion_coefficients: CT.DistortionCoefficientsLike = None,
    new_camera_matrix: Optional[CT.CameraMatrixLike] = None,
) -> CT.Points3D:
    distortion_coefficients = np.asarray(distortion_coefficients, dtype=np.float64)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    np_points_2d = convert_points_to_unstructured_numpy(points_2d).squeeze()
    np_undistorted_points_2d = cv2.undistortPointsIter(
        src=np_points_2d,
        cameraMatrix=camera_matrix,
        distCoeffs=distortion_coefficients,
        R=None,
        P=new_camera_matrix,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 0.0001),
    )

    np_points_3d = cv2.convertPointsToHomogeneous(np_undistorted_points_2d)
    return np_points_3d.squeeze()


def project_points(
    points_3d: CT.Points3DLike,
    camera_matrix: CT.CameraMatrixLike,
    distortion_coefficients: CT.DistortionCoefficientsLike = None,
) -> CT.Points2D:
    rvec = tvec = np.zeros((1, 1, 3))

    distortion_coefficients = np.asarray(distortion_coefficients, dtype=np.float64)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)

    np_points_3d = convert_points_to_unstructured_numpy(points_3d).squeeze()
    if np_points_3d.shape[1] == 2:
        np_points_3d = cv2.convertPointsToHomogeneous(np_points_3d)

    points_2d, _ = cv2.projectPoints(
        objectPoints=np_points_3d,
        rvec=rvec,
        tvec=tvec,
        cameraMatrix=camera_matrix,
        distCoeffs=distortion_coefficients,
    )
    return points_2d.squeeze()


def undistort_rectify_map(
    camera_matrix: CT.CameraMatrixLike,
    width: int,
    height: int,
    distortion_coefficients: CT.DistortionCoefficientsLike = None,
    new_camera_matrix: Optional[CT.CameraMatrixLike] = None,
) -> tuple[CT.UndistortRectifyMap, CT.UndistortRectifyMap]:
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion_coefficients,
        None,
        new_camera_matrix,
        (width, height),
        cv2.CV_32FC1,
    )
    return map1, map2
