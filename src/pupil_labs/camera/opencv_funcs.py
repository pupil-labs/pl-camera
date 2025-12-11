import cv2
import numpy as np

from . import custom_types as CT


def undistort_points(
    points_2d: CT.Points2D,
    camera_matrix: CT.CameraMatrix,
    distortion_coefficients: CT.DistortionCoefficients | None = None,
    new_camera_matrix: CT.CameraMatrix | None = None,
) -> CT.Points3D:
    points_3d = cv2.undistortPointsIter(
        src=points_2d,
        cameraMatrix=camera_matrix,
        distCoeffs=distortion_coefficients,
        R=None,
        P=new_camera_matrix,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 0.0001),
    )

    # Convert to homogeneous coordinates to obtain proper 3D vectors
    points_3d = cv2.convertPointsToHomogeneous(points_3d)

    # Remove unnecessary dimension introduced by OpenCV
    points_3d = points_3d[:, 0, :]

    return points_3d


def _project_points(
    points_3d: CT.Points3D,
    camera_matrix: CT.CameraMatrix,
    distortion_coefficients: CT.DistortionCoefficients | None = None,
) -> tuple[CT.Points2D, np.ndarray]:
    rvec = tvec = np.zeros((1, 1, 3))

    projected, jacobian = cv2.projectPoints(
        objectPoints=points_3d,
        rvec=rvec,
        tvec=tvec,
        cameraMatrix=camera_matrix,
        distCoeffs=distortion_coefficients,
    )

    # Remove unnecessary dimension introduced by OpenCV
    projected = projected[:, 0, :]

    return np.array(projected).astype(np.float64).squeeze(), jacobian


def project_points(
    points_3d: CT.Points3D,
    camera_matrix: CT.CameraMatrix,
    distortion_coefficients: CT.DistortionCoefficients | None = None,
) -> CT.Points2D:
    return _project_points(points_3d, camera_matrix, distortion_coefficients)[0]


def project_points_with_jacobian(
    points_3d: CT.Points3D,
    camera_matrix: CT.CameraMatrix,
    distortion_coefficients: CT.DistortionCoefficients | None = None,
) -> tuple[CT.Points2D, np.ndarray]:
    return _project_points(points_3d, camera_matrix, distortion_coefficients)
