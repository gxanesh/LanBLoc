"""
Camera Calibration Module

Utilities for camera intrinsic/extrinsic calibration and stereo rectification.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import yaml
import logging
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


@dataclass
class CameraCalibration:
    """
    Single camera calibration parameters.

    Attributes:
        camera_matrix: 3x3 intrinsic camera matrix
        dist_coeffs: Distortion coefficients (k1, k2, p1, p2, k3, ...)
        image_size: Image dimensions (width, height)
        reprojection_error: RMS reprojection error from calibration
    """
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: Tuple[int, int]
    reprojection_error: float = 0.0

    @property
    def fx(self) -> float:
        """Focal length in x direction (pixels)."""
        return self.camera_matrix[0, 0]

    @property
    def fy(self) -> float:
        """Focal length in y direction (pixels)."""
        return self.camera_matrix[1, 1]

    @property
    def cx(self) -> float:
        """Principal point x coordinate."""
        return self.camera_matrix[0, 2]

    @property
    def cy(self) -> float:
        """Principal point y coordinate."""
        return self.camera_matrix[1, 2]

    def undistort(self, image: np.ndarray) -> np.ndarray:
        """Undistort an image using calibration parameters."""
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

    def save(self, path: str) -> None:
        """Save calibration to YAML file."""
        data = {
            'camera_matrix': self.camera_matrix.tolist(),
            'dist_coeffs': self.dist_coeffs.tolist(),
            'image_size': list(self.image_size),
            'reprojection_error': self.reprojection_error
        }
        with open(path, 'w') as f:
            yaml.safe_dump(data, f)

    @classmethod
    def load(cls, path: str) -> "CameraCalibration":
        """Load calibration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return cls(
            camera_matrix=np.array(data['camera_matrix']),
            dist_coeffs=np.array(data['dist_coeffs']),
            image_size=tuple(data['image_size']),
            reprojection_error=data.get('reprojection_error', 0.0)
        )


@dataclass
class StereoCalibration:
    """
    Stereo camera calibration parameters.

    Attributes:
        left_camera: Left camera calibration
        right_camera: Right camera calibration
        R: Rotation matrix between cameras
        T: Translation vector between cameras
        E: Essential matrix
        F: Fundamental matrix
        baseline: Stereo baseline in meters
    """
    left_camera: CameraCalibration
    right_camera: CameraCalibration
    R: np.ndarray
    T: np.ndarray
    E: np.ndarray
    F: np.ndarray
    baseline: float

    # Rectification parameters (computed lazily)
    _R1: Optional[np.ndarray] = None
    _R2: Optional[np.ndarray] = None
    _P1: Optional[np.ndarray] = None
    _P2: Optional[np.ndarray] = None
    _Q: Optional[np.ndarray] = None
    _rect_maps: Optional[Tuple[np.ndarray, ...]] = None

    def compute_rectification(self) -> None:
        """Compute stereo rectification parameters."""
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.left_camera.camera_matrix,
            self.left_camera.dist_coeffs,
            self.right_camera.camera_matrix,
            self.right_camera.dist_coeffs,
            self.left_camera.image_size,
            self.R,
            self.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0
        )

        self._R1 = R1
        self._R2 = R2
        self._P1 = P1
        self._P2 = P2
        self._Q = Q

        # Compute rectification maps
        map1_L, map2_L = cv2.initUndistortRectifyMap(
            self.left_camera.camera_matrix,
            self.left_camera.dist_coeffs,
            R1, P1,
            self.left_camera.image_size,
            cv2.CV_32FC1
        )

        map1_R, map2_R = cv2.initUndistortRectifyMap(
            self.right_camera.camera_matrix,
            self.right_camera.dist_coeffs,
            R2, P2,
            self.right_camera.image_size,
            cv2.CV_32FC1
        )

        self._rect_maps = (map1_L, map2_L, map1_R, map2_R)

        logger.info("Stereo rectification computed")

    def rectify(
            self,
            left_image: np.ndarray,
            right_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify a stereo image pair.

        Args:
            left_image: Left camera image
            right_image: Right camera image

        Returns:
            Rectified (left, right) images
        """
        if self._rect_maps is None:
            self.compute_rectification()

        map1_L, map2_L, map1_R, map2_R = self._rect_maps

        left_rect = cv2.remap(left_image, map1_L, map2_L, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_image, map1_R, map2_R, cv2.INTER_LINEAR)

        return left_rect, right_rect

    def save(self, path: str) -> None:
        """Save stereo calibration to YAML file."""
        data = {
            'left_camera': {
                'camera_matrix': self.left_camera.camera_matrix.tolist(),
                'dist_coeffs': self.left_camera.dist_coeffs.tolist(),
                'image_size': list(self.left_camera.image_size),
            },
            'right_camera': {
                'camera_matrix': self.right_camera.camera_matrix.tolist(),
                'dist_coeffs': self.right_camera.dist_coeffs.tolist(),
                'image_size': list(self.right_camera.image_size),
            },
            'R': self.R.tolist(),
            'T': self.T.tolist(),
            'E': self.E.tolist(),
            'F': self.F.tolist(),
            'baseline': self.baseline
        }

        with open(path, 'w') as f:
            yaml.safe_dump(data, f)

    @classmethod
    def load(cls, path: str) -> "StereoCalibration":
        """Load stereo calibration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        left_camera = CameraCalibration(
            camera_matrix=np.array(data['left_camera']['camera_matrix']),
            dist_coeffs=np.array(data['left_camera']['dist_coeffs']),
            image_size=tuple(data['left_camera']['image_size'])
        )

        right_camera = CameraCalibration(
            camera_matrix=np.array(data['right_camera']['camera_matrix']),
            dist_coeffs=np.array(data['right_camera']['dist_coeffs']),
            image_size=tuple(data['right_camera']['image_size'])
        )

        return cls(
            left_camera=left_camera,
            right_camera=right_camera,
            R=np.array(data['R']),
            T=np.array(data['T']),
            E=np.array(data['E']),
            F=np.array(data['F']),
            baseline=data['baseline']
        )

    @classmethod
    def load_from_opencv_xml(cls, xml_path: str) -> "StereoCalibration":
        """
        Load stereo calibration from OpenCV XML format.

        Args:
            xml_path: Path to OpenCV stereo calibration XML file

        Returns:
            StereoCalibration object
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        def parse_matrix(element):
            """Parse OpenCV matrix from XML element."""
            rows = int(element.find('rows').text)
            cols = int(element.find('cols').text)
            data_text = element.find('data').text
            data = np.array([float(x) for x in data_text.split()])
            return data.reshape(rows, cols)

        # Parse camera matrices
        K_left = parse_matrix(root.find('K_left'))
        K_right = parse_matrix(root.find('K_right'))
        dist_left = parse_matrix(root.find('dist_left')).flatten()
        dist_right = parse_matrix(root.find('dist_right')).flatten()

        # Parse stereo extrinsics
        R = parse_matrix(root.find('R'))
        T = parse_matrix(root.find('T')).flatten()
        E = parse_matrix(root.find('E'))
        F = parse_matrix(root.find('F'))

        # Parse rectification parameters if available
        R1 = parse_matrix(root.find('R1')) if root.find('R1') is not None else None
        R2 = parse_matrix(root.find('R2')) if root.find('R2') is not None else None
        P1 = parse_matrix(root.find('P1')) if root.find('P1') is not None else None
        P2 = parse_matrix(root.find('P2')) if root.find('P2') is not None else None
        Q = parse_matrix(root.find('Q')) if root.find('Q') is not None else None

        # Determine image size from rectification maps or default
        left_map_x = root.find('Left_Stereo_Map_x')
        if left_map_x is not None:
            img_height = int(left_map_x.find('rows').text)
            img_width = int(left_map_x.find('cols').text)
            image_size = (img_width, img_height)
        else:
            image_size = (640, 480)  # Default

        # Create camera calibrations
        left_camera = CameraCalibration(
            camera_matrix=K_left,
            dist_coeffs=dist_left,
            image_size=image_size
        )

        right_camera = CameraCalibration(
            camera_matrix=K_right,
            dist_coeffs=dist_right,
            image_size=image_size
        )

        # Compute baseline from translation vector
        baseline = np.linalg.norm(T)

        # Create stereo calibration
        stereo = cls(
            left_camera=left_camera,
            right_camera=right_camera,
            R=R,
            T=T,
            E=E,
            F=F,
            baseline=baseline
        )

        # Set pre-computed rectification parameters if available
        if all(x is not None for x in [R1, R2, P1, P2, Q]):
            stereo._R1 = R1
            stereo._R2 = R2
            stereo._P1 = P1
            stereo._P2 = P2
            stereo._Q = Q

            # Compute rectification maps
            map1_L, map2_L = cv2.initUndistortRectifyMap(
                K_left, dist_left, R1, P1, image_size, cv2.CV_32FC1
            )
            map1_R, map2_R = cv2.initUndistortRectifyMap(
                K_right, dist_right, R2, P2, image_size, cv2.CV_32FC1
            )
            stereo._rect_maps = (map1_L, map2_L, map1_R, map2_R)

            logger.info("Loaded pre-computed rectification parameters")

        logger.info(f"Loaded stereo calibration from {xml_path}")
        logger.info(f"  Image size: {image_size}")
        logger.info(f"  Baseline: {baseline:.2f} mm")

        return stereo


def calibrate_camera(
        images: List[np.ndarray],
        pattern_size: Tuple[int, int],
        square_size: float,
        flags: int = 0
) -> CameraCalibration:
    """
    Calibrate a single camera using checkerboard images.

    Args:
        images: List of calibration images
        pattern_size: Checkerboard inner corners (columns, rows)
        square_size: Size of checkerboard square in meters
        flags: OpenCV calibration flags

    Returns:
        CameraCalibration object
    """
    # Prepare object points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    obj_points = []
    img_points = []
    image_size = None

    for img in images:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            # Refine corners
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )

            obj_points.append(objp)
            img_points.append(corners)

    if len(obj_points) < 3:
        raise ValueError(f"Only {len(obj_points)} valid calibration images found")

    # Calibrate
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None, flags=flags
    )

    logger.info(f"Camera calibrated with RMS error: {ret:.4f}")

    return CameraCalibration(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,
        reprojection_error=ret
    )


def calibrate_stereo(
        left_images: List[np.ndarray],
        right_images: List[np.ndarray],
        pattern_size: Tuple[int, int],
        square_size: float,
        flags: int = cv2.CALIB_FIX_INTRINSIC
) -> StereoCalibration:
    """
    Calibrate a stereo camera pair.

    Args:
        left_images: List of left camera calibration images
        right_images: List of right camera calibration images
        pattern_size: Checkerboard inner corners
        square_size: Square size in meters
        flags: OpenCV stereo calibration flags

    Returns:
        StereoCalibration object
    """
    # First calibrate individual cameras
    left_calib = calibrate_camera(left_images, pattern_size, square_size)
    right_calib = calibrate_camera(right_images, pattern_size, square_size)

    # Prepare object points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    obj_points = []
    left_points = []
    right_points = []

    for left_img, right_img in zip(left_images, right_images):
        # Convert to grayscale
        if len(left_img.shape) == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_img

        if len(right_img.shape) == 3:
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        else:
            right_gray = right_img

        # Find corners in both images
        ret_L, corners_L = cv2.findChessboardCorners(left_gray, pattern_size, None)
        ret_R, corners_R = cv2.findChessboardCorners(right_gray, pattern_size, None)

        if ret_L and ret_R:
            corners_L = cv2.cornerSubPix(
                left_gray, corners_L, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            corners_R = cv2.cornerSubPix(
                right_gray, corners_R, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )

            obj_points.append(objp)
            left_points.append(corners_L)
            right_points.append(corners_R)

    if len(obj_points) < 3:
        raise ValueError(f"Only {len(obj_points)} valid stereo pairs found")

    # Stereo calibration
    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_points,
        left_points,
        right_points,
        left_calib.camera_matrix,
        left_calib.dist_coeffs,
        right_calib.camera_matrix,
        right_calib.dist_coeffs,
        left_calib.image_size,
        flags=flags
    )

    # Compute baseline
    baseline = np.linalg.norm(T)

    logger.info(f"Stereo calibrated: RMS={ret:.4f}, baseline={baseline:.4f}m")

    # Update camera calibrations
    left_calib.camera_matrix = K1
    left_calib.dist_coeffs = D1
    right_calib.camera_matrix = K2
    right_calib.dist_coeffs = D2

    stereo_calib = StereoCalibration(
        left_camera=left_calib,
        right_camera=right_calib,
        R=R,
        T=T,
        E=E,
        F=F,
        baseline=baseline
    )

    # Compute rectification
    stereo_calib.compute_rectification()

    return stereo_calib


def get_trilat_stereo_calibration() -> StereoCalibration:
    """
    Get pre-calibrated stereo camera parameters for the TriLat dataset.

    This returns hardcoded calibration data for the stereo camera system
    used in the TriLat campus dataset collection.

    Camera: Stereo camera with ~101.58mm baseline
    Image resolution: 640x480

    Returns:
        StereoCalibration object with pre-computed rectification
    """
    # Left camera intrinsics
    K_left = np.array([
        [812.05758235568624, 0.0, 323.31408515421867],
        [0.0, 830.95653798935859, 130.01151290953925],
        [0.0, 0.0, 1.0]
    ])

    # Left camera distortion coefficients (k1, k2, p1, p2, k3)
    dist_left = np.array([
        -0.0048172892818360213,
        1.1665338541421777,
        -0.053659291271298412,
        0.00066455478299043056,
        -3.8793461751787559
    ])

    # Right camera intrinsics
    K_right = np.array([
        [808.95302198875254, 0.0, 302.3723149394142],
        [0.0, 821.76132098158143, 152.23723661527475],
        [0.0, 0.0, 1.0]
    ])

    # Right camera distortion coefficients (k1, k2, p1, p2, k3)
    dist_right = np.array([
        0.078432594447713111,
        0.11873251098810297,
        -0.044173793118491012,
        -0.016917989740622127,
        -0.49748588852387554
    ])

    # Rotation matrix between cameras
    R = np.array([
        [0.99908229921033809, -0.017949289244548312, 0.038889361272719698],
        [0.020411197518043045, 0.99774996451096287, -0.063862284129615213],
        [-0.037655576220050092, 0.064597456095327868, 0.99720069506872488]
    ])

    # Translation vector (mm) - baseline ~101.58mm
    T = np.array([-101.58189652104497, -2.4021076857082995, 4.382036314849449])

    # Essential matrix
    E = np.array([
        [0.0010101402943274301, -4.5273466233905806, -2.1155366056130349],
        [0.55289006963753495, 6.4832776633029932, 101.46795241054369],
        [0.32649511543981324, -101.39644977447625, 6.580668371657632]
    ])

    # Fundamental matrix
    F = np.array([
        [3.5992052319570572e-10, -1.5764390308146769e-06, -0.00040727510691888257],
        [1.9392835416141518e-07, 2.2223152185625279e-06, 0.028549731315769204],
        [6.4475698396245168e-05, -0.028423047085812902, 1.0]
    ])

    # Rectification rotation matrices
    R1 = np.array([
        [0.99998007069629613, 0.0028544551422451693, -0.0056311895787198099],
        [-0.0030339890599737993, 0.99947891785082543, -0.032135458330741669],
        [0.0055365260420711312, 0.032151902861009406, 0.99946765931759973]
    ])

    R2 = np.array([
        [0.99879215938886623, 0.023618443882825215, -0.043085861391870164],
        [-0.02222086840483832, 0.99922037123569929, 0.032632540736603412],
        [0.043823000247135868, -0.031635720572559416, 0.99853829462529597]
    ])

    # Projection matrices
    P1 = np.array([
        [1081.8708395231049, 0.0, 373.12105178833008, 0.0],
        [0.0, 1081.8708395231049, 38.21363639831543, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ])

    P2 = np.array([
        [1081.8708395231049, 0.0, 373.12105178833008, -110031.39205339375],
        [0.0, 1081.8708395231049, 38.21363639831543, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ])

    # Disparity-to-depth matrix (Q)
    Q = np.array([
        [1.0, 0.0, 0.0, -373.12105178833008],
        [0.0, 1.0, 0.0, -38.21363639831543],
        [0.0, 0.0, 0.0, 1081.8708395231049],
        [0.0, 0.0, 0.0098323834619679917, 0.0]
    ])

    image_size = (640, 480)

    # Create camera calibrations
    left_camera = CameraCalibration(
        camera_matrix=K_left,
        dist_coeffs=dist_left,
        image_size=image_size
    )

    right_camera = CameraCalibration(
        camera_matrix=K_right,
        dist_coeffs=dist_right,
        image_size=image_size
    )

    # Compute baseline in mm
    baseline = np.linalg.norm(T)

    # Create stereo calibration
    stereo = StereoCalibration(
        left_camera=left_camera,
        right_camera=right_camera,
        R=R,
        T=T,
        E=E,
        F=F,
        baseline=baseline
    )

    # Set pre-computed rectification parameters
    stereo._R1 = R1
    stereo._R2 = R2
    stereo._P1 = P1
    stereo._P2 = P2
    stereo._Q = Q

    # Compute rectification maps
    map1_L, map2_L = cv2.initUndistortRectifyMap(
        K_left, dist_left, R1, P1, image_size, cv2.CV_32FC1
    )
    map1_R, map2_R = cv2.initUndistortRectifyMap(
        K_right, dist_right, R2, P2, image_size, cv2.CV_32FC1
    )
    stereo._rect_maps = (map1_L, map2_L, map1_R, map2_R)

    logger.info("Loaded TriLat stereo calibration")
    logger.info(f"  Image size: {image_size}")
    logger.info(f"  Baseline: {baseline:.2f} mm")
    logger.info(f"  Focal length (rectified): {P1[0, 0]:.2f} px")

    return stereo


def compute_depth_from_disparity(
        disparity: np.ndarray,
        Q: np.ndarray
) -> np.ndarray:
    """
    Convert disparity map to depth using Q matrix.

    Args:
        disparity: Disparity map from stereo matching
        Q: Disparity-to-depth mapping matrix (4x4)

    Returns:
        Depth map in same units as baseline in calibration
    """
    # Reproject to 3D
    points_3d = cv2.reprojectImageTo3D(disparity, Q)

    # Extract depth (Z coordinate)
    depth = points_3d[:, :, 2]

    # Mask invalid disparities
    depth[disparity <= 0] = 0

    return depth


def compute_depth_from_stereo(
        left_rect: np.ndarray,
        right_rect: np.ndarray,
        stereo_calib: StereoCalibration,
        num_disparities: int = 64,
        block_size: int = 15
) -> np.ndarray:
    """
    Compute depth map from rectified stereo pair.

    Args:
        left_rect: Rectified left image
        right_rect: Rectified right image
        stereo_calib: Stereo calibration with Q matrix
        num_disparities: Number of disparities (must be divisible by 16)
        block_size: Block size for matching (odd number)

    Returns:
        Depth map in same units as baseline
    """
    # Convert to grayscale if needed
    if len(left_rect.shape) == 3:
        left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    else:
        left_gray = left_rect

    if len(right_rect.shape) == 3:
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
    else:
        right_gray = right_rect

    # Create stereo matcher
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size ** 2,
        P2=32 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    # Compute disparity
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

    # Convert to depth
    if stereo_calib._Q is not None:
        depth = compute_depth_from_disparity(disparity, stereo_calib._Q)
    else:
        # Fallback: compute depth from disparity directly
        # depth = baseline * focal_length / disparity
        focal = stereo_calib._P1[0, 0] if stereo_calib._P1 is not None else stereo_calib.left_camera.fx
        depth = np.zeros_like(disparity)
        valid_mask = disparity > 0
        depth[valid_mask] = stereo_calib.baseline * focal / disparity[valid_mask]

    return depth