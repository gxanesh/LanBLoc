"""
Landmark-Based Localization Module (Algorithm 2: LanBLoc-3L)

This module implements the main localization algorithm that combines
landmark detection, stereo depth estimation, and trilateration for
position estimation in GPS-denied environments.

Algorithm 2: Landmark-Based Localization (LanBLoc-3L)
------------------------------------------------------
Input:
    - Stereo images (I_L, I_R)
    - Camera calibration {f_x, B, rectification maps}
    - Landmark database M: l_i -> (x_i, y_i)
    - Minimum landmarks N_req = 3

Output:
    - Current 2D position (x, y)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path
import logging
import yaml

from .stereo_depth import StereoDepth, StereoDepthResult, failure_from_stereo_depth
from .trilateration import Trilateration, TrilaterationResult

logger = logging.getLogger(__name__)


@dataclass
class LandmarkDetection:
    """Detected landmark with bounding box."""
    landmark_id: str
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    class_id: int = 0


@dataclass
class LocalizationResult:
    """Result from LanBLoc localization."""
    position: Tuple[float, float]
    success: bool
    landmarks_detected: int
    landmarks_used: int
    landmark_distances: Dict[str, float] = field(default_factory=dict)
    residual: float = 0.0
    error_message: Optional[str] = None
    
    @property
    def x(self) -> float:
        return self.position[0]
    
    @property
    def y(self) -> float:
        return self.position[1]


class LanBLoc:
    """
    Landmark-Based Localization (LanBLoc-3L).
    
    Main localization class implementing Algorithm 2. Combines YOLO-based
    landmark detection, stereo depth estimation, and trilateration for
    accurate position estimation.
    
    Attributes:
        landmark_db: Database mapping landmark IDs to known positions
        stereo_depth: Stereo depth estimator
        trilateration: Trilateration solver
        detector: YOLO landmark detector
        min_landmarks: Minimum landmarks required (N_req)
    
    Example:
        >>> from lanbloc import LanBLoc
        >>> from lanbloc.data import LandmarkDatabase
        >>> 
        >>> db = LandmarkDatabase.from_dataset("data/landmark_stereov1_corrupt")
        >>> lanbloc = LanBLoc(landmark_db=db)
        >>> 
        >>> result = lanbloc.localize(left_image, right_image)
        >>> if result.success:
        ...     print(f"Position: ({result.x:.2f}, {result.y:.2f})")
    """
    
    def __init__(
        self,
        landmark_db: Dict[str, Tuple[float, float]],
        stereo_depth: Optional[StereoDepth] = None,
        trilateration: Optional[Trilateration] = None,
        detector: Optional[Any] = None,
        min_landmarks: int = 3,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        rectification_maps: Optional[Tuple[np.ndarray, ...]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize LanBLoc localization system.
        
        Args:
            landmark_db: Dictionary mapping landmark IDs to (x, y) positions
            stereo_depth: StereoDepth instance (created from config if None)
            trilateration: Trilateration instance (created from config if None)
            detector: YOLO detector instance (optional)
            min_landmarks: Minimum landmarks required for localization (N_req)
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            rectification_maps: Stereo rectification maps
            config: Configuration dictionary
        """
        self.landmark_db = landmark_db
        self.min_landmarks = min_landmarks
        self.config = config or {}
        
        # Camera parameters
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.rectification_maps = rectification_maps
        
        # Initialize stereo depth estimator
        if stereo_depth is not None:
            self.stereo_depth = stereo_depth
        else:
            self.stereo_depth = StereoDepth.from_config(self.config)
        
        # Initialize trilateration solver
        if trilateration is not None:
            self.trilateration = trilateration
        else:
            self.trilateration = Trilateration.from_config(self.config)
        
        # Initialize detector (lazy loading)
        self._detector = detector
        self._detector_loaded = detector is not None
        
        logger.info(f"LanBLoc initialized with {len(landmark_db)} landmarks, "
                   f"min_landmarks={min_landmarks}")
    
    def _load_detector(self) -> None:
        """Lazy load YOLO detector."""
        if self._detector_loaded:
            return
        
        try:
            from ..detection.yolo_detector import YOLOLandmarkDetector
            
            det_config = self.config.get("detection", {})
            self._detector = YOLOLandmarkDetector(
                weights=det_config.get("weights", "yolov8s.pt"),
                confidence_threshold=det_config.get("confidence_threshold", 0.5),
                device=det_config.get("device", "cuda")
            )
            self._detector_loaded = True
            logger.info("YOLO detector loaded")
            
        except ImportError as e:
            logger.warning(f"Could not load YOLO detector: {e}")
            self._detector = None
            self._detector_loaded = True
    
    def preprocess(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess stereo images (normalize/denoise).
        
        Corresponds to Line 2 of Algorithm 2.
        
        Args:
            left_image: Left camera image
            right_image: Right camera image
            
        Returns:
            Preprocessed (left, right) images
        """
        # Normalize
        left_norm = cv2.normalize(left_image, None, 0, 255, cv2.NORM_MINMAX)
        right_norm = cv2.normalize(right_image, None, 0, 255, cv2.NORM_MINMAX)
        
        # Optional: denoise
        # left_norm = cv2.fastNlMeansDenoisingColored(left_norm, None, 10, 10, 7, 21)
        # right_norm = cv2.fastNlMeansDenoisingColored(right_norm, None, 10, 10, 7, 21)
        
        return left_norm, right_norm
    
    def rectify_undistort(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify and undistort stereo images.
        
        Corresponds to Line 3 of Algorithm 2: RECTIFYUNDISTORT(I_L, I_R)
        
        Args:
            left_image: Left camera image
            right_image: Right camera image
            
        Returns:
            Rectified (left, right) images
        """
        if self.rectification_maps is not None:
            # Apply rectification maps
            map1_L, map2_L, map1_R, map2_R = self.rectification_maps
            left_rect = cv2.remap(left_image, map1_L, map2_L, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_image, map1_R, map2_R, cv2.INTER_LINEAR)
            return left_rect, right_rect
        
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            # Apply undistortion only
            left_undist = cv2.undistort(left_image, self.camera_matrix, self.dist_coeffs)
            right_undist = cv2.undistort(right_image, self.camera_matrix, self.dist_coeffs)
            return left_undist, right_undist
        
        # Return as-is if no calibration available
        return left_image, right_image
    
    def detect_landmarks(
        self,
        image: np.ndarray
    ) -> List[LandmarkDetection]:
        """
        Detect landmarks in image using YOLO.
        
        Corresponds to Line 4 of Algorithm 2: YOLOV11SDETECT(I_L)
        
        Args:
            image: Input image (typically left rectified image)
            
        Returns:
            List of detected landmarks with bounding boxes
        """
        self._load_detector()
        
        if self._detector is None:
            logger.warning("No detector available, returning empty detections")
            return []
        
        detections = self._detector.detect(image)
        
        logger.debug(f"Detected {len(detections)} landmarks")
        return detections
    
    def detect_landmarks_manual(
        self,
        landmark_ids: List[str],
        bboxes: List[Tuple[int, int, int, int]]
    ) -> List[LandmarkDetection]:
        """
        Create landmark detections from manual annotations.
        
        Useful for evaluation with ground truth detections.
        
        Args:
            landmark_ids: List of landmark IDs
            bboxes: List of bounding boxes
            
        Returns:
            List of LandmarkDetection objects
        """
        detections = []
        for lid, bbox in zip(landmark_ids, bboxes):
            detections.append(LandmarkDetection(
                landmark_id=lid,
                bbox=bbox,
                confidence=1.0
            ))
        return detections
    
    def localize(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        detections: Optional[List[LandmarkDetection]] = None
    ) -> LocalizationResult:
        """
        Perform landmark-based localization.
        
        Main entry point implementing Algorithm 2 (LanBLoc-3L).
        
        Args:
            left_image: Left camera image
            right_image: Right camera image
            detections: Optional pre-computed detections (skips YOLO if provided)
            
        Returns:
            LocalizationResult with estimated position
        """
        try:
            # Line 2: Capture and preprocess
            left_proc, right_proc = self.preprocess(left_image, right_image)
            
            # Line 3: Rectify and undistort
            left_rect, right_rect = self.rectify_undistort(left_proc, right_proc)
            
            # Line 4: Detect landmarks (if not provided)
            if detections is None:
                detections = self.detect_landmarks(left_rect)
            
            logger.info(f"Processing {len(detections)} detected landmarks")
            
            # Line 5: Initialize lists for landmarks and distances
            L = []  # Landmark coordinates
            D_hat = []  # Estimated distances
            landmark_distances = {}
            
            # Lines 6-13: Process each detected landmark
            for detection in detections:
                # Line 7: Look up landmark coordinates from database
                landmark_id = detection.landmark_id
                
                if landmark_id not in self.landmark_db:
                    logger.warning(f"Landmark {landmark_id} not in database, skipping")
                    continue
                
                x_i, y_i = self.landmark_db[landmark_id]
                
                # Line 8: Estimate depth using stereo (Algorithm 1)
                depth_result = self.stereo_depth.compute(
                    left_rect,
                    right_rect,
                    roi=detection.bbox
                )
                
                # Lines 9-11: Check for depth estimation failure
                if failure_from_stereo_depth(depth_result):
                    logger.debug(f"Depth estimation failed for {landmark_id}")
                    continue
                
                Z_lm = depth_result.landmark_depth
                
                # Line 12: Add to lists
                L.append((x_i, y_i))
                D_hat.append(Z_lm)
                landmark_distances[landmark_id] = Z_lm
                
                logger.debug(f"Landmark {landmark_id}: pos=({x_i:.2f}, {y_i:.2f}), "
                           f"dist={Z_lm:.2f}m")
            
            # Lines 14-16: Check if enough landmarks
            if len(L) < self.min_landmarks:
                logger.warning(f"Insufficient landmarks: {len(L)} < {self.min_landmarks}")
                return LocalizationResult(
                    position=(0.0, 0.0),
                    success=False,
                    landmarks_detected=len(detections),
                    landmarks_used=len(L),
                    landmark_distances=landmark_distances,
                    error_message=f"Insufficient landmarks: {len(L)} < {self.min_landmarks}"
                )
            
            # Lines 17-18: Compute position via trilateration
            trilat_result = self.trilateration.solve(L, D_hat)
            
            # Line 19: Return position
            logger.info(f"Localization successful: ({trilat_result.position[0]:.2f}, "
                       f"{trilat_result.position[1]:.2f})")
            
            return LocalizationResult(
                position=trilat_result.position,
                success=trilat_result.success,
                landmarks_detected=len(detections),
                landmarks_used=len(L),
                landmark_distances=landmark_distances,
                residual=trilat_result.residual
            )
            
        except Exception as e:
            logger.error(f"Localization failed: {e}")
            return LocalizationResult(
                position=(0.0, 0.0),
                success=False,
                landmarks_detected=0,
                landmarks_used=0,
                error_message=str(e)
            )
    
    def localize_with_known_distances(
        self,
        landmark_ids: List[str],
        distances: List[float]
    ) -> LocalizationResult:
        """
        Localize using pre-computed distances (skip detection and depth estimation).
        
        Useful for evaluation and testing.
        
        Args:
            landmark_ids: List of landmark IDs
            distances: List of distances to each landmark
            
        Returns:
            LocalizationResult with estimated position
        """
        # Build landmark coordinates and distances
        L = []
        D_hat = []
        landmark_distances = {}
        
        for lid, dist in zip(landmark_ids, distances):
            if lid not in self.landmark_db:
                logger.warning(f"Landmark {lid} not in database")
                continue
            
            x_i, y_i = self.landmark_db[lid]
            L.append((x_i, y_i))
            D_hat.append(dist)
            landmark_distances[lid] = dist
        
        if len(L) < self.min_landmarks:
            return LocalizationResult(
                position=(0.0, 0.0),
                success=False,
                landmarks_detected=len(landmark_ids),
                landmarks_used=len(L),
                landmark_distances=landmark_distances,
                error_message=f"Insufficient landmarks: {len(L)} < {self.min_landmarks}"
            )
        
        # Trilateration
        trilat_result = self.trilateration.solve(L, D_hat)
        
        return LocalizationResult(
            position=trilat_result.position,
            success=trilat_result.success,
            landmarks_detected=len(landmark_ids),
            landmarks_used=len(L),
            landmark_distances=landmark_distances,
            residual=trilat_result.residual
        )
    
    @classmethod
    def from_config(cls, config_path: str) -> "LanBLoc":
        """
        Create LanBLoc instance from configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Configured LanBLoc instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load landmark database
        from ..data.landmark_db import LandmarkDatabase
        
        dataset_config = config.get("dataset", {})
        root_dir = dataset_config.get("root_dir", "data/landmark_stereov1_corrupt")
        
        landmark_db = LandmarkDatabase.from_dataset(root_dir)
        
        # Create instance
        return cls(
            landmark_db=landmark_db.get_2d_positions(),
            min_landmarks=config.get("localization", {}).get("min_landmarks", 3),
            config=config
        )
    
    def set_camera_calibration(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        R1: Optional[np.ndarray] = None,
        R2: Optional[np.ndarray] = None,
        P1: Optional[np.ndarray] = None,
        P2: Optional[np.ndarray] = None,
        image_size: Optional[Tuple[int, int]] = None
    ) -> None:
        """
        Set camera calibration parameters.
        
        Args:
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            R1, R2: Rotation matrices for rectification
            P1, P2: Projection matrices for rectification
            image_size: Image dimensions (width, height)
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        if all(x is not None for x in [R1, R2, P1, P2, image_size]):
            # Compute rectification maps
            map1_L, map2_L = cv2.initUndistortRectifyMap(
                camera_matrix, dist_coeffs, R1, P1, image_size, cv2.CV_32FC1
            )
            map1_R, map2_R = cv2.initUndistortRectifyMap(
                camera_matrix, dist_coeffs, R2, P2, image_size, cv2.CV_32FC1
            )
            self.rectification_maps = (map1_L, map2_L, map1_R, map2_R)
