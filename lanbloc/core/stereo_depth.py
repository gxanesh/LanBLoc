"""
Stereo Depth Estimation Module (Algorithm 1)

This module implements metric depth estimation from stereo landmark ROIs.
The algorithm computes a disparity map using stereo matching and converts
it to metric depth using the camera's focal length and baseline.

Algorithm 1: Metric Depth from Stereo Landmark ROI
--------------------------------------------------
Input:
    - Rectified stereo images (I_L, I_R)
    - Camera focal length f (in pixels)
    - Stereo baseline B (in meters)
    - Landmark ROI (x_min, y_min, w, h)
    - Valid depth threshold N_min

Output:
    - Depth map Z
    - Scalar landmark distance Z_lm
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class StereoDepthResult:
    """Result from stereo depth computation."""
    depth_map: np.ndarray
    landmark_depth: float
    disparity_map: np.ndarray
    valid_depth_count: int
    success: bool
    error_message: Optional[str] = None


class StereoDepth:
    """
    Stereo depth estimation for landmark distance measurement.
    
    Implements Algorithm 1: Metric Depth from Stereo Landmark ROI.
    
    Attributes:
        focal_length: Camera focal length in pixels
        baseline: Stereo baseline in meters
        config: Additional configuration parameters
    
    Example:
        >>> stereo = StereoDepth(focal_length=700, baseline=0.12)
        >>> result = stereo.compute(left_img, right_img, roi=(100, 100, 50, 50))
        >>> print(f"Landmark depth: {result.landmark_depth:.2f} m")
    """
    
    def __init__(
        self,
        focal_length: float,
        baseline: float,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize stereo depth estimator.
        
        Args:
            focal_length: Camera focal length in pixels
            baseline: Stereo baseline in meters
            config: Optional configuration dictionary with keys:
                - algorithm: "BM" or "SGBM" (default: "SGBM")
                - num_disparities: Number of disparities (default: 128)
                - block_size: Matching block size (default: 11)
                - min_disparity: Minimum disparity (default: 0)
                - disp_scale: Disparity scale factor (default: 16)
        """
        self.focal_length = focal_length
        self.baseline = baseline
        self.config = config or {}
        
        # Set default configuration values
        self.algorithm = self.config.get("algorithm", "SGBM")
        self.num_disparities = self.config.get("num_disparities", 128)
        self.block_size = self.config.get("block_size", 11)
        self.min_disparity = self.config.get("min_disparity", 0)
        self.disp_scale = self.config.get("disp_scale", 16)
        self.min_valid_depth = self.config.get("min_valid_depth", 100)
        
        # Initialize stereo matcher
        self._init_stereo_matcher()
        
        logger.info(f"StereoDepth initialized: f={focal_length}, B={baseline}, "
                   f"algorithm={self.algorithm}")
    
    def _init_stereo_matcher(self) -> None:
        """Initialize the stereo matching algorithm."""
        if self.algorithm.upper() == "BM":
            self.stereo_matcher = cv2.StereoBM_create(
                numDisparities=self.num_disparities,
                blockSize=self.block_size
            )
        else:  # SGBM
            sgbm_config = self.config.get("sgbm", {})
            P1 = sgbm_config.get("P1", 8 * 3 * self.block_size ** 2)
            P2 = sgbm_config.get("P2", 32 * 3 * self.block_size ** 2)
            
            self.stereo_matcher = cv2.StereoSGBM_create(
                minDisparity=self.min_disparity,
                numDisparities=self.num_disparities,
                blockSize=self.block_size,
                P1=P1,
                P2=P2,
                disp12MaxDiff=sgbm_config.get("disp12_max_diff", 1),
                preFilterCap=sgbm_config.get("prefilter_cap", 63),
                uniquenessRatio=sgbm_config.get("uniqueness_ratio", 10),
                speckleWindowSize=sgbm_config.get("speckle_window_size", 100),
                speckleRange=sgbm_config.get("speckle_range", 32),
                mode=sgbm_config.get("mode", cv2.STEREO_SGBM_MODE_HH)
            )
    
    def compute_disparity(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray
    ) -> np.ndarray:
        """
        Compute disparity map from stereo images.
        
        Args:
            left_image: Left rectified image
            right_image: Right rectified image
            
        Returns:
            Disparity map (scaled by disp_scale)
        """
        # Convert to grayscale if needed
        if len(left_image.shape) == 3:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_image
            
        if len(right_image.shape) == 3:
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        else:
            right_gray = right_image
        
        # Compute raw disparity (Line 2 of Algorithm 1)
        disparity_raw = self.stereo_matcher.compute(left_gray, right_gray)
        
        return disparity_raw
    
    def disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        """
        Convert disparity map to depth map.
        
        Uses the formula: Z(x,y) = (f * B) / D(x,y)
        
        Args:
            disparity: Raw disparity map (scaled)
            
        Returns:
            Depth map in meters
        """
        # Scale disparity to actual values (Line 3 of Algorithm 1)
        disparity_scaled = disparity.astype(np.float64) / self.disp_scale
        
        # Create valid disparity mask (Line 4)
        valid_mask = disparity_scaled > 0
        
        # Initialize depth map with NaN (Line 5)
        depth = np.full_like(disparity_scaled, np.nan)
        
        # Compute depth where disparity is valid (Lines 6-8)
        # Z(x,y) = (f * B) / D(x,y)
        depth[valid_mask] = (self.focal_length * self.baseline) / disparity_scaled[valid_mask]
        
        return depth
    
    def compute(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
        min_valid_depth: Optional[int] = None
    ) -> StereoDepthResult:
        """
        Compute metric depth for a landmark ROI.
        
        This is the main entry point implementing Algorithm 1.
        
        Args:
            left_image: Left rectified image
            right_image: Right rectified image
            roi: Region of interest (x_min, y_min, width, height).
                 If None, uses entire image.
            min_valid_depth: Minimum valid depth points required (N_min).
                            Overrides config value if provided.
        
        Returns:
            StereoDepthResult containing depth map and landmark distance
        """
        if min_valid_depth is None:
            min_valid_depth = self.min_valid_depth
        
        try:
            # Step 1: Compute disparity map (Line 2)
            disparity = self.compute_disparity(left_image, right_image)
            
            # Step 2: Convert to depth (Lines 3-8)
            depth_map = self.disparity_to_depth(disparity)
            
            # Step 3: Extract ROI if provided (Lines 9-11)
            if roi is not None:
                x_min, y_min, w, h = roi
                x1 = x_min + w
                y1 = y_min + h
                
                # Ensure ROI is within image bounds
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x1 = min(depth_map.shape[1], x1)
                y1 = min(depth_map.shape[0], y1)
                
                # Extract patch (Line 11)
                depth_patch = depth_map[y_min:y1, x_min:x1]
            else:
                depth_patch = depth_map
            
            # Step 4: Get valid (finite) depth values (Line 12)
            valid_depths = depth_patch[np.isfinite(depth_patch)]
            
            # Step 5: Check if enough valid depths (Line 13)
            if len(valid_depths) < min_valid_depth:
                logger.warning(f"Insufficient valid depth: {len(valid_depths)} < {min_valid_depth}")
                return StereoDepthResult(
                    depth_map=depth_map,
                    landmark_depth=np.nan,
                    disparity_map=disparity,
                    valid_depth_count=len(valid_depths),
                    success=False,
                    error_message=f"Insufficient valid depth: {len(valid_depths)} < {min_valid_depth}"
                )
            
            # Step 6: Compute robust landmark depth using median (Line 16)
            landmark_depth = np.median(valid_depths)
            
            logger.debug(f"Landmark depth: {landmark_depth:.2f}m from {len(valid_depths)} valid points")
            
            return StereoDepthResult(
                depth_map=depth_map,
                landmark_depth=landmark_depth,
                disparity_map=disparity,
                valid_depth_count=len(valid_depths),
                success=True
            )
            
        except Exception as e:
            logger.error(f"Stereo depth computation failed: {e}")
            return StereoDepthResult(
                depth_map=np.array([]),
                landmark_depth=np.nan,
                disparity_map=np.array([]),
                valid_depth_count=0,
                success=False,
                error_message=str(e)
            )
    
    def compute_batch(
        self,
        left_images: list,
        right_images: list,
        rois: Optional[list] = None
    ) -> list:
        """
        Compute depth for multiple stereo pairs.
        
        Args:
            left_images: List of left images
            right_images: List of right images
            rois: Optional list of ROIs (one per pair)
            
        Returns:
            List of StereoDepthResult objects
        """
        if rois is None:
            rois = [None] * len(left_images)
        
        results = []
        for left, right, roi in zip(left_images, right_images, rois):
            result = self.compute(left, right, roi)
            results.append(result)
        
        return results
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "StereoDepth":
        """
        Create StereoDepth instance from configuration dictionary.
        
        Args:
            config: Configuration with 'camera' and 'stereo' sections
            
        Returns:
            Configured StereoDepth instance
        """
        camera_config = config.get("camera", {})
        stereo_config = config.get("stereo", {})
        
        return cls(
            focal_length=camera_config.get("focal_length", 700.0),
            baseline=camera_config.get("baseline", 0.12),
            config=stereo_config
        )


def failure_from_stereo_depth(result: StereoDepthResult) -> bool:
    """
    Check if stereo depth computation failed.
    
    This function corresponds to FAILUREFROMSTEREODEPTH() in Algorithm 2.
    
    Args:
        result: Result from StereoDepth.compute()
        
    Returns:
        True if the computation failed (insufficient valid depth)
    """
    return not result.success
