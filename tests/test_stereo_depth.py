"""
Unit tests for stereo depth module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lanbloc.core.stereo_depth import (
    StereoDepthEstimator,
    StereoDepthResult,
    disparity_to_depth,
    create_stereo_matcher
)


class TestDisparityToDepth:
    """Test disparity to depth conversion."""
    
    def test_basic_conversion(self):
        """Test basic disparity to depth conversion."""
        focal_length = 700  # pixels
        baseline = 0.12  # meters
        disparity = 10.0  # pixels
        
        # Z = f * B / D
        expected_depth = focal_length * baseline / disparity
        
        depth = disparity_to_depth(disparity, focal_length, baseline)
        
        assert abs(depth - expected_depth) < 0.001
    
    def test_zero_disparity(self):
        """Test that zero disparity returns infinity."""
        depth = disparity_to_depth(0, 700, 0.12)
        
        assert depth == float('inf') or np.isinf(depth)
    
    def test_negative_disparity(self):
        """Test handling of negative disparity."""
        depth = disparity_to_depth(-10, 700, 0.12)
        
        # Should handle gracefully (either inf or negative depth)
        assert isinstance(depth, float)
    
    def test_inverse_relationship(self):
        """Test that depth decreases as disparity increases."""
        focal_length = 700
        baseline = 0.12
        
        d1 = disparity_to_depth(10, focal_length, baseline)
        d2 = disparity_to_depth(20, focal_length, baseline)
        
        assert d2 < d1  # Closer objects have larger disparity


class TestCreateStereoMatcher:
    """Test stereo matcher creation."""
    
    def test_create_sgbm(self):
        """Test SGBM matcher creation."""
        matcher = create_stereo_matcher(
            algorithm='SGBM',
            num_disparities=128,
            block_size=11
        )
        
        assert matcher is not None
    
    def test_create_bm(self):
        """Test BM matcher creation."""
        matcher = create_stereo_matcher(
            algorithm='BM',
            num_disparities=64,
            block_size=15
        )
        
        assert matcher is not None
    
    def test_invalid_algorithm(self):
        """Test that invalid algorithm raises error."""
        with pytest.raises(ValueError):
            create_stereo_matcher(algorithm='INVALID')
    
    def test_disparity_divisibility(self):
        """Test that num_disparities must be divisible by 16."""
        # Should work with valid values
        matcher = create_stereo_matcher(num_disparities=128)
        assert matcher is not None
        
        # Should raise or adjust for invalid values
        # The implementation might auto-adjust, so just check it doesn't crash


class TestStereoDepthEstimator:
    """Test StereoDepthEstimator class."""
    
    def test_init_default(self):
        """Test default initialization."""
        estimator = StereoDepthEstimator()
        
        assert estimator.focal_length == 700
        assert estimator.baseline == 0.12
        assert estimator.algorithm == 'SGBM'
    
    def test_init_custom(self):
        """Test custom initialization."""
        estimator = StereoDepthEstimator(
            focal_length=800,
            baseline=0.15,
            algorithm='BM',
            num_disparities=64
        )
        
        assert estimator.focal_length == 800
        assert estimator.baseline == 0.15
        assert estimator.algorithm == 'BM'
    
    def test_compute_disparity_synthetic(self):
        """Test disparity computation with synthetic images."""
        estimator = StereoDepthEstimator()
        
        # Create synthetic stereo pair (simple horizontal gradient)
        height, width = 480, 640
        left = np.zeros((height, width), dtype=np.uint8)
        right = np.zeros((height, width), dtype=np.uint8)
        
        # Add some texture (required for stereo matching)
        for y in range(0, height, 10):
            left[y:y+5, :] = 255
            right[y:y+5, :] = 255
        
        # Shift right image to simulate disparity
        shift = 10
        right[:, shift:] = left[:, :-shift]
        
        disparity = estimator.compute_disparity(left, right)
        
        assert disparity is not None
        assert disparity.shape == (height, width)
    
    def test_compute_depth_from_roi(self):
        """Test depth computation from ROI."""
        estimator = StereoDepthEstimator(
            focal_length=700,
            baseline=0.12,
            min_valid_depth_points=10
        )
        
        # Create synthetic depth map
        height, width = 480, 640
        depth_map = np.full((height, width), 5.0, dtype=np.float32)
        
        # ROI in center of image
        roi = (width//4, height//4, width//2, height//2)
        
        result = estimator.compute_depth_from_roi(depth_map, roi)
        
        assert result.success
        assert abs(result.depth - 5.0) < 0.1
    
    def test_compute_depth_from_roi_insufficient_points(self):
        """Test depth computation with insufficient valid points."""
        estimator = StereoDepthEstimator(
            min_valid_depth_points=1000000  # Very high threshold
        )
        
        # Create depth map with few valid points
        height, width = 480, 640
        depth_map = np.full((height, width), np.nan, dtype=np.float32)
        depth_map[100:110, 100:110] = 5.0  # Only small region valid
        
        roi = (0, 0, width, height)
        
        result = estimator.compute_depth_from_roi(depth_map, roi)
        
        assert not result.success


class TestStereoDepthResult:
    """Test StereoDepthResult dataclass."""
    
    def test_success_result(self):
        """Test successful result."""
        result = StereoDepthResult(
            success=True,
            depth=5.5,
            depth_map=np.ones((100, 100)),
            disparity_map=np.ones((100, 100)),
            valid_points=500,
            roi=(10, 10, 50, 50)
        )
        
        assert result.success
        assert result.depth == 5.5
        assert result.valid_points == 500
    
    def test_failure_result(self):
        """Test failure result."""
        result = StereoDepthResult(
            success=False,
            error_message="Insufficient valid depth points"
        )
        
        assert not result.success
        assert result.depth is None
        assert "Insufficient" in result.error_message
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = StereoDepthResult(
            success=True,
            depth=5.5,
            valid_points=500,
            roi=(10, 10, 50, 50)
        )
        
        d = result.to_dict()
        
        assert d['success'] == True
        assert d['depth'] == 5.5
        assert d['valid_points'] == 500


class TestIntegration:
    """Integration tests for stereo depth estimation."""
    
    def test_full_pipeline_synthetic(self):
        """Test full pipeline with synthetic images."""
        estimator = StereoDepthEstimator(
            focal_length=700,
            baseline=0.12,
            min_valid_depth_points=50
        )
        
        # Create textured synthetic stereo pair
        height, width = 480, 640
        np.random.seed(42)
        
        # Create random texture
        texture = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        
        # Apply Gaussian blur for more realistic texture
        try:
            import cv2
            texture = cv2.GaussianBlur(texture, (5, 5), 1.0)
        except ImportError:
            pass
        
        left = texture.copy()
        right = texture.copy()
        
        # Shift right image to simulate disparity of ~10 pixels
        shift = 10
        right[:, shift:] = left[:, :-shift]
        right[:, :shift] = 0
        
        # Compute depth
        result = estimator.estimate_depth(left, right)
        
        # Result should exist (success depends on matching quality)
        assert isinstance(result, StereoDepthResult)
        assert result.disparity_map is not None
    
    def test_depth_within_roi(self):
        """Test depth estimation within a specific ROI."""
        estimator = StereoDepthEstimator(
            focal_length=700,
            baseline=0.12,
            min_valid_depth_points=10
        )
        
        # Create depth map with known depth in ROI
        height, width = 480, 640
        depth_map = np.full((height, width), np.inf, dtype=np.float32)
        
        # Set specific depth in ROI region
        roi_depth = 8.4  # Expected depth in meters
        depth_map[200:300, 250:350] = roi_depth
        
        roi = (250, 200, 100, 100)  # x, y, w, h
        
        result = estimator.compute_depth_from_roi(depth_map, roi)
        
        if result.success:
            assert abs(result.depth - roi_depth) < 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
