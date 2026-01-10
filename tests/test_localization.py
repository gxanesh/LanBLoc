"""
Unit tests for localization module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lanbloc.core.localization import (
    LanBLoc,
    LocalizationResult,
    LocalizationConfig
)
from lanbloc.data.landmark_db import LandmarkDatabase, TRILATERATION_DATA


class TestLocalizationConfig:
    """Test LocalizationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = LocalizationConfig()
        
        assert config.min_landmarks == 3
        assert config.focal_length == 700
        assert config.baseline == 0.12
        assert config.stereo_algorithm == 'SGBM'
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = LocalizationConfig(
            min_landmarks=4,
            focal_length=800,
            baseline=0.15,
            stereo_algorithm='BM',
            confidence_threshold=0.6
        )
        
        assert config.min_landmarks == 4
        assert config.focal_length == 800
        assert config.baseline == 0.15
        assert config.stereo_algorithm == 'BM'
        assert config.confidence_threshold == 0.6


class TestLocalizationResult:
    """Test LocalizationResult dataclass."""
    
    def test_success_result(self):
        """Test successful result."""
        result = LocalizationResult(
            success=True,
            x=100.0,
            y=200.0,
            num_landmarks=4,
            detected_landmarks=['l1', 'l2', 'l3', 'l4'],
            residual=0.001
        )
        
        assert result.success
        assert result.x == 100.0
        assert result.y == 200.0
        assert result.num_landmarks == 4
    
    def test_failure_result(self):
        """Test failure result."""
        result = LocalizationResult(
            success=False,
            error_message="Insufficient landmarks detected"
        )
        
        assert not result.success
        assert result.x is None
        assert result.y is None
        assert "Insufficient" in result.error_message
    
    def test_position_property(self):
        """Test position property."""
        result = LocalizationResult(
            success=True,
            x=100.0,
            y=200.0,
            num_landmarks=3,
            detected_landmarks=['l1', 'l2', 'l3'],
            residual=0.001
        )
        
        assert result.position == (100.0, 200.0)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = LocalizationResult(
            success=True,
            x=100.0,
            y=200.0,
            num_landmarks=4,
            detected_landmarks=['l1', 'l2', 'l3', 'l4'],
            residual=0.001
        )
        
        d = result.to_dict()
        
        assert d['success'] == True
        assert d['x'] == 100.0
        assert d['y'] == 200.0
        assert d['num_landmarks'] == 4


class TestLanBLoc:
    """Test LanBLoc class."""
    
    @pytest.fixture
    def landmark_db(self):
        """Create landmark database for tests."""
        return LandmarkDatabase.from_builtin()
    
    @pytest.fixture
    def lanbloc(self, landmark_db):
        """Create LanBLoc instance for tests."""
        return LanBLoc(landmark_db=landmark_db.get_2d_positions())
    
    def test_init_default(self):
        """Test default initialization."""
        db = LandmarkDatabase.from_builtin()
        lanbloc = LanBLoc(landmark_db=db.get_2d_positions())
        
        assert lanbloc is not None
        assert lanbloc.config.min_landmarks == 3
    
    def test_init_with_config(self, landmark_db):
        """Test initialization with custom config."""
        config = LocalizationConfig(
            min_landmarks=4,
            focal_length=800
        )
        
        lanbloc = LanBLoc(
            landmark_db=landmark_db.get_2d_positions(),
            config=config
        )
        
        assert lanbloc.config.min_landmarks == 4
        assert lanbloc.config.focal_length == 800
    
    def test_localize_with_known_distances(self, landmark_db):
        """Test localization with known distances."""
        lanbloc = LanBLoc(landmark_db=landmark_db.get_2d_positions())
        
        # Use trilat1 data: n35 distances to l1, l2, l3
        trilat = TRILATERATION_DATA['trilat1']
        distances_from_n35 = trilat['distanceFromN35To']
        
        landmark_ids = ['l1', 'l2', 'l3']
        distances = [distances_from_n35[lid] for lid in landmark_ids]
        
        result = lanbloc.localize_with_known_distances(landmark_ids, distances)
        
        assert result.success
        assert result.num_landmarks == 3
        assert result.x is not None
        assert result.y is not None
    
    def test_localize_with_known_distances_all_4(self, landmark_db):
        """Test localization with 4 landmarks."""
        lanbloc = LanBLoc(landmark_db=landmark_db.get_2d_positions())
        
        # Use all 4 landmarks from trilat1
        trilat = TRILATERATION_DATA['trilat1']
        distances_from_n35 = trilat['distanceFromN35To']
        
        landmark_ids = ['l1', 'l2', 'l3', 'l4']
        distances = [distances_from_n35[lid] for lid in landmark_ids]
        
        result = lanbloc.localize_with_known_distances(landmark_ids, distances)
        
        assert result.success
        assert result.num_landmarks == 4
    
    def test_localize_with_known_distances_insufficient(self, landmark_db):
        """Test localization with insufficient landmarks."""
        lanbloc = LanBLoc(landmark_db=landmark_db.get_2d_positions())
        
        # Only 2 landmarks (need 3 minimum)
        result = lanbloc.localize_with_known_distances(
            ['l1', 'l2'],
            [43.87, 40.44]
        )
        
        assert not result.success
        assert "Insufficient" in result.error_message or result.num_landmarks < 3
    
    def test_localize_with_unknown_landmarks(self, landmark_db):
        """Test localization with unknown landmarks."""
        lanbloc = LanBLoc(landmark_db=landmark_db.get_2d_positions())
        
        # Include unknown landmark
        result = lanbloc.localize_with_known_distances(
            ['l1', 'l2', 'unknown_landmark'],
            [43.87, 40.44, 50.0]
        )
        
        # Should handle gracefully
        assert isinstance(result, LocalizationResult)


class TestLanBLocWithMultipleTrilats:
    """Test LanBLoc with different trilateration sets."""
    
    @pytest.fixture
    def lanbloc(self):
        """Create LanBLoc instance."""
        db = LandmarkDatabase.from_builtin()
        return LanBLoc(landmark_db=db.get_2d_positions())
    
    def test_trilat2(self, lanbloc):
        """Test with trilat2 data."""
        trilat = TRILATERATION_DATA['trilat2']
        distances = trilat['distanceFromN37To']
        
        landmark_ids = list(distances.keys())[:3]
        dist_values = [distances[lid] for lid in landmark_ids]
        
        result = lanbloc.localize_with_known_distances(landmark_ids, dist_values)
        
        assert result.success
    
    def test_trilat3(self, lanbloc):
        """Test with trilat3 data."""
        trilat = TRILATERATION_DATA['trilat3']
        distances = trilat['distanceFromN41To']
        
        landmark_ids = list(distances.keys())
        dist_values = [distances[lid] for lid in landmark_ids]
        
        result = lanbloc.localize_with_known_distances(landmark_ids, dist_values)
        
        assert result.success
    
    def test_all_trilat_sets(self, lanbloc):
        """Test with all trilateration sets."""
        for trilat_id, trilat in TRILATERATION_DATA.items():
            # Get first node's distances
            node = trilat['nodes'][0]
            dist_key = f"distanceFrom{node.upper()}To"
            distances = trilat[dist_key]
            
            landmark_ids = list(distances.keys())
            dist_values = [distances[lid] for lid in landmark_ids]
            
            result = lanbloc.localize_with_known_distances(landmark_ids, dist_values)
            
            assert result.success, f"Failed for {trilat_id} with node {node}"


class TestConfigFromFile:
    """Test loading configuration from file."""
    
    def test_from_config_default(self):
        """Test loading from default config location."""
        config_path = Path(__file__).parent.parent / 'config' / 'default_config.yaml'
        
        if config_path.exists():
            lanbloc = LanBLoc.from_config(str(config_path))
            assert lanbloc is not None
    
    def test_from_nonexistent_config(self):
        """Test handling of nonexistent config file."""
        with pytest.raises((FileNotFoundError, Exception)):
            LanBLoc.from_config('nonexistent_config.yaml')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
