"""
Pytest configuration and shared fixtures.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_landmark_positions():
    """Sample landmark positions for testing."""
    return {
        'l1': (-155.47, -5021.05),
        'l2': (-155.53, -5021.07),
        'l3': (-155.51, -5021.10),
        'l4': (-155.43, -5021.10)
    }


@pytest.fixture
def sample_distances():
    """Sample distances from node to landmarks."""
    return {
        'l1': 43.87,
        'l2': 40.44,
        'l3': 38.94,
        'l4': 70.90
    }


@pytest.fixture
def sample_stereo_images():
    """Generate sample stereo images for testing."""
    import numpy as np
    
    height, width = 480, 640
    np.random.seed(42)
    
    # Create textured image
    texture = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    
    left = texture.copy()
    right = texture.copy()
    
    # Simulate disparity
    shift = 10
    right[:, shift:] = left[:, :-shift]
    right[:, :shift] = 0
    
    return left, right


@pytest.fixture
def trilat1_data():
    """Trilateration data for trilat1."""
    from lanbloc.data.landmark_db import TRILATERATION_DATA
    return TRILATERATION_DATA['trilat1']


@pytest.fixture
def lanbloc_instance():
    """Create a LanBLoc instance with default settings."""
    from lanbloc import LanBLoc
    from lanbloc.data import LandmarkDatabase
    
    db = LandmarkDatabase.from_builtin()
    return LanBLoc(landmark_db=db.get_2d_positions())
