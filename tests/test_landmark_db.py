"""
Unit tests for landmark database module.
"""

import pytest
import tempfile
import json
import yaml
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lanbloc.data.landmark_db import (
    Landmark,
    LandmarkDatabase,
    NodeDatabase,
    TRILATERATION_DATA,
    get_trilateration_data
)


class TestLandmark:
    """Test Landmark dataclass."""
    
    def test_init_3d(self):
        """Test initialization with 3D position."""
        lm = Landmark(
            landmark_id='l1',
            position_xyz=(100.0, 200.0, 300.0)
        )
        
        assert lm.landmark_id == 'l1'
        assert lm.position_xyz == (100.0, 200.0, 300.0)
        assert lm.x == 100.0
        assert lm.y == 200.0
        assert lm.z == 300.0
    
    def test_position_2d(self):
        """Test 2D position property."""
        lm = Landmark(
            landmark_id='l1',
            position_xyz=(100.0, 200.0, 300.0)
        )
        
        assert lm.position_2d == (100.0, 200.0)
    
    def test_optional_fields(self):
        """Test optional fields."""
        lm = Landmark(
            landmark_id='l1',
            position_xyz=(100.0, 200.0, 300.0),
            position_latlon=(37.9555, -91.7735),
            name='Test Landmark',
            category='building'
        )
        
        assert lm.position_latlon == (37.9555, -91.7735)
        assert lm.name == 'Test Landmark'
        assert lm.category == 'building'


class TestLandmarkDatabase:
    """Test LandmarkDatabase class."""
    
    def test_init_empty(self):
        """Test empty initialization."""
        db = LandmarkDatabase()
        
        assert len(db) == 0
    
    def test_init_with_landmarks(self):
        """Test initialization with landmarks."""
        landmarks = {
            'l1': Landmark('l1', (100.0, 200.0, 300.0)),
            'l2': Landmark('l2', (150.0, 250.0, 350.0))
        }
        
        db = LandmarkDatabase(landmarks)
        
        assert len(db) == 2
        assert 'l1' in db
        assert 'l2' in db
    
    def test_add_landmark_2d(self):
        """Test adding landmark with 2D position."""
        db = LandmarkDatabase()
        
        db.add_landmark('l1', (100.0, 200.0))
        
        assert 'l1' in db
        lm = db.get_landmark('l1')
        assert lm.position_xyz == (100.0, 200.0, 0.0)
    
    def test_add_landmark_3d(self):
        """Test adding landmark with 3D position."""
        db = LandmarkDatabase()
        
        db.add_landmark('l1', (100.0, 200.0, 300.0))
        
        lm = db.get_landmark('l1')
        assert lm.position_xyz == (100.0, 200.0, 300.0)
    
    def test_get_position(self):
        """Test getting landmark position."""
        db = LandmarkDatabase()
        db.add_landmark('l1', (100.0, 200.0, 300.0))
        
        pos = db.get_position('l1')
        
        assert pos == (100.0, 200.0, 300.0)
    
    def test_get_position_nonexistent(self):
        """Test getting position of nonexistent landmark."""
        db = LandmarkDatabase()
        
        pos = db.get_position('nonexistent')
        
        assert pos is None
    
    def test_get_2d_position(self):
        """Test getting 2D position."""
        db = LandmarkDatabase()
        db.add_landmark('l1', (100.0, 200.0, 300.0))
        
        pos = db.get_2d_position('l1')
        
        assert pos == (100.0, 200.0)
    
    def test_get_2d_positions(self):
        """Test getting all 2D positions."""
        db = LandmarkDatabase()
        db.add_landmark('l1', (100.0, 200.0, 300.0))
        db.add_landmark('l2', (150.0, 250.0, 350.0))
        
        positions = db.get_2d_positions()
        
        assert len(positions) == 2
        assert positions['l1'] == (100.0, 200.0)
        assert positions['l2'] == (150.0, 250.0)
    
    def test_get_3d_positions(self):
        """Test getting all 3D positions."""
        db = LandmarkDatabase()
        db.add_landmark('l1', (100.0, 200.0, 300.0))
        db.add_landmark('l2', (150.0, 250.0, 350.0))
        
        positions = db.get_3d_positions()
        
        assert len(positions) == 2
        assert positions['l1'] == (100.0, 200.0, 300.0)
        assert positions['l2'] == (150.0, 250.0, 350.0)
    
    def test_iteration(self):
        """Test database iteration."""
        db = LandmarkDatabase()
        db.add_landmark('l1', (100.0, 200.0, 300.0))
        db.add_landmark('l2', (150.0, 250.0, 350.0))
        
        landmarks = list(db)
        
        assert len(landmarks) == 2
        assert all(isinstance(lm, Landmark) for lm in landmarks)
    
    def test_from_dict(self):
        """Test creating database from dictionary."""
        positions = {
            'l1': (100.0, 200.0),
            'l2': (150.0, 250.0, 350.0)
        }
        
        db = LandmarkDatabase.from_dict(positions)
        
        assert len(db) == 2
        assert db.get_2d_position('l1') == (100.0, 200.0)
    
    def test_from_builtin(self):
        """Test creating database from built-in data."""
        db = LandmarkDatabase.from_builtin()
        
        assert len(db) > 0
        assert 'l1' in db
        assert 'l34' in db
    
    def test_save_load_json(self):
        """Test saving and loading from JSON."""
        db = LandmarkDatabase()
        db.add_landmark('l1', (100.0, 200.0, 300.0))
        db.add_landmark('l2', (150.0, 250.0, 350.0))
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            db.save(temp_path)
            
            loaded_db = LandmarkDatabase.load(temp_path)
            
            assert len(loaded_db) == 2
            assert 'l1' in loaded_db
            assert loaded_db.get_position('l1') == (100.0, 200.0, 300.0)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_save_load_yaml(self):
        """Test saving and loading from YAML."""
        db = LandmarkDatabase()
        db.add_landmark('l1', (100.0, 200.0, 300.0))
        
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            db.save(temp_path)
            
            loaded_db = LandmarkDatabase.load(temp_path)
            
            assert 'l1' in loaded_db
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestNodeDatabase:
    """Test NodeDatabase class."""
    
    def test_init_empty(self):
        """Test empty initialization."""
        db = NodeDatabase()
        
        assert len(db.nodes) == 0
    
    def test_init_with_nodes(self):
        """Test initialization with nodes."""
        nodes = {
            'n35': (100.0, 200.0, 300.0),
            'n36': (150.0, 250.0, 350.0)
        }
        
        db = NodeDatabase(nodes)
        
        assert len(db.nodes) == 2
    
    def test_get_position(self):
        """Test getting node position."""
        db = NodeDatabase({'n35': (100.0, 200.0, 300.0)})
        
        pos = db.get_position('n35')
        
        assert pos == (100.0, 200.0, 300.0)
    
    def test_get_position_nonexistent(self):
        """Test getting position of nonexistent node."""
        db = NodeDatabase()
        
        pos = db.get_position('n99')
        
        assert pos is None
    
    def test_get_2d_position(self):
        """Test getting 2D position."""
        db = NodeDatabase({'n35': (100.0, 200.0, 300.0)})
        
        pos = db.get_2d_position('n35')
        
        assert pos == (100.0, 200.0)
    
    def test_from_builtin(self):
        """Test creating from built-in data."""
        db = NodeDatabase.from_builtin()
        
        assert len(db.nodes) > 0
        assert 'n35' in db.nodes
        assert 'n60' in db.nodes


class TestTrilaterationData:
    """Test trilateration ground truth data."""
    
    def test_data_exists(self):
        """Test that trilateration data exists."""
        assert len(TRILATERATION_DATA) > 0
    
    def test_all_trilat_sets(self):
        """Test that all 10 trilateration sets exist."""
        for i in range(1, 11):
            trilat_id = f'trilat{i}'
            assert trilat_id in TRILATERATION_DATA, f"Missing {trilat_id}"
    
    def test_trilat_structure(self):
        """Test structure of trilateration data."""
        for trilat_id, data in TRILATERATION_DATA.items():
            assert 'nodes' in data, f"{trilat_id} missing 'nodes'"
            assert 'landmarks' in data, f"{trilat_id} missing 'landmarks'"
            
            # Each node should have distances
            for node in data['nodes']:
                dist_key = f"distanceFrom{node.upper()}To"
                assert dist_key in data, f"{trilat_id} missing distances for {node}"
    
    def test_trilat1_details(self):
        """Test specific values in trilat1."""
        trilat1 = TRILATERATION_DATA['trilat1']
        
        assert trilat1['nodes'] == ['n35', 'n36']
        assert trilat1['landmarks'] == ['l1', 'l2', 'l3', 'l4']
        assert trilat1['distanceFromN35To']['l1'] == 43.87
        assert trilat1['distanceFromN36To']['l1'] == 48.7382
    
    def test_get_trilateration_data(self):
        """Test get_trilateration_data function."""
        data = get_trilateration_data()
        
        assert data == TRILATERATION_DATA
    
    def test_landmarks_per_trilat(self):
        """Test that each trilat has at least 3 landmarks."""
        for trilat_id, data in TRILATERATION_DATA.items():
            assert len(data['landmarks']) >= 3, \
                f"{trilat_id} has only {len(data['landmarks'])} landmarks"
    
    def test_distance_values_positive(self):
        """Test that all distance values are positive."""
        for trilat_id, data in TRILATERATION_DATA.items():
            for key, value in data.items():
                if key.startswith('distanceFrom'):
                    for lm, dist in value.items():
                        assert dist > 0, \
                            f"Non-positive distance in {trilat_id}: {key}[{lm}] = {dist}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
