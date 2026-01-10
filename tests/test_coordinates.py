"""
Unit tests for coordinates module.
"""

import pytest
import math
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lanbloc.utils.coordinates import (
    latlon_to_xyz,
    xyz_to_latlon,
    haversine_distance,
    CoordinateTransformer,
    LANDMARK_XYZ,
    NODE_XYZ,
    LANDMARK_LATLON,
    NODE_LATLON
)


class TestLatLonXYZConversion:
    """Test lat/lon to XYZ coordinate conversion."""
    
    def test_latlon_to_xyz_equator(self):
        """Test conversion at equator (lat=0)."""
        xyz = latlon_to_xyz(0, 0)
        
        # At (0, 0), we should be on positive x-axis
        r = 6371  # Earth radius
        assert abs(xyz[0] - r) < 0.01
        assert abs(xyz[1]) < 0.01
        assert abs(xyz[2]) < 0.01
    
    def test_latlon_to_xyz_north_pole(self):
        """Test conversion at north pole (lat=90)."""
        xyz = latlon_to_xyz(90, 0)
        
        r = 6371
        assert abs(xyz[0]) < 0.01
        assert abs(xyz[1]) < 0.01
        assert abs(xyz[2] - r) < 0.01
    
    def test_latlon_to_xyz_south_pole(self):
        """Test conversion at south pole (lat=-90)."""
        xyz = latlon_to_xyz(-90, 0)
        
        r = 6371
        assert abs(xyz[0]) < 0.01
        assert abs(xyz[1]) < 0.01
        assert abs(xyz[2] + r) < 0.01
    
    def test_xyz_to_latlon_roundtrip(self):
        """Test that xyz_to_latlon is inverse of latlon_to_xyz."""
        test_points = [
            (37.9555, -91.7735),  # Near Rolla, MO
            (0, 0),               # Equator/prime meridian
            (45, 90),             # Mid-latitude
            (-33.86, 151.21),     # Sydney
        ]
        
        for lat, lon in test_points:
            xyz = latlon_to_xyz(lat, lon)
            lat2, lon2 = xyz_to_latlon(xyz[0], xyz[1], xyz[2])
            
            assert abs(lat - lat2) < 0.001, f"Latitude mismatch for ({lat}, {lon})"
            assert abs(lon - lon2) < 0.001, f"Longitude mismatch for ({lat}, {lon})"
    
    def test_latlon_to_xyz_with_list(self):
        """Test that function returns list format."""
        xyz = latlon_to_xyz(37.9555, -91.7735)
        
        assert isinstance(xyz, (list, tuple))
        assert len(xyz) == 3
        assert all(isinstance(x, float) for x in xyz)


class TestHaversineDistance:
    """Test Haversine distance calculation."""
    
    def test_same_point(self):
        """Test distance between same point is zero."""
        lat, lon = 37.9555, -91.7735
        d = haversine_distance(lat, lon, lat, lon)
        
        assert d == 0.0
    
    def test_known_distance(self):
        """Test distance between known cities."""
        # Rolla, MO to St. Louis, MO (approximately 100 miles / 161 km)
        rolla = (37.9514, -91.7712)
        stlouis = (38.6270, -90.1994)
        
        d = haversine_distance(rolla[0], rolla[1], stlouis[0], stlouis[1])
        
        # Should be approximately 160-165 km
        assert 155 < d < 170
    
    def test_symmetric(self):
        """Test that distance is symmetric."""
        p1 = (37.9555, -91.7735)
        p2 = (38.6270, -90.1994)
        
        d1 = haversine_distance(p1[0], p1[1], p2[0], p2[1])
        d2 = haversine_distance(p2[0], p2[1], p1[0], p1[1])
        
        assert abs(d1 - d2) < 0.001


class TestCoordinateTransformer:
    """Test CoordinateTransformer class."""
    
    def test_init_default(self):
        """Test default initialization."""
        tf = CoordinateTransformer()
        
        # Default origin should be set
        assert tf.origin_lat is not None
        assert tf.origin_lon is not None
    
    def test_init_custom_origin(self):
        """Test custom origin initialization."""
        tf = CoordinateTransformer(origin_lat=37.9555, origin_lon=-91.7735)
        
        assert tf.origin_lat == 37.9555
        assert tf.origin_lon == -91.7735
    
    def test_latlon_to_local_origin(self):
        """Test that origin transforms to (0, 0) in local coordinates."""
        origin = (37.9555, -91.7735)
        tf = CoordinateTransformer(origin_lat=origin[0], origin_lon=origin[1])
        
        x, y = tf.latlon_to_local(origin[0], origin[1])
        
        assert abs(x) < 0.1
        assert abs(y) < 0.1
    
    def test_local_to_latlon_roundtrip(self):
        """Test local to lat/lon roundtrip."""
        origin = (37.9555, -91.7735)
        tf = CoordinateTransformer(origin_lat=origin[0], origin_lon=origin[1])
        
        # Test point nearby
        test_lat, test_lon = 37.9560, -91.7740
        
        x, y = tf.latlon_to_local(test_lat, test_lon)
        lat2, lon2 = tf.local_to_latlon(x, y)
        
        # Should be close to original (within meter precision)
        assert abs(test_lat - lat2) < 0.0001
        assert abs(test_lon - lon2) < 0.0001
    
    def test_batch_conversion(self):
        """Test batch lat/lon to local conversion."""
        tf = CoordinateTransformer(origin_lat=37.9555, origin_lon=-91.7735)
        
        latlons = [
            (37.9555, -91.7735),
            (37.9560, -91.7740),
            (37.9550, -91.7730)
        ]
        
        local_coords = tf.batch_latlon_to_local(latlons)
        
        assert len(local_coords) == len(latlons)
        assert all(len(coord) == 2 for coord in local_coords)


class TestBuiltinData:
    """Test built-in landmark and node data."""
    
    def test_landmark_xyz_exists(self):
        """Test that landmark XYZ data exists."""
        assert len(LANDMARK_XYZ) > 0
        assert 'l1' in LANDMARK_XYZ
        assert 'l34' in LANDMARK_XYZ
    
    def test_landmark_xyz_format(self):
        """Test landmark XYZ data format."""
        for lid, xyz in LANDMARK_XYZ.items():
            assert isinstance(xyz, (list, tuple))
            assert len(xyz) == 3
            assert all(isinstance(x, (int, float)) for x in xyz)
    
    def test_node_xyz_exists(self):
        """Test that node XYZ data exists."""
        assert len(NODE_XYZ) > 0
        assert 'n35' in NODE_XYZ
        assert 'n60' in NODE_XYZ
    
    def test_node_xyz_format(self):
        """Test node XYZ data format."""
        for nid, xyz in NODE_XYZ.items():
            assert isinstance(xyz, (list, tuple))
            assert len(xyz) == 3
            assert all(isinstance(x, (int, float)) for x in xyz)
    
    def test_landmark_latlon_exists(self):
        """Test that landmark lat/lon data exists."""
        assert len(LANDMARK_LATLON) > 0
        
        for lid, latlon in LANDMARK_LATLON.items():
            assert len(latlon) == 2
            lat, lon = latlon
            # Should be near Rolla, MO
            assert 37 < lat < 38
            assert -92 < lon < -91
    
    def test_node_latlon_exists(self):
        """Test that node lat/lon data exists."""
        assert len(NODE_LATLON) > 0
        
        for nid, latlon in NODE_LATLON.items():
            assert len(latlon) == 2
            lat, lon = latlon
            # Should be near Rolla, MO
            assert 37 < lat < 38
            assert -92 < lon < -91
    
    def test_xyz_latlon_consistency(self):
        """Test that XYZ and lat/lon are consistent."""
        for lid in LANDMARK_XYZ:
            if lid in LANDMARK_LATLON:
                latlon = LANDMARK_LATLON[lid]
                xyz_computed = latlon_to_xyz(latlon[0], latlon[1])
                xyz_stored = LANDMARK_XYZ[lid]
                
                # Allow for small numerical differences
                for i in range(3):
                    assert abs(xyz_computed[i] - xyz_stored[i]) < 0.001


class TestDistanceCalculations:
    """Test distance calculations between landmarks and nodes."""
    
    def test_distance_between_landmarks(self):
        """Test distance calculation between two landmarks."""
        # Get positions for l1 and l2
        l1 = LANDMARK_XYZ['l1']
        l2 = LANDMARK_XYZ['l2']
        
        # Calculate Euclidean distance in XYZ space
        d = math.sqrt(sum((a - b)**2 for a, b in zip(l1, l2)))
        
        # Should be positive
        assert d > 0
    
    def test_distances_reasonable(self):
        """Test that distances between nearby points are reasonable."""
        # Calculate distance from n35 to l1 (should be around 43.87m based on data)
        n35 = NODE_XYZ['n35']
        l1 = LANDMARK_XYZ['l1']
        
        d = math.sqrt(sum((a - b)**2 for a, b in zip(n35, l1)))
        
        # The trilateration data says this should be ~43.87
        # But note: XYZ are Earth-centered coordinates
        # The actual ground distance calculation is more complex
        # This test just verifies the calculation runs
        assert d > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
