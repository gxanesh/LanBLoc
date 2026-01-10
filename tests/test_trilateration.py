"""
Unit tests for trilateration module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lanbloc.core.trilateration import (
    TrilaterationSolver,
    TrilaterationResult,
    trilaterate_2d,
    trilaterate_3d
)


class TestTrilaterationSolver:
    """Test cases for TrilaterationSolver class."""
    
    def test_init_default(self):
        """Test default initialization."""
        solver = TrilaterationSolver()
        assert solver.method == 'LBFGS-B'
        assert solver.tol == 1e-9
        assert solver.max_iter == 1000
    
    def test_init_custom(self):
        """Test custom initialization."""
        solver = TrilaterationSolver(method='L-BFGS-B', tol=1e-6, max_iter=500)
        assert solver.method == 'L-BFGS-B'
        assert solver.tol == 1e-6
        assert solver.max_iter == 500
    
    def test_solve_2d_exact(self):
        """Test 2D trilateration with exact distances."""
        solver = TrilaterationSolver()
        
        # Known position at (0, 0)
        landmarks = {
            'A': (3.0, 0.0),
            'B': (0.0, 4.0),
            'C': (-3.0, 0.0)
        }
        
        # Exact distances from origin
        distances = {
            'A': 3.0,
            'B': 4.0,
            'C': 3.0
        }
        
        result = solver.solve(landmarks, distances)
        
        assert result.success
        assert abs(result.x) < 0.1  # Close to 0
        assert abs(result.y) < 0.1  # Close to 0
        assert result.residual < 0.01
    
    def test_solve_2d_known_position(self):
        """Test 2D trilateration with known target position."""
        solver = TrilaterationSolver()
        
        # Target position
        target = (5.0, 5.0)
        
        # Landmarks
        landmarks = {
            'A': (0.0, 0.0),
            'B': (10.0, 0.0),
            'C': (5.0, 10.0)
        }
        
        # Compute exact distances
        distances = {}
        for lid, pos in landmarks.items():
            d = np.sqrt((target[0] - pos[0])**2 + (target[1] - pos[1])**2)
            distances[lid] = d
        
        result = solver.solve(landmarks, distances)
        
        assert result.success
        assert abs(result.x - target[0]) < 0.01
        assert abs(result.y - target[1]) < 0.01
    
    def test_solve_2d_noisy(self):
        """Test 2D trilateration with noisy distances."""
        solver = TrilaterationSolver()
        
        # Target position
        target = (3.0, 4.0)
        
        # Landmarks
        landmarks = {
            'A': (0.0, 0.0),
            'B': (6.0, 0.0),
            'C': (3.0, 8.0),
            'D': (-2.0, 4.0)
        }
        
        # Compute distances with small noise
        np.random.seed(42)
        distances = {}
        for lid, pos in landmarks.items():
            d = np.sqrt((target[0] - pos[0])**2 + (target[1] - pos[1])**2)
            noise = np.random.normal(0, 0.1)  # 10cm noise
            distances[lid] = d + noise
        
        result = solver.solve(landmarks, distances)
        
        assert result.success
        # Allow for larger error due to noise
        assert abs(result.x - target[0]) < 0.5
        assert abs(result.y - target[1]) < 0.5
    
    def test_solve_insufficient_landmarks(self):
        """Test that solver handles insufficient landmarks."""
        solver = TrilaterationSolver()
        
        landmarks = {
            'A': (0.0, 0.0),
            'B': (1.0, 0.0)
        }
        distances = {
            'A': 1.0,
            'B': 1.0
        }
        
        result = solver.solve(landmarks, distances)
        
        # With only 2 landmarks, solution may still be found but less accurate
        # The solver should handle this gracefully
        assert isinstance(result, TrilaterationResult)
    
    def test_solve_3d(self):
        """Test 3D trilateration."""
        solver = TrilaterationSolver()
        
        # Target position in 3D
        target = (1.0, 2.0, 3.0)
        
        # Landmarks in 3D
        landmarks = {
            'A': (0.0, 0.0, 0.0),
            'B': (4.0, 0.0, 0.0),
            'C': (0.0, 4.0, 0.0),
            'D': (0.0, 0.0, 6.0)
        }
        
        # Compute exact distances
        distances = {}
        for lid, pos in landmarks.items():
            d = np.sqrt(sum((t - p)**2 for t, p in zip(target, pos)))
            distances[lid] = d
        
        result = solver.solve(landmarks, distances, is_3d=True)
        
        assert result.success
        assert abs(result.x - target[0]) < 0.01
        assert abs(result.y - target[1]) < 0.01
        assert result.z is not None
        assert abs(result.z - target[2]) < 0.01
    
    def test_result_to_dict(self):
        """Test TrilaterationResult conversion to dict."""
        result = TrilaterationResult(
            x=1.5, y=2.5, z=3.5,
            success=True,
            residual=0.001,
            num_landmarks=4,
            landmarks_used=['A', 'B', 'C', 'D']
        )
        
        d = result.to_dict()
        
        assert d['x'] == 1.5
        assert d['y'] == 2.5
        assert d['z'] == 3.5
        assert d['success'] == True
        assert d['residual'] == 0.001
        assert d['num_landmarks'] == 4


class TestTrilaterateFunctions:
    """Test standalone trilateration functions."""
    
    def test_trilaterate_2d(self):
        """Test trilaterate_2d convenience function."""
        landmarks = {
            'A': (0.0, 0.0),
            'B': (10.0, 0.0),
            'C': (5.0, 8.66)  # Equilateral triangle
        }
        
        # Target at centroid
        target = (5.0, 2.89)
        
        distances = {}
        for lid, pos in landmarks.items():
            distances[lid] = np.sqrt((target[0] - pos[0])**2 + (target[1] - pos[1])**2)
        
        result = trilaterate_2d(landmarks, distances)
        
        assert result.success
        assert abs(result.x - target[0]) < 0.1
        assert abs(result.y - target[1]) < 0.1
    
    def test_trilaterate_3d(self):
        """Test trilaterate_3d convenience function."""
        landmarks = {
            'A': (0.0, 0.0, 0.0),
            'B': (10.0, 0.0, 0.0),
            'C': (0.0, 10.0, 0.0),
            'D': (0.0, 0.0, 10.0)
        }
        
        target = (2.0, 3.0, 4.0)
        
        distances = {}
        for lid, pos in landmarks.items():
            distances[lid] = np.sqrt(sum((t - p)**2 for t, p in zip(target, pos)))
        
        result = trilaterate_3d(landmarks, distances)
        
        assert result.success
        assert abs(result.x - target[0]) < 0.1
        assert abs(result.y - target[1]) < 0.1
        assert result.z is not None
        assert abs(result.z - target[2]) < 0.1


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_collinear_landmarks(self):
        """Test with collinear landmarks (degenerate case)."""
        solver = TrilaterationSolver()
        
        # All landmarks on a line
        landmarks = {
            'A': (0.0, 0.0),
            'B': (5.0, 0.0),
            'C': (10.0, 0.0)
        }
        
        distances = {
            'A': 5.0,
            'B': 2.5,
            'C': 5.0
        }
        
        result = solver.solve(landmarks, distances)
        
        # Should handle gracefully even if solution is not unique
        assert isinstance(result, TrilaterationResult)
    
    def test_inconsistent_distances(self):
        """Test with geometrically inconsistent distances."""
        solver = TrilaterationSolver()
        
        landmarks = {
            'A': (0.0, 0.0),
            'B': (10.0, 0.0),
            'C': (5.0, 10.0)
        }
        
        # Distances that don't correspond to any point
        distances = {
            'A': 1.0,
            'B': 1.0,
            'C': 1.0
        }
        
        result = solver.solve(landmarks, distances)
        
        # Should return a result but with high residual
        assert isinstance(result, TrilaterationResult)
        # Residual should be high due to inconsistency
        assert result.residual > 1.0
    
    def test_empty_input(self):
        """Test with empty inputs."""
        solver = TrilaterationSolver()
        
        result = solver.solve({}, {})
        
        assert not result.success
    
    def test_mismatched_keys(self):
        """Test with mismatched landmark and distance keys."""
        solver = TrilaterationSolver()
        
        landmarks = {
            'A': (0.0, 0.0),
            'B': (10.0, 0.0),
            'C': (5.0, 10.0)
        }
        
        distances = {
            'A': 5.0,
            'X': 5.0,  # Key doesn't exist in landmarks
            'Y': 5.0
        }
        
        result = solver.solve(landmarks, distances)
        
        # Should handle gracefully
        assert isinstance(result, TrilaterationResult)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
