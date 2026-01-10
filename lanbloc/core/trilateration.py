"""
Trilateration Module

This module implements 2D trilateration for position estimation from
distances to known landmarks. It includes both least-squares initial
estimation and LBFGS-B refinement.

The trilateration problem: Given N landmarks with known positions
(x_i, y_i) and measured distances d_i to those landmarks, find the
observer's position (x, y).
"""

import numpy as np
from scipy.optimize import minimize, least_squares
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrilaterationResult:
    """Result from trilateration computation."""
    position: Tuple[float, float]
    residual: float
    num_landmarks: int
    success: bool
    iterations: int = 0
    error_message: Optional[str] = None


class Trilateration:
    """
    2D Trilateration solver for position estimation.
    
    Implements least-squares trilateration with optional LBFGS-B refinement.
    
    Example:
        >>> trilat = Trilateration()
        >>> landmarks = [(0, 0), (10, 0), (5, 10)]
        >>> distances = [5.0, 5.0, 5.0]
        >>> result = trilat.solve(landmarks, distances)
        >>> print(f"Position: {result.position}")
    """
    
    def __init__(
        self,
        method: str = "LBFGS-B",
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
        initial_estimate: str = "centroid"
    ):
        """
        Initialize trilateration solver.
        
        Args:
            method: Optimization method ("LBFGS-B", "Powell", "Nelder-Mead")
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance
            initial_estimate: Method for initial estimate ("centroid", "weighted", "nearest")
        """
        self.method = method
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.initial_estimate = initial_estimate
        
        logger.info(f"Trilateration initialized: method={method}")
    
    def _compute_initial_estimate(
        self,
        landmarks: np.ndarray,
        distances: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute initial position estimate.
        
        Args:
            landmarks: Array of landmark positions (N, 2)
            distances: Array of distances to landmarks (N,)
            
        Returns:
            Initial position estimate (x, y)
        """
        if self.initial_estimate == "weighted":
            # Weight by inverse distance (closer landmarks have more weight)
            weights = 1.0 / (distances + 1e-6)
            weights /= weights.sum()
            x0 = np.sum(landmarks[:, 0] * weights)
            y0 = np.sum(landmarks[:, 1] * weights)
        elif self.initial_estimate == "nearest":
            # Start at nearest landmark
            nearest_idx = np.argmin(distances)
            x0, y0 = landmarks[nearest_idx]
        else:  # centroid
            x0 = np.mean(landmarks[:, 0])
            y0 = np.mean(landmarks[:, 1])
        
        return x0, y0
    
    def least_squares_trilateration(
        self,
        landmarks: np.ndarray,
        distances: np.ndarray
    ) -> Tuple[float, float]:
        """
        Solve trilateration using linear least squares.
        
        This provides the initial estimate for Algorithm 2 (Line 17).
        Uses the linearization approach for trilateration.
        
        Args:
            landmarks: Array of landmark positions (N, 2)
            distances: Array of distances to landmarks (N,)
            
        Returns:
            Position estimate (x, y)
        """
        N = len(landmarks)
        
        if N < 2:
            raise ValueError("At least 2 landmarks required for trilateration")
        
        if N == 2:
            # Special case: 2 landmarks - use geometric solution
            return self._solve_two_landmarks(landmarks, distances)
        
        # Linearized least squares approach
        # For each pair of landmarks (i, j):
        # x_i^2 - x_j^2 + y_i^2 - y_j^2 = d_j^2 - d_i^2 + 2x(x_i - x_j) + 2y(y_i - y_j)
        
        # Reference to first landmark
        x0, y0 = landmarks[0]
        d0 = distances[0]
        
        # Build linear system Ax = b
        A = np.zeros((N - 1, 2))
        b = np.zeros(N - 1)
        
        for i in range(1, N):
            xi, yi = landmarks[i]
            di = distances[i]
            
            A[i - 1, 0] = 2 * (xi - x0)
            A[i - 1, 1] = 2 * (yi - y0)
            b[i - 1] = (d0**2 - di**2) - (x0**2 - xi**2) - (y0**2 - yi**2)
        
        # Solve using least squares
        try:
            result, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            return float(result[0]), float(result[1])
        except np.linalg.LinAlgError:
            # Fall back to centroid if linear algebra fails
            logger.warning("Linear least squares failed, using centroid")
            return self._compute_initial_estimate(landmarks, distances)
    
    def _solve_two_landmarks(
        self,
        landmarks: np.ndarray,
        distances: np.ndarray
    ) -> Tuple[float, float]:
        """
        Solve trilateration with exactly 2 landmarks.
        
        Returns the midpoint of the two intersection points.
        """
        x1, y1 = landmarks[0]
        x2, y2 = landmarks[1]
        d1, d2 = distances
        
        # Distance between landmarks
        D = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if D == 0:
            return float(x1), float(y1)
        
        # Check if circles intersect
        if D > d1 + d2 or D < abs(d1 - d2):
            # No intersection, return weighted midpoint
            t = d1 / (d1 + d2)
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return float(x), float(y)
        
        # Find intersection points
        a = (d1**2 - d2**2 + D**2) / (2 * D)
        h = np.sqrt(max(0, d1**2 - a**2))
        
        # Point on line between landmarks
        px = x1 + a * (x2 - x1) / D
        py = y1 + a * (y2 - y1) / D
        
        # Return midpoint of intersection line
        return float(px), float(py)
    
    def _objective_function(
        self,
        position: np.ndarray,
        landmarks: np.ndarray,
        distances: np.ndarray
    ) -> float:
        """
        Objective function for optimization.
        
        Computes sum of squared errors between measured and predicted distances.
        This corresponds to the objective in Line 18 of Algorithm 2.
        
        Args:
            position: Current position estimate [x, y]
            landmarks: Array of landmark positions (N, 2)
            distances: Array of measured distances (N,)
            
        Returns:
            Sum of squared errors
        """
        x, y = position
        
        # Predicted distances
        predicted = np.sqrt((landmarks[:, 0] - x)**2 + (landmarks[:, 1] - y)**2)
        
        # Sum of squared errors
        return np.sum((predicted - distances)**2)
    
    def _gradient(
        self,
        position: np.ndarray,
        landmarks: np.ndarray,
        distances: np.ndarray
    ) -> np.ndarray:
        """
        Gradient of the objective function.
        
        Args:
            position: Current position estimate [x, y]
            landmarks: Array of landmark positions (N, 2)
            distances: Array of measured distances (N,)
            
        Returns:
            Gradient vector [dE/dx, dE/dy]
        """
        x, y = position
        
        # Predicted distances
        dx = landmarks[:, 0] - x
        dy = landmarks[:, 1] - y
        predicted = np.sqrt(dx**2 + dy**2)
        
        # Avoid division by zero
        predicted = np.maximum(predicted, 1e-10)
        
        # Gradient components
        error = predicted - distances
        grad_x = -2 * np.sum(error * dx / predicted)
        grad_y = -2 * np.sum(error * dy / predicted)
        
        return np.array([grad_x, grad_y])
    
    def solve(
        self,
        landmarks: List[Tuple[float, float]],
        distances: List[float],
        initial_position: Optional[Tuple[float, float]] = None
    ) -> TrilaterationResult:
        """
        Solve trilateration problem.
        
        Implements the position estimation from Algorithm 2 (Lines 17-18).
        
        Args:
            landmarks: List of landmark positions [(x1, y1), (x2, y2), ...]
            distances: List of distances to landmarks [d1, d2, ...]
            initial_position: Optional initial position estimate
            
        Returns:
            TrilaterationResult with estimated position
        """
        # Convert to numpy arrays
        landmarks_arr = np.array(landmarks)
        distances_arr = np.array(distances)
        
        N = len(landmarks)
        
        if N < 3:
            logger.warning(f"Only {N} landmarks provided, result may be unreliable")
        
        try:
            # Step 1: Get initial estimate (Line 17)
            if initial_position is not None:
                x0, y0 = initial_position
            else:
                x0, y0 = self.least_squares_trilateration(landmarks_arr, distances_arr)
            
            logger.debug(f"Initial estimate: ({x0:.2f}, {y0:.2f})")
            
            # Step 2: Refine using optimization (Line 18)
            result = minimize(
                self._objective_function,
                x0=[x0, y0],
                args=(landmarks_arr, distances_arr),
                method=self.method,
                jac=lambda p: self._gradient(p, landmarks_arr, distances_arr),
                options={
                    'maxiter': self.max_iterations,
                    'gtol': self.tolerance
                }
            )
            
            final_x, final_y = result.x
            residual = np.sqrt(result.fun / N)  # RMSE
            
            logger.debug(f"Refined estimate: ({final_x:.2f}, {final_y:.2f}), "
                        f"residual: {residual:.4f}")
            
            return TrilaterationResult(
                position=(float(final_x), float(final_y)),
                residual=float(residual),
                num_landmarks=N,
                success=result.success,
                iterations=result.nit
            )
            
        except Exception as e:
            logger.error(f"Trilateration failed: {e}")
            return TrilaterationResult(
                position=(0.0, 0.0),
                residual=float('inf'),
                num_landmarks=N,
                success=False,
                error_message=str(e)
            )
    
    def solve_3d(
        self,
        landmarks: List[Tuple[float, float, float]],
        distances: List[float],
        initial_position: Optional[Tuple[float, float, float]] = None
    ) -> Tuple[float, float, float]:
        """
        Solve 3D trilateration problem.
        
        Args:
            landmarks: List of 3D landmark positions
            distances: List of distances to landmarks
            initial_position: Optional initial 3D position
            
        Returns:
            Estimated 3D position (x, y, z)
        """
        landmarks_arr = np.array(landmarks)
        distances_arr = np.array(distances)
        
        N = len(landmarks)
        
        # Initial estimate
        if initial_position is not None:
            x0 = np.array(initial_position)
        else:
            x0 = np.mean(landmarks_arr, axis=0)
        
        # 3D objective function
        def objective_3d(pos):
            predicted = np.sqrt(np.sum((landmarks_arr - pos)**2, axis=1))
            return np.sum((predicted - distances_arr)**2)
        
        # Optimize
        result = minimize(
            objective_3d,
            x0=x0,
            method=self.method,
            options={'maxiter': self.max_iterations}
        )
        
        return tuple(result.x)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Trilateration":
        """
        Create Trilateration instance from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configured Trilateration instance
        """
        loc_config = config.get("localization", {})
        
        return cls(
            method=loc_config.get("optimization_method", "LBFGS-B"),
            max_iterations=loc_config.get("max_iterations", 1000),
            tolerance=loc_config.get("tolerance", 1e-8),
            initial_estimate=loc_config.get("initial_estimate", "centroid")
        )


def least_squares_trilateration(
    landmarks: List[Tuple[float, float]],
    distances: List[float]
) -> Tuple[float, float]:
    """
    Convenience function for least-squares trilateration.
    
    This corresponds to LEASTSQUARESTRILATERATION() in Algorithm 2.
    
    Args:
        landmarks: List of landmark positions
        distances: List of distances
        
    Returns:
        Position estimate (x, y)
    """
    solver = Trilateration()
    return solver.least_squares_trilateration(
        np.array(landmarks),
        np.array(distances)
    )
