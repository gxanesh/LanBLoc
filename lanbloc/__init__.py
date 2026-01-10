"""
LanBLoc: Landmark-Based Localization for GPS-Denied Environments
================================================================

A visual localization framework using stereo vision and deep learning-based
landmark recognition for position estimation.

Modules:
    core: Core algorithms (stereo depth, localization, trilateration)
    detection: YOLO-based landmark detection
    calibration: Camera calibration utilities
    utils: Coordinate transformations and visualization
    data: Dataset loading and landmark database

Example:
    >>> from lanbloc import LanBLoc
    >>> from lanbloc.data import LandmarkDatabase
    >>> 
    >>> # Initialize
    >>> db = LandmarkDatabase.from_dataset("data/landmark_stereov1_corrupt")
    >>> lanbloc = LanBLoc(landmark_db=db)
    >>> 
    >>> # Localize
    >>> position = lanbloc.localize(left_image, right_image)
    >>> print(f"Position: ({position.x}, {position.y})")
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your-email@example.com"

from .core.localization import LanBLoc
from .core.stereo_depth import StereoDepth
from .core.trilateration import Trilateration
from .data.landmark_db import LandmarkDatabase
from .data.dataset import StereoDataset

__all__ = [
    "LanBLoc",
    "StereoDepth", 
    "Trilateration",
    "LandmarkDatabase",
    "StereoDataset",
    "__version__",
]
