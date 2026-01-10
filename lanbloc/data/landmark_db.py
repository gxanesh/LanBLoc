"""
Landmark Database Module

Manages known landmark positions for the LanBLoc localization system.
"""

import json
import yaml
import numpy as np
from typing import Dict, Tuple, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
import logging

from ..utils.coordinates import latlon_to_xyz, LANDMARK_XYZ, NODE_XYZ

logger = logging.getLogger(__name__)


@dataclass
class Landmark:
    """
    A known landmark with position and metadata.
    
    Attributes:
        landmark_id: Unique identifier (e.g., 'l1', 'l2')
        position_xyz: 3D Cartesian position (x, y, z)
        position_latlon: Optional geographic position (lat, lon)
        name: Optional human-readable name
        category: Optional landmark category
    """
    landmark_id: str
    position_xyz: Tuple[float, float, float]
    position_latlon: Optional[Tuple[float, float]] = None
    name: Optional[str] = None
    category: Optional[str] = None
    
    @property
    def x(self) -> float:
        return self.position_xyz[0]
    
    @property
    def y(self) -> float:
        return self.position_xyz[1]
    
    @property
    def z(self) -> float:
        return self.position_xyz[2]
    
    @property
    def position_2d(self) -> Tuple[float, float]:
        """Return 2D position (x, y)."""
        return (self.position_xyz[0], self.position_xyz[1])


class LandmarkDatabase:
    """
    Database of known landmarks for localization.
    
    The database maps landmark IDs to their known positions. Supports
    both 2D and 3D localization.
    
    Attributes:
        landmarks: Dictionary mapping IDs to Landmark objects
    
    Example:
        >>> db = LandmarkDatabase.from_dataset("data/landmark_stereov1_corrupt")
        >>> pos = db.get_position("l1")
        >>> print(f"Landmark l1 at {pos}")
    """
    
    def __init__(self, landmarks: Optional[Dict[str, Landmark]] = None):
        """
        Initialize landmark database.
        
        Args:
            landmarks: Optional dictionary of landmarks
        """
        self.landmarks: Dict[str, Landmark] = landmarks or {}
        
        logger.info(f"LandmarkDatabase initialized with {len(self.landmarks)} landmarks")
    
    def add_landmark(
        self,
        landmark_id: str,
        position: Union[Tuple[float, float], Tuple[float, float, float]],
        **kwargs
    ) -> None:
        """
        Add a landmark to the database.
        
        Args:
            landmark_id: Unique identifier
            position: Position as (x, y) or (x, y, z)
            **kwargs: Additional landmark attributes
        """
        if len(position) == 2:
            position_xyz = (position[0], position[1], 0.0)
        else:
            position_xyz = position
        
        self.landmarks[landmark_id] = Landmark(
            landmark_id=landmark_id,
            position_xyz=position_xyz,
            **kwargs
        )
    
    def get_landmark(self, landmark_id: str) -> Optional[Landmark]:
        """Get landmark by ID."""
        return self.landmarks.get(landmark_id)
    
    def get_position(self, landmark_id: str) -> Optional[Tuple[float, float, float]]:
        """Get landmark 3D position."""
        landmark = self.landmarks.get(landmark_id)
        return landmark.position_xyz if landmark else None
    
    def get_2d_position(self, landmark_id: str) -> Optional[Tuple[float, float]]:
        """Get landmark 2D position (x, y)."""
        landmark = self.landmarks.get(landmark_id)
        return landmark.position_2d if landmark else None
    
    def get_2d_positions(self) -> Dict[str, Tuple[float, float]]:
        """Get all landmark 2D positions."""
        return {lid: lm.position_2d for lid, lm in self.landmarks.items()}
    
    def get_3d_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """Get all landmark 3D positions."""
        return {lid: lm.position_xyz for lid, lm in self.landmarks.items()}
    
    def get_landmarks_for_trilat(
        self,
        trilat_id: str,
        trilat_data: Dict[str, Any]
    ) -> List[str]:
        """
        Get landmark IDs for a trilateration set.
        
        Args:
            trilat_id: Trilateration set ID (e.g., 'trilat1')
            trilat_data: Trilateration data dictionary
            
        Returns:
            List of landmark IDs
        """
        if trilat_id not in trilat_data:
            return []
        return trilat_data[trilat_id].get('landmarks', [])
    
    def __len__(self) -> int:
        return len(self.landmarks)
    
    def __contains__(self, landmark_id: str) -> bool:
        return landmark_id in self.landmarks
    
    def __iter__(self):
        return iter(self.landmarks.values())
    
    def save(self, path: str) -> None:
        """
        Save database to file.
        
        Args:
            path: Output file path (.json or .yaml)
        """
        data = {
            lid: {
                'position_xyz': lm.position_xyz,
                'position_latlon': lm.position_latlon,
                'name': lm.name,
                'category': lm.category
            }
            for lid, lm in self.landmarks.items()
        }
        
        path = Path(path)
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            with open(path, 'w') as f:
                yaml.safe_dump(data, f)
        
        logger.info(f"Database saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "LandmarkDatabase":
        """
        Load database from file.
        
        Args:
            path: Input file path (.json or .yaml)
            
        Returns:
            LandmarkDatabase instance
        """
        path = Path(path)
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
        else:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
        
        landmarks = {}
        for lid, ldata in data.items():
            landmarks[lid] = Landmark(
                landmark_id=lid,
                position_xyz=tuple(ldata['position_xyz']),
                position_latlon=tuple(ldata['position_latlon']) if ldata.get('position_latlon') else None,
                name=ldata.get('name'),
                category=ldata.get('category')
            )
        
        return cls(landmarks)
    
    @classmethod
    def from_dict(
        cls,
        positions: Dict[str, Tuple[float, ...]],
        latlon_positions: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> "LandmarkDatabase":
        """
        Create database from position dictionary.
        
        Args:
            positions: Dictionary mapping IDs to (x, y) or (x, y, z) positions
            latlon_positions: Optional dictionary of lat/lon positions
            
        Returns:
            LandmarkDatabase instance
        """
        landmarks = {}
        for lid, pos in positions.items():
            if len(pos) == 2:
                pos_xyz = (pos[0], pos[1], 0.0)
            else:
                pos_xyz = pos
            
            pos_latlon = latlon_positions.get(lid) if latlon_positions else None
            
            landmarks[lid] = Landmark(
                landmark_id=lid,
                position_xyz=pos_xyz,
                position_latlon=pos_latlon
            )
        
        return cls(landmarks)
    
    @classmethod
    def from_dataset(cls, root_dir: str) -> "LandmarkDatabase":
        """
        Create database from dataset directory.
        
        Reads landmark positions from the dataset structure.
        Falls back to built-in coordinates if files not found.
        
        Args:
            root_dir: Root directory of the dataset
            
        Returns:
            LandmarkDatabase instance
        """
        root = Path(root_dir)
        
        # Try to read from location files in dataset
        landmarks = {}
        
        for trilat_dir in root.glob("trilat*"):
            for landmark_dir in trilat_dir.glob("landmark*"):
                location_file = landmark_dir / "location"
                
                if location_file.exists():
                    try:
                        with open(location_file, 'r') as f:
                            content = f.read().strip()
                            # Parse location (format depends on your data)
                            # Assuming format: "lat,lon" or "x,y,z"
                            parts = [float(x) for x in content.split(',')]
                            
                            # Extract landmark ID from directory name
                            lid = landmark_dir.name.replace('landmark', 'l')
                            
                            if len(parts) == 2:
                                # lat, lon - convert to xyz
                                xyz = latlon_to_xyz(parts[0], parts[1])
                                latlon = tuple(parts)
                            else:
                                xyz = tuple(parts)
                                latlon = None
                            
                            landmarks[lid] = Landmark(
                                landmark_id=lid,
                                position_xyz=xyz,
                                position_latlon=latlon
                            )
                    except Exception as e:
                        logger.warning(f"Could not read {location_file}: {e}")
        
        # If no landmarks found, use built-in coordinates
        if not landmarks:
            logger.info("Using built-in landmark coordinates")
            return cls.from_builtin()
        
        return cls(landmarks)
    
    @classmethod
    def from_builtin(cls) -> "LandmarkDatabase":
        """
        Create database from built-in landmark coordinates.
        
        Uses the pre-computed XYZ coordinates from the trilateration data.
        
        Returns:
            LandmarkDatabase instance
        """
        landmarks = {}
        
        for lid, xyz in LANDMARK_XYZ.items():
            landmarks[lid] = Landmark(
                landmark_id=lid,
                position_xyz=xyz
            )
        
        return cls(landmarks)


class NodeDatabase:
    """
    Database of node (observation point) positions for evaluation.
    
    Similar to LandmarkDatabase but for node ground truth positions.
    """
    
    def __init__(self, nodes: Optional[Dict[str, Tuple[float, float, float]]] = None):
        """
        Initialize node database.
        
        Args:
            nodes: Dictionary mapping node IDs to positions
        """
        self.nodes = nodes or {}
    
    def get_position(self, node_id: str) -> Optional[Tuple[float, float, float]]:
        """Get node position."""
        return self.nodes.get(node_id)
    
    def get_2d_position(self, node_id: str) -> Optional[Tuple[float, float]]:
        """Get node 2D position."""
        pos = self.nodes.get(node_id)
        return (pos[0], pos[1]) if pos else None
    
    @classmethod
    def from_builtin(cls) -> "NodeDatabase":
        """Create from built-in node coordinates."""
        return cls(NODE_XYZ)
    
    @classmethod
    def from_dataset(cls, root_dir: str) -> "NodeDatabase":
        """Load node positions from dataset."""
        root = Path(root_dir)
        nodes = {}
        
        # Look for node_location files
        for node_loc_file in root.rglob("node_location"):
            try:
                with open(node_loc_file, 'r') as f:
                    # Parse node locations
                    # Format depends on your data structure
                    pass
            except Exception as e:
                logger.warning(f"Could not read {node_loc_file}: {e}")
        
        if not nodes:
            logger.info("Using built-in node coordinates")
            return cls.from_builtin()
        
        return cls(nodes)


# Trilateration data from the original dataset
TRILATERATION_DATA = {
    'trilat1': {
        'nodes': ['n35', 'n36'],
        'landmarks': ['l1', 'l2', 'l3', 'l4'],
        'distanceFromN35To': {'l1': 43.87, 'l2': 40.4416, 'l3': 38.941, 'l4': 70.9046},
        'distanceFromN36To': {'l1': 48.7382, 'l2': 47.2863, 'l3': 35.9235, 'l4': 63.469}
    },
    'trilat2': {
        'nodes': ['n37', 'n38', 'n39', 'n40'],
        'landmarks': ['l5', 'l6', 'l7', 'l8'],
        'distanceFromN37To': {'l5': 56.7951, 'l6': 41.0379, 'l7': 77.9306, 'l8': 33.9431},
        'distanceFromN38To': {'l5': 45.4514, 'l6': 45.7008, 'l7': 71.5641, 'l8': 29.0393},
        'distanceFromN39To': {'l5': 42.2358, 'l6': 49.4733, 'l7': 46.093, 'l8': 49.5754},
        'distanceFromN40To': {'l5': 37.3341, 'l6': 66.2053, 'l7': 45.7337, 'l8': 57.5021}
    },
    'trilat3': {
        'nodes': ['n41', 'n42'],
        'landmarks': ['l9', 'l10', 'l11'],
        'distanceFromN41To': {'l9': 301.0519, 'l10': 404.2565, 'l11': 346.0815},
        'distanceFromN42To': {'l9': 313.4865, 'l10': 415.1959, 'l11': 356.9422}
    },
    'trilat4': {
        'nodes': ['n43', 'n44'],
        'landmarks': ['l12', 'l13', 'l14', 'l15'],
        'distanceFromN43To': {'l12': 345.4517, 'l13': 309.4366, 'l14': 361.3245, 'l15': 383.0193},
        'distanceFromN44To': {'l12': 358.3242, 'l13': 332.5309, 'l14': 383.1429, 'l15': 405.6193}
    },
    'trilat5': {
        'nodes': ['n45', 'n46'],
        'landmarks': ['l16', 'l17', 'l18'],
        'distanceFromN45To': {'l16': 378.3449, 'l17': 411.6999, 'l18': 331.956},
        'distanceFromN46To': {'l16': 369.3734, 'l17': 390.5518, 'l18': 310.5561}
    },
    'trilat6': {
        'nodes': ['n47', 'n48', 'n49'],
        'landmarks': ['l19', 'l20', 'l21', 'l22'],
        'distanceFromN47To': {'l19': 331.9468, 'l20': 419.645, 'l21': 342.2365, 'l22': 340.2365},
        'distanceFromN48To': {'l19': 226.1938, 'l20': 312.8568, 'l21': 235.66, 'l22': 235.66},
        'distanceFromN49To': {'l19': 220.8149, 'l20': 303.4425, 'l21': 228.6897, 'l22': 228.6897}
    },
    'trilat7': {
        'nodes': ['n50', 'n51'],
        'landmarks': ['l23', 'l24', 'l25'],
        'distanceFromN50To': {'l23': 216.4899, 'l24': 203.3214, 'l25': 238.2552},
        'distanceFromN51To': {'l23': 241.5702, 'l24': 228.8935, 'l25': 263.8305}
    },
    'trilat8': {
        'nodes': ['n52', 'n53', 'n54'],
        'landmarks': ['l26', 'l27', 'l28'],
        'distanceFromN52To': {'l26': 72.3674, 'l27': 117.6282, 'l28': 94.8749},
        'distanceFromN53To': {'l26': 188.5074, 'l27': 139.0653, 'l28': 153.9165},
        'distanceFromN54To': {'l26': 180.8588, 'l27': 131.1457, 'l28': 148.1084}
    },
    'trilat9': {
        'nodes': ['n55', 'n56', 'n57'],
        'landmarks': ['l29', 'l30', 'l31'],
        'distanceFromN55To': {'l29': 285.6226, 'l30': 228.5951, 'l31': 202.8682},
        'distanceFromN56To': {'l29': 281.6801, 'l30': 222.4362, 'l31': 197.3177},
        'distanceFromN57To': {'l29': 277.3108, 'l30': 221.0898, 'l31': 195.0798}
    },
    'trilat10': {
        'nodes': ['n58', 'n59', 'n60'],
        'landmarks': ['l32', 'l33', 'l34'],
        'distanceFromN58To': {'l32': 186.6573, 'l33': 208.3273, 'l34': 139.8556},
        'distanceFromN59To': {'l32': 205.3355, 'l33': 227.6478, 'l34': 158.8327},
        'distanceFromN60To': {'l32': 214.5594, 'l33': 239.4777, 'l34': 168.9824}
    }
}


def get_trilateration_data() -> Dict[str, Any]:
    """Get the full trilateration data dictionary."""
    return TRILATERATION_DATA
