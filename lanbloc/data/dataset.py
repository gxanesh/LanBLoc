"""
Dataset loading utilities for LanBLoc.

This module provides dataset classes for loading stereo image pairs
from the landmark_stereov1_corrupt dataset structure.

Dataset Structure:
    landmark_stereov1_corrupt/
    ├── trilat1/ ... trilat10/           # 10 trilateration sets
    │   ├── landmark1/ ... landmarkN/
    │   │   ├── location                 # Landmark coordinates
    │   │   └── stereo_images/
    │   │       ├── node_location        # Ground truth node positions
    │   │       ├── stereoL/
    │   │       │   └── <node>_<seq>.png # e.g., 1_img1.png
    │   │       └── stereoR/
    │   │           └── <node>_<seq>.png

Image naming convention: <node_position>_<sequence>.png
    - prefix indicates observation point (node)
    - stereoL and stereoR with same name form a stereo pair
"""

import os
import re
import glob
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterator, Any
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


logger = logging.getLogger(__name__)


@dataclass
class StereoImagePair:
    """A stereo image pair with metadata."""
    left_path: str
    right_path: str
    node_id: str
    sequence: str
    landmark_id: str
    trilat_id: str
    left_image: Optional[np.ndarray] = None
    right_image: Optional[np.ndarray] = None
    
    def load_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load stereo images from disk."""
        if not HAS_CV2:
            raise ImportError("OpenCV is required to load images")
        
        self.left_image = cv2.imread(self.left_path)
        self.right_image = cv2.imread(self.right_path)
        
        if self.left_image is None:
            raise FileNotFoundError(f"Could not load left image: {self.left_path}")
        if self.right_image is None:
            raise FileNotFoundError(f"Could not load right image: {self.right_path}")
            
        return self.left_image, self.right_image
    
    def load_images_gray(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load stereo images as grayscale."""
        if not HAS_CV2:
            raise ImportError("OpenCV is required to load images")
        
        self.left_image = cv2.imread(self.left_path, cv2.IMREAD_GRAYSCALE)
        self.right_image = cv2.imread(self.right_path, cv2.IMREAD_GRAYSCALE)
        
        if self.left_image is None:
            raise FileNotFoundError(f"Could not load left image: {self.left_path}")
        if self.right_image is None:
            raise FileNotFoundError(f"Could not load right image: {self.right_path}")
            
        return self.left_image, self.right_image


@dataclass
class LandmarkData:
    """Data for a single landmark in a trilateration set."""
    landmark_id: str
    trilat_id: str
    landmark_dir: str
    location_file: Optional[str] = None
    location: Optional[Tuple[float, float]] = None  # lat, lon or x, y
    stereo_pairs: List[StereoImagePair] = field(default_factory=list)
    node_locations: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    def get_pairs_by_node(self, node_id: str) -> List[StereoImagePair]:
        """Get all stereo pairs captured from a specific node."""
        return [p for p in self.stereo_pairs if p.node_id == node_id]
    
    def get_unique_nodes(self) -> List[str]:
        """Get list of unique node IDs that captured this landmark."""
        return list(set(p.node_id for p in self.stereo_pairs))


@dataclass
class TrilatData:
    """Data for a trilateration set."""
    trilat_id: str
    trilat_dir: str
    landmarks: Dict[str, LandmarkData] = field(default_factory=dict)
    ground_truth_distances: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def get_all_stereo_pairs(self) -> List[StereoImagePair]:
        """Get all stereo pairs across all landmarks."""
        pairs = []
        for landmark in self.landmarks.values():
            pairs.extend(landmark.stereo_pairs)
        return pairs
    
    def get_pairs_by_node(self, node_id: str) -> Dict[str, List[StereoImagePair]]:
        """Get stereo pairs for each landmark from a specific node."""
        result = {}
        for lm_id, landmark in self.landmarks.items():
            pairs = landmark.get_pairs_by_node(node_id)
            if pairs:
                result[lm_id] = pairs
        return result
    
    def get_unique_nodes(self) -> List[str]:
        """Get all unique node IDs across all landmarks."""
        nodes = set()
        for landmark in self.landmarks.values():
            nodes.update(landmark.get_unique_nodes())
        return sorted(nodes)


class StereoDataset:
    """
    Dataset loader for stereo image pairs from the landmark_stereov1_corrupt dataset.
    
    This class handles loading individual stereo image pairs and their metadata.
    
    Example:
        # >>> dataset = StereoDataset('/path/to/landmark_stereov1_corrupt')
        # >>> for pair in dataset:
        # ...     left, right = pair.load_images()
        # ...     print(f"Processing {pair.landmark_id} from node {pair.node_id}")
    """
    
    def __init__(self, root_dir: str, trilat_ids: Optional[List[str]] = None):
        """
        Initialize the stereo dataset.
        
        Args:
            root_dir: Root directory of the landmark_stereov1_corrupt dataset
            trilat_ids: Optional list of trilateration set IDs to load (e.g., ['trilat1', 'trilat2'])
                       If None, loads all available trilateration sets.
        """
        self.root_dir = Path(root_dir)
        self.trilat_ids = trilat_ids
        self.stereo_pairs: List[StereoImagePair] = []
        
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root directory not found: {root_dir}")
        
        self._scan_dataset()
    
    def _parse_image_filename(self, filename: str) -> Tuple[str, str]:
        """
        Parse image filename to extract node ID and sequence number.
        
        Filename format: <node_position>_<sequence>.png
        Examples: 1_img1.png, 2_img3.png
        
        Returns:
            Tuple of (node_id, sequence)
        """
        name = Path(filename).stem  # Remove extension
        
        # Try pattern: <number>_<sequence>
        match = re.match(r'^(\d+)_(.+)$', name)
        if match:
            return match.group(1), match.group(2)
        
        # Fallback: use whole name as node_id
        return name, '1'
    
    def _scan_dataset(self):
        """Scan the dataset directory and build the list of stereo pairs."""
        self.stereo_pairs = []
        
        # Find trilateration directories
        trilat_pattern = self.root_dir / 'trilat*'
        trilat_dirs = sorted(glob.glob(str(trilat_pattern)))
        
        if not trilat_dirs:
            # Maybe the root is a single trilat set
            if (self.root_dir / 'landmark1').exists():
                trilat_dirs = [str(self.root_dir)]
        
        for trilat_dir in trilat_dirs:
            trilat_path = Path(trilat_dir)
            trilat_id = trilat_path.name
            
            # Filter by trilat_ids if specified
            if self.trilat_ids and trilat_id not in self.trilat_ids:
                continue
            
            # Find landmark directories
            landmark_dirs = sorted(glob.glob(str(trilat_path / 'landmark*')))
            
            for landmark_dir in landmark_dirs:
                landmark_path = Path(landmark_dir)
                landmark_id = landmark_path.name
                
                # Find stereo image directories
                stereo_images_dir = landmark_path / 'stereo_images'
                if not stereo_images_dir.exists():
                    # Try without stereo_images subdirectory
                    stereo_l_dir = landmark_path / 'stereoL'
                    stereo_r_dir = landmark_path / 'stereoR'
                else:
                    stereo_l_dir = stereo_images_dir / 'stereoL'
                    stereo_r_dir = stereo_images_dir / 'stereoR'
                
                if not stereo_l_dir.exists() or not stereo_r_dir.exists():
                    logger.warning(f"Missing stereo directories in {landmark_dir}")
                    continue
                
                # Find left images and match with right
                left_images = glob.glob(str(stereo_l_dir / '*.png'))
                left_images.extend(glob.glob(str(stereo_l_dir / '*.jpg')))
                left_images.extend(glob.glob(str(stereo_l_dir / '*.jpeg')))
                
                for left_path in sorted(left_images):
                    filename = Path(left_path).name
                    right_path = stereo_r_dir / filename
                    
                    if not right_path.exists():
                        logger.warning(f"Missing right stereo pair for: {left_path}")
                        continue
                    
                    node_id, sequence = self._parse_image_filename(filename)
                    
                    pair = StereoImagePair(
                        left_path=str(left_path),
                        right_path=str(right_path),
                        node_id=node_id,
                        sequence=sequence,
                        landmark_id=landmark_id,
                        trilat_id=trilat_id
                    )
                    self.stereo_pairs.append(pair)
        
        logger.info(f"Found {len(self.stereo_pairs)} stereo pairs in dataset")
    
    def __len__(self) -> int:
        return len(self.stereo_pairs)
    
    def __getitem__(self, idx: int) -> StereoImagePair:
        return self.stereo_pairs[idx]
    
    def __iter__(self) -> Iterator[StereoImagePair]:
        return iter(self.stereo_pairs)
    
    def get_by_trilat(self, trilat_id: str) -> List[StereoImagePair]:
        """Get all stereo pairs for a specific trilateration set."""
        return [p for p in self.stereo_pairs if p.trilat_id == trilat_id]
    
    def get_by_landmark(self, landmark_id: str, trilat_id: Optional[str] = None) -> List[StereoImagePair]:
        """Get all stereo pairs for a specific landmark."""
        pairs = [p for p in self.stereo_pairs if p.landmark_id == landmark_id]
        if trilat_id:
            pairs = [p for p in pairs if p.trilat_id == trilat_id]
        return pairs
    
    def get_by_node(self, node_id: str) -> List[StereoImagePair]:
        """Get all stereo pairs captured from a specific node position."""
        return [p for p in self.stereo_pairs if p.node_id == node_id]
    
    def get_unique_trilats(self) -> List[str]:
        """Get list of unique trilateration set IDs."""
        return sorted(set(p.trilat_id for p in self.stereo_pairs))
    
    def get_unique_landmarks(self, trilat_id: Optional[str] = None) -> List[str]:
        """Get list of unique landmark IDs."""
        pairs = self.stereo_pairs
        if trilat_id:
            pairs = [p for p in pairs if p.trilat_id == trilat_id]
        return sorted(set(p.landmark_id for p in pairs))
    
    def get_unique_nodes(self, trilat_id: Optional[str] = None) -> List[str]:
        """Get list of unique node IDs."""
        pairs = self.stereo_pairs
        if trilat_id:
            pairs = [p for p in pairs if p.trilat_id == trilat_id]
        return sorted(set(p.node_id for p in pairs))


class TrilatDataset:
    """
    Dataset loader organized by trilateration sets.
    
    This class provides a hierarchical view of the dataset organized
    by trilateration sets, landmarks, and stereo pairs.
    
    Example:
        # >>> dataset = TrilatDataset('/path/to/landmark_stereov1_corrupt')
        # >>> for trilat in dataset:
        # ...     print(f"Trilat {trilat.trilat_id}: {len(trilat.landmarks)} landmarks")
        # ...     for lm_id, landmark in trilat.landmarks.items():
        # ...         print(f"  {lm_id}: {len(landmark.stereo_pairs)} pairs")
    """
    
    def __init__(self, root_dir: str, trilat_ids: Optional[List[str]] = None,
                 trilateration_data: Optional[Dict] = None):
        """
        Initialize the trilateration dataset.
        
        Args:
            root_dir: Root directory of the landmark_stereov1_corrupt dataset
            trilat_ids: Optional list of trilateration set IDs to load
            trilateration_data: Optional dictionary with ground truth distances
                               (from data_trilateration.py format)
        """
        self.root_dir = Path(root_dir)
        self.trilat_ids = trilat_ids
        self.trilateration_data = trilateration_data or {}
        self.trilat_sets: Dict[str, TrilatData] = {}
        
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root directory not found: {root_dir}")
        
        self._scan_dataset()
    
    def _read_location_file(self, filepath: str) -> Optional[Tuple[float, float]]:
        """Read location from a location file."""
        try:
            with open(filepath, 'r') as f:
                content = f.read().strip()
                # Try parsing as "lat, lon" or "lat lon"
                parts = re.split(r'[,\s]+', content)
                if len(parts) >= 2:
                    return float(parts[0]), float(parts[1])
        except Exception as e:
            logger.warning(f"Could not read location file {filepath}: {e}")
        return None
    
    def _read_node_location_file(self, filepath: str) -> Dict[str, Tuple[float, float]]:
        """Read node locations from a node_location file."""
        locations = {}
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = re.split(r'[,\s:]+', line)
                    if len(parts) >= 3:
                        node_id = parts[0]
                        lat, lon = float(parts[1]), float(parts[2])
                        locations[node_id] = (lat, lon)
        except Exception as e:
            logger.warning(f"Could not read node location file {filepath}: {e}")
        return locations
    
    def _parse_image_filename(self, filename: str) -> Tuple[str, str]:
        """Parse image filename to extract node ID and sequence number."""
        name = Path(filename).stem
        match = re.match(r'^(\d+)_(.+)$', name)
        if match:
            return match.group(1), match.group(2)
        return name, '1'
    
    def _scan_dataset(self):
        """Scan the dataset directory and build the hierarchical structure."""
        self.trilat_sets = {}
        
        # Find trilateration directories
        trilat_pattern = self.root_dir / 'trilat*'
        trilat_dirs = sorted(glob.glob(str(trilat_pattern)))
        
        if not trilat_dirs:
            logger.warning(f"No trilateration directories found in {self.root_dir}")
            return
        
        for trilat_dir in trilat_dirs:
            trilat_path = Path(trilat_dir)
            trilat_id = trilat_path.name
            
            # Filter by trilat_ids if specified
            if self.trilat_ids and trilat_id not in self.trilat_ids:
                continue
            
            trilat_data = TrilatData(
                trilat_id=trilat_id,
                trilat_dir=str(trilat_path)
            )
            
            # Load ground truth distances if available
            if trilat_id in self.trilateration_data:
                gt_data = self.trilateration_data[trilat_id]
                # Extract distances from format like 'distanceFromN35To'
                for key, distances in gt_data.items():
                    if key.startswith('distanceFrom'):
                        node_id = key.replace('distanceFrom', '').replace('To', '').lower()
                        trilat_data.ground_truth_distances[node_id] = distances
            
            # Find landmark directories
            landmark_dirs = sorted(glob.glob(str(trilat_path / 'landmark*')))
            
            for landmark_dir in landmark_dirs:
                landmark_path = Path(landmark_dir)
                landmark_id = landmark_path.name
                
                landmark_data = LandmarkData(
                    landmark_id=landmark_id,
                    trilat_id=trilat_id,
                    landmark_dir=str(landmark_path)
                )
                
                # Check for location file
                location_file = landmark_path / 'location'
                if location_file.exists():
                    landmark_data.location_file = str(location_file)
                    landmark_data.location = self._read_location_file(str(location_file))
                
                # Find stereo image directories
                stereo_images_dir = landmark_path / 'stereo_images'
                if stereo_images_dir.exists():
                    stereo_l_dir = stereo_images_dir / 'stereoL'
                    stereo_r_dir = stereo_images_dir / 'stereoR'
                    
                    # Check for node_location file
                    node_loc_file = stereo_images_dir / 'node_location'
                    if node_loc_file.exists():
                        landmark_data.node_locations = self._read_node_location_file(str(node_loc_file))
                else:
                    stereo_l_dir = landmark_path / 'stereoL'
                    stereo_r_dir = landmark_path / 'stereoR'
                
                if not stereo_l_dir.exists() or not stereo_r_dir.exists():
                    logger.warning(f"Missing stereo directories in {landmark_dir}")
                    trilat_data.landmarks[landmark_id] = landmark_data
                    continue
                
                # Find and match stereo pairs
                left_images = glob.glob(str(stereo_l_dir / '*.png'))
                left_images.extend(glob.glob(str(stereo_l_dir / '*.jpg')))
                left_images.extend(glob.glob(str(stereo_l_dir / '*.jpeg')))
                
                for left_path in sorted(left_images):
                    filename = Path(left_path).name
                    right_path = stereo_r_dir / filename
                    
                    if not right_path.exists():
                        logger.warning(f"Missing right stereo pair for: {left_path}")
                        continue
                    
                    node_id, sequence = self._parse_image_filename(filename)
                    
                    pair = StereoImagePair(
                        left_path=str(left_path),
                        right_path=str(right_path),
                        node_id=node_id,
                        sequence=sequence,
                        landmark_id=landmark_id,
                        trilat_id=trilat_id
                    )
                    landmark_data.stereo_pairs.append(pair)
                
                trilat_data.landmarks[landmark_id] = landmark_data
            
            self.trilat_sets[trilat_id] = trilat_data
        
        # Log summary
        total_pairs = sum(
            len(lm.stereo_pairs)
            for trilat in self.trilat_sets.values()
            for lm in trilat.landmarks.values()
        )
        logger.info(f"Loaded {len(self.trilat_sets)} trilateration sets with {total_pairs} total stereo pairs")
    
    def __len__(self) -> int:
        return len(self.trilat_sets)
    
    def __getitem__(self, trilat_id: str) -> TrilatData:
        return self.trilat_sets[trilat_id]
    
    def __iter__(self) -> Iterator[TrilatData]:
        return iter(self.trilat_sets.values())
    
    def __contains__(self, trilat_id: str) -> bool:
        return trilat_id in self.trilat_sets
    
    def keys(self) -> List[str]:
        return list(self.trilat_sets.keys())
    
    def values(self) -> List[TrilatData]:
        return list(self.trilat_sets.values())
    
    def items(self):
        return self.trilat_sets.items()
    
    def get_all_stereo_pairs(self) -> List[StereoImagePair]:
        """Get all stereo pairs across all trilateration sets."""
        pairs = []
        for trilat in self.trilat_sets.values():
            pairs.extend(trilat.get_all_stereo_pairs())
        return pairs
    
    def get_trilat_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for each trilateration set."""
        stats = {}
        for trilat_id, trilat in self.trilat_sets.items():
            num_landmarks = len(trilat.landmarks)
            num_pairs = sum(len(lm.stereo_pairs) for lm in trilat.landmarks.values())
            unique_nodes = trilat.get_unique_nodes()
            
            stats[trilat_id] = {
                'num_landmarks': num_landmarks,
                'num_pairs': num_pairs,
                'num_nodes': len(unique_nodes),
                'nodes': unique_nodes,
                'landmarks': list(trilat.landmarks.keys())
            }
        return stats
    
    def summary(self) -> str:
        """Get a summary string of the dataset."""
        lines = [f"TrilatDataset: {self.root_dir}"]
        lines.append(f"Trilateration sets: {len(self.trilat_sets)}")
        
        for trilat_id, trilat in sorted(self.trilat_sets.items()):
            num_pairs = sum(len(lm.stereo_pairs) for lm in trilat.landmarks.values())
            lines.append(f"  {trilat_id}: {len(trilat.landmarks)} landmarks, {num_pairs} pairs")
            for lm_id, lm in sorted(trilat.landmarks.items()):
                nodes = lm.get_unique_nodes()
                lines.append(f"    {lm_id}: {len(lm.stereo_pairs)} pairs from nodes {nodes}")
        
        return '\n'.join(lines)


def create_evaluation_dataset(root_dir: str, trilateration_data: Dict,
                             landmark_xyz: Dict, node_xyz: Dict) -> TrilatDataset:
    """
    Create a dataset with full ground truth information for evaluation.
    
    Args:
        root_dir: Root directory of the landmark_stereov1_corrupt dataset
        trilateration_data: Dictionary with trilateration ground truth
        landmark_xyz: Dictionary mapping landmark IDs to XYZ coordinates
        node_xyz: Dictionary mapping node IDs to XYZ coordinates
    
    Returns:
        TrilatDataset with ground truth information populated
    """
    dataset = TrilatDataset(root_dir, trilateration_data=trilateration_data)
    
    # Add landmark coordinates
    for trilat in dataset.trilat_sets.values():
        for lm_id, landmark in trilat.landmarks.items():
            # Map landmark directory name to canonical ID
            # e.g., 'landmark1' -> 'l1' (depends on dataset structure)
            canonical_id = lm_id.replace('landmark', 'l')
            if canonical_id in landmark_xyz:
                xyz = landmark_xyz[canonical_id]
                landmark.location = (xyz[0], xyz[1])  # Store X, Y as location
    
    return dataset


# Convenience function for loading with built-in ground truth
def load_dataset(root_dir: str, with_ground_truth: bool = True) -> TrilatDataset:
    """
    Load the landmark_stereov1_corrupt dataset.
    
    Args:
        root_dir: Root directory of the dataset
        with_ground_truth: If True, include built-in ground truth data
    
    Returns:
        TrilatDataset instance
    """
    if with_ground_truth:
        from .landmark_db import TRILATERATION_DATA
        return TrilatDataset(root_dir, trilateration_data=TRILATERATION_DATA)
    return TrilatDataset(root_dir)
