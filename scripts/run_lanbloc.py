#!/usr/bin/env python3
"""
LanBLoc: Landmark-Based Localization

Run LanBLoc localization on a stereo image pair.

Usage:
    python run_lanbloc.py --left LEFT_IMAGE --right RIGHT_IMAGE [options]
    python run_lanbloc.py --dataset PATH --trilat TRILAT_ID --node NODE_ID [options]

Examples:
    # Single stereo pair
    python run_lanbloc.py --left left.png --right right.png --config config.yaml
    
    # From dataset
    python run_lanbloc.py --dataset ./landmark_stereov1_corrupt --trilat trilat1 --node 1
    
    # With known distances (bypass stereo depth)
    python run_lanbloc.py --landmarks l1,l2,l3 --distances 43.87,40.44,38.94
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lanbloc import LanBLoc
from lanbloc.data import LandmarkDatabase, TrilatDataset, LANDMARK_XYZ, NODE_XYZ
from lanbloc.utils.visualization import visualize_localization, plot_ground_truth_vs_estimated

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def parse_list(value: str) -> List[str]:
    """Parse comma-separated string to list."""
    return [v.strip() for v in value.split(',') if v.strip()]


def parse_float_list(value: str) -> List[float]:
    """Parse comma-separated string to list of floats."""
    return [float(v.strip()) for v in value.split(',') if v.strip()]


def run_single_pair(args, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run localization on a single stereo image pair."""
    if not HAS_CV2:
        raise ImportError("OpenCV is required to load images")
    
    # Load images
    left_image = cv2.imread(args.left)
    right_image = cv2.imread(args.right)
    
    if left_image is None:
        raise FileNotFoundError(f"Could not load left image: {args.left}")
    if right_image is None:
        raise FileNotFoundError(f"Could not load right image: {args.right}")
    
    # Initialize LanBLoc
    landmark_db = LandmarkDatabase.from_builtin()
    
    if args.config:
        lanbloc = LanBLoc.from_config(args.config)
    else:
        lanbloc = LanBLoc(landmark_db=landmark_db.get_2d_positions())
    
    # Run localization
    result = lanbloc.localize(left_image, right_image)
    
    return {
        'x': result.x,
        'y': result.y,
        'success': result.success,
        'num_landmarks': result.num_landmarks,
        'detected_landmarks': result.detected_landmarks,
        'residual': result.residual
    }


def run_with_known_distances(args) -> Dict[str, Any]:
    """Run localization with known landmark distances (bypass stereo depth)."""
    landmarks = parse_list(args.landmarks)
    distances = parse_float_list(args.distances)
    
    if len(landmarks) != len(distances):
        raise ValueError(f"Number of landmarks ({len(landmarks)}) must match distances ({len(distances)})")
    
    if len(landmarks) < 3:
        raise ValueError("At least 3 landmarks required for trilateration")
    
    # Initialize LanBLoc
    landmark_db = LandmarkDatabase.from_builtin()
    lanbloc = LanBLoc(landmark_db=landmark_db.get_2d_positions())
    
    # Run localization with known distances
    result = lanbloc.localize_with_known_distances(landmarks, distances)
    
    return {
        'x': result.x,
        'y': result.y,
        'success': result.success,
        'num_landmarks': result.num_landmarks,
        'detected_landmarks': result.detected_landmarks,
        'residual': result.residual
    }


def run_from_dataset(args, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run localization on images from the dataset."""
    if not HAS_CV2:
        raise ImportError("OpenCV is required to load images")
    
    # Load dataset
    dataset = TrilatDataset(args.dataset)
    
    if args.trilat not in dataset:
        raise ValueError(f"Trilateration set '{args.trilat}' not found in dataset")
    
    trilat = dataset[args.trilat]
    
    # Get stereo pairs from specified node
    pairs_by_landmark = trilat.get_pairs_by_node(args.node)
    
    if not pairs_by_landmark:
        available_nodes = trilat.get_unique_nodes()
        raise ValueError(f"No stereo pairs found for node '{args.node}'. Available: {available_nodes}")
    
    # Initialize LanBLoc
    landmark_db = LandmarkDatabase.from_builtin()
    
    if args.config:
        lanbloc = LanBLoc.from_config(args.config)
    else:
        lanbloc = LanBLoc(landmark_db=landmark_db.get_2d_positions())
    
    results = []
    
    # Process each landmark's stereo pair
    for lm_id, pairs in pairs_by_landmark.items():
        for pair in pairs:
            left_img, right_img = pair.load_images()
            
            logging.info(f"Processing {lm_id} (sequence {pair.sequence}) from node {pair.node_id}")
            
            # Run detection and depth estimation for this landmark
            result = lanbloc.localize(left_img, right_img)
            
            results.append({
                'landmark': lm_id,
                'sequence': pair.sequence,
                'success': result.success,
                'x': result.x if result.success else None,
                'y': result.y if result.success else None
            })
    
    return {
        'trilat': args.trilat,
        'node': args.node,
        'results': results
    }


def visualize_result(result: Dict[str, Any], landmark_db: LandmarkDatabase, 
                    output_path: Optional[str] = None):
    """Visualize localization result."""
    if result.get('success'):
        fig = visualize_localization(
            estimated_pos=(result['x'], result['y']),
            landmark_positions=landmark_db.get_2d_positions(),
            detected_landmarks=result.get('detected_landmarks', []),
            title="LanBLoc Localization Result"
        )
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            logging.info(f"Visualization saved to {output_path}")
        else:
            try:
                import matplotlib.pyplot as plt
                plt.show()
            except:
                logging.warning("Could not display figure (no display available)")


def main():
    parser = argparse.ArgumentParser(
        description="LanBLoc: Landmark-Based Localization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input modes
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument('--left', '-l', help='Path to left stereo image')
    input_group.add_argument('--right', '-r', help='Path to right stereo image')
    input_group.add_argument('--dataset', '-d', help='Path to landmark_stereov1_corrupt dataset')
    input_group.add_argument('--trilat', '-t', help='Trilateration set ID (e.g., trilat1)')
    input_group.add_argument('--node', '-n', help='Node ID to process')
    
    # Known distances mode
    dist_group = parser.add_argument_group('Known Distances Mode')
    dist_group.add_argument('--landmarks', help='Comma-separated landmark IDs (e.g., l1,l2,l3)')
    dist_group.add_argument('--distances', help='Comma-separated distances (e.g., 43.87,40.44,38.94)')
    
    # Configuration
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument('--config', '-c', help='Path to YAML configuration file')
    config_group.add_argument('--model', '-m', help='Path to YOLO model weights')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output', '-o', help='Output file for results (JSON)')
    output_group.add_argument('--visualize', '-v', action='store_true', help='Visualize result')
    output_group.add_argument('--plot-output', help='Save visualization to file')
    
    # Other options
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    
    try:
        # Determine mode and run
        if args.landmarks and args.distances:
            logger.info("Running with known distances (no stereo images)")
            result = run_with_known_distances(args)
            
        elif args.dataset and args.trilat and args.node:
            logger.info(f"Running on dataset: {args.trilat}/{args.node}")
            result = run_from_dataset(args, config)
            
        elif args.left and args.right:
            logger.info(f"Running on stereo pair: {args.left}, {args.right}")
            result = run_single_pair(args, config)
            
        else:
            parser.print_help()
            print("\nError: Please specify either:")
            print("  1. --left and --right for a stereo pair")
            print("  2. --dataset, --trilat, and --node for dataset mode")
            print("  3. --landmarks and --distances for known distances mode")
            sys.exit(1)
        
        # Print results
        print("\n" + "="*50)
        print("LOCALIZATION RESULTS")
        print("="*50)
        
        if 'results' in result:
            # Dataset mode with multiple results
            for r in result['results']:
                status = "✓" if r['success'] else "✗"
                pos = f"({r['x']:.4f}, {r['y']:.4f})" if r['success'] else "Failed"
                print(f"  {status} {r['landmark']} seq {r['sequence']}: {pos}")
        else:
            # Single result
            if result['success']:
                print(f"  Position: ({result['x']:.4f}, {result['y']:.4f})")
                print(f"  Landmarks used: {result['num_landmarks']}")
                print(f"  Detected: {result['detected_landmarks']}")
                print(f"  Residual: {result['residual']:.6f}")
            else:
                print("  Localization failed")
        
        print("="*50)
        
        # Save results
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to {args.output}")
        
        # Visualize
        if args.visualize or args.plot_output:
            if 'x' in result and result.get('success'):
                landmark_db = LandmarkDatabase.from_builtin()
                visualize_result(result, landmark_db, args.plot_output)
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == '__main__':
    main()
