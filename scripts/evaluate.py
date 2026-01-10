#!/usr/bin/env python3
"""
LanBLoc Evaluation Script

Evaluate LanBLoc localization accuracy on the landmark_stereov1_corrupt dataset.
Computes metrics including RMSE, mean error, and success rate.

Usage:
    python evaluate.py --dataset /path/to/landmark_stereov1_corrupt [options]
    python evaluate.py --use-ground-truth-distances  # Use known distances

Examples:
    # Full evaluation with stereo depth estimation
    python evaluate.py --dataset ./landmark_stereov1_corrupt --config ../config/default_config.yaml
    
    # Evaluation using ground truth distances (trilateration only)
    python evaluate.py --use-ground-truth-distances --output results.json
    
    # Evaluate specific trilateration sets
    python evaluate.py --dataset ./landmark_stereov1_corrupt --trilats trilat1,trilat2
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import math

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lanbloc import LanBLoc
from lanbloc.core.trilateration import trilaterate, TrilaterationResult
from lanbloc.data import (
    LandmarkDatabase, NodeDatabase, TrilatDataset,
    TRILATERATION_DATA, LANDMARK_XYZ, NODE_XYZ
)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def compute_euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Compute 2D Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def compute_3d_distance(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    """Compute 3D Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)


def compute_metrics(errors: List[float]) -> Dict[str, float]:
    """Compute evaluation metrics from error list."""
    if not errors:
        return {
            'count': 0,
            'rmse': float('nan'),
            'mae': float('nan'),
            'median': float('nan'),
            'std': float('nan'),
            'min': float('nan'),
            'max': float('nan')
        }
    
    errors = np.array(errors)
    return {
        'count': len(errors),
        'rmse': float(np.sqrt(np.mean(errors**2))),
        'mae': float(np.mean(errors)),
        'median': float(np.median(errors)),
        'std': float(np.std(errors)),
        'min': float(np.min(errors)),
        'max': float(np.max(errors))
    }


def evaluate_with_ground_truth_distances(
    trilateration_data: Dict,
    landmark_xyz: Dict,
    node_xyz: Dict,
    trilat_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluate trilateration using ground truth distances.
    
    This mode bypasses stereo depth estimation and tests trilateration accuracy
    directly using known landmark distances from the dataset.
    
    Args:
        trilateration_data: Dictionary with trilateration ground truth
        landmark_xyz: Dictionary mapping landmark IDs to XYZ coordinates
        node_xyz: Dictionary mapping node IDs to XYZ coordinates
        trilat_ids: Optional list of trilateration set IDs to evaluate
    
    Returns:
        Dictionary with evaluation results
    """
    logger = logging.getLogger(__name__)
    
    results = {
        'mode': 'ground_truth_distances',
        'timestamp': datetime.now().isoformat(),
        'trilat_results': {},
        'errors': [],
        'all_node_results': []
    }
    
    if trilat_ids is None:
        trilat_ids = list(trilateration_data.keys())
    
    for trilat_id in trilat_ids:
        if trilat_id not in trilateration_data:
            logger.warning(f"Trilateration set '{trilat_id}' not found")
            continue
        
        trilat = trilateration_data[trilat_id]
        nodes = trilat.get('nodes', [])
        landmarks = trilat.get('landmarks', [])
        
        logger.info(f"\nEvaluating {trilat_id}: {len(nodes)} nodes, {len(landmarks)} landmarks")
        
        trilat_result = {
            'nodes': nodes,
            'landmarks': landmarks,
            'node_results': []
        }
        
        # Get landmark positions (2D: x, y from XYZ)
        landmark_positions = {}
        for lm_id in landmarks:
            if lm_id in landmark_xyz:
                xyz = landmark_xyz[lm_id]
                landmark_positions[lm_id] = (xyz[0], xyz[1])
            else:
                logger.warning(f"Landmark {lm_id} not found in landmark_xyz")
        
        # Evaluate each node
        for node_id in nodes:
            # Get ground truth position
            if node_id not in node_xyz:
                logger.warning(f"Node {node_id} not found in node_xyz")
                continue
            
            gt_xyz = node_xyz[node_id]
            gt_position = (gt_xyz[0], gt_xyz[1])
            
            # Get distances from this node to landmarks
            distance_key = f"distanceFrom{node_id.upper()}To"
            if distance_key not in trilat:
                # Try different case variations
                for key in trilat.keys():
                    if key.lower() == distance_key.lower():
                        distance_key = key
                        break
                else:
                    logger.warning(f"No distance data for node {node_id}")
                    continue
            
            distances = trilat[distance_key]
            
            # Prepare data for trilateration
            positions = []
            dist_list = []
            used_landmarks = []
            
            for lm_id in landmarks:
                if lm_id in landmark_positions and lm_id in distances:
                    positions.append(landmark_positions[lm_id])
                    dist_list.append(distances[lm_id])
                    used_landmarks.append(lm_id)
            
            if len(positions) < 3:
                logger.warning(f"Insufficient landmarks for node {node_id}")
                continue
            
            # Run trilateration
            try:
                result = trilaterate(
                    positions=positions,
                    distances=dist_list,
                    method='lbfgsb'
                )
                
                if result.success:
                    estimated = (result.x, result.y)
                    error = compute_euclidean_distance(estimated, gt_position)
                    
                    node_result = {
                        'node_id': node_id,
                        'success': True,
                        'estimated': estimated,
                        'ground_truth': gt_position,
                        'error': error,
                        'residual': result.residual,
                        'landmarks_used': used_landmarks
                    }
                    
                    results['errors'].append(error)
                    logger.info(f"  {node_id}: Error = {error:.4f}m")
                else:
                    node_result = {
                        'node_id': node_id,
                        'success': False,
                        'ground_truth': gt_position,
                        'landmarks_used': used_landmarks
                    }
                    logger.warning(f"  {node_id}: Trilateration failed")
                
            except Exception as e:
                node_result = {
                    'node_id': node_id,
                    'success': False,
                    'error_message': str(e),
                    'ground_truth': gt_position
                }
                logger.error(f"  {node_id}: Error - {e}")
            
            trilat_result['node_results'].append(node_result)
            results['all_node_results'].append(node_result)
        
        results['trilat_results'][trilat_id] = trilat_result
    
    # Compute overall metrics
    results['metrics'] = compute_metrics(results['errors'])
    
    return results


def evaluate_with_stereo_images(
    dataset_path: str,
    config_path: Optional[str] = None,
    trilat_ids: Optional[List[str]] = None,
    node_xyz: Dict = None
) -> Dict[str, Any]:
    """
    Evaluate LanBLoc on stereo images from the dataset.
    
    This mode runs the full pipeline including YOLO detection and stereo depth.
    
    Args:
        dataset_path: Path to landmark_stereov1_corrupt dataset
        config_path: Optional path to configuration file
        trilat_ids: Optional list of trilateration set IDs to evaluate
        node_xyz: Dictionary mapping node IDs to XYZ coordinates
    
    Returns:
        Dictionary with evaluation results
    """
    if not HAS_CV2:
        raise ImportError("OpenCV is required for stereo image evaluation")
    
    logger = logging.getLogger(__name__)
    
    # Load dataset
    dataset = TrilatDataset(dataset_path, trilat_ids=trilat_ids,
                           trilateration_data=TRILATERATION_DATA)
    
    # Initialize LanBLoc
    landmark_db = LandmarkDatabase.from_builtin()
    
    if config_path:
        lanbloc = LanBLoc.from_config(config_path)
    else:
        lanbloc = LanBLoc(landmark_db=landmark_db.get_2d_positions())
    
    results = {
        'mode': 'stereo_images',
        'timestamp': datetime.now().isoformat(),
        'dataset_path': dataset_path,
        'trilat_results': {},
        'errors': [],
        'all_node_results': []
    }
    
    node_xyz = node_xyz or NODE_XYZ
    
    for trilat_id, trilat in dataset.items():
        logger.info(f"\nProcessing {trilat_id}")
        
        trilat_result = {
            'landmarks': list(trilat.landmarks.keys()),
            'node_results': []
        }
        
        # Group stereo pairs by node
        unique_nodes = trilat.get_unique_nodes()
        
        for node_id in unique_nodes:
            pairs_by_landmark = trilat.get_pairs_by_node(node_id)
            
            if len(pairs_by_landmark) < 3:
                logger.warning(f"  {node_id}: Insufficient landmarks ({len(pairs_by_landmark)})")
                continue
            
            # Get ground truth if available
            full_node_id = f"n{node_id}" if not node_id.startswith('n') else node_id
            gt_position = None
            if full_node_id in node_xyz:
                gt_xyz = node_xyz[full_node_id]
                gt_position = (gt_xyz[0], gt_xyz[1])
            
            # Process each landmark and collect depths
            detected_landmarks = []
            estimated_distances = []
            
            for lm_id, pairs in pairs_by_landmark.items():
                # Use first pair for each landmark
                pair = pairs[0]
                
                try:
                    left_img, right_img = pair.load_images()
                    
                    # Run localization (will detect and estimate depth)
                    result = lanbloc.localize(left_img, right_img)
                    
                    # Collect detected landmarks and distances
                    if result.success:
                        detected_landmarks.extend(result.detected_landmarks)
                        # Note: distances would come from depth estimation
                        
                except Exception as e:
                    logger.warning(f"  Error processing {lm_id}: {e}")
            
            # Record result
            node_result = {
                'node_id': node_id,
                'landmarks_processed': list(pairs_by_landmark.keys()),
                'ground_truth': gt_position
            }
            
            # If we have a localization result
            # (simplified - full implementation would aggregate across landmarks)
            
            trilat_result['node_results'].append(node_result)
            results['all_node_results'].append(node_result)
        
        results['trilat_results'][trilat_id] = trilat_result
    
    # Compute metrics
    results['metrics'] = compute_metrics(results['errors'])
    
    return results


def print_results(results: Dict[str, Any]):
    """Print evaluation results in a formatted table."""
    print("\n" + "="*70)
    print("LANBLOC EVALUATION RESULTS")
    print("="*70)
    print(f"Mode: {results['mode']}")
    print(f"Timestamp: {results['timestamp']}")
    
    # Per-trilat summary
    print("\n" + "-"*70)
    print("Per-Trilateration Set Results:")
    print("-"*70)
    
    for trilat_id, trilat_result in results['trilat_results'].items():
        node_results = trilat_result.get('node_results', [])
        successful = [n for n in node_results if n.get('success', False)]
        
        errors = [n['error'] for n in successful if 'error' in n]
        
        print(f"\n{trilat_id}:")
        print(f"  Nodes evaluated: {len(node_results)}")
        print(f"  Successful: {len(successful)}")
        
        if errors:
            print(f"  RMSE: {np.sqrt(np.mean(np.array(errors)**2)):.4f}m")
            print(f"  Mean Error: {np.mean(errors):.4f}m")
    
    # Overall metrics
    print("\n" + "-"*70)
    print("Overall Metrics:")
    print("-"*70)
    
    metrics = results.get('metrics', {})
    print(f"  Total evaluations: {metrics.get('count', 0)}")
    print(f"  RMSE: {metrics.get('rmse', float('nan')):.6f}m")
    print(f"  Mean Absolute Error: {metrics.get('mae', float('nan')):.6f}m")
    print(f"  Median Error: {metrics.get('median', float('nan')):.6f}m")
    print(f"  Std Dev: {metrics.get('std', float('nan')):.6f}m")
    print(f"  Min Error: {metrics.get('min', float('nan')):.6f}m")
    print(f"  Max Error: {metrics.get('max', float('nan')):.6f}m")
    
    # Success rate
    all_nodes = results.get('all_node_results', [])
    if all_nodes:
        success_count = sum(1 for n in all_nodes if n.get('success', False))
        success_rate = 100 * success_count / len(all_nodes)
        print(f"  Success Rate: {success_rate:.1f}%")
    
    print("\n" + "="*70)


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to JSON file."""
    # Convert tuples to lists for JSON serialization
    def convert_tuples(obj):
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, dict):
            return {k: convert_tuples(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_tuples(item) for item in obj]
        return obj
    
    serializable = convert_tuples(results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    logging.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LanBLoc localization on landmark_stereov1_corrupt dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--use-ground-truth-distances', '-g', action='store_true',
                           help='Evaluate using ground truth distances (bypass stereo depth)')
    mode_group.add_argument('--use-stereo-images', '-s', action='store_true',
                           help='Evaluate using stereo images (full pipeline)')
    
    # Dataset options
    parser.add_argument('--dataset', '-d', help='Path to landmark_stereov1_corrupt dataset')
    parser.add_argument('--trilats', '-t', help='Comma-separated trilat IDs to evaluate')
    
    # Configuration
    parser.add_argument('--config', '-c', help='Path to configuration YAML')
    
    # Output options
    parser.add_argument('--output', '-o', help='Output JSON file for results')
    parser.add_argument('--log-file', help='Save logs to file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose, args.log_file)
    logger = logging.getLogger(__name__)
    
    # Parse trilat IDs
    trilat_ids = None
    if args.trilats:
        trilat_ids = [t.strip() for t in args.trilats.split(',')]
    
    try:
        start_time = time.time()
        
        if args.use_stereo_images:
            if not args.dataset:
                parser.error("--dataset required for stereo image evaluation")
            
            logger.info("Evaluating with stereo images (full pipeline)")
            results = evaluate_with_stereo_images(
                dataset_path=args.dataset,
                config_path=args.config,
                trilat_ids=trilat_ids
            )
            
        else:
            # Default to ground truth distances mode
            logger.info("Evaluating with ground truth distances")
            results = evaluate_with_ground_truth_distances(
                trilateration_data=TRILATERATION_DATA,
                landmark_xyz=LANDMARK_XYZ,
                node_xyz=NODE_XYZ,
                trilat_ids=trilat_ids
            )
        
        elapsed = time.time() - start_time
        results['elapsed_time'] = elapsed
        
        # Print results
        print_results(results)
        print(f"\nEvaluation completed in {elapsed:.2f}s")
        
        # Save results
        if args.output:
            save_results(results, args.output)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == '__main__':
    main()
