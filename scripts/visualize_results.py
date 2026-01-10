#!/usr/bin/env python3
"""
LanBLoc Visualization Script

Generate publication-quality figures from evaluation results.

Usage:
    python visualize_results.py --results results.json --output figures/
    python visualize_results.py --plot-landmarks
    python visualize_results.py --plot-trajectories --dataset /path/to/data

Examples:
    # Visualize evaluation results
    python visualize_results.py --results evaluation_results.json --output ./figures
    
    # Plot landmark positions
    python visualize_results.py --plot-landmarks --output landmark_map.png
    
    # Plot ground truth vs estimated
    python visualize_results.py --results results.json --comparison-plot
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from lanbloc.data import LandmarkDatabase, NodeDatabase, LANDMARK_XYZ, NODE_XYZ


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_results(results_path: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_landmarks(landmark_xyz: Dict, node_xyz: Optional[Dict] = None,
                  output_path: Optional[str] = None, title: str = "Landmark Map"):
    """
    Plot landmark positions on a 2D map.
    
    Args:
        landmark_xyz: Dictionary mapping landmark IDs to XYZ coordinates
        node_xyz: Optional dictionary mapping node IDs to XYZ coordinates
        output_path: Optional path to save the figure
        title: Plot title
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib is required for visualization")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Extract X, Y coordinates for landmarks
    lm_x = [xyz[0] for xyz in landmark_xyz.values()]
    lm_y = [xyz[1] for xyz in landmark_xyz.values()]
    lm_ids = list(landmark_xyz.keys())
    
    # Plot landmarks
    ax.scatter(lm_x, lm_y, c='blue', s=100, marker='^', label='Landmarks', zorder=5)
    
    # Annotate landmarks
    for i, lm_id in enumerate(lm_ids):
        ax.annotate(lm_id, (lm_x[i], lm_y[i]), 
                   textcoords="offset points", xytext=(5, 5),
                   fontsize=8, color='blue')
    
    # Plot nodes if provided
    if node_xyz:
        node_x = [xyz[0] for xyz in node_xyz.values()]
        node_y = [xyz[1] for xyz in node_xyz.values()]
        node_ids = list(node_xyz.keys())
        
        ax.scatter(node_x, node_y, c='red', s=80, marker='o', label='Nodes', zorder=5)
        
        for i, node_id in enumerate(node_ids):
            ax.annotate(node_id, (node_x[i], node_y[i]),
                       textcoords="offset points", xytext=(5, -10),
                       fontsize=8, color='red')
    
    ax.set_xlabel('X (km from Earth center)', fontsize=12)
    ax.set_ylabel('Y (km from Earth center)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved landmark map to {output_path}")
    
    return fig, ax


def plot_error_distribution(errors: List[float], output_path: Optional[str] = None,
                           title: str = "Localization Error Distribution"):
    """
    Plot histogram of localization errors.
    
    Args:
        errors: List of error values (in meters)
        output_path: Optional path to save the figure
        title: Plot title
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib is required for visualization")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.4f}m')
    ax1.axvline(np.median(errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.4f}m')
    ax1.set_xlabel('Error (m)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Error Distribution', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2 = axes[1]
    bp = ax2.boxplot(errors, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax2.set_ylabel('Error (m)', fontsize=12)
    ax2.set_title('Error Box Plot', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f"RMSE: {np.sqrt(np.mean(np.array(errors)**2)):.4f}m\n"
    stats_text += f"MAE: {np.mean(errors):.4f}m\n"
    stats_text += f"Std: {np.std(errors):.4f}m\n"
    stats_text += f"Min: {np.min(errors):.4f}m\n"
    stats_text += f"Max: {np.max(errors):.4f}m"
    
    ax2.text(1.3, np.median(errors), stats_text, fontsize=10,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved error distribution to {output_path}")
    
    return fig, axes


def plot_ground_truth_vs_estimated(results: Dict[str, Any], 
                                  output_path: Optional[str] = None,
                                  title: str = "Ground Truth vs Estimated Positions"):
    """
    Plot ground truth positions against estimated positions.
    
    Args:
        results: Evaluation results dictionary
        output_path: Optional path to save the figure
        title: Plot title
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib is required for visualization")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    all_nodes = results.get('all_node_results', [])
    
    gt_x, gt_y = [], []
    est_x, est_y = [], []
    node_ids = []
    
    for node_result in all_nodes:
        if not node_result.get('success', False):
            continue
        
        gt = node_result.get('ground_truth')
        est = node_result.get('estimated')
        
        if gt and est:
            gt_x.append(gt[0])
            gt_y.append(gt[1])
            est_x.append(est[0])
            est_y.append(est[1])
            node_ids.append(node_result.get('node_id', ''))
    
    # Plot ground truth
    ax.scatter(gt_x, gt_y, c='green', s=100, marker='o', label='Ground Truth', zorder=5)
    
    # Plot estimated
    ax.scatter(est_x, est_y, c='red', s=100, marker='x', label='Estimated', zorder=5)
    
    # Draw error lines
    for i in range(len(gt_x)):
        ax.plot([gt_x[i], est_x[i]], [gt_y[i], est_y[i]], 
               'k--', alpha=0.5, linewidth=1)
        
        # Annotate node
        ax.annotate(node_ids[i], (gt_x[i], gt_y[i]),
                   textcoords="offset points", xytext=(5, 5),
                   fontsize=8)
    
    ax.set_xlabel('X (km from Earth center)', fontsize=12)
    ax.set_ylabel('Y (km from Earth center)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved comparison plot to {output_path}")
    
    return fig, ax


def plot_per_trilat_errors(results: Dict[str, Any], output_path: Optional[str] = None,
                          title: str = "Error by Trilateration Set"):
    """
    Plot errors grouped by trilateration set.
    
    Args:
        results: Evaluation results dictionary
        output_path: Optional path to save the figure
        title: Plot title
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib is required for visualization")
    
    trilat_results = results.get('trilat_results', {})
    
    trilat_ids = []
    errors_per_trilat = []
    
    for trilat_id, trilat_result in sorted(trilat_results.items()):
        node_results = trilat_result.get('node_results', [])
        errors = [n['error'] for n in node_results if n.get('success') and 'error' in n]
        
        if errors:
            trilat_ids.append(trilat_id)
            errors_per_trilat.append(errors)
    
    if not errors_per_trilat:
        logging.warning("No errors to plot")
        return None, None
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Box plot
    bp = ax.boxplot(errors_per_trilat, labels=trilat_ids, patch_artist=True)
    
    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(trilat_ids)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Add mean markers
    means = [np.mean(e) for e in errors_per_trilat]
    ax.scatter(range(1, len(trilat_ids)+1), means, marker='D', color='red', 
              s=50, zorder=3, label='Mean')
    
    ax.set_xlabel('Trilateration Set', fontsize=12)
    ax.set_ylabel('Error (m)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate x labels if needed
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved per-trilat error plot to {output_path}")
    
    return fig, ax


def create_evaluation_report(results: Dict[str, Any], output_dir: str):
    """
    Create a comprehensive evaluation report with multiple figures.
    
    Args:
        results: Evaluation results dictionary
        output_dir: Directory to save figures
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib is required for visualization")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Creating evaluation report in {output_dir}")
    
    # 1. Plot landmark map
    plot_landmarks(
        LANDMARK_XYZ, NODE_XYZ,
        output_path=str(output_dir / 'landmark_map.png'),
        title='Landmark and Node Positions'
    )
    
    # 2. Plot error distribution
    errors = results.get('errors', [])
    if errors:
        plot_error_distribution(
            errors,
            output_path=str(output_dir / 'error_distribution.png'),
            title='LanBLoc Localization Error Distribution'
        )
    
    # 3. Plot ground truth vs estimated
    plot_ground_truth_vs_estimated(
        results,
        output_path=str(output_dir / 'gt_vs_estimated.png'),
        title='Ground Truth vs Estimated Positions'
    )
    
    # 4. Plot per-trilat errors
    plot_per_trilat_errors(
        results,
        output_path=str(output_dir / 'per_trilat_errors.png'),
        title='Error by Trilateration Set'
    )
    
    # 5. Create summary figure
    create_summary_figure(
        results,
        output_path=str(output_dir / 'summary.png')
    )
    
    logger.info(f"Evaluation report complete. Figures saved to {output_dir}")


def create_summary_figure(results: Dict[str, Any], output_path: Optional[str] = None):
    """
    Create a multi-panel summary figure.
    
    Args:
        results: Evaluation results dictionary
        output_path: Optional path to save the figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib is required for visualization")
    
    fig = plt.figure(figsize=(16, 12))
    
    # Panel 1: Landmark Map (top-left)
    ax1 = fig.add_subplot(2, 2, 1)
    
    lm_x = [xyz[0] for xyz in LANDMARK_XYZ.values()]
    lm_y = [xyz[1] for xyz in LANDMARK_XYZ.values()]
    node_x = [xyz[0] for xyz in NODE_XYZ.values()]
    node_y = [xyz[1] for xyz in NODE_XYZ.values()]
    
    ax1.scatter(lm_x, lm_y, c='blue', s=60, marker='^', label='Landmarks')
    ax1.scatter(node_x, node_y, c='red', s=40, marker='o', label='Nodes')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Landmark and Node Positions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Panel 2: Error Histogram (top-right)
    ax2 = fig.add_subplot(2, 2, 2)
    
    errors = results.get('errors', [])
    if errors:
        ax2.hist(errors, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax2.axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.4f}')
        ax2.set_xlabel('Error (m)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Panel 3: GT vs Estimated (bottom-left)
    ax3 = fig.add_subplot(2, 2, 3)
    
    all_nodes = results.get('all_node_results', [])
    for node_result in all_nodes:
        if not node_result.get('success', False):
            continue
        gt = node_result.get('ground_truth')
        est = node_result.get('estimated')
        if gt and est:
            ax3.scatter(gt[0], gt[1], c='green', s=60, marker='o')
            ax3.scatter(est[0], est[1], c='red', s=60, marker='x')
            ax3.plot([gt[0], est[0]], [gt[1], est[1]], 'k--', alpha=0.5, linewidth=0.5)
    
    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Ground Truth'),
        Line2D([0], [0], marker='x', color='red', markersize=10, label='Estimated')
    ]
    ax3.legend(handles=legend_elements)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('Ground Truth vs Estimated')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # Panel 4: Metrics Table (bottom-right)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    metrics = results.get('metrics', {})
    
    table_data = [
        ['Metric', 'Value'],
        ['Total Evaluations', str(metrics.get('count', 0))],
        ['RMSE', f"{metrics.get('rmse', 0):.6f} m"],
        ['Mean Absolute Error', f"{metrics.get('mae', 0):.6f} m"],
        ['Median Error', f"{metrics.get('median', 0):.6f} m"],
        ['Std Dev', f"{metrics.get('std', 0):.6f} m"],
        ['Min Error', f"{metrics.get('min', 0):.6f} m"],
        ['Max Error', f"{metrics.get('max', 0):.6f} m"]
    ]
    
    table = ax4.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc='center',
        cellLoc='center',
        colWidths=[0.5, 0.4]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white')
    
    ax4.set_title('Evaluation Metrics', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle('LanBLoc Evaluation Summary', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved summary figure to {output_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualize LanBLoc evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input options
    parser.add_argument('--results', '-r', help='Path to evaluation results JSON')
    
    # Plot types
    parser.add_argument('--plot-landmarks', action='store_true',
                       help='Plot landmark positions')
    parser.add_argument('--comparison-plot', action='store_true',
                       help='Plot ground truth vs estimated')
    parser.add_argument('--error-distribution', action='store_true',
                       help='Plot error distribution')
    parser.add_argument('--per-trilat', action='store_true',
                       help='Plot errors by trilateration set')
    parser.add_argument('--summary', action='store_true',
                       help='Create summary figure')
    parser.add_argument('--full-report', action='store_true',
                       help='Generate full evaluation report')
    
    # Output options
    parser.add_argument('--output', '-o', help='Output file or directory')
    parser.add_argument('--show', action='store_true', help='Display plots')
    
    # Other options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    if not HAS_MATPLOTLIB:
        logger.error("Matplotlib is required for visualization. Install with: pip install matplotlib")
        sys.exit(1)
    
    try:
        # Load results if provided
        results = None
        if args.results:
            results = load_results(args.results)
        
        # Generate requested plots
        if args.plot_landmarks:
            output = args.output or 'landmark_map.png'
            plot_landmarks(LANDMARK_XYZ, NODE_XYZ, output_path=output)
        
        elif args.full_report:
            if not results:
                parser.error("--results required for full report")
            output_dir = args.output or './figures'
            create_evaluation_report(results, output_dir)
        
        elif args.comparison_plot:
            if not results:
                parser.error("--results required for comparison plot")
            output = args.output or 'gt_vs_estimated.png'
            plot_ground_truth_vs_estimated(results, output_path=output)
        
        elif args.error_distribution:
            if not results:
                parser.error("--results required for error distribution")
            errors = results.get('errors', [])
            output = args.output or 'error_distribution.png'
            plot_error_distribution(errors, output_path=output)
        
        elif args.per_trilat:
            if not results:
                parser.error("--results required for per-trilat plot")
            output = args.output or 'per_trilat_errors.png'
            plot_per_trilat_errors(results, output_path=output)
        
        elif args.summary:
            if not results:
                parser.error("--results required for summary")
            output = args.output or 'summary.png'
            create_summary_figure(results, output_path=output)
        
        else:
            parser.print_help()
            print("\nPlease specify a plot type (--plot-landmarks, --full-report, etc.)")
        
        if args.show:
            plt.show()
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == '__main__':
    main()
