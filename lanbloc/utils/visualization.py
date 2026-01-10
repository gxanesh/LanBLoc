"""
Visualization Utilities for LanBLoc

Functions for plotting trajectories, landmarks, and localization results.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import matplotlib.colors as mcolors
import logging

logger = logging.getLogger(__name__)


def plot_landmarks(
    landmarks: Dict[str, Tuple[float, float]],
    ax: Optional[plt.Axes] = None,
    color: str = 'blue',
    marker: str = '^',
    size: int = 100,
    labels: bool = True,
    **kwargs
) -> plt.Axes:
    """
    Plot landmark positions.
    
    Args:
        landmarks: Dictionary mapping landmark IDs to (x, y) positions
        ax: Matplotlib axes (created if None)
        color: Marker color
        marker: Marker style
        size: Marker size
        labels: Whether to show landmark labels
        **kwargs: Additional scatter plot arguments
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    x = [pos[0] for pos in landmarks.values()]
    y = [pos[1] for pos in landmarks.values()]
    
    ax.scatter(x, y, c=color, marker=marker, s=size, label='Landmarks', **kwargs)
    
    if labels:
        for lid, (lx, ly) in landmarks.items():
            ax.annotate(lid, (lx, ly), textcoords="offset points", 
                       xytext=(5, 5), fontsize=8)
    
    return ax


def plot_trajectory(
    positions: List[Tuple[float, float]],
    ax: Optional[plt.Axes] = None,
    color: str = 'red',
    marker: str = 'o',
    linewidth: float = 1.5,
    label: str = 'Trajectory',
    show_direction: bool = True,
    **kwargs
) -> plt.Axes:
    """
    Plot a trajectory of positions.
    
    Args:
        positions: List of (x, y) positions
        ax: Matplotlib axes (created if None)
        color: Line and marker color
        marker: Marker style
        linewidth: Line width
        label: Legend label
        show_direction: Show direction arrows
        **kwargs: Additional plot arguments
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    if not positions:
        return ax
    
    x = [p[0] for p in positions]
    y = [p[1] for p in positions]
    
    # Plot trajectory line
    ax.plot(x, y, c=color, linewidth=linewidth, label=label, **kwargs)
    
    # Plot points
    ax.scatter(x, y, c=color, marker=marker, s=50, zorder=5)
    
    # Mark start and end
    ax.scatter([x[0]], [y[0]], c='green', marker='s', s=100, 
               label='Start', zorder=6)
    ax.scatter([x[-1]], [y[-1]], c='orange', marker='*', s=150, 
               label='End', zorder=6)
    
    # Show direction arrows
    if show_direction and len(positions) > 1:
        for i in range(0, len(positions) - 1, max(1, len(positions) // 10)):
            dx = x[i + 1] - x[i]
            dy = y[i + 1] - y[i]
            if dx != 0 or dy != 0:
                ax.annotate('', xy=(x[i + 1], y[i + 1]), xytext=(x[i], y[i]),
                           arrowprops=dict(arrowstyle='->', color=color, lw=1))
    
    return ax


def plot_ground_truth_vs_estimated(
    ground_truth: List[Tuple[float, float]],
    estimated: List[Tuple[float, float]],
    landmarks: Optional[Dict[str, Tuple[float, float]]] = None,
    ax: Optional[plt.Axes] = None,
    show_errors: bool = True,
    title: str = 'Ground Truth vs Estimated Trajectory',
    **kwargs
) -> plt.Axes:
    """
    Plot ground truth and estimated trajectories for comparison.
    
    Args:
        ground_truth: List of ground truth (x, y) positions
        estimated: List of estimated (x, y) positions
        landmarks: Optional landmark positions to plot
        ax: Matplotlib axes (created if None)
        show_errors: Draw lines between GT and estimated
        title: Plot title
        **kwargs: Additional arguments
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot landmarks
    if landmarks:
        plot_landmarks(landmarks, ax=ax, color='blue', alpha=0.6)
    
    # Plot ground truth
    plot_trajectory(ground_truth, ax=ax, color='green', 
                   label='Ground Truth', marker='s')
    
    # Plot estimated
    plot_trajectory(estimated, ax=ax, color='red',
                   label='Estimated', marker='o')
    
    # Draw error lines
    if show_errors:
        for gt, est in zip(ground_truth, estimated):
            ax.plot([gt[0], est[0]], [gt[1], est[1]], 
                   'k--', alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    return ax


def visualize_localization(
    estimated_position: Tuple[float, float],
    landmarks: Dict[str, Tuple[float, float]],
    distances: Dict[str, float],
    ground_truth: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = 'Localization Result',
    show_circles: bool = True,
    **kwargs
) -> plt.Axes:
    """
    Visualize a single localization result with distance circles.
    
    Args:
        estimated_position: Estimated (x, y) position
        landmarks: Dictionary of landmark positions
        distances: Dictionary of distances to each landmark
        ground_truth: Optional ground truth position
        ax: Matplotlib axes (created if None)
        title: Plot title
        show_circles: Show distance circles around landmarks
        **kwargs: Additional arguments
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot landmarks
    colors = list(mcolors.TABLEAU_COLORS.values())
    used_landmarks = [lid for lid in distances.keys() if lid in landmarks]
    
    for i, lid in enumerate(used_landmarks):
        lx, ly = landmarks[lid]
        color = colors[i % len(colors)]
        
        # Plot landmark
        ax.scatter([lx], [ly], c=color, marker='^', s=150, 
                  label=f'{lid}', zorder=5)
        ax.annotate(lid, (lx, ly), textcoords="offset points",
                   xytext=(5, 5), fontsize=10, fontweight='bold')
        
        # Plot distance circle
        if show_circles and lid in distances:
            circle = Circle((lx, ly), distances[lid], fill=False,
                          color=color, linestyle='--', alpha=0.5)
            ax.add_patch(circle)
    
    # Plot estimated position
    ax.scatter([estimated_position[0]], [estimated_position[1]], 
              c='red', marker='*', s=300, label='Estimated', zorder=10)
    
    # Plot ground truth
    if ground_truth:
        ax.scatter([ground_truth[0]], [ground_truth[1]],
                  c='green', marker='s', s=200, label='Ground Truth', zorder=10)
        
        # Draw error line
        ax.plot([ground_truth[0], estimated_position[0]],
               [ground_truth[1], estimated_position[1]],
               'k--', alpha=0.5, linewidth=2)
        
        # Calculate error
        error = np.sqrt((estimated_position[0] - ground_truth[0])**2 +
                       (estimated_position[1] - ground_truth[1])**2)
        ax.set_title(f'{title}\nError: {error:.4f}')
    else:
        ax.set_title(title)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    return ax


def plot_error_histogram(
    errors: List[float],
    ax: Optional[plt.Axes] = None,
    bins: int = 20,
    title: str = 'Localization Error Distribution',
    xlabel: str = 'Error',
    **kwargs
) -> plt.Axes:
    """
    Plot histogram of localization errors.
    
    Args:
        errors: List of error values
        ax: Matplotlib axes (created if None)
        bins: Number of histogram bins
        title: Plot title
        xlabel: X-axis label
        **kwargs: Additional histogram arguments
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(errors, bins=bins, edgecolor='black', alpha=0.7, **kwargs)
    
    # Add statistics
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    std_error = np.std(errors)
    
    ax.axvline(mean_error, color='red', linestyle='--', 
               label=f'Mean: {mean_error:.4f}')
    ax.axvline(median_error, color='green', linestyle='-.',
               label=f'Median: {median_error:.4f}')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.set_title(f'{title}\n(std: {std_error:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_error_over_time(
    errors: List[float],
    timestamps: Optional[List[float]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = 'Localization Error Over Time',
    **kwargs
) -> plt.Axes:
    """
    Plot localization error over time/sequence.
    
    Args:
        errors: List of error values
        timestamps: Optional list of timestamps
        ax: Matplotlib axes (created if None)
        title: Plot title
        **kwargs: Additional plot arguments
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    
    if timestamps is None:
        timestamps = list(range(len(errors)))
    
    ax.plot(timestamps, errors, 'b-o', markersize=4, **kwargs)
    ax.axhline(np.mean(errors), color='red', linestyle='--',
               label=f'Mean: {np.mean(errors):.4f}')
    
    ax.set_xlabel('Time/Sequence')
    ax.set_ylabel('Error')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def create_evaluation_figure(
    ground_truth: List[Tuple[float, float]],
    estimated: List[Tuple[float, float]],
    landmarks: Dict[str, Tuple[float, float]],
    title: str = 'LanBLoc Evaluation',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a comprehensive evaluation figure with multiple subplots.
    
    Args:
        ground_truth: List of ground truth positions
        estimated: List of estimated positions
        landmarks: Dictionary of landmark positions
        title: Figure title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Calculate errors
    errors = [np.sqrt((gt[0] - est[0])**2 + (gt[1] - est[1])**2)
              for gt, est in zip(ground_truth, estimated)]
    
    # Subplot 1: Trajectory comparison
    ax1 = fig.add_subplot(2, 2, 1)
    plot_ground_truth_vs_estimated(ground_truth, estimated, landmarks, ax=ax1,
                                   title='Trajectory Comparison')
    
    # Subplot 2: Error histogram
    ax2 = fig.add_subplot(2, 2, 2)
    plot_error_histogram(errors, ax=ax2)
    
    # Subplot 3: Error over time
    ax3 = fig.add_subplot(2, 2, 3)
    plot_error_over_time(errors, ax=ax3)
    
    # Subplot 4: Error statistics table
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    stats_text = f"""
    Evaluation Statistics
    =====================
    
    Samples:     {len(errors)}
    RMSE:        {np.sqrt(np.mean(np.array(errors)**2)):.6f}
    MAE:         {np.mean(errors):.6f}
    Median:      {np.median(errors):.6f}
    Std:         {np.std(errors):.6f}
    Min Error:   {np.min(errors):.6f}
    Max Error:   {np.max(errors):.6f}
    
    Landmarks:   {len(landmarks)}
    """
    
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
             fontsize=12, verticalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig
