from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9


def plot_mdist_vs_criterion(data: pd.DataFrame,
                           save_path: Optional[str] = None) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Plot m-distance
    ax.plot(data['c'], data['m_dist_avg'], 'o-', 
            label='m-distance', linewidth=2, markersize=4)
    
    # Add reference line at mean
    mean_mdist = data['m_dist_avg'].mean()
    ax.axhline(mean_mdist, color='gray', linestyle='--', alpha=0.5,
              label=f'Mean = {mean_mdist:.3f}')
    
    ax.set_xlabel('Type 1 Criterion (c)')
    ax.set_ylabel('M-distance')
    ax.set_title('M-distance vs Criterion\n(Demonstrating Invariance)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_mean_conf_vs_dprime(data: pd.DataFrame,
                            save_path: Optional[str] = None) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    
    ax.plot(data['d_prime'], data['mean_conf'], 'o-',
            linewidth=2, markersize=4, color='darkblue')
    
    ax.set_xlabel("Type 1 Sensitivity (d')")
    ax.set_ylabel('Mean Confidence')
    ax.set_title("Mean Confidence vs d'\n(Showing Sensitivity)")
    ax.grid(alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_all_metrics_vs_parameter(data: pd.DataFrame,
                                  param_name: str,
                                  metrics: List[str] = None,
                                  save_path: Optional[str] = None) -> Figure:
    if metrics is None:
        metrics = ['m_dist_avg', 'm_dist2_avg', 'mean_conf']
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(6, 4*len(metrics)))
    
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        axes[i].plot(data[param_name], data[metric], 'o-',
                    linewidth=2, markersize=4)
        axes[i].set_xlabel(param_name)
        axes[i].set_ylabel(metric)
        axes[i].set_title(f'{metric} vs {param_name}')
        axes[i].grid(alpha=0.3)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_correlation_matrix(data: pd.DataFrame,
                           metrics: List[str] = None,
                           method: str = 'pearson',
                           save_path: Optional[str] = None) -> Figure:
    if metrics is None:
        metrics = ['d_prime', 'c', 'meta_d', 'm_dist_avg', 
                  'm_dist2_avg', 'mean_conf']
    
    # Filter to available metrics
    available = [m for m in metrics if m in data.columns]
    
    # Compute correlation matrix
    corr_mat = data[available].corr(method=method)
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    
    sns.heatmap(corr_mat, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True,
                cbar_kws={'label': f'{method.capitalize()} r'},
                ax=ax)
    
    ax.set_title(f'{method.capitalize()} Correlation Matrix')
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_bland_altman(x: np.ndarray,
                     y: np.ndarray,
                     labels: Tuple[str, str] = ('Method 1', 'Method 2'),
                     save_path: Optional[str] = None) -> Figure:
    from .stats import bland_altman_analysis
    
    # Compute statistics
    ba = bland_altman_analysis(x, y)
    
    # Calculate plotting values
    mean_vals = (x + y) / 2
    diff_vals = x - y
    
    # Create plot
    fig, ax = plt.subplots(figsize=(7, 6))
    
    ax.scatter(mean_vals, diff_vals, alpha=0.6, s=30)
    
    # Mean difference line
    ax.axhline(ba['bias'], color='blue', linestyle='-', linewidth=2,
              label=f"Bias = {ba['bias']:.3f}")
    
    # Limits of agreement
    ax.axhline(ba['upper_limit'], color='red', linestyle='--', linewidth=1.5,
              label=f"ULoA = {ba['upper_limit']:.3f}")
    ax.axhline(ba['lower_limit'], color='red', linestyle='--', linewidth=1.5,
              label=f"LLoA = {ba['lower_limit']:.3f}")
    
    # Zero line
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel(f'Mean of {labels[0]} and {labels[1]}')
    ax.set_ylabel(f'Difference ({labels[0]} - {labels[1]})')
    ax.set_title('Bland-Altman Plot')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_scatter_with_identity(x: np.ndarray,
                               y: np.ndarray,
                               labels: Tuple[str, str] = ('X', 'Y'),
                               save_path: Optional[str] = None) -> Figure:
    from .stats import pearson_correlation
    
    # Compute correlation
    corr = pearson_correlation(x, y)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.scatter(x, y, alpha=0.6, s=30)
    
    # Identity line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='Identity')
    
    # Regression line
    mask = np.isfinite(x) & np.isfinite(y)
    z = np.polyfit(x[mask], y[mask], 1)
    p = np.poly1d(z)
    ax.plot(x[mask], p(x[mask]), 'r-', alpha=0.5, linewidth=1.5,
           label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
    
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(f"r = {corr['r']:.3f}, p = {corr['p']:.4f}")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_simulation_summary(results: Dict[str, pd.DataFrame],
                           save_dir: str = 'outputs/figures') -> Dict[str, Figure]:
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    figures = {}
    
    # Plot 1: M-distance vs criterion
    if 'vary_criterion' in results:
        fig = plot_mdist_vs_criterion(
            results['vary_criterion'],
            save_path=f"{save_dir}/mdist_vs_criterion.png"
        )
        figures['mdist_vs_criterion'] = fig
    
    # Plot 2: M-distance vs d'
    if 'vary_dprime' in results:
        fig = plot_all_metrics_vs_parameter(
            results['vary_dprime'],
            param_name='dprime',
            metrics=['m_dist_avg', 'm_dist2_avg', 'mean_conf'],
            save_path=f"{save_dir}/metrics_vs_dprime.png"
        )
        figures['metrics_vs_dprime'] = fig
    
    # Plot 3: Effects of tau parameters
    if 'vary_tauN' in results:
        fig = plot_all_metrics_vs_parameter(
            results['vary_tauN'],
            param_name='tauN',
            metrics=['m_dist_avg', 'm_dist2_avg'],
            save_path=f"{save_dir}/metrics_vs_tauN.png"
        )
        figures['metrics_vs_tauN'] = fig
    
    # Plot 4: Correlation matrix
    if 'vary_criterion' in results:
        fig = plot_correlation_matrix(
            results['vary_criterion'],
            save_path=f"{save_dir}/correlation_matrix.png"
        )
        figures['correlation_matrix'] = fig
    
    print(f"Generated {len(figures)} figures in {save_dir}/")
    
    return figures


def plot_heatmap_2d(data: pd.DataFrame,
                   x_var: str,
                   y_var: str,
                   z_var: str,
                   save_path: Optional[str] = None) -> Figure:
    # Pivot data for heatmap
    pivot = data.pivot_table(values=z_var, index=y_var, columns=x_var)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(pivot, annot=False, cmap='viridis', ax=ax,
                cbar_kws={'label': z_var})
    
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_title(f'{z_var} as function of {x_var} and {y_var}')
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    
    return fig
