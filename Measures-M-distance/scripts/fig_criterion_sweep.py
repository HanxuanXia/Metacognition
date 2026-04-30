import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from sdt_shared import (
    SDTParameters, simulate_sdt_trials, compute_all_metrics,
    COLORS, OUTPUT_DIR_FIG4, aggregate_replicates, plt, t_dist,
)

OUT = OUTPUT_DIR_FIG4

def run_simulations(sigma_meta=0.001):
    """Run all three sweep simulations for Figure 4."""
    print("=" * 70)
    print(f"FIGURE 4 - Criterion Sweeps (σ_meta = {sigma_meta})")
    print("=" * 70)

    dprime       = 2.0
    n_trials     = 50000
    n_replicates = 20

    print("\n1. Criterion (c) sweep...")
    c_vals = np.linspace(-1.0, 1.0, 21)
    results_crit = []
    tauN, tauY = -0.5, 0.5

    for i, c in enumerate(tqdm(c_vals, desc="c")):
        for rep in range(n_replicates):
            params = SDTParameters(
                dprime=dprime, c=c, tauN=tauN, tauY=tauY,
                sigma_meta=sigma_meta, nTrials=n_trials, nRatings=2,
            )
            seed = i*100 + rep*1000 + int(sigma_meta*10000)
            trials  = simulate_sdt_trials(params, seed=seed)
            metrics = compute_all_metrics(trials, nRatings=2)
            results_crit.append({
                'c': c, 'replicate': rep, 'sigma_meta': sigma_meta,
                'd_prime':      metrics['d_prime'],
                'meta_d':       metrics['meta_d'],
                'M_ratio':      metrics['M_ratio'],
                'm_dist_R1':    metrics['m_dist_R1'],
                'm_dist_R2':    metrics['m_dist_R2'],
                'm_dist2_R1':   metrics['m_dist2_R1'],
                'm_dist2_R2':   metrics['m_dist2_R2'],
                'mean_conf_R1': metrics['mean_conf_R1'],
                'mean_conf_R2': metrics['mean_conf_R2'],
            })
    df_crit = pd.DataFrame(results_crit)

    print("\n2. τ_N sweep...")
    tauN_vals = np.linspace(-1.5, -0.05, 21)
    results_tauN = []
    c, tauY = 0.0, 0.5

    for i, tauN in enumerate(tqdm(tauN_vals, desc="τ_N")):
        for rep in range(n_replicates):
            params = SDTParameters(
                dprime=dprime, c=c, tauN=tauN, tauY=tauY,
                sigma_meta=sigma_meta, nTrials=n_trials, nRatings=2,
            )
            seed = i*100 + rep*2000 + int(sigma_meta*10000)
            trials  = simulate_sdt_trials(params, seed=seed)
            metrics = compute_all_metrics(trials, nRatings=2)
            results_tauN.append({
                'tauN': tauN, 'replicate': rep, 'sigma_meta': sigma_meta,
                'm_dist_R1':    metrics['m_dist_R1'],
                'm_dist_R2':    metrics['m_dist_R2'],
                'm_dist2_R1':   metrics['m_dist2_R1'],
                'm_dist2_R2':   metrics['m_dist2_R2'],
                'mean_conf_R1': metrics['mean_conf_R1'],
                'mean_conf_R2': metrics['mean_conf_R2'],
            })
    df_tauN = pd.DataFrame(results_tauN)

    print("\n3. τ_Y sweep...")
    tauY_vals = np.linspace(0.05, 1.5, 21)
    results_tauY = []
    tauN = -0.5

    for i, tauY in enumerate(tqdm(tauY_vals, desc="τ_Y")):
        for rep in range(n_replicates):
            params = SDTParameters(
                dprime=dprime, c=c, tauN=tauN, tauY=tauY,
                sigma_meta=sigma_meta, nTrials=n_trials, nRatings=2,
            )
            seed = i*100 + rep*3000 + int(sigma_meta*10000)
            trials  = simulate_sdt_trials(params, seed=seed)
            metrics = compute_all_metrics(trials, nRatings=2)
            results_tauY.append({
                'tauY': tauY, 'replicate': rep, 'sigma_meta': sigma_meta,
                'm_dist_R1':    metrics['m_dist_R1'],
                'm_dist_R2':    metrics['m_dist_R2'],
                'm_dist2_R1':   metrics['m_dist2_R1'],
                'm_dist2_R2':   metrics['m_dist2_R2'],
                'mean_conf_R1': metrics['mean_conf_R1'],
                'mean_conf_R2': metrics['mean_conf_R2'],
            })
    df_tauY = pd.DataFrame(results_tauY)

    return df_crit, df_tauN, df_tauY

def plot(df_crit, df_tauN, df_tauY, sigma_meta, save_path):
    """Generate layout with A (1x3) on left for Criterion, B (2x3) on right for tauN/tauY."""
    df_crit_agg = aggregate_replicates(
        df_crit, ['c'],
        ['mean_conf_R1', 'mean_conf_R2', 'm_dist_R1', 'm_dist_R2',
         'm_dist2_R1', 'm_dist2_R2'],
    )
    df_tauN_agg = aggregate_replicates(
        df_tauN, ['tauN'],
        ['mean_conf_R1', 'mean_conf_R2', 'm_dist_R1', 'm_dist_R2',
         'm_dist2_R1', 'm_dist2_R2'],
    )
    df_tauY_agg = aggregate_replicates(
        df_tauY, ['tauY'],
        ['mean_conf_R1', 'mean_conf_R2', 'm_dist_R1', 'm_dist_R2',
         'm_dist2_R1', 'm_dist2_R2'],
    )

    n_rep     = df_crit['replicate'].nunique()
    ci_factor = t_dist.ppf(0.975, df=n_rep - 1)

    def _panel(ax, x_vals, agg_df, key, xlabel, ylabel,
               ylim=None, hline=False):
        lines = {}
        for resp, color, marker in [
            ('R1', COLORS['R1'], 'o'),
            ('R2', COLORS['R2'], 's'),
        ]:
            y     = agg_df[f'{key}_{resp}_mean'].values
            y_err = agg_df[f'{key}_{resp}_sem'].values * ci_factor
            (ln,) = ax.plot(x_vals, y, color=color,
                            linewidth=1.8, marker=marker, markersize=4,
                            markevery=2, zorder=3)
            ax.fill_between(x_vals, y - y_err, y + y_err,
                            color=color, alpha=0.15, zorder=2)
            lines[resp] = ln
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
        if ylim:
            ax.set_ylim(*ylim)
        if hline:
            ax.axhline(0, color='#999999', linestyle='--',
                       linewidth=0.8, alpha=0.6)
        return lines

    # Layout: Left (1x3) for A group, Right (2x3) for B group
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)
    
    # LEFT SIDE - GROUP A: Criterion sweep (1x3, spans 3 rows, 1 column)
    ax_A1 = fig.add_subplot(gs[0, 0])
    ax_A2 = fig.add_subplot(gs[1, 0])
    ax_A3 = fig.add_subplot(gs[2, 0])
    
    # RIGHT SIDE - GROUP B: tauN and tauY (2x3 grid, columns 1-2)
    ax_B1 = fig.add_subplot(gs[0, 1])   # Mean Conf vs tauN
    ax_B2 = fig.add_subplot(gs[0, 2])   # Mean Conf vs tauY
    ax_B3 = fig.add_subplot(gs[1, 1])   # m-dist vs tauN
    ax_B4 = fig.add_subplot(gs[1, 2])   # m-dist vs tauY
    ax_B5 = fig.add_subplot(gs[2, 1])   # m-dist2 vs tauN
    ax_B6 = fig.add_subplot(gs[2, 2])   # m-dist2 vs tauY

    c_x    = df_crit_agg['c'].values
    tauN_x = df_tauN_agg['tauN'].values
    tauY_x = df_tauY_agg['tauY'].values

    # GROUP A: Criterion sweep (left, 1x3)
    lns_A = _panel(ax_A1, c_x, df_crit_agg, 'mean_conf', 
                   'Criterion', 'Prop. Confident')
    _panel(ax_A2, c_x, df_crit_agg, 'm_dist', 
           'Criterion', 'm-distance', ylim=(0, 0.5), hline=True)
    _panel(ax_A3, c_x, df_crit_agg, 'm_dist2', 
           'Criterion', 'm-distance2', ylim=(0, 1), hline=True)

    # GROUP B: tauN and tauY sweeps (right, 2x3)
    # Row 1: Mean Confidence
    _panel(ax_B1, tauN_x, df_tauN_agg, 'mean_conf', 
           'C2$_{R1}$', 'Prop. Confident')
    _panel(ax_B2, tauY_x, df_tauY_agg, 'mean_conf', 
           'C2$_{R2}$', 'Prop. Confident')    # Row 2: m-distance
    _panel(ax_B3, tauN_x, df_tauN_agg, 'm_dist', 
           'C2$_{R1}$', 'm-distance', hline=True)
    _panel(ax_B4, tauY_x, df_tauY_agg, 'm_dist', 
           'C2$_{R2}$', 'm-distance', hline=True)

    # Row 3: m-distance2
    _panel(ax_B5, tauN_x, df_tauN_agg, 'm_dist2', 
           'C2$_{R1}$', 'm-distance2', hline=True)
    lns = _panel(ax_B6, tauY_x, df_tauY_agg, 'm_dist2', 
           'C2$_{R2}$', 'm-distance2', hline=True)

    # Add title with sigma_meta
    fig.suptitle(r'$\sigma_{\mathrm{meta}} = ' + f'{sigma_meta}' + r'$',
                fontsize=13, fontweight='bold', y=0.995)

    # Add group labels A and B - positioned appropriately
    # Group A label (left side, 1x3 column)
    fig.text(0.05, 0.96, 'A', fontsize=20, fontweight='bold', 
             ha='left', va='top', transform=fig.transFigure)
    
    # Group B label (right side, 2x3 columns)
    fig.text(0.35, 0.96, 'B', fontsize=20, fontweight='bold', 
             ha='left', va='top', transform=fig.transFigure)

    fig.legend(
        handles=[lns['R1'], lns['R2']],
        labels=['Respond R1', 'Respond R2'],
        loc='lower center',
        ncol=2,
        frameon=True,
        framealpha=0.9,
        edgecolor='#cccccc',
        fontsize=11,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved → {save_path}")
    return fig


def run_simulations_with_noise(sigma_meta=0.5):
    """Run criterion sweep with specified metacognitive noise level."""
    print("=" * 70)
    print(f"FIGURE 4B – Criterion sweep with σ_meta = {sigma_meta}")
    print("=" * 70)

    dprime       = 2.0
    n_trials     = 50000
    n_replicates = 20

    print(f"\nCriterion (c) sweep with noise σ_meta = {sigma_meta}...")
    c_vals = np.linspace(-1.0, 1.0, 21)
    results_crit = []
    tauN, tauY = -0.5, 0.5

    for i, c in enumerate(tqdm(c_vals, desc="c")):
        for rep in range(n_replicates):
            params = SDTParameters(
                dprime=dprime, c=c, tauN=tauN, tauY=tauY,
                sigma_meta=sigma_meta, nTrials=n_trials, nRatings=2,
            )
            trials  = simulate_sdt_trials(params, seed=i*100 + rep*10000 + int(sigma_meta*1000))
            metrics = compute_all_metrics(trials, nRatings=2)
            results_crit.append({
                'c': c, 'replicate': rep, 'sigma_meta': sigma_meta,
                'd_prime':      metrics['d_prime'],
                'meta_d':       metrics['meta_d'],
                'M_ratio':      metrics['M_ratio'],
                'm_dist_R1':    metrics['m_dist_R1'],
                'm_dist_R2':    metrics['m_dist_R2'],
                'm_dist2_R1':   metrics['m_dist2_R1'],
                'm_dist2_R2':   metrics['m_dist2_R2'],
                'mean_conf_R1': metrics['mean_conf_R1'],
                'mean_conf_R2': metrics['mean_conf_R2'],
            })
    return pd.DataFrame(results_crit)


def plot_criterion_single(df_criterion, sigma_meta, save_path):
    """Plot criterion sweep for a single σ_meta value."""
    
    df_agg = aggregate_replicates(
        df_criterion, ['c'],
        ['mean_conf_R1', 'mean_conf_R2', 'm_dist_R1', 'm_dist_R2',
         'm_dist2_R1', 'm_dist2_R2'],
    )
    n_rep = df_criterion['replicate'].nunique()
    ci_factor = t_dist.ppf(0.975, df=n_rep - 1)
    c_x = df_agg['c'].values
    
    fig, axes = plt.subplots(1, 3, figsize=(13, 4),
                            gridspec_kw={'hspace': 0.40, 'wspace': 0.35})
    
    lines = {}
    
    # Panel 1: Mean Confidence
    ax = axes[0]
    for resp, color, marker in [('R1', COLORS['R1'], 'o'), ('R2', COLORS['R2'], 's')]:
        y     = df_agg[f'mean_conf_{resp}_mean'].values
        y_err = df_agg[f'mean_conf_{resp}_sem'].values * ci_factor
        (ln,) = ax.plot(c_x, y, color=color,
                       linewidth=1.8, marker=marker, markersize=4,
                       markevery=2, zorder=3)
        ax.fill_between(c_x, y - y_err, y + y_err,
                       color=color, alpha=0.15, zorder=2)
        lines[resp] = ln
    ax.set_xlabel('Criterion $c$', fontsize=10)
    ax.set_ylabel('Prop. Confident', fontsize=10)
    ax.grid(alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)
    
    # Panel 2: m-distance
    ax = axes[1]
    for resp, color, marker in [('R1', COLORS['R1'], 'o'), ('R2', COLORS['R2'], 's')]:
        y     = df_agg[f'm_dist_{resp}_mean'].values
        y_err = df_agg[f'm_dist_{resp}_sem'].values * ci_factor
        ax.plot(c_x, y, color=color,
               linewidth=1.8, marker=marker, markersize=4,
               markevery=2, zorder=3)
        ax.fill_between(c_x, y - y_err, y + y_err,
                       color=color, alpha=0.15, zorder=2)
    ax.set_xlabel('Criterion $c$', fontsize=10)
    ax.set_ylabel('m-distance', fontsize=10)
    ax.axhline(0, color='#999999', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_ylim(-0.05, 0.5)
    ax.grid(alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)
    
    # Panel 3: m-distance2
    ax = axes[2]
    for resp, color, marker in [('R1', COLORS['R1'], 'o'), ('R2', COLORS['R2'], 's')]:
        y     = df_agg[f'm_dist2_{resp}_mean'].values
        y_err = df_agg[f'm_dist2_{resp}_sem'].values * ci_factor
        ax.plot(c_x, y, color=color,
               linewidth=1.8, marker=marker, markersize=4,
               markevery=2, zorder=3)
        ax.fill_between(c_x, y - y_err, y + y_err,
                       color=color, alpha=0.15, zorder=2)
    ax.set_xlabel('Criterion $c$', fontsize=10)
    ax.set_ylabel('m-distance2', fontsize=10)
    ax.axhline(0, color='#999999', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_ylim(-0.05, 1.0)
    ax.grid(alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)
    
    # Add title with sigma_meta value
    fig.suptitle(r'$\sigma_{\mathrm{meta}} = ' + f'{sigma_meta}' + r'$',
                fontsize=12, fontweight='bold', y=1.02)
    
    # Shared legend
    fig.legend(
        handles=[lines['R1'], lines['R2']],
        labels=['Respond R1', 'Respond R2'],
        loc='lower center', ncol=2,
        frameon=True, framealpha=0.9, edgecolor='#cccccc',
        fontsize=11, bbox_to_anchor=(0.5, -0.15),
    )
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved → {save_path}")
    return fig


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)

    print("\n" + "=" * 70)
    print("GENERATING FIGURE 4 - CRITERION SWEEP ACROSS NOISE LEVELS")
    print("=" * 70)
    
    # Figure 4 with σ_meta = 0.001 (Low Noise)
    df_crit_low, df_tauN_low, df_tauY_low = run_simulations(sigma_meta=0.001)
    df_crit_low.to_csv(f'{OUT}/sdt_criterion_lowNoise.csv', index=False)
    df_tauN_low.to_csv(f'{OUT}/sdt_tauN_lowNoise.csv',      index=False)
    df_tauY_low.to_csv(f'{OUT}/sdt_tauY_lowNoise.csv',      index=False)
    
    plot(df_crit_low, df_tauN_low, df_tauY_low, sigma_meta=0.001,
         save_path=f'{OUT}/criterion_sweep_sigma0p001.png')

    # Figure 4 with σ_meta = 0.5 (High Noise)
    df_crit_high, df_tauN_high, df_tauY_high = run_simulations(sigma_meta=0.5)
    df_crit_high.to_csv(f'{OUT}/sdt_criterion_highNoise.csv', index=False)
    df_tauN_high.to_csv(f'{OUT}/sdt_tauN_highNoise.csv',      index=False)
    df_tauY_high.to_csv(f'{OUT}/sdt_tauY_highNoise.csv',      index=False)
    
    plot(df_crit_high, df_tauN_high, df_tauY_high, sigma_meta=0.5,
         save_path=f'{OUT}/criterion_sweep_sigma0p5.png')

    print("\n" + "=" * 70)
    print("Figure 4 complete.")
    print("=" * 70)
