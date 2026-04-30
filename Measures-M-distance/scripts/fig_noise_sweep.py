import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from sdt_shared import (
    SDTParameters, simulate_sdt_trials, compute_all_metrics,
    COLORS, OUTPUT_DIR_FIG5, aggregate_replicates, plt, t_dist,
)

OUT = OUTPUT_DIR_FIG5

def run_noise_sweep():
    print("=" * 70)
    print("Metacognitive Noise (σ_meta) Sweep")
    print("=" * 70)

    dprime       = 2.0
    c            = 0.0
    tauN         = -0.5
    tauY         =  0.5
    n_trials     = 50000
    n_replicates = 20

    # Extended noise sweep: from 0.001 to 100 with logarithmic spacing
    sigma_vals = np.logspace(-3, 2, 35)

    results = []
    for i, sigma_meta in enumerate(tqdm(sigma_vals, desc="σ_meta")):
        for rep in range(n_replicates):
            params = SDTParameters(
                dprime=dprime, c=c,
                tauN=tauN, tauY=tauY,
                sigma_meta=sigma_meta,
                nTrials=n_trials, nRatings=2,
            )
            trials  = simulate_sdt_trials(params, seed=i*100 + rep*4000)
            metrics = compute_all_metrics(trials, nRatings=2)
            results.append({
                'sigma_meta':   sigma_meta,
                'replicate':    rep,
                'mean_conf_R1': metrics['mean_conf_R1'],
                'mean_conf_R2': metrics['mean_conf_R2'],
                'm_dist_R1':    metrics['m_dist_R1'],
                'm_dist_R2':    metrics['m_dist_R2'],
                'm_dist2_R1':   metrics['m_dist2_R1'],
                'm_dist2_R2':   metrics['m_dist2_R2'],
                'meta_d':       metrics['meta_d'],
                'M_ratio':      metrics['M_ratio'],
            })

    return pd.DataFrame(results)


def plot(df_noise, save_path):
    agg = aggregate_replicates(
        df_noise, ['sigma_meta'],
        ['mean_conf_R1', 'mean_conf_R2',
         'm_dist_R1',   'm_dist_R2',
         'm_dist2_R1',  'm_dist2_R2'],
    )

    sigma_vals = agg['sigma_meta'].values
    n_rep      = df_noise['replicate'].nunique()
    ci_factor  = t_dist.ppf(0.975, df=n_rep - 1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2),
                             gridspec_kw={'wspace': 0.35})

    panels = [
        ('mean_conf', 'Prop. Confident'),
        ('m_dist',    'm-distance'),
        ('m_dist2',   'm-distance2'),
    ]

    lines = {}
    for ax, (key, ylabel) in zip(axes, panels):
        for resp, color, marker in [
            ('R1', COLORS['R1'], 'o'),
            ('R2', COLORS['R2'], 's'),
        ]:
            y     = agg[f'{key}_{resp}_mean'].values
            y_err = agg[f'{key}_{resp}_sem'].values * ci_factor
            (ln,) = ax.plot(sigma_vals, y, color=color,
                            linewidth=1.8, marker=marker, markersize=6,
                            zorder=3)
            ax.fill_between(sigma_vals, y - y_err, y + y_err,
                            color=color, alpha=0.15, zorder=2)
            lines[resp] = ln

        ax.set_xscale('log')
        ax.set_xlabel('σ_meta', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(alpha=0.25, which='both')
        ax.spines[['top', 'right']].set_visible(False)

    # Shared legend
    fig.legend(
        handles=[lines['R1'], lines['R2']],
        labels=['Respond R1', 'Respond R2'],
        loc='lower center', ncol=2,
        frameon=True, framealpha=0.9, edgecolor='#cccccc',
        fontsize=11, bbox_to_anchor=(0.5, -0.08),
    )

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure 5 saved → {save_path}")
    return fig


def plot_supplementary(df_noise, save_path):
    agg = aggregate_replicates(
        df_noise, ['sigma_meta'],
        ['meta_d', 'M_ratio'],
    )

    sigma_vals = agg['sigma_meta'].values
    n_rep      = df_noise['replicate'].nunique()
    ci_factor  = t_dist.ppf(0.975, df=n_rep - 1)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4),
                             gridspec_kw={'wspace': 0.38})

    # Panel A: meta-d′
    ax = axes[0]
    y  = agg['meta_d_mean'].values
    ye = agg['meta_d_sem'].values * ci_factor
    ax.plot(sigma_vals, y, color=COLORS['gray'],
            linewidth=1.8, marker='D', markersize=6, zorder=3)
    ax.fill_between(sigma_vals, y - ye, y + ye,
                    color=COLORS['gray'], alpha=0.15, zorder=2)
    ax.set_xscale('log')
    ax.set_xlabel('σ_meta', fontsize=10)
    ax.set_ylabel("meta-d′", fontsize=10)
    ax.grid(alpha=0.25, which='both')
    ax.spines[['top', 'right']].set_visible(False)

    # Panel B: M-ratio
    ax = axes[1]
    y  = agg['M_ratio_mean'].values
    ye = agg['M_ratio_sem'].values * ci_factor
    ax.plot(sigma_vals, y, color=COLORS['orange'],
            linewidth=1.8, marker='^', markersize=6,
            linestyle='--', zorder=3)
    ax.fill_between(sigma_vals, y - ye, y + ye,
                    color=COLORS['orange'], alpha=0.15, zorder=2)
    ax.set_xscale('log')
    ax.set_xlabel('σ_meta', fontsize=10)
    ax.set_ylabel('M-ratio', fontsize=10)
    ax.grid(alpha=0.25, which='both')
    ax.spines[['top', 'right']].set_visible(False)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure S1 saved → {save_path}")
    return fig


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)

    df_noise = run_noise_sweep()
    df_noise.to_csv(f'{OUT}/sdt_noise_sweep.csv', index=False)

    plot(df_noise, save_path=f'{OUT}/noise_sweep.png')
    plot_supplementary(df_noise, save_path=f'{OUT}/metad_mratio_vs_noise.png')

    print("\n" + "=" * 70)
    print("Figure 5 + Figure S1 complete.")
    print("=" * 70)
