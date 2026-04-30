import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from sdt_shared import (
    SDTParameters, simulate_sdt_trials, compute_all_metrics,
    COLORS, OUTPUT_DIR_FIG6, aggregate_replicates, plt, t_dist,
)

OUT = OUTPUT_DIR_FIG6

def run_metad_sweep(sigma_meta=0.001):
    print("=" * 70)
    print(f"- meta-d' (d') Sweep  (σ_meta = {sigma_meta})")
    print("=" * 70)

    c            = 0.0
    tauN         = -0.5
    tauY         =  0.5
    n_trials     = 50000
    n_replicates = 20

    dprime_vals = [0.5, 1.0, 1.5, 2.0]

    results = []
    for i, dprime in enumerate(tqdm(dprime_vals, desc="d'")):
        for rep in range(n_replicates):
            params = SDTParameters(
                dprime=dprime, c=c,
                tauN=tauN, tauY=tauY,
                sigma_meta=sigma_meta,
                nTrials=n_trials, nRatings=2,
            )
            seed = i*100 + rep*5000 + int(sigma_meta*10000)
            trials  = simulate_sdt_trials(params, seed=seed)
            metrics = compute_all_metrics(trials, nRatings=2)
            results.append({
                'dprime':       dprime,
                'replicate':    rep,
                'sigma_meta':   sigma_meta,
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

def plot(df_metad, sigma_meta, save_path):
    agg = aggregate_replicates(
        df_metad, ['dprime'],
        ['mean_conf_R1', 'mean_conf_R2',
         'm_dist_R1',   'm_dist_R2',
         'm_dist2_R1',  'm_dist2_R2'],
    )

    dprime_vals = agg['dprime'].values
    n_rep       = df_metad['replicate'].nunique()
    ci_factor   = t_dist.ppf(0.975, df=n_rep - 1)

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
            (ln,) = ax.plot(dprime_vals, y, color=color,
                            linewidth=1.8, marker=marker, markersize=6,
                            zorder=3)
            ax.fill_between(dprime_vals, y - y_err, y + y_err,
                            color=color, alpha=0.15, zorder=2)
            lines[resp] = ln

        ax.set_xlabel("d′", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)

    # Add title with sigma_meta
    fig.suptitle(r'$\sigma_{\mathrm{meta}} = ' + f'{sigma_meta}' + r'$',
                fontsize=12, fontweight='bold', y=1.02)

    # Shared legend
    fig.legend(
        handles=[lines['R1'], lines['R2']],
        labels=['Respond R1', 'Respond R2'],
        loc='lower center', ncol=2,
        frameon=True, framealpha=0.9, edgecolor='#cccccc',
        fontsize=11, bbox_to_anchor=(0.5, -0.08),
    )

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved → {save_path}")
    return fig


def plot_supplementary(df_metad, sigma_meta, save_path):
    agg = aggregate_replicates(df_metad, ['dprime'], ['meta_d'])

    dprime_vals = agg['dprime'].values
    n_rep       = df_metad['replicate'].nunique()
    ci_factor   = t_dist.ppf(0.975, df=n_rep - 1)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    y  = agg['meta_d_mean'].values
    ye = agg['meta_d_sem'].values * ci_factor

    ax.plot(dprime_vals, y, color=COLORS['gray'],
            linewidth=1.8, marker='D', markersize=6,
            label="meta-d′", zorder=3)
    ax.fill_between(dprime_vals, y - ye, y + ye,
                    color=COLORS['gray'], alpha=0.15, zorder=2)
    ax.plot(dprime_vals, dprime_vals, color='black',
            linewidth=1.2, linestyle='--', label="identity (d′)")

    ax.set_xlabel("d′", fontsize=11)
    ax.set_ylabel("meta-d′", fontsize=11)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)

    # Add title
    fig.suptitle(r'$\sigma_{\mathrm{meta}} = ' + f'{sigma_meta}' + r'$',
                fontsize=11, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved → {save_path}")
    return fig


def plot_2x2(df_metad, sigma_meta, save_path):
    agg_main = aggregate_replicates(
        df_metad, ['dprime'],
        ['mean_conf_R1', 'mean_conf_R2',
         'm_dist_R1',   'm_dist_R2',
         'm_dist2_R1',  'm_dist2_R2',
         'meta_d'],
    )

    dprime_vals = agg_main['dprime'].values
    n_rep       = df_metad['replicate'].nunique()
    ci_factor   = t_dist.ppf(0.975, df=n_rep - 1)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2),
                             gridspec_kw={'wspace': 0.32, 'hspace': 0.32})
    axes = axes.ravel()

    panels = [
        ('mean_conf', 'Prop. Confident'),
        ('m_dist',    'm-distance'),
        ('m_dist2',   'm-distance2'),
    ]

    lines = {}
    for ax, (key, ylabel) in zip(axes[:3], panels):
        for resp, color, marker in [
            ('R1', COLORS['R1'], 'o'),
            ('R2', COLORS['R2'], 's'),
        ]:
            y     = agg_main[f'{key}_{resp}_mean'].values
            y_err = agg_main[f'{key}_{resp}_sem'].values * ci_factor
            (ln,) = ax.plot(dprime_vals, y, color=color,
                            linewidth=1.8, marker=marker, markersize=6,
                            zorder=3)
            ax.fill_between(dprime_vals, y - y_err, y + y_err,
                            color=color, alpha=0.15, zorder=2)
            lines[resp] = ln

        ax.set_xlabel("d′", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)

    # Bottom-right panel: meta-d' identity plot
    ax = axes[3]
    y  = agg_main['meta_d_mean'].values
    ye = agg_main['meta_d_sem'].values * ci_factor
    ax.plot(dprime_vals, y, color=COLORS['gray'],
            linewidth=1.8, marker='D', markersize=6,
            label="meta-d′", zorder=3)
    ax.fill_between(dprime_vals, y - ye, y + ye,
                    color=COLORS['gray'], alpha=0.15, zorder=2)
    ax.plot(dprime_vals, dprime_vals, color='black',
            linewidth=1.2, linestyle='--', label="identity (d′)")
    ax.set_xlabel("d′", fontsize=10)
    ax.set_ylabel("meta-d′", fontsize=10)
    ax.grid(alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc='best')

    for label, ax in zip(['A', 'B', 'C', 'D'], axes):
        ax.text(-0.14, 1.06, label, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')

    fig.suptitle(r'$\sigma_{\mathrm{meta}} = ' + f'{sigma_meta}' + r'$',
                 fontsize=13, fontweight='bold', y=0.98)

    fig.legend(
        handles=[lines['R1'], lines['R2']],
        labels=['Respond R1', 'Respond R2'],
        loc='lower center', ncol=2,
        frameon=True, framealpha=0.9, edgecolor='#cccccc',
        fontsize=10, bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved → {save_path}")
    return fig

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)

    print("\n" + "=" * 70)
    print("GENERATING FIGURE 6 - M-DISTANCE SWEEP ACROSS NOISE LEVELS")
    print("=" * 70)

    # Figure 6A: Low noise (σ_meta = 0.001)
    df_metad_low = run_metad_sweep(sigma_meta=0.001)
    df_metad_low.to_csv(f'{OUT}/sdt_metad_sweep_lowNoise.csv', index=False)
    plot(df_metad_low, sigma_meta=0.001, save_path=f'{OUT}/metad_sweep_lowNoise.png')
    plot_supplementary(df_metad_low, sigma_meta=0.001, save_path=f'{OUT}/metad_identity_lowNoise.png')
    plot_2x2(df_metad_low, sigma_meta=0.001, save_path=f'{OUT}/2x2_lowNoise.png')

    # Figure 6B: High noise (σ_meta = 0.5)
    df_metad_high = run_metad_sweep(sigma_meta=0.5)
    df_metad_high.to_csv(f'{OUT}/sdt_metad_sweep_highNoise.csv', index=False)
    plot(df_metad_high, sigma_meta=0.5, save_path=f'{OUT}/metad_sweep_highNoise.png')
    plot_supplementary(df_metad_high, sigma_meta=0.5, save_path=f'{OUT}/metad_identity_highNoise.png')
    plot_2x2(df_metad_high, sigma_meta=0.5, save_path=f'{OUT}/2x2_highNoise.png')

    print("\n" + "=" * 70)
    print("Figure 6 + Figure S2 complete.")
    print("=" * 70)
