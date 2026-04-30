import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t as t_dist
from tqdm import tqdm

# Add src to path (works regardless of where the script is called from)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mdistance import (
    SDTParameters,
    simulate_sdt_trials,
    compute_all_metrics,
)

sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'lines.linewidth': 2.0,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
})

COLORS = {
    'R1':   '#2E86AB',  # Blue
    'R2':   '#A23B72',  # Red/Purple
    'gray': '#666666',
    'orange': '#E07B39',
}

OUTPUT_DIR_FIG4 = 'outputs/fig4_criterion_sweep'
OUTPUT_DIR_FIG5 = 'outputs/fig5_noise_sweep'
OUTPUT_DIR_FIG6 = 'outputs/fig6_metad_sweep'

OUTPUT_DIR = OUTPUT_DIR_FIG4


def aggregate_replicates(df, group_cols, value_cols):
    """Group by group_cols, return mean ± SEM for each value column."""
    result = (
        df.groupby(group_cols)[value_cols]
        .agg(['mean', 'sem'])
        .reset_index()
    )
    result.columns = ['_'.join(col).strip('_') for col in result.columns.values]
    return result
