__version__ = "0.1.0"
__author__ = "Hanxuan Xia"

# Core functions
from .simulations import simulate_sdt_trials, sweep_parameter
from .metad_fitting import fit_metad_mle
from .mdistance import compute_mdist, compute_mdist2, compute_mean_confidence, compute_all_metrics
from .sdt import SDTParameters, compute_type1_sdt
from .data_io import trials_to_counts, counts_to_trials

__all__ = [
    # Simulation
    "simulate_sdt_trials",
    "sweep_parameter",
    # SDT
    "SDTParameters",
    "compute_type1_sdt",
    # Meta-d' fitting
    "fit_metad_mle",
    # M-distance metrics
    "compute_mdist",
    "compute_mdist2",
    "compute_mean_confidence",
    "compute_all_metrics",
    # Data conversion
    "trials_to_counts",
    "counts_to_trials",
]
