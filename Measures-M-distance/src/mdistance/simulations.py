from typing import Dict, Optional, List
import numpy as np
import pandas as pd
from tqdm import tqdm

from .sdt import SDTParameters


def simulate_sdt_trials(params: SDTParameters,
                       seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    nTrials = params.nTrials
    nTrials_per_stim = nTrials // 2

    # Generate evidence: S1 ~ N(-d'/2, σ1²),  S2 ~ N(+d'/2, σ2²)
    evidence_S1 = rng.normal(-params.dprime / 2, params.sigma1, nTrials_per_stim)
    evidence_S2 = rng.normal( params.dprime / 2, params.sigma2, nTrials_per_stim)

    evidence = np.concatenate([evidence_S1, evidence_S2])
    stimulus = np.concatenate([
        np.zeros(nTrials_per_stim, dtype=int),
        np.ones(nTrials_per_stim, dtype=int)
    ])

    # Shuffle trial order (more realistic; no effect on statistics)
    perm = rng.permutation(nTrials)
    evidence = evidence[perm]
    stimulus = stimulus[perm]

    # Type-1 decision
    # R2 (1) if evidence > c, else R1 (0)
    c = params.c
    response = (evidence > c).astype(int)
    correct  = (stimulus == response).astype(int)

    # Metacognitive (noisy) evidence 
    metacog_noise = rng.normal(0, params.sigma_meta, nTrials)
    evidence_meta = evidence + metacog_noise

    # Absolute positions of type-2 criteria
    #   tN = c + τN   (τN < 0  →  tN is LEFT  of c, on the R1 side)
    #   tY = c + τY   (τY > 0  →  tY is RIGHT of c, on the R2 side)
    tN = c + params.tauN
    tY = c + params.tauY

    # Type-2 confidence rule (strictly side-restricted)

    r1_mask = (response == 0)
    r2_mask = (response == 1)

    confident_R1 = r1_mask & (evidence_meta < c) & (evidence_meta < tN)
    confident_R2 = r2_mask & (evidence_meta > c) & (evidence_meta > tY)

    confidence = np.zeros(nTrials, dtype=int)
    confidence[confident_R1 | confident_R2] = 1

    # Ordinal rating scale:  1=R1-high, 2=R1-low, 3=R2-low, 4=R2-high
    rating = np.zeros(nTrials, dtype=int)
    rating[r1_mask & (confidence == 1)] = 1
    rating[r1_mask & (confidence == 0)] = 2
    rating[r2_mask & (confidence == 0)] = 3
    rating[r2_mask & (confidence == 1)] = 4

    return {
        'stimulus':      stimulus,
        'response':      response,
        'correct':       correct,
        'confidence':    confidence,
        'rating':        rating,
        'evidence':      evidence,
        'evidence_meta': evidence_meta,
        'metacog_noise': metacog_noise,
    }


def sweep_parameter(param_name: str,
                   param_values: np.ndarray,
                   default_params: SDTParameters,
                   seed: Optional[int] = None,
                   show_progress: bool = True) -> pd.DataFrame:
    
    from .mdistance import compute_all_metrics
    
    results = []
    
    iterator = tqdm(param_values) if show_progress else param_values
    
    for i, value in enumerate(iterator):
        # Create new parameters with modified value
        params_dict = {
            'dprime': default_params.dprime,
            'c': default_params.c,
            'tauN': default_params.tauN,
            'tauY': default_params.tauY,
            'nTrials': default_params.nTrials,
            'sigma1': default_params.sigma1,
            'sigma2': default_params.sigma2,
            'sigma_meta': default_params.sigma_meta,  # Include metacognitive noise
            'nRatings': default_params.nRatings,
        }
        
        # Update the parameter being swept
        params_dict[param_name] = value
        params = SDTParameters(**params_dict)
        
        # Simulate trials
        trial_seed = seed + i if seed is not None else None
        trials = simulate_sdt_trials(params, seed=trial_seed)
        
        # Compute all metrics
        try:
            metrics = compute_all_metrics(trials, nRatings=params.nRatings)
            metrics[param_name] = value
            metrics['param_value'] = value
            metrics['param_name'] = param_name
            
            # Add ground truth parameters
            metrics['true_dprime'] = params.dprime
            metrics['true_c'] = params.c
            metrics['true_tauN'] = params.tauN
            metrics['true_tauY'] = params.tauY
            
            results.append(metrics)
        except Exception as e:
            print(f"Error at {param_name}={value}: {e}")
            continue
    
    return pd.DataFrame(results)


def run_simulation_grid(dprime_values: np.ndarray,
                       c_values: np.ndarray,
                       m_ratio_values: np.ndarray,
                       trial_counts: List[int],
                       seed: Optional[int] = None) -> pd.DataFrame:
    from .mdistance import compute_all_metrics
    from .metad_fitting import fit_metad_mle
    from .data_io import trials_to_counts_simple
    
    results = []
    total_sims = (len(dprime_values) * len(c_values) * 
                  len(m_ratio_values) * len(trial_counts))
    
    pbar = tqdm(total=total_sims, desc="Grid simulation")
    
    sim_id = 0
    for dprime in dprime_values:
        for c in c_values:
            for m_ratio in m_ratio_values:
                for nTrials in trial_counts:
                    # M-ratio affects tau positions
                    # Assume symmetric taus: distance = m_ratio * dprime / 4
                    tau_dist = m_ratio * dprime / 4
                    
                    params = SDTParameters(
                        dprime=dprime,
                        c=c,
                        tauN=-tau_dist,
                        tauY=tau_dist,
                        nTrials=nTrials,
                    )
                    
                    # Simulate
                    trial_seed = seed + sim_id if seed is not None else None
                    trials = simulate_sdt_trials(params, seed=trial_seed)
                    
                    # Compute metrics
                    try:
                        metrics = compute_all_metrics(trials)
                        
                        # Add grid parameters
                        metrics['true_dprime'] = dprime
                        metrics['true_c'] = c
                        metrics['true_m_ratio'] = m_ratio
                        metrics['nTrials'] = nTrials
                        metrics['sim_id'] = sim_id
                        
                        results.append(metrics)
                    except Exception as e:
                        print(f"Error in sim {sim_id}: {e}")
                    
                    sim_id += 1
                    pbar.update(1)
    
    pbar.close()
    return pd.DataFrame(results)


def run_all_simulations(output_dir: str = "outputs/simulations",
                       seed: int = 42) -> Dict[str, pd.DataFrame]:
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Default parameters (matching MATLAB)
    defaults = SDTParameters(
        dprime=2.0,
        c=0.0,
        tauN=-0.5,  # -dprime/4
        tauY=0.5,   # dprime/4
        nTrials=100000,
        sigma1=1.0,
        sigma2=1.0,
    )
    
    results = {}
    
    # Simulation 1: Vary type 1 criterion
    print("Simulation 1: Varying criterion...")
    c_values = np.arange(-1.5, 1.6, 0.1)
    results['vary_criterion'] = sweep_parameter(
        'c', c_values, defaults, seed=seed
    )
    results['vary_criterion'].to_csv(
        f"{output_dir}/vary_criterion.csv", index=False
    )
    
    # Simulation 2a: Vary tauN
    print("Simulation 2a: Varying tauN...")
    tau_values = np.arange(-1.0, 1.1, 0.1)
    results['vary_tauN'] = sweep_parameter(
        'tauN', tau_values, defaults, seed=seed
    )
    results['vary_tauN'].to_csv(
        f"{output_dir}/vary_tauN.csv", index=False
    )
    
    # Simulation 2b: Vary tauY
    print("Simulation 2b: Varying tauY...")
    results['vary_tauY'] = sweep_parameter(
        'tauY', tau_values, defaults, seed=seed
    )
    results['vary_tauY'].to_csv(
        f"{output_dir}/vary_tauY.csv", index=False
    )
    
    # Simulation 3: Vary type 1 sensitivity (d')
    print("Simulation 3: Varying d'...")
    dprime_values = np.arange(0.5, 3.1, 0.1)
    results['vary_dprime'] = sweep_parameter(
        'dprime', dprime_values, defaults, seed=seed
    )
    results['vary_dprime'].to_csv(
        f"{output_dir}/vary_dprime.csv", index=False
    )
    
    # Simulation 4: Vary trial count
    print("Simulation 4: Varying trial count...")
    trial_counts = np.arange(100, 1100, 100)
    trial_results = []
    
    for nTrials in tqdm(trial_counts):
        params = SDTParameters(
            dprime=defaults.dprime,
            c=defaults.c,
            tauN=defaults.tauN,
            tauY=defaults.tauY,
            nTrials=int(nTrials),
        )
        
        # Run multiple subjects per trial count
        nSubjects = 15
        for subj in range(nSubjects):
            trials = simulate_sdt_trials(params, seed=seed + int(nTrials)*100 + subj)
            metrics = compute_all_metrics(trials)
            metrics['nTrials'] = nTrials
            metrics['subject'] = subj
            trial_results.append(metrics)
    
    results['vary_trialcount'] = pd.DataFrame(trial_results)
    results['vary_trialcount'].to_csv(
        f"{output_dir}/vary_trialcount.csv", index=False
    )
    
    print(f"All simulations complete. Results saved to {output_dir}/")
    return results


def compute_all_metrics(trials):
    """Import here to avoid circular dependency."""
    from .mdistance import compute_all_metrics as _compute
    return _compute(trials)
