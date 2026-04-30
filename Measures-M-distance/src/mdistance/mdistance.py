from typing import Dict, Optional
import numpy as np
import pandas as pd


def compute_mdist(fit_result: Dict) -> Dict[str, float]:
    meta_d = fit_result['meta_d']
    meta_c = fit_result['meta_c']
    meta_c2_R1 = fit_result['meta_c2_R1']
    meta_c2_R2 = fit_result['meta_c2_R2']
    
    if (abs(meta_d) < 1e-10
            or np.isnan(meta_c) or np.isnan(meta_c2_R1) or np.isnan(meta_c2_R2)):
        return {
            'm_dist_R1': np.nan,
            'm_dist_R2': np.nan,
            'm_dist_avg': np.nan,
            'm_dist_abs_avg': np.nan,
        }
    
    # Compute normalized m-distance
    m_dist_R1 = (meta_c - meta_c2_R1) / abs(meta_d)
    m_dist_R2 = (meta_c2_R2 - meta_c) / abs(meta_d)
    
    # Average (signed and absolute)
    m_dist_avg = (m_dist_R1 + m_dist_R2) / 2
    m_dist_abs_avg = (abs(m_dist_R1) + abs(m_dist_R2)) / 2
    
    return {
        'm_dist_R1': m_dist_R1,
        'm_dist_R2': m_dist_R2,
        'm_dist_avg': m_dist_avg,
        'm_dist_abs_avg': m_dist_abs_avg,
    }


def compute_mdist2(fit_result: Dict) -> Dict[str, float]:
    meta_c = fit_result['meta_c']
    meta_c2_R1 = fit_result['meta_c2_R1']
    meta_c2_R2 = fit_result['meta_c2_R2']
    
    m_dist2_R1 = meta_c - meta_c2_R1
    m_dist2_R2 = meta_c2_R2 - meta_c
    
    m_dist2_avg = (m_dist2_R1 + m_dist2_R2) / 2
    m_dist2_abs_avg = (abs(m_dist2_R1) + abs(m_dist2_R2)) / 2
    
    return {
        'm_dist2_R1': m_dist2_R1,
        'm_dist2_R2': m_dist2_R2,
        'm_dist2_avg': m_dist2_avg,
        'm_dist2_abs_avg': m_dist2_abs_avg,
    }


def compute_mean_confidence(trials: Dict[str, np.ndarray],
                           by_response: bool = False,
                           by_correctness: bool = False) -> Dict[str, float]:
    if 'rating' not in trials:
        raise ValueError("compute_mean_confidence requires trials['rating'] "
                         "(1=R1-high, 2=R1-low, 3=R2-low, 4=R2-high)")

    rating   = trials['rating'].astype(float)
    response = trials['response']

    result = {'mean_conf_all': np.mean(rating)}

    if by_response:
        # Per-side confidence on a 0-1 scale (0=low, 1=high)
        mask_R1 = response == 0
        mask_R2 = response == 1
        # R1: rating 1→conf 1, rating 2→conf 0
        conf_R1 = 2.0 - rating[mask_R1]          
        # R2: rating 3→conf 0, rating 4→conf 1
        conf_R2 = rating[mask_R2] - 3.0         
        result['mean_conf_R1'] = float(np.mean(conf_R1)) if mask_R1.any() else np.nan
        result['mean_conf_R2'] = float(np.mean(conf_R2)) if mask_R2.any() else np.nan

    if by_correctness:
        correct = trials['correct']
        result['mean_conf_correct']   = np.mean(rating[correct == 1])
        result['mean_conf_incorrect'] = np.mean(rating[correct == 0])

    return result


def compute_all_metrics(trials: Dict[str, np.ndarray],
                       fit_result: Optional[Dict] = None,
                       nRatings: int = 2) -> Dict[str, float]:
    from .sdt import compute_type1_sdt
    from .metad_fitting import fit_metad_simple_binary
    from .data_io import trials_to_counts_simple
    
    # Compute type 1 SDT
    sdt = compute_type1_sdt(trials)
    
    # Fit meta-d' if not provided
    if fit_result is None:
        if nRatings == 2:
            fit_result = fit_metad_simple_binary(trials)
        else:
            from .metad_fitting import fit_metad_mle
            nR_S1, nR_S2 = trials_to_counts_simple(trials, nRatings)
            fit_result = fit_metad_mle(nR_S1, nR_S2, nRatings=nRatings)
    
    # Compute m-distance
    mdist = compute_mdist(fit_result)
    mdist2 = compute_mdist2(fit_result)
    mean_conf = compute_mean_confidence(trials, by_response=True)
    
    # Combine all metrics
    result = {
        # Type 1 SDT
        'd_prime': sdt['d_prime'],
        'c': sdt['c'],
        'HR': sdt['HR'],
        'FAR': sdt['FAR'],
        
        # Meta-d'
        'meta_d': fit_result['meta_d'],
        'meta_c': fit_result['meta_c'],
        'meta_c2_R1': fit_result['meta_c2_R1'],
        'meta_c2_R2': fit_result['meta_c2_R2'],
        'M_ratio': fit_result['meta_d'] / sdt['d_prime'] if sdt['d_prime'] != 0 else np.nan,
        
        # M-distance (normalized)
        'm_dist_R1': mdist['m_dist_R1'],
        'm_dist_R2': mdist['m_dist_R2'],
        'm_dist_avg': mdist['m_dist_avg'],
        
        # M-dist2 (unnormalized)
        'm_dist2_R1': mdist2['m_dist2_R1'],
        'm_dist2_R2': mdist2['m_dist2_R2'],
        'm_dist2_avg': mdist2['m_dist2_avg'],
        
        # Mean confidence (overall and response-specific)
        'mean_conf': mean_conf['mean_conf_all'],
        'mean_conf_R1': mean_conf['mean_conf_R1'],
        'mean_conf_R2': mean_conf['mean_conf_R2'],
        
        # Fit quality
        'logL': fit_result['logL'],
        'fit_success': fit_result['success'],
    }
    
    return result


def compare_metrics(trials_list: list,
                   labels: Optional[list] = None) -> pd.DataFrame:
    if labels is None:
        labels = [f"Dataset_{i+1}" for i in range(len(trials_list))]
    
    results = []
    for trials, label in zip(trials_list, labels):
        metrics = compute_all_metrics(trials)
        metrics['dataset'] = label
        results.append(metrics)
    
    return pd.DataFrame(results)
