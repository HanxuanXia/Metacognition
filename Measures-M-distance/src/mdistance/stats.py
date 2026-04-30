from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
from scipy import stats
import warnings


def pearson_correlation(x: np.ndarray, 
                       y: np.ndarray) -> Dict[str, float]:
    # Remove NaN/Inf
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 3:
        return {'r': np.nan, 'p': np.nan, 'n': len(x_clean)}
    
    r, p = stats.pearsonr(x_clean, y_clean) 
    return {'r': r, 'p': p, 'n': len(x_clean)}


def spearman_correlation(x: np.ndarray,
                        y: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 3:
        return {'rho': np.nan, 'p': np.nan, 'n': len(x_clean)}
    
    rho, p = stats.spearmanr(x_clean, y_clean)
    
    return {'rho': rho, 'p': p, 'n': len(x_clean)}


def bland_altman_analysis(x: np.ndarray,
                         y: np.ndarray,
                         confidence: float = 0.95) -> Dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    diff = x_clean - y_clean
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    
    # Limits of agreement (mean ± 1.96*SD)
    z = stats.norm.ppf((1 + confidence) / 2)
    lower_limit = mean_diff - z * std_diff
    upper_limit = mean_diff + z * std_diff
    
    # Confidence interval for bias
    se_mean = std_diff / np.sqrt(len(diff))
    ci_lower = mean_diff - z * se_mean
    ci_upper = mean_diff + z * se_mean
    
    return {
        'bias': mean_diff,
        'std': std_diff,
        'lower_limit': lower_limit,
        'upper_limit': upper_limit,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n': len(diff),
    }


def compute_icc(x: np.ndarray,
               y: np.ndarray,
               icc_type: str = 'ICC(2,1)') -> Dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    n = len(x_clean)
    k = 2  # Two raters
    
    # Organize as ratings matrix
    ratings = np.column_stack([x_clean, y_clean])
    
    # Compute ICC(2,1) - two-way random effects, single rater
    # Using variance components from ANOVA
    
    # Grand mean
    grand_mean = np.mean(ratings)
    
    # Between-subjects variance
    subject_means = np.mean(ratings, axis=1)
    SS_between = k * np.sum((subject_means - grand_mean) ** 2)
    MS_between = SS_between / (n - 1)
    
    # Within-subjects variance
    SS_within = np.sum((ratings - subject_means[:, np.newaxis]) ** 2)
    MS_within = SS_within / (n * (k - 1))
    
    # Residual variance
    rater_means = np.mean(ratings, axis=0)
    SS_rater = n * np.sum((rater_means - grand_mean) ** 2)
    MS_rater = SS_rater / (k - 1)
    
    SS_error = SS_within - SS_rater
    MS_error = SS_error / ((n - 1) * (k - 1))
    
    # ICC(2,1) formula
    ICC = (MS_between - MS_error) / (MS_between + (k - 1) * MS_error + k * (MS_rater - MS_error) / n)
    
    # Confidence interval (approximate)
    F_lower = MS_between / MS_error / stats.f.ppf(0.975, n - 1, n * (k - 1))
    F_upper = MS_between / MS_error / stats.f.ppf(0.025, n - 1, n * (k - 1))
    
    ci_lower = (F_lower - 1) / (F_lower + k - 1)
    ci_upper = (F_upper - 1) / (F_upper + k - 1)
    
    return {
        'ICC': ICC,
        'ci_lower': max(0, ci_lower),  # ICC bounded at 0
        'ci_upper': min(1, ci_upper),  # ICC bounded at 1
        'n': n,
    }


def partial_correlation(x: np.ndarray,
                       y: np.ndarray,
                       covariates: np.ndarray,
                       method: str = 'pearson') -> Dict[str, float]:
    from scipy.linalg import lstsq
    
    # Ensure covariates is 2D
    if covariates.ndim == 1:
        covariates = covariates[:, np.newaxis]
    
    # Remove NaN
    mask = (np.isfinite(x) & np.isfinite(y) & 
            np.all(np.isfinite(covariates), axis=1))
    x_clean = x[mask]
    y_clean = y[mask]
    cov_clean = covariates[mask]
    
    n = len(x_clean)
    k = cov_clean.shape[1]
    
    if n < k + 3:
        return {'r_partial': np.nan, 'p': np.nan, 'n': n}
    
    # Add intercept
    X_cov = np.column_stack([np.ones(n), cov_clean])
    
    # Residualize x and y
    resid_x = x_clean - X_cov @ lstsq(X_cov, x_clean)[0]
    resid_y = y_clean - X_cov @ lstsq(X_cov, y_clean)[0]
    
    # Correlate residuals
    if method == 'pearson':
        r_partial, p = stats.pearsonr(resid_x, resid_y)
    elif method == 'spearman':
        r_partial, p = stats.spearmanr(resid_x, resid_y)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {'r_partial': r_partial, 'p': p, 'n': n}


def bootstrap_test(x: np.ndarray,
                  y: np.ndarray,
                  n_bootstrap: int = 10000,
                  statistic: str = 'mean_diff') -> Dict[str, float]:
    # Clean data
    x_clean = x[np.isfinite(x)]
    y_clean = y[np.isfinite(y)]
    
    # Observed statistic
    if statistic == 'mean_diff':
        obs_stat = np.mean(x_clean) - np.mean(y_clean)
    elif statistic == 'median_diff':
        obs_stat = np.median(x_clean) - np.median(y_clean)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")
    
    # Bootstrap under null (pooled data)
    pooled = np.concatenate([x_clean, y_clean])
    n_x = len(x_clean)
    n_y = len(y_clean)
    
    boot_stats = []
    for _ in range(n_bootstrap):
        # Resample from pooled
        boot_sample = np.random.choice(pooled, size=len(pooled), replace=True)
        boot_x = boot_sample[:n_x]
        boot_y = boot_sample[n_x:]
        
        if statistic == 'mean_diff':
            boot_stat = np.mean(boot_x) - np.mean(boot_y)
        else:
            boot_stat = np.median(boot_x) - np.median(boot_y)
        
        boot_stats.append(boot_stat)
    
    boot_stats = np.array(boot_stats)
    
    # Two-tailed p-value
    p_value = np.mean(np.abs(boot_stats) >= np.abs(obs_stat))
    
    return {
        'statistic': obs_stat,
        'p_value': p_value,
        'ci_lower': np.percentile(boot_stats, 2.5),
        'ci_upper': np.percentile(boot_stats, 97.5),
    }


def correlation_matrix(data: pd.DataFrame,
                      variables: List[str],
                      method: str = 'pearson') -> pd.DataFrame:
    subset = data[variables].dropna()
    
    if method == 'pearson':
        corr_mat = subset.corr(method='pearson')
    elif method == 'spearman':
        corr_mat = subset.corr(method='spearman')
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return corr_mat


def compare_metrics_stats(data: pd.DataFrame,
                         metric1: str,
                         metric2: str) -> Dict:
    x = data[metric1].values
    y = data[metric2].values
    
    results = {
        'pearson': pearson_correlation(x, y),
        'spearman': spearman_correlation(x, y),
        'bland_altman': bland_altman_analysis(x, y),
        'icc': compute_icc(x, y),
    }
    
    # Add controlling for d' if available
    if 'd_prime' in data.columns:
        results['partial_pearson_control_d'] = partial_correlation(
            x, y, data['d_prime'].values, method='pearson'
        )
    
    # Add controlling for c if available
    if 'c' in data.columns:
        results['partial_pearson_control_c'] = partial_correlation(
            x, y, data['c'].values, method='pearson'
        )
    
    # Both d' and c
    if 'd_prime' in data.columns and 'c' in data.columns:
        covariates = np.column_stack([data['d_prime'].values, data['c'].values])
        results['partial_pearson_control_d_c'] = partial_correlation(
            x, y, covariates, method='pearson'
        )
    
    return results
