from typing import Dict, Tuple, Optional, Callable
import numpy as np
from scipy import stats, optimize
from scipy.optimize import Bounds, minimize
import warnings


HAS_NUMBA = False  # Disabled for compatibility with scipy.optimize

warnings.filterwarnings('ignore', message='delta_grad == 0.0')


def fit_meta_d_logL(
    guess: np.ndarray,
    nR_S1: np.ndarray,
    nR_S2: np.ndarray,
    nRatings: int,
    d1: float,
    t1c1: float,
    s: float = 1.0
) -> float:

    meta_d1 = guess[0]
    t2c1 = guess[1:]
    
    # Define mean and SD of S1 and S2 distributions
    S1mu = -meta_d1 / 2
    S1sd = 1.0
    S2mu = meta_d1 / 2
    S2sd = S1sd / s
    
    # Adjust so type 1 criterion is at 0
    # (simplifies optimization bounds)
    S1mu = S1mu - (meta_d1 * (t1c1 / d1)) if abs(d1) > 1e-10 else S1mu
    S2mu = S2mu - (meta_d1 * (t1c1 / d1)) if abs(d1) > 1e-10 else S2mu
    t1c1 = 0.0
    
    # Get type 2 response counts
    # S1 responses: correct=R1, incorrect=R2
    nC_rS1 = nR_S1[:nRatings]  
    nI_rS1 = nR_S2[:nRatings]  
    
    # S2 responses: correct=R2, incorrect=R1
    nC_rS2 = nR_S2[nRatings:]  
    nI_rS2 = nR_S1[nRatings:]  
    
    # Use scipy.stats.norm.cdf for normal CDF (stable and fast enough)
    norm_cdf = stats.norm.cdf
    
    # Get type 2 probabilities
    C_area_rS1 = norm_cdf(t1c1, S1mu, S1sd)
    I_area_rS1 = norm_cdf(t1c1, S2mu, S2sd)
    C_area_rS2 = 1.0 - norm_cdf(t1c1, S2mu, S2sd)
    I_area_rS2 = 1.0 - norm_cdf(t1c1, S1mu, S1sd)
    
    # Build type 2 criteria array
    t2c1x = [-np.inf]
    t2c1x.extend(t2c1[:(nRatings - 1)])
    t2c1x.append(t1c1)
    t2c1x.extend(t2c1[(nRatings - 1):])
    t2c1x.append(np.inf)
    
    # Calculate probabilities for each rating bin
    prC_rS1 = []
    prI_rS1 = []
    for i in range(nRatings):
        prC = (norm_cdf(t2c1x[i + 1], S1mu, S1sd) - 
               norm_cdf(t2c1x[i], S1mu, S1sd)) / C_area_rS1
        prI = (norm_cdf(t2c1x[i + 1], S2mu, S2sd) - 
               norm_cdf(t2c1x[i], S2mu, S2sd)) / I_area_rS1
        prC_rS1.append(prC)
        prI_rS1.append(prI)
    
    prC_rS2 = []
    prI_rS2 = []
    for i in range(nRatings):
        idx = nRatings + i
        prC = ((1.0 - norm_cdf(t2c1x[idx], S2mu, S2sd)) - 
               (1.0 - norm_cdf(t2c1x[idx + 1], S2mu, S2sd))) / C_area_rS2
        prI = ((1.0 - norm_cdf(t2c1x[idx], S1mu, S1sd)) - 
               (1.0 - norm_cdf(t2c1x[idx + 1], S1mu, S1sd))) / I_area_rS2
        prC_rS2.append(prC)
        prI_rS2.append(prI)
    
    # Calculate log-likelihood
    logL = 0.0
    for i in range(nRatings):
        logL += nC_rS1[i] * np.log(np.clip(prC_rS1[i], 1e-10, 1))
        logL += nI_rS1[i] * np.log(np.clip(prI_rS1[i], 1e-10, 1))
        logL += nC_rS2[i] * np.log(np.clip(prC_rS2[i], 1e-10, 1))
        logL += nI_rS2[i] * np.log(np.clip(prI_rS2[i], 1e-10, 1))
    
    # Return negative log-likelihood for minimization
    if np.isnan(logL) or np.isinf(logL):
        return 1e300
    
    return -logL


def fit_metad_mle(
    nR_S1: np.ndarray,
    nR_S2: np.ndarray,
    s: float = 1.0,
    nRatings: Optional[int] = None,
    padding: bool = True,
    padAmount: Optional[float] = None,
    verbose: int = 0
) -> Dict:

    # Convert to numpy arrays
    nR_S1 = np.array(nR_S1, dtype=float)
    nR_S2 = np.array(nR_S2, dtype=float)
    
    # Validate input
    if len(nR_S1) != len(nR_S2):
        raise ValueError("nR_S1 and nR_S2 must have same length")
    if len(nR_S1) % 2 != 0:
        raise ValueError("Input arrays must have even length")
    
    # Determine nRatings
    if nRatings is None:
        nRatings = len(nR_S1) // 2
    
    # Apply padding to avoid zeros
    if padding:
        if padAmount is None:
            padAmount = 1.0 / (2.0 * nRatings)
        nR_S1 = nR_S1 + padAmount
        nR_S2 = nR_S2 + padAmount
    
    # Compute type 1 SDT parameters
    nS1 = np.sum(nR_S1)
    nS2 = np.sum(nR_S2)
    
    # Hits and False Alarms
    H = np.sum(nR_S2[nRatings:])
    M = np.sum(nR_S2[:nRatings:])
    FA = np.sum(nR_S1[nRatings:])
    CR = np.sum(nR_S1[:nRatings])  
    
    # Rates
    HR = H / (H + M)
    FAR = FA / (FA + CR)
    
    # Clip to avoid infinities
    HR = np.clip(HR, 1e-10, 1 - 1e-10)
    FAR = np.clip(FAR, 1e-10, 1 - 1e-10)
    
    # Type 1 parameters
    d1 = stats.norm.ppf(HR) - stats.norm.ppf(FAR)
    c1 = -0.5 * (stats.norm.ppf(HR) + stats.norm.ppf(FAR))
    
    # We space them evenly between -(d1/2) and 0, and between 0 and +(d1/2).
    n_t2c    = 2 * (nRatings - 1)       
    n_t2c_R1 = nRatings - 1             
    n_t2c_R2 = nRatings - 1              

    half = max(d1 / 2.0, 0.5)          

    # R1 criteria: n_t2c_R1 evenly spaced in (-half, 0), strictly negative
    if n_t2c_R1 > 0:
        guess_t2c_R1 = list(np.linspace(-half, -0.2, n_t2c_R1))
    else:
        guess_t2c_R1 = []

    # R2 criteria: n_t2c_R2 evenly spaced in (0, +half), strictly positive
    if n_t2c_R2 > 0:
        guess_t2c_R2 = list(np.linspace(0.2, half, n_t2c_R2))
    else:
        guess_t2c_R2 = []

    guess = np.array([d1] + guess_t2c_R1 + guess_t2c_R2)

    bounds_lower = [-10.0] + [-10.0] * n_t2c_R1 + [1e-6] * n_t2c_R2
    bounds_upper = [10.0]  + [-1e-6] * n_t2c_R1 + [10.0] * n_t2c_R2
    bounds = Bounds(bounds_lower, bounds_upper)

    # We only need the within-side ordering constraints.
    A  = []
    lb = []
    ub = []

    # R1 ordering: t2c[i] - t2c[i+1] <= -gap
    for i in range(n_t2c_R1 - 1):
        row = np.zeros(1 + n_t2c)
        row[1 + i]     =  1.0   # t2c_R1[i]
        row[1 + i + 1] = -1.0   # -t2c_R1[i+1]
        A.append(row)
        lb.append(-np.inf)
        ub.append(-0.01)

    # R2 ordering: t2c[nR1 + i] - t2c[nR1 + i+1] <= -gap
    for i in range(n_t2c_R2 - 1):
        row = np.zeros(1 + n_t2c)
        row[1 + n_t2c_R1 + i]     =  1.0
        row[1 + n_t2c_R1 + i + 1] = -1.0
        A.append(row)
        lb.append(-np.inf)
        ub.append(-0.01)

    constraints = optimize.LinearConstraint(np.array(A), lb, ub) if A else None
    
    # Optimize
    try:
        result = minimize(
            fit_meta_d_logL,
            guess,
            args=(nR_S1, nR_S2, nRatings, d1, c1, s),
            method='trust-constr',
            bounds=bounds,
            constraints=constraints,
            options={'verbose': verbose, 'maxiter': 1000}
        )
        
        meta_d = result.x[0]
        t2c = result.x[1:]
        logL = -result.fun
        success = result.success
        
    except Exception as e:
        if verbose > 0:
            print(f"Optimization failed: {e}")
            print("Falling back to L-BFGS-B without constraints...")
        
        # Fallback to simpler optimizer
        result = minimize(
            fit_meta_d_logL,
            guess,
            args=(nR_S1, nR_S2, nRatings, d1, c1, s),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        meta_d = result.x[0]
        t2c = result.x[1:]
        logL = -result.fun
        success = result.success
    
    # Extract type 2 criteria for R1 and R2
    # n_t2c_R1 = nRatings - 1 criteria on each side (matches vector layout above)
    t2c_R1 = t2c[:n_t2c_R1]
    t2c_R2 = t2c[n_t2c_R1:]
    
    # Use the criterion closest to the decision boundary on each side.
    if len(t2c_R1) > 0:
        meta_c2_R1 = np.max(t2c_R1)
    else:
        meta_c2_R1 = -0.5
    
    if len(t2c_R2) > 0:
        meta_c2_R2 = np.min(t2c_R2)
    else:
        meta_c2_R2 = 0.5
    
    # meta_c is 0 because the likelihood was fit in criterion-centred space
    meta_c = 0.0
    
    return {
        'meta_d': meta_d,
        'meta_c': meta_c,
        't2c_R1': t2c_R1,
        't2c_R2': t2c_R2,
        'meta_c2_R1': meta_c2_R1,
        'meta_c2_R2': meta_c2_R2,
        'd_prime': d1,
        'c': c1,
        'logL': logL,
        'success': success,
        'HR': HR,
        'FAR': FAR,
    }


def fit_metad_simple_binary(trials: Dict[str, np.ndarray],
                           padding: bool = True,
                           verbose: int = 0) -> Dict:
    from .data_io import trials_to_counts_simple
    
    # Convert to counts
    nR_S1, nR_S2 = trials_to_counts_simple(trials, nRatings=2)
    
    # Fit
    return fit_metad_mle(nR_S1, nR_S2, nRatings=2, padding=padding, verbose=verbose)
