from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np
from scipy import stats


@dataclass
class SDTParameters:
    dprime: float = 2.0
    c: float = 0.0
    tauN: float = -0.5
    tauY: float = 0.5
    nTrials: int = 1000
    sigma1: float = 1.0
    sigma2: float = 1.0
    sigma_meta: float = 0.0
    padCells: bool = True
    nRatings: int = 2
    
    def __post_init__(self):
        if self.dprime < 0:
            raise ValueError("dprime must be non-negative")
        if self.nTrials <= 0:
            raise ValueError("nTrials must be positive")
        if self.sigma1 <= 0 or self.sigma2 <= 0:
            raise ValueError("sigma values must be positive")
        if self.nRatings < 2:
            raise ValueError("nRatings must be at least 2")
    
    @property
    def s_ratio(self) -> float:
        """Ratio of standard deviations: sigma1/sigma2."""
        return self.sigma1 / self.sigma2
    
    @property
    def tauN_absolute(self) -> float:
        return self.c + self.tauN
    
    @property
    def tauY_absolute(self) -> float:
        return self.c + self.tauY


def compute_type1_sdt(trials: Dict[str, np.ndarray], 
                      padCells: bool = True) -> Dict[str, float]:
    stimulus = trials['stimulus']
    response = trials['response']
    
    # Count outcomes
    s1_trials = stimulus == 0
    s2_trials = stimulus == 1
    r1_responses = response == 0
    r2_responses = response == 1
    
    nCRs = np.sum(s1_trials & r1_responses)    
    nFAs = np.sum(s1_trials & r2_responses)    
    nMisses = np.sum(s2_trials & r1_responses) 
    nHits = np.sum(s2_trials & r2_responses)   
    
    # Apply padding if requested
    padding = 0.5 if padCells else 0.0
    
    # Compute rates
    HR = (nHits + padding) / (nHits + nMisses + 2 * padding)
    FAR = (nFAs + padding) / (nFAs + nCRs + 2 * padding)
    
    # Clip to avoid infinite values
    HR = np.clip(HR, 1e-10, 1 - 1e-10)
    FAR = np.clip(FAR, 1e-10, 1 - 1e-10)
    
    # Compute d' and c
    d_prime = dprime_from_rates(HR, FAR)
    c = criterion_from_rates(HR, FAR)
    
    return {
        'd_prime': d_prime,
        'c': c,
        'HR': HR,
        'FAR': FAR,
        'nHits': int(nHits),
        'nMisses': int(nMisses),
        'nFAs': int(nFAs),
        'nCRs': int(nCRs),
    }


def compute_type2_sdt(trials: Dict[str, np.ndarray],
                      padCells: bool = True) -> Dict[str, float]:
    correct = trials['correct']
    confidence = trials['confidence']
    
    # Count outcomes (assuming binary confidence: 0=low, 1=high)
    n_correct_high_conf = np.sum(correct & (confidence == 1))
    n_correct_low_conf = np.sum(correct & (confidence == 0))
    n_incorrect_high_conf = np.sum(~correct & (confidence == 1))
    n_incorrect_low_conf = np.sum(~correct & (confidence == 0))
    
    padding = 0.5 if padCells else 0.0
    
    # Type 2 HR: P(high confidence | correct)
    type2_HR = (n_correct_high_conf + padding) / \
               (n_correct_high_conf + n_correct_low_conf + 2 * padding)
    
    # Type 2 FAR: P(high confidence | incorrect)
    type2_FAR = (n_incorrect_high_conf + padding) / \
                (n_incorrect_high_conf + n_incorrect_low_conf + 2 * padding)
    
    type2_HR = np.clip(type2_HR, 1e-10, 1 - 1e-10)
    type2_FAR = np.clip(type2_FAR, 1e-10, 1 - 1e-10)
    
    # Type 2 d'
    type2_dprime = stats.norm.ppf(type2_HR) - stats.norm.ppf(type2_FAR)
    
    return {
        'type2_HR': type2_HR,
        'type2_FAR': type2_FAR,
        'type2_dprime': type2_dprime,
    }


def dprime_from_rates(HR: float, FAR: float) -> float:
    HR = np.clip(HR, 1e-10, 1 - 1e-10)
    FAR = np.clip(FAR, 1e-10, 1 - 1e-10)
    return stats.norm.ppf(HR) - stats.norm.ppf(FAR)


def criterion_from_rates(HR: float, FAR: float) -> float:
    HR = np.clip(HR, 1e-10, 1 - 1e-10)
    FAR = np.clip(FAR, 1e-10, 1 - 1e-10)
    return -0.5 * (stats.norm.ppf(HR) + stats.norm.ppf(FAR))


def rates_from_dprime_c(d_prime: float, c: float) -> Tuple[float, float]:
    HR = stats.norm.cdf(d_prime / 2 - c)
    FAR = stats.norm.cdf(-d_prime / 2 - c)
    return HR, FAR
