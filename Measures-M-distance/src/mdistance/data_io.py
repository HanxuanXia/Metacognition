from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd


def trials_to_counts(trials: Dict[str, np.ndarray],
                     nRatings: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    stimulus = trials['stimulus']
    response = trials['response']
    
    # Handle rating format
    if 'rating' in trials:
        # Signed ratings: -nRatings to -1, 1 to nRatings
        # or unsigned: 1 to 2*nRatings
        rating = trials['rating']
    elif 'confidence' in trials and nRatings == 2:
        # Binary confidence: convert to ratings
        # conf=1 → rating=2, conf=0 → rating=1
        rating = trials['confidence'] + 1
    else:
        raise ValueError("trials must contain 'rating' or 'confidence' (for binary)")
    
    # Initialize count arrays
    nR_S1 = np.zeros(2 * nRatings, dtype=int)
    nR_S2 = np.zeros(2 * nRatings, dtype=int)
    
    # Process each trial
    for i in range(len(stimulus)):
        stim = stimulus[i]
        resp = response[i]
        rat = rating[i]
        
        # Convert signed ratings to index (0 to 2*nRatings-1)
        if rat < 0:
            # Negative rating: R1 response with confidence
            idx = nRatings + rat  # e.g., -3 → 0, -2 → 1, -1 → 2
        elif rat > 0 and rat <= nRatings:
            # Positive rating: could be R1 (low conf) or R2
            if resp == 0:  # R1 response
                idx = nRatings - rat  # e.g., 1 → 2, 2 → 1, 3 → 0
            else:  # R2 response
                idx = nRatings + rat - 1  # e.g., 1 → 3, 2 → 4, 3 → 5
        else:
            raise ValueError(f"Invalid rating: {rat}")
        
        # Increment count
        if stim == 0:  # S1 trial
            nR_S1[idx] += 1
        else:  # S2 trial
            nR_S2[idx] += 1
    
    return nR_S1, nR_S2


def trials_to_counts_simple(trials: Dict[str, np.ndarray],
                            nRatings: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    stimulus = trials['stimulus']
    response = trials['response']
    
    # Construct combined rating: response determines direction
    # R1 responses: ratings nRatings down to 1
    # R2 responses: ratings 1 up to nRatings
    if 'rating' in trials:
        rating = trials['rating']
    else:
        # Binary confidence
        confidence = trials['confidence']
        rating = np.zeros(len(stimulus), dtype=int)
        for i in range(len(stimulus)):
            if response[i] == 0:  # R1
                rating[i] = nRatings - confidence[i]  
            else:  # R2
                rating[i] = nRatings + confidence[i] + 1  
    
    # Initialize counts
    nR_S1 = np.zeros(2 * nRatings, dtype=int)
    nR_S2 = np.zeros(2 * nRatings, dtype=int)
    
    # Count trials
    for i in range(len(stimulus)):
        idx = int(rating[i]) - 1  # Convert to 0-indexed
        if stimulus[i] == 0:
            nR_S1[idx] += 1
        else:
            nR_S2[idx] += 1
    
    return nR_S1, nR_S2


def counts_to_trials(nR_S1: np.ndarray, 
                     nR_S2: np.ndarray) -> Dict[str, np.ndarray]:
    nRatings = len(nR_S1) // 2
    
    # Calculate total trials
    total_trials = int(np.sum(nR_S1) + np.sum(nR_S2))
    
    # Pre-allocate arrays
    stimulus = np.zeros(total_trials, dtype=int)
    response = np.zeros(total_trials, dtype=int)
    rating = np.zeros(total_trials, dtype=int)
    
    trial_idx = 0
    
    # Process S1 trials
    for i in range(len(nR_S1)):
        count = int(nR_S1[i])
        for _ in range(count):
            stimulus[trial_idx] = 0  # S1
            if i < nRatings:
                response[trial_idx] = 0  # R1
                rating[trial_idx] = nRatings - i  
            else:
                response[trial_idx] = 1  # R2
                rating[trial_idx] = i - nRatings + 1
            trial_idx += 1
    
    # Process S2 trials
    for i in range(len(nR_S2)):
        count = int(nR_S2[i])
        for _ in range(count):
            stimulus[trial_idx] = 1  # S2
            if i < nRatings:
                response[trial_idx] = 0  # R1
                rating[trial_idx] = nRatings - i
            else:
                response[trial_idx] = 1  # R2
                rating[trial_idx] = i - nRatings + 1
            trial_idx += 1
    
    # Add correct field
    correct = (stimulus == response).astype(int)
    
    return {
        'stimulus': stimulus,
        'response': response,
        'rating': rating,
        'correct': correct,
    }


def add_padding(nR_S1: np.ndarray, 
                nR_S2: np.ndarray, 
                padding: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    return nR_S1 + padding, nR_S2 + padding


def load_empirical_data(filepath: str, 
                       format: str = 'csv') -> pd.DataFrame:
    if format == 'csv':
        return pd.read_csv(filepath)
    elif format == 'excel':
        return pd.read_excel(filepath)
    elif format == 'mat':
        # Load MATLAB .mat file
        from scipy.io import loadmat
        data = loadmat(filepath)
        # Convert to DataFrame (structure depends on .mat format)
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_results(data: pd.DataFrame, 
                filepath: str,
                format: str = 'csv'):
    if format == 'csv':
        data.to_csv(filepath, index=False)
    elif format == 'excel':
        data.to_excel(filepath, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")


def validate_trials(trials: Dict[str, np.ndarray]) -> bool:
    required_keys = ['stimulus', 'response']
    
    for key in required_keys:
        if key not in trials:
            raise ValueError(f"Missing required key: {key}")
    
    # Check array lengths match
    n_trials = len(trials['stimulus'])
    for key in trials:
        if len(trials[key]) != n_trials:
            raise ValueError(f"Length mismatch for key: {key}")
    
    # Check values in valid range
    if not np.all(np.isin(trials['stimulus'], [0, 1])):
        raise ValueError("stimulus must contain only 0 or 1")
    
    if not np.all(np.isin(trials['response'], [0, 1])):
        raise ValueError("response must contain only 0 or 1")
    
    return True
