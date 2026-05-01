# Measures-M-distance: Assessing Signal-Detection Theoretic Measures of Metacognitive Confidence

## Overview

This project compares three **threshold-related measures of metacognitive confidence**—**proportion confident**, **m-distance**, and **m-distance2**—under systematic manipulations of criterion placement, metacognitive noise, and type 1 sensitivity within the Signal Detection Theory (SDT) framework.

## Motivation

Confidence judgements are central to metacognition because they provide a behavioural report of how certain an observer is about a decision. However, confidence is not a simple readout of accuracy. Several factors influence confidence beyond objective performance:

- **Attention**: Can change confidence even when objective performance is held constant
- **Prior information**: Can bias subjective confidence reports
- **History and context**: Previous decisions and states influence current confidence

An important distinction exists between three related but distinct quantities:

1. **Confidence frequency**: How often an observer reports high confidence
2. **Metacognitive sensitivity**: How well confidence distinguishes correct from incorrect decisions
3. **Metacognitive threshold**: How much evidence is required before an observer reports high confidence

This project systematically compares threshold-related measures by examining how they behave under variations in type 1 criterion, response-specific confidence criteria, metacognitive noise, and type 1 sensitivity.

## Key Findings

The project reveals that:

1. **Proportion confident** is a simple descriptive measure but is affected by both threshold and type 1 sensitivity
2. **m-distance** measures response-specific confidence criterion distance relative to meta-d', providing criterion-invariant threshold estimates under ordinary conditions
3. **m-distance2** uses the same fitted criterion distance as m-distance but leaves distance in evidence units (unbounded), providing stable comparison for understanding the role of meta-d' normalization

## Project Structure

```
Measures-M-distance/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── scripts/                           # Main analysis scripts
│   ├── fig_criterion_sweep.py        # Criterion translation effects
│   ├── fig_noise_sweep.py            # Metacognitive noise sensitivity
│   ├── fig_metad_sweep.py            # Type 1 sensitivity dependence
│   └── sdt_shared.py                 # Shared utilities and constants
├── src/                               # Core simulation library
│   └── mdistance/
│       ├── __init__.py               # Package initialization
│       ├── sdt.py                    # SDT model implementation
│       ├── simulations.py            # Trial simulation functions
│       ├── mdistance.py              # Core m-distance calculations
│       ├── metad_fitting.py          # Meta-d' estimation
│       ├── stats.py                  # Statistical utilities
│       ├── plots.py                  # Plotting functions
│       └── data_io.py                # Data input/output
└── outputs/                           # Generated figures and data
    ├── criterion_sweep/              # Criterion manipulation results
    ├── noise_sweep/                  # Metacognitive noise effects
    └── metad_sweep/                  # Type 1 sensitivity effects
```

## Theoretical Background

### Signal Detection Theory Framework

The project uses a binary decision SDT model:

- **Type 1 decision**: Based on evidence $X$ ~ $\mathcal{N}(d'/2, 1)$ for signal trials and $X$ ~ $\mathcal{N}(-d'/2, 1)$ for noise trials
- **Type 1 criterion**: $c$ separates the two response regions
- **Confidence judgment**: Based on evidence $X_{\text{conf}} = X + \epsilon$, where $\epsilon$ ~ $\mathcal{N}(0, \sigma_{\text{meta}}^2)$
- **Meta-d'**: The level of type 1 sensitivity needed to reproduce the observed relationship between confidence and accuracy
- **Response-specific confidence criteria**: $\text{meta-}c_{2R_1}$ and $\text{meta-}c_{2R_2}$ determine how far evidence must fall into each response region before high confidence is reported

### Key Measures Compared

1. **Proportion confident**: $p(\text{confident}) = \frac{\text{number of confident responses}}{\text{total number of responses}}$
   - Simple descriptive measure, but not specific to threshold
   - Can change with type 1 sensitivity or response bias, not just threshold

2. **m-distance**: Response-specific criterion distance normalized by meta-d'
   - $\text{m-dist}_{R_1} = \frac{|\text{meta-}c - \text{meta-}c_{2R_1}|}{\text{meta-}d'}$
   - $\text{m-dist}_{R_2} = \frac{|\text{meta-}c_{2R_2} - \text{meta-}c|}{\text{meta-}d'}$
   - Criterion-invariant measure of threshold relative to metacognitive sensitivity

3. **m-distance2**: Response-specific criterion distance in evidence units (unnormalized)
   - $\text{m-dist2}_{R_1} = |\text{meta-}c - \text{meta-}c_{2R_1}|$
   - $\text{m-dist2}_{R_2} = |\text{meta-}c_{2R_2} - \text{meta-}c|$
   - Comparison quantity to isolate the effect of meta-d' normalization

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/HanxuanXia/Metacognition.git
cd Metacognition/Measures-M-distance
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

The project requires:
- **numpy** (≥1.20.0): Numerical computing
- **scipy** (≥1.7.0): Scientific computing
- **pandas** (≥1.3.0): Data manipulation
- **matplotlib** (≥3.4.0): Visualization
- **seaborn**: Statistical data visualization
- **tqdm**: Progress bar utilities

## Usage

### Running Simulations

All simulations are organized into three main analysis scripts. Each script manipulates a different aspect of the SDT framework:

#### 1. Criterion Sweep Analysis
```bash
cd scripts
python fig_criterion_sweep.py
```

This script investigates the stability of threshold measures under variations in:
- **Type 1 criterion (c)**: The decision boundary for type 1 decisions
- **Confidence thresholds (τₙ, τᵧ)**: The evidence thresholds for confidence reports by response

The script runs two separate analyses:
- **Low metacognitive noise** (σ_meta = 0.001): Near-perfect metacognitive sensitivity
- **High metacognitive noise** (σ_meta = 0.5): Degraded metacognitive sensitivity

**Output**: 
- PNG figures: 3×3 grids showing criterion and threshold effects (Group A: criterion sweep, Group B: tau sweeps)
- CSV files: Aggregated statistics for each parameter value

#### 2. Metacognitive Noise Sweep
```bash
python fig_noise_sweep.py
```

This script examines how confidence measures degrade as internal metacognitive noise increases from σ_meta = 0.001 (near-perfect metacognition) to σ_meta = 100 (random noise) using logarithmic spacing.

The main simulation question is: When metacognitive sensitivity deteriorates and meta-d' declines towards zero, which measures remain interpretable?

**Output**: 
- **noise_sweep.png**: Line plots showing trajectories of proportion confident, m-distance, and m-distance2
- **metad_mratio_vs_noise.png**: Supplementary figure showing meta-d' and M-ratio across noise levels
- **sdt_noise_sweep.csv**: Aggregated data across noise conditions

#### 3. Type 1 Sensitivity Sweep
```bash
python fig_metad_sweep.py
```

This script explores how threshold measures depend on task difficulty (d' ranging from 0.5 to 2.0) under different metacognitive noise conditions.

Examines interactions between:
- Type 1 sensitivity (d')
- Metacognitive sensitivity (meta-d')
- Threshold measure values

**Output**: 
- **metad_sweep_lowNoise.png / highNoise.png**: 2×2 grids showing all three measures and meta-d' identity plots
- **metad_identity_*.png**: Meta-d' vs d' scatter plots with identity lines
- **2x2_*.png**: Summary grids combining all measures
- **sdt_metad_sweep_*.csv**: Type 1 sensitivity sweep results

## Configuration

All simulation parameters can be modified in the scripts or in `sdt_shared.py`:

```python
# Key parameters
n_trials = 50000          # Number of trials per condition
n_replicates = 20         # Number of simulation replicates
dprime = 2.0              # Type 1 sensitivity (d')
sigma_meta = 0.001        # Metacognitive noise (0.001 = low, 0.5 = high)
c = 0.0                   # Decision criterion
tauN = -0.5               # Confidence threshold for "not confident"
tauY = 0.5                # Confidence threshold for "confident"
```

## Key Results

### Critical Findings

1. **Criterion Invariance Under Ordinary Conditions**: Both m-distance and m-distance2 remain largely invariant to type 1 criterion shifts, confirming their effectiveness as criterion-invariant threshold descriptors. This addresses a key limitation of proportion confident.

2. **Response-Specific Sensitivity**: Both m-distance and m-distance2 successfully isolate response-specific confidence criterion spacing while remaining insensitive to criterion placement, as originally intended by Sherman et al. (2018).

3. **Measure Behavior Under Metacognitive Degradation**: The key distinction emerges under high metacognitive noise:
   - **Proportion confident**: Changes with both threshold and task difficulty
   - **m-distance**: Becomes increasingly dependent on meta-d' when noise increases
   - **m-distance2**: Remains stable and bounded across all noise levels

4. **Interpretation of Normalization**: The comparison between m-distance and m-distance2 clarifies that when metacognitive sensitivity deteriorates:
   - m-distance's meta-d' normalization causes increasing dependence on sensitivity changes
   - m-distance2 preserves the fitted criterion spacing in its original units
   - These measures capture different aspects: threshold relative to sensitivity vs. raw criterion distance

## Output Files

### Figure Outputs

**Criterion Sweep Results:**
- **criterion_sweep_sigma0p001.png**: 3×3 grid with σ_meta = 0.001 (low noise)
  - Left (Group A): Criterion sweep showing proportion confident, m-distance, m-distance₂
  - Right (Group B): Confidence threshold sweeps for both responses
- **criterion_sweep_sigma0p5.png**: Same layout with σ_meta = 0.5 (high noise)

**Noise Sweep Results:**
- **noise_sweep.png**: Log-scale plots showing all three measures across noise levels (0.001 to 100)
- **metad_mratio_vs_noise.png**: Supplementary figure tracking meta-d' and M-ratio deterioration

**Type 1 Sensitivity Sweep Results:**
- **metad_sweep_lowNoise.png / highNoise.png**: 2×2 grids showing all measures vs d' plus meta-d' identity plot
- **metad_identity_lowNoise.png / highNoise.png**: Meta-d' vs d' identity plots with confidence bands
- **2x2_lowNoise.png / highNoise.png**: Compact summary grids

### Data Outputs

**Criterion Sweep Data:**
- **sdt_criterion_lowNoise.csv / highNoise.csv**: Criterion manipulation data
- **sdt_tauN_lowNoise.csv / highNoise.csv**: τ_N (negative response threshold) sweep data
- **sdt_tauY_lowNoise.csv / highNoise.csv**: τ_Y (positive response threshold) sweep data

**Noise Sweep Data:**
- **sdt_noise_sweep.csv**: Complete noise sweep results with all measures

**Type 1 Sensitivity Sweep Data:**
- **sdt_metad_sweep.csv**: Main metad sweep results
- **sdt_metad_sweep_lowNoise.csv / highNoise.csv**: Separated by noise condition

## Scientific Applications

This framework is particularly useful for:

- **Perceptual Decision Research**: Understanding how observers set confidence thresholds during sensory decisions
- **Metacognition Assessment**: Separating confidence frequency, metacognitive sensitivity, and threshold in empirical data
- **Clinical Research**: Assessing confidence calibration in clinical populations with potentially impaired metacognition
- **Developmental Psychology**: Investigating how metacognitive threshold placement changes during development
- **Comparative Measure Evaluation**: Choosing appropriate confidence measures based on research questions about confidence behavior
- **Theoretical Development**: Understanding how different aspects of confidence depend on metacognitive sensitivity

## Limitations & Future Directions

### Current Limitations

1. **Equal-variance SDT only**: Simulations use equal-variance assumption; unequal-variance models may show different patterns
2. **Binary confidence**: Uses binary confidence responses rather than graded scales (1-6 ratings)
3. **Additive Gaussian noise**: Metacognitive noise is modeled as additive Gaussian; other noise sources could be investigated
4. **No empirical validation**: Results are simulation-based; empirical replication with human data is needed

### Future Extensions

- [ ] Unequal-variance SDT models
- [ ] Graded confidence scales (1-6 ratings or continuous)
- [ ] Alternative noise distributions (e.g., multiplicative, non-Gaussian)
- [ ] Empirical replication with human participants
- [ ] Hierarchical Bayesian estimation approaches for parameter recovery
- [ ] Multi-dimensional decision spaces (>2 alternatives)
- [ ] Weakly-normalized threshold measures (partially normalized by meta-d')
- [ ] Application to human data from perceptual and memory tasks

## License

This project is licensed under the MIT License—see the LICENSE file for details.

## Acknowledgments

This work builds on foundational research in signal detection theory and metacognition:

- **Maxine T. Sherman** and colleagues: Development of m-distance as a criterion-invariant threshold metric
- **Brian Maniscalco and Hakwan Lau**: The meta-d' framework for estimating metacognitive sensitivity
- **David Green and John Swets**: Foundational signal detection theory
- **Adam Barrett and colleagues**: SDT-based metacognition measures and theoretical frameworks

The project compares these measures systematically to clarify what each captures and under what conditions its behavior is useful for interpreting confidence data.

---

**Last Updated**: April 2026  
