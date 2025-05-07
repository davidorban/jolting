# Monte Carlo Jolt Detection

This directory contains the implementation of the Monte Carlo Appendix component for the Jolting Technologies Hypothesis paper. It provides tools for generating synthetic capability curves and detecting jolts (positive third derivatives) in technological progress data.

## Overview

The Monte Carlo simulation framework evaluates the performance of jolt detection algorithms across three growth regimes (exponential, logistic, and jolt) with varying levels of noise. The framework implements multiple detection approaches, including a simple peak ratio-based detector and a more sophisticated hybrid detector that combines multiple criteria for improved accuracy.

## Directory Structure

```
monte_carlo_jolt/
├── mcjolt/                  # Core Python package
│   ├── __init__.py          # Package initialization
│   ├── generator.py         # Synthetic curve generation
│   └── estimator.py         # Derivative estimation and jolt detection
├── notebooks/               # Jupyter notebooks
│   └── Figure_MC.ipynb      # Analysis and visualization notebook
├── mc_style.mplstyle        # Matplotlib style for publication-ready figures
└── run_mc.py                # Script to run Monte Carlo simulations
```

## Components

### Jolt Detection Approaches

#### Simple Detector
The simple detector uses a peak ratio threshold to identify jolts. It compares the maximum third derivative value to the median absolute third derivative, providing a straightforward approach that works well for clear jolt signals.

#### Hybrid Detector
The hybrid detector combines multiple criteria to achieve a better balance between false positives and false negatives:
- **Peak Analysis**: Evaluates the prominence of peaks in the third derivative
- **Pattern Matching**: Checks if the pattern of first, second, and third derivatives matches the expected pattern for a jolt
- **Duration Analysis**: Considers the duration of positive third derivative regions

The hybrid detector achieves excellent results with low false positive rates (5-16%) while maintaining perfect jolt detection (0% false negatives).

### mcjolt Package

The `mcjolt` package provides the core functionality for the Monte Carlo simulations:

- **generator.py**: Generates synthetic capability curves for three growth regimes:
  - Exponential growth
  - Logistic (S-curve) growth
  - Jolt (super-exponential with positive third derivative)

- **estimator.py**: Implements methods for:
  - Smoothing time series data using Savitzky-Golay filters
  - Estimating derivatives using finite differences and automatic differentiation
  - Detecting jolts based on third derivative analysis
  - Bootstrap confidence interval estimation

### run_mc.py

This script runs Monte Carlo simulations to evaluate the performance of jolt detection algorithms:

- Generates synthetic data for each growth regime with varying noise levels
- Applies jolt detection and logs false-positive and false-negative rates
- Saves results in both CSV and Parquet formats
- Creates visualizations of the results

#### Usage

```bash
python run_mc.py [--device DEVICE] [--n-draws N_DRAWS] [--noise-grid NOISE_GRID]
```

Options:
- `--device`: Device to use for computation (cpu or cuda)
- `--n-draws`: Number of Monte Carlo draws per regime and noise level
- `--noise-grid`: Comma-separated list of noise standard deviations to test

### Figure_MC.ipynb

This Jupyter notebook analyzes the results of the Monte Carlo simulations and produces:

1. False-positive/negative heatmaps
2. ROC curves
3. ΔROC curves to quantify performance degradation with increasing noise
4. LaTeX-ready tables for the manuscript

## Output Files

The simulations generate the following output files:

- **results/mc_summary.csv**: CSV file with simulation results
- **results/mc_summary.parquet**: Parquet file with simulation results (faster loading)
- **figures/mc_heatmap.pdf**: Heatmap of false-positive and false-negative rates
- **figures/mc_roc_curves.pdf**: ROC curves for different noise levels
- **figures/mc_delta_roc_curves.pdf**: ΔROC curves showing performance degradation
- **tables/mc_results.tex**: LaTeX-ready table of simulation results

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Pandas
- Matplotlib
- Seaborn
- PyArrow (optional, for Parquet support)
- PyTorch (optional, for GPU acceleration)
- Jupyter (for running notebooks)

## Usage

```bash
# Run Monte Carlo simulations with default parameters (simple detector)
python run_mc.py

# Run with hybrid detector (recommended for best results)
python run_mc.py --device cpu --n-draws 200 --noise-grid 0.01,0.05,0.1 --use-hybrid-detector

# Run with simple detector and custom peak ratio threshold
python run_mc.py --device cpu --n-draws 200 --noise-grid 0.01,0.05,0.1 --peak-ratio-thresholds 7.5

# Run with hybrid detector and custom parameters
python run_mc.py --use-hybrid-detector --hybrid-final-threshold 0.65 --hybrid-peak-norm-factor 12.0

# Run with GPU acceleration (if available)
python run_mc.py --device cuda --n-draws 500 --noise-grid 0.01,0.03,0.05,0.07,0.1 --use-hybrid-detector
```

## Integration with Manuscript

The results from these simulations are used in the Jolting Technologies manuscript to validate the jolt detection methodology. The figures and tables generated by this code are directly referenced in the paper.

## License

MIT License

Copyright (c) 2025 Anonymous Authors

See the LICENSE file in the root directory for full license details.
