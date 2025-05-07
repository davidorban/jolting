# Jolting Technologies Hypothesis - Code Repository

This repository contains the code implementation for the empirical studies supporting the Jolting Technologies Hypothesis paper.

## Repository Structure

- `simulations/` - Contains simulation code for various empirical studies
  - `monte_carlo_jolt/` - Monte Carlo simulation for evaluating jolt detection algorithms
  - `agentbench/` - Analysis of AgentBench performance data (to be implemented)
  - `governance_pid/` - Governance PID-Lag model implementation (to be implemented)

## Monte Carlo Jolt Detection

The `monte_carlo_jolt` directory contains a complete implementation of a Monte Carlo simulation framework for evaluating jolt detection algorithms. The simulation generates synthetic capability curves for three growth regimes (exponential, logistic, and jolt) and evaluates the performance of various jolt detection approaches.

### Key Components

- `mcjolt/` - Python package implementing core functionality
  - `generator.py` - Synthetic capability curve generation
  - `estimator.py` - Smoothing, derivative estimation, and jolt detection algorithms
- `run_mc.py` - Main script to run Monte Carlo simulations
- `mc_style.mplstyle` - Matplotlib style for publication-ready figures
- `notebooks/` - Jupyter notebooks for visualization and analysis
- `results/` - Simulation results in CSV and Parquet formats
- `tables/` - LaTeX tables for publication

### Usage

```bash
# Basic usage with hybrid detector
python run_mc.py --device cpu --n-draws 200 --noise-grid 0.01,0.05,0.1 --use-hybrid-detector

# Using simple detector with custom peak ratio threshold
python run_mc.py --device cpu --n-draws 200 --noise-grid 0.01,0.05,0.1 --peak-ratio-thresholds 7.5

# Running with GPU acceleration (if available)
python run_mc.py --device cuda --n-draws 500 --noise-grid 0.01,0.05,0.1 --use-hybrid-detector
```

## Dependencies

See `requirements.txt` or `environment.yml` in the respective directories for the required dependencies.

## Citation

If you use this code in your research, please cite our paper:

```
[Citation information will be added after publication]
```

## License

MIT License

Copyright (c) 2025 Anonymous Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
