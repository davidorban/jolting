# Jolting Technologies Hypothesis - Code Repository

This repository contains the complete code implementation for the empirical studies supporting the Jolting Technologies Hypothesis paper. The repository includes three main components:

1. **Monte Carlo Jolt Detection**: A simulation framework for evaluating jolt detection algorithms
2. **AgentBench Analysis**: Application of jolt detection to AI agent capabilities data
3. **Governance PID-Lag Model**: A toy model simulating governance responses to technological acceleration and jolts

## Repository Structure

- `simulations/` - Contains simulation code for various empirical studies
  - `monte_carlo_jolt/` - Monte Carlo simulation for evaluating jolt detection algorithms
- `analysis/` - Contains analysis code for empirical studies
  - `agentbench/` - Analysis of AgentBench performance data
  - `imagenet/` - Analysis of ImageNet performance data (to be implemented)
- `governance/` - Contains governance models
  - `pid_lag/` - Governance PID-Lag model implementation
- `supplementary_materials/` - Contains supplementary materials for the paper
  - `figures/` - High-resolution figures for all empirical components
  - `tables/` - Data tables in CSV and LaTeX formats

## Monte Carlo Jolt Detection

The `monte_carlo_jolt` directory contains a complete implementation of a Monte Carlo simulation framework for evaluating jolt detection algorithms. The simulation generates synthetic capability curves for three growth regimes (exponential, logistic, and jolt) and evaluates the performance of various jolt detection approaches.

## AgentBench Jolt Analysis

The `analysis/agentbench` directory contains an implementation of jolt detection applied to AgentBench leaderboard data. It generates synthetic monthly snapshots of AI agent performance and analyzes them to detect periods of super-exponential acceleration. The implementation is designed to work with both synthetic data and real historical data when available.

### Key Components

- `agentbench_jolt/` - Python package implementing core functionality
  - `generator.py` - Synthetic AgentBench data generation
  - `analyzer.py` - Data preprocessing and jolt detection
- `run_agentbench_analysis.py` - Main script to run the analysis
- `run_stronger_jolt.py` - Script for testing with stronger jolt parameters
- `run_extreme_jolt.py` - Script for testing with extreme jolt parameters
- `notebooks/` - Jupyter notebooks for visualization and analysis
  - `Figure_AgentBench.ipynb` - Interactive notebook for creating publication-ready figures
- `data/` - Directory for storing synthetic and historical data
- `results/` - Directory for storing analysis results
  - `figures/` - Generated visualizations
  - `tables/` - LaTeX tables for publication

### Usage

```bash
# Generate synthetic data and run the analysis with default parameters
python run_agentbench_analysis.py

# Customize the synthetic data generation
python run_agentbench_analysis.py --start-date 2024-01-01 --num-months 18 --jolt-month 9 --jolt-magnitude 15.0

# Run with stronger jolt parameters for better detection
python run_stronger_jolt.py

# Run with extreme jolt parameters to ensure detection
python run_extreme_jolt.py
```

### Integration with Monte Carlo Module

The AgentBench analysis leverages the hybrid jolt detection algorithm developed in the Monte Carlo simulation component. This ensures consistency in methodology across different empirical studies and allows for direct comparison of results.

## Governance PID-Lag Toy Model

The `governance/pid_lag` directory contains an implementation of a PID controller with time lag to simulate governance responses to technological acceleration and jolts. The model captures key real-world governance challenges including response delays, implementation time, effectiveness decay, and effectiveness caps.

### Key Components

- `controller.py` - PID controller with lag implementation
  - PID controller: Adjusts governance response based on proportional, integral, and derivative terms
  - Lag model: Simulates real-world governance challenges like response delay and effectiveness decay
- `technology.py` - Technology progress models with jolts
  - Exponential growth with optional jolts
  - Risk calculation using logistic function
  - Governance effect simulation
- `run_pid_lag_model.py` - Main script to run simulations
- `notebooks/` - Jupyter notebooks for interactive exploration
  - `PID_Lag_Explorer.ipynb` - Interactive parameter exploration

### Usage

```bash
# Run a basic simulation with default parameters
python run_pid_lag_model.py

# Run with custom parameters
python run_pid_lag_model.py --jolt-time 50 --jolt-magnitude 0.3 --response-delay 5

# Run with no jolt (for baseline comparison)
python run_pid_lag_model.py --jolt-time -1
```

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
