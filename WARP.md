# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Overview

This repository implements empirical studies for the Jolting Technologies Hypothesis paper, focusing on jolt detection in technological capability curves. The codebase consists of three main research components:

1. **Monte Carlo Jolt Detection** (`simulations/monte_carlo_jolt/`) - Simulation framework for evaluating jolt detection algorithms
2. **AgentBench Analysis** (`analysis/agentbench/`) - Application of jolt detection to AI agent performance data  
3. **Governance PID-Lag Model** (`governance/pid_lag/`) - Governance response simulation to technological jolts

## Architecture

### Core Algorithm Components

The repository implements a **hybrid jolt detection system** that operates across all three research components:

- **Synthetic Data Generation**: Three growth regimes (exponential, logistic, jolt) with configurable noise
- **Derivative Estimation**: Multiple smoothing methods (Savitzky-Golay, splines, GAM) for robust derivative calculation
- **Jolt Detection**: Dual approach with simple peak ratio detection and advanced pattern matching

### Data Flow

1. **Generate**: Synthetic capability curves with specified growth patterns
2. **Smooth**: Apply smoothing to handle noisy real-world data
3. **Differentiate**: Calculate 1st, 2nd, and 3rd derivatives
4. **Detect**: Apply jolt detection algorithms to identify super-exponential acceleration
5. **Validate**: Monte Carlo evaluation of detection performance

### Package Structure

Each component is organized as a Python package:
- `mcjolt/` - Core Monte Carlo simulation algorithms
- `agentbench_jolt/` - AgentBench-specific data processing
- `pid_lag/` - Governance modeling (organized as modules rather than package)

## Common Development Commands

### Environment Setup
```bash
# For Monte Carlo simulations
cd simulations/monte_carlo_jolt
pip install -r requirements.txt
# or
conda env create -f environment.yml

# For AgentBench analysis  
cd analysis/agentbench
pip install -r requirements.txt
# or
conda env create -f environment.yml

# For Governance model
cd governance/pid_lag
pip install -r requirements.txt
# or
conda env create -f environment.yml
```

### Running Simulations

#### Monte Carlo Jolt Detection
```bash
cd simulations/monte_carlo_jolt

# Basic simulation with hybrid detector
python run_mc.py --device cpu --n-draws 200 --noise-grid 0.01,0.05,0.1 --use-hybrid-detector

# Simple detector with custom threshold
python run_mc.py --device cpu --n-draws 200 --noise-grid 0.01,0.05,0.1 --peak-ratio-thresholds 7.5

# GPU acceleration (if available)
python run_mc.py --device cuda --n-draws 500 --noise-grid 0.01,0.05,0.1 --use-hybrid-detector
```

#### AgentBench Analysis
```bash
cd analysis/agentbench

# Default analysis
python run_agentbench_analysis.py

# Custom parameters
python run_agentbench_analysis.py --start-date 2024-01-01 --num-months 18 --jolt-month 9 --jolt-magnitude 15.0

# Stronger jolt scenarios
python run_stronger_jolt.py
python run_extreme_jolt.py
```

#### Governance PID-Lag Model
```bash
cd governance/pid_lag

# Basic simulation
python run_pid_lag_model.py

# Custom parameters
python run_pid_lag_model.py --jolt-time 50 --jolt-magnitude 0.3 --response-delay 5

# Baseline without jolt
python run_pid_lag_model.py --jolt-time -1
```

### Jupyter Notebooks

Interactive analysis and visualization:
```bash
# Monte Carlo results visualization
jupyter notebook simulations/monte_carlo_jolt/notebooks/Figure_MC.ipynb

# AgentBench publication figures
jupyter notebook analysis/agentbench/notebooks/Figure_AgentBench.ipynb

# Governance model parameter exploration
jupyter notebook governance/pid_lag/notebooks/PID_Lag_Explorer.ipynb
```

## Key Implementation Details

### Jolt Detection Methods

The repository implements two complementary detection approaches:

1. **Simple Detection**: Uses peak ratio threshold (default 7.5) comparing maximum third derivative to median absolute third derivative
2. **Advanced Detection**: Adaptive thresholding with duration requirements and optional pattern validation

### Growth Regime Parameters

When working with synthetic data generation:

- **Exponential**: `{"growth_rate": float}` - Base exponential growth rate
- **Logistic**: `{"growth_rate": float, "capacity": float, "midpoint": float}` - S-curve parameters  
- **Jolt**: `{"base_rate": float, "jolt_magnitude": float, "jolt_time": float, "jolt_width": float}` - Super-exponential with timing

### Device Support

All Monte Carlo simulations support both CPU and CUDA acceleration. Use `--device cuda` for GPU acceleration when available.

## Documentation Website

The repository includes a Jekyll-based documentation site in `docs/` that contains:
- Research blog posts explaining the methodology
- Supplementary materials and figures
- Published paper (jolting-technologies-david-orban.pdf)

To work with the documentation:
```bash
cd docs
bundle install  # if using Jekyll locally
bundle exec jekyll serve
```