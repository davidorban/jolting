#!/usr/bin/env python
"""
Run AgentBench analysis with stronger jolt parameters to ensure detection.
This script generates synthetic data with a more pronounced jolt and runs the analysis
with more sensitive detection parameters.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from agentbench_jolt.generator import generate_synthetic_agentbench_data
from agentbench_jolt.analyzer import (
    preprocess_agentbench_data,
    detect_agentbench_jolt,
    plot_agentbench_jolt,
    create_latex_table
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run AgentBench analysis with stronger jolt parameters')
    
    # Data generation parameters
    parser.add_argument('--start-date', type=str, default='2023-01-01',
                        help='Start date for synthetic data (YYYY-MM-DD)')
    parser.add_argument('--num-months', type=int, default=24,
                        help='Number of months to generate')
    parser.add_argument('--jolt-month', type=int, default=12,
                        help='Month when jolt occurs (0-indexed, -1 for no jolt)')
    parser.add_argument('--jolt-magnitude', type=float, default=20.0,
                        help='Magnitude of the jolt (higher values = stronger jolt)')
    parser.add_argument('--jolt-width', type=int, default=3,
                        help='Width of the jolt in months')
    parser.add_argument('--noise-std', type=float, default=1.0,
                        help='Standard deviation of noise')
    
    # Detection parameters
    parser.add_argument('--hybrid-final-threshold', type=float, default=0.5,
                        help='Final threshold for hybrid detector score')
    parser.add_argument('--hybrid-peak-norm-factor', type=float, default=10.0,
                        help='Peak normalization factor for hybrid detector')
    parser.add_argument('--hybrid-duration-norm-multiplier', type=float, default=1.5,
                        help='Duration normalization multiplier for hybrid detector')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output-prefix', type=str, default='stronger_jolt',
                        help='Prefix for output files')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Print parameters
    print("Generating synthetic AgentBench data with stronger jolt parameters:")
    print(f"  Start date: {args.start_date}")
    print(f"  Number of months: {args.num_months}")
    print(f"  Jolt month: {args.jolt_month}")
    print(f"  Jolt magnitude: {args.jolt_magnitude}")
    print(f"  Jolt width: {args.jolt_width}")
    print(f"  Noise std: {args.noise_std}")
    print(f"  Seed: {args.seed}")
    
    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    
    # Generate synthetic data
    data_path = f"data/{args.output_prefix}_agentbench_data.csv"
    df = generate_synthetic_agentbench_data(
        start_date=args.start_date,
        num_months=args.num_months,
        jolt_month=args.jolt_month,
        jolt_magnitude=args.jolt_magnitude,
        jolt_width=args.jolt_width,
        noise_std=args.noise_std,
        seed=args.seed
    )
    df.to_csv(data_path, index=False)
    print(f"Saved synthetic data to {data_path}")
    
    # Preprocess data
    processed_df = preprocess_agentbench_data(df)
    processed_df.to_csv(f"results/{args.output_prefix}_processed_data.csv", index=False)
    
    # Detect jolt
    jolt_detected, jolt_info, derivs = detect_agentbench_jolt(
        processed_df,
        hybrid_final_threshold=args.hybrid_final_threshold,
        hybrid_peak_norm_factor=args.hybrid_peak_norm_factor,
        hybrid_duration_norm_multiplier=args.hybrid_duration_norm_multiplier
    )
    
    # Print results
    print("\nAnalysis Results:")
    print(f"  Jolt detected: {jolt_detected}")
    if jolt_detected:
        print(f"  Jolt date: {jolt_info.get('jolt_date', 'Unknown')}")
        print(f"  Jolt score: {jolt_info.get('score', 0):.2f}")
        print(f"  Peak score component: {jolt_info.get('components', {}).get('peak_score', 0):.2f}")
        print(f"  Pattern score component: {jolt_info.get('components', {}).get('pattern_score', 0):.2f}")
        print(f"  Duration score component: {jolt_info.get('components', {}).get('duration_score', 0):.2f}")
    
    # Create visualization
    fig = plot_agentbench_jolt(
        processed_df, 
        derivs, 
        jolt_info,
        title=f"AgentBench Performance Jolt Analysis (Stronger Parameters)",
        save_path=f"results/figures/{args.output_prefix}_jolt.png"
    )
    
    # Create LaTeX table
    create_latex_table(
        processed_df, 
        jolt_info, 
        f"results/tables/{args.output_prefix}_jolt.tex"
    )
    
    print("\nResults saved to:")
    print(f"  Figure: results/figures/{args.output_prefix}_jolt.png")
    print(f"  LaTeX table: results/tables/{args.output_prefix}_jolt.tex")
    print(f"  Processed data: results/{args.output_prefix}_processed_data.csv")

if __name__ == "__main__":
    main()
