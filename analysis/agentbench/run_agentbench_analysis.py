#!/usr/bin/env python
"""
AgentBench Jolt Analysis

This script generates synthetic AgentBench leaderboard data and analyzes it
to detect jolts (super-exponential acceleration) in AI agent performance.

Usage:
    python run_agentbench_analysis.py [--start-date START_DATE] [--num-months NUM_MONTHS]
                                     [--jolt-month JOLT_MONTH] [--jolt-magnitude JOLT_MAGNITUDE]
                                     [--noise-std NOISE_STD] [--seed SEED]
"""

import os
import argparse
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from agentbench_jolt.generator import generate_model_performance, save_synthetic_data
from agentbench_jolt.analyzer import (
    preprocess_agentbench_data,
    detect_agentbench_jolt,
    plot_agentbench_jolt,
    create_latex_table
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run AgentBench jolt detection analysis")
    
    parser.add_argument("--start-date", type=str, default="2024-05-01",
                        help="Start date for synthetic data (YYYY-MM-DD)")
    parser.add_argument("--num-months", type=int, default=12,
                        help="Number of months to generate data for")
    parser.add_argument("--jolt-month", type=int, default=6,
                        help="Month in which the jolt occurs (0-indexed, -1 for no jolt)")
    parser.add_argument("--jolt-magnitude", type=float, default=10.0,
                        help="Magnitude of the jolt")
    parser.add_argument("--noise-std", type=float, default=2.0,
                        help="Standard deviation of random noise")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    return parser.parse_args()


def main():
    """Main function to run the AgentBench analysis."""
    args = parse_args()
    
    # Create directories if they don't exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    # Convert jolt_month -1 to None (no jolt)
    jolt_month = args.jolt_month if args.jolt_month >= 0 else None
    
    # Parse start date
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    
    print(f"Generating synthetic AgentBench data with:")
    print(f"  Start date: {args.start_date}")
    print(f"  Number of months: {args.num_months}")
    print(f"  Jolt month: {jolt_month if jolt_month is not None else 'None (no jolt)'}")
    print(f"  Jolt magnitude: {args.jolt_magnitude}")
    print(f"  Noise std: {args.noise_std}")
    print(f"  Seed: {args.seed}")
    
    # Generate synthetic data
    synthetic_data = generate_model_performance(
        start_date=start_date,
        num_months=args.num_months,
        jolt_month=jolt_month,
        jolt_magnitude=args.jolt_magnitude,
        noise_std=args.noise_std,
        seed=args.seed
    )
    
    # Save synthetic data
    data_path = "data/synthetic_agentbench_data.csv"
    save_synthetic_data(synthetic_data, data_path)
    print(f"Saved synthetic data to {data_path}")
    
    # Preprocess data
    processed_df = preprocess_agentbench_data(
        synthetic_data,
        metric='median_score',
        aggregation='max'
    )
    
    # Detect jolt
    jolt_detected, jolt_info, derivs = detect_agentbench_jolt(
        processed_df,
        hybrid_final_threshold=0.6,
        hybrid_peak_norm_factor=15.0,
        hybrid_duration_norm_multiplier=2.0
    )
    
    # Plot results
    fig = plot_agentbench_jolt(
        processed_df, 
        derivs, 
        jolt_info,
        title="AgentBench Performance Jolt Analysis (Synthetic Data)",
        save_path="results/figures/agentbench_jolt.png"
    )
    
    # Create LaTeX table
    create_latex_table(
        processed_df, 
        jolt_info, 
        "results/tables/agentbench_jolt.tex"
    )
    
    # Save processed data
    processed_df.to_csv("results/processed_agentbench_data.csv", index=False)
    
    # Print results
    print("\nAnalysis Results:")
    print(f"  Jolt detected: {jolt_detected}")
    if jolt_detected:
        print(f"  Jolt date: {jolt_info.get('jolt_date', 'Unknown')}")
        print(f"  Jolt score: {jolt_info.get('score', 0):.2f}")
        print(f"  Peak score component: {jolt_info.get('components', {}).get('peak_score', 0):.2f}")
        print(f"  Pattern score component: {jolt_info.get('components', {}).get('pattern_score', 0):.2f}")
        print(f"  Duration score component: {jolt_info.get('components', {}).get('duration_score', 0):.2f}")
    
    print("\nResults saved to:")
    print(f"  Figure: results/figures/agentbench_jolt.png")
    print(f"  LaTeX table: results/tables/agentbench_jolt.tex")
    print(f"  Processed data: results/processed_agentbench_data.csv")


if __name__ == "__main__":
    main()
