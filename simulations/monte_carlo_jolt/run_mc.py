"""
Monte Carlo Jolt Detection Simulation

This script runs Monte Carlo simulations to evaluate the performance of jolt detection
algorithms across three growth regimes: exponential, logistic, and jolt.

The script generates synthetic capability curves, applies jolt detection, and logs
false-positive and false-negative rates. Results are saved in both Parquet and CSV formats.

Usage:
    python run_mc.py [--device DEVICE] [--n-draws N_DRAWS] [--noise-grid NOISE_GRID] \
                     [--peak-ratio-thresholds PEAK_RATIO_THRESHOLDS] [--use-hybrid-detector] \
                     [--hybrid-final-threshold HYBRID_FINAL_THRESHOLD] [--hybrid-peak-norm-factor HYBRID_PEAK_NORM_FACTOR] \
                     [--hybrid-duration-norm-multiplier HYBRID_DURATION_NORM_MULTIPLIER]

Options:
    --device        Device to use for computation (cpu or cuda)
    --n-draws       Number of Monte Carlo draws per regime and noise level
    --noise-grid    Comma-separated list of noise standard deviations to test
    --peak-ratio-thresholds Comma-separated list of peak ratio thresholds to test (for simple_detection mode)
    --use-hybrid-detector Flag to use the hybrid jolt detector. Default: False
    --hybrid-final-threshold Final threshold for hybrid detector score. Default: 0.6
    --hybrid-peak-norm-factor Peak normalization factor for hybrid detector. Default: 15.0
    --hybrid-duration-norm-multiplier Duration normalization multiplier for hybrid detector. Default: 2.0
"""

import os
import argparse
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Import mcjolt package
from mcjolt.generator import generate_series
from mcjolt.estimator import estimate_derivatives, detect_jolt, detect_jolt_hybrid # Added detect_jolt_hybrid

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Monte Carlo jolt detection simulations")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to use for computation")
    parser.add_argument("--n-draws", type=int, default=200, 
                        help="Number of Monte Carlo draws per regime and noise level")
    parser.add_argument("--noise-grid", type=str, default="0.001,0.005,0.01,0.02,0.05,0.1,0.2",
                        help="Comma-separated list of noise standard deviations to test")
    # Arguments for the simple detector (used if --use-hybrid-detector is False)
    parser.add_argument("--peak-ratio-thresholds", type=str, default="7.5", # Default to a single value if not sweeping
                        help="Comma-separated list of peak ratio thresholds to test for simple_detection mode")
    # Arguments for the hybrid detector
    parser.add_argument("--use-hybrid-detector", action="store_true", 
                        help="Use the hybrid jolt detector instead of the simple/standard one. Default: False")
    parser.add_argument("--hybrid-final-threshold", type=float, default=0.6,
                        help="Final threshold for hybrid detector score. Default: 0.6")
    parser.add_argument("--hybrid-peak-norm-factor", type=float, default=15.0,
                        help="Peak normalization factor for hybrid detector. Default: 15.0")
    parser.add_argument("--hybrid-duration-norm-multiplier", type=float, default=2.0,
                        help="Duration normalization multiplier for hybrid detector. Default: 2.0")
    # Add min_duration for hybrid detector as well, as it's used by it
    parser.add_argument("--min-duration", type=int, default=2,
                        help="Minimum duration for jolt detection (used by hybrid and standard). Default: 2")

    return parser.parse_args()


def run_simulation(
    regime: str,
    params: Dict,
    noise_std: float,
    n_draws: int,
    device: str,
    use_hybrid_detector: bool,
    detector_params: Dict, # Dictionary to hold parameters for the chosen detector
    n_points: int = 100,
) -> Tuple[float, float]:
    """
    Run Monte Carlo simulation for a specific growth regime and noise level.
    """
    true_jolt = (regime == "jolt")
    false_positives = 0
    false_negatives = 0
    
    for i in range(n_draws):
        seed = SEED + i 
        t, y = generate_series(
            regime, 
            params, 
            n_points=n_points, 
            noise_std=noise_std,
            device=device,
            seed=seed
        )
        
        derivs = estimate_derivatives(t, y, method="savgol")
        
        if use_hybrid_detector:
            jolt_detected, jolt_info = detect_jolt_hybrid(
                third_deriv=derivs["third_deriv"],
                second_deriv=derivs["second_deriv"],
                first_deriv=derivs["first_deriv"],
                min_duration=detector_params["min_duration"],
                final_threshold=detector_params["hybrid_final_threshold"],
                peak_norm_factor=detector_params["hybrid_peak_norm_factor"],
                duration_norm_multiplier=detector_params["hybrid_duration_norm_multiplier"]
                # Add weights and adaptive_threshold_base if you want to make them configurable via CLI too
            )
        else:
            # Using the original detect_jolt which has simple_detection mode
            jolt_detected, jolt_info = detect_jolt(
                third_deriv=derivs["third_deriv"],
                second_deriv=derivs["second_deriv"],
                first_deriv=derivs["first_deriv"],
                pattern_check=True, # Assuming pattern_check is True for non-hybrid standard mode
                simple_detection=True, # Using simple_detection for this branch
                peak_ratio_threshold=detector_params["peak_ratio_threshold"],
                min_duration=detector_params["min_duration"]
            )
        
        if jolt_detected and not true_jolt:
            false_positives += 1
        elif not jolt_detected and true_jolt:
            false_negatives += 1
    
    false_positive_rate = false_positives / n_draws if not true_jolt else 0.0
    false_negative_rate = false_negatives / n_draws if true_jolt else 0.0
    
    return false_positive_rate, false_negative_rate


def create_regime_params():
    """Create parameters for each growth regime."""
    return {
        "exponential": {"growth_rate": 2.0},
        "logistic": {"growth_rate": 10.0, "capacity": 1.0, "midpoint": 0.5},
        "jolt": {"base_rate": 2.0, "jolt_magnitude": 10.0, "jolt_time": 0.5, "jolt_width": 0.02}
    }


def create_results_directory():
    """Create directory for results if it doesn't exist."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    (results_dir / "tables").mkdir(exist_ok=True) 
    return results_dir


def save_results(results_df, results_dir, filename_suffix=""):
    """Save results to CSV and Parquet formats."""
    csv_path = results_dir / f"mc_summary{filename_suffix}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    try:
        parquet_path = results_dir / f"mc_summary{filename_suffix}.parquet"
        results_df.to_parquet(parquet_path, index=False)
        print(f"Results saved to {parquet_path}")
    except Exception as e:
        print(f"Could not save as Parquet: {e}")


def create_latex_table(results_df, results_dir, filename_suffix=""):
    """Create LaTeX-ready table of results. Adapts if peak_ratio_threshold is a column."""
    if "peak_ratio_threshold" in results_df.columns and results_df["peak_ratio_threshold"].nunique() > 1:
        # Handle parameter sweep table (multiple peak_ratio_thresholds)
        all_latex_tables = ""
        for pr_threshold in results_df["peak_ratio_threshold"].unique():
            df_subset = results_df[results_df["peak_ratio_threshold"] == pr_threshold]
            table_data = df_subset.pivot_table(
                index="noise_std", 
                columns="regime", 
                values=["false_positive_rate", "false_negative_rate"]
            )
            caption = f"Monte Carlo Simulation Results (Peak Ratio Threshold: {pr_threshold})"
            label = f"tab:mc_results_pr_{str(pr_threshold).replace('.', '_')}"
            if "detector_type" in df_subset.columns:
                detector_name = df_subset["detector_type"].iloc[0]
                caption = f"Monte Carlo Results ({detector_name}, PRT: {pr_threshold})"
                label = f"tab:mc_{detector_name}_prt_{str(pr_threshold).replace('.', '_')}"

            latex_table = table_data.to_latex(
                float_format="%.3f",
                multicolumn=True,
                multicolumn_format="c",
                bold_rows=True,
                caption=caption,
                label=label
            )
            all_latex_tables += f"\n\n% Detector: {detector_name if 'detector_name' in locals() else 'Simple'}, Peak Ratio Threshold: {pr_threshold}\n"
            all_latex_tables += latex_table
        table_path = results_dir / "tables" / f"mc_results{filename_suffix}.tex"
        with open(table_path, "w") as f:
            f.write(all_latex_tables)
        print(f"LaTeX tables saved to {table_path}")
    else:
        # Handle single run (e.g., hybrid or simple with one PRT)
        table_data = results_df.pivot_table(
            index="noise_std", 
            columns="regime", 
            values=["false_positive_rate", "false_negative_rate"]
        )
        caption = "Monte Carlo Simulation Results"
        label = "tab:mc_results_single_run"
        if "detector_type" in results_df.columns:
            detector_name = results_df["detector_type"].iloc[0]
            caption = f"Monte Carlo Simulation Results ({detector_name})"
            if "peak_ratio_threshold" in results_df.columns:
                 prt = results_df["peak_ratio_threshold"].iloc[0]
                 caption += f" PRT: {prt}"
            label = f"tab:mc_results_{detector_name.lower().replace(' ','_')}"

        latex_table = table_data.to_latex(
            float_format="%.3f",
            multicolumn=True,
            multicolumn_format="c",
            bold_rows=True,
            caption=caption,
            label=label
        )
        table_path = results_dir / "tables" / f"mc_results{filename_suffix}.tex"
        with open(table_path, "w") as f:
            f.write(latex_table)
        print(f"LaTeX table saved to {table_path}")


def main():
    """Main function to run Monte Carlo simulations."""
    args = parse_args()
    device = args.device
    n_draws = args.n_draws
    noise_grid = [float(x) for x in args.noise_grid.split(",")]
    
    print(f"Running Monte Carlo simulations with:")
    print(f"  Device: {device}")
    print(f"  Number of draws: {n_draws}")
    print(f"  Noise levels: {noise_grid}")

    detector_params = {
        "min_duration": args.min_duration
    }
    if args.use_hybrid_detector:
        print("  Using Hybrid Detector")
        detector_params.update({
            "hybrid_final_threshold": args.hybrid_final_threshold,
            "hybrid_peak_norm_factor": args.hybrid_peak_norm_factor,
            "hybrid_duration_norm_multiplier": args.hybrid_duration_norm_multiplier
        })
        detector_name_suffix = "_hybrid"
        detector_type_for_df = "Hybrid"
        print(f"    Hybrid Final Threshold: {args.hybrid_final_threshold}")
        print(f"    Hybrid Peak Norm Factor: {args.hybrid_peak_norm_factor}")
        print(f"    Hybrid Duration Norm Multiplier: {args.hybrid_duration_norm_multiplier}")
    else:
        # This branch handles simple detector, potentially with a sweep if multiple PRTs are given
        peak_ratio_thresholds_list = [float(x) for x in args.peak_ratio_thresholds.split(",")]
        print(f"  Using Simple Detector with Peak Ratio Threshold(s): {peak_ratio_thresholds_list}")
        detector_name_suffix = "_simple_prt_sweep" if len(peak_ratio_thresholds_list) > 1 else f"_simple_prt_{peak_ratio_thresholds_list[0]}"
        detector_type_for_df = "Simple"

    regime_params = create_regime_params()
    results_dir = create_results_directory()
    results = []
    start_time = time.time()

    # Determine loop for peak_ratio_thresholds based on detector type
    prt_loop_values = [None] # For hybrid, PRT is not directly iterated here, but part of detector_params
    if not args.use_hybrid_detector:
        prt_loop_values = peak_ratio_thresholds_list

    for pr_thresh_val_for_loop in tqdm(prt_loop_values, desc="Detector Configurations"):
        current_detector_params = detector_params.copy()
        if not args.use_hybrid_detector:
            current_detector_params["peak_ratio_threshold"] = pr_thresh_val_for_loop
        
        for regime in ["exponential", "logistic", "jolt"]:
            params = regime_params[regime]
            desc_text = f"Regime: {regime}"
            if not args.use_hybrid_detector:
                desc_text += f" (PRT: {pr_thresh_val_for_loop})"
            else:
                desc_text += f" (Hybrid)"

            for noise_std in tqdm(noise_grid, desc=desc_text, leave=False):
                fp_rate, fn_rate = run_simulation(
                    regime, 
                    params, 
                    noise_std, 
                    n_draws, 
                    device,
                    use_hybrid_detector=args.use_hybrid_detector,
                    detector_params=current_detector_params
                )
                result_entry = {
                    "detector_type": detector_type_for_df,
                    "regime": regime,
                    "noise_std": noise_std,
                    "false_positive_rate": fp_rate,
                    "false_negative_rate": fn_rate
                }
                if not args.use_hybrid_detector:
                    result_entry["peak_ratio_threshold"] = pr_thresh_val_for_loop
                # Add hybrid specific params to results if needed for detailed logging
                if args.use_hybrid_detector:
                    result_entry["hybrid_final_threshold"] = args.hybrid_final_threshold
                    # Add other hybrid params if desired in output CSV
                results.append(result_entry)
    
    elapsed_time = time.time() - start_time
    print(f"Simulations completed in {elapsed_time:.2f} seconds")
    
    results_df = pd.DataFrame(results)
    
    save_results(results_df, results_dir, filename_suffix=detector_name_suffix)
    create_latex_table(results_df, results_dir, filename_suffix=detector_name_suffix)
    # Heatmap generation would need to be adapted or run for specific configurations.

if __name__ == "__main__":
    main()

