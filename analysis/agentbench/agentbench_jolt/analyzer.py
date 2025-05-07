"""
AgentBench Data Analysis Module

This module provides functions for preprocessing AgentBench data,
estimating derivatives, and detecting jolts in performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage

# Import the hybrid jolt detector from the Monte Carlo module
import sys
import os

# Add the Monte Carlo module to the path
monte_carlo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                               '../../../simulations/monte_carlo_jolt'))
if monte_carlo_path not in sys.path:
    sys.path.append(monte_carlo_path)

from mcjolt.estimator import estimate_derivatives, detect_jolt_hybrid


def preprocess_agentbench_data(
    df: pd.DataFrame,
    metric: str = 'median_score',
    aggregation: str = 'max',
    min_date: Optional[str] = None,
    max_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Preprocess AgentBench data for jolt detection analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        AgentBench leaderboard data
    metric : str, default='median_score'
        Performance metric to analyze
    aggregation : str, default='max'
        How to aggregate model scores for each date ('max', 'mean', 'median')
    min_date : str or None, default=None
        Minimum date to include (YYYY-MM-DD)
    max_date : str or None, default=None
        Maximum date to include (YYYY-MM-DD)
        
    Returns
    -------
    pd.DataFrame
        Preprocessed data with columns:
        - date: Date in YYYY-MM-DD format
        - date_ordinal: Date as ordinal number for numerical analysis
        - performance: Aggregated performance metric
    """
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Filter by date if specified
    if min_date:
        data = data[data['date'] >= min_date]
    if max_date:
        data = data[data['date'] <= max_date]
    
    # Convert date to datetime if it's not already
    if not pd.api.types.is_datetime64_dtype(data['date']):
        data['date'] = pd.to_datetime(data['date'])
    
    # Sort by date
    data = data.sort_values('date')
    
    # Aggregate performance by date
    if aggregation == 'max':
        agg_data = data.groupby('date')[metric].max().reset_index()
    elif aggregation == 'mean':
        agg_data = data.groupby('date')[metric].mean().reset_index()
    elif aggregation == 'median':
        agg_data = data.groupby('date')[metric].median().reset_index()
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    # Convert date to ordinal for numerical analysis
    agg_data['date_ordinal'] = agg_data['date'].apply(lambda x: x.toordinal())
    
    # Rename the metric column to 'performance'
    agg_data = agg_data.rename(columns={metric: 'performance'})
    
    return agg_data


def detect_agentbench_jolt(
    df: pd.DataFrame,
    window_length: Optional[int] = None,
    polyorder: int = 3,
    method: str = 'savgol',
    hybrid_final_threshold: float = 0.6,
    hybrid_peak_norm_factor: float = 15.0,
    hybrid_duration_norm_multiplier: float = 2.0
) -> Tuple[bool, Dict, Dict]:
    """
    Detect jolts in AgentBench performance data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed AgentBench data with date_ordinal and performance columns
    window_length : int or None, default=None
        Window length for smoothing (None for automatic selection)
    polyorder : int, default=3
        Polynomial order for smoothing
    method : str, default='savgol'
        Smoothing method ('savgol' or 'spline')
    hybrid_final_threshold : float, default=0.6
        Final threshold for hybrid detector score
    hybrid_peak_norm_factor : float, default=15.0
        Peak normalization factor for hybrid detector
    hybrid_duration_norm_multiplier : float, default=2.0
        Duration normalization multiplier for hybrid detector
        
    Returns
    -------
    Tuple[bool, Dict, Dict]
        - Whether a jolt was detected
        - Jolt detection details
        - Dictionary containing the derivatives and smoothed data
    """
    # Extract x and y values
    x = df['date_ordinal'].values
    y = df['performance'].values
    
    # Normalize x to [0, 1] for better numerical stability
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
    
    # Estimate derivatives
    derivs = estimate_derivatives(
        x_norm, y, 
        window_length=window_length, 
        polyorder=polyorder, 
        method=method
    )
    
    # Add smoothed data to derivatives dictionary
    derivs['smoothed'] = derivs.get('smoothed_data', y)
    
    # Detect jolt using the hybrid detector
    jolt_detected, jolt_info = detect_jolt_hybrid(
        third_deriv=derivs['third_deriv'],
        second_deriv=derivs['second_deriv'],
        first_deriv=derivs['first_deriv'],
        final_threshold=hybrid_final_threshold,
        peak_norm_factor=hybrid_peak_norm_factor,
        duration_norm_multiplier=hybrid_duration_norm_multiplier
    )
    
    # Add date information to jolt_info
    if jolt_detected and 'jolt_index' in jolt_info:
        jolt_idx = jolt_info['jolt_index']
        if 0 <= jolt_idx < len(x):
            jolt_date_ordinal = x[jolt_idx]
            jolt_date = datetime.fromordinal(int(jolt_date_ordinal))
            jolt_info['jolt_date'] = jolt_date.strftime('%Y-%m-%d')
    
    return jolt_detected, jolt_info, derivs


def plot_agentbench_jolt(
    df: pd.DataFrame,
    derivs: Dict[str, np.ndarray],
    jolt_info: Dict,
    title: str = "AgentBench Performance Jolt Analysis",
    figsize: Tuple[int, int] = (15, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a visualization of AgentBench performance with jolt detection.
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed AgentBench data
    derivs : Dict
        Dictionary containing derivatives and smoothed data
    jolt_info : Dict
        Jolt detection details
    title : str, default='AgentBench Performance Jolt Analysis'
        Figure title
    figsize : Tuple[int, int], default=(15, 12)
        Figure size
    save_path : str or None, default=None
        Path to save the figure (None to not save)
        
    Returns
    -------
    plt.Figure
        Matplotlib figure with the visualization
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Get the dates and convert from ordinal
    dates = [datetime.fromordinal(int(d)) for d in df['date_ordinal']]
    
    # Plot 1: Performance over time with smoothed curve
    ax1 = axes[0]
    ax1.scatter(dates, df['performance'], color='blue', alpha=0.7, label='Actual Performance')
    ax1.plot(dates, derivs['smoothed'], color='darkblue', linewidth=2, label='Smoothed Performance')
    
    # Highlight jolt region if detected
    if jolt_info.get('positive_regions', []):
        for start, end in jolt_info['positive_regions']:
            if start < len(dates) and end <= len(dates):
                jolt_region_dates = dates[start:end]
                jolt_region_y = derivs['smoothed'][start:end]
                ax1.fill_between(jolt_region_dates, 0, jolt_region_y, color='red', alpha=0.2)
    
    # Mark jolt point if detected
    if 'jolt_index' in jolt_info and 0 <= jolt_info['jolt_index'] < len(dates):
        jolt_idx = jolt_info['jolt_index']
        jolt_date = dates[jolt_idx]
        jolt_y = derivs['smoothed'][jolt_idx]
        ax1.scatter([jolt_date], [jolt_y], color='red', s=100, zorder=5, 
                   label=f"Jolt Point ({jolt_info.get('jolt_date', 'Unknown')})")
    
    ax1.set_title('AgentBench Performance Over Time', fontsize=14)
    ax1.set_ylabel('Performance Score', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: First and second derivatives
    ax2 = axes[1]
    ax2.plot(dates, derivs['first_deriv'], color='green', linewidth=2, label='First Derivative (Velocity)')
    ax2.plot(dates, derivs['second_deriv'], color='purple', linewidth=2, label='Second Derivative (Acceleration)')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    ax2.set_title('First and Second Derivatives', fontsize=14)
    ax2.set_ylabel('Derivative Value', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Third derivative with jolt detection
    ax3 = axes[2]
    ax3.plot(dates, derivs['third_deriv'], color='red', linewidth=2, label='Third Derivative (Jolt)')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Highlight jolt regions
    if jolt_info.get('positive_regions', []):
        for start, end in jolt_info['positive_regions']:
            if start < len(dates) and end <= len(dates):
                jolt_region_dates = dates[start:end]
                jolt_region_y = derivs['third_deriv'][start:end]
                ax3.fill_between(jolt_region_dates, 0, jolt_region_y, 
                                color='red', alpha=0.3, label='Jolt Region')
    
    # Add jolt score information
    if 'score' in jolt_info:
        score_text = f"Jolt Score: {jolt_info['score']:.2f}"
        if jolt_info['score'] > 0.6:
            score_text += " (Significant Jolt Detected)"
        ax3.text(0.02, 0.95, score_text, transform=ax3.transAxes, 
                fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax3.set_title('Third Derivative (Jolt)', fontsize=14)
    ax3.set_ylabel('Jolt Magnitude', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # Overall figure settings
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save figure if path is provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_latex_table(
    df: pd.DataFrame,
    jolt_info: Dict,
    output_path: str
) -> None:
    """
    Create a LaTeX table summarizing the AgentBench jolt detection results.
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed AgentBench data
    jolt_info : Dict
        Jolt detection details
    output_path : str
        Path to save the LaTeX table
    """
    # Extract key metrics
    start_date = df['date'].min().strftime('%Y-%m-%d')
    end_date = df['date'].max().strftime('%Y-%m-%d')
    num_months = len(df)
    
    initial_performance = df['performance'].iloc[0]
    final_performance = df['performance'].iloc[-1]
    growth_percentage = (final_performance - initial_performance) / initial_performance * 100
    
    # Jolt information
    jolt_detected = 'Yes' if jolt_info.get('score', 0) > 0.6 else 'No'
    jolt_date = jolt_info.get('jolt_date', 'N/A')
    jolt_score = jolt_info.get('score', 0)
    
    # Component scores
    peak_score = jolt_info.get('components', {}).get('peak_score', 0)
    pattern_score = jolt_info.get('components', {}).get('pattern_score', 0)
    duration_score = jolt_info.get('components', {}).get('duration_score', 0)
    
    # Create LaTeX table
    latex_content = r"""
\begin{table}[ht]
\centering
\caption{AgentBench Jolt Detection Results}
\label{tab:agentbench_jolt}
\begin{tabular}{ll}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Analysis Period & %s to %s \\
Number of Months & %d \\
Initial Performance & %.2f \\
Final Performance & %.2f \\
Overall Growth & %.2f\%% \\
\midrule
Jolt Detected & %s \\
Jolt Date & %s \\
Jolt Score & %.2f \\
\midrule
Peak Score Component & %.2f \\
Pattern Score Component & %.2f \\
Duration Score Component & %.2f \\
\bottomrule
\end{tabular}
\end{table}
""" % (
    start_date, end_date, num_months,
    initial_performance, final_performance, growth_percentage,
    jolt_detected, jolt_date, jolt_score,
    peak_score, pattern_score, duration_score
)
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(latex_content)


if __name__ == "__main__":
    # Example usage
    import os
    
    # Load synthetic data
    data_path = "../data/synthetic_agentbench_data.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        
        # Preprocess data
        processed_df = preprocess_agentbench_data(df)
        
        # Detect jolt
        jolt_detected, jolt_info, derivs = detect_agentbench_jolt(processed_df)
        
        # Plot results
        fig = plot_agentbench_jolt(processed_df, derivs, jolt_info)
        
        # Save results
        os.makedirs("../results", exist_ok=True)
        os.makedirs("../results/tables", exist_ok=True)
        os.makedirs("../results/figures", exist_ok=True)
        
        fig.savefig("../results/figures/agentbench_jolt.png", dpi=300)
        create_latex_table(processed_df, jolt_info, "../results/tables/agentbench_jolt.tex")
        
        print(f"Jolt detected: {jolt_detected}")
        print(f"Jolt date: {jolt_info.get('jolt_date', 'Unknown')}")
        print(f"Jolt score: {jolt_info.get('score', 0):.2f}")
    else:
        print(f"Data file not found: {data_path}")
        print("Please generate synthetic data first using the generator module.")
