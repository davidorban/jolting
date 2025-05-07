"""
Synthetic AgentBench Data Generator

This module generates synthetic monthly AgentBench leaderboard data
for testing jolt detection algorithms when historical data is unavailable.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union


def generate_model_performance(
    start_date: datetime,
    num_months: int = 12,
    num_models: int = 20,
    base_performance: float = 40.0,
    growth_rate: float = 5.0,
    jolt_month: Optional[int] = 6,
    jolt_magnitude: float = 10.0,
    jolt_width: float = 1.0,
    noise_std: float = 2.0,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic AgentBench leaderboard data with monthly snapshots.
    
    Parameters
    ----------
    start_date : datetime
        Starting date for the synthetic data
    num_months : int, default=12
        Number of months to generate data for
    num_models : int, default=20
        Number of models to include in each snapshot
    base_performance : float, default=40.0
        Initial median performance score
    growth_rate : float, default=5.0
        Monthly growth rate in performance (percentage points per month)
    jolt_month : int or None, default=6
        Month in which the jolt occurs (0-indexed, None for no jolt)
    jolt_magnitude : float, default=10.0
        Magnitude of the jolt (additional percentage points)
    jolt_width : float, default=1.0
        Width of the jolt effect in months
    noise_std : float, default=2.0
        Standard deviation of random noise
    seed : int or None, default=None
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        Synthetic AgentBench leaderboard data with columns:
        - date: Snapshot date (YYYY-MM-DD)
        - model: Model name
        - median_score: Overall median score
        - os_score: Operating system task score
        - db_score: Database task score
        - kg_score: Knowledge graph task score
        - web_score: Web browsing task score
        - code_score: Coding task score
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate dates for each month
    dates = [start_date + timedelta(days=30*i) for i in range(num_months)]
    date_strs = [d.strftime('%Y-%m-%d') for d in dates]
    
    # Generate model names
    model_prefixes = ['GPT', 'Claude', 'Llama', 'Gemini', 'PaLM', 'Falcon', 'Bloom', 'Grok']
    model_suffixes = ['Pro', 'Ultra', 'Max', 'Plus', 'XL', 'Advanced', 'Elite']
    model_sizes = ['7B', '13B', '34B', '70B', '175B']
    
    models = []
    for i in range(num_models):
        prefix = np.random.choice(model_prefixes)
        suffix = np.random.choice(model_suffixes) if np.random.random() > 0.5 else ""
        size = np.random.choice(model_sizes)
        models.append(f"{prefix}-{size}{'-'+suffix if suffix else ''}")
    
    # Generate performance trajectory with optional jolt
    base_trajectory = np.zeros(num_months)
    for i in range(num_months):
        base_trajectory[i] = base_performance + growth_rate * i
    
    # Add jolt if specified
    if jolt_month is not None:
        for i in range(num_months):
            # Gaussian-shaped jolt centered at jolt_month
            jolt_effect = jolt_magnitude * np.exp(-0.5 * ((i - jolt_month) / jolt_width)**2)
            base_trajectory[i] += jolt_effect
    
    # Generate data for all models and months
    data = []
    for i, date_str in enumerate(date_strs):
        # Base performance for this month with some random variation
        month_performance = base_trajectory[i]
        
        for model in models:
            # Add model-specific variation
            model_factor = 0.8 + 0.4 * np.random.random()  # 0.8 to 1.2
            model_base = month_performance * model_factor
            
            # Add random noise
            median_score = model_base + noise_std * np.random.randn()
            
            # Generate task-specific scores with correlation to median
            task_noise = noise_std * 0.5  # Less noise for individual tasks
            os_score = median_score * (0.9 + 0.2 * np.random.random()) + task_noise * np.random.randn()
            db_score = median_score * (0.9 + 0.2 * np.random.random()) + task_noise * np.random.randn()
            kg_score = median_score * (0.9 + 0.2 * np.random.random()) + task_noise * np.random.randn()
            web_score = median_score * (0.9 + 0.2 * np.random.random()) + task_noise * np.random.randn()
            code_score = median_score * (0.9 + 0.2 * np.random.random()) + task_noise * np.random.randn()
            
            # Ensure scores are non-negative
            median_score = max(0, median_score)
            os_score = max(0, os_score)
            db_score = max(0, db_score)
            kg_score = max(0, kg_score)
            web_score = max(0, web_score)
            code_score = max(0, code_score)
            
            data.append({
                'date': date_str,
                'model': model,
                'median_score': round(median_score, 2),
                'os_score': round(os_score, 2),
                'db_score': round(db_score, 2),
                'kg_score': round(kg_score, 2),
                'web_score': round(web_score, 2),
                'code_score': round(code_score, 2)
            })
    
    return pd.DataFrame(data)


def save_synthetic_data(
    df: pd.DataFrame,
    output_path: str,
    include_metadata: bool = True
) -> None:
    """
    Save synthetic AgentBench data to CSV with optional metadata.
    
    Parameters
    ----------
    df : pd.DataFrame
        Synthetic data to save
    output_path : str
        Path to save the CSV file
    include_metadata : bool, default=True
        Whether to include metadata as comments at the top of the CSV
    """
    if include_metadata:
        with open(output_path, 'w') as f:
            f.write("# Synthetic AgentBench Leaderboard Data\n")
            f.write("# Generated for the Jolting Technologies Hypothesis project\n")
            f.write(f"# Generation date: {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write("# This is synthetic data for development purposes\n")
            f.write("# Real historical data should be used when available\n")
            f.write("#\n")
        
        df.to_csv(output_path, mode='a', index=False)
    else:
        df.to_csv(output_path, index=False)


def generate_synthetic_agentbench_data(
    start_date: str = "2024-05-01",
    num_months: int = 12,
    num_models: int = 20,
    base_performance: float = 40.0,
    growth_rate: float = 5.0,
    jolt_month: Optional[int] = 6,
    jolt_magnitude: float = 10.0,
    jolt_width: float = 1.0,
    noise_std: float = 2.0,
    seed: Optional[int] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate synthetic AgentBench data and optionally save to file.
    
    Parameters
    ----------
    start_date : str, default="2024-05-01"
        Starting date for the synthetic data (YYYY-MM-DD)
    num_months : int, default=12
        Number of months to generate data for
    num_models : int, default=20
        Number of models to include in each snapshot
    base_performance : float, default=40.0
        Initial median performance score
    growth_rate : float, default=5.0
        Monthly growth rate in performance (percentage points per month)
    jolt_month : int or None, default=6
        Month in which the jolt occurs (0-indexed, None for no jolt)
    jolt_magnitude : float, default=10.0
        Magnitude of the jolt (additional percentage points)
    jolt_width : float, default=1.0
        Width of the jolt effect in months
    noise_std : float, default=2.0
        Standard deviation of random noise
    seed : int or None, default=None
        Random seed for reproducibility
    output_path : str or None, default=None
        Path to save the CSV file (None to not save)
        
    Returns
    -------
    pd.DataFrame
        Synthetic AgentBench leaderboard data
    """
    # Convert string date to datetime
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    
    # Generate the data
    df = generate_model_performance(
        start_date=start_date,
        num_months=num_months,
        num_models=num_models,
        base_performance=base_performance,
        growth_rate=growth_rate,
        jolt_month=jolt_month if jolt_month >= 0 else None,
        jolt_magnitude=jolt_magnitude,
        jolt_width=jolt_width,
        noise_std=noise_std,
        seed=seed
    )
    
    # Save to file if output_path is provided
    if output_path:
        save_synthetic_data(df, output_path)
    
    return df


if __name__ == "__main__":
    # Example usage
    synthetic_data = generate_synthetic_agentbench_data(
        start_date="2024-05-01",
        num_months=12,
        jolt_month=6,
        seed=42,
        output_path="synthetic_agentbench_data.csv"
    )
    
    print(f"Generated synthetic data for {len(synthetic_data)} entries")
    print(f"Unique dates: {synthetic_data['date'].nunique()}")
    print(f"Unique models: {synthetic_data['model'].nunique()}")
