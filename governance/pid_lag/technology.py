"""
Technology Progress Models

This module implements models for technological progress with jolts
to be used with the PID-Lag governance controller.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass


@dataclass
class TechProgressParams:
    """Parameters for technological progress model."""
    base_growth_rate: float = 0.05  # Base exponential growth rate
    jolt_time: Optional[float] = None  # Time of jolt (None for no jolt)
    jolt_magnitude: float = 0.2  # Magnitude of jolt
    jolt_duration: float = 5.0  # Duration of jolt effect
    jolt_decay: float = 0.5  # Decay rate of jolt effect
    noise_std: float = 0.01  # Standard deviation of noise
    initial_value: float = 1.0  # Initial technology level
    
    def __post_init__(self):
        """Validate parameters."""
        if self.base_growth_rate < 0:
            raise ValueError("Base growth rate must be non-negative")
        if self.jolt_magnitude < 0:
            raise ValueError("Jolt magnitude must be non-negative")
        if self.jolt_duration <= 0:
            raise ValueError("Jolt duration must be positive")
        if self.jolt_decay < 0 or self.jolt_decay > 1:
            raise ValueError("Jolt decay must be between 0 and 1")
        if self.noise_std < 0:
            raise ValueError("Noise standard deviation must be non-negative")
        if self.initial_value <= 0:
            raise ValueError("Initial value must be positive")


def exponential_growth(
    t: float,
    params: TechProgressParams
) -> float:
    """
    Calculate exponential growth at time t.
    
    Parameters
    ----------
    t : float
        Time
    params : TechProgressParams
        Parameters for technological progress
        
    Returns
    -------
    float
        Technology level at time t
    """
    return params.initial_value * np.exp(params.base_growth_rate * t)


def jolt_effect(
    t: float,
    params: TechProgressParams
) -> float:
    """
    Calculate jolt effect at time t.
    
    Parameters
    ----------
    t : float
        Time
    params : TechProgressParams
        Parameters for technological progress
        
    Returns
    -------
    float
        Jolt effect at time t
    """
    if params.jolt_time is None or t < params.jolt_time:
        return 0.0
    
    time_since_jolt = t - params.jolt_time
    if time_since_jolt > params.jolt_duration:
        decay_time = time_since_jolt - params.jolt_duration
        return params.jolt_magnitude * np.exp(-params.jolt_decay * decay_time)
    else:
        # Ramp up during jolt duration
        ramp_fraction = time_since_jolt / params.jolt_duration
        return params.jolt_magnitude * ramp_fraction


def tech_progress_with_jolt(
    t: float,
    params: TechProgressParams,
    governance_effect: float = 0.0,
    seed: Optional[int] = None
) -> float:
    """
    Calculate technology progress with jolt and governance effect.
    
    Parameters
    ----------
    t : float
        Time
    params : TechProgressParams
        Parameters for technological progress
    governance_effect : float, default=0.0
        Effect of governance on technology progress
    seed : int, optional
        Random seed for noise
        
    Returns
    -------
    float
        Technology level at time t
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Base exponential growth
    base_level = exponential_growth(t, params)
    
    # Add jolt effect
    jolt = jolt_effect(t, params)
    
    # Apply governance effect (reduces growth)
    gov_factor = max(0, 1 - governance_effect)
    
    # Add noise
    noise = np.random.normal(0, params.noise_std)
    
    # Calculate final technology level
    tech_level = base_level * (1 + jolt) * gov_factor + noise
    
    return max(0, tech_level)  # Ensure non-negative


def calculate_tech_risk(
    tech_level: float,
    risk_threshold: float = 10.0,
    risk_steepness: float = 2.0
) -> float:
    """
    Calculate risk associated with technology level.
    
    Parameters
    ----------
    tech_level : float
        Current technology level
    risk_threshold : float, default=10.0
        Technology level at which risk is 0.5
    risk_steepness : float, default=2.0
        Steepness of the risk curve
        
    Returns
    -------
    float
        Risk level between 0 and 1
    """
    # Logistic function for risk
    risk = 1 / (1 + np.exp(-risk_steepness * (tech_level - risk_threshold)))
    return risk


def simulate_tech_progress(
    params: TechProgressParams,
    governance_func: Optional[Callable[[float, float], float]] = None,
    time_range: Tuple[float, float] = (0, 100),
    time_step: float = 1.0,
    risk_threshold: float = 10.0,
    risk_steepness: float = 2.0,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Simulate technological progress with optional governance.
    
    Parameters
    ----------
    params : TechProgressParams
        Parameters for technological progress
    governance_func : Callable[[float, float], float], optional
        Function that takes (time, tech_level) and returns governance effect
    time_range : Tuple[float, float], default=(0, 100)
        Start and end times for simulation
    time_step : float, default=1.0
        Time step for simulation
    risk_threshold : float, default=10.0
        Technology level at which risk is 0.5
    risk_steepness : float, default=2.0
        Steepness of the risk curve
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing simulation results:
        - 'time': Time points
        - 'tech_level': Technology levels
        - 'jolt_effect': Jolt effects
        - 'governance_effect': Governance effects
        - 'risk': Risk levels
    """
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize simulation
    start_time, end_time = time_range
    time_points = np.arange(start_time, end_time + time_step, time_step)
    n_points = len(time_points)
    
    # Initialize result arrays
    tech_level = np.zeros(n_points)
    jolt_effects = np.zeros(n_points)
    governance_effects = np.zeros(n_points)
    risk_levels = np.zeros(n_points)
    
    # Run simulation
    for i, t in enumerate(time_points):
        # Calculate jolt effect
        jolt_effects[i] = jolt_effect(t, params)
        
        # Calculate governance effect if provided
        gov_effect = 0.0
        if governance_func is not None and i > 0:
            gov_effect = governance_func(t, tech_level[i-1])
        governance_effects[i] = gov_effect
        
        # Calculate technology level
        tech_level[i] = tech_progress_with_jolt(
            t, params, gov_effect, seed=None  # Already set seed
        )
        
        # Calculate risk
        risk_levels[i] = calculate_tech_risk(
            tech_level[i], risk_threshold, risk_steepness
        )
    
    # Return results
    return {
        'time': time_points,
        'tech_level': tech_level,
        'jolt_effect': jolt_effects,
        'governance_effect': governance_effects,
        'risk': risk_levels
    }
