"""
Synthetic capability curve generator for Monte Carlo jolt detection.

This module provides functions to generate synthetic capability curves for three growth regimes:
exponential, logistic, and jolt (super-exponential with positive third derivative).
"""

import numpy as np
import torch
from typing import Dict, Tuple, Literal, Optional, Union

# Define growth regime types
GrowthRegime = Literal["exponential", "logistic", "jolt"]


def generate_series(
    kind: GrowthRegime,
    params: Dict,
    n_points: int = 100,
    noise_std: float = 0.01,
    device: Optional[str] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic capability curve for a specified growth regime.
    
    Parameters
    ----------
    kind : {"exponential", "logistic", "jolt"}
        The growth regime to generate.
    params : dict
        Parameters for the specific growth regime:
        - exponential: {"growth_rate": float}
        - logistic: {"growth_rate": float, "capacity": float, "midpoint": float}
        - jolt: {"base_rate": float, "jolt_magnitude": float, "jolt_time": float, "jolt_width": float}
    n_points : int, default=100
        Number of time points to generate.
    noise_std : float, default=0.01
        Standard deviation of Gaussian noise to add to the curve.
    device : str, optional
        Device to use for computation ("cpu" or "cuda"). If None, uses CPU.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    t : ndarray
        Time points (x-axis).
    y : ndarray
        Capability values (y-axis).
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
    
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Generate time points
    t = np.linspace(0, 1, n_points)
    
    # Generate base curve based on growth regime
    if kind == "exponential":
        y = _generate_exponential(t, params, device)
    elif kind == "logistic":
        y = _generate_logistic(t, params, device)
    elif kind == "jolt":
        y = _generate_jolt(t, params, device)
    else:
        raise ValueError(f"Unknown growth regime: {kind}")
    
    # Add noise
    if noise_std > 0:
        if device == "cuda" and torch.cuda.is_available():
            noise = torch.normal(0, noise_std, size=y.shape, device=device)
            y = y + noise
        else:
            noise = np.random.normal(0, noise_std, size=y.shape)
            y = y + noise
    
    # Convert to numpy if using torch
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    
    return t, y


def _generate_exponential(
    t: np.ndarray, 
    params: Dict, 
    device: str
) -> Union[np.ndarray, torch.Tensor]:
    """
    Generate an exponential growth curve.
    
    Parameters
    ----------
    t : ndarray
        Time points.
    params : dict
        Must contain "growth_rate" key.
    device : str
        Device to use for computation.
        
    Returns
    -------
    y : ndarray or torch.Tensor
        Exponential curve values.
    """
    growth_rate = params.get("growth_rate", 2.0)
    
    if device == "cuda" and torch.cuda.is_available():
        t_tensor = torch.tensor(t, dtype=torch.float32, device=device)
        return torch.exp(growth_rate * t_tensor)
    else:
        return np.exp(growth_rate * t)


def _generate_logistic(
    t: np.ndarray, 
    params: Dict, 
    device: str
) -> Union[np.ndarray, torch.Tensor]:
    """
    Generate a logistic (S-curve) growth curve.
    
    Parameters
    ----------
    t : ndarray
        Time points.
    params : dict
        Must contain "growth_rate", "capacity", and "midpoint" keys.
    device : str
        Device to use for computation.
        
    Returns
    -------
    y : ndarray or torch.Tensor
        Logistic curve values.
    """
    growth_rate = params.get("growth_rate", 10.0)
    capacity = params.get("capacity", 1.0)
    midpoint = params.get("midpoint", 0.5)
    
    if device == "cuda" and torch.cuda.is_available():
        t_tensor = torch.tensor(t, dtype=torch.float32, device=device)
        return capacity / (1 + torch.exp(-growth_rate * (t_tensor - midpoint)))
    else:
        return capacity / (1 + np.exp(-growth_rate * (t - midpoint)))


def _generate_jolt(
    t: np.ndarray, 
    params: Dict, 
    device: str
) -> Union[np.ndarray, torch.Tensor]:
    """
    Generate a jolt growth curve (super-exponential with positive third derivative).
    
    Parameters
    ----------
    t : ndarray
        Time points.
    params : dict
        Must contain "base_rate", "jolt_magnitude", "jolt_time", and "jolt_width" keys.
    device : str
        Device to use for computation.
        
    Returns
    -------
    y : ndarray or torch.Tensor
        Jolt curve values.
    """
    base_rate = params.get("base_rate", 2.0)
    jolt_magnitude = params.get("jolt_magnitude", 10.0)  # Dramatically increased to 10.0
    jolt_time = params.get("jolt_time", 0.5)  # Keep at 0.5
    jolt_width = params.get("jolt_width", 0.02)  # Even narrower for a more distinct signal
    
    if device == "cuda" and torch.cuda.is_available():
        t_tensor = torch.tensor(t, dtype=torch.float32, device=device)
        
        # Base exponential growth
        base_growth = torch.exp(base_rate * t_tensor)
        
        # Jolt component (accelerating growth after jolt_time)
        jolt_component = jolt_magnitude * torch.sigmoid((t_tensor - jolt_time) / jolt_width)
        accelerated_rate = base_rate + jolt_component
        
        # Final curve with jolt
        return torch.exp(accelerated_rate * t_tensor)
    else:
        # Base exponential growth
        base_growth = np.exp(base_rate * t)
        
        # Jolt component (accelerating growth after jolt_time)
        jolt_component = jolt_magnitude * (1 / (1 + np.exp(-(t - jolt_time) / jolt_width)))
        accelerated_rate = base_rate + jolt_component
        
        # Final curve with jolt
        return np.exp(accelerated_rate * t)


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Generate curves for each regime
    t, y_exp = generate_series("exponential", {"growth_rate": 2.0}, seed=42)
    t, y_log = generate_series("logistic", {"growth_rate": 10.0, "capacity": 1.0, "midpoint": 0.5}, seed=42)
    t, y_jolt = generate_series("jolt", {"base_rate": 2.0, "jolt_magnitude": 2.0, "jolt_time": 0.7, "jolt_width": 0.1}, seed=42)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(t, y_exp)
    plt.title("Exponential Growth")
    plt.ylabel("Capability")
    
    plt.subplot(3, 1, 2)
    plt.plot(t, y_log)
    plt.title("Logistic Growth")
    plt.ylabel("Capability")
    
    plt.subplot(3, 1, 3)
    plt.plot(t, y_jolt)
    plt.title("Jolt Growth")
    plt.xlabel("Time")
    plt.ylabel("Capability")
    
    plt.tight_layout()
    plt.savefig("growth_regimes.png")
    plt.close()
    
    print("Generated example curves for three growth regimes.")
