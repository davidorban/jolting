"""
Jolt detection and derivative estimation module.

This module provides functions to smooth time series data, estimate derivatives,
and detect jolts (positive third derivatives) in capability curves.
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
import statsmodels.api as sm
from typing import Dict, Tuple, Optional, List, Union


def estimate_derivatives(
    t: np.ndarray,
    y: np.ndarray,
    window_length: Optional[int] = None,
    polyorder: int = 3,
    method: str = "savgol",
    smooth_factor: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Estimate the first, second, and third derivatives of a time series.
    """
    n_points = len(t)
    if window_length is None:
        window_length = min(n_points // 5 * 2 + 1, n_points - 2 if n_points > polyorder + 2 else n_points -1) # ensure n_points > window_length
        window_length = max(window_length, polyorder + 1 if polyorder + 1 < n_points else (n_points -1 if n_points % 2 == 0 else n_points) ) # ensure window is large enough and odd
        if window_length % 2 == 0:
            window_length = window_length + 1 if window_length + 1 < n_points else (window_length -1 if window_length >1 else 1)
        window_length = max(1, window_length) # Ensure window_length is at least 1
        if n_points <= window_length : # handle very small n_points
            window_length = n_points if n_points % 2 != 0 else (n_points -1 if n_points > 1 else 1)
            if window_length <=0 : window_length =1
            if polyorder >= window_length: polyorder = window_length -1
            if polyorder < 0: polyorder = 0

    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-9) # Avoid division by zero if t.max() == t.min()
    
    if method == "savgol":
        if n_points <= window_length or polyorder >= window_length:
             # Fallback for very short series or problematic parameters
            y_smooth = y
            first_deriv = np.zeros_like(y)
            second_deriv = np.zeros_like(y)
            third_deriv = np.zeros_like(y)
        else:
            y_smooth = savgol_filter(y, window_length, polyorder)
            dt = t_norm[1] - t_norm[0] if len(t_norm) > 1 else 1.0
            if dt == 0: dt = 1.0 # Avoid division by zero
            first_deriv = np.gradient(y_smooth, dt)
            second_deriv = np.gradient(first_deriv, dt)
            third_deriv = np.gradient(second_deriv, dt)
        
    elif method == "spline":
        if smooth_factor is None:
            smooth_factor = _find_optimal_smooth_factor(t_norm, y)
        try:
            spline = UnivariateSpline(t_norm, y, s=smooth_factor)
            y_smooth = spline(t_norm)
            first_deriv = spline.derivative(1)(t_norm)
            second_deriv = spline.derivative(2)(t_norm)
            third_deriv = spline.derivative(3)(t_norm)
        except Exception:
            # Fallback if spline fails
            y_smooth = y
            first_deriv = np.zeros_like(y)
            second_deriv = np.zeros_like(y)
            third_deriv = np.zeros_like(y)
            
    elif method == "gam":
        try:
            X = sm.add_constant(t_norm)
            gam_model = sm.GAM(y, X).fit()
            y_smooth = gam_model.predict(X)
            dt = t_norm[1] - t_norm[0] if len(t_norm) > 1 else 1.0
            if dt == 0: dt = 1.0
            first_deriv = np.gradient(y_smooth, dt)
            second_deriv = np.gradient(first_deriv, dt)
            third_deriv = np.gradient(second_deriv, dt)
        except Exception:
            y_smooth = y
            first_deriv = np.zeros_like(y)
            second_deriv = np.zeros_like(y)
            third_deriv = np.zeros_like(y)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {
        "y_smooth": y_smooth,
        "first_deriv": first_deriv,
        "second_deriv": second_deriv,
        "third_deriv": third_deriv
    }

def detect_jolt(
    third_deriv: np.ndarray,
    second_deriv: Optional[np.ndarray] = None,
    first_deriv: Optional[np.ndarray] = None,
    threshold: Optional[float] = None, # For standard method adaptive threshold base
    min_duration: int = 2,
    pattern_check: bool = True, # For standard method
    simple_detection: bool = True, 
    peak_ratio_threshold: float = 7.5 # For simple method
) -> Tuple[bool, Dict]:
    """
    Detects jolts. 
    If simple_detection is True, uses peak_ratio_threshold.
    Otherwise, uses an adaptive threshold, min_duration, and optionally pattern_check.
    """
    if third_deriv is None or len(third_deriv) == 0:
        return False, {"max_jolt": 0.0, "jolt_index": -1, "jolt_duration": 0, "positive_regions": [], "pattern_score": 0.0, "peak_ratio": 0.0}

    max_third_deriv_val = np.max(third_deriv) if len(third_deriv) > 0 else 0.0
    jolt_idx_val = np.argmax(third_deriv) if len(third_deriv) > 0 else -1

    if simple_detection:
        median_third_deriv_abs = np.median(np.abs(third_deriv)) if len(third_deriv) > 0 else 0.0
        peak_ratio = max_third_deriv_val / (median_third_deriv_abs + 1e-10)
        
        if peak_ratio > peak_ratio_threshold:
            return True, {
                "max_jolt": max_third_deriv_val,
                "jolt_index": jolt_idx_val,
                "jolt_duration": min_duration, # Simplified assumption for simple mode
                "positive_regions": [(max(0, jolt_idx_val - min_duration // 2), min(len(third_deriv), jolt_idx_val + min_duration // 2 + 1))],
                "pattern_score": peak_ratio / (peak_ratio_threshold * 1.5), # Normalized score attempt
                "peak_ratio": peak_ratio
            }
        else:
            return False, {"max_jolt": max_third_deriv_val, "jolt_index": jolt_idx_val, "jolt_duration": 0, "positive_regions": [], "pattern_score": peak_ratio / (peak_ratio_threshold * 1.5), "peak_ratio": peak_ratio}

    # Standard detection approach (if not simple_detection or simple_detection conditions not met)
    current_threshold = threshold
    if current_threshold is None:
        mean_third_deriv = np.mean(third_deriv)
        std_third_deriv = np.std(third_deriv)
        data_range = np.max(third_deriv) - np.min(third_deriv) if len(third_deriv) > 0 else 0.0
        threshold1 = mean_third_deriv + 1.5 * std_third_deriv
        threshold2 = 0.2 * data_range
        current_threshold = max(threshold1, threshold2, 0.1) 
    
    positive_mask = third_deriv > current_threshold
    positive_regions = _find_consecutive_regions(positive_mask)
    valid_regions = [region for region in positive_regions if region[1] - region[0] >= min_duration]
    
    if not valid_regions:
        return False, {"max_jolt": max_third_deriv_val, "jolt_index": jolt_idx_val, "jolt_duration": 0, "positive_regions": [], "pattern_score": 0.0, "peak_ratio": 0.0}
    
    jolt_duration_val = sum(end - start for start, end in valid_regions)
    pattern_score_val = 0.0
    if pattern_check and second_deriv is not None and first_deriv is not None and len(second_deriv) == len(third_deriv) and len(first_deriv) == len(third_deriv):
        pattern_score_val = _check_jolt_pattern(third_deriv, second_deriv, first_deriv, jolt_idx_val)
        if pattern_score_val < 0.5: # Default threshold for pattern matching score
            return False, {"max_jolt": max_third_deriv_val, "jolt_index": jolt_idx_val, "jolt_duration": jolt_duration_val, "positive_regions": valid_regions, "pattern_score": pattern_score_val, "peak_ratio": 0.0}
    
    return True, {"max_jolt": max_third_deriv_val, "jolt_index": jolt_idx_val, "jolt_duration": jolt_duration_val, "positive_regions": valid_regions, "pattern_score": pattern_score_val, "peak_ratio": 0.0}

def _check_jolt_pattern(
    third_deriv: np.ndarray,
    second_deriv: np.ndarray,
    first_deriv: np.ndarray,
    jolt_index: int,
    window_size: int = 10
) -> float:
    """
    Checks derivative pattern around a potential jolt point.
    """
    n_points = len(third_deriv)
    if n_points == 0 or jolt_index < 0 or jolt_index >= n_points:
        return 0.0
    
    # Ensure jolt_index is valid for windowing, adjust window_size if series is too short
    actual_window_size = min(window_size, n_points // 3, jolt_index, n_points - 1 - jolt_index)
    if actual_window_size <=0 : actual_window_size = 1 # ensure window size is at least 1 for small series

    start_idx = max(0, jolt_index - actual_window_size)
    end_idx = min(n_points, jolt_index + actual_window_size + 1)
    
    if start_idx >= end_idx: return 0.0 # Invalid window

    third_window = third_deriv[start_idx:end_idx]
    second_window = second_deriv[start_idx:end_idx]
    first_window = first_deriv[start_idx:end_idx]

    if len(third_window) == 0: return 0.0
    
    third_deriv_max_window = np.max(third_window) if len(third_window) > 0 else 0.0
    if third_deriv_max_window <= 1e-9: # Effectively zero or negative
        return 0.0
    
    third_deriv_median_abs_window = np.median(np.abs(third_window)) if len(third_window) > 0 else 0.0
    peak_prominence = third_deriv_max_window / (third_deriv_median_abs_window + 1e-10)
    peak_score = min(1.0, peak_prominence / 10.0)  # Normalize, 10.0 is an empirical factor
    
    second_score = 0.0
    if len(second_window) > 2 :
        mid_idx_window = jolt_index - start_idx # This is the index of jolt_index within the window
        if mid_idx_window > 0 and mid_idx_window < len(second_window) -1:
            # Check if second_deriv is increasing before and decreasing after the jolt_index within the window
            # Need at least 2 points to calculate diff
            before_segment = second_window[:mid_idx_window]
            after_segment = second_window[mid_idx_window:]
            before_increasing = np.mean(np.diff(before_segment) > 0) if len(before_segment) > 1 else 0.0
            after_decreasing = np.mean(np.diff(after_segment) < 0) if len(after_segment) > 1 else 0.0
            second_score = 0.5 * (before_increasing + after_decreasing)
        elif len(second_window) > 1: # if mid_idx is at boundary, take overall trend
            second_score = 0.5 * (np.mean(np.diff(second_window)>0) + np.mean(np.diff(second_window)<0)) # less specific

    first_deriv_positive_score = np.mean(first_window > 0) if len(first_window) > 0 else 0.0
    
    pattern_score = 0.6 * peak_score + 0.3 * second_score + 0.1 * first_deriv_positive_score
    return pattern_score


def _find_consecutive_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Finds consecutive regions where mask is True."""
    if len(mask) == 0: return []
    regions = []
    in_region = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_region:
            in_region = True
            start = i
        elif not val and in_region:
            in_region = False
            regions.append((start, i))
    if in_region:
        regions.append((start, len(mask)))
    return regions

def _find_optimal_smooth_factor(t: np.ndarray, y: np.ndarray) -> float:
    """Find optimal smoothing factor using AIC (simplified)."""
    if len(t) < 5: return 0.1 # Default for very short series
    s_values = np.logspace(-3, 1, 10)
    aic_values = []
    for s_val in s_values:
        try:
            spline = UnivariateSpline(t, y, s=s_val)
            y_pred = spline(t)
            residuals = y - y_pred
            rss = np.sum(residuals**2)
            if rss < 1e-9: rss = 1e-9 # Avoid log(0)
            n = len(y)
            # Simplified AIC: n * log(RSS/n) + 2 * (num_knots)
            # num_knots can be approximated or use a fixed penalty for complexity
            aic = n * np.log(rss / n) + 2 * (len(spline.get_knots()) + 2) 
            aic_values.append(aic)
        except Exception:
            aic_values.append(np.inf)
    if not aic_values or all(v == np.inf for v in aic_values):
         return 0.1 # Default if all spline fits fail
    return s_values[np.argmin(aic_values)]

def detect_jolt_hybrid(
    third_deriv: np.ndarray,
    second_deriv: np.ndarray,
    first_deriv: np.ndarray,
    min_duration: int = 2,
    weights: Optional[Dict[str, float]] = None,
    final_threshold: float = 0.6,
    peak_norm_factor: float = 15.0,
    duration_norm_multiplier: float = 2.0,
    adaptive_threshold_base: Optional[float] = None # For duration region finding
) -> Tuple[bool, Dict]:
    """
    Hybrid jolt detection using a weighted score of peak ratio, pattern, and duration.
    """
    if third_deriv is None or len(third_deriv) == 0:
        return False, {"score": 0.0, "components": {}}

    _weights = weights if weights else {"peak": 0.4, "pattern": 0.4, "duration": 0.2}

    # 1. Overall Peak Score Component
    max_td = np.max(third_deriv) if len(third_deriv) > 0 else 0.0
    median_abs_td = np.median(np.abs(third_deriv)) if len(third_deriv) > 0 else 0.0
    overall_peak_ratio = max_td / (median_abs_td + 1e-10)
    peak_score_component = min(1.0, overall_peak_ratio / peak_norm_factor)

    # 2. Pattern Score Component
    jolt_idx = np.argmax(third_deriv) if len(third_deriv) > 0 else -1
    if jolt_idx == -1 or second_deriv is None or first_deriv is None or not (len(second_deriv) == len(third_deriv) and len(first_deriv) == len(third_deriv)):
        pattern_score_component = 0.0
    else:
        pattern_score_component = _check_jolt_pattern(third_deriv, second_deriv, first_deriv, jolt_idx)

    # 3. Duration Score Component
    current_adaptive_thresh = adaptive_threshold_base
    if current_adaptive_thresh is None: # Use default adaptive logic if base not provided
        mean_td = np.mean(third_deriv)
        std_td = np.std(third_deriv)
        range_td = np.max(third_deriv) - np.min(third_deriv) if len(third_deriv) > 0 else 0.0
        thresh1 = mean_td + 1.5 * std_td
        thresh2 = 0.2 * range_td
        current_adaptive_thresh = max(thresh1, thresh2, 0.1)

    positive_mask = third_deriv > current_adaptive_thresh
    positive_regions = _find_consecutive_regions(positive_mask)
    valid_regions = [region for region in positive_regions if region[1] - region[0] >= min_duration]
    
    max_duration_found = 0
    if valid_regions:
        max_duration_found = max(end - start for start, end in valid_regions)
    
    duration_score_component = min(1.0, max_duration_found / (duration_norm_multiplier * min_duration))

    # Combine scores
    final_score = (
        _weights["peak"] * peak_score_component +
        _weights["pattern"] * pattern_score_component +
        _weights["duration"] * duration_score_component
    )
    
    jolt_detected = final_score > final_threshold
    
    info = {
        "score": final_score,
        "components": {
            "peak_score": peak_score_component,
            "pattern_score": pattern_score_component,
            "duration_score": duration_score_component,
            "overall_peak_ratio": overall_peak_ratio,
            "max_duration_found": max_duration_found
        },
        "max_jolt": max_td,
        "jolt_index": jolt_idx,
        "jolt_duration": max_duration_found, # representative duration
        "positive_regions": valid_regions # regions from duration check
    }
    return jolt_detected, info


