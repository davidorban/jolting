"""
Monte Carlo Jolt (mcjolt) package.

This package provides tools for generating synthetic capability curves and detecting jolts
(positive third derivatives) in technological progress data.
"""

from .generator import generate_series
from .estimator import estimate_derivatives, detect_jolt

__all__ = ["generate_series", "estimate_derivatives", "detect_jolt"]
