"""OpenAir - Air Bearing Analysis Tool.

This package provides tools for analyzing and visualizing air bearing performance.
"""
from .app import app
from .airbearings import (
    AxisymmetricBearing,
    get_kappa,
    get_Qsc,
    get_beta,
    solve_bearing
)

# Define what should be available when using 'from OpenAir import *'
__all__ = [
    'app',
    'AxisymmetricBearing',
    'get_kappa',
    'get_Qsc',
    'get_beta',
    'solve_bearing'
]