"""OpenAir - Air Bearing Analysis Tool.

This package provides tools for analyzing and visualizing air bearing performance.
"""
from .app import app
from .bearings import *
from .solvers import get_kappa, get_Qsc, get_beta, solve_bearing
from .plots import *

# Define what should be available when using 'from OpenAir import *'
__all__ = [
    'app',
    'circular_bearing',
    'annular_bearing',
    'InfiniteLinearBearing',
    'get_kappa',
    'get_Qsc',
    'get_beta',
    'solve_bearing',
    'plot_bearing_shape',
    'plot_key_results',
]