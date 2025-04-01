"""OpenAir - Air Bearing Analysis Tool.

This package provides tools for analyzing and visualizing air bearing performance.
"""

# Version information
__version__ = '0.1.0'

# Import constants from config (where they're actually defined)
from .config import ANALYTIC, NUMERIC, DEMO_MODE

# Import specific bearing classes
from .bearings import (
    RectangularBearing, 
    CircularBearing,
    AnnularBearing,
    InfiniteLinearBearing,
)

# Import utility functions from where they're defined
from .utils import (
    get_kappa,
    get_Qsc,
    get_beta,
    Result,
    get_geom,
    get_area,
)

# Import solver function
from .solvers import solve_bearing

# Import visualization functions
from .plots import plot_bearing_shape, plot_key_results

# Import app
from .app.app import app

# Define what should be available when using 'from openairbearing import *'
__all__ = [
    # App
    'app',
    
    # Bearing types
    'RectangularBearing',
    'CircularBearing',
    'AnnularBearing',
    'InfiniteLinearBearing',
    
    # Bearing parameters
    'get_kappa',
    'get_Qsc',
    'get_beta',
    'get_geom',
    'get_area',
    
    # Result type
    'Result',
    
    # Solver
    'solve_bearing',
    'ANALYTIC',
    'NUMERIC',
    
    # Configuration
    'DEMO_MODE',
    
    # Visualization
    'plot_bearing_shape',
    'plot_key_results',
]