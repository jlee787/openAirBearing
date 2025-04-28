"""OpenAir - Air Bearing Analysis Tool.

This package provides tools for analyzing and visualizing air bearing performance.
"""

# Version information
__version__ = "0.1.0"

# Import constants from config (where they're actually defined)
from .config import DEMO_MODE

# Import specific bearing classes
from .bearings import (
    RectangularBearing,
    CircularBearing,
    AnnularBearing,
    InfiniteLinearBearing,
    JournalBearing,
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
from .solvers import solve_bearing, get_pressure_numeric

# Import visualization functions
from .plots import (
    plot_bearing_shape,
    plot_key_results,
    plot_load_capacity,
    plot_stiffness,
    plot_pressure_distribution,
    plot_supply_flow_rate,
    plot_chamber_flow_rate,
    plot_ambient_flow_rate,
)

# Import app
from .app.app import app

# Define what should be available when using 'from openairbearing import *'
__all__ = [
    # App
    "app",
    # Bearing types
    "RectangularBearing",
    "CircularBearing",
    "AnnularBearing",
    "InfiniteLinearBearing",
    "JournalBearing",
    # Bearing parameters
    "get_kappa",
    "get_Qsc",
    "get_beta",
    "get_geom",
    "get_area",
    # Result type
    "Result",
    # Solver
    "solve_bearing",
    "get_pressure_numeric",
    # Configuration
    "DEMO_MODE",
    # Visualization
    "plot_bearing_shape",
    "plot_key_results",
    "plot_load_capacity",
    "plot_stiffness",
    "plot_pressure_distribution",
    "plot_supply_flow_rate",
    "plot_chamber_flow_rate",
    "plot_ambient_flow_rate",
]
