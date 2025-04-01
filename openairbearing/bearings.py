import numpy as np
from dataclasses import dataclass, field

from openairbearing.utils import get_area, get_geom, get_kappa, get_beta

@dataclass
class BaseBearing:
    """Base class for all bearing types."""
    pa: float = 101325
    pc: float = pa
    ps: float = 0.6e6 + pa

    rho: float = 1.293
    mu: float = 1.85e-5

    hp: float = 4.5e-3
    
    ha_min: float = 1e-6
    ha_max: float = 20e-6
    
    xa: float = 37 / 2 *1e-3
    xc: float = 0
    ya: float = 0
    nh: int = 20
    nx: int = 30
    ny: int = 1

    Psi: float = 0

    error_type: str = "none"
    error: float = 0e-6

    blocked: bool = False
    block_x: float = 25.2e-3 / 2
    block_w: float = 1e-3

    Qsc: float = 3  # L/min
    psc: float = 0.6e6 + pa

    x: np.ndarray = field(init=False)
    dx: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    dy: np.ndarray = field(init=False)

    ha: np.ndarray = field(init=False)
    A: float = field(init=False)
    kappa: float = field(init=False)
    beta: float = field(init=False)
    geom: np.ndarray = field(init=False)

    case: str = "base"
    type: str = "bearing"
    csys: str = "cartesian"

    def __post_init__(self):
        self.ha = np.linspace(self.ha_min, self.ha_max, self.nh).T
        self.x = np.linspace(self.xc, self.xa, self.nx)
        self.y = np.linspace(0, self.ya, self.ny)
        self.dx = np.gradient(self.x)
        self.dy = 1 if self.ny == 1 else np.gradient(self.y)
        self.A = get_area(self)
        self.geom = get_geom(self)
        self.kappa = get_kappa(self)
        self.beta = get_beta(self)

@dataclass
class CircularBearing(BaseBearing):
    """Base class for circular thrust bearing"""
    case: str = "circular"
    type: str = "bearing"
    csys: str = "polar"

    xc: float = 1e-6
    xa: float = 37e-3 / 2
    Qsc: float = 2.8  # L/min

    def __post_init__(self):
        super().__post_init__()
        self.psc = 0.6e6 + self.pa
       
@dataclass
class AnnularBearing(BaseBearing):
    """Base class for annular bearing"""
    
    case: str = "annular"
    type: str = "seal"
    csys: str = "polar"
   
    xa: float = 58e-3 / 2
    xc: float = 25e-3 / 2
    
    Qsc: float = 3  # L/min
   
    def __post_init__(self):
        super().__post_init__()
        self.psc = 0.6e6 + self.pa

@dataclass
class InfiniteLinearBearing(BaseBearing):
    """Base class for Infinitely long linear bearing bearing"""
   
    case: str = "infinite"
    type: str = "seal"
    csys: str = "cartesian"

    ps: float = 0.41e6
    xa: float = 40e-3 

    Qsc: float = 37  # L/min

    def __post_init__(self):
        super().__post_init__()
        self.psc = 0.41e6 + self.pa

@dataclass
class RectangularBearing(BaseBearing):
    """Base class for rectangular thrust bearing"""
   
    case: str = "rectangular"
    type: str = "bearing"
    csys: str = "cartesian"
   
    xa: float = 80e-3
    ya: float = 40e-3
    nx: int = 40
    ny: int = 20

    ps: float = 0.41e6

    Qsc: float = 2.94  # L/min

    def __post_init__(self):
        super().__post_init__()
        self.psc = 0.41e6 + self.pa
        self.x = np.linspace(-self.xa / 2, self.xa / 2, self.nx)
        self.y = np.linspace(-self.ya / 2, self.ya / 2, self.ny)
        self.geom = get_geom(self) # calculate after x y

