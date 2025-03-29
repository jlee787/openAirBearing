import numpy as np
from dataclasses import dataclass, field


@dataclass
class BaseBearing:
    """Base class for all bearing types."""
    pa: float = 101325
    pc: float = pa
    ps: float = 0.6e6 + pa

    rho: float = 1.293
    mu: float = 1.85e-5

    hp: float = 4.5e-3
    
    ha_min: float = 0.5e-6
    ha_max: float = 30e-6
    
    xa: float = 37 / 2 *1e-3
    xc: float = 0
    ya: float = 0
    nh: int = 25
    nx: int = 20
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

    xa: float = 40e-3 
    Qsc: float = 40  # L/min

    def __post_init__(self):
        super().__post_init__()
        self.psc = 0.6e6 + self.pa

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


def get_area(bearing):
    b = bearing
    match b.case:
        case "circular":
            A = np.pi * b.xa ** 2
        case "annular":
            A = np.pi * (b.xa**2 - b.xc**2)
        case "infinite":
            A = b.xa
        case "rectangular":
            A = b.xa * b.ya
        case _:
            raise ValueError(f"Unknown case: {b.case}")

    return A
    
def get_geom(bearing):
    """
    Calculate the geometry of the bearing.  
    """
    b = bearing
    if b.ny == 1:
        match b.error_type:
            case "none":
                geom = np.zeros(b.nx)
            case "linear":
                geom = b.error * (1 - b.x / b.xa) #- np.min(b.error * (1 - b.x**2 / b.xa**2))
            case "quadratic":
                geom = b.error * (1 - b.x**2 / b.xa**2) #- np.min(b.error * (1 - b.x**2 / b.xa**2))
            case _:
                raise ValueError(f"Unknown error type: {b.error_type}")

    else:
        if b.csys == "cartesian":

            x = b.x[:, None] 
            y = b.y[None, :]

            match b.error_type:
                case "none":
                    geom = np.zeros((b.nx, b.ny))
                case "linear":
                    geom = b.error * (np.abs(x) / b.xa + np.abs(y) / b.ya)
                case "quadratic":
                    geom = b.error * 2 * ((x / b.xa) ** 2 + (y / b.ya) ** 2)
                case _:
                    raise ValueError(f"Unknown error type: {b.error_type}")

        elif b.csys == "polar":
            r = b.x[:, None]
            theta = b.y[None, :]

            match b.error_type:
                case "none":
                    geom = np.zeros((b.nx, b.ny))
                case "linear":
                    geom = b.error * (1 - r / b.xa)
                case "quadratic":
                    geom = b.error * (1 - (r / b.xa) ** 2)
                case _:
                    raise ValueError(f"Unknown error type: {b.error_type}")
                
        else:
            raise ValueError(f"Unknown coordinate system: {b.csys}")

    return geom - np.min(geom)

def get_beta(bearing):
    """
    Calculate the porous feeding parameter.

    Returns:
        float: The porous feeding parameter, beta.
    """
    b = bearing
    beta = 6 * b.kappa * b.xa**2 / (b.hp * b.ha**3)
    return beta

def get_kappa(bearing):
    """
    Calculate the permeability.

    Returns:
        float: Permeability, kappa.
    """
    b = bearing

    if getattr( b, 'blocked', False):
        kappa = 2 * b.Qsc / 6e4 * b.mu * b.hp * b.pa / (b.block_A * (b.psc**2 - b.pa**2))
    else:
        kappa = 2 * b.Qsc / 6e4 * b.mu * b.hp * b.pa / (b.A * (b.psc**2 - b.pa**2))
    return round_to_sig_dig(kappa, 3)

def get_Qsc(bearing):
    """
    Calculate the permeability.

    Returns:
        float: Permeability, kappa.
    """
    b = bearing

    if b.blocked:
        Qsc = b.kappa * 6e4 * b.block_A * (b.psc**2 - b.pa**2) / (2 * b.mu * b.hp * b.pa)
    else:
        Qsc = b.kappa * 6e4 * b.A * (b.psc**2 - b.pa**2) / (2 * b.mu * b.hp * b.pa)
    return round_to_sig_dig(Qsc, 3) 

def round_to_sig_dig(number, digits):
    return np.round(number, -int(np.floor(np.log10(np.abs(number)))) + (digits - 1))