from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import i0, k0
from plotly.io import show


from plots import plot_key_results 

ANALYTIC = True
NUMERIC = False

@dataclass
class CircularBearing:
    """Base class for circular thrust bearing"""
    pa: float = 101325
    pc: float = 101325
    ps: float = 0.6e6 + pa
    rho: float = 1.293
    mu: float = 1.85e-5
    hp: float = 4.5e-3
    ha_min: float = 0.5e-6
    ha_max: float = 30e-6
    n_ha: float = 25
    ra: float = 37e-3 / 2
    nr: int = 20
    Psi: float = 0

    c: float = 0e-6

    blocked: bool = False
    block_r: float = 25.2e-3 / 2
    block_w: float = 1e-3

    Qsc: float = 2.8  # L/min
    psc: float = 0.6e6 + pa

    r: np.ndarray = field(init=False)
    dr: np.ndarray = field(init=False)
    ha: np.ndarray = field(init=False)
    A: float = field(init=False)
    kappa: float = field(init=False)
    beta: float = field(init=False)
    geom: np.ndarray = field(init=False)
    block_in: np.ndarray = field(init=False)

    case: str = "circular"
    type: str = "bearing"
    csys: str = "polar"

    def __post_init__(self):
        self.r = np.linspace(1e-4, self.ra, self.nr)
        self.ha = np.linspace(self.ha_min, self.ha_max, self.n_ha).T
        self.dr = np.gradient(self.r)
        self.A = np.pi * self.ra**2
        self.geom = self.c * (1 - self.r**2 / self.ra**2) - np.min(self.c * (1 - self.r**2 / self.ra**2))
        self.block_in = np.logical_and(self.r > self.block_r, self.r < (self.block_r + self.block_w))
        self.block_A = np.pi * (self.ra**2 - (self.block_r + self.block_w)**2 + self.block_r**2)
        self.kappa = get_kappa(self)
        self.beta = get_beta(self)

@dataclass
class InfiniteLinearBearing:
    """Base class for Infinitely long linear bearing bearing"""
    pa: float = 101325
    pc: float = 101325
    ps: float = 0.6e6 + pa
    rho: float = 1.293
    mu: float = 1.85e-5
    hp: float = 4.5e-3
    ha_min: float = 0.5e-6
    ha_max: float = 30e-6
    n_ha: float = 25
    ra: float = 40e-3 
    nr: int = 30
    Psi: float = 0

    c: float = 0e-6

    blocked: bool = False
    block_r: float = 25.2e-3 / 2
    block_w: float = 1e-3

    Qsc: float = 100  # L/min
    psc: float = 0.6e6 + pa

    r: np.ndarray = field(init=False)
    dr: np.ndarray = field(init=False)
    ha: np.ndarray = field(init=False)
    A: float = field(init=False)
    kappa: float = field(init=False)
    beta: float = field(init=False)
    geom: np.ndarray = field(init=False)

    case: str = "infinite"
    type: str = "seal"
    csys: str = "cartesian"

    def __post_init__(self):
        self.r = np.linspace(0, self.ra, self.nr)
        self.ha = np.linspace(self.ha_min, self.ha_max, self.n_ha).T
        self.dr = np.gradient(self.r)
        self.A = 1*self.ra
        self.geom = self.c * (1 - self.r**2 / self.ra**2) - np.min(self.c * (1 - self.r**2 / self.ra**2))
        self.kappa = get_kappa(self)
        self.beta = get_beta(self)

@dataclass
class AnnularBearing:
    """Base class for annular bearing"""
    pa: float = 101325
    pc: float = 101325
    ps: float = 0.6e6 + pa
    rho: float = 1.293
    mu: float = 1.85e-5
    hp: float = 4.5e-3
    ha_min: float = 0.5e-6
    ha_max: float = 30e-6
    n_ha: float = 25
    ra: float = 58e-3 / 2
    rc: float = 25e-3 / 2
    nr: int = 20
    Psi: float = 0

    c: float = 0e-6

    blocked: bool = False

    Qsc: float = 3  # L/min
    psc: float = 0.6e6 + pa

    r: np.ndarray = field(init=False)
    dr: np.ndarray = field(init=False)
    ha: np.ndarray = field(init=False)
    A: float = field(init=False)
    kappa: float = field(init=False)
    beta: float = field(init=False)
    geom: np.ndarray = field(init=False)
    
    case: str = "annular"
    type: str = "seal"
    csys: str = "polar"

    def __post_init__(self):
        self.r = np.linspace(self.rc, self.ra, self.nr)
        self.ha = np.linspace(self.ha_min, self.ha_max, self.n_ha).T
        self.dr = np.gradient(self.r)
        self.A = np.pi * (self.ra**2 - self.rc**2)
        self.geom = self.c * (1 - self.r**2 / self.ra**2) - np.min(self.c * (1 - self.r**2 / self.ra**2))
        self.kappa = get_kappa(self)
        self.beta = get_beta(self)

@dataclass
class Result:
    """Class to hold bearing calculation results"""
    name: str
    p: np.ndarray
    w: np.ndarray
    k: np.ndarray
    qs: np.ndarray
    qa: np.ndarray
    qc: np.ndarray

def solve_bearing(bearing, soltype: bool) -> Result:
    if soltype == ANALYTIC:
        name = "Analytic"
        match bearing.case:
            case "circular":
                p = get_pressure_analytic_circular(bearing)
            case "annular":
                p = get_pressure_analytic_annular(bearing)
            case "infinite":
                p = get_pressure_analytic_infinite(bearing)
    elif soltype == NUMERIC:
        name = "Numeric"
        p = get_pressure_numeric(bearing)
    
    w = get_load_capacity(bearing, p)
    k = get_stiffness(bearing, w)
    qs, qa, qc = get_volumetric_flow(bearing=bearing, p=p, soltype=soltype)

    return Result(name=name, p=p, w=w, k=k, qs=qs, qa=qa, qc=qc)

def get_kappa(bearing):
    """
    Calculate the permeability.

    Returns:
        float: Permeability, kappa.
    """
    b = bearing

    if getattr(bearing, 'blocked', False):
        kappa = 2 * b.Qsc / 6e4 * b.mu * b.hp * b.pa / (b.block_A * (b.psc**2 - b.pa**2))
    else:
        kappa = 2 * b.Qsc / 6e4 * b.mu * b.hp * b.pa / (b.A * (b.psc**2 - b.pa**2))

    sig_digits = 3
    kappa = np.round(kappa, -int(np.floor(np.log10(abs(kappa)))) + (sig_digits - 1))
    return kappa

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

    sig_digits = 3
    Qsc = np.round(Qsc, -int(np.floor(np.log10(abs(Qsc)))) + (sig_digits - 1))
    return Qsc

def get_beta(bearing):
    """
    Calculate the porous feeding parameter.

    Returns:
        float: The porous feeding parameter, beta.
    """
    b = bearing
    beta = 6 * b.kappa * b.ra**2 / (b.hp * b.ha**3)
    return beta

def get_load_capacity(bearing, p):
    """
    Calculate the load capacity of the bearing.

    Args:
        p (numpy.ndarray): The pressure distribution.

    Returns:
        numpy.ndarray: The calculated load capacity.
    """
    b = bearing

    if bearing.csys == "polar":
        w = np.sum(np.pi * (p - b.pa) * np.gradient(b.r**2)[:, np.newaxis], axis=0)
    elif bearing.csys == "cartesian":
        w = (np.sum((p - b.pa) * np.gradient(b.r)[:, np.newaxis], axis=0))
    return w

def get_stiffness(bearing, w):
    """
    Calculate the stiffness.

    Args:
        w (numpy.ndarray): The load or deflection values.

    Returns:
        numpy.ndarray: The stiffness values.
    """
    b = bearing
    k = -np.gradient(w, b.ha.flatten())
    return k

def get_volumetric_flow(bearing, p: np.ndarray, soltype: bool) -> tuple:
    """Calculate volumetric flow rates through the bearing.

    Args:
        bearing: Bearing instance containing geometry and properties
        p (np.ndarray): Pressure distribution array
        soltype (bool): Solution type (ANALYTIC or NUMERIC)

    Returns:
        tuple: (qs, qa, qc) where:
            - qs (np.ndarray): Supply flow rate (L/min)
            - qa (np.ndarray): Ambient flow rate (L/min)
            - qc (np.ndarray): Chamber flow rate (L/min)
    """
    b = bearing

    if soltype == ANALYTIC:
        h = b.ha[np.newaxis, :]
        r = b.r[:, np.newaxis]
    elif soltype == NUMERIC:
        h = b.ha[np.newaxis, :] + b.geom[:, np.newaxis]
        r = b.r[:, np.newaxis]

    q = (-6e4 * np.pi * h**3 * r  * b.rho *
            np.gradient(p**2, axis=0) / (12 * b.mu * b.pa * np.gradient(b.r)[:, np.newaxis]))
    
    qa = q[-1, :]
    qc = q[1, :]
    qs = qa - qc
    return qs, qa, qc


def get_pressure_analytic_infinite(bearing):
    """
    Calculates the solution for the pressure distribution in infinitely long bearings and seals.
    """

    b = bearing

    f = (2 * b.beta) ** 0.5
    slip = (1 + b.Psi) ** 0.5

    # nondimensionals
    Pa = 1
    Ra = 1
    R = b.r / b.ra
    Ps = b.ps / b.pa
    Pc = b.pc / b.pa

    exp_f = np.exp((f * Ra) / slip)

    numer1 = -Pc**2 + Ps**2 + exp_f * (Pa**2 - Ps**2)
    numer2 = exp_f * (-Pa**2 + Ps**2 + exp_f * (Pc**2 - Ps**2))

    denom = -1 + np.exp((2 * f * Ra) / slip)
 
    C1 = numer1 / denom
    C2 = numer2 / denom

    p = b.pa * (Ps**2 + C1 * np.exp(np.outer(R, f) / slip) + C2 * np.exp(-np.outer(R, f) / slip)) ** 0.5
    return p

def get_pressure_analytic_annular(bearing):
    """
    Calculates the Bessel function solution for the pressure distribution in annular bearings and seals.
    """

    b = bearing

    f = (2 * b.beta) ** 0.5

    # nondimensionals
    Pa = 1
    Ra = 1
    R = b.r / b.ra
    Ps = b.ps / b.pa
    Pc = b.pc / b.pa
    Rc = b.rc / b.ra

    numer1 = (Pa**2 - Ps**2) * k0(f * Rc) + (Ps**2 - Pc**2) * k0(f * Ra)
    numer2 = (Pa**2 - Ps**2) * i0(f * Rc) + (Ps**2 - Pc**2) * i0(f * Ra)

    denom = i0(f * Rc) * k0(f * Ra) - i0(f * Ra) * k0(f * Rc)

    C1 = numer1 / denom
    C2 = numer2 / denom

    p = b.pa * (Ps**2 - C1 * i0(np.outer(R, f)) + C2 * k0(np.outer(R, f))) ** 0.5
    return p

def get_pressure_analytic_circular(bearing):
    """
    Calculates the Bessel function solution for the pressure distribution in circluar trust bearings.
    """
    b = bearing
    p = b.ps * (1 - (1 - b.pa**2 / b.ps**2) *
            i0(np.outer(b.r/b.ra, (2 * b.beta) ** 0.5)) / i0((2 * b.beta) ** 0.5)) ** 0.5
    return p

def get_pressure_numeric(bearing):
    b = bearing
    p = np.zeros((len(b.r), len(b.ha)))

    # uniform kappa
    kappa = b.kappa * np.ones_like(b.r)

    # Partially blocked restrictors, set blocked region to 0 permeability
    if b.blocked:
        kappa[b.block_in] = 0 
    
    # porous feeding terms
    porous_source = - kappa / (2 * b.hp * b.mu)
    
    for i in range(len(b.ha)):
        h = b.ha[i] + b.geom
        
        if b.csys == "polar":
            epsilon = (1 + b.Psi) * b.r * h**3 / (24 * b.mu)
            coefficient = sp.diags(1 / b.r, 0)
        elif b.csys == "cartesian":
            epsilon = (1 + b.Psi) * h**3 / (24 * b.mu)
            coefficient = 1

        diff_mat = build_diff_matrix(coefficient, epsilon, b.dr)
        A = sp.lil_matrix(diff_mat + sp.diags(porous_source, 0))
        
        f = b.ps**2 * porous_source
        
        # Boundary conditions
        if b.type == "bearing": 
            # symmetry at r=0
            A[0, 1] = - A[0, 0] 
            f[0] = 0 
        elif b.type == "seal": 
            # symmetry at r=rc
            A[0, 0] = 1
            A[0, 1] = 0
            f[0] = b.pc**2 
        
        # dirilect at r=ra
        A[-1, -2] = 0 
        A[-1, -1] = 1
        f[-1] = b.pa**2

        A = A.tocsr()
        p[:, i] = spla.spsolve(A, f)**0.5

    return p

def build_diff_matrix(coef: np.ndarray, eps: np.ndarray, dr: np.ndarray) -> sp.csr_matrix:
    """Construct finite-difference matrix for coefficient @ D_r(epsilon * D_r(f(r)))

    Builds a sparse matrix representing the discretized differential operator
    using second-order central differences with variable coefficients.
    """

    N = len(dr)

    # Compute epsilon at half-points
    eps_half = (eps[:-1] + eps[1:]) / 2

    # Finite difference second derivative matrix with variable coefficient
    diag_main = np.zeros(N)
    diag_upper = np.zeros(N-1)
    diag_lower = np.zeros(N-1)

    # interior points with 3 point stencil
    diag_main[1:-1] = -(eps_half[1:] + eps_half[:-1]) / dr[1:-1]**2
    diag_upper[1:] = eps_half[1:] / dr[1:-1]**2
    diag_lower[:-1] = eps_half[:-1] / dr[1:-1]**2

    # Assemble sparse matrix
    L_mat = coef @ sp.diags([diag_lower, diag_main, diag_upper], [-1, 0, 1], format="csr")

    return L_mat


if __name__ == "__main__":
    bearing = AnnularBearing()  # Instantiate the class
    result = solve_bearing(bearing, soltype=NUMERIC)
    figure = plot_key_results(bearing, result)
    show(figure)

