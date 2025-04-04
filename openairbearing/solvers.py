from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import i0, k0

from openairbearing.utils import (
    Result,
    get_load_capacity,
    get_stiffness,
    get_volumetric_flow,
)


def solve_bearing(bearing, soltype: bool) -> Result:
    match soltype:
        case "analytic":
            name = "analytic"
            match bearing.case:
                case "circular":
                    p = get_pressure_analytic_circular(bearing)
                case "annular":
                    p = get_pressure_analytic_annular(bearing)
                case "infinite":
                    p = get_pressure_analytic_infinite(bearing)
                case _:
                    return Result(
                        name="none",
                        p=np.array([]),
                        w=np.array([]),
                        k=np.array([]),
                        qs=np.array([]),
                        qa=np.array([]),
                        qc=np.array([]),
                    )
        case "numeric":
            name = "numeric"
            # if bearing.case == "rectangular":
            #     p = get_pressure_2d_numeric(bearing)
            # else:
            p = get_pressure_numeric(bearing)
        case "numeric2d":
            name = "numeric2d"
            p = get_pressure_2d_numeric(bearing)
        case _:
            raise ValueError("Invalid solution type")

    w = get_load_capacity(bearing=bearing, p=p)
    k = get_stiffness(bearing=bearing, w=w)
    qs, qa, qc = get_volumetric_flow(bearing=bearing, p=p, soltype=soltype)

    return Result(name=name, p=p, w=w, k=k, qs=qs, qa=qa, qc=qc)


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
    R = b.x / b.xa
    Ps = b.ps / b.pa
    Pc = b.pc / b.pa

    exp_f = np.exp((f * Ra) / slip)

    numer1 = -(Pc**2) + Ps**2 + exp_f * (Pa**2 - Ps**2)
    numer2 = exp_f * (-(Pa**2) + Ps**2 + exp_f * (Pc**2 - Ps**2))

    denom = -1 + np.exp((2 * f * Ra) / slip)

    C1 = numer1 / denom
    C2 = numer2 / denom

    p = (
        b.pa
        * (
            Ps**2
            + C1 * np.exp(np.outer(R, f) / slip)
            + C2 * np.exp(-np.outer(R, f) / slip)
        )
        ** 0.5
    )
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
    R = b.x / b.xa
    Ps = b.ps / b.pa
    Pc = b.pc / b.pa
    Rc = b.xc / b.xa

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
    p = (
        b.ps
        * (
            1
            - (1 - b.pa**2 / b.ps**2)
            * i0(np.outer(b.x / b.xa, (2 * b.beta) ** 0.5))
            / i0((2 * b.beta) ** 0.5)
        )
        ** 0.5
    )
    return p


def get_pressure_numeric(bearing):
    b = bearing
    p = np.zeros((len(b.x), len(b.ha)))

    # uniform kappa
    kappa = b.kappa * np.ones_like(b.x)

    # Partially blocked restrictors, set blocked region to 0 permeability
    if b.blocked:
        kappa[b.block_in] = 0

    # porous feeding terms
    porous_source = -kappa / (2 * b.hp * b.mu)

    for i in range(len(b.ha)):
        h = b.ha[i] + b.geom

        if b.csys == "polar":
            epsilon = (1 + b.Psi) * b.x * h**3 / (24 * b.mu)
            coefficient = sp.diags(1 / b.x, 0)
        elif b.csys == "cartesian":
            epsilon = (1 + b.Psi) * h**3 / (24 * b.mu)
            coefficient = 1

        diff_mat = build_diff_matrix(coefficient, epsilon, b.dx)
        A = sp.lil_matrix(diff_mat + sp.diags(porous_source, 0))

        f = b.ps**2 * porous_source

        # Boundary conditions
        if b.type == "bearing":
            # symmetry at r=0
            A[0, 1] = -A[0, 0]
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
        p[:, i] = spla.spsolve(A, f) ** 0.5

    return p


def build_diff_matrix(
    coef: np.ndarray, eps: np.ndarray, dr: np.ndarray
) -> sp.csr_matrix:
    """Construct finite-difference matrix for coefficient @ D_r(epsilon * D_r(f(r)))

    Builds a sparse matrix representing the discretized differential operator
    using second-order central differences with variable coefficients.
    """

    N = len(dr)

    # Compute epsilon at half-points
    eps_half = (eps[:-1] + eps[1:]) / 2
    # Finite difference second derivative matrix with variable coefficient
    diag_main = np.zeros(N)
    diag_upper = np.zeros(N - 1)
    diag_lower = np.zeros(N - 1)

    # interior points with 3 point stencil
    diag_main[1:-1] = -(eps_half[1:] + eps_half[:-1]) / dr[1:-1] ** 2
    diag_upper[1:] = eps_half[1:] / dr[1:-1] ** 2
    diag_lower[:-1] = eps_half[:-1] / dr[1:-1] ** 2

    # Assemble sparse matrix
    L_mat = sp.diags([diag_lower, diag_main, diag_upper], [-1, 0, 1], format="csr")
    # Handle also scalar coefficients
    if isinstance(coef, (float, int)):
        return coef * L_mat
    else:
        return coef @ L_mat


def get_pressure_2d_numeric(bearing):
    """
    Solve the 2D pressure distribution for the bearing using finite differences.

    Args:
        bearing: Bearing object containing geometry and properties.

    Returns:
        np.ndarray: 2D pressure distribution array.
    """
    b = bearing
    N = b.nx
    M = b.ny

    p = np.zeros((M, N, len(b.ha)))

    # Uniform kappa
    kappa = b.kappa * np.ones((N, M))

    # Partially blocked restrictors, set blocked region to 0 permeability
    if b.blocked:
        kappa[b.block_in] = 0

    # Porous feeding terms
    porous_source = -kappa / (2 * b.hp * b.mu)

    for i, ha in enumerate(b.ha):
        h = ha + b.geom
        if b.csys == "polar":
            epsilon_r = (1 + b.Psi) * b.x[:, None] * h**3 / (24 * b.mu)
            epsilon_theta = (1 + b.Psi) * b.x[:, None] * h**3 / (24 * b.mu)
            epsilon = (epsilon_r, epsilon_theta)
            coefficient = sp.diags(1 / np.repeat(b.x, b.ny), 0)
        elif b.csys == "cartesian":
            epsilon = (1 + b.Psi) * h**3 / (24 * b.mu)
            coefficient = 1

        # boundary conditions
        if b.case == "rectangular":
            bc = {
                "west": "Dirichlet",
                "east": "Dirichlet",
                "north": "Dirichlet",
                "south": "Dirichlet",
            }
        elif b.case == "circular":
            bc = {
                "west": "Neumann",
                "east": "Dirichlet",
                "north": "Periodic",
                "south": "Periodic",
            }
        elif b.case == "annular":
            bc = {
                "west": "Dirichlet",
                "east": "Dirichlet",
                "north": "Periodic",
                "south": "Periodic",
            }

        # Build the 2D differential matrix
        A = build_2d_diff_matrix(
            coef=coefficient,
            epsilon=epsilon,
            porous_source=porous_source,
            dx=b.dx,
            dy=b.dy,
            bc=bc,
            N=N,
            M=M,
        )

        # Right-hand side (forcing term)
        f = (b.ps**2 * porous_source).flatten()

        if bc["west"] == "Dirichlet":
            f[0::N] = b.pa**2
        if bc["east"] == "Dirichlet":
            f[N - 1 :: N] = b.pa**2
        if bc["north"] == "Dirichlet":
            f[-N:] = b.pa**2
        if bc["south"] == "Dirichlet":
            f[:N] = b.pa**2

        # Solve the linear system
        A = A.tocsr()
        p_flat = spla.spsolve(A, f)
        p[:, :, i] = p_flat.reshape((M, N)) ** 0.5

    return p


def build_2d_diff_matrix(
    coef: np.ndarray,
    epsilon: float | tuple | np.ndarray,
    porous_source: np.ndarray,
    dx: float,
    dy: float,
    bc: dict,
    N: int,
    M: int,
) -> sp.csr_matrix:
    """
    Construct a finite-difference matrix for 2D differential operator:
    coef @ D_r(epsilon * D_r(f(r))) + coef @ D_x(epsilon * D_x(f(x)))

    Args:
        coef (np.ndarray): Coefficient matrix.
        eps (np.ndarray): Epsilon matrix (variable coefficients) (N,M).
        dr (np.ndarray): Radial grid spacing.
        dx (np.ndarray): angular grid spacing.
        bc (dict): Dictionary specifying boundary conditions for "left", "right", "top", and "bottom".

    Returns:
        sp.csr_matrix: Sparse matrix representing the 2D differential operator.
    """

    if isinstance(epsilon, tuple):
        epsilon_x, epsilon_y = epsilon
        eps_x = (epsilon_x[:-1, :] + epsilon_y[1:, :]) / 2
        eps_y = (epsilon_y[:, :-1] + epsilon_y[:, 1:]) / 2
    else:  # Assume epsilon is a 2D array
        eps_x = (epsilon[:-1, :] + epsilon[1:, :]) / 2
        eps_y = (epsilon[:, :-1] + epsilon[:, 1:]) / 2

    # pad stencil epsilon to match the size of the matrix
    eps_w = np.vstack([eps_x, np.zeros((1, M))])
    eps_e = np.vstack([np.zeros((1, M)), eps_x])
    eps_s = np.hstack([eps_y, np.zeros((N, 1))])
    eps_n = np.hstack([np.zeros((N, 1)), eps_y])

    # Form sparse matrix diagonals
    diag_center = (
        -((eps_w + eps_e) / dx[:, None] ** 2 + (eps_n + eps_s) / dy[None, :] ** 2)
        + porous_source
    ).flatten("F")
    diag_west = (eps_w / dx[:, None] ** 2).flatten("F")[:-1]
    diag_east = (eps_e / dx[:, None] ** 2).flatten("F")[1:]
    diag_north = (eps_n / dy[None, :] ** 2).flatten("F")[:-N]
    diag_south = (eps_s / dy[None, :] ** 2).flatten("F")[N:]

    # Boundary conditions
    diag_east[N - 1 :: N] = 0
    diag_west[N - 2 :: N] = 0

    diag_east[:N] = 0
    diag_east[-N:] = 0

    diag_west[:N] = 0
    diag_west[-N:] = 0

    diag_south[::N] = 0
    diag_south[N - 1 :: N] = 0

    diag_north[::N] = 0
    diag_north[N - 1 :: N] = 0

    if bc["west"] == "Dirichlet":
        diag_center[0::N] = 1
        diag_east[::N] = 0
    if bc["east"] == "Dirichlet":
        diag_center[N - 1 :: N] = 1
        diag_west[N - 1 :: N] = 0
    if bc["north"] == "Dirichlet":
        diag_center[-N:] = 1
        diag_south[-N:] = 0
    if bc["south"] == "Dirichlet":
        diag_center[:N] = 1
        diag_north[:N] = 0

    L_mat = sp.diags(
        [diag_center, diag_east, diag_west, diag_north, diag_south],
        [0, 1, -1, N, -N],
        format="csr",
    )

    # Handle scalar coefficients
    if isinstance(coef, (float, int)):
        return coef * L_mat
    else:
        return coef @ L_mat
