import numpy as np
from dataclasses import dataclass

from openairbearing.config import ANALYTIC, NUMERIC


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


def get_area(bearing):
    b = bearing
    match b.case:
        case "circular":
            A = np.pi * b.xa**2
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
                geom = b.error * (1 - b.x / b.xa)
            case "quadratic":
                geom = b.error * (1 - b.x**2 / b.xa**2)
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

    if getattr(b, "blocked", False):
        kappa = (
            2 * b.Qsc / 6e4 * b.mu * b.hp * b.pa / (b.block_A * (b.psc**2 - b.pa**2))
        )
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
        Qsc = (
            b.kappa * 6e4 * b.block_A * (b.psc**2 - b.pa**2) / (2 * b.mu * b.hp * b.pa)
        )
    else:
        Qsc = b.kappa * 6e4 * b.A * (b.psc**2 - b.pa**2) / (2 * b.mu * b.hp * b.pa)
    return round_to_sig_dig(Qsc, 3)


def round_to_sig_dig(number, digits):
    return np.round(number, -int(np.floor(np.log10(np.abs(number)))) + (digits - 1))


def get_dA(bearing) -> np.ndarray:
    b = bearing
    if b.ny == 1:
        match b.csys:
            case "polar":
                dA = np.pi * np.gradient(b.x**2)
                dA[[0, -1]] = dA[[0, -1]] / 2
            case "cartesian":
                dA = b.dx.copy()
                dA[[0, -1]] = dA[[0, -1]] / 2
            case _:
                raise ValueError("Error: invalid csys in dA calculation")
    else:
        match b.csys:
            case "polar":
                dA = np.pi * np.gradient(b.x**2)[None, :] * b.dy[:, None]
                dA[[0, -1], :] = dA[[0, -1], :] / 2
            case "cartesian":
                dA = (
                    b.dx[
                        None,
                        :,
                    ]
                    * b.dy[:, None]
                )
                dA[[0, -1], :] = dA[[0, -1], :] / 2
                dA[:, [0, -1]] = dA[:, [0, -1]] / 2
            case _:
                raise ValueError("Error: invalid csys in dA calculation")
    return dA


def get_load_capacity(bearing, p: np.ndarray) -> np.ndarray:
    """
    Calculate the load capacity of the bearing.

    Args:
        p (numpy.ndarray): The pressure distribution.

    Returns:
        numpy.ndarray: The calculated load capacity.
    """
    b = bearing
    dA = get_dA(b)
    p_rel = p - b.pa
    if b.ny == 1:
        w = np.sum(p_rel * dA[:, None], axis=0)
    else:
        w = np.sum(p_rel * dA[:, :, None], axis=(0, 1))
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
    if b.ny == 1:
        if soltype == ANALYTIC:
            h = b.ha
            # print("a: ", p[[0, 1],-1]*1e-6)
        elif soltype == NUMERIC:
            h = b.ha + b.geom[:, None]
            # print("n: ", p[[0, 1],-1]*1e-6)

        if b.csys == "polar":
            q = (
                -6e4
                * h**3
                * b.rho
                * np.gradient(p**2, axis=0)
                * np.pi
                * b.x[:, None]
                / (12 * b.mu * b.pa * b.dx[:, None])
            )
        elif b.csys == "cartesian":
            q = (
                -6e4
                * h**3
                * b.rho
                * np.gradient(p**2, axis=0)
                / (12 * b.mu * b.pa * b.dx[:, None])
            )
        else:
            raise ValueError("Invalid csys")

        qa = q[-1, :]
        qc = q[1, :]
        qs = qa - qc
    else:
        if soltype == ANALYTIC:
            raise TypeError("Analytic 2d attempted")
        if soltype == NUMERIC:
            h = b.ha[None, None, :] + b.geom.T[:, :, None]

            qx = (
                -6e4
                * h**3
                * b.rho
                * np.gradient(p**2, axis=1)
                * b.dy[:, None, None]
                / (12 * b.mu * b.pa * b.dx[None, :, None])
            )
            qy = (
                -6e4
                * h**3
                * b.rho
                * np.gradient(p**2, axis=0)
                * b.dx[None, :, None]
                / (12 * b.mu * b.pa * b.dy[:, None, None])
            )

            qa = np.sum(np.abs(qx[:, (0, -1), :]), axis=(0, 1)) + np.sum(
                abs(qy[(0, -1), :, :]), axis=(0, 1)
            )
            qc = 0
            qs = qa - qc
    return qs, qa, qc
