# openairbearing/solvers_fem.py
import numpy as np
from dataclasses import dataclass
from skfem import MeshLine, MeshTri, Basis, BilinearForm, LinearForm, asm, condense, solve
from skfem.element import ElementLineP2, ElementLineP1 ,ElementTriP1
from skfem.helpers import grad, dot
#from skfem import *

from openairbearing.fem_utils import (
    load_circular_axial,
    qfilm_circular_rim,
    load_infinite_axial_per_width,
    qfilm_planar_edge,
    load_rect_axial,  # ∫ (p - pa) dx  [per unit width]
    qfilm_planar_edge,              # boundary flow on planar edges; weight=1
)

@dataclass
class FemArtifacts:
    mesh: object | None = None
    basis: object | None = None
    r_nodes: np.ndarray | None = None

def _sweep_over_h(nh, solve_one_h):
    """Generic h-sweep helper returning stacked arrays."""
    p_list, W, qa, qc, qs = [], [], [], [], []
    for i in range(nh):
        p_vec, W_i, qa_i, qc_i, qs_i = solve_one_h(i)
        p_list.append(p_vec)
        W.append(W_i); qa.append(qa_i); qc.append(qc_i); qs.append(qs_i)
    return np.column_stack(p_list), np.array(W), np.array(qa), np.array(qc), np.array(qs)

#interpoltates P2 solution to bearing.x grid
""" def _interp_to_grid(basis, vec, x_eval):
    xnodes = np.array(basis.doflocs[0])
    order  = np.argsort(xnodes)
    return np.interp(x_eval, xnodes[order], vec[order]), xnodes, order """
def _interp_to_grid(basis, dof_values, x_eval):
    """
    Interpolate a nodal DOF vector defined on `basis` nodes onto `x_eval`.
    Works for P1/P2. Returns a 1D ndarray ONLY.
    """
    xnodes = np.asarray(basis.doflocs[0]).ravel()
    order  = np.argsort(xnodes)
    return np.interp(x_eval, xnodes[order], np.asarray(dof_values).ravel()[order])

def _fem1d_circular_annular(b, *, return_artifacts: bool):
    # geometry split
    circular = (b.case == "circular")
    r0 = 0.0 if circular else float(b.xc)  # natural at center only for solid disk
    r1 = float(b.xa)

    # mesh/basis
    ne = max(2, int(b.nx) - 1)
    mesh = MeshLine(np.linspace(r0, r1, ne + 1)).with_boundaries({
        "inner": lambda x: np.isclose(x[0], r0),
        "outer": lambda x: np.isclose(x[0], r1),
    })
    basis = Basis(mesh, ElementLineP2())

    # forms for η := ps^2 - p^2   (1/r d_r (r dη/dr) + (S/ε) η = 0)
    def assemble_matrix(K2):
        @BilinearForm
        def a(u, v, w):
            r = w.x[0]
            return r * (grad(u)[0] * grad(v)[0]) + K2 * r * u * v
        return a.assemble(basis)

    # fem_utils post-processing forms
    load_form = load_circular_axial()
    q_form    = qfilm_circular_rim(mu=b.mu, p_ref=b.pa)

    r_nodes = basis.doflocs[0]
    order = np.argsort(r_nodes)
    r_sorted = r_nodes[order]

    def solve_one_h(i):
        # local film height (axisymmetric slice; grooves etc. can alter later)
        h = float(b.ha[i])

        # S/ε consistent with FD parameters:
        #   S = kappa/(2 μ hp),  ε = (1+Ψ) h^3/(24 μ)  ⇒ K2 = S/ε = 12 κ / ((1+Ψ) hp h^3)
        K2 = (12.0 * b.kappa) / ((1.0 + b.Psi) * b.hp * h**3)

        A = assemble_matrix(K2)

        # Dirichlet lift for η = ps^2 - p^2
        x0 = basis.zeros()
        D  = basis.get_dofs("outer")
        x0[D] = (b.ps**2 - b.pa**2)
        if not circular:             # annular or circular with center hole
            Di = basis.get_dofs("inner")
            x0[Di] = (b.ps**2 - b.pc**2)
            D = D | Di                              # both ends Dirichlet
        # else: solid disk -> natural at r=0

        eta = solve(*condense(A, x=x0, D=D))
        p_dofs = np.sqrt(np.maximum(b.ps**2 - eta, 0.0))

        # Return p on constructor’s r-grid so plots & utils keep working
        #p_on_x = np.interp(b.x, r_sorted, p_dofs[order])  #change to use the interp_to_grid function? answe no the the function convolutes
        p_on_x = _interp_to_grid(basis, p_dofs, b.x)
        
        
        # fem_utils loads/flows on the FEM field
        pfield = basis.interpolate(p_dofs)
        W  = asm(load_form, basis, p=pfield, pa=b.pa).sum()
        fb_out = basis.boundary("outer")
        fb_in  = basis.boundary("inner")
        qa = asm(q_form, fb_out, p=p_dofs, h=h).sum()                         # to ambient
        qc = -asm(q_form, fb_in,  p=p_dofs, h=h).sum() if fb_in.nelems else 0 # to chamber
        qs = qa - qc                                                          # porous source

        return p_on_x, W, qa, qc, qs

    p, W, qa, qc, qs = _sweep_over_h(b.nh, solve_one_h)

    artifacts = FemArtifacts(mesh=mesh, basis=basis, r_nodes=r_nodes) if return_artifacts else None
    return p, W, qa, qc, qs, artifacts

def _fem1d_infinite(b, *, return_artifacts: bool):
    """
    Infinite linear bearing (per-unit width), x in [0, xa].
    η := ps^2 - p^2,  d^2η/dx^2 = (S/ε) η  ⇒  ∫ η' v' + ∫ K2 η v = 0,
    with K2 = S/ε = 12 κ / ((1+Ψ) hp h^3).
    Dirichlet: left→pc, right→pa.  Loads/flows via fem_utils (per width).
    Returns: p(b.x, h), W[h] (N/m), qa/qc/qs[h] (L/min per m), artifacts.
    """
    # --- mesh/basis (P2) ---
    xL, xR = float(b.x.min()), float(b.x.max())
    ne = max(2, int(b.nx) - 1)
    mesh = MeshLine(np.linspace(xL, xR, ne + 1)).with_boundaries({
        "left":  lambda x: np.isclose(x[0], xL),
        "right": lambda x: np.isclose(x[0], xR),
    })
    basis = Basis(mesh, ElementLineP2())

    # --- bilinear form for η ---
    def assemble_matrix(K2):
        @BilinearForm
        def a(u, v, w):
            return grad(u)[0] * grad(v)[0] + K2 * u * v
        return a.assemble(basis)

    # --- fem_utils forms (planar) ---
    load_form = load_infinite_axial_per_width()
    q_form    = qfilm_planar_edge(mu=b.mu, p_ref=b.pa)

    # output arrays
    P  = np.zeros((b.nx, b.nh))   # pressure on constructor grid
    W  = np.zeros(b.nh)           # N/m
    qa = np.zeros(b.nh)           # L/min per m width (to ambient, +x)
    qc = np.zeros(b.nh)           # L/min per m width (to chamber, -x; sign fixed below)

    # store node coords for artifacts
    xnodes = np.array(basis.doflocs[0])

    for j, h in enumerate(map(float, b.ha)):
        # K2 = 12 κ / ((1+Ψ) hp h^3)
        K2 = 12.0 * b.kappa / ((1.0 + b.Psi) * b.hp * (h**3))
        A  = assemble_matrix(K2)

        # Dirichlet lift on η = ps^2 - p^2
        x0v = basis.zeros()
        D   = basis.get_dofs("right"); x0v[D]  = (b.ps**2 - b.pa**2)
        Dl  = basis.get_dofs("left");  x0v[Dl] = (b.ps**2 - b.pc**2)
        D   = D | Dl

        eta   = solve(*condense(A, x=x0v, D=D))
        p_dofs = np.sqrt(np.maximum(b.ps**2 - eta, 0.0))

        # resample to constructor grid (works for P1/P2)
        x_nodes = np.asarray(basis.doflocs[0]).ravel()
        ordr    = np.argsort(x_nodes)
        #p_on_x  = np.interp(b.x, x_nodes[ordr], p_dofs[ordr])
        p_on_x= _interp_to_grid(basis, p_dofs, b.x)
        P[:, j] = p_on_x

        # assemble loads/flows on the FEM field
        pfield = basis.interpolate(p_dofs)

        W[j]  = asm(load_form, basis, p=pfield, pa=b.pa).sum()
        fb_r  = basis.boundary("right")
        fb_l  = basis.boundary("left")
        qa[j] = asm(q_form, fb_r, p=p_dofs, h=h).sum()          # to ambient (+x)
        qc[j] = -asm(q_form, fb_l, p=p_dofs, h=h).sum()         # into chamber

    qs = qa - qc

    artifacts = FemArtifacts(mesh=mesh, basis=basis, r_nodes=xnodes) if return_artifacts else None
    return P, W, qa, qc, qs, artifacts

def _fem2d_rectangular(b, *, return_artifacts: bool):
    """
    2D rectangular pad, Cartesian: Δη = K2 η with
        K2 = 12 κ / ((1+Ψ) hp h^3)   (constant over the pad for now)
    BC: all four edges ambient Dirichlet: p = pa (=> η = ps^2 - pa^2).
    Loads & boundary flows via fem_utils.
    Returns:
        P: (ny, nx, nh) absolute pressure on the constructor grid
        W: (nh,) [N]
        qa,qc,qs: (nh,) [L/min]
        artifacts: {mesh, basis} or None
    """
    # --- mesh/basis on constructor grid (tensor grid) ---
    xgrid, ygrid = b.x, b.y
    mesh = MeshTri.init_tensor(xgrid, ygrid).with_boundaries({
        "left":   lambda x: np.isclose(x[0], xgrid[0]),
        "right":  lambda x: np.isclose(x[0], xgrid[-1]),
        "bottom": lambda x: np.isclose(x[1], ygrid[0]),
        "top":    lambda x: np.isclose(x[1], ygrid[-1]),
    })
    basis = Basis(mesh, ElementTriP1())

    # --- weak form: ∫ ∇η·∇v + ∫ K2 η v = 0 ---
    def assemble_matrix_2d(K2):
        @BilinearForm
        def a(u, v, w):
            return dot(grad(u), grad(v)) + K2 * u * v
        return a.assemble(basis)

    # fem_utils forms
    load_form = load_rect_axial()
    q_edge    = qfilm_planar_edge(mu=b.mu, p_ref=b.pa)

    nx, ny, nh = b.nx, b.ny, b.nh
    P  = np.zeros((ny, nx, nh))
    W  = np.zeros(nh)
    qa = np.zeros(nh)
    qc = np.zeros(nh)

    # precompute nodal (x,y) ordering so we can reshape DOFs onto (ny, nx)
    xnodes = np.asarray(basis.doflocs[0]).ravel()
    ynodes = np.asarray(basis.doflocs[1]).ravel()
    order  = np.lexsort((xnodes, ynodes))  # sort by y, then x
    # sanity: nodes should be a full tensor grid with lengths len(xgrid)*len(ygrid)
    # reshape map:
    def dof_to_grid(vec):
        return np.asarray(vec)[order].reshape(len(ygrid), len(xgrid))

    for j, h in enumerate(map(float, b.ha)):
        # K2 = S/ε = 12 κ / ((1+Ψ) hp h^3)
        K2 = 12.0 * b.kappa / ((1.0 + b.Psi) * b.hp * h**3)

        A  = assemble_matrix_2d(K2)

        # Dirichlet on all edges: η = ps^2 - pa^2
        D_left  = basis.get_dofs("left")
        D_right = basis.get_dofs("right")
        D_bot   = basis.get_dofs("bottom")
        D_top   = basis.get_dofs("top")
        D = D_left | D_right | D_bot | D_top

        x0 = basis.zeros()
        x0[D] = (b.ps**2 - b.pa**2)

        eta    = solve(*condense(A, x=x0, D=D))
        p_dofs = np.sqrt(np.maximum(b.ps**2 - eta, 0.0))

        # --- return pressure on constructor grid (ny, nx) ---
        P[:, :, j] = dof_to_grid(p_dofs)

        # --- loads (volume) ---
        pfield = basis.interpolate(p_dofs)
        W[j]   = asm(load_form, basis, p=pfield, pa=b.pa).sum()

        # --- boundary flows (ambient on all edges) ---
        fbL, fbR = basis.boundary('left'), basis.boundary('right')
        fbB, fbT = basis.boundary('bottom'), basis.boundary('top')

        # IMPORTANT: pass DOF vector (not interpolated field) to boundary forms
        qa_L = asm(q_edge, fbL, p=p_dofs, h=h).sum()
        qa_R = asm(q_edge, fbR, p=p_dofs, h=h).sum()
        qa_B = asm(q_edge, fbB, p=p_dofs, h=h).sum()
        qa_T = asm(q_edge, fbT, p=p_dofs, h=h).sum()

        qa[j] = qa_L + qa_R + qa_B + qa_T
        qc[j] = 0.0   # no chamber edges in this pad

    qs = qa - qc
    artifacts = {"mesh": mesh, "basis": basis} if return_artifacts else None
    return P, W, qa, qc, qs, artifacts

def solve_bearing_fem(bearing, *, return_artifacts: bool = False):
    """Public FEM facade. Dispatch by bearing.case."""
    b = bearing
    if b.case in ("circular", "annular"):
        return _fem1d_circular_annular(b, return_artifacts=return_artifacts)
    elif b.case == "infinite":
         return _fem1d_infinite(b, return_artifacts=return_artifacts)    
    elif b.case == "rectangular":
        return _fem2d_rectangular(bearing, return_artifacts=return_artifacts)
    elif b.case == "cone":
        raise NotImplementedError("FEM 'cone' coming next.")
    elif b.case in ("spherical_rasnick", "spherical"):
        raise NotImplementedError("FEM 'spherical' coming next.")
    elif b.case == "journal":
        raise NotImplementedError("FEM 'journal' coming next.")
    else:
        raise ValueError(f"Unsupported case for FEM: {b.case}")
    
""" D. Naming guidance for the remaining 6 FEM cases

Use this consistent naming inside solvers_fem.py:

_fem1d_circular_annular(b, ...) ✅ (done)

_fem1d_infinite(b, ...) (x–only, per-unit width; planar forms: load_infinite_axial_per_width, qfilm_planar_edge)

_fem2d_rectangular(b, ...) (x–y planar mesh; load_rect_axial, qfilm_planar_edge)

_fem2d_cone(b, ...) (slant-r, φ metric; you already started cone helpers in fem_utils)

_fem2d_spherical(b, ...) (θ′–φ′ metric; use your spherical load/edge forms)

_fem2d_journal(b, ...) (φ–z cylinder; journal load/edge forms)

…and map them in solve_bearing_fem.

This gives you one place to click to find each case, while keeping the public entry point stable (solve_bearing_fem) and your UI/API unchanged. """
