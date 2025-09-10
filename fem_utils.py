# weakforms_loadflow.py
# Unified load & boundary-flow forms for common air-bearing geometries.
# Usage notes:
# - All load forms integrate GAUGE pressure by default: (p - pa).
# - All flow forms use the p^2 formulation:
#       q_n = -(h^3/(24 μ p_ref)) * ∂_n(p^2) = -(h^3/(12 μ p_ref)) * p * (∇p·n)
#   and return L/min via the 6e4 factor.
# - For flow forms, pass fields: w['p'] (absolute pressure) and w['h'] (film height).

import numpy as np
from skfem import LinearForm
from skfem.helpers import grad, dot


# =========================
# Core, geometry-agnostic
# =========================

def make_load_component(area_weight, n_dot_dir=lambda w: 1.0, *, subtract_pa: bool = True):
    """
    Generic surface-load component:
        W_dir = ∬ area_weight(w) * (n · dir)(w) * [p - pa (optional)] dΩ_param
    - area_weight(w): Jacobian from parameter domain to physical area
    - n_dot_dir(w):   projection of unit normal onto desired direction
    """
    @LinearForm
    def load_form(v, w):
        p  = w.p
        pa = getattr(w, "pa", 0.0)
        pg = (p - pa) if subtract_pa else p
        return area_weight(w) * n_dot_dir(w) * pg * v
    return load_form


def make_qfilm_p2(boundary_weight, *, mu: float, p_ref: float, h=None):
    """
    Unified boundary flow Q [L/min] via p^2-form:
        Q = ∮ 6e4 * boundary_weight(w) * [ -(h^3/(24 μ p_ref)) * 2 p (∇p·n) ] ds_param
    Provide, at assemble-time: fields 'p' (absolute) and 'h'.
    """
    
    
    """
    Unified boundary flow Q [L/min] via p^2-form:
        Q = ∮ 6e4 * W(w) * [ -(h^3/(24 μ p_ref)) * 2 p (∇p·n) ] ds
    If `h` is:
      - callable(w): evaluated at facet quadrature points (captured in closure)
      - scalar: captured constant
      - None: expect `h` to be provided at assemble(..., h=...) as scalar/array/DiscreteField
    """
    K = 60000.0 / (24.0 * mu * p_ref)

    if callable(h):
        @LinearForm
        def qfilm(v, w):
            hval = h(w)                    # callable, no kwarg
            pval = w['p']
            dpdn = dot(grad(pval), w.n)
            qn   = -K * (hval**3) * (2.0 * pval * dpdn)
            return boundary_weight(w) * qn * v
        return qfilm

    if h is not None and np.isscalar(h):
        h_const = float(h)
        @LinearForm
        def qfilm(v, w):
            pval = w['p']
            dpdn = dot(grad(pval), w.n)
            qn   = -K * (h_const**3) * (2.0 * pval * dpdn)
            return boundary_weight(w) * qn * v
        return qfilm

    @LinearForm
    def qfilm(v, w):
        hval = w['h']                      # scalar/array/DiscreteField only
        pval = w['p']
        dpdn = dot(grad(pval), w.n)
        qn   = -K * (hval**3) * (2.0 * pval * dpdn)
        return boundary_weight(w) * qn * v
    return qfilm


# =========================
# Circular / Annular thrust
# (axisymmetric 1D in r)
# =========================

area_circular = lambda w: 2.0 * np.pi * w.x[0]      # dA = 2π r dr
n_axial_plate = lambda w: 1.0                       # flat plate normal

def load_circular_axial():
    return make_load_component(area_circular, n_axial_plate)

# Inner/outer rim flow (on a 1D r-mesh, boundary is two points; weight = circumference)
def qfilm_circular_rim(*, mu: float, p_ref: float):
    return make_qfilm_p2(lambda w: 2.0 * np.pi * w.x[0], mu=mu, p_ref=p_ref)


# =========================
# Rectangular (planar 2D)
# & “Infinite” (per-unit width)
# =========================

area_rect     = lambda w: 1.0
n_axial_rect  = lambda w: 1.0

def load_rect_axial():
    return make_load_component(area_rect, n_axial_rect)

def load_infinite_axial_per_width():
    # same integrand as rectangular; interpretation is per-unit width
    return make_load_component(area_rect, n_axial_rect)

# For planar meshes, skfem’s boundary integral already uses physical ds; weight = 1.0.
def qfilm_planar_edge(*, mu: float, p_ref: float):
    return make_qfilm_p2(lambda w: 1.0, mu=mu, p_ref=p_ref)


# =========================
# Journal (cylinder, φ–z)
# =========================

def journal_helpers(R: float):
    area = lambda w: R                        # dA = R dφ dz
    n_x  = lambda w: np.cos(w.x[0])          # e_r · e_x Vertical load
    n_y  = lambda w: np.sin(w.x[0])          # e_r · e_y
    n_r  = lambda w: 1.0                     # along e_r
    return area, n_x, n_y, n_r

def make_load_form_journal_x(R: float):
    area, n_x, *_ = journal_helpers(R)
    return make_load_component(area, n_x)

def make_load_form_journal_y(R: float):
    area, _, n_y, _ = journal_helpers(R)
    return make_load_component(area, n_y)

def make_load_form_journal_radial(R: float):
    area, *_, n_r = journal_helpers(R)
    return make_load_component(area, n_r)

# Boundary flow weights:
journal_weight_z_ends = lambda R: (lambda w: R)   # edges at z = const
journal_weight_phi_sides = lambda: (lambda w: 1.0)  # edges at φ = const (metric in grad)

def qfilm_journal_z_end(R: float, *, mu: float, p_ref: float):
    return make_qfilm_p2(journal_weight_z_ends(R), mu=mu, p_ref=p_ref)

def qfilm_journal_phi_side(*, mu: float, p_ref: float):
    return make_qfilm_p2(journal_weight_phi_sides(), mu=mu, p_ref=p_ref)


# =========================
# Cone (slant-r, φ), half-angle α
# α = angle between axis and surface generator
# =========================

def cone_helpers_slant(alpha: float):
    s, c = np.sin(alpha), np.cos(alpha)
    area             = lambda w: w.x[0] * s          # dA = r sinα dr dφ
    n_dot_axial      = lambda w: s                   # n · e_z
    n_dot_radial     = lambda w: c                   # n · e_r
    weight_r_const   = lambda w: w.x[0] * s          # ds on r=const: r sinα dφ
    weight_phi_const = lambda w: 1.0                 # ds on φ=const: dr
    return area, n_dot_axial, n_dot_radial, weight_r_const, weight_phi_const

def make_load_forms_cone_slant(alpha: float):
    area, n_z, n_r, _, _ = cone_helpers_slant(alpha)
    load_axial  = make_load_component(area, n_z)    # r sin^2α (p - pa)
    load_radial = make_load_component(area, n_r)    # r sinα cosα (p - pa)
    return load_axial, load_radial

# ↓ Add h and pass it through
def qfilm_cone_r_const_slant(alpha: float, *, mu: float, p_ref: float, h=None):
    _, _, _, w_rconst, _ = cone_helpers_slant(alpha)
    return make_qfilm_p2(w_rconst, mu=mu, p_ref=p_ref, h=h)

def qfilm_cone_phi_const_slant(alpha: float, *, mu: float, p_ref: float, h=None):
    _, _, _, _, w_phiconst = cone_helpers_slant(alpha)
    return make_qfilm_p2(w_phiconst, mu=mu, p_ref=p_ref, h=h)


# =========================
# Spherical cap (θ′, φ′), Rasnick mapping used in your code
# n = (cosθ′, sinθ′ sinφ′, -sinθ′ cosφ′)
# =========================
def map_rot_to_orig(thp, php):
    """
    Rasnick rotation (+90° about y): z' ≡ x(original)
      x =  R cos θ′
      y =  R sin θ′ sin φ′
      z = -R sin θ′ cos φ′
    => cos θ = z/R = -sin θ′ cos φ′
       φ = atan2(y, x) = atan2(sin θ′ sin φ′, cos θ′)
    """
    cos_th = -np.sin(thp) * np.cos(php)
    th = np.arccos(np.clip(cos_th, -1.0, 1.0))
    ph = np.arctan2(np.sin(thp) * np.sin(php), np.cos(thp))
    return th, ph

def sphere_helpers(R: float):
    area = lambda w: (R**2) * np.sin(w.x[0])              # dA = R^2 sinθ′ dθ′ dφ′
    n_x  = lambda w: np.cos(w.x[0])
    n_y  = lambda w: np.sin(w.x[0]) * np.sin(w.x[1])
    n_z  = lambda w: -np.sin(w.x[0]) * np.cos(w.x[1])
    w_theta_const = lambda w: np.sin(w.x[0])              # θ′-const edges (your earlier derivation)
    w_phi_const   = lambda w: 1.0                         # φ′-const edges
    return area, n_x, n_y, n_z, w_theta_const, w_phi_const

""" def make_load_forms_sphere_xyz(R: float):
    area, n_x, n_y, n_z, *_ = sphere_helpers(R)
    Wx = make_load_component(area, n_x)
    Wy = make_load_component(area, n_y)
    Wz = make_load_component(area, n_z)
    return Wx, Wy, Wz """
# --- Load components on a spherical cap (absolute or gauge) ---
def make_load_forms_sphere_xyz(R: float, *, subtract_pa: bool = True):
    area, n_x, n_y, n_z, *_ = sphere_helpers(R)
    Wx = make_load_component(area, n_x, subtract_pa=subtract_pa)
    Wy = make_load_component(area, n_y, subtract_pa=subtract_pa)
    Wz = make_load_component(area, n_z, subtract_pa=subtract_pa)
    return Wx, Wy, Wz
""" def qfilm_sphere_theta_edge(R: float, *, mu: float, p_ref: float):
    *_, w_th, _ = sphere_helpers(R)
    return make_qfilm_p2(w_th, mu=mu, p_ref=p_ref) """
def qfilm_sphere_theta_edge(R: float, *, mu: float, p_ref: float, h=None):
    
    """
    p²-form ambient-liter flow on θ′-constant edges of a spherical cap:
      Q = ∮ 6e4 * [ -(h^3/(24 μ p_ref)) * ∂_n(p^2) ] * (R sinθ′ dφ′)
        = ∮ 6e4 * [ -(h^3/(24 μ p_ref)) * 2 p (∇p·n) ] * (sinθ′) d(φ′)  (in param coords)
    Pass:
      - h=None to expect w['h'] (scalar/array/DiscreteField), OR
      - h=callable(w) to evaluate local thickness at the facet points.
    """
    *_, w_th, _ = sphere_helpers(R)
    return make_qfilm_p2(w_th, mu=mu, p_ref=p_ref, h=h)

""" def qfilm_sphere_phi_edge(R: float, *, mu: float, p_ref: float):
    *_, _, w_ph = sphere_helpers(R)
    return make_qfilm_p2(w_ph, mu=mu, p_ref=p_ref) """

def qfilm_sphere_phi_edge(R: float, *, mu: float, p_ref: float, h=None):

    *_, _, w_ph = sphere_helpers(R)
    return make_qfilm_p2(w_ph, mu=mu, p_ref=p_ref, h=h)
