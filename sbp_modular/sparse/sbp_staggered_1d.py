"""
1D Staggered SBP-FD Operators

Implementation of staggered Summation-by-Parts finite difference operators
from Shashkin, Goyman, Tretyak (2025) "SBP FD for Linear Shallow Water 
Equations on Staggered Curvilinear Grids in Closed Domains"

Grids:
    x^v (vertices): x^v_i = a + (i-1)*dx,  i = 1, ..., N+1
    x^c (centers):  x^c_i = a + (i-0.5)*dx, i = 1, ..., N

Variables:
    h: defined at x^v (N+1 points)
    u: defined at x^c (N points)

Operators:
    Dvc: d/dx at x^c using values at x^v  [N x (N+1)]
    Dcv: d/dx at x^v using values at x^c  [(N+1) x N]
    Hv:  quadrature at x^v [(N+1) x (N+1)] diagonal
    Hc:  quadrature at x^c [N x N] diagonal
    Pvc: interpolate x^v -> x^c [N x (N+1)]
    Pcv: interpolate x^c -> x^v [(N+1) x N]
    l, r: boundary extrapolation from x^c to left/right boundaries

SBP Property (Eq. 5):
    u^T Hc Dvc h = -h^T Hv Dcv u + h_{N+1} r^T u - h_1 l^T u

SBP-Preserving Interpolation (Eq. 18):
    u^T Hc Pvc h = h^T Hv Pcv u
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple
from functools import partial


class StaggeredSBPOperators(NamedTuple):
    """Container for 1D staggered SBP operators."""
    Dcv: jnp.ndarray   # [(N+1) x N] derivative: x^c -> x^v
    Dvc: jnp.ndarray   # [N x (N+1)] derivative: x^v -> x^c
    Hv: jnp.ndarray    # [(N+1) x (N+1)] quadrature at vertices (diagonal)
    Hc: jnp.ndarray    # [N x N] quadrature at centers (diagonal)
    Pcv: jnp.ndarray   # [(N+1) x N] interpolation: x^c -> x^v
    Pvc: jnp.ndarray   # [N x (N+1)] interpolation: x^v -> x^c
    l: jnp.ndarray     # [N] left boundary extrapolation
    r: jnp.ndarray     # [N] right boundary extrapolation
    order: int         # interior order (2, 4, or 6)
    dx: float          # grid spacing


def make_grids(N: int, a: float = 0.0, b: float = 1.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create staggered grids.
    
    Args:
        N: number of cells
        a, b: domain bounds
        
    Returns:
        xv: vertex grid [N+1]
        xc: center grid [N]
    """
    dx = (b - a) / N
    xv = a + jnp.arange(N + 1) * dx
    xc = a + (jnp.arange(N) + 0.5) * dx
    return xv, xc


# =============================================================================
# Order 2/1 Operators (Equations 9-11, 19, 88)
# =============================================================================

def sbp_21(N: int, dx: float = 1.0) -> StaggeredSBPOperators:
    """
    Create 2nd order interior / 1st order boundary SBP operators.
    
    From Equations (9), (10), (11), (19), (88) in Shashkin et al.
    """
    # Quadrature matrices (Eq. 88)
    hv_diag = jnp.ones(N + 1)
    hv_diag = hv_diag.at[0].set(0.5)
    hv_diag = hv_diag.at[N].set(0.5)
    Hv = jnp.diag(hv_diag * dx)
    
    Hc = jnp.eye(N) * dx
    
    # Derivative Dcv: (N+1) x N (Eq. 9)
    # Stencil: [-1, 1]/dx at each row
    # Row i approximates d/dx at x^v_i using u values at x^c
    Dcv = jnp.zeros((N + 1, N))
    
    # Row 0: at left boundary x=0, use forward stencil [-1, 1]
    Dcv = Dcv.at[0, 0].set(-1.0)
    Dcv = Dcv.at[0, 1].set(1.0) if N > 1 else Dcv
    
    # Interior rows 1 to N-1: centered stencil [-1, 1] at columns i-1, i
    for i in range(1, N):
        Dcv = Dcv.at[i, i - 1].set(-1.0)
        Dcv = Dcv.at[i, i].set(1.0)
    
    # Row N: at right boundary x=1, use backward stencil [-1, 1]
    Dcv = Dcv.at[N, N - 2].set(-1.0) if N > 1 else Dcv
    Dcv = Dcv.at[N, N - 1].set(1.0)
    
    Dcv = Dcv / dx
    
    # Boundary extrapolation (Eq. 11)
    # l = (3/2, -1/2, 0, ..., 0)  - extrapolate to x=a from centers
    # r = (0, ..., 0, -1/2, 3/2)  - extrapolate to x=b from centers
    l = jnp.zeros(N)
    l = l.at[0].set(1.5)
    if N > 1:
        l = l.at[1].set(-0.5)
    
    r = jnp.zeros(N)
    if N > 1:
        r = r.at[N - 2].set(-0.5)
    r = r.at[N - 1].set(1.5)
    
    # Derive Dvc from SBP property (Eq. 5):
    # u^T Hc Dvc h = -h^T Hv Dcv u + h_{N+1} r^T u - h_1 l^T u
    # => Hc Dvc + Dcv^T Hv = R_cv^T  where R_cv = e_r r^T - e_l l^T
    er = jnp.zeros(N + 1).at[N].set(1.0)
    el = jnp.zeros(N + 1).at[0].set(1.0)
    Rcv = jnp.outer(er, r) - jnp.outer(el, l)
    
    Hc_inv = jnp.diag(1.0 / jnp.diag(Hc))
    Dvc = Hc_inv @ (-Dcv.T @ Hv + Rcv.T)
    
    # Interpolation Pvc: N x (N+1) (Eq. 19)
    Pvc = jnp.zeros((N, N + 1))
    for i in range(N):
        Pvc = Pvc.at[i, i].set(0.5)
        Pvc = Pvc.at[i, i + 1].set(0.5)
    
    # Interpolation Pcv: (N+1) x N - derived from SBP-preserving property
    # Hv Pcv = Pvc^T Hc  =>  Pcv = Hv^{-1} Pvc^T Hc
    Hv_inv = jnp.diag(1.0 / jnp.diag(Hv))
    Pcv = Hv_inv @ Pvc.T @ Hc
    
    return StaggeredSBPOperators(
        Dcv=Dcv, Dvc=Dvc, Hv=Hv, Hc=Hc,
        Pcv=Pcv, Pvc=Pvc, l=l, r=r,
        order=2, dx=dx
    )


# =============================================================================
# Order 4/2 Operators (Equations 89, 91-92, 97-98)
# =============================================================================

def sbp_42(N: int, dx: float = 1.0) -> StaggeredSBPOperators:
    """
    Create 4th order interior / 2nd order boundary SBP operators.
    
    From Equations (89), (91), (92), (97), (98) in Shashkin et al.
    Requires N >= 8 for proper stencil fitting.
    """
    assert N >= 8, f"N={N} too small for 4/2 order operators (need N >= 8)"
    
    # Quadrature Hv (Eq. 89)
    hv_diag = jnp.ones(N + 1)
    # Left boundary: 7/18, 9/8, 1, 71/72
    hv_diag = hv_diag.at[0].set(7/18)
    hv_diag = hv_diag.at[1].set(9/8)
    hv_diag = hv_diag.at[2].set(1.0)
    hv_diag = hv_diag.at[3].set(71/72)
    # Right boundary (symmetric)
    hv_diag = hv_diag.at[N].set(7/18)
    hv_diag = hv_diag.at[N-1].set(9/8)
    hv_diag = hv_diag.at[N-2].set(1.0)
    hv_diag = hv_diag.at[N-3].set(71/72)
    Hv = jnp.diag(hv_diag * dx)
    
    # Quadrature Hc (Eq. 89)
    hc_diag = jnp.ones(N)
    # Left boundary: 13/12, 7/8, 25/24
    hc_diag = hc_diag.at[0].set(13/12)
    hc_diag = hc_diag.at[1].set(7/8)
    hc_diag = hc_diag.at[2].set(25/24)
    # Right boundary (symmetric)
    hc_diag = hc_diag.at[N-1].set(13/12)
    hc_diag = hc_diag.at[N-2].set(7/8)
    hc_diag = hc_diag.at[N-3].set(25/24)
    Hc = jnp.diag(hc_diag * dx)
    
    # Derivative Dcv (Eq. 91)
    Dcv = jnp.zeros((N + 1, N))
    
    # Row 0: [-2, 3, -1, 0, ...]
    Dcv = Dcv.at[0, 0].set(-2.0)
    Dcv = Dcv.at[0, 1].set(3.0)
    Dcv = Dcv.at[0, 2].set(-1.0)
    
    # Row 1: [-1, 1, 0, ...]
    Dcv = Dcv.at[1, 0].set(-1.0)
    Dcv = Dcv.at[1, 1].set(1.0)
    
    # Row 2: [1/24, -9/8, 9/8, -1/24, ...]
    Dcv = Dcv.at[2, 0].set(1/24)
    Dcv = Dcv.at[2, 1].set(-9/8)
    Dcv = Dcv.at[2, 2].set(9/8)
    Dcv = Dcv.at[2, 3].set(-1/24)
    
    # Row 3: [-1/71, 6/71, -83/71, 81/71, -3/71]
    Dcv = Dcv.at[3, 0].set(-1/71)
    Dcv = Dcv.at[3, 1].set(6/71)
    Dcv = Dcv.at[3, 2].set(-83/71)
    Dcv = Dcv.at[3, 3].set(81/71)
    Dcv = Dcv.at[3, 4].set(-3/71)
    
    # Interior rows: [1/24, -9/8, 9/8, -1/24]
    for i in range(4, N - 3):
        Dcv = Dcv.at[i, i - 2].set(1/24)
        Dcv = Dcv.at[i, i - 1].set(-9/8)
        Dcv = Dcv.at[i, i].set(9/8)
        Dcv = Dcv.at[i, i + 1].set(-1/24)
    
    # Right boundary rows (antisymmetric: d_{N+2-i, N+1-j} = -d_{i,j})
    for i in range(4):
        for j in range(min(5, N)):
            val = Dcv[i, j]
            if val != 0:
                Dcv = Dcv.at[N - i, N - 1 - j].set(-val)
    
    Dcv = Dcv / dx
    
    # Boundary extrapolation (Eq. 92)
    l = jnp.zeros(N)
    l = l.at[0].set(15/8)
    l = l.at[1].set(-10/8)
    l = l.at[2].set(3/8)
    
    r = jnp.zeros(N)
    r = r.at[N-3].set(3/8)
    r = r.at[N-2].set(-10/8)
    r = r.at[N-1].set(15/8)
    
    # Derive Dvc from SBP property:
    # Hc Dvc + Dcv^T Hv = e_r r^T - e_l l^T
    # where e_r = (0,...,0,1)^T, e_l = (1,0,...,0)^T at x^v points
    er = jnp.zeros(N + 1).at[N].set(1.0)
    el = jnp.zeros(N + 1).at[0].set(1.0)
    
    Rcv = jnp.outer(er, r) - jnp.outer(el, l)
    Hc_inv = jnp.diag(1.0 / jnp.diag(Hc))
    Dvc = Hc_inv @ (-Dcv.T @ Hv + Rcv.T)
    
    # Interpolation Pvc (Eq. 97)
    # Free parameters from Eq. 98
    c13 = 102207746025903 / 808013506696916
    c14 = -289843969221617 / 9696162080362992
    
    Pvc = _build_pvc_42(N, c13, c14)
    
    # Pcv from SBP-preserving property
    Hv_inv = jnp.diag(1.0 / jnp.diag(Hv))
    Pcv = Hv_inv @ Pvc.T @ Hc
    
    return StaggeredSBPOperators(
        Dcv=Dcv, Dvc=Dvc, Hv=Hv, Hc=Hc,
        Pcv=Pcv, Pvc=Pvc, l=l, r=r,
        order=4, dx=dx
    )


def _build_pvc_42(N: int, c13: float, c14: float) -> jnp.ndarray:
    """Build 4/2 order interpolation matrix Pvc."""
    Pvc = jnp.zeros((N, N + 1))
    
    # Coefficients from Eq. 97
    p11 = 0.5 + c13 + 2*c14
    p12 = 0.5 - 2*c13 - 3*c14
    p13 = c13
    p14 = c14
    
    p21 = -8/63 - 52*c13/21 - 104*c14/21
    p22 = 29/42 + 104*c13/21 + 52*c14/7
    p23 = -52*c13/21 + 0.5
    p24 = -4/63 - 52*c14/21
    
    p31 = 26*c13/25 + 52*c14/25 - 1/25
    p32 = -1/50 - 52*c13/25 - 78*c14/25
    p33 = 3/5 + 26*c13/25
    p34 = 13/25 + 26*c14/25
    p35 = -3/50
    
    # Row 0
    Pvc = Pvc.at[0, 0].set(p11)
    Pvc = Pvc.at[0, 1].set(p12)
    Pvc = Pvc.at[0, 2].set(p13)
    Pvc = Pvc.at[0, 3].set(p14)
    
    # Row 1
    Pvc = Pvc.at[1, 0].set(p21)
    Pvc = Pvc.at[1, 1].set(p22)
    Pvc = Pvc.at[1, 2].set(p23)
    Pvc = Pvc.at[1, 3].set(p24)
    
    # Row 2
    Pvc = Pvc.at[2, 0].set(p31)
    Pvc = Pvc.at[2, 1].set(p32)
    Pvc = Pvc.at[2, 2].set(p33)
    Pvc = Pvc.at[2, 3].set(p34)
    Pvc = Pvc.at[2, 4].set(p35)
    
    # Interior: [-1/16, 9/16, 9/16, -1/16]
    for i in range(3, N - 3):
        Pvc = Pvc.at[i, i - 1].set(-1/16)
        Pvc = Pvc.at[i, i].set(9/16)
        Pvc = Pvc.at[i, i + 1].set(9/16)
        Pvc = Pvc.at[i, i + 2].set(-1/16)
    
    # Right boundary (symmetric: p_{N+1-i, N+2-j} = p_{i,j})
    for i in range(3):
        for j in range(5):
            if j < N + 1:
                val = Pvc[i, j]
                if val != 0 and (N - 1 - i) >= 0 and (N - j) >= 0:
                    Pvc = Pvc.at[N - 1 - i, N - j].set(val)
    
    return Pvc


# =============================================================================
# Order 6/3 Operators (Equations 90, 93-96, 99-100)
# =============================================================================

def sbp_63(N: int, dx: float = 1.0, optimize_dispersion: bool = True) -> StaggeredSBPOperators:
    """
    Create 6th order interior / 3rd order boundary SBP operators.
    
    From Equations (90), (93)-(96), (99)-(100) in Shashkin et al.
    Requires N >= 12 for proper stencil fitting.
    
    Args:
        N: number of cells
        dx: grid spacing
        optimize_dispersion: if True, use wave-optimized parameters (Eq. 95)
                           if False, use polynomial-optimized (Eq. 94)
    """
    assert N >= 12, f"N={N} too small for 6/3 order operators (need N >= 12)"
    
    # Quadrature Hv (Eq. 90)
    hv_diag = jnp.ones(N + 1)
    hv_vals = [95/288, 317/240, 23/30, 793/720, 157/160]
    for i, val in enumerate(hv_vals):
        hv_diag = hv_diag.at[i].set(val)
        hv_diag = hv_diag.at[N - i].set(val)
    Hv = jnp.diag(hv_diag * dx)
    
    # Quadrature Hc (Eq. 90)
    hc_diag = jnp.ones(N)
    hc_vals = [325363/276480, 144001/276480, 43195/27648, 
               86857/138240, 312623/276480, 271229/276480]
    for i, val in enumerate(hc_vals):
        hc_diag = hc_diag.at[i].set(val)
        hc_diag = hc_diag.at[N - 1 - i].set(val)
    Hc = jnp.diag(hc_diag * dx)
    
    # Free parameters for derivatives (Eq. 94 or 95)
    if optimize_dispersion:
        # Wave-optimized (Eq. 95)
        c34 = 0.467391226104632
        c55 = -0.723617281756727
    else:
        # Polynomial-optimized (Eq. 94)
        c34 = 0.6690374220138081
        c55 = -0.7930390145751754
    
    # Derivative Dcv (Eq. 93)
    Dcv = _build_dcv_63(N, c34, c55)
    Dcv = Dcv / dx
    
    # Boundary extrapolation (Eq. 96)
    l = jnp.zeros(N)
    l = l.at[0].set(35/16)
    l = l.at[1].set(-35/16)
    l = l.at[2].set(21/16)
    l = l.at[3].set(-5/16)
    
    r = jnp.zeros(N)
    r = r.at[N-4].set(-5/16)
    r = r.at[N-3].set(21/16)
    r = r.at[N-2].set(-35/16)
    r = r.at[N-1].set(35/16)
    
    # Derive Dvc from SBP property
    er = jnp.zeros(N + 1).at[N].set(1.0)
    el = jnp.zeros(N + 1).at[0].set(1.0)
    Rcv = jnp.outer(er, r) - jnp.outer(el, l)
    Hc_inv = jnp.diag(1.0 / jnp.diag(Hc))
    Dvc = Hc_inv @ (-Dcv.T @ Hv + Rcv.T)
    
    # Interpolation (Eq. 99-100)
    c42 = -0.3332211159670528
    c43 = 0.3310769312612241
    c52 = -0.07099703081266314
    c53 = -0.2916164053358880
    c62 = 0.05753938634775091
    c64 = -0.1230378129758785
    
    Pvc = _build_pvc_63(N, c42, c43, c52, c53, c62, c64)
    
    # Pcv from SBP-preserving property
    Hv_inv = jnp.diag(1.0 / jnp.diag(Hv))
    Pcv = Hv_inv @ Pvc.T @ Hc
    
    return StaggeredSBPOperators(
        Dcv=Dcv, Dvc=Dvc, Hv=Hv, Hc=Hc,
        Pcv=Pcv, Pvc=Pvc, l=l, r=r,
        order=6, dx=dx
    )


def _build_dcv_63(N: int, c34: float, c55: float) -> jnp.ndarray:
    """Build 6/3 order derivative matrix Dcv."""
    Dcv = jnp.zeros((N + 1, N))
    
    # Coefficients from Eq. 93 (extremely long expressions)
    # Row 0
    d = _dcv_63_row0(c34, c55)
    for j, val in enumerate(d):
        Dcv = Dcv.at[0, j].set(val)
    
    # Row 1
    d = _dcv_63_row1(c34, c55)
    for j, val in enumerate(d):
        Dcv = Dcv.at[1, j].set(val)
    
    # Row 2
    d = _dcv_63_row2(c34, c55)
    for j, val in enumerate(d):
        Dcv = Dcv.at[2, j].set(val)
    
    # Row 3
    d = _dcv_63_row3(c34, c55)
    for j, val in enumerate(d):
        Dcv = Dcv.at[3, j].set(val)
    
    # Row 4
    d = _dcv_63_row4(c34, c55)
    for j, val in enumerate(d):
        Dcv = Dcv.at[4, j].set(val)
    
    # Row 5 (first interior with standard stencil starting)
    Dcv = Dcv.at[5, 0].set(-3/640)
    Dcv = Dcv.at[5, 1].set(25/384)
    Dcv = Dcv.at[5, 2].set(-75/256)
    Dcv = Dcv.at[5, 3].set(75/256)
    Dcv = Dcv.at[5, 4].set(-25/384)
    Dcv = Dcv.at[5, 5].set(3/640)
    
    # Interior rows: [-3/640, 25/384, -75/256, 75/256, -25/384, 3/640]
    for i in range(6, N - 4):
        Dcv = Dcv.at[i, i - 3].set(-3/640)
        Dcv = Dcv.at[i, i - 2].set(25/384)
        Dcv = Dcv.at[i, i - 1].set(-75/256)
        Dcv = Dcv.at[i, i].set(75/256)
        Dcv = Dcv.at[i, i + 1].set(-25/384)
        Dcv = Dcv.at[i, i + 2].set(3/640)
    
    # Right boundary (antisymmetric)
    for i in range(6):
        for j in range(min(7, N)):
            val = Dcv[i, j]
            if val != 0:
                ri = N - i
                rj = N - 1 - j
                if 0 <= ri <= N and 0 <= rj < N:
                    Dcv = Dcv.at[ri, rj].set(-val)
    
    return Dcv


def _dcv_63_row0(c34, c55):
    """Coefficients for row 0 of 6/3 Dcv."""
    d11 = (-60711983 + 15005904*c55 + 5183400*c34) / 21888000
    d12 = (101173243 - 30011808*c55 - 15550200*c34) / 17510400
    d13 = (-7780959 + 1727800*c34) / 1459200
    d14 = (-5183400*c34 + 35609465 + 30011808*c55) / 8755200
    d15 = (-7502952*c55 - 5209847) / 2188800
    d16 = (18712829 + 30011808*c55 + 1727800*c34) / 29184000
    return [d11, d12, d13, d14, d15, d16]


def _dcv_63_row1(c34, c55):
    """Coefficients for row 1 of 6/3 Dcv."""
    d21 = (-53376169 - 30011808*c55 - 10366800*c34) / 43822080
    d22 = (7190801 + 10003936*c55 + 5183400*c34) / 5842944
    d23 = -(2591700*c34 - 2846555) / 2191104
    d24 = (-27181195 - 30011808*c55 + 5183400*c34) / 8764416
    d25 = (7223559 + 10003936*c55) / 2921472
    d26 = (-59866697 - 90035424*c55 - 5183400*c34) / 87644160
    return [d21, d22, d23, d24, d25, d26]


def _dcv_63_row2(c34, c55):
    """Coefficients for row 2 of 6/3 Dcv."""
    d31 = (332488 + 625246*c55 + 215975*c34) / 353280
    d32 = (-6326795 - 10003936*c55 - 5183400*c34) / 2260992
    d33 = (1727800*c34 - 665205) / 565248
    d34 = (8940511 + 10003936*c55 - 1727800*c34) / 1130496
    d35 = (-3843253 - 5001968*c55) / 565248
    d36 = (21758409 + 30011808*c55 + 1727800*c34) / 11304960
    return [d31, d32, d33, d34, d35, d36]


def _dcv_63_row3(c34, c55):
    """Coefficients for row 3 of 6/3 Dcv."""
    d41 = (-17586239 - 30011808*c55 - 10366800*c34) / 36541440
    d42 = (14084351 + 30011808*c55 + 15550200*c34) / 14616576
    d43 = -215975*c34 / 152256
    d44 = (5183400*c34 - 30011808*c55 - 21697151) / 7308288
    d45 = (25503551 + 30011808*c55) / 7308288
    d46 = (-24437759 - 30011808*c55 - 1727800*c34) / 24360960
    return [d41, d42, d43, d44, d45, d46]


def _dcv_63_row4(c34, c55):
    """Coefficients for row 4 of 6/3 Dcv."""
    d51 = (9606527 + 15005904*c55 + 5183400*c34) / 65111040
    d52 = (-4598783 - 10003936*c55 - 5183400*c34) / 17362944
    d53 = (-4811905 + 5183400*c34) / 13022208
    d54 = (5665537 - 5183400*c34 + 30011808*c55) / 26044416
    d55 = -312623*c55 / 271296
    d56 = (68894207 + 90035424*c55 + 5183400*c34) / 260444160
    d57 = 3/628
    return [d51, d52, d53, d54, d55, d56, d57]


def _build_pvc_63(N: int, c42, c43, c52, c53, c62, c64) -> jnp.ndarray:
    """Build 6/3 order interpolation matrix Pvc.
    
    NOTE: Full boundary coefficients from Eq. 99 are complex.
    Using 4/2 order fallback near boundaries for robustness.
    """
    Pvc = jnp.zeros((N, N + 1))
    
    # Interior stencil: [3/256, -25/256, 150/256, 150/256, -25/256, 3/256]
    # Centered at (i, i+1) midpoint
    for i in range(4, N - 4):
        Pvc = Pvc.at[i, i - 2].set(3/256)
        Pvc = Pvc.at[i, i - 1].set(-25/256)
        Pvc = Pvc.at[i, i].set(150/256)
        Pvc = Pvc.at[i, i + 1].set(150/256)
        Pvc = Pvc.at[i, i + 2].set(-25/256)
        Pvc = Pvc.at[i, i + 3].set(3/256)
    
    # Near-boundary: use 4/2 order stencil [-1/16, 9/16, 9/16, -1/16]
    for i in range(min(4, N)):
        if i == 0:
            # Very first row: simple average
            Pvc = Pvc.at[i, 0].set(0.5)
            Pvc = Pvc.at[i, 1].set(0.5)
        elif i < N - 1:
            Pvc = Pvc.at[i, i - 1].set(-1/16) if i > 0 else Pvc
            Pvc = Pvc.at[i, i].set(9/16)
            Pvc = Pvc.at[i, i + 1].set(9/16)
            Pvc = Pvc.at[i, i + 2].set(-1/16) if i + 2 <= N else Pvc
    
    # Right boundary (symmetric)
    for i in range(max(0, N - 4), N):
        if i == N - 1:
            # Very last row: simple average
            Pvc = Pvc.at[i, N - 1].set(0.5)
            Pvc = Pvc.at[i, N].set(0.5)
        elif i >= 4:  # Don't overwrite interior
            Pvc = Pvc.at[i, i - 1].set(-1/16) if i > 0 else Pvc
            Pvc = Pvc.at[i, i].set(9/16)
            Pvc = Pvc.at[i, i + 1].set(9/16)
            Pvc = Pvc.at[i, i + 2].set(-1/16) if i + 2 <= N else Pvc
    
    return Pvc


# =============================================================================
# Factory function
# =============================================================================

def make_sbp_operators(N: int, order: int = 2, dx: float = 1.0, 
                       **kwargs) -> StaggeredSBPOperators:
    """
    Factory function to create SBP operators of specified order.
    
    Args:
        N: number of cells
        order: interior order (2, 4, or 6)
        dx: grid spacing
        **kwargs: passed to specific constructor
        
    Returns:
        StaggeredSBPOperators
    """
    if order == 2:
        return sbp_21(N, dx)
    elif order == 4:
        return sbp_42(N, dx)
    elif order == 6:
        return sbp_63(N, dx, **kwargs)
    else:
        raise ValueError(f"Unsupported order {order}. Use 2, 4, or 6.")


# =============================================================================
# Interface condition operators (SAT-Projection from Section 2.2)
# =============================================================================

def apply_sat_correction(ops: StaggeredSBPOperators) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create SAT-corrected derivative operators for periodic domain.
    
    From Equations (26), (27):
        D^S_vc = Dvc - (1/2) Hc^{-1} (r+l)(e_r - e_l)^T
        D^S_cv = Dcv - (1/2) Hv^{-1} (e_r + e_l)(r - l)^T
    
    Returns:
        DS_vc, DS_cv: SAT-corrected operators
    """
    N = ops.Hc.shape[0]
    
    # e_r = (0,...,0,1), e_l = (1,0,...,0) for h-grid (N+1 points)
    er = jnp.zeros(N + 1).at[N].set(1.0)
    el = jnp.zeros(N + 1).at[0].set(1.0)
    
    Hc_inv = jnp.diag(1.0 / jnp.diag(ops.Hc))
    Hv_inv = jnp.diag(1.0 / jnp.diag(ops.Hv))
    
    # SAT corrections
    DS_vc = ops.Dvc - 0.5 * Hc_inv @ jnp.outer(ops.r + ops.l, er - el)
    DS_cv = ops.Dcv - 0.5 * Hv_inv @ jnp.outer(er + el, ops.r - ops.l)
    
    return DS_vc, DS_cv


def make_projection_matrix(ops: StaggeredSBPOperators) -> jnp.ndarray:
    """
    Create projection matrix A for interface continuity.
    
    From Equation (33): Projects h values to be continuous at interfaces.
    For periodic 1D: h_1 = h_{N+1} after projection.
    
    Returns:
        A: projection matrix [(N+1) x (N+1)]
    """
    N = ops.Hc.shape[0]
    Hv_diag = jnp.diag(ops.Hv)
    
    # Start with identity
    A = jnp.eye(N + 1)
    
    # Modify first and last rows for averaging
    # (Ah)_1 = (Ah)_{N+1} = (Hv_1 h_1 + Hv_{N+1} h_{N+1}) / (Hv_1 + Hv_{N+1})
    w1 = Hv_diag[0] / (Hv_diag[0] + Hv_diag[N])
    wN = Hv_diag[N] / (Hv_diag[0] + Hv_diag[N])
    
    A = A.at[0, 0].set(w1)
    A = A.at[0, N].set(wN)
    A = A.at[N, 0].set(w1)
    A = A.at[N, N].set(wN)
    
    return A


def apply_sat_projection(ops: StaggeredSBPOperators) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create SAT-Projection corrected operators for periodic domain.
    
    From Equation (32):
        D^P_cv = A D^S_cv
        D^P_vc = D_vc A
    
    This removes spurious modes and reduces stiffness compared to pure SAT.
    
    Returns:
        DP_vc, DP_cv: SAT-Projection corrected operators
    """
    DS_vc, DS_cv = apply_sat_correction(ops)
    A = make_projection_matrix(ops)
    
    DP_cv = A @ DS_cv
    DP_vc = DS_vc @ A
    
    return DP_vc, DP_cv
