"""
operator_dispatch.py — Dense/Sparse Backend Dispatch for SBP Operators
======================================================================

Provides a uniform interface for applying SBP operators (Dvc, Dcv, Pvc, Pcv)
regardless of whether the underlying implementation uses dense matmul or
sparse stencil application.

Usage:
    from operator_dispatch import make_operators

    op = make_operators(N, dx, backend='auto')

    # In RHS (batched over panels):
    dv1_dt = -g * op.Dvc.ax0(h_proj)   # einsum('ij,pjk->pik', Dvc, h_proj)
    dv2_dt = -g * op.Dvc.ax1(h_proj)   # einsum('pij,kj->pik', h_proj, Dvc)

    # In contravariant (single panel via vmap):
    v2_at_h = op.Pcv.ax1(v2)           # v2 @ Pcv.T
    cross_v1 = op.Pvc.ax0(cross_h)     # Pvc @ cross_h

    # Access raw matrix (always available):
    mat = op.Dvc.matrix

    # Access other operator data:
    Hv_diag = op.Hv_diag
    Hc_diag = op.Hc_diag
    l = op.l    # left extrapolation
    r = op.r    # right extrapolation
"""
import numpy as np

import jax
import jax.numpy as jnp

from .sbp_staggered_1d import sbp_42


# ============================================================
# Stencil extraction (self-contained, no external dependency)
# ============================================================

def _extract_stencil(mat):
    """
    Decompose dense SBP operator into interior stencil + boundary blocks.
    Returns numpy arrays.
    """
    rows, cols = mat.shape
    mid = rows // 2

    # Interior stencil from middle row
    nz_mid = np.nonzero(mat[mid])[0]
    offsets = nz_mid - mid
    weights = mat[mid, nz_mid]

    # Find largest contiguous block of interior rows (bit-exact match)
    is_interior = np.zeros(rows, dtype=bool)
    for i in range(rows):
        nz = np.nonzero(mat[i])[0]
        is_interior[i] = (
            len(nz) == len(offsets) and
            np.array_equal(nz - i, offsets) and
            np.array_equal(mat[i, nz], weights)
        )

    best_start, best_len, cur_start, cur_len = 0, 0, 0, 0
    for i in range(rows):
        if is_interior[i]:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
        else:
            cur_len = 0

    int_start = best_start
    int_end = best_start + best_len - 1
    n_top = int_start
    n_bot = rows - 1 - int_end

    # Top boundary dense block
    top_max_col = 0
    for i in range(n_top):
        nz = np.nonzero(mat[i])[0]
        if len(nz) > 0:
            top_max_col = max(top_max_col, nz[-1])
    top_ncols = top_max_col + 1
    top_block = np.zeros((n_top, top_ncols))
    for i in range(n_top):
        nz = np.nonzero(mat[i])[0]
        top_block[i, nz] = mat[i, nz]

    # Bottom boundary dense block
    bot_min_col = cols
    for i in range(int_end + 1, rows):
        nz = np.nonzero(mat[i])[0]
        if len(nz) > 0:
            bot_min_col = min(bot_min_col, nz[0])
    bot_ncols = cols - bot_min_col
    bot_block = np.zeros((n_bot, bot_ncols))
    for i in range(int_end + 1, rows):
        nz = np.nonzero(mat[i])[0]
        bot_block[i - (int_end + 1), nz - bot_min_col] = mat[i, nz]

    return {
        'weights': jnp.array(weights),
        'offsets': offsets,  # numpy int array
        'int_start': int_start,
        'int_end': int_end,
        'n_top': n_top,
        'n_bot': n_bot,
        'rows': rows,
        'cols': cols,
        'top_block': jnp.array(top_block),
        'top_ncols': top_ncols,
        'bot_block': jnp.array(bot_block),
        'bot_col0': bot_min_col,
    }


# ============================================================
# Sparse apply (slice-and-accumulate)
# ============================================================

def _sparse_ax0(s, x):
    """Apply operator along axis -2 using sparse stencil."""
    w = s['weights']
    offsets = s['offsets']
    int_start = s['int_start']
    n_int = s['int_end'] - int_start + 1

    interior = w[0] * x[..., int_start + offsets[0]:int_start + offsets[0] + n_int, :]
    for k in range(1, len(offsets)):
        start = int_start + offsets[k]
        interior = interior + w[k] * x[..., start:start + n_int, :]

    x_top = x[..., :s['top_ncols'], :]
    top = jnp.einsum('ij,...jk->...ik', s['top_block'], x_top)

    x_bot = x[..., s['bot_col0']:, :]
    bot = jnp.einsum('ij,...jk->...ik', s['bot_block'], x_bot)

    return jnp.concatenate([top, interior, bot], axis=-2)


def _sparse_ax1(s, x):
    """Apply operator along axis -1 using sparse stencil (X @ Op.T)."""
    w = s['weights']
    offsets = s['offsets']
    int_start = s['int_start']
    n_int = s['int_end'] - int_start + 1

    interior = w[0] * x[..., :, int_start + offsets[0]:int_start + offsets[0] + n_int]
    for k in range(1, len(offsets)):
        start = int_start + offsets[k]
        interior = interior + w[k] * x[..., :, start:start + n_int]

    x_top = x[..., :, :s['top_ncols']]
    top = jnp.einsum('...ij,kj->...ik', x_top, s['top_block'])

    x_bot = x[..., :, s['bot_col0']:]
    bot = jnp.einsum('...ij,kj->...ik', x_bot, s['bot_block'])

    return jnp.concatenate([top, interior, bot], axis=-1)


# ============================================================
# Dense apply
# ============================================================

def _dense_ax0(mat, x):
    """Apply dense matrix along axis -2: Op @ X[..., :, :]."""
    return jnp.einsum('ij,...jk->...ik', mat, x)


def _dense_ax1(mat, x):
    """Apply dense matrix along axis -1: X[..., :, :] @ Op.T."""
    return jnp.einsum('...ij,kj->...ik', x, mat)


# ============================================================
# Single operator wrapper
# ============================================================

class OpApply:
    """
    Wraps a single SBP operator with ax0/ax1 dispatch.

    op.ax0(X)  — apply Op along second-to-last axis  (Op @ X)
    op.ax1(X)  — apply Op along last axis             (X @ Op.T)
    op.matrix  — raw dense matrix (always available)
    """
    def __init__(self, name, mat_jax, backend):
        self.name = name
        self.matrix = mat_jax
        self._backend = backend

        if backend == 'sparse':
            mat_np = np.array(mat_jax)
            stencil = _extract_stencil(mat_np)
            self._stencil = stencil
            self.ax0 = lambda x, s=stencil: _sparse_ax0(s, x)
            self.ax1 = lambda x, s=stencil: _sparse_ax1(s, x)
        else:
            self.ax0 = lambda x, m=mat_jax: _dense_ax0(m, x)
            self.ax1 = lambda x, m=mat_jax: _dense_ax1(m, x)
            self._stencil = None

    @property
    def backend(self):
        return self._backend

    def __repr__(self):
        r, c = self.matrix.shape
        return f"OpApply({self.name}, {r}×{c}, {self._backend})"


# ============================================================
# Hardware detection
# ============================================================

def detect_backend():
    """
    Choose backend based on hardware.

    Returns 'dense' or 'sparse'.

    Logic:
      - GPU with strong FP64 (A30, A100, H100, V100): dense
        Tensor cores handle the 99% wasted ops faster than sparse indexing.
      - GPU with weak FP64 (L40S, T4, consumer): sparse
        FP64 throughput too low to brute-force the waste.
      - CPU: dense
        AMX/AVX matmul nearly as fast, and simpler.
      - TPU: dense
        Systolic array optimized for dense matmul.
    """
    dev = jax.devices()[0]
    platform = dev.platform.lower()

    if platform == 'gpu':
        name = getattr(dev, 'device_kind', '').lower()
        # GPUs with good FP64 tensor cores
        strong_fp64 = ['a30', 'a100', 'h100', 'h200', 'v100', 'b200', 'gh200']
        for model in strong_fp64:
            if model in name:
                return 'dense'
        # Default GPU (L40S, T4, RTX, etc): sparse may help
        return 'sparse'
    elif platform == 'tpu':
        return 'dense'
    else:
        # CPU: dense is fine (AVX/AMX matmul)
        return 'dense'


# ============================================================
# Main factory
# ============================================================

class OperatorSet:
    """
    Complete set of SBP operators with backend dispatch.

    Attributes:
        Dvc, Dcv, Pvc, Pcv : OpApply — operator wrappers with ax0/ax1
        Hv, Hc             : jnp arrays — quadrature matrices
        Hv_diag, Hc_diag   : jnp arrays — diagonal of quadrature matrices
        l, r               : jnp arrays — left/right extrapolation vectors
        N, dx              : int, float — grid parameters
        backend             : str — 'dense' or 'sparse'
    """
    def __init__(self, N, dx, backend='auto'):
        self.N = N
        self.dx = dx

        # Build raw operators
        ops = sbp_42(N, dx)

        # Choose backend
        if backend == 'auto':
            self.backend = detect_backend()
        else:
            self.backend = backend

        # Wrap each operator
        self.Dvc = OpApply('Dvc', jnp.array(ops.Dvc), self.backend)
        self.Dcv = OpApply('Dcv', jnp.array(ops.Dcv), self.backend)
        self.Pvc = OpApply('Pvc', jnp.array(ops.Pvc), self.backend)
        self.Pcv = OpApply('Pcv', jnp.array(ops.Pcv), self.backend)

        # Quadrature and extrapolation (always dense — they're small)
        self.Hv = jnp.array(ops.Hv)
        self.Hc = jnp.array(ops.Hc)
        self.Hv_diag = jnp.diag(self.Hv)
        self.Hc_diag = jnp.diag(self.Hc)
        self.l = jnp.array(ops.l)
        self.r = jnp.array(ops.r)

    def __repr__(self):
        return (f"OperatorSet(N={self.N}, backend='{self.backend}', "
                f"Dvc={self.Dvc.matrix.shape}, Dcv={self.Dcv.matrix.shape})")


def make_operators(N, dx, backend='auto'):
    """
    Factory function for SBP operator set.

    Parameters
    ----------
    N : int — cells per panel edge
    dx : float — grid spacing
    backend : str — 'dense', 'sparse', or 'auto' (detect from hardware)

    Returns
    -------
    OperatorSet with .Dvc, .Dcv, .Pvc, .Pcv (each with .ax0, .ax1, .matrix)
    """
    return OperatorSet(N, dx, backend=backend)
