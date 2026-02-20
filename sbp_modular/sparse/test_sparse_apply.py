"""
test_sparse_apply.py — Compare sparse apply techniques against dense matmul
============================================================================

Two strategies for applying the extracted stencil operators:

  Strategy A (slice): Interior via shifted-slice accumulation, boundary via
    small dense matvec. Simple, no library dependency.

  Strategy B (conv): Interior via jax.lax.conv_general_dilated (1D convolution),
    boundary same as A. Lets XLA fuse the stencil with surrounding ops.

Both must match dense einsum('ij,pjk->pik', Op, X) to machine precision.

Usage:
    python test_sparse_apply.py
"""
import sys
import os
import time

project_dir = '/mnt/project'
if os.path.isdir(project_dir):
    sys.path.insert(0, project_dir)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from sbp_staggered_1d import sbp_42
from test_stencil_extraction import extract_stencil


# ============================================================
# Prepare stencil for JAX (convert boundary to dense submatrices)
# ============================================================

def prepare_stencil_jax(stencil):
    """
    Convert extracted stencil to JAX-friendly form with dense boundary blocks.

    Returns dict with all arrays as jnp, plus:
        'top_block': (n_top, top_ncols) dense matrix
        'bot_block': (n_bot, bot_ncols) dense matrix
        'top_ncols': int
        'bot_col0':  int — first column of bottom block
    """
    rows = stencil['rows']
    cols = stencil['cols']
    n_top = stencil['n_top']
    n_bot = stencil['n_bot']

    # Build top dense block: rows 0..n_top-1
    top_max_col = 0
    for _, col_indices, _ in stencil['top_rows']:
        if len(col_indices) > 0:
            top_max_col = max(top_max_col, col_indices[-1])
    top_ncols = top_max_col + 1

    top_block = np.zeros((n_top, top_ncols))
    for row_idx, col_indices, weights in stencil['top_rows']:
        top_block[row_idx, col_indices] = weights

    # Build bottom dense block
    bot_min_col = cols
    for _, col_indices, _ in stencil['bot_rows']:
        if len(col_indices) > 0:
            bot_min_col = min(bot_min_col, col_indices[0])

    bot_ncols = cols - bot_min_col
    bot_block = np.zeros((n_bot, bot_ncols))
    for row_idx, col_indices, weights in stencil['bot_rows']:
        local_row = row_idx - (rows - n_bot)
        bot_block[local_row, col_indices - bot_min_col] = weights

    return {
        'interior_weights': jnp.array(stencil['interior_weights']),
        'interior_offsets': stencil['interior_offsets'],  # keep as numpy int
        'int_start': stencil['int_start'],
        'int_end': stencil['int_end'],
        'n_top': n_top,
        'n_bot': n_bot,
        'rows': rows,
        'cols': cols,
        'top_block': jnp.array(top_block),
        'bot_block': jnp.array(bot_block),
        'top_ncols': top_ncols,
        'bot_col0': bot_min_col,
    }


# ============================================================
# Strategy A: Slice-and-accumulate
# ============================================================

def apply_slice_ax0(s, x):
    """
    Apply operator along axis -2: result[..., i, :] = sum_k w[k] * x[..., i+off[k], :]

    x:      (..., cols, K)
    result: (..., rows, K)
    """
    w = s['interior_weights']
    offsets = s['interior_offsets']
    int_start = s['int_start']
    int_end = s['int_end']
    n_int = int_end - int_start + 1

    # Interior: shifted slices
    interior = w[0] * x[..., int_start + offsets[0]:int_start + offsets[0] + n_int, :]
    for k in range(1, len(offsets)):
        start = int_start + offsets[k]
        interior = interior + w[k] * x[..., start:start + n_int, :]

    # Top boundary: small dense matmul
    x_top = x[..., :s['top_ncols'], :]
    top = jnp.einsum('ij,...jk->...ik', s['top_block'], x_top)

    # Bottom boundary
    x_bot = x[..., s['bot_col0']:, :]
    bot = jnp.einsum('ij,...jk->...ik', s['bot_block'], x_bot)

    return jnp.concatenate([top, interior, bot], axis=-2)


def apply_slice_ax1(s, x):
    """
    Apply operator along axis -1: result[..., :, i] = sum_k w[k] * x[..., :, i+off[k]]

    x:      (..., K, cols)
    result: (..., K, rows)
    """
    w = s['interior_weights']
    offsets = s['interior_offsets']
    int_start = s['int_start']
    int_end = s['int_end']
    n_int = int_end - int_start + 1

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
# Strategy B: 1D convolution for interior
# ============================================================

def apply_conv_ax0(s, x):
    """
    Apply operator along axis -2 using lax.conv for interior.

    Interior stencil as 1D convolution: kernel applied along the "cols" axis.
    Boundary handled same as Strategy A.
    """
    w = s['interior_weights']
    offsets = s['interior_offsets']
    int_start = s['int_start']
    int_end = s['int_end']
    n_int = int_end - int_start + 1

    # For conv: need to extract the right slice of input and set up kernel
    # Offsets like [-1,0,1,2] mean kernel spans 4 points, first output aligns
    # with input index int_start + offsets[0]
    min_off = int(offsets[0])
    max_off = int(offsets[-1])
    kernel_len = max_off - min_off + 1

    # Build kernel: offsets mapped to positions in kernel
    kernel = jnp.zeros(kernel_len)
    for wt, o in zip(w, offsets):
        kernel = kernel.at[int(o) - min_off].set(wt)

    # Input slice that the conv will read
    in_start = int_start + min_off
    in_end = int_end + max_off + 1  # exclusive
    x_slice = x[..., in_start:in_end, :]  # (..., in_len, K)

    # Reshape for lax.conv: need (batch, spatial, channels) -> (batch, channels, spatial)
    # Original: (..., spatial, K). Flatten leading dims.
    orig_shape = x_slice.shape
    K = orig_shape[-1]
    spatial = orig_shape[-2]
    batch = int(np.prod(orig_shape[:-2])) if len(orig_shape) > 2 else 1

    x_flat = x_slice.reshape(batch, spatial, K)   # (B, S, K)
    x_conv = jnp.transpose(x_flat, (0, 2, 1))     # (B, K, S) — treat K as channels

    # Kernel shape for conv: (out_channels, in_channels, kernel_size)
    # We want same operation on each "channel" independently
    # Use depthwise: (K, 1, kernel_len) with feature_group_count=K
    kern_conv = jnp.broadcast_to(kernel[None, None, :], (K, 1, kernel_len))

    result_conv = jax.lax.conv_general_dilated(
        x_conv, kern_conv,
        window_strides=(1,),
        padding='VALID',
        feature_group_count=K,
        dimension_numbers=('NCH', 'OIH', 'NCH'),
    )  # (B, K, n_int)

    result_flat = jnp.transpose(result_conv, (0, 2, 1))  # (B, n_int, K)
    interior = result_flat.reshape(*orig_shape[:-2], n_int, K)

    # Boundaries same as Strategy A
    x_top = x[..., :s['top_ncols'], :]
    top = jnp.einsum('ij,...jk->...ik', s['top_block'], x_top)

    x_bot = x[..., s['bot_col0']:, :]
    bot = jnp.einsum('ij,...jk->...ik', s['bot_block'], x_bot)

    return jnp.concatenate([top, interior, bot], axis=-2)


# ============================================================
# Tests
# ============================================================

def test_correctness_ax0():
    """Verify both strategies match dense einsum('ij,pjk->pik', Op, X)."""
    print("\n" + "=" * 65)
    print("TEST 1: Correctness — apply along axis 0 (Op @ X, batched)")
    print("=" * 65)

    passed = True
    for N in [16, 48, 96, 384]:
        dx = (np.pi / 2) / N
        M = N + 1
        ops = sbp_42(N, dx)

        for name, x_cols in [('Dvc', M), ('Dcv', N), ('Pvc', M), ('Pcv', N)]:
            mat = np.array(getattr(ops, name))
            stencil = extract_stencil(mat, dx)
            sj = prepare_stencil_jax(stencil)
            mat_j = jnp.array(mat)

            key = jax.random.PRNGKey(42)
            X = jax.random.normal(key, (6, x_cols, M))

            dense = jnp.einsum('ij,pjk->pik', mat_j, X)
            result_a = apply_slice_ax0(sj, X)
            result_b = apply_conv_ax0(sj, X)

            err_a = float(jnp.max(jnp.abs(dense - result_a)))
            err_b = float(jnp.max(jnp.abs(dense - result_b)))
            ok_a = err_a < 1e-11
            ok_b = err_b < 1e-11
            ok = ok_a and ok_b
            passed = passed and ok

            if not ok or N == 384:
                print(f"  N={N:3d} {name}: slice={err_a:.1e} {'✓' if ok_a else '✗'}  "
                      f"conv={err_b:.1e} {'✓' if ok_b else '✗'}")

    print(f"\n  {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


def test_correctness_ax1():
    """Verify Strategy A ax1 matches dense einsum('pij,kj->pik', X, Op)."""
    print("\n" + "=" * 65)
    print("TEST 2: Correctness — apply along axis 1 (X @ Op.T, batched)")
    print("=" * 65)

    passed = True
    for N in [16, 48, 96, 384]:
        dx = (np.pi / 2) / N
        M = N + 1
        ops = sbp_42(N, dx)

        for name, x_rows in [('Dvc', M), ('Dcv', M), ('Pvc', M), ('Pcv', M)]:
            mat = np.array(getattr(ops, name))
            x_cols = mat.shape[1]  # cols of Op = what we contract over
            stencil = extract_stencil(mat, dx)
            sj = prepare_stencil_jax(stencil)
            mat_j = jnp.array(mat)

            key = jax.random.PRNGKey(123)
            X = jax.random.normal(key, (6, x_rows, x_cols))

            dense = jnp.einsum('pij,kj->pik', X, mat_j)
            result_a = apply_slice_ax1(sj, X)

            err_a = float(jnp.max(jnp.abs(dense - result_a)))
            ok = err_a < 1e-11
            passed = passed and ok

            if not ok or N == 384:
                print(f"  N={N:3d} {name}: slice={err_a:.1e} {'✓' if ok else '✗'}")

    print(f"\n  {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


def test_performance():
    """Timing comparison: dense vs slice vs conv at N=384."""
    print("\n" + "=" * 65)
    print("TEST 3: Performance — N=384 (100 reps, CPU)")
    print("=" * 65)

    N = 384; M = N + 1
    dx = (np.pi / 2) / N
    ops = sbp_42(N, dx)
    nreps = 100

    print(f"  {'Op':>5} {'axis':>5} {'dense ms':>10} {'slice ms':>10} "
          f"{'conv ms':>10} {'slice x':>8} {'conv x':>8}")
    print(f"  {'-'*60}")

    key = jax.random.PRNGKey(0)

    for name in ['Dvc', 'Dcv', 'Pvc', 'Pcv']:
        mat = np.array(getattr(ops, name))
        stencil = extract_stencil(mat, dx)
        sj = prepare_stencil_jax(stencil)
        mat_j = jnp.array(mat)
        r, c = mat.shape

        # --- ax0: einsum('ij,pjk->pik', Op, X) ---
        X0 = jax.random.normal(key, (6, c, M))

        # Warmup all three
        d0 = jnp.einsum('ij,pjk->pik', mat_j, X0)
        s0 = apply_slice_ax0(sj, X0)
        c0 = apply_conv_ax0(sj, X0)
        jax.block_until_ready(d0); jax.block_until_ready(s0); jax.block_until_ready(c0)

        t = time.time()
        for _ in range(nreps):
            d0 = jnp.einsum('ij,pjk->pik', mat_j, X0)
        jax.block_until_ready(d0)
        dense_ms = (time.time() - t) / nreps * 1000

        t = time.time()
        for _ in range(nreps):
            s0 = apply_slice_ax0(sj, X0)
        jax.block_until_ready(s0)
        slice_ms = (time.time() - t) / nreps * 1000

        t = time.time()
        for _ in range(nreps):
            c0 = apply_conv_ax0(sj, X0)
        jax.block_until_ready(c0)
        conv_ms = (time.time() - t) / nreps * 1000

        sl_x = dense_ms / slice_ms if slice_ms > 0 else 0
        cv_x = dense_ms / conv_ms if conv_ms > 0 else 0
        print(f"  {name:>5} {'ax0':>5} {dense_ms:10.3f} {slice_ms:10.3f} "
              f"{conv_ms:10.3f} {sl_x:7.1f}x {cv_x:7.1f}x")

        # --- ax1: einsum('pij,kj->pik', X, Op) ---
        X1 = jax.random.normal(key, (6, M, c))

        d1 = jnp.einsum('pij,kj->pik', X1, mat_j)
        s1 = apply_slice_ax1(sj, X1)
        jax.block_until_ready(d1); jax.block_until_ready(s1)

        t = time.time()
        for _ in range(nreps):
            d1 = jnp.einsum('pij,kj->pik', X1, mat_j)
        jax.block_until_ready(d1)
        dense_ms = (time.time() - t) / nreps * 1000

        t = time.time()
        for _ in range(nreps):
            s1 = apply_slice_ax1(sj, X1)
        jax.block_until_ready(s1)
        slice_ms = (time.time() - t) / nreps * 1000

        sl_x = dense_ms / slice_ms if slice_ms > 0 else 0
        print(f"  {name:>5} {'ax1':>5} {dense_ms:10.3f} {slice_ms:10.3f} "
              f"{'---':>10} {sl_x:7.1f}x {'---':>8}")

    return True  # informational


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  Sparse Apply Tests")
    print("  Strategy A: slice-and-accumulate")
    print("  Strategy B: 1D convolution (ax0 only)")
    print("=" * 65)

    results = {}
    results['ax0_correct'] = test_correctness_ax0()
    results['ax1_correct'] = test_correctness_ax1()
    results['performance']  = test_performance()

    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<15} {status}")
        all_pass = all_pass and passed

    print()
    if all_pass:
        print("  All tests passed. Ready to substitute into RHS.")
    else:
        print("  Some tests failed.")
    print("=" * 65)
