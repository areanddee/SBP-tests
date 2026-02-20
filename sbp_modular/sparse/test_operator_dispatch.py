"""
test_operator_dispatch.py — Verify dispatch layer correctness
==============================================================

Tests:
  1. Dense and sparse backends produce identical results (to machine precision)
  2. Auto-detection returns a valid backend
  3. Both backends work for all operator/axis combinations
  4. Both backends handle batched (6,M,K) and single-panel (M,K) inputs
  5. Performance comparison on detected hardware

Usage:
    python test_operator_dispatch.py
"""
import sys
import os
import time

project_dir = '/mnt/project'
if os.path.isdir(project_dir) and project_dir not in sys.path:
    sys.path.insert(0, project_dir)

sys.path.insert(0, '/home/claude')

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from operator_dispatch import make_operators, detect_backend


def get_device_info():
    dev = jax.devices()[0]
    return dev.platform.upper(), getattr(dev, 'device_kind', str(dev))


# ============================================================
# Test 1: Dense vs Sparse agreement
# ============================================================

def test_dense_sparse_agreement():
    """Both backends must produce identical results to machine precision."""
    print("\n" + "=" * 65)
    print("TEST 1: Dense vs Sparse agreement")
    print("=" * 65)

    passed = True
    key = jax.random.PRNGKey(42)

    for N in [16, 48, 96, 192, 384]:
        dx = (np.pi / 2) / N
        M = N + 1

        op_d = make_operators(N, dx, backend='dense')
        op_s = make_operators(N, dx, backend='sparse')

        for name in ['Dvc', 'Dcv', 'Pvc', 'Pcv']:
            od = getattr(op_d, name)
            os_ = getattr(op_s, name)
            r, c = od.matrix.shape

            # Batched input (6 panels)
            key, subkey = jax.random.split(key)
            X0 = jax.random.normal(subkey, (6, c, M))

            d0 = od.ax0(X0)
            s0 = os_.ax0(X0)
            err0 = float(jnp.max(jnp.abs(d0 - s0)))

            # ax1: X @ Op.T
            key, subkey = jax.random.split(key)
            X1 = jax.random.normal(subkey, (6, M, c))

            d1 = od.ax1(X1)
            s1 = os_.ax1(X1)
            err1 = float(jnp.max(jnp.abs(d1 - s1)))

            ok = err0 < 1e-11 and err1 < 1e-11
            passed = passed and ok

            if not ok or N == 384:
                print(f"  N={N:3d} {name}: ax0={err0:.1e} ax1={err1:.1e}  "
                      f"{'✓' if ok else '✗'}")

    print(f"\n  {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


# ============================================================
# Test 2: Single-panel (no batch dim) works
# ============================================================

def test_single_panel():
    """Verify ax0/ax1 work for unbatched (M,K) inputs — needed for vmap."""
    print("\n" + "=" * 65)
    print("TEST 2: Single-panel (unbatched) inputs")
    print("=" * 65)

    passed = True
    key = jax.random.PRNGKey(99)
    N = 96
    dx = (np.pi / 2) / N
    M = N + 1

    for backend in ['dense', 'sparse']:
        op = make_operators(N, dx, backend=backend)

        for name in ['Dvc', 'Dcv', 'Pvc', 'Pcv']:
            o = getattr(op, name)
            r, c = o.matrix.shape

            key, subkey = jax.random.split(key)
            X = jax.random.normal(subkey, (c, M))
            result_ax0 = o.ax0(X)
            expected_ax0 = o.matrix @ X
            err0 = float(jnp.max(jnp.abs(result_ax0 - expected_ax0)))

            key, subkey = jax.random.split(key)
            X1 = jax.random.normal(subkey, (M, c))
            result_ax1 = o.ax1(X1)
            expected_ax1 = X1 @ o.matrix.T
            err1 = float(jnp.max(jnp.abs(result_ax1 - expected_ax1)))

            ok = err0 < 1e-11 and err1 < 1e-11
            passed = passed and ok

            if not ok:
                print(f"  {backend} {name}: ax0={err0:.1e} ax1={err1:.1e}  ✗")

    if passed:
        print(f"  Both backends correct for single-panel inputs  ✓")

    print(f"\n  {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


# ============================================================
# Test 3: Auto-detection
# ============================================================

def test_auto_detection():
    """Verify auto backend detection runs and returns valid choice."""
    print("\n" + "=" * 65)
    print("TEST 3: Auto backend detection")
    print("=" * 65)

    platform, dev_name = get_device_info()
    backend = detect_backend()

    print(f"  Platform: {platform} ({dev_name})")
    print(f"  Selected: {backend}")

    ok = backend in ('dense', 'sparse')
    print(f"\n  {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


# ============================================================
# Test 4: Operator metadata accessible
# ============================================================

def test_metadata():
    """Verify Hv, Hc, l, r, matrix all accessible."""
    print("\n" + "=" * 65)
    print("TEST 4: Operator metadata")
    print("=" * 65)

    N = 48
    dx = (np.pi / 2) / N
    op = make_operators(N, dx, backend='dense')

    checks = [
        ('Hv_diag', (N + 1,)),
        ('Hc_diag', (N,)),
        ('l', (N,)),
        ('r', (N,)),
    ]

    passed = True
    for attr, expected_shape in checks:
        val = getattr(op, attr)
        ok = val.shape == expected_shape
        passed = passed and ok
        print(f"  {attr}: shape={val.shape} expected={expected_shape}  "
              f"{'✓' if ok else '✗'}")

    # Check matrices
    for name, expected_shape in [
        ('Dvc', (N, N + 1)),
        ('Dcv', (N + 1, N)),
        ('Pvc', (N, N + 1)),
        ('Pcv', (N + 1, N)),
    ]:
        mat = getattr(op, name).matrix
        ok = mat.shape == expected_shape
        passed = passed and ok
        print(f"  {name}.matrix: shape={mat.shape} expected={expected_shape}  "
              f"{'✓' if ok else '✗'}")

    print(f"\n  {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


# ============================================================
# Test 5: Performance comparison
# ============================================================

def test_performance():
    """Timing comparison: dense vs sparse on detected hardware."""
    platform, dev_name = get_device_info()

    print("\n" + "=" * 65)
    print(f"TEST 5: Performance — N=384 ({platform}: {dev_name})")
    print("=" * 65)

    N = 384
    dx = (np.pi / 2) / N
    M = N + 1
    nreps = 200

    op_d = make_operators(N, dx, backend='dense')
    op_s = make_operators(N, dx, backend='sparse')

    key = jax.random.PRNGKey(0)

    print(f"  {nreps} reps, batched (6, *, *)")
    print(f"  {'Op':>5} {'axis':>5} {'dense µs':>10} {'sparse µs':>10} {'winner':>22}")
    print(f"  {'-'*57}")

    for name in ['Dvc', 'Dcv', 'Pvc', 'Pcv']:
        od = getattr(op_d, name)
        os_ = getattr(op_s, name)
        _, c = od.matrix.shape

        for axis in [0, 1]:
            key, subkey = jax.random.split(key)
            if axis == 0:
                X = jax.random.normal(subkey, (6, c, M))
                fn_d = lambda x=X: od.ax0(x)
                fn_s = lambda x=X: os_.ax0(x)
                label = 'ax0'
            else:
                X = jax.random.normal(subkey, (6, M, c))
                fn_d = lambda x=X: od.ax1(x)
                fn_s = lambda x=X: os_.ax1(x)
                label = 'ax1'

            # Warmup
            jax.block_until_ready(fn_d())
            jax.block_until_ready(fn_s())

            t = time.time()
            for _ in range(nreps):
                r_d = fn_d()
            jax.block_until_ready(r_d)
            dense_us = (time.time() - t) / nreps * 1e6

            t = time.time()
            for _ in range(nreps):
                r_s = fn_s()
            jax.block_until_ready(r_s)
            sparse_us = (time.time() - t) / nreps * 1e6

            ratio = max(dense_us, sparse_us) / max(min(dense_us, sparse_us), 1)
            if dense_us <= sparse_us:
                winner = f"dense {ratio:.0f}x faster"
            else:
                winner = f"sparse {ratio:.0f}x faster"

            print(f"  {name:>5} {label:>5} {dense_us:10.0f} {sparse_us:10.0f} {winner:>22}")

    auto = detect_backend()
    print(f"\n  Auto-selected backend: {auto}")
    return True  # informational


# ============================================================
# Test 6: RHS integration smoke test
# ============================================================

def test_rhs_smoke():
    """
    Verify that using OperatorSet in an RHS-like computation gives
    identical results to using raw matrices.
    """
    print("\n" + "=" * 65)
    print("TEST 6: RHS integration smoke test")
    print("=" * 65)

    N = 48
    dx = (np.pi / 2) / N
    M = N + 1

    # Build both backends
    op_d = make_operators(N, dx, backend='dense')
    op_s = make_operators(N, dx, backend='sparse')

    # Raw matrices for reference
    Dvc = op_d.Dvc.matrix
    Dcv = op_d.Dcv.matrix
    Pvc = op_d.Pvc.matrix
    Pcv = op_d.Pcv.matrix

    key = jax.random.PRNGKey(77)
    h = jax.random.normal(key, (6, M, M))
    key, subkey = jax.random.split(key)
    v1 = jax.random.normal(subkey, (6, N, M))
    key, subkey = jax.random.split(key)
    v2 = jax.random.normal(subkey, (6, M, N))

    # --- Gradient (raw einsum) ---
    grad1_ref = jnp.einsum('ij,pjk->pik', Dvc, h)
    grad2_ref = jnp.einsum('pij,kj->pik', h, Dvc)

    # --- Gradient (dispatch) ---
    for label, op in [('dense', op_d), ('sparse', op_s)]:
        grad1 = op.Dvc.ax0(h)
        grad2 = op.Dvc.ax1(h)
        err1 = float(jnp.max(jnp.abs(grad1 - grad1_ref)))
        err2 = float(jnp.max(jnp.abs(grad2 - grad2_ref)))
        print(f"  Gradient ({label}): ax0={err1:.1e}, ax1={err2:.1e}")

    # --- Divergence (raw einsum) ---
    div1_ref = jnp.einsum('ij,pjk->pik', Dcv, v1)
    div2_ref = jnp.einsum('pij,kj->pik', v2, Dcv)

    for label, op in [('dense', op_d), ('sparse', op_s)]:
        div1 = op.Dcv.ax0(v1)
        div2 = op.Dcv.ax1(v2)
        err1 = float(jnp.max(jnp.abs(div1 - div1_ref)))
        err2 = float(jnp.max(jnp.abs(div2 - div2_ref)))
        print(f"  Divergence ({label}): ax0={err1:.1e}, ax1={err2:.1e}")

    # --- Contravariant-like (single panel via vmap, raw @) ---
    def contra_ref(v2_p):
        return v2_p @ Pcv.T

    def contra_dispatch(v2_p, op):
        return op.Pcv.ax1(v2_p)

    ref = jax.vmap(contra_ref)(v2)
    passed = True
    for label, op in [('dense', op_d), ('sparse', op_s)]:
        result = jax.vmap(lambda v, o=op: contra_dispatch(v, o))(v2)
        err = float(jnp.max(jnp.abs(result - ref)))
        ok = err < 1e-11
        passed = passed and ok
        print(f"  Pcv.ax1 via vmap ({label}): err={err:.1e}  {'✓' if ok else '✗'}")

    print(f"\n  {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    platform, dev_name = get_device_info()
    print("=" * 65)
    print("  Operator Dispatch Tests")
    print(f"  Device: {platform} ({dev_name})")
    print("=" * 65)

    results = {}
    results['agreement']  = test_dense_sparse_agreement()
    results['single']     = test_single_panel()
    results['autodetect'] = test_auto_detection()
    results['metadata']   = test_metadata()
    results['performance'] = test_performance()
    results['rhs_smoke']  = test_rhs_smoke()

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
        print("  Dispatch layer verified. Ready to integrate into RHS.")
    else:
        print("  Some tests failed.")
    print("=" * 65)
