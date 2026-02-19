# JAX GPU Profiling Guide

Practical guide for profiling the cubed-sphere SWE code on GPU.

## Quick Start: Trace-Based Profiling

### 1. Add a profiling block to the test

Insert after JIT warmup, before the main run:

```python
import jax

# Warmup (already done)
h, v1, v2 = step_fn(h, v1, v2, dt)
jax.block_until_ready(h)

# Profile 10 steps
jax.profiler.start_trace("/tmp/jax-trace")
for _ in range(10):
    h, v1, v2 = step_fn(h, v1, v2, dt)
jax.block_until_ready(h)
jax.profiler.stop_trace()
print("Trace saved to /tmp/jax-trace")
```

### 2. View with TensorBoard

```bash
pip install tensorboard-plugin-profile
tensorboard --logdir /tmp/jax-trace --port 6006
```

Open `http://localhost:6006` in a browser. Key tabs:

- **Trace Viewer**: Timeline of GPU ops. Look for gaps (= idle GPU waiting on host) and long bars (= expensive ops).
- **Overview Page**: Summary of GPU utilization, step time breakdown.
- **Op Profile**: Which XLA ops consume the most time.

If running on a remote HPC node, tunnel the port:

```bash
ssh -L 6006:localhost:6006 user@hpc-node
```

### 3. What to look for

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Long gaps between GPU ops | Python dispatch overhead | `lax.fori_loop` (done) |
| Many tiny scatter ops in sequence | Sequential `.at[].set/add()` | Batch into single scatter |
| Large memcpy bars | Host↔device transfers | Keep data on device, avoid `float()` in hot path |
| Single op dominates | Expensive einsum or matmul | Check if operation can be restructured |


## Alternative: Simple Timing

If TensorBoard is not available, bracket individual components:

```python
import time

# Profile individual RHS components
h_proj = project_h(h)
jax.block_until_ready(h_proj)

# Time gradient
t0 = time.time()
for _ in range(100):
    dv1 = -g * jnp.einsum('ij,pjk->pik', Dvc, h_proj)
jax.block_until_ready(dv1)
print(f"Gradient: {(time.time()-t0)/100*1000:.2f} ms/call")

# Time contravariant velocity
t0 = time.time()
for _ in range(100):
    v1c, v2c = _contra_vmap(v1, v2)
jax.block_until_ready(v1c)
print(f"Contravariant: {(time.time()-t0)/100*1000:.2f} ms/call")

# Time divergence
t0 = time.time()
for _ in range(100):
    div = (jnp.einsum('ij,pjk->pik', Dcv, u1_all) +
           jnp.einsum('pij,kj->pik', u2_all, Dcv))
jax.block_until_ready(div)
print(f"Divergence: {(time.time()-t0)/100*1000:.2f} ms/call")

# Time SAT
t0 = time.time()
for _ in range(100):
    div2 = add_sat(div, u1_all, u2_all, v1, v2)
jax.block_until_ready(div2)
print(f"SAT: {(time.time()-t0)/100*1000:.2f} ms/call")

# Time projection
t0 = time.time()
for _ in range(100):
    hp = project_h(h)
jax.block_until_ready(hp)
print(f"Projection: {(time.time()-t0)/100*1000:.2f} ms/call")
```

This requires exposing the internal functions. Add a helper to `make_cubed_sphere_swe` that returns them, or just inline the profiling code temporarily.


## Standalone Profiling Script

Drop this next to the test file. Run as `python profile_rhs.py`.

```python
"""Profile individual RHS components at N=384."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from test_stag_step5_Nconv import (
    make_cubed_sphere_swe, make_rk4_step, equiangular_to_cartesian
)

N = 384; H0 = 1.0; g = 1.0
print(f"Building system N={N}...")
sys_d = make_cubed_sphere_swe(N, H0, g)

# Create IC
grids = sys_d['grids']
xi_v = grids['xi_v']
xi1_2d, xi2_2d = jnp.meshgrid(xi_v, xi_v, indexing='ij')
h = jnp.zeros((6, N+1, N+1))
for p in range(6):
    X, Y, Z = equiangular_to_cartesian(xi1_2d, xi2_2d, p)
    d = jnp.arccos(jnp.clip(Z, -1.0, 1.0))
    h = h.at[p].set(0.01 * jnp.exp(-d**2 / 0.18))
v1 = jnp.zeros((6, N, N+1))
v2 = jnp.zeros((6, N+1, N))

# Warmup
rhs = sys_d['rhs']
print("JIT compiling RHS...", flush=True)
t0 = time.time()
dh, dv1, dv2 = rhs(h, v1, v2)
jax.block_until_ready(dh)
print(f"JIT: {time.time()-t0:.1f}s")

# Time full RHS
nreps = 50
t0 = time.time()
for _ in range(nreps):
    dh, dv1, dv2 = rhs(h, v1, v2)
jax.block_until_ready(dh)
rhs_ms = (time.time() - t0) / nreps * 1000
print(f"\nFull RHS: {rhs_ms:.2f} ms/call  ({nreps} reps)")

# Time full RK4 step
step_fn, run_n_steps = make_rk4_step(rhs)
dx = sys_d['dx']
c = np.sqrt(g * H0)
dt = 1.0 / (N * 3)

print("JIT compiling RK4 step...", flush=True)
t0 = time.time()
h2, v12, v22 = step_fn(h, v1, v2, dt)
jax.block_until_ready(h2)
print(f"JIT: {time.time()-t0:.1f}s")

nreps = 20
t0 = time.time()
for _ in range(nreps):
    h2, v12, v22 = step_fn(h, v1, v2, dt)
jax.block_until_ready(h2)
step_ms = (time.time() - t0) / nreps * 1000
print(f"RK4 step: {step_ms:.2f} ms/call  ({nreps} reps)")
print(f"Expected RHS share: {4 * rhs_ms:.2f} ms  (4 evals)")

# Time fori_loop batch
nsteps_test = 100
print(f"\nJIT compiling fori_loop ({nsteps_test} steps)...", flush=True)
t0 = time.time()
h3, v13, v23 = run_n_steps(h, v1, v2, dt, nsteps_test)
jax.block_until_ready(h3)
print(f"JIT + run: {time.time()-t0:.1f}s")

t0 = time.time()
h3, v13, v23 = run_n_steps(h, v1, v2, dt, nsteps_test)
jax.block_until_ready(h3)
fori_ms = (time.time() - t0) * 1000
print(f"fori_loop {nsteps_test} steps: {fori_ms:.1f} ms total, "
      f"{fori_ms/nsteps_test:.2f} ms/step")
print(f"Python loop estimate: {step_ms * nsteps_test:.1f} ms")
print(f"fori_loop speedup: {step_ms * nsteps_test / fori_ms:.2f}x")

# Trace-based profile (optional)
print("\n--- Generating trace profile ---")
jax.profiler.start_trace("/tmp/jax-trace-rhs")
for _ in range(10):
    dh, dv1, dv2 = rhs(h, v1, v2)
jax.block_until_ready(dh)
jax.profiler.stop_trace()
print("Trace saved to /tmp/jax-trace-rhs")
print("View with: tensorboard --logdir /tmp/jax-trace-rhs")
```


## GPU Memory and Utilization

Check GPU utilization during a run:

```bash
# In a separate terminal while the test is running:
watch -n 1 nvidia-smi

# Or for a log:
nvidia-smi dmon -s um -d 1 > gpu_log.txt &
# Then run your test, then kill the logger
```

Key metrics:
- **SM%**: GPU compute utilization. Should be >80% during the run phase.
- **Mem%**: GPU memory utilization.
- **FB Used**: GPU memory consumed. At N=384 with f64, expect ~6×385² × 8 bytes × ~20 arrays ≈ 140 MB — tiny for a modern GPU.

Low SM% during the run phase means the GPU is starved for work, which points to either host dispatch overhead or sequential dependencies in the XLA graph.


## XLA Compilation Flags

For more aggressive optimization:

```bash
# Dump XLA HLO for inspection
XLA_FLAGS="--xla_dump_to=/tmp/xla-dump" python test_stag_step5_Nconv.py

# Enable op fusion and layout optimization
XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" python test_stag_step5_Nconv.py
```

The HLO dump shows exactly what XLA compiled. Look for:
- `scatter` ops (from `.at[].set/add()`) — count how many are sequential
- `fusion` ops — more fusion = better GPU utilization
- `while` ops — this is the `fori_loop` body


## Suspected Bottlenecks (Current Code)

Ranked by estimated impact at N=384:

1. **`project_h`** — called 2× per RHS (8× per RK4 step). Loops over 12 edges + 8 corners doing sequential `.at[].set()`. ~256 scatter ops per RK4 step.

2. **`build_cartesian_sat_fn`** — called 1× per RHS (4× per step). Each of 12 edges does: extrapolate covariant → Cartesian transform → average → back-transform → consensus flux → 2 scatter `.at[].add()`. ~96 scatter ops per step, plus expensive per-edge computation.

3. **`compute_contravariant`** via `vmap` — called 1× per RHS. This should vectorize well, but the Pvc/Pcv interpolation matmuls are small per-panel.

Profile first to confirm ranking before optimizing.
