"""
bench_exchange_scaling.py — Multi-GPU Halo Exchange Benchmark
==============================================================

Measures the cost of 12-edge cubed-sphere halo exchange in isolation,
scaling across 1, 2, 3, 6 GPUs using JAX NamedSharding.

The 6 cube faces are sharded along the first array dimension using a
1D Mesh ('faces',). The 4-stage communication schedule from
halo_exchange.py avoids race conditions regardless of device count.

Usage:
    # On 8×H100 node:
    python bench_exchange_scaling.py                    # all configs
    python bench_exchange_scaling.py --ngpus 6          # single config
    python bench_exchange_scaling.py --N 384 --iters 500

Output:
    - Latency per exchange (μs)
    - Effective bandwidth (GB/s)
    - Scaling efficiency vs 1-GPU baseline
    - Cross-device vs local edge counts

Requires: 8×H100 (or adjust --ngpus to available device count)
Uses: halo_exchange.py connectivity (Rich's pattern, not Shashkin's)
"""
import os
import sys
import time
import argparse
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from halo_exchange import (
    create_communication_schedule,
    make_halo_exchange,
    extend_to_include_ghosts,
)


# ============================================================
# Device / Sharding setup
# ============================================================

def setup_sharding(num_gpus):
    """
    Create a 1D Mesh over the face dimension.

    Args:
        num_gpus: Number of devices to use (1, 2, 3, or 6)

    Returns:
        mesh: jax.sharding.Mesh
        sharding: NamedSharding for (6, ...) arrays
    """
    devices = jax.devices()[:num_gpus]
    print(f"  Devices: {len(devices)} × {devices[0].device_kind}")
    for d in devices:
        print(f"    {d}")

    mesh = Mesh(np.array(devices), ('faces',))
    sharding = NamedSharding(mesh, P('faces',))

    return mesh, sharding


def count_cross_device_edges(num_gpus):
    """
    Count how many of the 12 edges cross device boundaries.

    With NamedSharding P('faces',) on 6 faces:
      6 GPUs: faces [0],[1],[2],[3],[4],[5] — all 12 edges cross devices
      3 GPUs: faces [0,1],[2,3],[4,5] — some edges local
      2 GPUs: faces [0,1,2],[3,4,5] — more edges local
      1 GPU:  all faces local — 0 cross-device edges
    """
    schedule = create_communication_schedule()

    # Determine which face goes to which device
    faces_per_device = 6 // num_gpus
    def device_of(face_id):
        return face_id // faces_per_device

    cross = 0
    local = 0
    details = []
    for stage in schedule:
        for (fa, ea), (fb, eb), op in stage:
            is_cross = device_of(fa) != device_of(fb)
            if is_cross:
                cross += 1
            else:
                local += 1
            details.append((fa, ea, fb, eb, op, is_cross))

    return cross, local, details


# ============================================================
# Benchmark kernel
# ============================================================

def run_benchmark(N, num_gpus, n_warmup=20, n_iters=200):
    """
    Benchmark one complete halo exchange at resolution N on num_gpus devices.

    Returns dict with timing statistics.
    """
    print(f"\n{'='*65}")
    print(f"  Benchmark: N={N}, GPUs={num_gpus}")
    print(f"{'='*65}")

    # --- Setup sharding ---
    mesh, sharding = setup_sharding(num_gpus)

    # --- Cross-device edge analysis ---
    cross, local, details = count_cross_device_edges(num_gpus)
    print(f"  Edges: {cross} cross-device, {local} local, {cross+local} total")

    # --- Create test field ---
    # Use (6, N+2, N+2) ghost-extended field (matches halo_exchange API)
    field_interior = jnp.ones((6, N, N), dtype=jnp.float64)
    for face in range(6):
        field_interior = field_interior.at[face].set(float(face))

    field_ghosts = extend_to_include_ghosts(field_interior, N)

    # Shard the field across devices
    field_sharded = jax.device_put(field_ghosts, sharding)
    print(f"  Field shape: {field_ghosts.shape}, "
          f"dtype={field_ghosts.dtype}, "
          f"size={field_ghosts.nbytes / 1024:.1f} KB")
    print(f"  Sharding: {field_sharded.sharding}")

    # --- Build exchange function ---
    schedule = create_communication_schedule()
    exchange_fn = make_halo_exchange(schedule, N)

    # --- Warmup (JIT compile + cache) ---
    print(f"  Warming up ({n_warmup} iterations)...", end='', flush=True)
    for _ in range(n_warmup):
        result = exchange_fn(field_sharded)
        jax.block_until_ready(result)
        field_sharded = result
    print(" done")

    # --- Benchmark ---
    print(f"  Benchmarking ({n_iters} iterations)...", end='', flush=True)
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        result = exchange_fn(field_sharded)
        jax.block_until_ready(result)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        field_sharded = result
    print(" done")

    times = np.array(times) * 1e6  # convert to microseconds

    # --- Bandwidth calculation ---
    # Each edge exchanges 2 × N values (bidirectional) × 8 bytes (float64)
    bytes_per_edge = 2 * N * 8
    total_bytes = 12 * bytes_per_edge
    cross_bytes = cross * bytes_per_edge
    mean_time_s = np.mean(times) * 1e-6
    total_bw = total_bytes / mean_time_s / 1e9  # GB/s
    cross_bw = cross_bytes / mean_time_s / 1e9 if cross > 0 else 0.0

    stats = {
        'N': N,
        'num_gpus': num_gpus,
        'cross_edges': cross,
        'local_edges': local,
        'mean_us': np.mean(times),
        'std_us': np.std(times),
        'min_us': np.min(times),
        'max_us': np.max(times),
        'p50_us': np.percentile(times, 50),
        'p99_us': np.percentile(times, 99),
        'total_bytes': total_bytes,
        'cross_bytes': cross_bytes,
        'total_bw_gbs': total_bw,
        'cross_bw_gbs': cross_bw,
    }

    print(f"\n  Results:")
    print(f"    Mean:   {stats['mean_us']:10.1f} μs")
    print(f"    Median: {stats['p50_us']:10.1f} μs")
    print(f"    Min:    {stats['min_us']:10.1f} μs")
    print(f"    Max:    {stats['max_us']:10.1f} μs")
    print(f"    Std:    {stats['std_us']:10.1f} μs")
    print(f"    P99:    {stats['p99_us']:10.1f} μs")
    print(f"    Data:   {total_bytes:d} bytes/exchange "
          f"({cross_bytes} cross-device)")
    print(f"    BW:     {total_bw:.3f} GB/s total, "
          f"{cross_bw:.3f} GB/s cross-device")

    return stats


# ============================================================
# Scaling sweep
# ============================================================

def run_scaling_sweep(N, gpu_configs, n_warmup=20, n_iters=200):
    """Run benchmark across multiple GPU configurations."""
    print("=" * 65)
    print(f"  HALO EXCHANGE SCALING BENCHMARK")
    print(f"  N = {N}, field = (6, {N+2}, {N+2}) float64")
    print(f"  Connectivity: Rich's 4-stage schedule (halo_exchange.py)")
    print(f"  Available devices: {len(jax.devices())} × "
          f"{jax.devices()[0].device_kind}")
    print("=" * 65)

    # Edge analysis for all configs
    print(f"\n  Cross-device edge counts:")
    print(f"  {'GPUs':>6} {'Cross':>8} {'Local':>8} {'Total':>8}")
    print(f"  {'-'*34}")
    for ng in gpu_configs:
        c, l, _ = count_cross_device_edges(ng)
        print(f"  {ng:6d} {c:8d} {l:8d} {c+l:8d}")

    # Run benchmarks
    all_stats = []
    for ng in gpu_configs:
        stats = run_benchmark(N, ng, n_warmup=n_warmup, n_iters=n_iters)
        all_stats.append(stats)

    # --- Summary table ---
    print(f"\n{'='*65}")
    print(f"  SCALING SUMMARY (N={N})")
    print(f"{'='*65}")
    print(f"  {'GPUs':>6} {'Cross':>7} {'Mean(μs)':>10} {'Med(μs)':>10} "
          f"{'Min(μs)':>10} {'BW(GB/s)':>10} {'Speedup':>8}")
    print(f"  {'-'*63}")

    baseline = all_stats[0]['mean_us']  # 1-GPU baseline
    for s in all_stats:
        speedup = baseline / s['mean_us']
        print(f"  {s['num_gpus']:6d} {s['cross_edges']:7d} "
              f"{s['mean_us']:10.1f} {s['p50_us']:10.1f} "
              f"{s['min_us']:10.1f} {s['total_bw_gbs']:10.3f} "
              f"{speedup:8.2f}x")

    # --- Per-stage analysis for 6-GPU case ---
    if 6 in gpu_configs:
        print(f"\n  4-Stage Schedule (6 GPUs):")
        schedule = create_communication_schedule()
        for i, stage in enumerate(schedule):
            pairs = []
            for (fa, ea), (fb, eb), op in stage:
                pairs.append(f"({fa},{ea})↔({fb},{eb})[{op}]")
            print(f"    Stage {i}: {', '.join(pairs)}")

    return all_stats


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU halo exchange benchmark")
    parser.add_argument('--N', type=int, default=384,
                        help='Grid resolution per panel (default: 384)')
    parser.add_argument('--ngpus', type=int, default=None,
                        help='Single GPU count to test (default: sweep 1,2,3,6)')
    parser.add_argument('--iters', type=int, default=200,
                        help='Benchmark iterations (default: 200)')
    parser.add_argument('--warmup', type=int, default=20,
                        help='Warmup iterations (default: 20)')
    args = parser.parse_args()

    n_available = len(jax.devices())
    print(f"JAX devices available: {n_available}")

    if args.ngpus is not None:
        gpu_configs = [args.ngpus]
    else:
        # Standard scaling configs: 1, 2, 3, 6
        gpu_configs = [c for c in [1, 2, 3, 6] if c <= n_available]

    if not gpu_configs:
        print(f"ERROR: No valid GPU configs for {n_available} devices")
        sys.exit(1)

    # Verify 6 divides evenly for each config
    for ng in gpu_configs:
        if 6 % ng != 0:
            print(f"WARNING: 6 faces doesn't divide evenly across {ng} GPUs")
            print(f"  NamedSharding requires 6 % num_gpus == 0")
            gpu_configs.remove(ng)

    all_stats = run_scaling_sweep(
        args.N, gpu_configs,
        n_warmup=args.warmup, n_iters=args.iters
    )

    print(f"\n{'='*65}")
    print(f"  Benchmark complete.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
