def test_convergence(variant=1):
    """
    Spatial convergence test against exact spectral solution.

    Compares numerical h-field at T=25 days against the exact Legendre
    polynomial expansion solution (Shashkin Eq. 84, f=0).

    Reference solutions are pre-generated zarr files in:
        reference_solutions/gauss{1,2}/N{024,048,...}.zarr

    variant 1: Gaussian centered at panel center (0,0) on panel 0
    variant 2: Gaussian centered at cube vertex (pi/4,pi/4) on panel 0

    Shashkin Table 2, Ch42 convergence rates:
      Variant 1: l2 = 4.25, linf = 3.98
      Variant 2: l2 = 3.81, linf = 3.43
    """
    import time as _time
    import zarr

    label = "panel center (0,0)" if variant == 1 else "cube vertex (pi/4,pi/4)"
    print("\n" + "=" * 65)
    print(f"TEST 7: Spatial Convergence -- Gaussian variant {variant}")
    print(f"        Center at {label}")
    print(f"        Reference: exact spectral solution")
    print("=" * 65)

    Ns = [24, 48, 96, 192]
    H0 = 1.0; g = 1.0; c = np.sqrt(g * H0)
    T_end = 25.0

    print(f"  T = {T_end}, H0 = {H0}, g = {g}")
    print(f"  N values: {Ns}")

    # --- Load exact references ---
    ref_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "reference_solutions", f"gauss{variant}")
    print(f"  Reference dir: {ref_dir}")

    h_refs = {}
    for N in Ns:
        path = os.path.join(ref_dir, f"N{N:03d}.zarr")
        if not os.path.exists(path):
            print(f"  ERROR: reference not found: {path}")
            print(f"  Run: python generate_reference_solutions.py")
            return False
        store = zarr.open(path, mode='r')
        h_refs[N] = jnp.array(np.array(store['h_exact']))
        if N == Ns[0]:
            attrs = dict(store.attrs)
            print(f"  Spectral params: sigma={attrs['sigma']:.6f}, "
                  f"amp={float(attrs.get('amp', 1.0)):.1f}, "
                  f"L_max={attrs['L_max']}, T_end={attrs['T_end']}")

    # --- Gaussian IC using great-circle distance ---
    # Shashkin Eq. 84: h(0) = exp(-16 theta^2)
    sigma_ic = 1.0 / (4.0 * np.sqrt(2.0))   # matches exp(-16 theta^2)
    amp_ic = 1.0

    def make_gaussian_ic(N_loc, grids_loc):
        xi_v_loc = grids_loc['xi_v']
        xi1_2d, xi2_2d = jnp.meshgrid(xi_v_loc, xi_v_loc, indexing='ij')
        h_ic = jnp.zeros((6, N_loc + 1, N_loc + 1))
        v1_ic = jnp.zeros((6, N_loc, N_loc + 1))
        v2_ic = jnp.zeros((6, N_loc + 1, N_loc))
        if variant == 1:
            X0, Y0, Z0 = 0.0, 0.0, 1.0
        else:
            X0, Y0, Z0 = equiangular_to_cartesian(
                jnp.array(jnp.pi / 4), jnp.array(jnp.pi / 4), 0)
            X0, Y0, Z0 = float(X0), float(Y0), float(Z0)
        for p in range(6):
            X, Y, Z = equiangular_to_cartesian(xi1_2d, xi2_2d, p)
            cos_d = jnp.clip(X * X0 + Y * Y0 + Z * Z0, -1.0, 1.0)
            d = jnp.arccos(cos_d)
            h_ic = h_ic.at[p].set(amp_ic * jnp.exp(-d**2 / (2 * sigma_ic**2)))
        return h_ic, v1_ic, v2_ic

    # --- Run all resolutions ---
    run_results = {}
    for N in Ns:
        sys_d = make_cubed_sphere_swe(N, H0, g)
        grids = sys_d['grids']
        dx = sys_d['dx']
        dt = 1.0 / (N * 3)     # Shashkin Table 1
        CFL = c * dt / dx
        nsteps = int(np.ceil(T_end / dt))
        dt = T_end / nsteps     # hit T_end exactly

        h, v1, v2 = make_gaussian_ic(N, grids)
        mass0 = compute_mass(h, sys_d['Wh'], sys_d['Jh'])
        step_fn = make_rk4_step(sys_d['rhs'])

        print(f"\n  N = {N:3d}: dx = {dx:.4e}, dt = {dt:.4e}, CFL = {CFL:.4e}, steps = {nsteps}",
              end='', flush=True)

        t0 = _time.time()
        h, v1, v2 = step_fn(h, v1, v2, dt)  # JIT warmup
        jax.block_until_ready(h)
        jit_t = _time.time() - t0
        print(f"  (JIT {jit_t:.1f}s)", end='', flush=True)

        t0 = _time.time()
        for s in range(1, nsteps):
            h, v1, v2 = step_fn(h, v1, v2, dt)
        jax.block_until_ready(h)
        run_t = _time.time() - t0
        print(f"  (run {run_t:.1f}s)", flush=True)

        mass_f = compute_mass(h, sys_d['Wh'], sys_d['Jh'])
        mass_err = abs(mass_f - mass0)
        print(f"         mass_err = {mass_err:.2e}")

        run_results[N] = {
            'h': h, 'dx': dx,
            'Jh': sys_d['Jh'], 'Wh': sys_d['Wh'],
        }

    # --- Compute errors vs exact spectral solution ---
    print(f"\n  Errors vs exact spectral solution (T = {T_end}):")
    print(f"  {'N':>5} {'dx':>12} {'L2 err':>12} {'L2 rate':>8} "
          f"{'Linf err':>12} {'Linf rate':>10}")
    print(f"  {'-'*65}")

    errors = []
    for N in Ns:
        diff = run_results[N]['h'] - h_refs[N]
        Jh = run_results[N]['Jh']
        Wh = run_results[N]['Wh']

        # Quadrature-weighted L2 on the sphere
        l2 = float(jnp.sqrt(jnp.sum(diff**2 * Jh[None] * Wh[None])))
        linf = float(jnp.max(jnp.abs(diff)))

        errors.append({'N': N, 'dx': run_results[N]['dx'], 'l2': l2, 'linf': linf})

        if len(errors) >= 2:
            l2_rate = (np.log(errors[-2]['l2'] / errors[-1]['l2']) /
                       np.log(errors[-2]['dx'] / errors[-1]['dx']))
            linf_rate = (np.log(errors[-2]['linf'] / errors[-1]['linf']) /
                         np.log(errors[-2]['dx'] / errors[-1]['dx']))
            l2_str = f"{l2_rate:8.2f}"
            linf_str = f"{linf_rate:10.2f}"
        else:
            l2_str = "     ---"
            linf_str = "       ---"

        print(f"  {N:5d} {run_results[N]['dx']:12.4e} {l2:12.4e} {l2_str} "
              f"{linf:12.4e} {linf_str}")

    # --- Summary ---
    if len(errors) >= 2:
        final_l2 = (np.log(errors[-2]['l2'] / errors[-1]['l2']) /
                    np.log(errors[-2]['dx'] / errors[-1]['dx']))
        final_linf = (np.log(errors[-2]['linf'] / errors[-1]['linf']) /
                      np.log(errors[-2]['dx'] / errors[-1]['dx']))
    else:
        final_l2 = 0.0
        final_linf = 0.0

    print(f"\n  Final L2 convergence rate:   {final_l2:.2f}")
    print(f"  Final Linf convergence rate: {final_linf:.2f}")

    # Shashkin Table 2 Ch42 rates
    if variant == 1:
        print(f"  Shashkin Ch42 reference:     l2=4.25, linf=3.98")
        threshold = 2.5
    else:
        print(f"  Shashkin Ch42 reference:     l2=3.81, linf=3.43")
        threshold = 1.8

    passed = final_l2 > threshold
    print(f"  Expected L2 rate > {threshold:.1f}")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed
