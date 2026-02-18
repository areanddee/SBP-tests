# SAT Refactoring Plan: Cartesian-Averaged Velocity at Panel Interfaces

## Problem Statement

The current SAT correction blindly averages mass flux `u = J·v^contra` from 
neighboring panels. On 8 of 12 cubed-sphere edges, the local coordinate systems 
are misaligned (rotated, reflected, or axis-swapped), so the mass flux components 
represent **different physical directions**. Averaging them is physically 
meaningless and produces O(1) errors that don't converge.

The 4 equatorial E↔W edges work by accident because their ξ¹ axes are aligned.

## Solution: Cartesian Averaging (FV3 approach)

At each shared edge:
1. Extrapolate **covariant velocity** (v_1, v_2) to the boundary (both panels)
2. Convert each panel's covariant velocity to **Cartesian** (Vx, Vy, Vz)
3. Average in Cartesian: V_avg = 0.5 * (V_A + V_B)
4. Convert V_avg back to each panel's **covariant frame**
5. Compute mass flux from the averaged velocity and apply SAT penalty

This is coordinate-system-free and automatically handles all edge types.

## Why Covariant (Not Contravariant)?

The SWE momentum equation evolves **covariant** v_i (Shashkin Eq. 50):
   dv_i/dt = -g ∂h/∂ξ^i

The state variables v1, v2 in test_stag_step4.py are covariant components.
The code then computes contravariant v^i = Q^{ij} v_j via compute_contravariant()
to get the mass flux u = J * v^contra.

For the SAT, we need to extrapolate from cell-center velocities to the boundary.
These cell-center velocities are covariant. The extrapolation l^T v and r^T v 
gives covariant components at the boundary.

## Incremental Steps with Unit Tests

---

### Step 1: velocity_transforms.py — Correct Naming ✅ DONE

**File**: velocity_transforms.py (in outputs/)
**Changes**:
- Renamed `covariant_to_cartesian` → `contravariant_to_cartesian`
- Renamed `cartesian_to_covariant` → `cartesian_to_contravariant`  
- Added TRUE `cartesian_to_covariant` (returns v_i = V·a_i without raising)
- Added TRUE `covariant_to_cartesian` (raises index then expands)
- All roundtrip tests pass

**Test**: `velocity_transforms.py` self-test ✅

---

### Step 2: Extrapolate Covariant Velocity to Boundary

**Goal**: Given the staggered-grid covariant velocity fields v1(N,N+1) and 
v2(N+1,N), extrapolate **both components** to each edge of the h-grid.

**Key insight**: At an E/W boundary, v1 lives at cell-centers in ξ¹ and needs 
extrapolation via l/r. But v2 lives at cell-centers in ξ² — it already has 
values at the ξ¹ boundary (v2 is (N+1,N), so v2[0,:] and v2[N,:] are AT the 
E/W boundary). However, v2 needs interpolation to the h-grid (N+1 points) 
via Pcv.

At a N/S boundary, the roles reverse.

**Function signature**:
```python
def extrapolate_covariant_to_boundary(v1, v2, panel, edge, ops):
    """
    Extrapolate both covariant components to an h-grid boundary.
    
    Returns: v1_bnd (N+1,), v2_bnd (N+1,) — covariant velocity at boundary
    """
```

**Details for each edge**:
- Edge E (ξ¹=max):
  - v1_bnd[j] = r^T @ v1[:, j]           (extrapolate in ξ¹, for each j)
  - v2_bnd[j] = (Pcv @ v2[N, :])[j] ??   NO — v2 at ξ¹=N is v2[N,:] shape (N,)
    → Need Pcv to interpolate from (N,) cell-centers to (N+1,) vertices in ξ²
    → v2_bnd = Pcv @ v2[N, :]   # (N+1,N) @ ... no, v2[N,:] is (N,), Pcv is (N+1,N)
    → v2_bnd = Pcv @ v2[N, :]   # YES: (N+1,N) @ (N,) = (N+1,)

- Edge W (ξ¹=min):  
  - v1_bnd[j] = l^T @ v1[:, j]
  - v2_bnd = Pcv @ v2[0, :]

- Edge N (ξ²=max):
  - v1_bnd = Pcv @ v1[:, N]   # v1[:,N] is (N,), Pcv is (N+1,N) → (N+1,)
    WAIT: v1 is (N,N+1). v1[:,N] is column N, shape (N,). YES.
  - v2_bnd[i] = r^T @ v2[i, :]   for each i
    → v2_bnd = v2 @ r  # (N+1,N) @ (N,) = (N+1,) ✓

- Edge S (ξ²=min):
  - v1_bnd = Pcv @ v1[:, 0]
  - v2_bnd = v2 @ l

**Unit test (Step 2)**:
- Use an analytic covariant velocity field (from Cartesian via cartesian_to_covariant)
- Extrapolate to boundary using the function
- Compare with analytic covariant velocity evaluated directly at boundary
- Verify O(h^p) convergence (p=2 at boundary, p=4 interior)

---

### Step 3: Convert Boundary Covariant Velocity to Cartesian

**Goal**: Given covariant (v1_bnd, v2_bnd) at h-grid boundary points on a 
specific panel, compute the Cartesian velocity (Vx, Vy, Vz).

**This is just**: `covariant_to_cartesian(v1_bnd, v2_bnd, xi1_bnd, xi2_bnd, panel)`

**Unit test (Step 3)**:
- Start with known Cartesian field V(X,Y,Z)
- Project to covariant on each panel
- Extrapolate to boundary (Step 2)
- Convert back to Cartesian (Step 3)
- Compare with analytic V at boundary points
- Verify Cartesian velocity matches between adjacent panels at shared edge 
  (this MUST hold to machine precision for exact fields, or O(h^p) for 
  extrapolated fields)

---

### Step 4: Cartesian Averaging at Shared Edges

**Goal**: For each of the 12 edges, compute the Cartesian-averaged velocity.

**Algorithm for edge (pa, ea) ↔ (pb, eb)**:
1. Extrapolate covariant to boundary: (v1_a, v2_a) and (v1_b, v2_b)
2. Convert to Cartesian: V_A and V_B  (at corresponding physical points)
3. Average: V_avg = 0.5 * (V_A + V_B_aligned)   [align indices for reversed edges]
4. Convert V_avg back to covariant in panel A's frame → (v1_avg_a, v2_avg_a)
5. Convert V_avg back to covariant in panel B's frame → (v1_avg_b, v2_avg_b)

**Unit test (Step 4)**:
- Use analytic Cartesian field (solid body rotation)
- The averaged velocity at each edge should equal the analytic velocity 
  (since both sides give the same V for exact data)
- Check that v1_avg and v2_avg in each panel's frame satisfy:
  covariant_to_cartesian(v1_avg, v2_avg, panel) = V_analytic  
- This is THE critical test. If this passes, the averaging is correct.
- Run for all 3 test fields: SolidBody-Z, SolidBody-X, Dipole
- ALL 12 edges must show O(h^p) error, including axis-swap edges

---

### Step 5: Refactored SAT Using Averaged Velocity

**Goal**: Replace the current `build_sat_fn` with one that uses Cartesian-averaged
velocity to compute the mass flux penalty.

**Algorithm (per edge)**:
1. Compute Cartesian-averaged covariant velocity at boundary (Step 4)
2. Compute the "consensus" mass flux from the averaged velocity:
   - Raise index: v^i_avg = Q^{ij} v_j_avg  (using local panel's metric)
   - Mass flux: u_avg = J * v^i_avg  (normal component only)
3. SAT penalty = own_flux - consensus_flux  (scaled by Hv_inv and sign)

Concretely, for panel A at edge E:
```
  v1_avg, v2_avg = cartesian_average(...)   # covariant, in panel A's frame
  # The normal flux at E boundary comes from v^1 = Q^{11}*v1_avg + Q^{12}*v2_avg
  # Mass flux at boundary = J1 * v^1_avg
  u_avg = J_bnd * (Q11_bnd * v1_avg + Q12_bnd * v2_avg)
  
  # Extrapolated own flux
  u_own = r^T @ u1[panel_a]   (as before)
  
  # SAT penalty  
  sat = -sign * Hv_inv * (u_own - u_avg)
```

Wait — this changes the SAT structure. Currently:
  sat = -sign * 0.5 * Hv_inv * (flux_own + ss * flux_nbr)

With averaging, the "consensus" IS the average, so:
  sat = -sign * Hv_inv * (flux_own - flux_consensus)

For a smooth exact solution, flux_own ≈ flux_consensus → sat ≈ 0. ✓
For discontinuous interfaces, sat penalizes the jump. ✓

**But we need to verify energy conservation is maintained!**

The energy proof requires the SAT contributions from both sides of an edge
to telescope. With the new formulation:
  sat_a = -sign_a * Hv_inv * (flux_own_a - flux_avg)
  sat_b = -sign_b * Hv_inv * (flux_own_b - flux_avg_in_b_frame)

The flux_avg is the SAME physical flux, just expressed in each panel's coords.
Energy: h_a * Hv * sat_a + h_b * Hv * sat_b 
      = h_a * (-sign_a)(flux_a - flux_avg_a) + h_b * (-sign_b)(flux_b - flux_avg_b)
      
With h continuous (projected): h_a = h_b at matching points.
And flux_avg is constructed from the average velocity, so:
  flux_avg_a = 0.5*(flux_a + T(flux_b))  ... not exactly, because the metric
  transformation is nonlinear.

**This needs careful analysis.** The energy proof may require us to stick closer 
to Shashkin's formulation (Section 4.3), which averages the velocity vector and
then recomputes fluxes locally. The key is that both panels use the SAME 
averaged velocity vector (expressed in their own coordinates).

**Unit test (Step 5)**:
- Run the full SWE gravity wave test with refactored SAT
- Check mass conservation (should remain machine-precision)
- Check energy conservation rate
- Run both variant 1 (panel center) and variant 2 (cube vertex)
- Convergence rate for variant 2 should improve from 1.4 toward ~3.8
- Compare convergence of variant 1 (should remain ~4.2 or improve)

---

## Edge Cases and Subtleties

### Corner points (3-panel junctions)
At cube vertices, three panels meet. Shashkin says "the vector component in 
parallel direction is used." This means: at a corner, only the tangential 
component gets averaged (the normal component from the third panel doesn't 
participate because its "normal" points in a completely different direction).

We may need special handling here, but the h-projection already handles 
3-panel corners for the scalar field. For velocity, the SAT edges should 
handle corners implicitly since each edge treats its own pair of panels.

### Metric at boundary for flux reconstruction
After averaging velocity in Cartesian and converting back to covariant,
we need the metric at the h-grid boundary to reconstruct the mass flux.
The metric is available at h-grid points (compute_metric at xi_v, xi_v).

### Staggered grid subtlety
v1 lives at (xi_c, xi_v) and v2 at (xi_v, xi_c). The extrapolation to 
h-grid boundaries involves different operations for each component.
Step 2 handles this carefully.

---

## File Dependencies

```
velocity_transforms.py  ← Step 1 (DONE)
    ↓
test_step5a_extrap_cov.py  ← Step 2 (extrapolate covariant)
    ↓
test_step5b_cart_bnd.py    ← Step 3 (boundary Cartesian conversion)  
    ↓
test_step5c_cart_avg.py    ← Step 4 (Cartesian averaging, THE critical test)
    ↓
test_stag_step4.py         ← Step 5 (refactored SAT + full convergence)
```
