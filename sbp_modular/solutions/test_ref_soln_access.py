import os, zarr, numpy as np

ref_dir = os.environ.get("SBP_REF_DIR", "reference_solutions")
print(f"SBP_REF_DIR = {ref_dir}")

for variant in [1, 2]:
    for N in [24, 48, 96, 192]:
        path = os.path.join(ref_dir, f"gauss{variant}", f"N{N:03d}.zarr")
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            continue
        store = zarr.open(path, mode='r')
        h = np.array(store['h_exact'])
        print(f"  gauss{variant}/N{N:03d}: shape={h.shape}, "
              f"min={h.min():.6e}, max={h.max():.6e}")
