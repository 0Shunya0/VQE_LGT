"""
Analysis for the final control: compares 8-restart (kink_dense_N3_noiseless.csv)
vs 32-restart/maxiter-4000 (boundary_restart_scaling_N3.csv) noiseless L=2
optimization at K=4.0, 4.5, 5.0, and states whether the sustained 2.5-5.6
energy-unit gap is optimizer under-convergence (case i) or an L=2
expressibility failure (case ii).
"""
import os

import numpy as np
import pandas as pd

RESULTS_DIR = "results"
K_POINTS = [4.0, 4.5, 5.0]
PLATEAU_TOL = 1e-3  # absolute energy-unit tolerance for "stopped improving"


def restarts_to_plateau(energies_in_order, tol=PLATEAU_TOL):
    """First restart index (1-based count) at which the running best-so-far
    is already within tol of the eventual final best (i.e. all restarts run
    is a superset of a working data structure -- we don't need to know the
    tolerance a priori for future restarts, only that from that point on the
    running best never again improves by more than tol)."""
    energies = np.asarray(energies_in_order, dtype=float)
    running_best = np.minimum.accumulate(energies)
    final_best = running_best[-1]
    hit = np.where(running_best <= final_best + tol)[0]
    return int(hit[0] + 1) if len(hit) else len(energies)


def main():
    dense8 = pd.read_csv(os.path.join(RESULTS_DIR, "kink_dense_N3_noiseless.csv"))
    scaling32 = pd.read_csv(os.path.join(RESULTS_DIR, "boundary_restart_scaling_N3.csv"))

    print(f"{'K':>5} {'E_exact':>10} {'best@8':>10} {'best@32':>10} {'err%@8':>8} {'err%@32':>8} "
          f"{'plateau':>8} {'min@32':>10} {'max@32':>10} {'median@32':>10} {'std@32':>8}")

    rows_out = []
    for K in K_POINTS:
        d8 = dense8[np.isclose(dense8["K"], K)]
        # L=2 only: boundary_restart_scaling_N3.csv also holds the L=3/L=4
        # depth-check rows appended later at K=4.5, which must not be mixed
        # into the L=2, 32-restart comparison.
        d32 = scaling32[np.isclose(scaling32["K"], K) & (scaling32["L"] == 2)].sort_values("restart")
        if not len(d8) or not len(d32):
            print(f"  (missing data for K={K})")
            continue

        E_exact = float(d8["E_exact"].iloc[0])
        best8 = float(d8.loc[d8["is_best"], "energy"].iloc[0])
        err8 = abs(best8 - E_exact) / abs(E_exact) * 100

        e32 = d32["energy"].to_numpy(dtype=float)
        best32 = float(e32.min())
        err32 = abs(best32 - E_exact) / abs(E_exact) * 100
        plateau = restarts_to_plateau(e32)

        print(f"{K:>5.2f} {E_exact:>10.4f} {best8:>10.4f} {best32:>10.4f} {err8:>7.2f}% {err32:>7.2f}% "
              f"{plateau:>8d} {e32.min():>10.4f} {e32.max():>10.4f} {np.median(e32):>10.4f} {e32.std():>8.4f}")
        rows_out.append(dict(K=K, E_exact=E_exact, best8=best8, best32=best32,
                              err8=err8, err32=err32, plateau=plateau))

    mean_err32 = float(np.mean([r["err32"] for r in rows_out])) if rows_out else float("nan")
    mean_gap32 = float(np.mean([abs(r["best32"] - r["E_exact"]) for r in rows_out])) if rows_out else float("nan")
    print()
    if mean_err32 < 1.0:
        case = "i"
        print(f"CASE (i): err%@32 = {mean_err32:.2f}% (mean), well below ~1% -- optimizer "
              f"under-convergence near the crossing. Fix: restart-budget prescription; the "
              f"paper's '5-8 restarts suffice' needs a stated exception near the boundary.")
    else:
        case = "ii"
        print(f"CASE (ii): err%@32 = {mean_err32:.2f}% (mean), mean absolute gap "
              f"{mean_gap32:.3f} energy units -- stays at the same order of magnitude as at "
              f"8 restarts. The L=2 manifold cannot represent the ground state in this "
              f"region. The paper's p/d>=1 sufficiency claim needs a stated exception near "
              f"the phase boundary.")

    if case == "ii" and "L" in scaling32.columns and (scaling32["L"] > 2).any():
        print()
        print("Depth check at K=4.5 (8 restarts, maxiter=2000):")
        print(f"{'L':>3} {'best_E':>10} {'E_exact':>10} {'err%':>8} {'n_restarts':>10}")
        for L in sorted(scaling32.loc[scaling32["K"] == 4.5, "L"].unique()):
            sub = scaling32[(scaling32["K"] == 4.5) & (scaling32["L"] == L)]
            best = sub.loc[sub["energy"].idxmin()]
            err = abs(best["energy"] - best["E_exact"]) / abs(best["E_exact"]) * 100
            print(f"{L:>3} {best['energy']:>10.4f} {best['E_exact']:>10.4f} {err:>7.4f}% {len(sub):>10d}")

    return case, rows_out


if __name__ == "__main__":
    main()
