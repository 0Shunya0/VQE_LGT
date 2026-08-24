"""
Analysis for the L=3 redo of the dense kink sweep (results/kink_dense_N3_L3_
noiseless.csv, results/kink_dense_N3_L3_noisy.csv), which supersedes the L=2
dense sweep invalidated by results/boundary_restart_scaling_N3.csv (L=2
cannot represent the ground state near K=4-5.25; L=3 recovers it).

Step 1: sanity gate -- confirm the L=3 noiseless curve actually tracks exact
across the whole window (expected ~0.01 energy units, not 3-5 like L=2).
Only proceeds to the transition analysis if this passes.

Step 2: integrated-slope-drop metric (NOT the fixed-K jump metric that
produced the discredited 99.6% L=2 figure) on all three curves -- exact,
L=3 noiseless, L=3 noisy (p=0.01) -- computed on the identical 13-point grid,
reporting suppression vs exact AND vs noiseless (the noise-attributable
number), the depolarizing-contraction check against the analytic w_eff, and
the noisy-vs-NOISELESS transition displacement.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os

import numpy as np
import pandas as pd

import schwinger.core as core

RESULTS_DIR = "results"
P = 0.01
EPS = core.SPAM_DEFAULT
N_CX_L3 = 30
SANITY_THRESHOLD = 0.1  # energy units; "order 0.01" expected, this is a generous margin


def _canonical_best(path):
    """For each K, prefer run_tag='basin64' rows (the 64-restart,
    maxiter=4000 basin-capture re-check at K=5.25/5.50/5.75) if present for
    that K; otherwise fall back to the original 'dense8' (8-restart)
    rows -- so the sanity gate and transition analysis both automatically
    use whichever data is canonical for a given K without needing separate
    code paths."""
    df = pd.read_csv(path)
    out = []
    for K, grp in df.groupby("K"):
        if "run_tag" in grp.columns and (grp["run_tag"] == "basin64").any():
            sub = grp[grp["run_tag"] == "basin64"]
        else:
            sub = grp
        row = sub.loc[sub["energy"].idxmin()]
        out.append((float(K), float(row["energy"])))
    out.sort(key=lambda t: t[0])
    return np.array([t[0] for t in out]), np.array([t[1] for t in out])


def report_basin_structure(path, K=5.50):
    """Full sorted list of the 64 final energies at K, so the basin
    structure (how many landed near the true value 0.4340 vs the
    competing polarized-state value 1.0000) is directly visible."""
    df = pd.read_csv(path)
    sub = df[(np.isclose(df["K"], K)) & (df.get("run_tag") == "basin64")]
    if not len(sub):
        print(f"  (no basin64 data at K={K} in {path})")
        return
    energies = np.sort(sub["energy"].to_numpy(dtype=float))
    E_exact = float(sub["E_exact"].iloc[0])
    print(f"  All {len(energies)} final energies at K={K} ({os.path.basename(path)}), sorted:")
    print("    " + ", ".join(f"{e:.4f}" for e in energies))
    near_true = int(np.sum(np.abs(energies - E_exact) < 0.05))
    near_polarized = int(np.sum(np.abs(energies - 1.0) < 0.05))
    other = len(energies) - near_true - near_polarized
    best = float(energies.min())
    print(f"  E_exact={E_exact:.4f}. Landed near true value (within 0.05): {near_true}/{len(energies)}. "
          f"Landed near polarized plateau 1.0000 (within 0.05): {near_polarized}/{len(energies)}. "
          f"Other: {other}. Best restart reached {best:.4f} (err vs exact = "
          f"{abs(best-E_exact)/abs(E_exact)*100:.4f}%).")
    return energies, E_exact


def _transition_stats(K_grid, E_curve):
    K_grid = np.asarray(K_grid, dtype=float)
    dE = np.gradient(np.asarray(E_curve, dtype=float), K_grid)
    dmax, dmin = float(np.max(dE)), float(np.min(dE))
    drop = dmax - dmin
    d2 = np.abs(np.diff(dE))
    i_steep = int(np.argmax(d2))
    K_steep = float(0.5 * (K_grid[i_steep] + K_grid[i_steep + 1]))
    if drop == 0:
        return dict(drop=0.0, K_steep=K_steep, width=float("nan"), dE=dE)
    s = (dE - dmin) / drop

    def find_cross(level):
        for j in range(len(s) - 1):
            a, b = s[j] - level, s[j + 1] - level
            if a * b <= 0 and s[j] != s[j + 1]:
                frac = (level - s[j]) / (s[j + 1] - s[j])
                return float(K_grid[j] + frac * (K_grid[j + 1] - K_grid[j]))
        return None

    K90, K10 = find_cross(0.9), find_cross(0.1)
    width = abs(K10 - K90) if (K90 is not None and K10 is not None) else float("nan")
    return dict(drop=drop, K_steep=K_steep, width=width, dE=dE)


def report_infidelity_headline():
    """L=2 vs L=3 cumulative infidelity at p=0.01, side by side -- a
    headline result for the paper, not a footnote."""
    print("=== Cumulative infidelity, L=2 vs L=3, at p=0.01 (per the task's framing) ===")
    for L, n_cx in [(2, core.NCX_N3_L2), (3, 30)]:
        infidelity = (1 - (1 - 0.01) ** n_cx) * 100
        w_eff = (1 - 0.01) ** n_cx * (1 - 2 * core.SPAM_DEFAULT)
        print(f"  L={L}: n_CX={n_cx}, cumulative infidelity 1-(0.99)^{n_cx} = {infidelity:.1f}%, "
              f"w_eff = (1-p)^{n_cx}*(1-2*eps) = {w_eff:.4f}")
    print()


def main():
    report_infidelity_headline()

    noiseless_path = os.path.join(RESULTS_DIR, "kink_dense_N3_L3_noiseless.csv")
    noisy_path = os.path.join(RESULTS_DIR, "kink_dense_N3_L3_noisy.csv")

    print("=== Basin structure at K=5.50 (the sanity-gate failure point) ===")
    nl_basin = report_basin_structure(noiseless_path, K=5.50)
    print()
    report_basin_structure(noisy_path, K=5.50)
    print()

    K_nl, E_noiseless = _canonical_best(noiseless_path)
    E_exact = np.array([core.exact_and_mix(3, k, convention="nu0only")[0] for k in K_nl])

    max_dev = float(np.max(np.abs(E_noiseless - E_exact)))
    worst_idx = int(np.argmax(np.abs(E_noiseless - E_exact)))
    print("=== Sanity gate: does L=3 noiseless track exact across K=[4,7]? ===")
    print(f"  (using run_tag='basin64' at K=5.25/5.50/5.75 where available, 'dense8' elsewhere)")
    print(f"  max |E_noiseless - E_exact| over 13 K points = {max_dev:.4f} energy units, "
          f"at K={K_nl[worst_idx]:.2f}")
    for k, en, ee in zip(K_nl, E_noiseless, E_exact):
        print(f"    K={k:.2f}  E_noiseless={en:.4f}  E_exact={ee:.4f}  diff={abs(en-ee):.4f}")

    if max_dev >= SANITY_THRESHOLD:
        print(f"\n  FAIL: max deviation {max_dev:.4f} >= threshold {SANITY_THRESHOLD}. "
              f"L=3 is NOT sufficient across the full window even at 64 restarts. STOPPING -- "
              f"the transition analysis below is NOT computed (no suppression numbers).")
        if nl_basin is not None:
            energies, E_exact_k = nl_basin
            hit_rate = float(np.mean(np.abs(energies - E_exact_k) < 0.05))
            best_reached = float(energies.min())
            closer_than_plateau = bool(np.any(np.abs(energies - E_exact_k) < np.abs(energies - 1.0)))
            print(f"\n  Basin-hit-rate extrapolation at K=5.50: {hit_rate*100:.1f}% of 64 restarts "
                  f"landed in the true-value basin.")
            if hit_rate > 0:
                for conf in (0.5, 0.9, 0.95, 0.99):
                    n_needed = int(np.ceil(np.log(1 - conf) / np.log(1 - hit_rate)))
                    print(f"    restarts needed for {conf*100:.0f}% chance of >=1 hit: {n_needed}")
            else:
                print(f"    0/64 restarts hit the true-value basin -- hit rate is 0, so no restart "
                      f"count can be extrapolated from this data; more restarts alone may not be "
                      f"the fix (or the true basin is astronomically small).")
            print(f"    Best restart reached {best_reached:.4f} "
                  f"({'closer to the true value than to the polarized plateau' if closer_than_plateau else 'no restart got closer to the true value than to the polarized plateau (1.0000)'}).")
        return
    print(f"\n  PASS: max deviation {max_dev:.4f} < threshold {SANITY_THRESHOLD}.\n")

    K_ny, E_noisy = _canonical_best(noisy_path)
    assert np.allclose(K_nl, K_ny), "noiseless and noisy sweeps used different K grids"

    s_exact = _transition_stats(K_nl, E_exact)
    s_noiseless = _transition_stats(K_nl, E_noiseless)
    s_noisy = _transition_stats(K_nl, E_noisy)

    print("=== Transition stats (integrated-slope-drop metric, K in [4,7]) ===")
    for name, s in [("exact", s_exact), ("L=3 noiseless", s_noiseless), ("L=3 noisy p=0.01", s_noisy)]:
        print(f"  {name:<18} drop={s['drop']:.4f} (from {s['dE'].max():.4f} to {s['dE'].min():.4f})  "
              f"K_steep={s['K_steep']:.3f}  width={s['width']:.4f}")

    sup_vs_exact = 100 * (1 - s_noisy["drop"] / s_exact["drop"]) if s_exact["drop"] else float("nan")
    sup_vs_noiseless = 100 * (1 - s_noisy["drop"] / s_noiseless["drop"]) if s_noiseless["drop"] else float("nan")

    print()
    print(f"  suppression vs exact      = 1 - drop_noisy/drop_exact      = {sup_vs_exact:.1f}%")
    print(f"  suppression vs noiseless  = 1 - drop_noisy/drop_noiseless  = {sup_vs_noiseless:.1f}%  <-- noise-attributable")
    ansatz_share = sup_vs_exact - sup_vs_noiseless
    print(f"  ansatz-attributable share (suppression vs exact MINUS suppression vs noiseless) = {ansatz_share:.1f} points")
    print(f"  i.e. of the total {sup_vs_exact:.1f}% reduction from exact, {sup_vs_noiseless:.1f} points "
          f"is attributable to noise (L=3 noiseless vs L=3 noisy) and the remaining "
          f"{ansatz_share:.1f} points is L=3's own residual gap from exact "
          f"(sanity-gate max deviation {max_dev:.4f}, so this remainder should be small).")

    slope_ratio = s_noisy["dE"][0] / s_noiseless["dE"][0]
    w_eff = (1 - P) ** N_CX_L3 * (1 - 2 * EPS)
    print()
    print(f"  Pre-transition slope at K=4.0: noiseless={s_noiseless['dE'][0]:.4f}, "
          f"noisy={s_noisy['dE'][0]:.4f}, ratio (noisy/noiseless)={slope_ratio:.4f}")
    print(f"  Analytic w_eff = (1-p)^{N_CX_L3} * (1-2*eps) = {w_eff:.4f}")
    print(f"  Difference: {abs(slope_ratio - w_eff):.4f} "
          f"({'agree within 0.05' if abs(slope_ratio - w_eff) < 0.05 else 'do NOT closely agree'})")

    displacement_vs_noiseless = s_noisy["K_steep"] - s_noiseless["K_steep"]
    print()
    print(f"  Transition displacement, noisy vs NOISELESS (the noise-attributable displacement): "
          f"K_steep(noisy)={s_noisy['K_steep']:.3f} - K_steep(noiseless)={s_noiseless['K_steep']:.3f} "
          f"= {displacement_vs_noiseless:+.3f}")
    print(f"  Width change, noisy vs noiseless: {s_noisy['width']:.3f} vs {s_noiseless['width']:.3f} "
          f"({'broadened' if s_noisy['width'] > s_noiseless['width'] else 'not broadened'})")


if __name__ == "__main__":
    main()
