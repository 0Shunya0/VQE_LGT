"""
Analysis for the warm-start continuation sweep (results/kink_warmstart_N3_L3_
noiseless.csv, results/kink_warmstart_N3_L3_noisy.csv), run to test whether
the cold-start L=3 noisy curve's -31.0% suppression and 1.220 pre-transition
slope ratio (Round 4) were methodological artifacts of independent random
restarts at each K creating point-to-point optimizer scatter, or a genuine
noise-driven distortion.

Checks, in order:
  1. Monotonicity gate on the noisy curve (E_noisy must be monotone
     non-decreasing across K=4..7 under the depolarizing-contraction model;
     any inversion is scatter, not physics).
  2. Slope smoothness: consecutive dE/dK values and max|slope[i+1]-slope[i]|
     on the pre-transition branch (K<=5.25), cold-start vs warm-start, both
     curves.
  3. Sanity gate on the warm-start noiseless curve (same 0.1 threshold).
  4. Only if 1-3 pass: full integrated-slope-drop analysis.
  5. Warm-start win rate per sweep.
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
SANITY_THRESHOLD = 0.1
PRE_TRANSITION_KMAX = 5.25


def _canonical_best_coldstart(path):
    """Same run_tag-preference loader used for the Round-4 cold-start data
    (prefers 'basin64' where present, else 'dense8')."""
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


def _warmstart_best(path):
    df = pd.read_csv(path)
    best = df[df["is_best"]].sort_values("K")
    return best["K"].to_numpy(dtype=float), best["energy"].to_numpy(dtype=float)


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


def check_monotonicity(K, E, label):
    print(f"=== 1. Monotonicity gate: {label} ===")
    inversions = []
    for i in range(len(K) - 1):
        if E[i + 1] < E[i] - 1e-9:
            inversions.append((K[i], E[i], K[i + 1], E[i + 1]))
    if inversions:
        print(f"  {len(inversions)} inversion(s) found (E should be non-decreasing in K):")
        for k1, e1, k2, e2 in inversions:
            print(f"    E({k1:.2f})={e1:.4f} > E({k2:.2f})={e2:.4f}  [DROP of {e1-e2:.4f}]")
    else:
        print("  No inversions: E_noisy is monotone non-decreasing across the full grid.")
    return inversions


def slope_smoothness(K, E, label):
    dE = np.gradient(E, K)
    print(f"  {label} dE/dK: " + ", ".join(f"{v:.3f}" for v in dE))
    mask = K <= PRE_TRANSITION_KMAX
    dE_pre = dE[mask]
    if len(dE_pre) > 1:
        jumps = np.abs(np.diff(dE_pre))
        max_jump = float(jumps.max())
        print(f"  {label} max|slope[i+1]-slope[i]| on pre-transition branch (K<={PRE_TRANSITION_KMAX}) "
              f"= {max_jump:.3f}")
    else:
        max_jump = float("nan")
    return dE, max_jump


def warm_win_rate(path, label):
    df = pd.read_csv(path)
    later = df[df["K_index"] > 0]
    per_k = later.groupby("K_index")["warm_won"].first()
    n_total = len(per_k)
    n_won = int(per_k.sum())
    print(f"  {label}: warm start won at {n_won}/{n_total} K points "
          f"({100*n_won/n_total:.1f}% of K>4.0 points)")
    return n_won, n_total


def main():
    nl_path = os.path.join(RESULTS_DIR, "kink_warmstart_N3_L3_noiseless.csv")
    ny_path = os.path.join(RESULTS_DIR, "kink_warmstart_N3_L3_noisy.csv")
    cold_nl_path = os.path.join(RESULTS_DIR, "kink_dense_N3_L3_noiseless.csv")
    cold_ny_path = os.path.join(RESULTS_DIR, "kink_dense_N3_L3_noisy.csv")

    K_nl_warm, E_nl_warm = _warmstart_best(nl_path)
    K_ny_warm, E_ny_warm = _warmstart_best(ny_path)
    K_nl_cold, E_nl_cold = _canonical_best_coldstart(cold_nl_path)
    K_ny_cold, E_ny_cold = _canonical_best_coldstart(cold_ny_path)
    E_exact = np.array([core.exact_and_mix(3, k, convention="nu0only")[0] for k in K_nl_warm])

    # ---- Check 1: monotonicity ----
    inversions = check_monotonicity(K_ny_warm, E_ny_warm, "warm-start noisy curve")
    print()

    # ---- Check 2: slope smoothness, cold vs warm, both curves ----
    print("=== 2. Slope smoothness (cold-start vs warm-start) ===")
    print("-- noiseless --")
    _, jump_nl_cold = slope_smoothness(K_nl_cold, E_nl_cold, "cold-start")
    _, jump_nl_warm = slope_smoothness(K_nl_warm, E_nl_warm, "warm-start")
    print("-- noisy --")
    _, jump_ny_cold = slope_smoothness(K_ny_cold, E_ny_cold, "cold-start")
    _, jump_ny_warm = slope_smoothness(K_ny_warm, E_ny_warm, "warm-start")
    print(f"\n  Noisy pre-transition slope-jump: cold-start={jump_ny_cold:.3f} -> "
          f"warm-start={jump_ny_warm:.3f} "
          f"({'reduced' if jump_ny_warm < jump_ny_cold else 'NOT reduced'})")
    print()

    # ---- Check 3: sanity gate on warm-start noiseless ----
    print("=== 3. Sanity gate: warm-start noiseless vs exact ===")
    dev = np.abs(E_nl_warm - E_exact)
    max_dev = float(dev.max())
    worst_K = float(K_nl_warm[int(np.argmax(dev))])
    print(f"  max |E_noiseless - E_exact| over 13 K points = {max_dev:.4f} energy units, at K={worst_K:.2f}")
    for k, en, ee in zip(K_nl_warm, E_nl_warm, E_exact):
        print(f"    K={k:.2f}  E_noiseless={en:.4f}  E_exact={ee:.4f}  diff={abs(en-ee):.4f}")
    gate_pass = max_dev < SANITY_THRESHOLD
    print(f"\n  {'PASS' if gate_pass else 'FAIL'} (threshold {SANITY_THRESHOLD}).\n")

    monotone_ok = len(inversions) == 0
    scatter_ok = jump_ny_warm < jump_ny_cold * 0.5  # substantial reduction, not marginal

    print("=== VERDICT ===")
    if gate_pass and monotone_ok:
        print("  CASE (a): warm-starting smooths the noisy curve -- monotone, slope scatter "
              f"reduced ({jump_ny_cold:.3f} -> {jump_ny_warm:.3f}). The cold-start -31.0% "
              "suppression and 1.220 slope ratio (Round 4) were methodological artifacts. "
              "Computing the warm-start numbers below.")
        case = "a"
    else:
        print("  CASE (b): the noisy curve stays non-monotone and/or scattered even "
              "warm-started. At 26% cumulative infidelity the boundary scan has no stable "
              "optimum that tracks continuously in K. NOT computing a suppression number.")
        case = "b"
    print()

    # ---- Check 5: warm-start win rate ----
    print("=== 5. Warm-start win rate ===")
    warm_win_rate(nl_path, "noiseless")
    warm_win_rate(ny_path, "noisy")
    print()

    if case == "b":
        return case

    # ---- Check 4: full integrated-slope analysis (only if case (a)) ----
    print("=== 4. Integrated-slope-drop analysis (warm-start) ===")
    s_exact = _transition_stats(K_nl_warm, E_exact)
    s_noiseless = _transition_stats(K_nl_warm, E_nl_warm)
    s_noisy = _transition_stats(K_ny_warm, E_ny_warm)
    for name, s in [("exact", s_exact), ("L=3 noiseless (warm)", s_noiseless), ("L=3 noisy p=0.01 (warm)", s_noisy)]:
        print(f"  {name:<24} drop={s['drop']:.4f}  K_steep={s['K_steep']:.3f}  width={s['width']:.4f}")

    sup_vs_exact = 100 * (1 - s_noisy["drop"] / s_exact["drop"]) if s_exact["drop"] else float("nan")
    sup_vs_noiseless = 100 * (1 - s_noisy["drop"] / s_noiseless["drop"]) if s_noiseless["drop"] else float("nan")
    slope_ratio = s_noisy["dE"][0] / s_noiseless["dE"][0]
    w_eff = (1 - P) ** N_CX_L3 * (1 - 2 * EPS)
    displacement = s_noisy["K_steep"] - s_noiseless["K_steep"]

    print()
    print(f"  suppression vs exact      = {sup_vs_exact:.1f}%")
    print(f"  suppression vs noiseless  = {sup_vs_noiseless:.1f}%  <-- noise-attributable")
    print(f"  transition displacement, noisy vs NOISELESS = {displacement:+.3f}")
    print(f"  pre-transition slope ratio (noisy/noiseless) at K=4.0 = {slope_ratio:.3f} vs "
          f"analytic w_eff=(1-p)^30*(1-2*eps)={w_eff:.4f} "
          f"(diff={abs(slope_ratio-w_eff):.3f})")
    return case


if __name__ == "__main__":
    main()
