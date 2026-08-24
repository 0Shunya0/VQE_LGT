"""
Analysis for the bidirectional (hysteresis) continuation sweep
(results/kink_hysteresis_N3_L3_noiseless.csv, results/kink_hysteresis_N3_L3_
noisy.csv). Standard method for locating a first-order transition
numerically: sweep K ascending and descending, each warm-started in its own
direction; where the two branches disagree by more than a tolerance is the
transition region (parameter continuation cannot cross a first-order
boundary, so a one-directional sweep's "monotonicity gate" was never a fair
test across that region -- this is the fix).
"""
import os

import numpy as np
import pandas as pd

import schwinger_core as core

RESULTS_DIR = "results"
P = 0.01
EPS = core.SPAM_DEFAULT
N_CX_L3 = 30
W_EFF = (1 - P) ** N_CX_L3 * (1 - 2 * EPS)
HYSTERESIS_TOL = 1e-3
PRE_RANGE = (4.25, 5.25)
POST_RANGE = (6.00, 7.00)


def _direction_best(path, direction):
    df = pd.read_csv(path)
    sub = df[(df["direction"] == direction) & (df["is_best"])].sort_values("K")
    return sub["K"].to_numpy(dtype=float), sub["energy"].to_numpy(dtype=float)


def _linfit(K, E):
    """Least-squares line E = a*K + b over the given points; returns
    (slope, max_abs_residual)."""
    slope, intercept = np.polyfit(K, E, 1)
    resid = np.abs(E - (slope * K + intercept))
    return float(slope), float(resid.max())


def _select_range(K, E, lo, hi):
    mask = (K >= lo - 1e-9) & (K <= hi + 1e-9)
    return K[mask], E[mask]


def report_curve(path, label):
    K_up, E_up = _direction_best(path, "up")
    K_down, E_down = _direction_best(path, "down")
    K_down_sorted_idx = np.argsort(K_down)
    K_down, E_down = K_down[K_down_sorted_idx], E_down[K_down_sorted_idx]
    assert np.allclose(K_up, K_down), f"{path}: up/down grids differ"
    K = K_up

    E_combined = np.minimum(E_up, E_down)

    print(f"=== {label}: up / down / combined ===")
    for k, eu, ed, ec in zip(K, E_up, E_down, E_combined):
        flag = " <-- disagree" if abs(eu - ed) > HYSTERESIS_TOL else ""
        print(f"    K={k:.2f}  up={eu:10.4f}  down={ed:10.4f}  |diff|={abs(eu-ed):.4f}  "
              f"combined={ec:10.4f}{flag}")

    diff = np.abs(E_up - E_down)
    disagree = diff > HYSTERESIS_TOL
    if disagree.any():
        idx = np.where(disagree)[0]
        window = (float(K[idx.min()]), float(K[idx.max()]))
        print(f"  Hysteresis window (|up-down|>{HYSTERESIS_TOL}, literal): K in [{window[0]:.2f}, {window[1]:.2f}]")
    else:
        window = None
        print(f"  No hysteresis window found (up and down agree everywhere within {HYSTERESIS_TOL}).")

    LARGE_TOL = 0.05
    large = diff > LARGE_TOL
    if large.any():
        idx = np.where(large)[0]
        large_window = (float(K[idx.min()]), float(K[idx.max()]))
        print(f"  Large-disagreement subset (|up-down|>{LARGE_TOL}): K in {sorted(K[idx].tolist())}")
    else:
        large_window = None
        print(f"  No K point exceeds the {LARGE_TOL} large-disagreement threshold.")

    # crossing point: where sign(up-down) changes
    s = np.sign(E_up - E_down)
    crossing = None
    for i in range(len(K) - 1):
        if s[i] != 0 and s[i + 1] != 0 and s[i] != s[i + 1]:
            # linear interpolate where (up-down) crosses zero
            d0, d1 = E_up[i] - E_down[i], E_up[i + 1] - E_down[i + 1]
            frac = d0 / (d0 - d1) if (d0 - d1) != 0 else 0.5
            crossing = float(K[i] + frac * (K[i + 1] - K[i]))
            break
    print(f"  Crossing point (up/down swap which is lower): "
          f"{'K~%.3f' % crossing if crossing is not None else 'none found'}")
    print()
    return K, E_up, E_down, E_combined, window, large_window


MONOTONICITY_TOL = 1e-4  # COBYLA-converged energies carry ~1e-6-1e-7 residual
                          # noise even on an exactly flat plateau; 1e-9 flags
                          # that noise as a false "inversion".


def monotonicity_gate(K, E, window, label):
    print(f"=== Monotonicity gate: {label} (combined curve) ===")
    inversions = []
    for i in range(len(K) - 1):
        if E[i + 1] < E[i] - MONOTONICITY_TOL:
            inside = window is not None and (window[0] - 0.3 <= K[i] <= window[1] + 0.3)
            inversions.append((K[i], E[i], K[i + 1], E[i + 1], inside))
    if not inversions:
        print("  No inversions.")
    else:
        for k1, e1, k2, e2, inside in inversions:
            tag = "INSIDE hysteresis window (expected -- first-order signature)" if inside else "OUTSIDE window -- genuine problem"
            print(f"  E({k1:.2f})={e1:.4f} > E({k2:.2f})={e2:.4f}  [{tag}]")
    outside_count = sum(1 for *_, inside in inversions if not inside)
    print()
    return inversions, outside_count


def slope_analysis(K, E_noiseless, E_noisy, label_prefix=""):
    print(f"=== Slope analysis {label_prefix}(on-branch linear fits) ===")
    results = {}
    for name, (lo, hi) in [("pre-transition", PRE_RANGE), ("post-transition", POST_RANGE)]:
        Kp, Ep_nl = _select_range(K, E_noiseless, lo, hi)
        _, Ep_ny = _select_range(K, E_noisy, lo, hi)
        slope_nl, resid_nl = _linfit(Kp, Ep_nl)
        slope_ny, resid_ny = _linfit(Kp, Ep_ny)
        ratio = slope_ny / slope_nl if abs(slope_nl) > 1e-6 else float("nan")
        print(f"  {name} branch, K in [{lo},{hi}]:")
        print(f"    noiseless: slope={slope_nl:.4f}, max residual={resid_nl:.4f}")
        print(f"    noisy:     slope={slope_ny:.4f}, max residual={resid_ny:.4f}")
        if np.isnan(ratio):
            print(f"    slope ratio: UNDEFINED (noiseless slope ~0 here -- E_exact is flat on this "
                  f"branch by construction; a ratio against ~0 is not meaningful, see kink-magnitude "
                  f"analysis below instead)")
        else:
            print(f"    slope ratio (noisy/noiseless) = {ratio:.4f} vs w_eff={W_EFF:.4f} "
                  f"(diff={abs(ratio-W_EFF):.4f})")
        results[name] = dict(slope_nl=slope_nl, slope_ny=slope_ny, resid_nl=resid_nl,
                              resid_ny=resid_ny, ratio=ratio)
    print()
    return results


def offset_analysis(K, E_noiseless, E_noisy):
    print("=== Offset analysis: E_noisy - E_noiseless, per branch ===")
    for name, (lo, hi) in [("pre-transition", PRE_RANGE), ("post-transition", POST_RANGE)]:
        Kp, Ep_nl = _select_range(K, E_noiseless, lo, hi)
        _, Ep_ny = _select_range(K, E_noisy, lo, hi)
        offset = Ep_ny - Ep_nl
        print(f"  {name} (K in [{lo},{hi}]):")
        for k, o in zip(Kp, offset):
            print(f"    K={k:.2f}: offset={o:.4f}")
        print(f"    mean={offset.mean():.4f}, std={offset.std():.4f} "
              f"({'consistent with constant' if offset.std() < 0.05 * abs(offset.mean()) else 'NOT constant'})")
    print()


def explain_pretransition_ratio():
    """The pre-transition slope ratio (noisy/noiseless) came out at ~1.0,
    far from w_eff -- not a bug, since the mixing model predicts
    ratio = w_eff + (1-w_eff)*(dEmix/dK)/(dEexact/dK), which only equals
    w_eff if dEmix/dK << dEexact/dK. Check whether E_mix is close to as
    steep as E_exact on this branch, which would explain a ratio near 1
    instead of near w_eff."""
    Kp = np.linspace(PRE_RANGE[0], PRE_RANGE[1], 5)
    Eexact = np.array([core.exact_and_mix(3, k, convention="nu0only")[0] for k in Kp])
    Emix = np.array([core.exact_and_mix(3, k, convention="nu0only")[1] for k in Kp])
    dEexact_dK, _ = _linfit(Kp, Eexact)
    dEmix_dK, _ = _linfit(Kp, Emix)
    predicted_ratio = W_EFF + (1 - W_EFF) * (dEmix_dK / dEexact_dK)
    print("=== Explaining the pre-transition slope ratio ===")
    print(f"  dE_exact/dK = {dEexact_dK:.4f}, dE_mix/dK = {dEmix_dK:.4f} on K in {PRE_RANGE}")
    print(f"  mixing-model-predicted ratio = w_eff + (1-w_eff)*(dEmix/dEexact) = {predicted_ratio:.4f}")
    print(f"  (observed ratio ~1.0 from slope_analysis above)")
    print()


def kink_and_baseline(slopes):
    print("=== Kink magnitude and suppression ===")
    kink_nl = slopes["pre-transition"]["slope_nl"] - slopes["post-transition"]["slope_nl"]
    kink_ny = slopes["pre-transition"]["slope_ny"] - slopes["post-transition"]["slope_ny"]
    suppression = 100 * (1 - kink_ny / kink_nl) if kink_nl else float("nan")
    print(f"  kink_noiseless = {kink_nl:.4f}, kink_noisy = {kink_ny:.4f}")
    print(f"  suppression = 1 - kink_noisy/kink_noiseless = {suppression:.1f}% (raw)")

    # analytic (1-w)*dEmix/dK baseline on the post-transition branch
    Kp = np.linspace(POST_RANGE[0], POST_RANGE[1], 5)
    Emix = np.array([core.exact_and_mix(3, k, convention="nu0only")[1] for k in Kp])
    dEmix_dK, _ = _linfit(Kp, Emix)
    baseline = (1 - W_EFF) * dEmix_dK
    raw_post_slope_ny = slopes["post-transition"]["slope_ny"]
    adjusted_post_slope_ny = raw_post_slope_ny - baseline
    print(f"  post-transition noisy slope: raw={raw_post_slope_ny:.4f}")
    print(f"  analytic baseline (1-w_eff)*dE_mix/dK = {baseline:.4f} (dE_mix/dK={dEmix_dK:.4f}, "
          f"w_eff={W_EFF:.4f})")
    print(f"  adjusted post-transition noisy slope (raw - baseline) = {adjusted_post_slope_ny:.4f}")
    kink_ny_adjusted = slopes["pre-transition"]["slope_ny"] - adjusted_post_slope_ny
    suppression_adjusted = 100 * (1 - kink_ny_adjusted / kink_nl) if kink_nl else float("nan")
    print(f"  kink_noisy (baseline-adjusted) = {kink_ny_adjusted:.4f}, "
          f"suppression (adjusted) = {suppression_adjusted:.1f}%")
    print()
    return dict(kink_nl=kink_nl, kink_ny=kink_ny, suppression=suppression,
                baseline=baseline, kink_ny_adjusted=kink_ny_adjusted,
                suppression_adjusted=suppression_adjusted)


def main():
    nl_path = os.path.join(RESULTS_DIR, "kink_hysteresis_N3_L3_noiseless.csv")
    ny_path = os.path.join(RESULTS_DIR, "kink_hysteresis_N3_L3_noisy.csv")

    K_nl, up_nl, down_nl, comb_nl, window_nl, large_nl = report_curve(nl_path, "NOISELESS")
    K_ny, up_ny, down_ny, comb_ny, window_ny, large_ny = report_curve(ny_path, "NOISY")
    assert np.allclose(K_nl, K_ny)
    K = K_nl

    # Use the physically-meaningful (large-disagreement) window for the
    # monotonicity carve-out, not the literal 1e-3 window -- at p=0.01 the
    # optimizer's own noise floor (COBYLA converging to slightly different
    # points in a noise-flattened landscape) already exceeds 1e-3 almost
    # everywhere, which would make the literal window meaningless as a
    # "this is the transition" signal. See hysteresis-window magnitude
    # breakdown printed above.
    inv_nl, outside_nl = monotonicity_gate(K, comb_nl, large_nl, "noiseless")
    inv_ny, outside_ny = monotonicity_gate(K, comb_ny, large_ny, "noisy")

    E_exact = np.array([core.exact_and_mix(3, k, convention="nu0only")[0] for k in K])
    max_dev_nl = float(np.max(np.abs(comb_nl - E_exact)))
    print(f"Sanity check, noiseless combined curve vs exact: max dev = {max_dev_nl:.4f}\n")

    slopes = slope_analysis(K, comb_nl, comb_ny)
    offset_analysis(K, comb_nl, comb_ny)
    explain_pretransition_ratio()

    print("=== VERDICT ===")
    pre_ok = slopes["pre-transition"]["resid_nl"] < 0.05 and slopes["pre-transition"]["resid_ny"] < 0.5
    post_ok = slopes["post-transition"]["resid_nl"] < 0.05 and slopes["post-transition"]["resid_ny"] < 0.5
    if outside_nl == 0 and outside_ny == 0 and pre_ok and post_ok:
        case = "a"
        print("  CASE (a): combined curve monotone outside the hysteresis window, both branches "
              "linear within small residuals. Computing kink suppression and reporting as the "
              "paper's numbers.")
        kink_result = kink_and_baseline(slopes)
    else:
        case = "b"
        print(f"  CASE (b): genuine instability remains (inversions outside window: "
              f"noiseless={outside_nl}, noisy={outside_ny}; branch-fit residuals: "
              f"pre_ok={pre_ok}, post_ok={post_ok}). NOT computing a suppression number.")
        kink_result = None

    return case, kink_result


if __name__ == "__main__":
    main()
