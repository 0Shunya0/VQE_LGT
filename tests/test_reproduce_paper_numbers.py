"""
Parses results/*.csv (and, for the kink-suppression number, re-runs the
Round-6 hysteresis analysis) and asserts the headline numbers recorded in
results/SUMMARY.md -- this repo's source of truth for "the paper's numbers"
(no main_final.tex is tracked in this repository; see README.md).

Run: python -m pytest tests/test_reproduce_paper_numbers.py
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
RESULTS_DIR = os.path.join(ROOT, "results")


def _csv(name):
    return pd.read_csv(os.path.join(RESULTS_DIR, name))


def test_gradient_variance():
    """GI: 24.15/23.02/21.19 at N=2,3,4. HW: 10.15/5.50/2.38."""
    df = _csv("gradvar_hw.csv")
    gi = {int(r.N): r.grad_var_mean for r in df[df.ansatz == "GI"].itertuples()}
    hw = {int(r.N): r.grad_var_mean for r in df[df.ansatz == "HW"].itertuples()}
    for N, expected in {2: 24.15, 3: 23.02, 4: 21.19}.items():
        assert abs(gi[N] - expected) < 0.01, f"GI N={N}: {gi[N]} vs {expected}"
    for N, expected in {2: 10.15, 3: 5.50, 4: 2.38}.items():
        assert abs(hw[N] - expected) < 0.01, f"HW N={N}: {hw[N]} vs {expected}"


def test_inloop_zne_boundary_residuals():
    """In-loop ZNE boundary residuals 0.86 / 1.23 / 4.14 at p=0.01/0.02/0.05,
    at the grid K nearest the N=3 boundary (|K|~5.6)."""
    df = _csv("inloop_zne_N3_mitigated.csv")
    k_near = df["K"].iloc[(df["K"] - 5.6).abs().argmin()]
    expected = {0.01: 0.86, 0.02: 1.23, 0.05: 4.14}
    for p, exp in expected.items():
        row = df[(df["p"] == p) & (df["K"] == k_near)].iloc[0]
        residual = abs(row["E_mit_linear"] - row["E_exact"])
        assert abs(residual - exp) < 0.01, f"p={p}: {residual} vs {exp}"


def test_n4_noiseless_control_mean_error():
    """N=4 noiseless control, mean err% at the noisy_N4_spot.csv-matched K
    points (0, 1.25, 2.5), = 0.24%."""
    df = _csv("noiseless_N4_control.csv")
    err0 = df[np.isclose(df["K"], 0.0)]["err_pct"].min()
    err125 = df[df["K_label"] == "noisy-matched-1.25"]["err_pct"].min()
    err25 = df[df["K_label"] == "noisy-matched-2.5"]["err_pct"].min()
    mean_err = float(np.mean([err0, err125, err25]))
    assert abs(mean_err - 0.24) < 0.01, mean_err


def test_l2_vs_l3_boundary_expressibility():
    """At K=4.5 (near the N=3 boundary): L=2 (32 restarts, maxiter=4000)
    still 65.2% wrong -- an expressibility failure, not under-convergence.
    L=3 (8 restarts, maxiter=2000) recovers it to 0.080%."""
    df = _csv("boundary_restart_scaling_N3.csv")

    def err_at(L):
        sub = df[(np.isclose(df["K"], 4.5)) & (df["L"] == L)]
        best = sub.loc[sub["energy"].idxmin()]
        return abs(best["energy"] - best["E_exact"]) / abs(best["E_exact"]) * 100

    assert abs(err_at(2) - 65.2) < 0.1, err_at(2)
    assert abs(err_at(3) - 0.080) < 0.005, err_at(3)


@pytest.mark.slow
def test_kink_suppression_round6():
    """Round-6 bidirectional-hysteresis kink suppression = 12.3% (raw),
    re-derived from results/kink_hysteresis_N3_L3_{noiseless,noisy}.csv via
    the same analysis as results/SUMMARY.md's Q2 section."""
    import analyze_kink_hysteresis_L3 as akh
    case, result = akh.main()
    assert case == "a"
    assert abs(result["suppression"] - 12.3) < 0.2, result["suppression"]


@pytest.mark.xfail(
    reason="'retained fraction' is not a quantity computed anywhere in this "
    "codebase or results/SUMMARY.md. Two natural definitions were tried "
    "against the actual hysteresis data and neither lands both branches in "
    "[0.92,1.01]: (1) a shared depolarizing-mixing-model least-squares solve "
    "for w per branch gave 0.89 (pre-transition) / 0.93 (post-transition); "
    "(2) a plain E_noisy-vs-E_exact regression slope gave 0.9974 pre-transition "
    "but diverges on the post-transition branch (E_exact is exactly flat "
    "there, so the regression has no signal to fit against). The exact "
    "formula needs to be confirmed before this assertion can be written "
    "honestly -- see the restructure conversation.",
    strict=True,
)
def test_retained_fraction_both_branches():
    raise NotImplementedError("retained-fraction formula not yet defined")
