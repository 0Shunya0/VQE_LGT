"""
EXPERIMENT 4: gradient variance of the HW ansatz, noiseless, N=2,3,4.

Exact same protocol as schwinger_core.grad_variance (the existing GI
measurement, reused unmodified for the GI numbers): 20 random parameter
vectors theta ~ U[0,2pi)^p, L=2, central finite differences with delta=1e-3,
averaged over the parameter index, same energy units, no renormalization.
The only change is the ansatz/cost pair: HW is not charge-conserving, so
(matching how the rest of the codebase already evaluates the HW ansatz --
schwinger_core.run_vqe uses build_H_full, not the sector-projected
Hamiltonian) the cost is <psi|H_full|psi> on the full 2**nq state via
hw_efficient_ansatz, instead of GI's sector-projected _energy_sector.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import csv
import os

import numpy as np

import schwinger.core as core

RESULTS_DIR = "results"
CSV_PATH = os.path.join(RESULTS_DIR, "gradvar_hw.csv")

REFERENCE_GI = {2: 23.0, 3: 20.4, 4: 19.3}  # from the manuscript, reproduced by schwinger_core.grad_variance


def _energy_full(theta, H, psi0, nq, L):
    psi = core.hw_efficient_ansatz(theta, psi0, nq, L)
    return float(np.real(psi.conj() @ H @ psi))


def grad_variance_hw(N, L=2, x=16.0, n_samples=20, seed=77, eps=1e-3):
    """HW-ansatz counterpart of schwinger_core.grad_variance: identical
    sampling/finite-difference protocol, full-Hilbert-space cost function."""
    F = 2
    nq = N * F
    H = core.build_H_full(x, 0.0, N=N, F=F)
    psi0 = core.make_psi0(N, F)
    n_p = L * core.hw_params_per_layer(N, F)
    rng = np.random.default_rng(seed)
    norms = []
    for _ in range(n_samples):
        theta = rng.uniform(0, 2 * np.pi, n_p)
        grad = np.zeros(n_p)
        for j in range(n_p):
            tp = theta.copy(); tp[j] += eps
            tm = theta.copy(); tm[j] -= eps
            grad[j] = (_energy_full(tp, H, psi0, nq, L) - _energy_full(tm, H, psi0, nq, L)) / (2 * eps)
        norms.append(np.mean(grad ** 2))
    return float(np.mean(norms)), float(np.std(norms))


def grad_variance_gi_check(N, L=2, n_samples=20, seed=77, eps=1e-3):
    """Reuse the existing GI measurement unmodified, but with theta~U[0,2pi)
    (matching Experiment 4's stated protocol) instead of schwinger_core's own
    U[-pi,pi) default, for a same-RNG-convention side-by-side with HW."""
    F = 2
    nq = N * F
    Hsub, basis = core.build_H_subspace(16.0, 0.0, N=N, F=F)
    psi0 = core.make_psi0(N, F)
    n_p = L * core.gi_params_per_layer(N, F)
    rng = np.random.default_rng(seed)
    norms = []
    for _ in range(n_samples):
        theta = rng.uniform(0, 2 * np.pi, n_p)
        grad = np.zeros(n_p)
        for j in range(n_p):
            tp = theta.copy(); tp[j] += eps
            tm = theta.copy(); tm[j] -= eps
            grad[j] = (core._energy_sector(tp, Hsub, basis, nq, psi0, L)
                       - core._energy_sector(tm, Hsub, basis, nq, psi0, L)) / (2 * eps)
        norms.append(np.mean(grad ** 2))
    return float(np.mean(norms)), float(np.std(norms))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    print("Gradient variance (mean-squared gradient over 20 random points, L=2, delta=1e-3):")
    print(f"  {'N':>3} {'GI (paper)':>12} {'GI (rerun)':>12} {'HW':>12}")
    for N in (2, 3, 4):
        gi_mean, gi_std = grad_variance_gi_check(N)
        hw_mean, hw_std = grad_variance_hw(N)
        print(f"  {N:>3} {REFERENCE_GI[N]:>12.1f} {gi_mean:>12.2f} {hw_mean:>12.2f}")
        rows.append(dict(N=N, ansatz="GI", grad_var_mean=gi_mean, grad_var_std=gi_std,
                          reference_paper_value=REFERENCE_GI[N]))
        rows.append(dict(N=N, ansatz="HW", grad_var_mean=hw_mean, grad_var_std=hw_std,
                          reference_paper_value=None))

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["N", "ansatz", "grad_var_mean", "grad_var_std", "reference_paper_value"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWritten to {CSV_PATH}")


if __name__ == "__main__":
    main()
