"""
Task B / Step 2: three noiseless/noisy reference values at the theta saved by
ibm_taskB_step1_theta.py (results/theta_N3_L2_K0.json), computed purely
classically -- no QPU. Reuses schwinger/core.py and schwinger/backend.py
unmodified:
  a) exact diagonalization ground-state energy, physical sector (core.eigh
     on build_H_subspace)
  b) noiseless statevector <W> at the saved theta (core.gauge_invariant_ansatz
     + build_H_full)
  c) depolarizing-channel <W> at ibm_marrakesh's measured median CZ error
     (excluding disabled couplers, error==1.0) plus SPAM eps=0.005
     (backend.NoisyEvaluator, ansatz='gi', density-matrix AerSimulator)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json

import numpy as np
from scipy.linalg import eigh

import schwinger.core as core
from schwinger.backend import NoisyEvaluator

N, L, X, K = 3, 2, 16.0, 0.0
F = 2

MEDIAN_CZ_ERROR = 0.002873171523854273  # ibm_marrakesh, median over enabled CZ pairs (16/352 disabled, error==1.0, excluded)
SPAM_EPS = 0.005


def main():
    with open("results/theta_N3_L2_K0.json") as f:
        saved = json.load(f)
    theta = np.array(saved["theta"])
    assert saved["N"] == N and saved["L"] == L and saved["K"] == K and saved["x"] == X

    # a) exact diagonalization, physical sector
    Hsub, basis = core.build_H_subspace(X, K, N=N, F=F)
    w = eigh(Hsub, eigvals_only=True)
    E_exact = float(w[0])

    # b) noiseless statevector <W>
    psi0 = core.make_psi0(N, F)
    nq = N * F
    H_full = core.build_H_full(X, K, N=N, F=F)
    psi = core.gauge_invariant_ansatz(theta, psi0, nq, L)
    E_noiseless = float(np.real(psi.conj() @ H_full @ psi))

    # c) depolarizing + SPAM at ibm_marrakesh's median CZ error
    ev = NoisyEvaluator(N=N, F=F, L=L, ansatz='gi', p=MEDIAN_CZ_ERROR, eps=SPAM_EPS, x=X)
    E_depol = ev.energy(theta, K)

    print(f"(a) exact diagonalization (physical sector): {E_exact:.6f}")
    print(f"(b) noiseless statevector <W>:                {E_noiseless:.6f}")
    print(f"(c) depolarizing+SPAM (p={MEDIAN_CZ_ERROR:.6f}, eps={SPAM_EPS}): {E_depol:.6f}")
    print()
    print(f"relative error (b) vs (a): {abs(E_noiseless - E_exact) / abs(E_exact) * 100:.4f}%")
    print(f"relative error (c) vs (a): {abs(E_depol - E_exact) / abs(E_exact) * 100:.4f}%")

    out = dict(E_exact=E_exact, E_noiseless=E_noiseless, E_depol=E_depol,
               median_cz_error=MEDIAN_CZ_ERROR, spam_eps=SPAM_EPS)
    with open("results/ibm_N3_L2_K0_references.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/ibm_N3_L2_K0_references.json")


if __name__ == "__main__":
    main()
