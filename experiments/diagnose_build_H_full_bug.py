"""
Diagnostic only -- reuses schwinger/core.py and schwinger/backend.py
unmodified. Does NOT touch core.py, does NOT plan/submit hardware.

Tests whether NoisyEvaluator.energy() (the function used as "W" throughout
every noisy-VQE production run in this codebase) inherits the build_H_full
electric-field bug, by feeding it the EXACT ground state at K=5 directly
(via a hand-built state-injection circuit, bypassing the parameterized
ansatz entirely) at p=0, eps=0 (fully noiseless), for both ansatz='gi' and
ansatz='hw'.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np
from scipy.linalg import eigh
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import qiskit_aer.library  # noqa: F401  -- registers QuantumCircuit.set_statevector

import schwinger.core as core
from schwinger.backend import NoisyEvaluator

X, K, N, F = 16.0, 5.0, 3, 2
NQ = N * F


def exact_ground_state_full():
    Hsub, basis = core.build_H_subspace(X, K, N=N, F=F)
    w, v = eigh(Hsub)
    psi_full = np.zeros(2 ** NQ, dtype=complex)
    for i, s in enumerate(basis):
        psi_full[s] = v[i, 0]
    return float(w[0]), psi_full


def main():
    E_exact, psi_full = exact_ground_state_full()
    print(f"Exact ground energy (build_H_subspace + eigh), K={K}, N={N}: {E_exact:.10f}")
    print()

    for ansatz in ('gi', 'hw'):
        ev = NoisyEvaluator(N=N, F=F, L=2, ansatz=ansatz, p=0.0, eps=0.0, x=X)

        # Bypass the parameterized ansatz entirely: inject the exact ground
        # state directly via a state-preparation circuit, then reuse
        # NoisyEvaluator's own W-construction/measurement machinery verbatim.
        inject = QuantumCircuit(ev.nq)
        inject.set_statevector(psi_full)
        ev.tqc = inject
        ev.n_params = 0

        H = core.build_H_full(ev.x, K, N=ev.N, F=ev.F)
        bound = inject.copy()
        bound.save_expectation_value(Operator(H), list(range(ev.nq)), label='W')
        res = ev.sim.run(bound).result()
        E_noisy_eval = float(np.real(res.data(0)['W']))

        diff = E_noisy_eval - E_exact
        print(f"ansatz={ansatz}: NoisyEvaluator-path energy of the EXACT ground state (p=0, eps=0) = {E_noisy_eval:.10f}")
        print(f"  diff from exact eigenvalue: {diff:+.10f}  ({abs(diff)/abs(E_exact)*100:.4f}% relative)")
        print()


if __name__ == "__main__":
    main()
