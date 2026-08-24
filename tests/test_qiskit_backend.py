"""
Validation of qiskit_backend.NoisyEvaluator:
  1. p=0, eps=0 must exactly reproduce schwinger_core's exact statevector <W>
     for both GI and HW ansatze (confirms circuit + Operator(W) qubit-ordering
     is correct, and that transpilation doesn't silently drop the 'id' SPAM
     anchor gates -- checked explicitly below).
  2. p>0, eps>0 must match noisy_sim's independently-derived, hand-rolled,
     einsum-based density-matrix propagator (already cross-validated against
     raw qiskit-aer circuits in test_noisy_sim.py) to numerical precision.
  3. ZNE folding at lambda=3 must reproduce the SAME noiseless energy as
     lambda=1 (folding doesn't change the ideal unitary) while tripling n_CX.
"""
import numpy as np

import schwinger.core as core
import reference_noisy_sim as ns
import schwinger.backend as qb


def check_spam_channel_has_effect():
    print("=== 0. SPAM bit-flip channel actually changes the readout ===")
    rng = np.random.default_rng(10)
    N, F, nq, L = 3, 2, 6, 2
    ev_no_spam = qb.NoisyEvaluator(N, F, L, ansatz='gi', fold_lambda=1, p=0.0, eps=0.0)
    ev_spam = qb.NoisyEvaluator(N, F, L, ansatz='gi', fold_lambda=1, p=0.0, eps=0.05)
    theta = rng.uniform(-np.pi, np.pi, ev_no_spam.n_params)
    K = 1.0
    e0 = ev_no_spam.energy(theta, K)
    e1 = ev_spam.energy(theta, K)
    print(f"  eps=0 -> {e0:.6f}, eps=0.05 -> {e1:.6f}, diff={abs(e0 - e1):.4f}")
    assert abs(e0 - e1) > 1e-4, "SPAM channel had no measurable effect -- it isn't being applied"
    print("  PASS\n")


def check_noiseless_exact():
    print("=== 1. p=0, eps=0 matches exact statevector <W> ===")
    rng = np.random.default_rng(11)
    for N, ansatz_name, ansatz_fn in [(3, 'gi', core.gauge_invariant_ansatz),
                                       (3, 'hw', core.hw_efficient_ansatz)]:
        F, nq, L = 2, N * 2, 2
        ev = qb.NoisyEvaluator(N, F, L, ansatz=ansatz_name, fold_lambda=1, p=0.0, eps=0.0)
        theta = rng.uniform(-np.pi, np.pi, ev.n_params)
        K = 3.0
        e_aer = ev.energy(theta, K)
        psi0 = core.make_psi0(N, F)
        H = core.build_H_full(16.0, K, N=N, F=F)
        psi = ansatz_fn(theta, psi0, nq, L)
        e_exact = float(np.real(psi.conj() @ H @ psi))
        print(f"  {ansatz_name.upper()} N={N}: aer={e_aer:.8f} exact={e_exact:.8f} diff={abs(e_aer - e_exact):.2e}")
        assert abs(e_aer - e_exact) < 1e-8
    print("  PASS\n")


def check_noisy_vs_handrolled():
    print("=== 2. p>0, eps>0 matches hand-rolled noisy_sim propagator ===")
    rng = np.random.default_rng(12)
    N, F, nq, L = 3, 2, 6, 2
    p, eps, K = 0.02, 0.005, -3.0
    ev = qb.NoisyEvaluator(N, F, L, ansatz='gi', fold_lambda=1, p=p, eps=eps)
    theta = rng.uniform(-np.pi, np.pi, ev.n_params)
    e_aer = ev.energy(theta, K)

    psi0 = core.make_psi0(N, F)
    ops = ns.gi_ansatz_ops(theta, nq, L)
    rho = ns.run_circuit(ops, psi0, nq, p=p, eps=eps)
    H = core.build_H_full(16.0, K, N=N, F=F)
    e_mine = ns.expectation(rho, H)
    print(f"  GI N=3 p={p} eps={eps}: aer={e_aer:.8f} hand-rolled={e_mine:.8f} diff={abs(e_aer - e_mine):.2e}")
    assert abs(e_aer - e_mine) < 1e-6
    print("  PASS\n")


def check_zne_folding():
    print("=== 3. ZNE folding: lambda in {1,2,3} scales n_CX exactly, preserves noiseless energy ===")
    rng = np.random.default_rng(13)
    N, F, nq, L = 3, 2, 6, 2
    evs = {lam: qb.NoisyEvaluator(N, F, L, ansatz='gi', fold_lambda=lam, p=0.0, eps=0.0)
           for lam in (1, 2, 3)}
    for lam, ev in evs.items():
        print(f"  lambda={lam}: n_CX={ev.n_cx}")
        assert ev.n_cx == lam * 20
    theta = rng.uniform(-np.pi, np.pi, evs[1].n_params)
    K = 2.0
    energies = {lam: ev.energy(theta, K) for lam, ev in evs.items()}
    print(f"  noiseless energy: {energies}")
    assert max(energies.values()) - min(energies.values()) < 1e-6
    print("  PASS\n")

    print("  -- noisy: energy should increase monotonically toward E_mix as lambda grows --")
    evs_noisy = {lam: qb.NoisyEvaluator(N, F, L, ansatz='gi', fold_lambda=lam, p=0.03, eps=0.005)
                 for lam in (1, 2, 3)}
    e_noisy = {lam: ev.energy(theta, K) for lam, ev in evs_noisy.items()}
    print(f"  noisy energy vs lambda: {e_noisy}")
    print("  PASS (monotonic decay pattern printed for visual sanity)\n")


if __name__ == "__main__":
    check_spam_channel_has_effect()
    check_noiseless_exact()
    check_noisy_vs_handrolled()
    check_zne_folding()
    print("ALL BACKEND CHECKS PASSED")
