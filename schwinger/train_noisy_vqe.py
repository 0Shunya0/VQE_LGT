"""
Train-noisy VQE driver: COBYLA optimizes <W> = Tr(W rho) evaluated on the
actual noisy density matrix (qiskit_backend.NoisyEvaluator) at every
optimizer step, replacing the post-hoc analytic contraction in
schwinger_core.noisy(). COBYLA settings (method, rhobeg) match the existing
noiseless driver (schwinger_core.run_vqe / run_vqe_scaling) exactly.
"""
import time
import numpy as np
from scipy.optimize import minimize

import schwinger_core as core
import qiskit_backend as qb

COBYLA_OPTIONS = {"rhobeg": 0.5}  # maxiter set per call, matching schwinger_core's pattern


def exact_charge_neutral_state(x, K, N, F=2):
    """Exact ground state in the Q_tot=0 sector, embedded in the full 2**nq space
    (same construction as schwinger_core.exact_ground_state_full)."""
    return core.exact_ground_state_full(x, K, N, F)


def run_noisy_vqe_point(evaluator, K, n_restarts=8, seed=42, maxiter=2000, verbose=False):
    """
    Optimize <W> at fixed (N,L,ansatz,fold_lambda,p,eps,K) over n_restarts random
    starts. Returns a dict with the best result and the full per-restart spread.
    """
    rng = np.random.default_rng(seed)
    cost = evaluator.cost_fn(K)
    n_p = evaluator.n_params

    per_restart = []
    for r in range(n_restarts):
        theta0 = rng.uniform(-np.pi, np.pi, n_p)
        t0 = time.time()
        res = minimize(cost, theta0, method="COBYLA",
                        options={**COBYLA_OPTIONS, "maxiter": maxiter})
        dt = time.time() - t0
        diverged = not np.isfinite(res.fun)
        hit_maxiter = (res.nfev >= maxiter) and not res.success
        per_restart.append(dict(
            restart=r, seed=seed, energy=float(res.fun) if not diverged else None,
            n_iter=int(res.nfev), converged=bool(res.success),
            hit_maxiter=bool(hit_maxiter), diverged=bool(diverged),
            theta=res.x.copy(), time_s=dt,
        ))
        if verbose:
            print(f"    restart {r+1}/{n_restarts}: E={res.fun:.6f} nfev={res.nfev} "
                  f"converged={res.success} ({dt:.1f}s)", flush=True)

    finite = [r for r in per_restart if r["energy"] is not None]
    best = min(finite, key=lambda r: r["energy"]) if finite else None
    return dict(K=K, best=best, per_restart=per_restart)


def diagnostics(evaluator, K, theta, N, F=2, x=16.0):
    """<W>, fidelity with the exact charge-neutral ground state, <Q_tot^2>, on
    the actual noisy density matrix produced by theta."""
    E, rho = evaluator.energy_and_state(theta, K)
    E_exact, psi_ref = exact_charge_neutral_state(x, K, N, F)
    Q_tot, Q2, N0, N1 = core.build_operators(N, F)
    fid = float(np.real(psi_ref.conj() @ rho @ psi_ref))
    qt2 = float(np.real(np.trace(Q2 @ rho)))
    err_pct = abs(E - E_exact) / max(abs(E_exact), 1e-9) * 100
    return dict(E=E, E_exact=E_exact, err_pct=err_pct, fidelity=fid, Q_tot2=qt2)


# ===== validation gate =========================================================

def validation_gate(seed=42, maxiter=2500, n_restarts=8):
    """
    p=0, eps=0 must reproduce the existing noiseless Table numbers:
        N=3, L=2 -> -43.560, err 0.015%
        N=4, L=5 -> -68.562, err 0.021%
    within 1e-3 absolute energy. Prints PASS/FAIL and returns bool.

    N=4's restart budget is doubled relative to N=3: COBYLA is a derivative-
    free simplex-type method with no reproducibility guarantee across
    different floating-point evaluation paths (Aer's C++ backend vs the
    reference driver's direct numpy matmul agree to 1e-13 per single
    evaluation -- see test_qiskit_backend.py check #1 -- but 75-parameter
    COBYLA trajectories are chaotically sensitive to that level of noise over
    ~2500 iterations, so 8 restarts alone landed 1.7e-2 short on a first
    attempt). More restarts is the correct fix, not a numerics bug: it makes
    the reported optimum robust to which local basin any single restart's
    perturbed trajectory happens to fall into.
    """
    print("=" * 70)
    print("VALIDATION GATE: p=0, eps=0 must reproduce noiseless Table numbers")
    print("=" * 70)
    checks = [
        (3, 2, -43.560, 0.015, n_restarts),
        (4, 5, -68.562, 0.021, n_restarts * 2),
    ]
    all_ok = True
    for N, L, E_ref, err_ref, n_r in checks:
        F = 2
        t0 = time.time()
        print(f"  building/transpiling N={N} L={L} evaluator...", flush=True)
        ev = qb.NoisyEvaluator(N, F, L, ansatz='gi', fold_lambda=1, p=0.0, eps=0.0)
        print(f"  running {n_r} restarts, maxiter={maxiter}...", flush=True)
        result = run_noisy_vqe_point(ev, K=0.0, n_restarts=n_r, seed=seed, maxiter=maxiter, verbose=True)
        dt = time.time() - t0
        energies = sorted(r["energy"] for r in result["per_restart"] if r["energy"] is not None)
        E = result["best"]["energy"]
        diff = abs(E - E_ref)
        ok = diff < 1e-3
        all_ok &= ok
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] N={N} L={L}: got {E:.6f}, reference {E_ref:.3f} "
              f"(paper err {err_ref}%), |diff|={diff:.2e} ({dt:.1f}s, {n_r} restarts)")
        print(f"         per-restart energies: {[round(e, 4) for e in energies]}")
    print("=" * 70)
    print("VALIDATION GATE: " + ("PASS" if all_ok else "FAIL - DO NOT PROCEED"))
    print("=" * 70)
    return all_ok


if __name__ == "__main__":
    ok = validation_gate()
    if not ok:
        raise SystemExit(1)
