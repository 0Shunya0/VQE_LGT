"""
Process-pool worker for the noisy-VQE sweeps. Each worker process builds its
own NoisyEvaluator (qiskit/Aer objects aren't shared across processes) and
runs a full (K,p[,lambda]) cell -- all n_restarts serially within the worker,
so the evaluator's one-time build/transpile cost is amortized over its whole
cell rather than paid per restart. Measured speedup on this 24-core machine:
~8.9x wall-clock with 20 worker processes vs. serial (8 restarts x ~20-25s
each in parallel finish in ~40-47s, not 8x that).

Thread-oversubscription guard: each worker process pins BLAS/OpenMP to a
single thread BEFORE numpy is imported (Windows uses spawn, so every worker
re-executes this module fresh) -- otherwise ~20 concurrent processes each
trying to multithread their own linear algebra would contend for the same 24
cores and erase the benefit of process-level parallelism.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import time

import numpy as np
from scipy.optimize import minimize

import schwinger.core as core
import schwinger.backend as qb

COBYLA_OPTIONS = {"rhobeg": 0.5}


def run_cell_worker(args):
    """
    args: dict with N,F,L,ansatz,fold_lambda,p,eps,x,K,n_restarts,seed,maxiter,
    and optionally capture_trace=True (Experiment 3 only: logs the running
    cost at every COBYLA iteration via a callback, ~2x eval cost from the
    extra re-evaluation per iteration -- worth it only for the small N=4 spot
    check, where seeing whether the optimizer stalls vs. genuinely converges
    in the flattened p/d>>1 landscape is itself part of the result).
    Returns a plain-dict (picklable) result: per-restart list, the best
    restart, and diagnostics (fidelity, <Q_tot^2>) computed on the best
    restart's actual noisy density matrix.
    """
    N, F, L = args["N"], args["F"], args["L"]
    ansatz, fold_lambda = args["ansatz"], args["fold_lambda"]
    p, eps, x, K = args["p"], args["eps"], args["x"], args["K"]
    n_restarts, seed, maxiter = args["n_restarts"], args["seed"], args["maxiter"]
    capture_trace = args.get("capture_trace", False)

    t_cell_start = time.time()
    ev = qb.NoisyEvaluator(N, F, L, ansatz=ansatz, fold_lambda=fold_lambda, p=p, eps=eps, x=x)
    cost = ev.cost_fn(K)
    rng = np.random.default_rng(seed)
    n_p = ev.n_params

    per_restart = []
    for r in range(n_restarts):
        theta0 = rng.uniform(-np.pi, np.pi, n_p)
        trace = None
        if capture_trace:
            trace = []
            res = minimize(cost, theta0, method="COBYLA",
                            options={**COBYLA_OPTIONS, "maxiter": maxiter},
                            callback=lambda xk: trace.append(float(cost(xk))))
        else:
            res = minimize(cost, theta0, method="COBYLA",
                            options={**COBYLA_OPTIONS, "maxiter": maxiter})
        diverged = not np.isfinite(res.fun)
        hit_maxiter = (res.nfev >= maxiter) and not res.success
        per_restart.append(dict(
            restart=r, seed=seed, energy=(float(res.fun) if not diverged else None),
            n_iter=int(res.nfev), converged=bool(res.success),
            hit_maxiter=bool(hit_maxiter), diverged=bool(diverged), trace=trace,
            theta=res.x.tolist(),
        ))

    finite = [r for r in per_restart if r["energy"] is not None]
    best = min(finite, key=lambda r: r["energy"]) if finite else None

    diag = None
    if best is not None:
        theta_best = np.array(best["theta"])
        E, rho = ev.energy_and_state(theta_best, K)
        E_exact, psi_ref = core.exact_ground_state_full(x, K, N, F)
        Q_tot, Q2, N0, N1 = core.build_operators(N, F)
        fid = float(np.real(psi_ref.conj() @ rho @ psi_ref))
        qt2 = float(np.real(np.trace(Q2 @ rho)))
        diag = dict(E_exact=float(E_exact), fidelity=fid, Q_tot2=qt2)

    return dict(N=N, L=L, ansatz=ansatz, fold_lambda=fold_lambda, p=p, K=K,
                n_cx=ev.n_cx, per_restart=per_restart,
                best_restart=(best["restart"] if best is not None else None),
                diag=diag, cell_time_s=time.time() - t_cell_start)
