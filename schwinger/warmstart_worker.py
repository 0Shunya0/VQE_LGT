"""
Single-restart worker for the warm-start continuation sweep. Unlike
parallel_worker.run_cell_worker (which runs a whole cell's restarts serially
inside one persistent process, amortizing the NoisyEvaluator build cost),
this dispatches ONE restart per task so a warm-start candidate (seeded with
the previous K's best theta) and several random-restart candidates can run
concurrently within a single K step, while K steps themselves stay strictly
serial (each depends on the previous one's winner). The per-task
NoisyEvaluator rebuild cost (~1-2s) is a small fraction of a restart's
optimization time (tens of seconds to minutes at maxiter=4000) so this is a
fine trade for the concurrency.

Same thread-pinning rationale as parallel_worker.py: pin BLAS/OpenMP to one
thread BEFORE importing numpy, since Windows spawn re-executes this module
fresh in every worker process.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import minimize

import schwinger.core as core
import schwinger.backend as qb

COBYLA_OPTIONS = {"rhobeg": 0.5}


def run_single_restart(args):
    """
    args: dict with N,F,L,ansatz,fold_lambda,p,eps,x,K,maxiter,init_mode
    ('warm' or 'random'), and either theta0 (list, for 'warm') or seed (int,
    for 'random'). Builds its own NoisyEvaluator, runs one COBYLA
    optimization from the given start point, and returns the result plus
    diagnostics (fidelity, <Q_tot^2>) computed on its own final state -- so
    the caller can pick a winner across candidates without a second pass.
    """
    N, F, L = args["N"], args["F"], args["L"]
    ansatz, fold_lambda = args["ansatz"], args["fold_lambda"]
    p, eps, x, K = args["p"], args["eps"], args["x"], args["K"]
    maxiter, init_mode = args["maxiter"], args["init_mode"]

    ev = qb.NoisyEvaluator(N, F, L, ansatz=ansatz, fold_lambda=fold_lambda, p=p, eps=eps, x=x)
    cost = ev.cost_fn(K)

    if init_mode == "warm":
        theta0 = np.array(args["theta0"], dtype=float)
        seed = None
    else:
        seed = args["seed"]
        rng = np.random.default_rng(seed)
        theta0 = rng.uniform(-np.pi, np.pi, ev.n_params)

    res = minimize(cost, theta0, method="COBYLA", options={**COBYLA_OPTIONS, "maxiter": maxiter})
    diverged = not np.isfinite(res.fun)
    hit_maxiter = (res.nfev >= maxiter) and not res.success

    diag = None
    if not diverged:
        E_exact, psi_ref = core.exact_ground_state_full(x, K, N, F)
        _, rho = ev.energy_and_state(res.x, K)
        Q_tot, Q2, N0, N1 = core.build_operators(N, F)
        fid = float(np.real(psi_ref.conj() @ rho @ psi_ref))
        qt2 = float(np.real(np.trace(Q2 @ rho)))
        diag = dict(E_exact=float(E_exact), fidelity=fid, Q_tot2=qt2)

    return dict(
        K=K, p=p, energy=(float(res.fun) if not diverged else None),
        theta=res.x.tolist(), n_iter=int(res.nfev), converged=bool(res.success),
        hit_maxiter=bool(hit_maxiter), diverged=bool(diverged),
        init_mode=init_mode, seed=seed, diag=diag,
    )
