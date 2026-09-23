"""
Odd-N (N=3) K=0 ground space is 2-fold degenerate. Recompute the K=0 fidelities
the results report against (a) the single vector core.exact_ground_state_full
picks and (b) the projector onto the full ground space, F_P = sum_i <g_i|rho|g_i>.

  * Sec IV B states: N=3 L=2, GI and HW, run_vqe with make_fig3's 8 restarts, seed 42.
  * In-loop noisy states behind results/noisy_inloop_N3.csv's K=0 fidelity column:
    re-run those 4 cells with exp01's exact seeds (10000 + p_idx*1000 + 16),
    then rebuild rho = NoisyEvaluator.energy_and_state(theta*).
Writes results/odd_N_fidelity_check.json only.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.linalg import eigh

import schwinger.core as core
import schwinger.backend as qb
from schwinger.parallel_worker import run_cell_worker

N, F, X, K = 3, 2, 16.0, 0.0
NQ = 6


def ground_space_vectors():
    Hs, basis = core.build_H_subspace(X, K, N=N, F=F)
    w, v = eigh(Hs)
    vecs = []
    for j in range(2):
        assert abs(w[j] - w[0]) < 1e-9
        g = np.zeros(2 ** NQ, complex)
        for i, s in enumerate(basis):
            g[s] = v[i, j]
        vecs.append(g)
    return vecs


def inloop_cell(p_idx_p):
    p_idx, p = p_idx_p
    res = run_cell_worker(dict(N=N, F=F, L=2, ansatz="gi", fold_lambda=1, p=p, eps=core.SPAM_DEFAULT, x=X, K=K,
                               n_restarts=8, seed=10000 + p_idx * 1000 + 16, maxiter=2000))
    best = res["per_restart"][res["best_restart"]]
    return dict(p=p, theta=best["theta"], energy=best["energy"], old_fid=res["diag"]["fidelity"])


def main():
    G = ground_space_vectors()
    out = {"sec_IVB": {}, "inloop": {}}

    # ---- in-loop cells (parallel)
    ps = [0.005, 0.01, 0.02, 0.05]
    with ProcessPoolExecutor(max_workers=4) as pool:
        cells = list(pool.map(inloop_cell, list(enumerate(ps))))
    _, psi_ref = core.exact_ground_state_full(X, K, N, F)
    for c in cells:
        ev = qb.NoisyEvaluator(N, F, 2, ansatz="gi", p=c["p"], eps=core.SPAM_DEFAULT, x=X)
        E, rho = ev.energy_and_state(np.array(c["theta"]), K)
        f_single = float(np.real(psi_ref.conj() @ rho @ psi_ref))
        f_proj = float(sum(np.real(g.conj() @ rho @ g) for g in G))
        out["inloop"][str(c["p"])] = dict(energy=E, fidelity_single_vector=f_single,
                                          fidelity_projector=f_proj, worker_reported_fidelity=c["old_fid"],
                                          per_ground_vector=[float(np.real(g.conj() @ rho @ g)) for g in G])
        print(f"in-loop p={c['p']}: E={E:.4f}  F_single={f_single:.4f} (worker {c['old_fid']:.4f})  F_projector={f_proj:.4f}", flush=True)

    # ---- Sec IV B: GI and HW, N=3 L=2 K=0
    H = core.build_H_full(X, K, N, F)
    Q_tot, Q2, N0, N1 = core.build_operators(N, F)
    psi0 = core.make_psi0(N, F)
    for name, fn, npar in [("GI", core.gauge_invariant_ansatz, 2 * ((NQ - 1) + NQ)),
                           ("HW", core.hw_efficient_ansatz, 2 * 2 * NQ)]:
        E, psi = core.run_vqe(fn, H, NQ, 2, npar, psi0, n_restarts=8)
        f_single = float(abs(psi_ref.conj() @ psi) ** 2)
        per = [float(abs(g.conj() @ psi) ** 2) for g in G]
        out["sec_IVB"][name] = dict(energy=E, Q_tot2=float(np.real(psi.conj() @ Q2 @ psi)),
                                    N0=float(np.real(psi.conj() @ N0 @ psi)),
                                    fidelity_single_vector=f_single, fidelity_projector=float(sum(per)),
                                    per_ground_vector=per)
        print(f"Sec IV B {name}: E={E:.4f}  F_single={f_single:.4f}  F_projector={sum(per):.4f}  per-vector={per}", flush=True)

    json.dump(out, open("results/odd_N_fidelity_check.json", "w"), indent=2)


if __name__ == "__main__":
    main()
