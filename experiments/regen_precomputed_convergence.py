"""
Task IV convergence check: for every (N,L) already in results/precomputed_N456.json
(seed 42 batch), run 3 more independent 6-restart batches (seeds 43,44,45) at the
same maxiter=2500. Report all 4 batch minima and the overall min over 24 restarts.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json
from concurrent.futures import ProcessPoolExecutor
from scipy.linalg import eigh
import schwinger.core as core

SEEDS = [43, 44, 45]
DIMS = {4: 70, 5: 252, 6: 924}


def run(args):
    N, L, seed = args
    E = core.run_vqe_scaling(N, L, n_restarts=6, maxiter=2500, seed=seed)
    return (N, L, seed, float(E))


if __name__ == "__main__":
    prior = json.load(open("results/precomputed_N456.json"))
    configs = [(r["N"], r["L"]) for r in prior]
    jobs = [(N, L, s) for (N, L) in configs for s in SEEDS]

    with ProcessPoolExecutor(max_workers=12) as pool:
        new_results = list(pool.map(run, jobs))

    by_cfg = {(N, L): {42: next(r["E"] for r in prior if r["N"] == N and r["L"] == L)} for N, L in configs}
    for N, L, seed, E in new_results:
        by_cfg[(N, L)][seed] = E

    out = []
    for N, L in configs:
        Hs, _ = core.build_H_subspace(16.0, 0.0, N=N, F=2)
        Ex = float(eigh(Hs, eigvals_only=True)[0])
        batches = by_cfg[(N, L)]
        overall_min = min(batches.values())
        nq = 2 * N
        n_params = L * ((nq - 1) + nq)
        row = dict(N=N, L=L, n_params=n_params, p_over_d=n_params / DIMS[N],
                   batch_minima={str(k): v for k, v in sorted(batches.items())},
                   overall_min=overall_min, E_exact=Ex,
                   err_pct=abs(overall_min - Ex) / abs(Ex) * 100)
        out.append(row)
        print(f"N={N} L={L} p/d={row['p_over_d']:.3f}  batches={row['batch_minima']}  "
              f"overall_min={overall_min:.4f}  err%={row['err_pct']:.3f}")

    json.dump(out, open("results/precomputed_N456_convergence.json", "w"), indent=2)
