"""Regenerate make_fig2.py's PRECOMPUTED (N=4,5,6) with run_vqe_scaling at the
same budget make_fig2 uses for N=2,3 (n_restarts=6, maxiter=2500, seed=42)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json
from concurrent.futures import ProcessPoolExecutor
from scipy.linalg import eigh
import schwinger.core as core

CONFIGS = [(4, L) for L in range(1, 6)] + [(5, 1), (5, 2), (6, 1)]

def run(cfg):
    N, L = cfg
    E = core.run_vqe_scaling(N, L, n_restarts=6, maxiter=2500)
    Hs, _ = core.build_H_subspace(16.0, 0.0, N=N, F=2)
    Ex = float(eigh(Hs, eigvals_only=True)[0])
    nq = 2 * N
    return dict(N=N, L=L, n_params=L * ((nq - 1) + nq), E=float(E), E_exact=Ex,
                err_pct=abs(E - Ex) / abs(Ex) * 100)

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=8) as pool:
        res = list(pool.map(run, CONFIGS))
    with open("results/precomputed_N456.json", "w") as f:
        json.dump(res, f, indent=2)
    for r in res:
        print(r)
