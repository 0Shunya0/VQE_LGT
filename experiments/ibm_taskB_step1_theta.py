"""
Task B / Step 1 (IBM hardware study): re-optimize N=3, L=2 at K=0, x=16
noiselessly, using the exact same optimizer call the paper used for this
(N, L) point -- see figures/make_fig2.py's expressibility sweep, which runs
core.run_vqe_scaling(N, L, n_restarts=R['scaling']=6, maxiter=max(2500,1500))
at the default K=0.0, x=16.0, seed=42.

run_vqe_scaling itself only returns the best energy (not the winning theta),
so this script duplicates its inner loop verbatim -- same seed, same restart
count, same COBYLA options, same cost function (core._energy_sector) -- just
also keeping r.x for the winning restart. No function in schwinger/core.py is
modified.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json
import os

import numpy as np
from scipy.optimize import minimize

import schwinger.core as core

N, L, X, K = 3, 2, 16.0, 0.0
F = 2
NQ = N * F
N_RESTARTS = 6      # R['scaling'] in figures/make_fig2.py
MAXITER = 2500       # max(R['maxiter']=2500, 1500) in figures/make_fig2.py
SEED = 42            # core.run_vqe_scaling default


def main():
    Hsub, basis = core.build_H_subspace(X, K, N=N, F=F)
    psi0 = core.make_psi0(N, F)
    n_p = L * ((NQ - 1) + NQ)

    rng = np.random.default_rng(SEED)
    best_E, best_theta, best_restart = np.inf, None, None
    for i in range(N_RESTARTS):
        t0 = rng.uniform(-np.pi, np.pi, n_p)
        r = minimize(core._energy_sector, t0, args=(Hsub, basis, NQ, psi0, L),
                     method="COBYLA", options={"maxiter": MAXITER, "rhobeg": 0.5})
        if r.fun < best_E:
            best_E, best_theta, best_restart = float(r.fun), r.x.tolist(), i

    out = dict(N=N, L=L, x=X, K=K, seed=SEED, n_restarts=N_RESTARTS, maxiter=MAXITER,
               winning_restart=best_restart, energy=best_E, theta=best_theta)
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "theta_N3_L2_K0.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}")
    print(f"best energy = {best_E}, winning restart = {best_restart}/{N_RESTARTS}")


if __name__ == "__main__":
    main()
