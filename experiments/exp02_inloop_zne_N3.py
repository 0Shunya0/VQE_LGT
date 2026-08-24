"""
EXPERIMENT 2: in-loop ZNE, N=3, L=2, GI ansatz. Parallelized across
(p,K,lambda) cells with a process pool (see parallel_worker.py), same
pattern and measured ~8.9x speedup as Experiment 1.

Same (K,p) grid and restart budget as Experiment 1, but each cell is optimized
independently at three noise-scaling factors lambda in {1,2,3} (n_CX 20/40/60
via qiskit_backend.fold_circuit's per-gate CNOT folding). From the three
best-of-8 energies per cell:
    E_mit (linear Richardson) = 1.5*E(lambda=1) - 0.5*E(lambda=3)
    E(lambda) = a + b*c^lambda fit through the 3 points (exact fit, 3 params/3
    points) -- the lambda->0 limit (a+b) is the "residual floor" once the
    depolarizing part is divided out, comparable to schwinger_core.zne_exponential.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.optimize import minimize as spminimize

import schwinger.core as core
from schwinger.parallel_worker import run_cell_worker

N, F, L = 3, 2, 2
X = 16.0
K_GRID = np.linspace(-16, 16, 33)
P_VALUES = [0.005, 0.01, 0.02, 0.05]
LAMBDAS = [1, 2, 3]
EPS = core.SPAM_DEFAULT
N_RESTARTS = 8
MAXITER = 2000
SEED_BASE = 20000
N_WORKERS = 20

RESULTS_DIR = "results"
CSV_PATH = os.path.join(RESULTS_DIR, "inloop_zne_N3.csv")
CKPT_PATH = os.path.join(RESULTS_DIR, "inloop_zne_N3_checkpoint.json")

CSV_FIELDS = ["p", "K", "K_index", "lambda", "n_CX", "restart", "seed",
              "energy", "E_exact", "err_pct", "n_iter", "converged",
              "hit_maxiter", "diverged", "is_best", "cell_time_s"]


def cell_seed(p_idx, k_idx, lam):
    return SEED_BASE + p_idx * 10000 + k_idx * 10 + lam


def load_checkpoint():
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH) as f:
            return set(json.load(f)["done_cells"])
    return set()


def save_checkpoint(done_cells):
    with open(CKPT_PATH, "w") as f:
        json.dump({"done_cells": sorted(done_cells)}, f)


def append_rows(rows):
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(rows)


def rows_from_result(result, k_idx):
    p, K, lam = result["p"], result["K"], result["fold_lambda"]
    diag = result["diag"]
    dt = result["cell_time_s"]
    rows = []
    for r in result["per_restart"]:
        rows.append(dict(p=p, K=K, K_index=k_idx, **{"lambda": lam}, n_CX=result["n_cx"],
                          restart=r["restart"], seed=r["seed"], energy=r["energy"],
                          E_exact=(diag["E_exact"] if diag else None),
                          err_pct=(abs(r["energy"] - diag["E_exact"]) / max(abs(diag["E_exact"]), 1e-9) * 100
                                   if (r["energy"] is not None and diag) else None),
                          n_iter=r["n_iter"], converged=r["converged"],
                          hit_maxiter=r["hit_maxiter"], diverged=r["diverged"],
                          is_best=(r["restart"] == result["best_restart"]), cell_time_s=dt))
    return rows


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    done_cells = load_checkpoint()

    cells = []
    for p_idx, p in enumerate(P_VALUES):
        for k_idx, K in enumerate(K_GRID):
            for lam in LAMBDAS:
                cell_key = f"{p_idx}_{k_idx}_{lam}"
                if cell_key not in done_cells:
                    cells.append((cell_key, p_idx, k_idx, p, float(K), lam))

    total_cells = len(P_VALUES) * len(K_GRID) * len(LAMBDAS)
    print(f"Experiment 2: {total_cells} (p,K,lambda) cells total, {len(done_cells)} already done, "
          f"{len(cells)} remaining. {N_WORKERS} parallel workers.", flush=True)
    if not cells:
        print("Nothing to do.")
        return

    t_sweep_start = time.time()
    n_reported = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {}
        for cell_key, p_idx, k_idx, p, K, lam in cells:
            args = dict(N=N, F=F, L=L, ansatz="gi", fold_lambda=lam, p=p, eps=EPS, x=X, K=K,
                        n_restarts=N_RESTARTS, seed=cell_seed(p_idx, k_idx, lam), maxiter=MAXITER)
            fut = pool.submit(run_cell_worker, args)
            futures[fut] = (cell_key, p_idx, k_idx, p, K, lam)

        for fut in as_completed(futures):
            cell_key, p_idx, k_idx, p, K, lam = futures[fut]
            result = fut.result()

            rows = rows_from_result(result, k_idx)
            append_rows(rows)
            done_cells.add(cell_key)
            save_checkpoint(done_cells)
            n_reported += 1

            best_e = next((r["energy"] for r in rows if r["is_best"]), None)
            elapsed = time.time() - t_sweep_start
            print(f"[{len(done_cells)}/{total_cells}] p={p} K={K:+.1f} lambda={lam} "
                  f"best_E={best_e} (elapsed {elapsed/60:.1f} min)", flush=True)

            if n_reported == min(N_WORKERS, len(cells)):
                rate = elapsed / n_reported
                remaining = len(cells) - n_reported
                est_remaining = remaining * rate
                print(f"\n>>> RUNTIME ESTIMATE: first batch of {n_reported} parallel cells took "
                      f"{elapsed/60:.1f} min ({rate:.1f}s/cell throughput-adjusted). "
                      f"{remaining} cells remain -> est. {est_remaining/60:.1f} min more.\n", flush=True)

    print(f"Experiment 2 done in {(time.time()-t_sweep_start)/60:.1f} min total this run.")


# ===== post-processing: Richardson mitigation, exponential-fit residual floor,
#       boundary residuals comparable to the post-hoc 3.8/10.5/33 numbers =====

def _fit_exp(lam_arr, E_arr):
    """Exact fit of E(lambda)=a+b*c^lambda through 3 points (same optimizer
    pattern as schwinger_core.zne_exponential: multi-start Nelder-Mead)."""
    def resid(params):
        a, b, c = params
        return float(np.sum((a + b * np.power(c, lam_arr) - E_arr) ** 2))
    best = None
    for c0 in (0.3, 0.6, 0.9):
        r = spminimize(resid, [E_arr[-1], E_arr[0] - E_arr[-1], c0], method="Nelder-Mead",
                        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
        if best is None or r.fun < best.fun:
            best = r
    a, b, c = best.x
    return a, b, c


def postprocess():
    import pandas as pd
    df = pd.read_csv(CSV_PATH)
    best = df[df["is_best"]].copy()
    piv = best.pivot_table(index=["p", "K"], columns="lambda", values="energy").reset_index()
    piv = piv.rename(columns={1: "E_lam1", 2: "E_lam2", 3: "E_lam3"})
    piv["E_mit_linear"] = 1.5 * piv["E_lam1"] - 0.5 * piv["E_lam3"]

    exact_map = best.drop_duplicates("K").set_index("K")["E_exact"].to_dict()
    piv["E_exact"] = piv["K"].map(exact_map)

    fit_rows = []
    for _, row in piv.iterrows():
        lam_arr = np.array([1.0, 2.0, 3.0])
        E_arr = np.array([row["E_lam1"], row["E_lam2"], row["E_lam3"]])
        a, b, c = _fit_exp(lam_arr, E_arr)
        fit_rows.append(dict(a=a, b=b, c=c, E_lam0=a + b,
                              residual_floor=abs((a + b) - row["E_exact"])))
    fit_df = pd.DataFrame(fit_rows)
    out = pd.concat([piv.reset_index(drop=True), fit_df], axis=1)
    out.to_csv(os.path.join(RESULTS_DIR, "inloop_zne_N3_mitigated.csv"), index=False)

    kb = 5.6
    boundary_K = out["K"].iloc[(out["K"] - kb).abs().argmin()]
    print(f"nearest grid K to boundary |K|~{kb}: {boundary_K}")
    for p_target in (0.01, 0.02, 0.05):
        sub = out[(out["p"] == p_target) & (out["K"] == boundary_K)]
        if len(sub):
            r = sub.iloc[0]
            res = abs(r["E_mit_linear"] - r["E_exact"])
            print(f"  in-loop ZNE boundary residual at p={p_target}: {res:.3f} "
                  f"(post-hoc reference: {'3.8' if p_target==0.01 else '10.5' if p_target==0.02 else '33'})")
    return out


if __name__ == "__main__":
    main()
    postprocess()
