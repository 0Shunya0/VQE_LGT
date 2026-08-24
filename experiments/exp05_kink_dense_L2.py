"""
Bug 1(b): dense in-loop K-sweep around the N=3 phase boundary (|K|~5.6, which
sits between the 33-point sweep's K=5 and K=6 grid points -- far too coarse
to resolve a kink). Same driver/settings as run_exp1.py, just a finer local
grid and a single p, so the kink can be measured on a grid fine enough to
matter and compared against the exact curve sampled on the SAME grid.

N=3, L=2, GI, p=0.01, K = linspace(4.0, 7.0, 13) (dK=0.25), 8 restarts,
maxiter 2000. Distinct SEED_BASE from run_exp1.py/run_exp2.py/run_exp3.py so
seeds cannot collide with any existing run.
"""
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import schwinger_core as core
from parallel_worker import run_cell_worker

N, F, L = 3, 2, 2
X = 16.0
K_GRID = np.linspace(4.0, 7.0, 13)
P = 0.01
EPS = core.SPAM_DEFAULT
N_RESTARTS = 8
MAXITER = 2000
SEED_BASE = 40000  # distinct from 10000 (exp1) / 20000 (exp2) / 30000 (exp3)
N_WORKERS = 13

RESULTS_DIR = "results"
CSV_PATH = os.path.join(RESULTS_DIR, "kink_dense_N3.csv")
CKPT_PATH = os.path.join(RESULTS_DIR, "kink_dense_N3_checkpoint.json")

CSV_FIELDS = ["p", "K", "K_index", "restart", "seed", "energy", "E_exact",
              "err_pct", "fidelity", "Q_tot2", "n_iter", "converged",
              "hit_maxiter", "diverged", "is_best", "cell_time_s"]


def cell_seed(k_idx):
    return SEED_BASE + k_idx


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
    diag = result["diag"]
    dt = result["cell_time_s"]
    rows = []
    for r in result["per_restart"]:
        row = dict(p=P, K=result["K"], K_index=k_idx, restart=r["restart"], seed=r["seed"],
                   energy=r["energy"], E_exact=(diag["E_exact"] if diag else None),
                   err_pct=(abs(r["energy"] - diag["E_exact"]) / max(abs(diag["E_exact"]), 1e-9) * 100
                            if (r["energy"] is not None and diag) else None),
                   fidelity=None, Q_tot2=None,
                   n_iter=r["n_iter"], converged=r["converged"],
                   hit_maxiter=r["hit_maxiter"], diverged=r["diverged"],
                   is_best=(r["restart"] == result["best_restart"]), cell_time_s=dt)
        if row["is_best"] and diag:
            row["fidelity"] = diag["fidelity"]
            row["Q_tot2"] = diag["Q_tot2"]
        rows.append(row)
    return rows


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    done_cells = load_checkpoint()

    cells = []
    for k_idx, K in enumerate(K_GRID):
        cell_key = f"{k_idx}"
        if cell_key not in done_cells:
            cells.append((cell_key, k_idx, float(K)))

    total_cells = len(K_GRID)
    print(f"Dense kink sweep: {total_cells} K cells total, {len(done_cells)} already done, "
          f"{len(cells)} remaining. {N_WORKERS} parallel workers.", flush=True)
    if not cells:
        print("Nothing to do.")
        return

    t_start = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {}
        for cell_key, k_idx, K in cells:
            args = dict(N=N, F=F, L=L, ansatz="gi", fold_lambda=1, p=P, eps=EPS, x=X, K=K,
                        n_restarts=N_RESTARTS, seed=cell_seed(k_idx), maxiter=MAXITER)
            fut = pool.submit(run_cell_worker, args)
            futures[fut] = (cell_key, k_idx, K)

        for fut in as_completed(futures):
            cell_key, k_idx, K = futures[fut]
            result = fut.result()
            rows = rows_from_result(result, k_idx)
            append_rows(rows)
            done_cells.add(cell_key)
            save_checkpoint(done_cells)

            best_e = next((r["energy"] for r in rows if r["is_best"]), None)
            print(f"[{len(done_cells)}/{total_cells}] K={K:+.2f} best_E={best_e} "
                  f"(elapsed {(time.time()-t_start)/60:.1f} min)", flush=True)

    print(f"Dense kink sweep done in {(time.time()-t_start)/60:.1f} min.")


if __name__ == "__main__":
    main()
