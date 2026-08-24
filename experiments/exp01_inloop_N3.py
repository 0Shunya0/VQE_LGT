"""
EXPERIMENT 1: in-loop noisy VQE, N=3, L=2, GI ansatz. Parallelized across
(p,K) cells with a process pool (see parallel_worker.py) -- ~8.9x measured
speedup with 20 workers on this 24-core machine, bringing the ~6.8h serial
estimate for this experiment down to roughly 20-25 minutes.

K sweep: 33 points over K in [-16,16] (schwinger_vqe_paper_final.ipynb's own
K_vals = np.linspace(-16, 16, 33), reused exactly). p in {0.005,0.01,0.02,0.05}.
8 random restarts per (K,p); best final energy kept, full spread recorded.
Checkpoints after every completed (K,p) cell. Prints a runtime estimate after
the first batch of cells completes and before the rest of the sweep.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import schwinger.core as core
from schwinger.parallel_worker import run_cell_worker

N, F, L = 3, 2, 2
X = 16.0
K_GRID = np.linspace(-16, 16, 33)
P_VALUES = [0.005, 0.01, 0.02, 0.05]
EPS = core.SPAM_DEFAULT  # 0.005
N_RESTARTS = 8
MAXITER = 2000
SEED_BASE = 10000
N_WORKERS = 20

RESULTS_DIR = "results"
CSV_PATH = os.path.join(RESULTS_DIR, "noisy_inloop_N3.csv")
CKPT_PATH = os.path.join(RESULTS_DIR, "noisy_inloop_N3_checkpoint.json")

CSV_FIELDS = ["p", "K", "K_index", "restart", "seed", "energy", "E_exact",
              "err_pct", "fidelity", "Q_tot2", "n_iter", "converged",
              "hit_maxiter", "diverged", "is_best", "cell_time_s"]


def cell_seed(p_idx, k_idx):
    return SEED_BASE + p_idx * 1000 + k_idx


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


def rows_from_result(result):
    p, K = result["p"], result["K"]
    diag = result["diag"]
    dt = result["cell_time_s"]  # actual worker-measured wall time for this cell
    rows = []
    for r in result["per_restart"]:
        row = dict(p=p, K=K, K_index=None, restart=r["restart"], seed=r["seed"],
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
    for p_idx, p in enumerate(P_VALUES):
        for k_idx, K in enumerate(K_GRID):
            cell_key = f"{p_idx}_{k_idx}"
            if cell_key not in done_cells:
                cells.append((cell_key, p_idx, k_idx, p, float(K)))

    total_cells = len(P_VALUES) * len(K_GRID)
    print(f"Experiment 1: {total_cells} (p,K) cells total, {len(done_cells)} already done, "
          f"{len(cells)} remaining. {N_WORKERS} parallel workers.", flush=True)
    if not cells:
        print("Nothing to do.")
        return

    t_sweep_start = time.time()
    n_reported = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {}
        for cell_key, p_idx, k_idx, p, K in cells:
            args = dict(N=N, F=F, L=L, ansatz="gi", fold_lambda=1, p=p, eps=EPS, x=X, K=K,
                        n_restarts=N_RESTARTS, seed=cell_seed(p_idx, k_idx), maxiter=MAXITER)
            fut = pool.submit(run_cell_worker, args)
            futures[fut] = (cell_key, p_idx, k_idx, p, K)

        for fut in as_completed(futures):
            cell_key, p_idx, k_idx, p, K = futures[fut]
            result = fut.result()

            rows = rows_from_result(result)
            for row in rows:
                row["K_index"] = k_idx
            append_rows(rows)
            done_cells.add(cell_key)
            save_checkpoint(done_cells)
            n_reported += 1

            best_e = next((r["energy"] for r in rows if r["is_best"]), None)
            elapsed = time.time() - t_sweep_start
            print(f"[{len(done_cells)}/{total_cells}] p={p} K={K:+.1f} best_E={best_e} "
                  f"(elapsed {elapsed/60:.1f} min)", flush=True)

            if n_reported == min(N_WORKERS, len(cells)):
                rate = elapsed / n_reported
                remaining = len(cells) - n_reported
                est_remaining = remaining * rate
                print(f"\n>>> RUNTIME ESTIMATE: first batch of {n_reported} parallel cells took "
                      f"{elapsed/60:.1f} min ({rate:.1f}s/cell throughput-adjusted). "
                      f"{remaining} cells remain -> est. {est_remaining/60:.1f} min more.\n", flush=True)

    print(f"Experiment 1 done in {(time.time()-t_sweep_start)/60:.1f} min total this run.")


if __name__ == "__main__":
    main()
