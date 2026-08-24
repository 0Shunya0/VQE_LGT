"""
Targeted basin-capture check at the L=3 sanity-gate failure point. Diagnosis
(from the previous run): kink_dense_N3_L3_noiseless.csv matches exact to a
constant 0.0060 pre-transition (K=4.00-5.25) and 0.0000 post-transition
(K>=5.75); the single failure is K=5.50, where the optimum locks onto the
polarized plateau (1.0000) instead of the true still-on-the-linear-branch
value (0.4340 = -1.566 + 8*0.25). The competing polarized state is only
0.566 higher and is a near-product state with a large basin -- this looks
like basin capture at a near-degenerate point (8 restarts too few to escape
it reliably), not an L=3 expressibility limit, so more restarts should fix
it if that diagnosis is right.

N=3, L=3, GI, K = 5.25, 5.50, 5.75 (the point that failed, plus its two
immediate neighbors as controls -- 5.25 and 5.75 should already be fine at
8 restarts, so they double as a check that 64 restarts doesn't change a
correct answer). Two sweeps: A' (p=0, eps=0) and B' (p=0.01, eps=SPAM_
DEFAULT), 64 restarts (not 8), maxiter=4000 (not 2000). Distinct SEED_BASE
from every prior run.

Appends to kink_dense_N3_L3_noiseless.csv / _noisy.csv with a new run_tag
column: existing rows are retroactively tagged 'dense8' (8 restarts,
maxiter=2000, the original sweep), new rows are tagged 'basin64' (64
restarts, maxiter=4000). Nothing is deleted; downstream analysis prefers
'basin64' rows for K in {5.25,5.50,5.75} and falls back to 'dense8'
everywhere else.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

import schwinger.core as core
from schwinger.parallel_worker import run_cell_worker

N, F, L = 3, 2, 3
X = 16.0
K_POINTS = [5.25, 5.50, 5.75]
K_INDEX_MAP = {5.25: 5, 5.50: 6, 5.75: 7}  # matches the original K=linspace(4,7,13) indices
N_RESTARTS = 64
MAXITER = 4000
SEED_BASE_NOISELESS = 110000  # distinct from every prior sweep (10000..100000)
SEED_BASE_NOISY = 120000
N_WORKERS = 6  # 3 K points x 2 sweeps, all independent

RESULTS_DIR = "results"
NOISELESS_CSV = os.path.join(RESULTS_DIR, "kink_dense_N3_L3_noiseless.csv")
NOISY_CSV = os.path.join(RESULTS_DIR, "kink_dense_N3_L3_noisy.csv")
CKPT_PATH = os.path.join(RESULTS_DIR, "kink_basin_check_checkpoint.json")

CSV_FIELDS = ["p", "K", "K_index", "run_tag", "restart", "seed", "energy", "E_exact",
              "err_pct", "fidelity", "Q_tot2", "n_iter", "converged",
              "hit_maxiter", "diverged", "is_best", "cell_time_s"]


def ensure_run_tag_column(csv_path, default_tag="dense8"):
    df = pd.read_csv(csv_path)
    if "run_tag" not in df.columns:
        df.insert(df.columns.get_loc("K_index") + 1, "run_tag", default_tag)
        df = df[CSV_FIELDS]
        df.to_csv(csv_path, index=False)
        print(f"Added run_tag='{default_tag}' to {len(df)} existing rows in {csv_path}", flush=True)


def load_checkpoint():
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH) as f:
            return set(json.load(f)["done_cells"])
    return set()


def save_checkpoint(done_cells):
    with open(CKPT_PATH, "w") as f:
        json.dump({"done_cells": sorted(done_cells)}, f)


def append_rows(csv_path, rows):
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writerows(rows)


def rows_from_result(result, k_idx, p_val):
    diag = result["diag"]
    dt = result["cell_time_s"]
    rows = []
    for r in result["per_restart"]:
        row = dict(p=p_val, K=result["K"], K_index=k_idx, run_tag="basin64",
                   restart=r["restart"], seed=r["seed"],
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
    ensure_run_tag_column(NOISELESS_CSV)
    ensure_run_tag_column(NOISY_CSV)
    done_cells = load_checkpoint()

    cells = []
    for K in K_POINTS:
        k_idx = K_INDEX_MAP[K]
        key_nl = f"nl_{k_idx}"
        if key_nl not in done_cells:
            cells.append((key_nl, k_idx, K, 0.0, 0.0, SEED_BASE_NOISELESS + k_idx, NOISELESS_CSV))
        key_ny = f"ny_{k_idx}"
        if key_ny not in done_cells:
            cells.append((key_ny, k_idx, K, 0.01, core.SPAM_DEFAULT, SEED_BASE_NOISY + k_idx, NOISY_CSV))

    total_cells = len(K_POINTS) * 2
    print(f"Basin-capture check: {total_cells} cells total, {len(done_cells)} already done, "
          f"{len(cells)} remaining. {N_WORKERS} parallel workers, {N_RESTARTS} restarts/cell, "
          f"maxiter={MAXITER}.", flush=True)
    if not cells:
        print("Nothing to do.")
        return

    t_start = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {}
        for cell_key, k_idx, K, p, eps, seed, csv_path in cells:
            args = dict(N=N, F=F, L=L, ansatz="gi", fold_lambda=1, p=p, eps=eps, x=X, K=K,
                        n_restarts=N_RESTARTS, seed=seed, maxiter=MAXITER)
            fut = pool.submit(run_cell_worker, args)
            futures[fut] = (cell_key, k_idx, K, p, csv_path)

        for fut in as_completed(futures):
            cell_key, k_idx, K, p, csv_path = futures[fut]
            result = fut.result()
            rows = rows_from_result(result, k_idx, p)
            append_rows(csv_path, rows)
            done_cells.add(cell_key)
            save_checkpoint(done_cells)

            best_e = next((r["energy"] for r in rows if r["is_best"]), None)
            print(f"[{len(done_cells)}/{total_cells}] {cell_key} K={K:+.2f} p={p} best_E={best_e} "
                  f"(elapsed {(time.time()-t_start)/60:.1f} min)", flush=True)

    print(f"Basin-capture check done in {(time.time()-t_start)/60:.1f} min.")


if __name__ == "__main__":
    main()
