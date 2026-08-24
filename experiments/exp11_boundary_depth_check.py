"""
Follow-up to the CASE (ii) finding in results/boundary_restart_scaling_N3.csv:
32 restarts at maxiter=4000 did not recover the exact ground state near the
K=4-5.25 near-degenerate crossing (err% unchanged from the 8-restart value at
every K tested), so the failure is L=2 expressibility, not optimizer
under-convergence. This checks whether extra ansatz depth (L=3, L=4) at the
worst point (K=4.5, 65.2% error at L=2) recovers it.

N=3, L=3 and L=4, GI, p=0, eps=0, K=4.5, 8 restarts, maxiter=2000. Distinct
SEED_BASE from every prior sweep. Appends to
results/boundary_restart_scaling_N3.csv with an added "L" column -- the
existing L=2 rows are rewritten with L=2 filled in (they only ever used L=2),
then the new L=3/L=4 rows are appended in the same schema.
"""
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from parallel_worker import run_cell_worker

N, F = 3, 2
X = 16.0
K = 4.5
P = 0.0
EPS = 0.0
N_RESTARTS = 8
MAXITER = 2000
L_VALUES = [3, 4]
SEED_BASE = 80000  # distinct from every prior sweep (10000..70000)
N_WORKERS = 2

RESULTS_DIR = "results"
CSV_PATH = os.path.join(RESULTS_DIR, "boundary_restart_scaling_N3.csv")
CKPT_PATH = os.path.join(RESULTS_DIR, "boundary_restart_scaling_N3_depthcheck_checkpoint.json")

CSV_FIELDS = ["p", "K", "K_index", "L", "restart", "seed", "energy", "E_exact",
              "err_pct", "fidelity", "Q_tot2", "n_iter", "converged",
              "hit_maxiter", "diverged", "is_best", "cell_time_s"]


def ensure_L_column():
    """Add L=2 to every existing row (the 32-restart control only ever used
    L=2) if the CSV doesn't already have an L column, once, idempotently."""
    df = pd.read_csv(CSV_PATH)
    if "L" not in df.columns:
        df.insert(df.columns.get_loc("K_index") + 1, "L", 2)
        df = df[CSV_FIELDS]
        df.to_csv(CSV_PATH, index=False)
        print(f"Added L=2 column to {len(df)} existing rows in {CSV_PATH}", flush=True)


def load_checkpoint():
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH) as f:
            return set(json.load(f)["done_cells"])
    return set()


def save_checkpoint(done_cells):
    with open(CKPT_PATH, "w") as f:
        json.dump({"done_cells": sorted(done_cells)}, f)


def append_rows(rows):
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writerows(rows)


def rows_from_result(result, L):
    diag = result["diag"]
    dt = result["cell_time_s"]
    rows = []
    for r in result["per_restart"]:
        row = dict(p=P, K=result["K"], K_index=0, L=L, restart=r["restart"], seed=r["seed"],
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
    ensure_L_column()
    done_cells = load_checkpoint()

    cells = [(str(L), L) for L in L_VALUES if str(L) not in done_cells]
    print(f"Depth check: {len(L_VALUES)} L cells total, {len(done_cells)} already done, "
          f"{len(cells)} remaining. {N_WORKERS} parallel workers.", flush=True)
    if not cells:
        print("Nothing to do.")
        return

    t_start = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {}
        for cell_key, L in cells:
            args = dict(N=N, F=F, L=L, ansatz="gi", fold_lambda=1, p=P, eps=EPS, x=X, K=K,
                        n_restarts=N_RESTARTS, seed=SEED_BASE + L, maxiter=MAXITER)
            fut = pool.submit(run_cell_worker, args)
            futures[fut] = (cell_key, L)

        for fut in as_completed(futures):
            cell_key, L = futures[fut]
            result = fut.result()
            rows = rows_from_result(result, L)
            append_rows(rows)
            done_cells.add(cell_key)
            save_checkpoint(done_cells)

            best_e = next((r["energy"] for r in rows if r["is_best"]), None)
            print(f"[{len(done_cells)}/{len(L_VALUES)}] L={L} K={K} best_E={best_e} "
                  f"(elapsed {(time.time()-t_start)/60:.1f} min)", flush=True)

    print(f"Depth check done in {(time.time()-t_start)/60:.1f} min.")


if __name__ == "__main__":
    main()
