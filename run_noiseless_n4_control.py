"""
Bug 3: noiseless control for the N=4 spot check. Same settings as
run_exp3.py (N=4, L=5, GI, 4 restarts, maxiter 2000) but p=0, eps=0, so any
degradation seen in noisy_N4_spot.csv can be attributed to noise only if the
noiseless case reaches good accuracy with the same restart budget. Also the
validation gate (p=0, eps=0) needed 16 restarts to hit sub-1e-3 accuracy at
N=4,L=5 -- so 4 restarts here is expected to be a much rougher optimum
regardless of noise; that comparison is exactly the point.

K points corrected per results/boundary_check.md (Bug 2): the N=4 boundary
that continues the N=2 (3.95) / N=3 (5.6) trend is the TERMINAL flavor-0
occupation crossing at K~6.4, not run_exp3.py's boundary_K(N=4)=2.5. Uses
K = 0, 3.2, 6.4 (0, half-boundary, boundary) -- NOT the same numeric K points
noisy_N4_spot.csv used (0, 1.25, 2.5).

ALSO runs K = 1.25 and K = 2.5 (labeled "noisy-matched-1.25" and
"noisy-matched-2.5") so noisy_N4_spot.csv's three K points (0, 1.25, 2.5) all
have a like-for-like noiseless counterpart at the exact same K, not just
K=0. The 3.2/6.4 rows stay in the CSV as the corrected-boundary points with
no noisy counterpart; see results/SUMMARY.md for how the two groups are
presented.
"""
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import schwinger_core as core
from parallel_worker import run_cell_worker

N, F, L = 4, 2, 5
X = 16.0
EPS = 0.0
P = 0.0
N_RESTARTS = 4
MAXITER = 2000
SEED_BASE = 50000  # distinct from 10000/20000/30000/40000
N_WORKERS = 5

KB_CORRECTED = 6.4  # from results/boundary_check.md
K_POINTS = [(0.0, "K=0"), (KB_CORRECTED / 2, "half-boundary"), (KB_CORRECTED, "boundary"),
            (1.25, "noisy-matched-1.25"), (2.5, "noisy-matched-2.5")]

RESULTS_DIR = "results"
CSV_PATH = os.path.join(RESULTS_DIR, "noiseless_N4_control.csv")
CKPT_PATH = os.path.join(RESULTS_DIR, "noiseless_N4_control_checkpoint.json")

CSV_FIELDS = ["p", "K", "K_label", "restart", "seed", "energy", "E_exact",
              "err_pct", "fidelity", "Q_tot2", "n_iter", "converged",
              "hit_maxiter", "diverged", "cell_time_s"]


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


def rows_from_result(result, K_label):
    diag = result["diag"]
    dt = result["cell_time_s"]
    rows = []
    for r in result["per_restart"]:
        row = dict(p=P, K=result["K"], K_label=K_label, restart=r["restart"], seed=r["seed"],
                   energy=r["energy"], E_exact=(diag["E_exact"] if diag else None),
                   err_pct=(abs(r["energy"] - diag["E_exact"]) / max(abs(diag["E_exact"]), 1e-9) * 100
                            if (r["energy"] is not None and diag) else None),
                   fidelity=None, Q_tot2=None,
                   n_iter=r["n_iter"], converged=r["converged"],
                   hit_maxiter=r["hit_maxiter"], diverged=r["diverged"], cell_time_s=dt)
        if r["restart"] == result["best_restart"] and diag:
            row["fidelity"] = diag["fidelity"]
            row["Q_tot2"] = diag["Q_tot2"]
        rows.append(row)
    return rows


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    done_cells = load_checkpoint()

    cells = []
    for k_idx, (K, K_label) in enumerate(K_POINTS):
        cell_key = f"{k_idx}"
        if cell_key not in done_cells:
            cells.append((cell_key, k_idx, K, K_label))

    total_cells = len(K_POINTS)
    print(f"Noiseless N=4 control: {total_cells} K cells total, {len(done_cells)} already done, "
          f"{len(cells)} remaining. {N_WORKERS} parallel workers.", flush=True)
    if not cells:
        print("Nothing to do.")
        return

    t_start = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {}
        for cell_key, k_idx, K, K_label in cells:
            args = dict(N=N, F=F, L=L, ansatz="gi", fold_lambda=1, p=P, eps=EPS, x=X, K=K,
                        n_restarts=N_RESTARTS, seed=SEED_BASE + k_idx, maxiter=MAXITER)
            fut = pool.submit(run_cell_worker, args)
            futures[fut] = (cell_key, K_label)

        for fut in as_completed(futures):
            cell_key, K_label = futures[fut]
            result = fut.result()
            rows = rows_from_result(result, K_label)
            append_rows(rows)
            done_cells.add(cell_key)
            save_checkpoint(done_cells)

            best_e = next((r["energy"] for r in rows if r["fidelity"] is not None), None)
            print(f"[{len(done_cells)}/{total_cells}] K={result['K']:+.2f} ({K_label}) "
                  f"best_E={best_e} (elapsed {(time.time()-t_start)/60:.1f} min)", flush=True)

    print("Noiseless N=4 control done.")


if __name__ == "__main__":
    main()
