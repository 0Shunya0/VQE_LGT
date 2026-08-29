"""
EXPERIMENT 3: N=4 spot check, L=5, GI ansatz (nq=8, 75 params, n_CX=70).
Parallelized across (p,K) cells with a process pool (see parallel_worker.py).

Only 3 K points (K=0, one interior point, one at the N=4 phase boundary,
located via schwinger_core.boundary_K(N=4) -- the existing driver's own
boundary finder, reused rather than re-derived), p in {0.01,0.02}, 4 restarts.
Convergence status (converged / hit_maxiter / diverged) is recorded per
restart so stalling in the flattened p/d>>1 landscape shows up directly in
the CSV, not just in the final energy.

K-POINT LABELLING (Round 1 / Round 7 note -- read this before using
results/noisy_N4_spot.csv). This run was generated when boundary_K(N=4)
still returned the buggy, non-terminal K=2.50 crossing instead of the
corrected terminal crossing at K=6.40 (see results/boundary_check.md). The
three K points it actually sampled are K = 0.00, 1.25, 2.50. Read those as
the low-K region matched to the noiseless restart-budget control (exp07 /
results/noiseless_N4_control.csv), which evaluates the same K values, NOT
as the phase-boundary region. The corrected N=4 boundary is K = 6.40, and
none of the three points here reach it. The data is not rerun (per the
standing rule that no numerical result changes); only its labelling was
ambiguous, and this note fixes that. The paper's Section VI E is already
consistent with this -- it says only "three chemical potentials" and makes
no boundary claim about them.
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

N, F, L = 4, 2, 5
X = 16.0
P_VALUES = [0.01, 0.02]
EPS = core.SPAM_DEFAULT
N_RESTARTS = 4
MAXITER = 2000
SEED_BASE = 30000
N_WORKERS = 6  # only 6 cells total; one worker per cell is enough

RESULTS_DIR = "results"
CSV_PATH = os.path.join(RESULTS_DIR, "noisy_N4_spot.csv")
TRACE_PATH = os.path.join(RESULTS_DIR, "noisy_N4_spot_traces.json")
CKPT_PATH = os.path.join(RESULTS_DIR, "noisy_N4_spot_checkpoint.json")

CSV_FIELDS = ["p", "K", "K_label", "restart", "seed", "energy", "E_exact",
              "err_pct", "fidelity", "Q_tot2", "n_iter", "converged",
              "hit_maxiter", "diverged", "cell_time_s"]


def k_points():
    kb = core.boundary_K(N=N, convention="nu0only")
    return [(0.0, "K=0"), (kb / 2, "interior"), (kb, "boundary")], kb


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


def append_trace(key, entry):
    data = {}
    if os.path.exists(TRACE_PATH):
        with open(TRACE_PATH) as f:
            data = json.load(f)
    data[key] = entry
    with open(TRACE_PATH, "w") as f:
        json.dump(data, f)


def rows_from_result(result, K_label):
    p, K = result["p"], result["K"]
    diag = result["diag"]
    dt = result["cell_time_s"]
    rows = []
    for r in result["per_restart"]:
        row = dict(p=p, K=K, K_label=K_label, restart=r["restart"], seed=r["seed"],
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
    kpts, kb = k_points()
    print(f"N=4 phase boundary located at |K|={kb:.3f} (via schwinger_core.boundary_K)", flush=True)

    cells = []
    for p_idx, p in enumerate(P_VALUES):
        for k_idx, (K, K_label) in enumerate(kpts):
            cell_key = f"{p_idx}_{k_idx}"
            if cell_key not in done_cells:
                cells.append((cell_key, p_idx, k_idx, p, K, K_label))

    total_cells = len(P_VALUES) * len(kpts)
    print(f"Experiment 3: {total_cells} (p,K) cells total, {len(done_cells)} already done, "
          f"{len(cells)} remaining. {N_WORKERS} parallel workers.", flush=True)
    if not cells:
        print("Nothing to do.")
        return

    t_start = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {}
        for cell_key, p_idx, k_idx, p, K, K_label in cells:
            args = dict(N=N, F=F, L=L, ansatz="gi", fold_lambda=1, p=p, eps=EPS, x=X, K=K,
                        n_restarts=N_RESTARTS, seed=SEED_BASE + p_idx * 100 + k_idx, maxiter=MAXITER,
                        capture_trace=True)
            fut = pool.submit(run_cell_worker, args)
            futures[fut] = (cell_key, K_label)

        for fut in as_completed(futures):
            cell_key, K_label = futures[fut]
            result = fut.result()
            rows = rows_from_result(result, K_label)
            append_rows(rows)
            if result["best_restart"] is not None:
                best_trace = result["per_restart"][result["best_restart"]]["trace"]
                append_trace(cell_key, dict(p=result["p"], K=result["K"], K_label=K_label,
                                             trace=best_trace))
            done_cells.add(cell_key)
            save_checkpoint(done_cells)

            best_e = next((r["energy"] for r in rows if r["fidelity"] is not None), None)
            all_hit_maxiter = all(r["hit_maxiter"] for r in rows)
            print(f"[{len(done_cells)}/{total_cells}] p={result['p']} K={result['K']:+.2f} ({K_label}) "
                  f"best_E={best_e} all_stalled={all_hit_maxiter} "
                  f"(elapsed {(time.time()-t_start)/60:.1f} min)", flush=True)

    print("Experiment 3 done.")


if __name__ == "__main__":
    main()
