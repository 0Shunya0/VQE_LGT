"""
Shared driver for the warm-start continuation sweep (N=3, L=3, GI,
K=linspace(4,7,13)), now supporting BOTH traversal directions so a
bidirectional (hysteresis) continuation can be built: an ascending pass
(K=4.00->7.00) and a descending pass (K=7.00->4.00), each strictly serial in
its own traversal order since every step's warm-start candidate depends on
the previous step's winner. The first point of whichever pass is being run
uses 8 random restarts (no previous point to warm-start from); every
subsequent point in that pass runs one warm-start candidate (seeded with the
previous step's winning theta) concurrently with 3 random-restart
candidates, keeping whichever wins -- so warm-starting can never do worse
than plain random restarts would have. maxiter=4000 throughout.

The 4 (or 8, at the pass's first point) candidates within a single K step
run in parallel via ProcessPoolExecutor; steps themselves run strictly in
traversal order in the main process. Checkpoints after every step (direction-
specific file, so the two passes can checkpoint independently even when
writing to the same CSV), storing the winning theta so a resume picks up the
correct warm-start seed for the next step.
"""
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from warmstart_worker import run_single_restart

RESULTS_DIR = "results"
N, F, L = 3, 2, 3
X = 16.0
K_GRID = np.linspace(4.0, 7.0, 13)
MAXITER = 4000
N_RANDOM_FIRST = 8   # at the pass's first point
N_RANDOM_LATER = 3   # alongside the warm start, at every subsequent point in the pass

CSV_FIELDS = ["p", "K", "K_index", "direction", "restart_id", "init_mode", "seed", "energy",
              "E_exact", "err_pct", "fidelity", "Q_tot2", "n_iter", "converged",
              "hit_maxiter", "diverged", "is_best", "warm_won"]


def _ckpt_path(csv_path, direction):
    return csv_path.replace(".csv", f"_{direction}_checkpoint.json")


def _load_checkpoint(csv_path, direction):
    p = _ckpt_path(csv_path, direction)
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        return d["last_pos"], np.array(d["last_best_theta"], dtype=float)
    return -1, None


def _save_checkpoint(csv_path, direction, pos, best_theta):
    with open(_ckpt_path(csv_path, direction), "w") as f:
        json.dump({"last_pos": pos, "last_best_theta": best_theta.tolist()}, f)


def _append_rows(csv_path, rows):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(rows)


def run_warmstart_sweep(p, eps, csv_path, seed_base, direction="up", n_workers=8,
                         ansatz="gi", fold_lambda=1):
    """direction='up': traverse K_GRID ascending (index 0..12).
    direction='down': traverse K_GRID descending (index 12..0).
    Both write to the same csv_path with a 'direction' column; each
    direction checkpoints independently so the two passes don't collide."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    order = list(range(len(K_GRID))) if direction == "up" else list(range(len(K_GRID) - 1, -1, -1))

    last_pos, prev_theta = _load_checkpoint(csv_path, direction)
    print(f"[{csv_path}][{direction}] resuming after pass-position={last_pos}" if last_pos >= 0
          else f"[{csv_path}][{direction}] starting fresh", flush=True)

    t_start = time.time()
    for pos, k_idx in enumerate(order):
        if pos <= last_pos:
            continue
        K = float(K_GRID[k_idx])

        candidates = []
        if pos == 0:
            for r in range(N_RANDOM_FIRST):
                candidates.append(dict(N=N, F=F, L=L, ansatz=ansatz, fold_lambda=fold_lambda,
                                        p=p, eps=eps, x=X, K=K, maxiter=MAXITER,
                                        init_mode="random", seed=seed_base + r))
        else:
            candidates.append(dict(N=N, F=F, L=L, ansatz=ansatz, fold_lambda=fold_lambda,
                                    p=p, eps=eps, x=X, K=K, maxiter=MAXITER,
                                    init_mode="warm", theta0=prev_theta.tolist()))
            for r in range(N_RANDOM_LATER):
                candidates.append(dict(N=N, F=F, L=L, ansatz=ansatz, fold_lambda=fold_lambda,
                                        p=p, eps=eps, x=X, K=K, maxiter=MAXITER,
                                        init_mode="random", seed=seed_base + pos * 100 + r))

        with ProcessPoolExecutor(max_workers=min(n_workers, len(candidates))) as pool:
            results = list(pool.map(run_single_restart, candidates))

        finite = [r for r in results if r["energy"] is not None]
        if not finite:
            raise RuntimeError(f"All candidates diverged at K={K} ({direction}), cannot continue chain")
        best = min(finite, key=lambda r: r["energy"])
        warm_won = (best["init_mode"] == "warm") if pos > 0 else None

        rows = []
        for i, r in enumerate(results):
            is_best = (r is best)
            row = dict(p=p, K=K, K_index=k_idx, direction=direction, restart_id=i,
                       init_mode=r["init_mode"], seed=r["seed"], energy=r["energy"],
                       E_exact=(r["diag"]["E_exact"] if r["diag"] else None),
                       err_pct=(abs(r["energy"] - r["diag"]["E_exact"]) / max(abs(r["diag"]["E_exact"]), 1e-9) * 100
                                if (r["energy"] is not None and r["diag"]) else None),
                       fidelity=(r["diag"]["fidelity"] if r["diag"] else None),
                       Q_tot2=(r["diag"]["Q_tot2"] if r["diag"] else None),
                       n_iter=r["n_iter"], converged=r["converged"], hit_maxiter=r["hit_maxiter"],
                       diverged=r["diverged"], is_best=is_best,
                       warm_won=(warm_won if pos > 0 else None))
            rows.append(row)
        _append_rows(csv_path, rows)

        prev_theta = np.array(best["theta"], dtype=float)
        _save_checkpoint(csv_path, direction, pos, prev_theta)

        elapsed = (time.time() - t_start) / 60
        print(f"[{csv_path}][{direction}] pos={pos} K_index={k_idx} K={K:+.2f} "
              f"best_E={best['energy']:.4f} winner={best['init_mode']} "
              f"(elapsed {elapsed:.1f} min)", flush=True)

    print(f"[{csv_path}][{direction}] pass done in {(time.time()-t_start)/60:.1f} min.", flush=True)
