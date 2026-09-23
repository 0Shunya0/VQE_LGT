"""
Ensemble version of the same-channel post-hoc control. The single-theta* result
depends on which of many equally-good noiseless optima the optimizer returned,
so: rerun exp08's noiseless optimization (identical seeds, 8 restarts/K) keeping
EVERY restart's theta, evaluate all of them at p=0.01 (no re-optimization), and
bootstrap random selections of one "equally good" restart per K.

A restart counts as equally good if its noiseless energy is within TOL of the
exact ground energy (so basin-captured / unconverged restarts are excluded).
Kink suppression and retained fractions use make_summary's own helpers.
New files only.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _os.path.join(_ROOT, "analysis"))
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import schwinger.core as core
import schwinger.backend as qb
from schwinger.parallel_worker import run_cell_worker

N, F, L, X = 3, 2, 3, 16.0
K_GRID = np.linspace(4.0, 7.0, 13)
SEED_BASE = 90000
TOL = 0.05
N_BOOT = 20000
STAGE1 = "results/posthoc_local_ensemble_thetas_N3_L3.json"
OUT = "results/posthoc_local_ensemble_N3_L3.json"


def optimize(k_idx):
    K = float(K_GRID[k_idx])
    res = run_cell_worker(dict(N=N, F=F, L=L, ansatz="gi", fold_lambda=1, p=0.0, eps=0.0, x=X, K=K,
                               n_restarts=8, seed=SEED_BASE + k_idx, maxiter=2000))
    return dict(K=K, E_exact=res["diag"]["E_exact"],
                restarts=[dict(E_noiseless=r["energy"], theta=r["theta"]) for r in res["per_restart"]])


def main():
    import make_summary as ms
    if _os.path.exists(STAGE1):
        cells = json.load(open(STAGE1))["cells"]
    else:
        with ProcessPoolExecutor(max_workers=13) as pool:
            cells = list(pool.map(optimize, range(len(K_GRID))))
        json.dump(dict(cells=cells), open(STAGE1, "w"))
    ev = qb.NoisyEvaluator(N, F, L, ansatz="gi", p=0.01, eps=core.SPAM_DEFAULT, x=X)
    K = np.array([c["K"] for c in cells])
    E_exact = np.array([c["E_exact"] for c in cells])
    # noisy post-hoc energy of every restart; mask of equally-good restarts
    E_ph = [[ev.energy(np.array(r["theta"]), c["K"]) for r in c["restarts"]] for c in cells]
    good = [[abs(r["E_noiseless"] - c["E_exact"]) <= TOL for r in c["restarts"]] for c in cells]
    print("equally-good restarts per K:", [sum(g) for g in good])
    print("post-hoc spread across good restarts per K (max-min):",
          [round(float(np.ptp([e for e, g in zip(E_ph[i], good[i]) if g])), 2) if any(good[i]) else None
           for i in range(len(cells))])
    usable = [i for i in range(len(cells)) if any(good[i])]
    fit_K = [i for i, k in enumerate(K) if (ms.PRE_RANGE[0] - 1e-9 <= k <= ms.PRE_RANGE[1] + 1e-9)
             or (ms.POST_RANGE[0] - 1e-9 <= k <= ms.POST_RANGE[1] + 1e-9)]
    assert all(i in usable for i in fit_K), "a fitted K has no equally-good restart"

    dEmix = {}
    for name, (lo, hi) in [("ordered", ms.PRE_RANGE), ("saturated", ms.POST_RANGE)]:
        Kmix = np.linspace(lo, hi, 5)
        dEmix[name] = ms._hyst_linfit(Kmix, np.array([core.exact_and_mix(3, k, convention="nu0only")[1] for k in Kmix]))[0]

    def metrics(E_noisy_curve):
        slopes = {}
        for name, (lo, hi) in [("pre", ms.PRE_RANGE), ("post", ms.POST_RANGE)]:
            Kp, Er = ms._hyst_select(K, E_exact, lo, hi)
            _, En = ms._hyst_select(K, E_noisy_curve, lo, hi)
            slopes[name] = (ms._hyst_linfit(Kp, Er)[0], ms._hyst_linfit(Kp, En)[0])
        kink_ref = slopes["pre"][0] - slopes["post"][0]
        kink_ny = slopes["pre"][1] - slopes["post"][1]
        rf_o = (dEmix["ordered"] - slopes["pre"][1]) / (dEmix["ordered"] - slopes["pre"][0])
        rf_s = (dEmix["saturated"] - slopes["post"][1]) / (dEmix["saturated"] - slopes["post"][0])
        return 100 * (1 - kink_ny / kink_ref), rf_o, rf_s

    rng = np.random.default_rng(0)
    draws = []
    for _ in range(N_BOOT):
        curve = np.full(len(K), np.nan)
        for i in range(len(K)):
            idx = [j for j, g in enumerate(good[i]) if g]
            curve[i] = E_ph[i][rng.choice(idx)] if idx else np.nan
        draws.append(metrics(np.where(np.isnan(curve), 0.0, curve)))
    d = np.array(draws)
    names = ["suppression_pct", "retained_ordered", "retained_saturated"]
    summ = {}
    for j, nme in enumerate(names):
        x = d[:, j]
        summ[nme] = dict(mean=float(x.mean()), std=float(x.std()), p05=float(np.percentile(x, 5)),
                         p25=float(np.percentile(x, 25)), median=float(np.median(x)),
                         p75=float(np.percentile(x, 75)), p95=float(np.percentile(x, 95)))
    summ["frac_draws_suppression_below_inloop_16.16"] = float(np.mean(d[:, 0] < 16.16))
    summ["frac_draws_suppression_below_global_26.77"] = float(np.mean(d[:, 0] < 26.77))
    summ["frac_draws_saturated_retained_above_inloop_0.920"] = float(np.mean(d[:, 2] > 0.920))
    json.dump(dict(K=K.tolist(), E_posthoc_all=E_ph, good=good, summary=summ, n_boot=N_BOOT, tol=TOL),
              open(OUT, "w"), indent=2)
    for k, v in summ.items():
        print(k, {a: round(b, 3) for a, b in v.items()} if isinstance(v, dict) else round(v, 4))


if __name__ == "__main__":
    main()
