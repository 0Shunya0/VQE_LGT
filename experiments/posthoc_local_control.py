"""
Post-hoc control with the SAME local noise model as the in-loop runs.

N=3, L=3, GI, K=linspace(4,7,13). Stage 1: noiseless optimization exactly as
exp08_kink_dense_L3_noiseless (8 restarts, maxiter 2000, seed 90000+k_idx),
SAVING the winning theta* per K. Stage 2: evaluate each fixed theta* with
NoisyEvaluator at p=0.01, eps=SPAM_DEFAULT (no re-optimization). Stage 3:
kink suppression and branch slopes / retained fractions computed with
analysis/make_summary.py's own helpers (_hyst_linfit, PRE_RANGE, POST_RANGE,
dE_mix from core.exact_and_mix). Writes only new files; nothing saved is changed.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _os.path.join(_ROOT, "analysis"))
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import schwinger.core as core
import schwinger.backend as qb
from schwinger.parallel_worker import run_cell_worker

N, F, L, X = 3, 2, 3, 16.0
K_GRID = np.linspace(4.0, 7.0, 13)
SEED_BASE = int(_os.environ.get("POSTHOC_SEED_BASE", 90000))   # 90000 = exp08_kink_dense_L3_noiseless's seed base
_TAG = _os.environ.get("POSTHOC_TAG", "")
THETA_PATH = f"results/posthoc_local_theta_N3_L3{_TAG}.json"
OUT_PATH = f"results/posthoc_local_control_N3_L3{_TAG}.json"


def optimize(k_idx):
    K = float(K_GRID[k_idx])
    res = run_cell_worker(dict(N=N, F=F, L=L, ansatz="gi", fold_lambda=1, p=0.0, eps=0.0, x=X, K=K,
                               n_restarts=8, seed=SEED_BASE + k_idx, maxiter=2000))
    best = res["per_restart"][res["best_restart"]]
    return dict(k_idx=k_idx, K=K, E_noiseless=best["energy"], theta=best["theta"],
                restart_energies=[r["energy"] for r in res["per_restart"]],
                E_exact=res["diag"]["E_exact"])


def main():
    import make_summary as ms
    t0 = time.time()
    if _os.path.exists(THETA_PATH):
        stage1 = json.load(open(THETA_PATH))["cells"]
        print("stage 1 loaded from", THETA_PATH, flush=True)
    else:
        with ProcessPoolExecutor(max_workers=13) as pool:
            stage1 = list(pool.map(optimize, range(len(K_GRID))))
        json.dump(dict(note="noiseless (p=0) GI L=3 N=3 optima, exp08 settings; theta saved per K",
                       cells=stage1), open(THETA_PATH, "w"), indent=2)
        print(f"stage 1 done in {(time.time()-t0)/60:.1f} min, saved {THETA_PATH}", flush=True)

    ev_noisy = qb.NoisyEvaluator(N, F, L, ansatz="gi", p=0.01, eps=core.SPAM_DEFAULT, x=X)
    ev_clean = qb.NoisyEvaluator(N, F, L, ansatz="gi", p=0.0, eps=0.0, x=X)
    K = np.array([c["K"] for c in stage1])
    E_nl = np.array([c["E_noiseless"] for c in stage1])
    E_exact = np.array([c["E_exact"] for c in stage1])
    E_ph = np.array([ev_noisy.energy(np.array(c["theta"]), c["K"]) for c in stage1])
    E_clean_check = np.array([ev_clean.energy(np.array(c["theta"]), c["K"]) for c in stage1])
    print("max |E_noiseless(opt) - E_noiseless(re-eval)| =", float(np.max(np.abs(E_nl - E_clean_check))))

    def analyse(E_ref, E_noisy, label):
        slopes = {}
        for name, (lo, hi) in [("pre", ms.PRE_RANGE), ("post", ms.POST_RANGE)]:
            Kp, Er = ms._hyst_select(K, E_ref, lo, hi)
            _, En = ms._hyst_select(K, E_noisy, lo, hi)
            slopes[name] = dict(slope_ref=ms._hyst_linfit(Kp, Er)[0], slope_noisy=ms._hyst_linfit(Kp, En)[0])
        kink_ref = slopes["pre"]["slope_ref"] - slopes["post"]["slope_ref"]
        kink_ny = slopes["pre"]["slope_noisy"] - slopes["post"]["slope_noisy"]
        supp = 100 * (1 - kink_ny / kink_ref)
        rf = {}
        for bname, key, (lo, hi) in [("ordered", "pre", ms.PRE_RANGE), ("saturated", "post", ms.POST_RANGE)]:
            Kmix = np.linspace(lo, hi, 5)
            Emix = np.array([core.exact_and_mix(3, k, convention="nu0only")[1] for k in Kmix])
            dE_mix = ms._hyst_linfit(Kmix, Emix)[0]
            rf[bname] = (dE_mix - slopes[key]["slope_noisy"]) / (dE_mix - slopes[key]["slope_ref"])
        return dict(label=label, slopes=slopes, kink_ref=kink_ref, kink_noisy=kink_ny,
                    suppression_pct=supp, retained_fraction=rf)

    out = dict(K=K.tolist(), E_noiseless_opt=E_nl.tolist(), E_exact=E_exact.tolist(),
               E_posthoc_local=E_ph.tolist(),
               analysis_noiseless_opt_reference=analyse(E_nl, E_ph, "reference = optimized noiseless energies"),
               analysis_exact_reference=analyse(E_exact, E_ph, "reference = exact ground energies"))
    json.dump(out, open(OUT_PATH, "w"), indent=2)
    for k in ("analysis_noiseless_opt_reference", "analysis_exact_reference"):
        a = out[k]
        print(f"\n{a['label']}")
        print("  slopes:", {n: {m: round(v, 4) for m, v in d.items()} for n, d in a["slopes"].items()})
        print(f"  kink: ref={a['kink_ref']:.4f} post-hoc-local={a['kink_noisy']:.4f} -> suppression {a['suppression_pct']:.2f}%")
        print(f"  retained fraction: ordered={a['retained_fraction']['ordered']:.3f} "
              f"saturated={a['retained_fraction']['saturated']:.3f}")
    print("\nper-K: K, E_exact, E_noiseless_opt, E_posthoc_local")
    for row in zip(K, E_exact, E_nl, E_ph):
        print("  %.2f  %9.4f  %9.4f  %9.4f" % row)


if __name__ == "__main__":
    main()
