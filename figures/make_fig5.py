"""
Regenerate fig5 (merged optimization-stability + convergence-trace figure)
from schwinger_vqe_paper_final.ipynb's Figure 5 cell. Replaces the former
separate stability (old Fig 5) and convergence (old Fig 6) figures.

The stability panels use np.random.default_rng(42); the convergence panel
uses the cell's inline conv_trace (seed=7). Both run COBYLA live, so the
result is deterministic given the scipy/numpy versions; if the regenerated
PNG does not match the committed one, the local scipy COBYLA differs from the
one the committed figure was made with.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from schwinger.core import (build_H_full, build_operators, make_psi0,
                            measure_state, gauge_invariant_ansatz, hw_efficient_ansatz)
import style

FIG_DIR = "pra_figures"
X = 16.0
R = dict(ksweep=8, penalty=10, stability=20, scaling=6, conv=5, maxiter=2500)

style.apply()


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    # Figure 5 (merged): optimization stability over R['stability'] restarts + convergence traces.
    N, F, nq = 2, 2, 4
    H0 = build_H_full(X, 0.0, N, F)
    Q_tot, Q2, N0_op, N1_op = build_operators(N, F)
    bs = [s for s in range(2 ** nq) if bin(s).count("1") == nq // 2]
    w0, v0 = eigh(H0[np.ix_(bs, bs)]); E_q0_0 = float(w0[0])
    psi_q0_0 = np.zeros(2 ** nq, dtype=complex)
    for i, s in enumerate(bs):
        psi_q0_0[s] = v0[i, 0]
    psi0 = make_psi0(N, F)
    print(f"Stability + convergence: {R['stability']} restarts at K=0, N=2, L=2")
    stability_data = {}
    for name, fn, nper in [("GI", gauge_invariant_ansatz, (nq - 1) + nq), ("HW", hw_efficient_ansatz, 2 * nq)]:
        rng = np.random.default_rng(42); obs_list = []
        for _ in range(R['stability']):
            t0 = rng.uniform(-np.pi, np.pi, 2 * nper)
            r = minimize(lambda th: float(np.real(fn(th, psi0, nq, 2).conj() @ H0 @ fn(th, psi0, nq, 2))),
                         t0, method='COBYLA', options={'maxiter': R['maxiter'], 'rhobeg': 0.5})
            psi_f = fn(r.x, psi0, nq, 2)
            o = measure_state(psi_f, Q_tot, Q2, N0_op, N1_op, psi_q0_0)
            o['E'] = float(np.real(psi_f.conj() @ H0 @ psi_f)); obs_list.append(o)
        stability_data[name] = obs_list

    def conv_trace(fn, H, nq, L, n_params, psi0, n_steps=60, n_tries=5, seed=7):
        rng = np.random.default_rng(seed); best, best_final = None, np.inf
        for _ in range(n_tries):
            theta = rng.uniform(-np.pi, np.pi, n_params); trace = []
            for step in range(n_steps):
                rho = max(0.5 * (0.95 ** step), 0.03)
                r = minimize(lambda th: float(np.real(fn(th, psi0, nq, L).conj() @ H @ fn(th, psi0, nq, L))),
                             theta, method='COBYLA', options={'maxiter': max(n_params + 2, 12), 'rhobeg': rho})
                theta = r.x
                trace.append(float(np.real(fn(theta, psi0, nq, L).conj() @ H @ fn(theta, psi0, nq, L))))
            if trace[-1] < best_final:
                best_final, best = trace[-1], trace
        return best
    tr_gi = conv_trace(gauge_invariant_ansatz, H0, nq, 2, 14, psi0, n_tries=R['conv'])
    tr_hw = conv_trace(hw_efficient_ansatz, H0, nq, 2, 16, psi0, n_tries=R['conv'])

    fig, axes = plt.subplots(1, 3, figsize=(style.FULL_WIDTH, 2.3))
    for ax, metric, ylabel, panel, ref, reflabel in [
            (axes[0], 'E', 'Final energy', '(a)', E_q0_0, f'exact: {E_q0_0:.2f}'),
            (axes[1], 'fidelity', 'Fidelity', '(b)', 1.0, 'perfect: 1.0')]:
        gv = [o[metric] for o in stability_data['GI']]; hv = [o[metric] for o in stability_data['HW']]
        vp = ax.violinplot([gv, hv], positions=[1, 2], showmeans=True, showmedians=True)
        vp['bodies'][0].set_facecolor('#1f77b4'); vp['bodies'][0].set_alpha(0.7)
        vp['bodies'][1].set_facecolor('#d62728'); vp['bodies'][1].set_alpha(0.7)
        ax.axhline(ref, color='k', ls='--', lw=1.2, label=reflabel)
        ax.set_xticks([1, 2]); ax.set_xticklabels(['GI', 'HW'])
        ax.set_ylabel(ylabel); ax.legend(fontsize=8); style.panel_label(ax, panel)
    axc = axes[2]
    axc.plot(range(len(tr_gi)), tr_gi, 'b-', lw=1.2, label='GI (14 params)')
    axc.plot(range(len(tr_hw)), tr_hw, 'r-', lw=1.2, label='HW (16 params)')
    axc.axhline(E_q0_0, color='k', ls='--', lw=1.2, label=f'exact: {E_q0_0:.2f}')
    band = abs(E_q0_0) * 0.01
    axc.fill_between(range(len(tr_gi)), E_q0_0 - band, E_q0_0 + band, alpha=0.15, color='green', label='1% band')
    axc.set_xlabel('Iteration'); axc.set_ylabel('VQE energy')
    axc.legend(fontsize=8); style.panel_label(axc, '(c)')
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig5_stability_convergence.png")
    plt.savefig(out_path)
    # Diagnostic (Round 8): the restart count that was in panels (a)/(b)
    # titles now goes to the caption.
    print(f"Saved {out_path}")
    print(f"  stability panels: {R['stability']} restarts per ansatz at K=0, N=2, L=2")
    print(f"  exact GS E_q0(K=0) = {E_q0_0:.4f}")


if __name__ == "__main__":
    main()
