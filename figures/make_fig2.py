"""
Regenerate fig2 (expressibility: error vs depth, p/d collapse, expressibility-
derived hardware cost) from schwinger_vqe_paper_final.ipynb's Figure 2 cell.

N=2,3 points re-run GI-ansatz VQE via schwinger.core.run_vqe_scaling (COBYLA,
seed=42, deterministic given the scipy/numpy versions); N=4,5,6 come from the
notebook's hardcoded `precomputed` table. See figures/README note if the
regenerated PNG does not match the committed one byte-for-byte -- that would
mean the local scipy COBYLA differs from the one the committed figure was
made with.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import math
import os
from math import comb

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import schwinger.core as core
import style

FIG_DIR = "pra_figures"
X = 16.0
R = dict(ksweep=8, penalty=10, stability=20, scaling=6, conv=5, maxiter=2500)

style.apply()


# Regenerated post-fix: best of 24 restarts, seeds 42-45 (4 independent
# 6-restart batches of core.run_vqe_scaling, maxiter=2500); see
# results/precomputed_N456_convergence.json.
PRECOMPUTED = {
    4: dict(E_exact=-68.5764, Q0=70, S_half=0.8042, nq=8,
            layers=[(1, 15, -35.0596, 48.875), (2, 30, -62.8888, 8.294), (3, 45, -66.5798, 2.912), (4, 60, -68.5420, 0.050), (5, 75, -68.5635, 0.019)]),
    5: dict(E_exact=-84.0895, Q0=252, S_half=1.0006, nq=10,
            layers=[(1, 19, -45.7461, 45.598), (2, 38, -78.4617, 6.693)]),
    6: dict(E_exact=-107.303, Q0=924, S_half=1.4056, nq=12,
            layers=[(1, 23, -57.1559, 46.734)]),
}


def exact_data_table():
    exact_data = {}
    for N in range(2, 7):
        nq = 2 * N
        Hsub, basis = core.build_H_subspace(X, 0.0, N=N)
        w, v = eigh(Hsub)
        psi_gs = np.zeros(2 ** nq, dtype=complex)
        for i, s in enumerate(basis):
            psi_gs[s] = v[i, 0]
        S = core.entanglement_entropy(psi_gs, list(range(nq // 2)), nq)
        exact_data[N] = dict(E_exact=float(w[0]), S_half=S, Q0=len(basis), nq=nq)
    return exact_data


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    exact_data = exact_data_table()

    all_data = {}
    for N in (2, 3):
        nq = N * 2; n_per = (nq - 1) + nq
        rows = []
        for L in (1, 2, 3):
            E = core.run_vqe_scaling(N, L, n_restarts=R['scaling'], maxiter=max(R['maxiter'], 1500))
            rows.append((L, L * n_per, E, abs(E - exact_data[N]['E_exact']) / abs(exact_data[N]['E_exact']) * 100))
        all_data[N] = dict(E_exact=exact_data[N]['E_exact'], Q0=exact_data[N]['Q0'],
                           S_half=exact_data[N]['S_half'], nq=nq, layers=rows)
    for N in (4, 5, 6):
        all_data[N] = PRECOMPUTED[N]

    fig = plt.figure(figsize=(style.FULL_WIDTH, 2.3))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.55)
    ax1, ax2, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    colors = {2: '#1f77b4', 3: '#d62728', 4: '#2ca02c', 5: '#9467bd', 6: '#8c564b'}
    markers = {2: 'o', 3: 's', 4: '^', 5: 'D', 6: 'P'}
    for N, d in all_data.items():
        nq = d['nq']; n_per = (nq - 1) + nq
        Lv = [t[0] for t in d['layers']]
        ev = [max(t[3], 5e-3) for t in d['layers']]
        pv = [t[1] for t in d['layers']]
        ax1.semilogy(Lv, ev, color=colors[N], marker=markers[N], lw=1.2, ms=4,
                     markeredgecolor='k', label=f'N={N}')
        Lth = d['Q0'] / n_per
        if Lth <= max(Lv) + 1:
            ax1.axvline(Lth, color=colors[N], ls=':', lw=1.0, alpha=0.6)
        ax2.semilogy([p / d['Q0'] for p in pv], ev, color=colors[N], marker=markers[N],
                     lw=1.2, ms=4, markeredgecolor='k', label=f'N={N}')
    ax1.axhline(1.0, color='gray', ls='--', lw=1.0, label='1% target')
    ax1.set_xlabel('L'); ax1.set_ylabel('Error (%)')
    style.panel_label(ax1, '(a)')
    ax1.set_xticks([1, 2, 3, 4, 5])
    ax2.axvline(1.0, color='k', ls='--', lw=1.2, alpha=0.8, label='$p/d=1$')
    ax2.axhline(1.0, color='gray', ls='--', lw=1.0)
    ax2.set_xlabel('$p/d$'); ax2.set_ylabel('Error (%)')
    style.panel_label(ax2, '(b)')
    ax2.legend(fontsize=8, loc='upper right')
    N_c = [2, 3, 4, 5]
    Lmin = [math.ceil(comb(N * 2, N) / ((N * 2 - 1) + N * 2)) for N in N_c]
    depth = [4 * L + 4 * N - 6 for N, L in zip(N_c, Lmin)]
    bar_col = ['#2ca02c' if L <= 3 else '#ff7f0e' if L <= 10 else '#d62728' for L in Lmin]
    xp = np.arange(len(N_c)); ax3b = ax3.twinx()
    ax3.bar(xp, Lmin, color=bar_col, edgecolor='k', lw=0.7, alpha=0.85)
    ax3b.plot(xp, depth, 'r^-', ms=4, lw=1.2, markeredgecolor='k')
    for xi, L in enumerate(Lmin):
        ax3.text(xi, L - 0.35, str(L), ha='center', va='top', fontsize=7,
                 fontweight='bold', color='white')
    ax3.set_ylim(top=max(Lmin) * 1.3)
    ax3b.set_ylim(top=max(depth) * 1.18)
    ax3.set_xticks(xp); ax3.set_xticklabels([str(N) for N in N_c])
    ax3.set_xlabel('N'); ax3.set_ylabel('$L_{\\mathrm{min}}$', color='k')
    ax3b.set_ylabel('Depth', color='red')
    style.panel_label(ax3, '(c)')
    ax3.axhline(10, color='orange', ls=':', lw=1.2, alpha=0.8, label='ion limit')
    ax3.legend(fontsize=8, loc='upper left')
    out_path = os.path.join(FIG_DIR, "fig2_expressibility.png")
    plt.savefig(out_path)
    print(f"Saved {out_path}")
    print(f"  panel (c): N={N_c}  L_min={Lmin}  2q depth={depth}")


if __name__ == "__main__":
    main()
