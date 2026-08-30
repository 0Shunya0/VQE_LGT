"""
Regenerate fig1 (baseline: Hilbert-space growth, ground-state entanglement,
finite-size energy per site) from schwinger_vqe_paper_final.ipynb's Figure 1
cell.

Pure exact-diagonalization replay: every quantity plotted comes from
schwinger.core.build_H_subspace + eigh + entanglement_entropy at K=0, plus
the L-layer parameter-budget arithmetic. No VQE, no random restarts, no seed
path -- the committed final_figs/fig1_baseline.png is reproduced exactly.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

import schwinger.core as core

FIG_DIR = "final_figs"
X = 16.0

plt.rcParams.update({"font.family": "serif", "font.size": 9, "figure.dpi": 120,
                     "savefig.bbox": "tight", "savefig.dpi": 150})


def _panel(ax, label):
    """Bare (a)/(b)/... panel label, top-left, just above the axes frame.
    Descriptive text lives in the manuscript caption, not on the figure."""
    ax.text(0.0, 1.02, label, transform=ax.transAxes, va="bottom", ha="left",
            fontweight="bold", fontsize=10)


def exact_data_table():
    """Table I quantities for N = 2..6 (K=0, x=16), same construction as the
    notebook's 'Exact ground states' cell."""
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    N_vals = list(exact_data.keys())
    Q0_dims = [exact_data[N]['Q0'] for N in N_vals]
    S_vals = [exact_data[N]['S_half'] for N in N_vals]
    EpN = [exact_data[N]['E_exact'] / N for N in N_vals]

    ax = axes[0]
    ax.bar(N_vals, Q0_dims, color='steelblue', edgecolor='k', lw=0.7, alpha=0.85)
    for N, d in zip(N_vals, Q0_dims):
        ax.text(N, d + 5, str(d), ha='center', fontsize=11, fontweight='bold')
    for L, col in [(1, 'red'), (2, 'orange'), (3, 'green')]:
        for iN, N in enumerate(N_vals):
            nq = N * 2; p = L * (nq - 1 + nq)
            ax.plot([N - 0.4, N + 0.4], [p, p], '-', color=col, lw=2, alpha=0.8,
                    label=f'L={L} params' if iN == 0 else '')
    ax.set_xlabel('Lattice sites N'); ax.set_ylabel('Q_tot=0 sector dimension')
    _panel(ax, '(a)')
    ax.legend(fontsize=8, loc='upper left'); ax.set_xticks(N_vals)

    ax = axes[1]
    ax.plot(N_vals, S_vals, 'purple', marker='s', lw=2.5, ms=10, markeredgecolor='k')
    for N, S in zip(N_vals, S_vals):
        ax.text(N + 0.07, S + 0.01, f'{S:.3f}', fontsize=9)
    ax.axhline(np.log(2), color='gray', ls='--', lw=1.5, label='ln2 = 0.693')
    ax.axhline(np.log(4), color='gray', ls=':', lw=1.5, label='ln4 = 1.386')
    ax.set_xlabel('Lattice sites N'); ax.set_ylabel('Half-system entropy S (nats)')
    _panel(ax, '(b)')
    ax.legend(fontsize=8); ax.set_xticks(N_vals)

    ax = axes[2]
    ax.plot(N_vals, EpN, 'ko-', lw=2.5, ms=10, markeredgecolor='k')
    for N, E in zip(N_vals, EpN):
        ax.text(N + 0.07, E - 0.3, f'{E:.2f}', fontsize=9)
    ax.set_xlabel('Lattice sites N'); ax.set_ylabel('E_exact / N')
    _panel(ax, '(c)')
    ax.set_xticks(N_vals)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig1_baseline.png")
    plt.savefig(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
