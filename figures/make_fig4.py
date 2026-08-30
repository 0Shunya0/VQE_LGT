"""
Regenerate fig4 (charge-penalty sweep for the HW ansatz at K=-14: energy,
leakage, fidelity vs lambda) from schwinger_vqe_paper_final.ipynb's Figure 4
cell.

Re-runs run_vqe (COBYLA, seed=42) with a lambda<Q_tot^2> penalty across the
lambda grid. Deterministic given the scipy/numpy versions; if the regenerated
PNG does not match the committed one, the local scipy COBYLA differs from the
one the committed figure was made with.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

from schwinger.core import (build_H_full, build_operators, make_psi0, run_vqe,
                            measure_state, gauge_invariant_ansatz, hw_efficient_ansatz)

FIG_DIR = "final_figs"
X = 16.0
R = dict(ksweep=8, penalty=10, stability=20, scaling=6, conv=5, maxiter=2500)

plt.rcParams.update({"font.family": "serif", "font.size": 9, "figure.dpi": 120,
                     "savefig.bbox": "tight", "savefig.dpi": 150})


def _panel(ax, label):
    """Bare (a)/(b)/... panel label, top-left, just above the axes frame.
    Descriptive text lives in the manuscript caption, not on the figure."""
    ax.text(0.0, 1.02, label, transform=ax.transAxes, va="bottom", ha="left",
            fontweight="bold", fontsize=10)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    N, F, nq = 2, 2, 4
    H_k14 = build_H_full(X, -14.0, N, F)
    Q_tot, Q2, N0_op, N1_op = build_operators(N, F)
    bs = [s for s in range(2 ** nq) if bin(s).count("1") == nq // 2]
    wv, vv = eigh(H_k14[np.ix_(bs, bs)]); E_q0_k14 = float(wv[0])
    psi_q0_k14 = np.zeros(2 ** nq, dtype=complex)
    for i, s in enumerate(bs):
        psi_q0_k14[s] = vv[i, 0]
    psi0 = make_psi0(N, F)
    print(f"Penalty sweep at K=-14: physical GS {E_q0_k14:.3f}, "
          f"global {float(np.real(np.linalg.eigvalsh(H_k14)[0])):.3f}")
    lambda_vals = [0, 0.5, 1, 2, 5, 10, 20, 50, 100]
    pen = []
    for lam in lambda_vals:
        E, psi = run_vqe(hw_efficient_ansatz, H_k14, nq, 2, 16, psi0,
                         n_restarts=R['penalty'], penalty_op=Q2, penalty_lam=lam)
        o = measure_state(psi, Q_tot, Q2, N0_op, N1_op, psi_q0_k14)
        pen.append(dict(lam=lam, E=E, **o))
    E_gi_ref, psi_gi_ref = run_vqe(gauge_invariant_ansatz, H_k14, nq, 2, 14, psi0, n_restarts=R['penalty'])
    o_gi = measure_state(psi_gi_ref, Q_tot, Q2, N0_op, N1_op, psi_q0_k14)

    # Diagnostic dump (Round 8): the plotted arrays, so a data-unchanged check
    # does not depend on pixels. The "crossover" is the smallest lambda at
    # which the HW+penalty state is back in the physical sector (fidelity
    # >= 0.99, equivalently <Q_tot^2> collapsed to ~0).
    print("  lambda   E          <Q_tot^2>   fidelity")
    for r in pen:
        print(f"  {r['lam']:>6}  {r['E']:>10.4f}  {r['Q_tot2']:>9.4f}  {r['fidelity']:>9.4f}")
    print(f"  GI reference fidelity = {o_gi['fidelity']:.4f}")
    crossover = next((r['lam'] for r in pen if r['fidelity'] >= 0.99), None)
    print(f"  penalty crossover (first lambda with fidelity >= 0.99): {crossover}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5)); lp = [max(l, 0.3) for l in lambda_vals]
    ax = axes[0]
    ax.semilogx(lp, [r['E'] for r in pen], 'r-o', lw=2, ms=7, markeredgecolor='k', label='HW + penalty')
    ax.axhline(E_q0_k14, color='blue', ls='--', lw=2, label=f'Physical GS: {E_q0_k14:.0f}')
    ax.axhline(float(np.real(np.linalg.eigvalsh(H_k14)[0])), color='gray', ls=':', lw=1.5, label='Global min')
    ax.set_xlabel(r'$\lambda$'); ax.set_ylabel('Energy'); ax.legend(fontsize=8); _panel(ax, '(a)')
    ax = axes[1]
    ax.semilogx(lp, [r['Q_tot2'] for r in pen], 'r-o', lw=2, ms=7, markeredgecolor='k', label='HW + penalty')
    ax.axhline(0, color='blue', ls='--', lw=2, label='GI: 0 (exact)')
    ax.set_xlabel(r'$\lambda$'); ax.set_ylabel(r'$\langle Q_{tot}^2\rangle$'); ax.legend(fontsize=9); _panel(ax, '(b)')
    ax = axes[2]
    ax.semilogx(lp, [r['fidelity'] for r in pen], 'r-o', lw=2, ms=7, markeredgecolor='k', label='HW + penalty')
    ax.axhline(o_gi['fidelity'], color='blue', ls='--', lw=2, label=f'GI: {o_gi["fidelity"]:.3f}')
    ax.set_xlabel(r'$\lambda$'); ax.set_ylabel('Fidelity'); ax.legend(fontsize=9); _panel(ax, '(c)')
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig4_penalty.png")
    plt.savefig(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
