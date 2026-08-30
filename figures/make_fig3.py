"""
Regenerate fig3 (GI vs HW across a K-sweep: energy, charge leakage, fidelity,
flavor numbers, error) from schwinger_vqe_paper_final.ipynb's Figure 3 cell.

Re-runs run_vqe (COBYLA, seed=42) at 33 K points for both ansatze -- a few
minutes. Deterministic given the scipy/numpy versions; if the regenerated PNG
does not match the committed one, the local scipy COBYLA differs from the one
the committed figure was made with.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import time

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

import schwinger.core as core
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
    N, F, nq, L = 2, 2, 4, 2
    psi0 = make_psi0(N, F)
    K_vals = np.linspace(-16, 16, 33)
    sweep = {k: [] for k in ['E_q0', 'E_glob', 'E_gi', 'E_hw', 'qt2_gi', 'qt2_hw',
                             'n0_gi', 'n0_hw', 'n1_gi', 'n1_hw', 'fid_gi', 'fid_hw']}
    print("Running K-sweep (N=2, L=2, 33 points)...")
    t0 = time.time()
    for ik, K in enumerate(K_vals):
        H = build_H_full(X, float(K), N, F)
        Q_tot, Q2, N0_op, N1_op = build_operators(N, F)
        bs = [s for s in range(2 ** nq) if bin(s).count("1") == nq // 2]
        w, v = eigh(H[np.ix_(bs, bs)]); E_q0 = float(w[0])
        psi_q0 = np.zeros(2 ** nq, dtype=complex)
        for i, s in enumerate(bs):
            psi_q0[s] = v[i, 0]
        E_glob = float(np.real(np.linalg.eigvalsh(H)[0]))
        Ev_gi, psi_gi = run_vqe(gauge_invariant_ansatz, H, nq, L, L * ((nq - 1) + nq), psi0, n_restarts=R['ksweep'])
        Ev_hw, psi_hw = run_vqe(hw_efficient_ansatz, H, nq, L, L * 2 * nq, psi0, n_restarts=R['ksweep'])
        og = measure_state(psi_gi, Q_tot, Q2, N0_op, N1_op, psi_q0)
        oh = measure_state(psi_hw, Q_tot, Q2, N0_op, N1_op, psi_q0)
        sweep['E_q0'].append(E_q0); sweep['E_glob'].append(E_glob)
        sweep['E_gi'].append(Ev_gi); sweep['E_hw'].append(Ev_hw)
        sweep['qt2_gi'].append(og['Q_tot2']); sweep['qt2_hw'].append(oh['Q_tot2'])
        sweep['n0_gi'].append(og['N0']); sweep['n0_hw'].append(oh['N0'])
        sweep['n1_gi'].append(og['N1']); sweep['n1_hw'].append(oh['N1'])
        sweep['fid_gi'].append(og['fidelity']); sweep['fid_hw'].append(oh['fidelity'])
    print(f"  done in {time.time() - t0:.0f}s")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10)); K = K_vals
    ax = axes[0, 0]
    ax.plot(K, sweep['E_q0'], 'k-', lw=2.5, label='Exact (Q=0)', zorder=3)
    ax.plot(K, sweep['E_glob'], 'k:', lw=1.5, alpha=0.4, label='Global min')
    ax.plot(K, sweep['E_gi'], 'b--', lw=2, label='Gauge-invariant')
    ax.plot(K, sweep['E_hw'], 'r--', lw=2, label='HW-efficient')
    ax.set_xlabel('K'); ax.set_ylabel('Energy'); ax.legend(fontsize=8, loc='lower left')
    _panel(ax, '(a)')
    ax = axes[0, 1]
    ax.plot(K, sweep['qt2_gi'], 'b-', lw=2.5, label='GI'); ax.plot(K, sweep['qt2_hw'], 'r-', lw=2.5, label='HW')
    ax.axhline(0, color='gray', ls=':', lw=1); ax.set_xlabel('K'); ax.set_ylabel(r'$\langle Q_{tot}^2\rangle$')
    ax.legend(); _panel(ax, '(b)')
    ax = axes[0, 2]
    ax.plot(K, sweep['fid_gi'], 'b-', lw=2.5, label='GI'); ax.plot(K, sweep['fid_hw'], 'r-', lw=2.5, label='HW')
    ax.axhline(1.0, color='gray', ls=':', lw=1); ax.set_xlabel('K'); ax.set_ylabel('Fidelity')
    ax.legend(); _panel(ax, '(c)')
    ax = axes[1, 0]
    ax.plot(K, sweep['n0_gi'], 'b-', lw=2, label='GI'); ax.plot(K, sweep['n0_hw'], 'r--', lw=2, label='HW')
    ax.set_xlabel('K'); ax.set_ylabel(r'$\langle N_0\rangle$'); ax.legend(); _panel(ax, '(d)')
    ax = axes[1, 1]
    ax.plot(K, sweep['n1_gi'], 'b-', lw=2, label='GI'); ax.plot(K, sweep['n1_hw'], 'r--', lw=2, label='HW')
    ax.set_xlabel('K'); ax.set_ylabel(r'$\langle N_1\rangle$'); ax.legend(); _panel(ax, '(e)')
    ax = axes[1, 2]
    eg = [abs(e - ex) / abs(ex) * 100 if abs(ex) > 0.1 else abs(e - ex) for e, ex in zip(sweep['E_gi'], sweep['E_q0'])]
    eh = [abs(e - ex) / abs(ex) * 100 if abs(ex) > 0.1 else abs(e - ex) for e, ex in zip(sweep['E_hw'], sweep['E_q0'])]
    ax.semilogy(K, [max(e, 1e-3) for e in eg], 'b-', lw=2.5, label='GI')
    ax.semilogy(K, [max(e, 1e-3) for e in eh], 'r-', lw=2.5, label='HW')
    ax.axhline(1.0, color='gray', ls='--', lw=1.5); ax.set_xlabel('K'); ax.set_ylabel('Energy error (%)')
    ax.legend(); _panel(ax, '(f)')
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig3_ksweep.png")
    plt.savefig(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
