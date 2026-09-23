"""
Regenerate fig7 (ZNE robustness vs per-CNOT depolarizing strength) from
schwinger_vqe_paper_final.ipynb's Figure 7 cell.

Pure analytic replay: the curve is schwinger.core.noisy() / zne_linear()
evaluated on a p grid at the N=3 boundary point. No VQE, no random restarts,
no seed path -- the committed final_figs/fig7_zne_robustness.png is
reproduced exactly.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os

import numpy as np
import matplotlib.pyplot as plt

import schwinger.core as core
import style

FIG_DIR = "pra_figures"

N_noise, NCX, conv = 3, core.NCX_N3_L2, "nu0only"

style.apply()


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    kb = core.boundary_K(N_noise, convention=conv)
    E0, Emix, _ = core.exact_and_mix(N_noise, kb - 0.05, convention=conv)
    ps = np.linspace(0.001, 0.05, 40)
    no = np.array([core.noisy(E0, Emix, p, NCX) for p in ps])
    zn = np.array([core.zne_linear(E0, Emix, p, NCX) for p in ps])

    fig, ax = plt.subplots(figsize=(style.COL_WIDTH, 2.6))
    ax.axhline(E0 / N_noise, color='k', ls='--', label='exact')
    ax.plot(ps * 100, no / N_noise, 'o-', color='#d62728', ms=2.5, label='noisy')
    ax.plot(ps * 100, zn / N_noise, 's-', color='#2ca02c', ms=2.5, label='ZNE')
    ax.axvline(1.0, color='#1f77b4', ls=':'); ax.axvline(2.0, color='#ff7f0e', ls=':')
    ax.set_xlabel('$p$ (%)'); ax.set_ylabel('$E/N$')
    ax.legend(frameon=False, fontsize=8, loc='lower right',
              handlelength=1.3, handletextpad=0.4, borderaxespad=0.3, labelspacing=0.3)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig7_zne_robustness.png")
    plt.savefig(out_path)

    res = {p: abs(core.zne_linear(E0, Emix, p, NCX) - E0) for p in (0.01, 0.02, 0.05)}
    print(f"Saved {out_path}")
    print(f"ZNE residual (energy units): p=1% -> {res[0.01]:.2f}, "
          f"p=2% -> {res[0.02]:.2f}, p=5% -> {res[0.05]:.2f}")


if __name__ == "__main__":
    main()
