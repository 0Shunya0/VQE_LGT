"""
Regenerate fig6 (phase boundary under noise) with the in-loop noisy-VQE curve
added alongside the existing exact / post-hoc-noisy / ZNE curves. Same style
as schwinger_vqe_paper_final.ipynb's Figure 6 cell: two panels, (a)/(b) panel
labels only, no per-panel titles beyond that, no suptitle.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import schwinger_core as core

RESULTS_DIR = "results"
FIG_DIR = "final_figs"
P_SHOW = 0.01  # representative p, matches the post-hoc figure's default

N_noise, NCX, conv = 3, core.NCX_N3_L2, "nu0only"


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # exact / post-hoc-noisy / ZNE curves, same construction as the original cell
    Kv = np.linspace(-8, 8, 81)
    ex, mx, S = np.array([core.exact_and_mix(N_noise, k, convention=conv) for k in Kv]).T
    no = core.noisy(ex, mx, P_SHOW, NCX)
    zn = core.zne_linear(ex, mx, P_SHOW, NCX)
    kb = core.boundary_K(N_noise, convention=conv)

    # in-loop noisy VQE curve from Experiment 1 (best-of-8 per K), restricted
    # to this panel's K-range and to p=P_SHOW
    inloop_path = os.path.join(RESULTS_DIR, "noisy_inloop_N3.csv")
    have_inloop = os.path.exists(inloop_path)
    if have_inloop:
        df = pd.read_csv(inloop_path)
        best = df[df["is_best"] & (df["p"] == P_SHOW) & (df["K"].abs() <= 8.0)].sort_values("K")
        K_il, E_il = best["K"].to_numpy(), best["energy"].to_numpy()

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.2))
    ax[0].plot(Kv, ex / N_noise, 'k--', label='exact')
    ax[0].plot(Kv, no / N_noise, 'o-', color='#d62728', ms=3,
               label=f'post-hoc noisy ({100 * (1 - (1 - P_SHOW) ** NCX):.1f}%)')
    ax[0].plot(Kv, zn / N_noise, 's-', color='#2ca02c', ms=3, label='post-hoc ZNE')
    if have_inloop:
        ax[0].plot(K_il, E_il / N_noise, '^-', color='#1f77b4', ms=5, lw=1.5,
                   label=f'in-loop noisy (p={P_SHOW})')
    ax[0].set_xlabel('$K$'); ax[0].set_ylabel(r'$E_0/N$ ($N=3$)'); ax[0].set_title('(a) energy under noise')
    ax[0].legend(frameon=False, fontsize=7)

    de, dn, dz = np.gradient(ex, Kv), np.gradient(no, Kv), np.gradient(zn, Kv)
    ax[1].plot(Kv, de, 'k--', label='exact')
    ax[1].plot(Kv, dn, 'o-', color='#d62728', ms=3, label='post-hoc noisy')
    ax[1].plot(Kv, dz, 's-', color='#2ca02c', ms=3, label='post-hoc ZNE')
    if have_inloop and len(K_il) > 2:
        d_il = np.gradient(E_il, K_il)
        ax[1].plot(K_il, d_il, '^-', color='#1f77b4', ms=5, lw=1.5, label='in-loop noisy')
    ax[1].axvline(kb, color='gray', ls=':'); ax[1].axvline(-kb, color='gray', ls=':')
    ax[1].set_xlabel('$K$'); ax[1].set_ylabel('$dE/dK$'); ax[1].set_title(f'(b) kink at |K|~{kb:.2f}')
    a2 = ax[1].twinx(); a2.plot(Kv, S, ':', color='#6a3d9a', lw=1.2)
    a2.set_ylabel('$S$ (nats)', color='#6a3d9a'); a2.tick_params(axis='y', colors='#6a3d9a')
    ax[1].legend(frameon=False, loc='upper left', fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig6_phase_boundary.png")
    plt.savefig(out_path)
    print(f"Saved {out_path} (in-loop curve {'included' if have_inloop else 'MISSING -- run Experiment 1 first'})")


if __name__ == "__main__":
    main()
