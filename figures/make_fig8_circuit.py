"""
Fig. 8: N=3, L=2 gauge-invariant ansatz circuit, logical vs heavy-hex-
transpiled.

Reuses build_gi_circuit from schwinger/backend.py unmodified (same builder
used for the noisy-VQE production runs and for the Task-A transpilation
study in _ibm_transpile_study.py). No ansatz code, parameters, or data are
changed here -- this script only draws the circuit.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os

import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

from schwinger.backend import build_gi_circuit
import style

FIG_DIR = "final_figures_pra"
N, L = 3, 2
NQ = 2 * N
SEED = 0
OPT_LEVEL = 3

style.apply()


def _render_panel(qc, tmp_path, fold=40):
    """Draw one circuit to its own tightly-cropped PNG (display copy only --
    global_phase zeroed just to drop the drawer's banner line; the returned
    transpiled circuit used for the printed gate/depth counts is untouched)."""
    qc_draw = qc.copy()
    qc_draw.global_phase = 0
    fig = qc_draw.draw(output='mpl', fold=fold, style={'fontsize': 8})
    fig.savefig(tmp_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    return plt.imread(tmp_path)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    qc, params, n_cx = build_gi_circuit(NQ, L)
    assert n_cx == 20

    backend = FakeSherbrooke()
    tqc = transpile(qc, backend=backend, optimization_level=OPT_LEVEL, seed_transpiler=SEED)

    img_a = _render_panel(qc, os.path.join(FIG_DIR, "_tmp_fig8_a.png"))
    img_b = _render_panel(tqc, os.path.join(FIG_DIR, "_tmp_fig8_b.png"))

    h_a = style.FULL_WIDTH * img_a.shape[0] / img_a.shape[1]
    h_b = style.FULL_WIDTH * img_b.shape[0] / img_b.shape[1]

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(style.FULL_WIDTH, h_a + h_b + 0.3),
        gridspec_kw={'height_ratios': [h_a, h_b], 'hspace': 0.15},
    )
    for ax, img in ((ax_a, img_a), (ax_b, img_b)):
        ax.imshow(img)
        ax.axis('off')
    style.panel_label(ax_a, '(a)', y=0.98)
    style.panel_label(ax_b, '(b)', y=0.98)

    out_path = os.path.join(FIG_DIR, "fig8_circuit.png")
    plt.savefig(out_path)
    for tmp in ("_tmp_fig8_a.png", "_tmp_fig8_b.png"):
        os.remove(os.path.join(FIG_DIR, tmp))
    print(f"Saved {out_path}")
    print(f"(a) logical: {qc.size()} instructions, depth {qc.depth()}, {n_cx} CX")
    ops = tqc.count_ops()
    n_2q = ops.get('ecr', 0) + ops.get('cx', 0)
    print(f"(b) transpiled: {tqc.size()} instructions, depth {tqc.depth()}, {n_2q} ECR/CX, seed={SEED}")


if __name__ == "__main__":
    main()
