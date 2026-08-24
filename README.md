# Disentangling Expressibility, Symmetry Protection, and Hardware Noise in Variational Quantum Simulation of the Two-Flavor Schwinger Model

Simulation code for the paper of the same title (arXiv:XXXX.XXXXX). It
implements the two-flavor Schwinger model on a Jordan-Wigner-mapped qubit
lattice, a gauge-invariant (charge-conserving) VQE ansatz alongside a
hardware-efficient baseline, and a train-noisy VQE pipeline — COBYLA
optimizing directly against an actual noisy density-matrix circuit
(depolarizing two-qubit noise + SPAM readout error, via qiskit-aer) rather
than a post-hoc analytic noise model — used to separate three effects that
are usually conflated in NISQ-era VQE studies: how much of a result's
distortion comes from ansatz expressibility, how much from symmetry
protection (or its absence), and how much is attributable to hardware noise
itself. No `.tex` source is tracked in this repository; the figure/table
map below refers to the manuscript's own figure/table numbers.

## Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.13, tested on Windows. `qiskit-aer`'s `AerSimulator(method="density_matrix")`
is the noisy-circuit backend for every train-noisy result in this repo.

## Repository layout

```
schwinger/      core simulation library (see below)
experiments/    sweep scripts that produce results/*.csv (exp01-exp14, plus
                two helpers: seed_hysteresis_ascending.py, _par_test_worker.py)
analysis/       post-processing: make_summary.py (-> results/SUMMARY.md) and
                the analyze_*.py / check_boundary.py scripts it draws on
figures/        figure-generating scripts (see the figure map below)
results/        every experiment's output: *.csv + *_checkpoint.json (raw
                data and provenance -- never regenerate by deleting these)
final_figs/     generated PNGs
notebooks/      the exploratory/pre-refactor Jupyter notebooks this codebase
                was extracted from, and their superseded figure output
                (notebooks/legacy_figs/) -- kept for provenance, not part of
                the reproducible pipeline
tests/          pytest suite
```

`schwinger/` is two library files kept as single modules on purpose
(`core.py` = `schwinger_core.py`, `backend.py` = `qiskit_backend.py`, both
just moved and import-path-fixed, not internally re-split) plus the
worker/driver files that depend on them:

| file | contents |
|---|---|
| `schwinger/core.py` | Hamiltonian construction, exact diagonalization, GI/HW ansatz (statevector form), observables, noiseless COBYLA drivers, post-hoc noise/ZNE, `boundary_K`, shot-noise estimator |
| `schwinger/backend.py` | Qiskit circuit builders, noise-model setup, ZNE circuit folding, `NoisyEvaluator` (the train-noisy backend) |
| `schwinger/parallel_worker.py` | process-pool cell worker + BLAS/OpenMP thread pinning used by every `experiments/exp*.py` sweep |
| `schwinger/warmstart_worker.py` | single-restart worker for warm-start continuation sweeps |
| `schwinger/kink_warmstart_core.py` | bidirectional (ascending/descending) warm-start continuation driver |
| `schwinger/train_noisy_vqe.py` | train-noisy VQE driver and the validation gate (see Tests) |

## Reproducing the experiments

Every sweep script checkpoints after each completed cell
(`results/<name>_checkpoint.json`, tracking which `(K,p[,L])` cells are
already done) and resumes automatically if interrupted — re-running the same
command after a crash or a Ctrl-C picks up where it left off rather than
restarting. Run all commands from the repository root.

| script | produces | run |
|---|---|---|
| `experiments/exp01_inloop_N3.py` | `results/noisy_inloop_N3.csv` | `python experiments/exp01_inloop_N3.py` — 20 workers, ~20-25 min |
| `experiments/exp02_inloop_zne_N3.py` | `results/inloop_zne_N3.csv`, `_mitigated.csv` | `python experiments/exp02_inloop_zne_N3.py` — 20 workers, 3x exp01's cell count |
| `experiments/exp03_N4_spot.py` | `results/noisy_N4_spot.csv`, `_traces.json` | `python experiments/exp03_N4_spot.py` — 6 workers, few minutes |
| `experiments/exp04_gradvar.py` | `results/gradvar_hw.csv` | `python experiments/exp04_gradvar.py` — serial, < 1 min |
| `experiments/exp05_kink_dense_L2.py` | `results/kink_dense_N3.csv` | `python experiments/exp05_kink_dense_L2.py` — 13 workers |
| `experiments/exp06_boundary_restart_scaling.py` | `results/boundary_restart_scaling_N3.csv` | `python experiments/exp06_boundary_restart_scaling.py` — 3 workers, 32 restarts/cell, maxiter=4000 |
| `experiments/exp07_noiseless_N4_control.py` | `results/noiseless_N4_control.csv` | `python experiments/exp07_noiseless_N4_control.py` — 5 workers |
| `experiments/exp08_kink_dense_L3_noiseless.py` / `_noisy.py` | `results/kink_dense_N3_L3_noiseless.csv` / `_noisy.csv` | `python experiments/exp08_kink_dense_L3_noiseless.py` (then `_noisy.py`) — 13 workers each |
| `experiments/exp09_kink_hysteresis_L3_noiseless.py` / `_noisy.py` | `results/kink_hysteresis_N3_L3_noiseless.csv` / `_noisy.csv` (descending leg; run `experiments/seed_hysteresis_ascending.py` first to seed the ascending leg from exp13/exp14's output) | `python experiments/seed_hysteresis_ascending.py && python experiments/exp09_kink_hysteresis_L3_noiseless.py` — ~15 min observed; `_noisy.py` ~22 min observed |
| `experiments/exp10_kink_dense_L2_noiseless.py` | `results/kink_dense_N3_noiseless.csv` | `python experiments/exp10_kink_dense_L2_noiseless.py` — 13 workers |
| `experiments/exp11_boundary_depth_check.py` | appends to `results/boundary_restart_scaling_N3.csv` (adds an `L` column) | `python experiments/exp11_boundary_depth_check.py` — 2 workers |
| `experiments/exp12_kink_basin_check_L3.py` | appends to `results/kink_dense_N3_L3_{noiseless,noisy}.csv` (adds a `run_tag` column) | `python experiments/exp12_kink_basin_check_L3.py` — 6 workers, 64 restarts/cell, maxiter=4000 |
| `experiments/exp13_kink_warmstart_L3_noiseless.py` / `exp14_..._noisy.py` | `results/kink_warmstart_N3_L3_noiseless.csv` / `_noisy.csv` (also the exp09 ascending leg, via the seed step above) | `python experiments/exp13_kink_warmstart_L3_noiseless.py` — ~15.6 min observed; `exp14` ~22.6 min observed |

`experiments/_par_test_worker.py <seed>` is an ad hoc single-restart timing
probe, not a numbered experiment. Each script's own module docstring states
its exact settings and which round of the audit (see `results/SUMMARY.md`'s
Corrections section) it belongs to.

## Analysis

```bash
python analysis/make_summary.py     # -> results/SUMMARY.md (all headline numbers, with a full
                                     #    Corrections history -- 6 audit rounds, nothing silently replaced)
python analysis/check_boundary.py   # -> results/boundary_check.md (independent boundary_K verification)
```

`analysis/analyze_kink_dense_L3.py`, `analyze_kink_warmstart_L3.py`, and
`analyze_kink_hysteresis_L3.py` are the three successive (Round 4/5/6) kink-
suppression analyses `make_summary.py` draws on; kept as separate scripts
rather than merged, since each is independently runnable and each documents
a distinct stage of the audit.

## Figures

| figure | script | status |
|---|---|---|
| Fig. 6 (phase boundary under noise) | `figures/make_fig6.py` | generates `final_figs/fig6_phase_boundary.png` |
| Figs. 1-5, 7, 8 | *(not yet extracted as standalone scripts)* | PNGs exist in `final_figs/`; the generating code lives in the exploratory notebooks under `notebooks/` and has not been pulled out into `figures/make_fig*.py` — see the `_editbak`-free git history for when this repo was restructured |

## Tests

```bash
python -m pytest                      # fast tests only
python -m pytest -m slow              # includes the validation gate + exact-diagonalization checks (several minutes)
python -m pytest tests/test_reproduce_paper_numbers.py
```

- `test_hamiltonian.py` — `[Q_tot, H] = 0`, sector dimension `C(2N,N)`,
  exact energies vs `REF_E0`, and `boundary_K` (see Bug fix below).
- `test_validation_gate.py` — the in-loop noisy-circuit evaluator at `p=0,
  eps=0` reproduces `N=3,L=2 -> -43.560` and `N=4,L=5 -> -68.562` to `1e-3`.
- `test_reproduce_paper_numbers.py` — parses `results/*.csv` and asserts the
  headline numbers below. One assertion (`retained fraction`) is marked
  `xfail`: it does not correspond to any quantity computed anywhere in this
  codebase or in `results/SUMMARY.md`, and two natural definitions tried
  against the actual data land outside the claimed range — see that test's
  docstring for both attempts. It needs a confirmed formula before it can
  be asserted honestly.

### Bug fix: `boundary_K`

`schwinger.core.boundary_K` located the first-order phase boundary as the
single largest jump in `dE/dK`. At `N=4` the two flavor-0 level crossings
(`K~2.5`, `N_0: 2->3`, and `K~6.4`, `N_0: 3->4`) produce *exactly* equal
slope jumps (`4.000000`, bit-for-bit), so `np.argmax`'s first-match tie-break
silently returned the non-terminal `K~2.5` crossing. Fixed to return the
terminal crossing (where `N_0` reaches its saturating value `N`), located
directly via `<N_0>` level crossings; `boundary_K(N=2,3,4)` now returns
`3.95, 5.60, 6.40` respectively, matching `results/boundary_check.md`'s
independent verification. An `all_crossings=True` flag returns every
crossing found, not just the terminal one. `experiments/exp03_N4_spot.py`'s
existing output (`results/noisy_N4_spot.csv`) was produced against the
*old* `boundary_K(N=4)=2.5` and is not rerun; see that script's docstring
and `results/boundary_check.md` for how to read its K-point labels.

## Results (headline numbers)

From `results/SUMMARY.md`, generated by `analysis/make_summary.py`:

- **In-loop vs post-hoc noisy estimate (N=3)**: mean relative gap in the
  central phase (`|K|<=4`) grows from 5.3% at `p=0.01` to 43.3% at `p=0.05`
  — the post-hoc analytic contraction is not a substitute for training
  against the actual noisy circuit.
- **Kink suppression (Round 6, bidirectional-hysteresis, N=3 L=3 p=0.01)**:
  12.3% raw; -27.9% after subtracting the analytic `(1-w_eff)*dE_mix/dK`
  baseline (the naive post-hoc mixing picture over-predicts the
  post-transition branch's growth; both numbers are reported, see
  `results/SUMMARY.md` §2 for why).
- **In-loop ZNE boundary residuals**: 0.86 / 1.23 / 4.14 at `p=0.01/0.02/0.05`
  (post-hoc reference: 3.8 / 10.5 / 33).
- **N=4 noiseless restart-budget control**: mean 0.24% error at the three
  K-points matched to `noisy_N4_spot.csv` — confirms the noise-driven
  degradation reported there is genuinely attributable to noise, not restart
  budget.
- **Gradient variance** (mean-squared, 20 random points, `L=2`): GI =
  24.15 / 23.02 / 21.19 at N=2/3/4; HW = 10.15 / 5.50 / 2.38.
- **L=2 vs L=3 expressibility near the N=3 boundary** (K=4.5): L=2 stays
  65.2% wrong even at 32 restarts/maxiter=4000 (an expressibility limit, not
  under-convergence); L=3 recovers it to 0.080% at 8 restarts.

`results/SUMMARY.md`'s Corrections section documents six rounds of
audit-driven fixes to these numbers in full, with every superseded or
provisional figure kept and explicitly annotated rather than deleted.
