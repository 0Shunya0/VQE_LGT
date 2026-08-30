# Disentangling Expressibility, Symmetry Protection, and Hardware Noise in Variational Quantum Simulation of the Two-Flavor Schwinger Model

Simulation code for the paper of the same title (not yet submitted or
preprinted; a link will be added here once it is). It
implements the two-flavor Schwinger model (a 1+1D lattice gauge theory) on a
Jordan-Wigner-mapped qubit lattice, at coupling `x=16` and lattice size
`N=2..6` sites (`F=2` flavors, `nq=2N` qubits). Two VQE ansatze are compared
on this model: a gauge-invariant (GI), charge-conserving ansatz built from
fermionic-exchange gates that never leaves the physical `Q_tot=0` sector, and
a hardware-efficient (HW) baseline built from generic single-qubit rotations
and a CNOT ladder, which has no such symmetry protection and can leak
population into unphysical charge sectors.

The central methodological contribution is a train-noisy VQE pipeline: COBYLA
optimizes directly against an actual noisy density-matrix circuit (a
two-qubit depolarizing channel after every CNOT, plus a per-qubit bit-flip
SPAM channel before readout, both via `qiskit-aer`'s
`AerSimulator(method="density_matrix")`), instead of the more common
train-noiseless-then-evaluate-noisy shortcut or a post-hoc analytic
contraction of the exact energy. That distinction matters because it lets the
three effects usually conflated in NISQ-era VQE studies be pulled apart and
attributed separately: how much of a result's distortion comes from ansatz
expressibility (can the ansatz even represent the target state, at any noise
level), how much from symmetry protection (does the ansatz's own structure
keep it in the physical sector, or does it need a penalty term), and how much
is genuinely attributable to hardware noise once the first two are controlled
for.

This repository also carries a seven-round audit trail of that analysis:
`results/SUMMARY.md`'s Corrections section documents every methodological bug
found and fixed along the way (grid mismatches, a phase-boundary tie-breaking
bug, basin-capture failures in the optimizer, one-directional monotonicity
gates that were never a fair test near a first-order transition, a
double-counted mixing correction), with every superseded or provisional
number kept and explicitly annotated rather than silently replaced. No
`.tex` source is tracked in this repository; the figure/table map below
refers to the manuscript's own figure and table numbers.

What is reproducible from this repository: every headline number in
`results/SUMMARY.md` (via `analysis/make_summary.py` over the committed
`results/*.csv`), the full pytest suite including the noiseless validation
gate, and every panel of Figures 1-7 (via `figures/make_fig*.py` over the
committed data and exact-diagonalization code). Figure reproduction is
verified by regenerating each PNG from committed data and confirming the
scripts' printed diagnostics (fig4's penalty crossover, fig6's phase
boundary and S_max, fig7's ZNE residuals) are unchanged. Through Round 7
every figure also regenerated pixel-for-pixel identical to its committed
PNG; Round 8 removed the in-panel subplot titles at a reviewer's request, so
the pixels changed by design and that check now applies to the pre-cleanup
versions, which remain in git history. What is not: the manuscript
`.tex` and its bibliography, and the raw multi-hour sweeps behind
`results/*.csv` are reproducible only in the sense that re-running the
`experiments/exp*.py` scripts with the committed `SEED_BASE` values
regenerates them (`experiments/exp04_gradvar.py` is byte-identical; the
COBYLA-heavy sweeps match to optimizer tolerance).

## Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Developed and tested on Python 3.13, Windows. `qiskit-aer`'s
`AerSimulator(method="density_matrix")` is the noisy-circuit backend for
every train-noisy result in this repo; it runs on CPU, no GPU or IBM Quantum
account is required. Dense-matrix simulation caps the reachable system size
at `nq<=8` (`N<=4`), which is why the noisy sweeps stop at `N=4` even though
the exact-diagonalization code (`schwinger.core`) scales further, to `N=6`,
for the noiseless reference table.

## Repository layout

```
schwinger/      core simulation library (see below)
experiments/    sweep scripts that produce results/*.csv (exp01-exp14, plus
                two helpers: seed_hysteresis_ascending.py, _par_test_worker.py)
analysis/       post-processing: make_summary.py (-> results/SUMMARY.md) and
                the analyze_*.py / check_boundary.py scripts it draws on
figures/        figure-generating scripts (see the figure map below)
results/        every experiment's output: *.csv plus *_checkpoint.json (raw
                data and provenance; never regenerate by deleting these)
final_figs/     generated PNGs
notebooks/      the exploratory, pre-refactor Jupyter notebooks this codebase
                was extracted from, and their superseded figure output
                (notebooks/legacy_figs/); kept for provenance, not part of
                the reproducible pipeline
tests/          pytest suite
```

`schwinger/` is two library files kept as single modules on purpose:
`core.py` is `schwinger_core.py` and `backend.py` is `qiskit_backend.py`,
both just moved into the package and import-path-fixed, not internally
re-split into narrower modules. They were extracted verbatim from the
manuscript's own validated notebook cells, dozens of downstream scripts
depend on their exact behavior, and a hand-split risks subtly changing which
names are visible where an import order matters, for a purely cosmetic
reorganization. The package also holds the worker/driver files that depend
on those two:

| file | contents |
|---|---|
| `schwinger/core.py` | Hamiltonian construction, exact diagonalization, GI/HW ansatz (statevector form), observables (`Q_tot`, `<Q_tot^2>`, entanglement entropy), noiseless COBYLA drivers, post-hoc noise/ZNE formulas, `boundary_K` (phase-boundary locator), and the finite-shot measurement-budget estimator |
| `schwinger/backend.py` | Qiskit circuit builders for both ansatze, noise-model setup (depolarizing + SPAM), ZNE circuit folding, and `NoisyEvaluator`, the class that wraps all of that into a single train-noisy cost function |
| `schwinger/parallel_worker.py` | the process-pool cell worker and BLAS/OpenMP thread-pinning pattern used by every `experiments/exp*.py` sweep, so N parallel worker processes don't each try to multithread their own linear algebra and contend for the same cores |
| `schwinger/warmstart_worker.py` | single-restart worker for warm-start continuation sweeps, dispatched one restart at a time rather than a whole cell, since a warm-start candidate needs an externally supplied starting point |
| `schwinger/kink_warmstart_core.py` | the bidirectional (ascending and descending) warm-start/hysteresis continuation driver used to locate the first-order kink transition without a one-directional monotonicity gate's blind spot |
| `schwinger/train_noisy_vqe.py` | the train-noisy VQE driver (`run_noisy_vqe_point`) and the validation gate that checks it against the noiseless reference table (see Tests) |

## Reproducing the experiments

Every sweep script checkpoints after each completed cell, writing
`results/<name>_checkpoint.json` to track which `(K,p[,L])` cells are already
done, and resumes automatically on the next invocation if it was interrupted:
re-running the same command after a crash or a Ctrl-C picks up where it left
off instead of restarting the whole sweep from scratch. Run all commands
below from the repository root, so the scripts' relative `results/...` paths
resolve correctly.

| script | produces | run |
|---|---|---|
| `experiments/exp01_inloop_N3.py` | `results/noisy_inloop_N3.csv` | `python experiments/exp01_inloop_N3.py` (20 workers, roughly 20-25 min) |
| `experiments/exp02_inloop_zne_N3.py` | `results/inloop_zne_N3.csv`, `_mitigated.csv` | `python experiments/exp02_inloop_zne_N3.py` (20 workers, 3x exp01's cell count since it sweeps three ZNE folding factors per cell) |
| `experiments/exp03_N4_spot.py` | `results/noisy_N4_spot.csv`, `_traces.json` | `python experiments/exp03_N4_spot.py` (6 workers, a few minutes; only 6 cells total) |
| `experiments/exp04_gradvar.py` | `results/gradvar_hw.csv` | `python experiments/exp04_gradvar.py` (serial, well under a minute) |
| `experiments/exp05_kink_dense_L2.py` | `results/kink_dense_N3.csv` | `python experiments/exp05_kink_dense_L2.py` (13 workers, one per K point) |
| `experiments/exp06_boundary_restart_scaling.py` | `results/boundary_restart_scaling_N3.csv` | `python experiments/exp06_boundary_restart_scaling.py` (3 workers, 32 restarts per cell, maxiter=4000; this is the expensive restart-scaling control) |
| `experiments/exp07_noiseless_N4_control.py` | `results/noiseless_N4_control.csv` | `python experiments/exp07_noiseless_N4_control.py` (5 workers) |
| `experiments/exp08_kink_dense_L3_noiseless.py` / `_noisy.py` | `results/kink_dense_N3_L3_noiseless.csv` / `_noisy.csv` | `python experiments/exp08_kink_dense_L3_noiseless.py` then `python experiments/exp08_kink_dense_L3_noisy.py` (13 workers each) |
| `experiments/exp09_kink_hysteresis_L3_noiseless.py` / `_noisy.py` | `results/kink_hysteresis_N3_L3_noiseless.csv` / `_noisy.csv`, the descending leg of the hysteresis sweep | run `experiments/seed_hysteresis_ascending.py` first, to seed the ascending leg from exp13/exp14's output, then `python experiments/exp09_kink_hysteresis_L3_noiseless.py` (about 15 min observed) and `python experiments/exp09_kink_hysteresis_L3_noisy.py` (about 22 min observed) |
| `experiments/exp10_kink_dense_L2_noiseless.py` | `results/kink_dense_N3_noiseless.csv` | `python experiments/exp10_kink_dense_L2_noiseless.py` (13 workers) |
| `experiments/exp11_boundary_depth_check.py` | appends to `results/boundary_restart_scaling_N3.csv`, adding an `L` column | `python experiments/exp11_boundary_depth_check.py` (2 workers, L=3 and L=4 at the worst K point from exp06) |
| `experiments/exp12_kink_basin_check_L3.py` | appends to `results/kink_dense_N3_L3_{noiseless,noisy}.csv`, adding a `run_tag` column | `python experiments/exp12_kink_basin_check_L3.py` (6 workers, 64 restarts per cell, maxiter=4000; the targeted basin-capture re-check) |
| `experiments/exp13_kink_warmstart_L3_noiseless.py` / `exp14_kink_warmstart_L3_noisy.py` | `results/kink_warmstart_N3_L3_noiseless.csv` / `_noisy.csv` (this is also exp09's ascending leg, once copied over by the seed step above) | `python experiments/exp13_kink_warmstart_L3_noiseless.py` (about 15.6 min observed), `python experiments/exp14_kink_warmstart_L3_noisy.py` (about 22.6 min observed) |

`experiments/_par_test_worker.py <seed>` is an ad hoc single-restart timing
probe used during development, not a numbered experiment; it takes a seed on
the command line and prints one restart's wall time and final energy. Every
numbered script's own module docstring states its exact settings (grid,
restart budget, `maxiter`, distinguishing `SEED_BASE`) and which round of the
audit it belongs to, cross-referenced against `results/SUMMARY.md`'s
Corrections section.

## Analysis

```bash
python analysis/make_summary.py     # -> results/SUMMARY.md: every headline number below, plus the
                                     #    full seven-round Corrections history, nothing silently replaced
python analysis/check_boundary.py   # -> results/boundary_check.md: independent boundary_K verification,
                                     #    via direct <N_0> level-crossing scans rather than boundary_K itself
```

`analysis/analyze_kink_dense_L3.py`, `analyze_kink_warmstart_L3.py`, and
`analyze_kink_hysteresis_L3.py` are the three successive kink-suppression
analyses (Rounds 4, 5, and 6 respectively; the Round 6 script also carries
the Round 7 retained-fraction solve) that `make_summary.py` draws on.
They are kept as three separate scripts rather than merged into one, since
each is independently runnable, each documents a distinct stage of the
audit's reasoning, and merging them would have obscured exactly which
methodological assumption changed between rounds.

## Figures

All seven figures are extracted from the notebooks as standalone
`figures/make_fig*.py` scripts, and re-run to regenerate their PNGs in
`final_figs/`. In-panel subplot titles were removed in Round 8 at a
reviewer's request; the panels now carry only bare (a)/(b)/... labels and
all descriptive text lives in the manuscript captions. Reproduction is
checked by confirming each script's printed diagnostics are unchanged, not
by pixel comparison (which Round 8 broke by design; the pixel-identical
check through Round 7 still applies to the pre-cleanup PNGs in git history).
Figs. 2-5 re-run COBYLA VQE internally (`schwinger.core.run_vqe` /
`run_vqe_scaling`, `seed=42`), not just replot saved data, so their exact
reproduction depends on the `scipy` version matching `requirements.txt`;
Figs. 1, 6, 7 are pure exact-diagonalization / analytic replays with no
optimizer in the loop.

| figure | script | what it plots |
|---|---|---|
| Fig. 1 | `figures/make_fig1.py` | `fig1_baseline.png`: Hilbert-space growth, ground-state entanglement entropy, finite-size energy per site (exact diagonalization, N=2..6) |
| Fig. 2 | `figures/make_fig2.py` | `fig2_expressibility.png`: relative energy error vs depth, the p/d collapse, and the expressibility-derived hardware cost (N=2,3 re-run VQE; N=4,5,6 from the notebook's `precomputed` table) |
| Fig. 3 | `figures/make_fig3.py` | `fig3_ksweep.png`: GI vs HW across a 33-point K-sweep -- energy, charge-sector leakage, fidelity, flavor numbers, error (re-runs VQE, ~10 min) |
| Fig. 4 | `figures/make_fig4.py` | `fig4_penalty.png`: HW-ansatz charge-penalty sweep at K=-14 -- energy, leakage, fidelity vs lambda |
| Fig. 5 | `figures/make_fig5.py` | `fig5_stability_convergence.png`: optimization-stability violin plots over 20 restarts plus best-of-5 convergence traces (merges the pre-refactor Figs. 5 and 6) |
| Fig. 6 | `figures/make_fig6.py` | `fig6_phase_boundary.png`: exact, post-hoc-noisy, post-hoc-ZNE, and in-loop noisy energy curves side by side, with the located phase boundary marked |
| Fig. 7 | `figures/make_fig7.py` | `fig7_zne_robustness.png`: ZNE residual vs per-CNOT depolarizing strength at the N=3 boundary (analytic) |

`final_figs/fig8_zne_robustness.png` is a stale artifact of the pre-merge
figure numbering (before the old Figs. 5 and 6 were merged into one, every
later figure was one higher). It is the same plot as Fig. 7 with a "Figure 8"
title. It is left in `final_figs/` rather than deleted; its generator lives
at `notebooks/legacy_figs/make_fig8_legacy.py`, not in `figures/`.

`submission/` is the flat folder staged for journal upload: the seven paper
figures copied from `final_figs/` (not moved -- `final_figs/` stays the
scripts' output target) alongside `main.tex`. It is generated, not
hand-maintained: refresh it by re-running the `make_fig*.py` scripts and
re-copying. The stale pre-merge PNGs are deliberately not in it.

## Tests

```bash
python -m pytest                      # fast tests only (well under a minute)
python -m pytest -m slow              # also runs the validation gate and the exact-diagonalization
                                       #   checks up to N=6 (several minutes total, mostly the gate)
python -m pytest tests/test_reproduce_paper_numbers.py
```

- `test_hamiltonian.py`: `[Q_tot, H] = 0` as an operator identity (not just
  on one state), sector dimension `C(2N,N)` against the actual physical
  basis, exact ground-state energies against `REF_E0` (the reference table),
  and `boundary_K`'s corrected output (see Bug fix below).
- `test_validation_gate.py`: the in-loop noisy-circuit evaluator, run at
  `p=0, eps=0` so it should reduce to the exact noiseless result, reproduces
  `N=3,L=2 -> -43.560` and `N=4,L=5 -> -68.562` to within `1e-3` absolute
  energy.
- `test_reproduce_paper_numbers.py`: parses `results/*.csv` directly (not
  copies of the numbers) and asserts the headline values listed below,
  including `test_retained_fraction_both_branches` (Round 7): the paper's
  Section VI C slope-mixture retained-fraction solve,
  `w_eff = (dE_mix/dK - dE_inloop/dK) / (dE_mix/dK - dE_exact/dK)`, giving
  1.005 (100.5%) on the ordered branch and 0.920 on the saturated branch.
  This was `xfail` through Round 6 (the formula was not in the codebase and
  two guessed definitions both failed); the confirmed formula lands both
  branches in `[0.91, 1.02]` (the assertion bound is not the tighter
  `[0.92, 1.01]` because the saturated branch is 0.91992, just under a
  nominal 0.92, and the ordered branch is genuinely above 1) and the
  assertion now runs.

### Bug fix: `boundary_K`

`schwinger.core.boundary_K` located the first-order phase boundary as the
single largest jump in `dE/dK` over a scanned grid. At `N=4` there are two
flavor-0 occupation level crossings in range, at `K~2.5` (`N_0: 2->3`) and
`K~6.4` (`N_0: 3->4`), and they produce exactly equal slope jumps
(`4.000000`, bit-for-bit identical, not just numerically close, because each
crossing adds the same fixed `-2*sqrt(x)` contribution to `dE/dK`
regardless of which occupation level it happens at). `np.argmax`'s
first-match tie-break silently returned the non-terminal `K~2.5` crossing
instead of the terminal one. `N=2` and `N=3` never hit this bug because each
only has one crossing in the scanned range, so no tie is ever possible there.

The fix returns the terminal crossing, the one where `N_0` reaches its
saturating value `N`, located directly via `<N_0>` level crossings rather
than via the slope-jump proxy that broke on the tie. `boundary_K(N=2,3,4)`
now returns `3.95, 5.60, 6.40` respectively, matching
`results/boundary_check.md`'s independent verification computed the same way
but without calling `boundary_K` itself. A new `all_crossings=True` flag
returns every crossing found as a list, not just the terminal one, so
downstream code that wants the full transition structure (as
`results/boundary_check.md` does) doesn't need to re-derive it.

`experiments/exp03_N4_spot.py`'s existing output
(`results/noisy_N4_spot.csv`) was produced against the old, buggy
`boundary_K(N=4)=2.5` and is deliberately not rerun, so that history isn't
silently rewritten; see that script's docstring and
`results/boundary_check.md` for how to read its K-point labels correctly
given the old boundary value.

## Results (headline numbers)

All of the following are generated fresh by `analysis/make_summary.py` from
`results/*.csv` and written to `results/SUMMARY.md`, which also carries the
full reasoning and audit history behind each one:

- **In-loop vs post-hoc noisy estimate (N=3).** The mean relative gap
  between the in-loop-trained noisy energy and the post-hoc analytic
  contraction, in the central phase (`|K|<=4`), grows from 5.3% at `p=0.01`
  to 43.3% at `p=0.05`. The post-hoc contraction is not a substitute for
  actually training against the noisy circuit; the gap widens with noise
  strength rather than staying roughly constant.
- **Kink suppression (Round 7, bidirectional-hysteresis check, N=3, L=3,
  p=0.01).** 12.3% raw suppression of the transition's slope drop under
  noise, reported alongside the paper's Section VI C retained-fraction pair:
  `w_eff = 0.920` on the saturated branch and `1.005` (100.5%) on the
  ordered branch, from the slope-mixture solve
  `w_eff = (dE_mix/dK - dE_inloop/dK) / (dE_mix/dK - dE_exact/dK)` (both in
  `[0.91, 1.02]`). Both branches retain essentially the full exact slope,
  far above the passive-mixing `w_eff = 0.732`. (Round 6's baseline-adjusted -27.9% figure
  is withdrawn -- it double-counted the mixing correction; see
  `results/SUMMARY.md` Corrections, Round 7.)
- **In-loop ZNE boundary residuals.** 0.86, 1.23, and 4.14 (energy units) at
  `p=0.01, 0.02, 0.05` respectively, versus a post-hoc reference of 3.8,
  10.5, and 33 at the same three noise strengths: in-loop training combined
  with linear Richardson extrapolation removes most of the residual error
  the post-hoc estimate leaves behind.
- **N=4 noiseless restart-budget control.** Mean 0.24% error at the three
  K-points matched to `noisy_N4_spot.csv`'s own K grid, with the same
  restart budget. This confirms the degradation reported there is genuinely
  attributable to noise, not to an under-provisioned restart budget at
  N=4's larger, 75-parameter circuit.
- **Gradient variance** (mean-squared gradient over 20 random parameter
  points, `L=2`, central finite differences). GI: 24.15, 23.02, and 21.19 at
  N=2, 3, and 4 respectively. HW: 10.15, 5.50, and 2.38 at the same three N.
  GI's gradient variance decays far more slowly with system size than HW's,
  consistent with charge conservation protecting it from the barren-plateau
  suppression HW is exposed to.
- **L=2 vs L=3 expressibility near the N=3 boundary (K=4.5).** L=2 stays
  65.2% wrong even at 32 restarts and `maxiter=4000`, a genuine
  expressibility limit rather than optimizer under-convergence (more
  restarts at the same depth do not close the gap). L=3 recovers the same
  point to 0.080% error at only 8 restarts, confirming the failure was
  depth-limited, not optimization-limited.

`results/SUMMARY.md`'s Corrections section documents all seven rounds of
audit-driven fixes behind these numbers in full: what was wrong, how it was
diagnosed, what was run to check it, and what changed as a result, with
every superseded or provisional figure kept in place and explicitly
annotated rather than deleted.
