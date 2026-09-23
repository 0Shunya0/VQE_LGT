"""
Task B follow-up: three EstimatorV2 hardware evaluations of the SAME fixed
theta (results/theta_N3_L2_K0.json), same observable W (corrected
build_H_full at K=0, x=16), all pinned to initial_layout [8,9,10,11,18,31]
(job daoeuotr85ps73ff6a70's exact physical qubits, same order), submitted
together in one Batch so they share a calibration window:

  Job 0: no mitigation (baseline)
  Job A: measure_mitigation=True only
  Job B: measure_mitigation=True, zne_mitigation=True, noise_factors=(1,3),
         extrapolator="linear"

No re-optimization. Reuses schwinger.core / schwinger.backend unmodified.
Three jobs, submitted once.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import datetime
import json
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import eigh
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2, Batch

import schwinger.core as core
from schwinger.backend import build_gi_circuit, state_prep_ops

N, L, X, K, F = 3, 2, 16.0, 0.0, 2
NQ = 6
NCX = 20
BACKEND_NAME = "ibm_marrakesh"
PINNED_LAYOUT = [8, 9, 10, 11, 18, 31]
SEED_TRANSPILER = 0
OPT_LEVEL = 3
DEFAULT_SHOTS = 4096
SPAM_PAPER = 0.005


def make_estimator(mode, measure_mitigation, zne_mitigation):
    est = EstimatorV2(mode=mode)
    est.options.default_shots = DEFAULT_SHOTS
    est.options.twirling.enable_gates = False
    est.options.twirling.enable_measure = False
    est.options.resilience.measure_mitigation = measure_mitigation
    est.options.resilience.zne_mitigation = zne_mitigation
    if zne_mitigation:
        est.options.resilience.zne.noise_factors = (1, 3)
        est.options.resilience.zne.extrapolator = "linear"
    return est


def refit_p(E, E_exact, E_mix, spam):
    w_eff = (E - E_mix) / (E_exact - E_mix)
    w_eff_nospam = w_eff / (1 - 2 * spam)
    if w_eff_nospam <= 0:
        return float('nan')
    return 1 - w_eff_nospam ** (1 / NCX)


def main():
    with open("results/theta_N3_L2_K0.json") as f:
        theta = np.array(json.load(f)["theta"])

    qc, params, n_cx = build_gi_circuit(NQ, L)
    ansatz_bound = qc.assign_parameters({params[i]: theta[i] for i in range(len(params))})
    psi0 = core.make_psi0(N, F)
    prep = QuantumCircuit(NQ)
    for q in state_prep_ops(psi0, NQ):
        prep.x(q)
    bound = QuantumCircuit(NQ)
    bound.compose(prep, inplace=True)
    bound.compose(ansatz_bound, inplace=True)

    H_full = core.build_H_full(X, K, N=N, F=F)
    paulis = core.decompose_to_paulis(H_full, NQ)
    W = SparsePauliOp.from_list([(label, coeff) for label, coeff, _ in paulis])

    Hsub, basis = core.build_H_subspace(X, K, N=N, F=F)
    w_spec = eigh(Hsub, eigvals_only=True)
    E_exact = float(w_spec[0])
    E_mix = float(np.mean(w_spec))

    service = QiskitRuntimeService()
    backend = service.backend(BACKEND_NAME)
    tqc = transpile(bound, backend=backend, optimization_level=OPT_LEVEL,
                     seed_transpiler=SEED_TRANSPILER, initial_layout=PINNED_LAYOUT)
    ops = tqc.count_ops()
    n_cz = ops.get('cz', 0) + ops.get('ecr', 0) + ops.get('cx', 0)
    depth = tqc.depth()
    physical_qubits = tqc.layout.final_index_layout()
    isa_W = W.apply_layout(tqc.layout)

    print(f"Circuit: {n_cz} CZ, depth {depth}, physical qubits {physical_qubits}")
    assert physical_qubits == PINNED_LAYOUT, f"layout mismatch: {physical_qubits} != {PINNED_LAYOUT}"
    assert n_cz == NCX, f"expected {NCX} CZ, got {n_cz}"

    jobs_spec = [
        ("Job0_baseline", False, False),
        ("JobA_readout", True, False),
        ("JobB_readout_zne", True, True),
    ]

    results = {}
    with Batch(backend=backend) as batch:
        submitted = []
        for name, meas_mit, zne_mit in jobs_spec:
            est = make_estimator(batch, meas_mit, zne_mit)
            job = est.run([(tqc, isa_W)])
            t_submit = time.time()
            print(f"{name}: submitted job {job.job_id()} at {datetime.datetime.now().isoformat()}")
            submitted.append((name, job, t_submit))

        for name, job, t_submit in submitted:
            print(f"Waiting on {name} ({job.job_id()})...")
            res = job.result()
            metrics = job.metrics()
            E = float(res[0].data.evs)
            std = float(res[0].data.stds) if hasattr(res[0].data, 'stds') else None
            qpu_s = metrics.get('usage', {}).get('qpu_charge_time_seconds') or metrics.get('usage', {}).get('quantum_seconds')
            calib_ts = metrics.get('timestamps', {})
            rel_err = abs(E - E_exact) / abs(E_exact) * 100
            results[name] = dict(job_id=job.job_id(), energy=E, std=std, rel_err_pct=rel_err,
                                  qpu_seconds=qpu_s, timestamps=calib_ts, t_submit=t_submit)
            print(f"  {name}: E={E:.6f}  rel_err={rel_err:.4f}%  qpu_s={qpu_s}  job_id={job.job_id()}")

    t0 = results["Job0_baseline"]["t_submit"]
    tA = results["JobA_readout"]["t_submit"]
    tB = results["JobB_readout_zne"]["t_submit"]
    print(f"\nSubmission gaps: Job0->JobA = {tA-t0:.2f}s, JobA->JobB = {tB-tA:.2f}s")

    E0 = results["Job0_baseline"]["energy"]
    EA = results["JobA_readout"]["energy"]
    EB = results["JobB_readout_zne"]["energy"]

    readout_fraction_batch = E0 - EA
    print(f"\nReadout fraction from this batch (Job0 - JobA) = {readout_fraction_batch:.6f}")

    step1 = json.load(open("results/ibm_N3_L2_K0_readout_corrected.json"))
    print(f"Step 1 classical correction of job 1 (E_raw - E_readout_corrected) = "
          f"{step1['E_raw'] - step1['E_readout_corrected']:.6f}")

    job1_E_raw = step1["E_raw"]
    print(f"\nJob0 (this batch, unmitigated) = {E0:.6f}  vs job 1 (original unmitigated) = {job1_E_raw:.6f}  "
          f"drift = {E0 - job1_E_raw:.6f}")

    # paper's ZNE model prediction, for comparison to Job B
    p_eff_job0 = refit_p(E0, E_exact, E_mix, SPAM_PAPER)
    zne_prediction = core.zne_linear(E_exact, E_mix, p_eff_job0, NCX, spam=SPAM_PAPER)
    print(f"\nJob B (readout+ZNE) = {EB:.6f}  rel_err={results['JobB_readout_zne']['rel_err_pct']:.4f}%")
    print(f"Paper's analytic ZNE model prediction (using p_eff refit from Job0): {zne_prediction:.6f}")
    print(f"Job B vs exact: diff={EB-E_exact:.6f}   Job B vs analytic ZNE prediction: diff={EB-zne_prediction:.6f}")

    out = dict(E_exact=E_exact, E_mix=E_mix, physical_qubits=physical_qubits, n_cz=n_cz, depth=depth,
               jobs=results,
               readout_fraction_batch=readout_fraction_batch,
               step1_readout_correction=step1['E_raw'] - step1['E_readout_corrected'],
               job0_vs_job1_drift=E0 - job1_E_raw,
               p_eff_job0_pct=p_eff_job0 * 100,
               zne_analytic_prediction=zne_prediction,
               jobB_vs_exact=EB - E_exact,
               jobB_vs_zne_prediction=EB - zne_prediction)
    with open("results/ibm_N3_L2_K0_mitigated.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/ibm_N3_L2_K0_mitigated.json")


if __name__ == "__main__":
    main()
