"""
Task B follow-up: plan (only) for two EstimatorV2 hardware evaluations of the
SAME fixed theta (results/theta_N3_L2_K0.json) on ibm_marrakesh --
Job A (readout mitigation only) and Job B (readout + linear ZNE). No
re-optimization; reuses schwinger.core / schwinger.backend unmodified.
STOPS before submission -- see ibm_taskB_mitigated_submit.py for that.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import datetime
import json

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService

import schwinger.core as core
from schwinger.backend import build_gi_circuit, state_prep_ops

N, L, X, K = 3, 2, 16.0, 0.0
F = 2
NQ = N * F
BACKEND_NAME = "ibm_marrakesh"
SEED_TRANSPILER = 0
OPT_LEVEL = 3
DEFAULT_SHOTS = 4096
SEC_PER_SHOT = 400e-6   # same order-of-magnitude assumption as ibm_taskB_step3_plan.py
PER_CIRCUIT_OVERHEAD_S = 0.02
PRIOR_JOB_ID = "daoeuotr85ps73ff6a70"
PRIOR_JOB_CREATION_UTC = "2026-09-21T08:48:35.184380+00:00"


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
    groups = W.group_commuting(qubit_wise=True)
    n_groups = len(groups)

    service = QiskitRuntimeService()
    backend = service.backend(BACKEND_NAME)
    tqc = transpile(bound, backend=backend, optimization_level=OPT_LEVEL, seed_transpiler=SEED_TRANSPILER)
    ops = tqc.count_ops()
    n_cz = ops.get('cz', 0) + ops.get('ecr', 0) + ops.get('cx', 0)
    depth = tqc.depth()
    physical_qubits = tqc.layout.final_index_layout() if tqc.layout else list(range(NQ))

    # readout errors nearest job daoeuotr85ps73ff6a70's run time
    dt = datetime.datetime.fromisoformat(PRIOR_JOB_CREATION_UTC)
    props = backend.properties(datetime=dt)
    used_historical = props is not None
    if props is None:
        props = backend.properties()
    readout_errors = {q: float(props.readout_error(q)) for q in physical_qubits}
    mean_readout = float(np.mean(list(readout_errors.values())))

    queue_len = backend.status().pending_jobs

    # cost estimate: Job A ~ n_groups circuits; Job B ~ n_groups * len(noise_factors) circuits
    # (+ a fixed TREX readout-calibration overhead, order-of-magnitude, shared once per job)
    trex_overhead_shots = DEFAULT_SHOTS  # order-of-magnitude: one extra "group" worth of shots
    jobA_shots = n_groups * DEFAULT_SHOTS + trex_overhead_shots
    jobA_circuits = n_groups + 1
    jobA_qpu_s = jobA_shots * SEC_PER_SHOT + jobA_circuits * PER_CIRCUIT_OVERHEAD_S

    n_noise_factors = 2  # (1, 3)
    jobB_shots = n_groups * n_noise_factors * DEFAULT_SHOTS + trex_overhead_shots
    jobB_circuits = n_groups * n_noise_factors + 1
    jobB_qpu_s = jobB_shots * SEC_PER_SHOT + jobB_circuits * PER_CIRCUIT_OVERHEAD_S

    print("=" * 70)
    print("PLAN ONLY -- NOT SUBMITTED")
    print("=" * 70)
    print(f"Backend: {BACKEND_NAME}  (queue length: {queue_len} pending jobs)")
    print(f"Transpiled circuit: {n_cz} CZ, depth {depth}, physical qubits {physical_qubits}")
    print(f"W: {len(paulis)} Pauli terms -> {n_groups} qubit-wise-commuting groups")
    print()
    print(f"Job A (readout mitigation only, TREX, gate twirling off):")
    print(f"  shots per circuit: {DEFAULT_SHOTS}  (precision via default_shots)")
    print(f"  estimated circuits: {jobA_circuits} ({n_groups} measurement groups + ~1 TREX calibration overhead)")
    print(f"  estimated total shots: {jobA_shots}")
    print(f"  estimated QPU time: {jobA_qpu_s:.1f} s")
    print()
    print(f"Job B (readout mitigation + linear ZNE, noise_factors=(1,3)):")
    print(f"  shots per circuit: {DEFAULT_SHOTS}")
    print(f"  estimated circuits: {jobB_circuits} ({n_groups} groups x {n_noise_factors} noise factors + ~1 TREX overhead)")
    print(f"  estimated total shots: {jobB_shots}")
    print(f"  estimated QPU time: {jobB_qpu_s:.1f} s")
    print()
    total_est = jobA_qpu_s + jobB_qpu_s
    print(f"TOTAL estimated QPU time (both jobs): {total_est:.1f} s   (budget: 180 s -- {'OK' if total_est <= 180 else 'OVER BUDGET'})")
    print()
    print(f"Readout-error calibration snapshot: {'historical (nearest job creation time)' if used_historical else 'CURRENT (historical snapshot unavailable)'}")
    print(f"  reference job: {PRIOR_JOB_ID}, created {PRIOR_JOB_CREATION_UTC}")
    print(f"  per-qubit readout error (physical qubits {physical_qubits}):")
    for q, e in readout_errors.items():
        print(f"    qubit {q}: {e:.5f}")
    print(f"  mean over these 6 qubits: {mean_readout:.5f}   (paper's assumed eps = 0.005)")
    print()
    print("STOPPED HERE per instructions. Waiting for go-ahead before any submission.")

    out = dict(jobA=dict(shots_per_circuit=DEFAULT_SHOTS, est_circuits=jobA_circuits,
                          est_total_shots=jobA_shots, est_qpu_s=jobA_qpu_s),
               jobB=dict(shots_per_circuit=DEFAULT_SHOTS, est_circuits=jobB_circuits,
                         est_total_shots=jobB_shots, est_qpu_s=jobB_qpu_s),
               total_est_qpu_s=total_est, n_cz=n_cz, depth=depth,
               physical_qubits=physical_qubits, readout_errors=readout_errors,
               mean_readout_error=mean_readout, paper_eps=0.005,
               queue_len=queue_len, used_historical_calibration=used_historical)
    with open("results/ibm_N3_L2_K0_mitigated_plan.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/ibm_N3_L2_K0_mitigated_plan.json")


if __name__ == "__main__":
    main()
