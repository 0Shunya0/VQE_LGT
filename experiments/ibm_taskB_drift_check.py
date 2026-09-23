"""
Diagnostic batch: is the job-1 -> Job-0 energy drift (Sampler pipeline vs
Estimator pipeline, ~24 min apart) a device change or a primitive/pipeline
difference? Four jobs in ONE Batch on ibm_marrakesh, submitted in order
S1, E1, S2, E2, all on layout [8,9,10,11,18,31], same fixed theta.

  S: job daoeuotr85ps73ff6a70's own ISA circuits (retrieved from its inputs),
     SamplerV2, 8192 shots/basis, no twirling, no DD; energy reconstructed
     exactly as in ibm_taskB_step4_submit.py.
  E: job daof9vdr85ps73ff6u10's own ISA circuit + observable (retrieved from
     its inputs), EstimatorV2, resilience_level=0, no twirling, no DD,
     default_shots=4096 (Job 0's precision).

Nothing is re-optimized. Four jobs, submitted once.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json
import time

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import eigh
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, EstimatorV2, Batch

import schwinger.core as core

N, K, X, F, NQ = 3, 0.0, 16.0, 2, 6
BACKEND_NAME = "ibm_marrakesh"
JOB1_ID = "daoeuotr85ps73ff6a70"
JOB0_ID = "daof9vdr85ps73ff6u10"
SAMPLER_SHOTS = 8192
ESTIMATOR_SHOTS = 4096
EST_S_QPU = 13.0   # job 1's actual qpu_charge_time_seconds
EST_E_QPU = 8.0    # Job 0's actual qpu_charge_time_seconds
BUDGET_S = 90.0


def energy_from_counts(counts_by_group, groups):
    total = 0.0
    for group, counts in zip(groups, counts_by_group):
        n_shots = sum(counts.values())
        for label, coeff in zip(group.paulis.to_labels(), group.coeffs):
            coeff = float(np.real(coeff))
            exp = 0.0
            for bitstring, cnt in counts.items():
                sign = 1
                for i, ch in enumerate(label):
                    if ch != 'I' and bitstring[i] == '1':
                        sign *= -1
                exp += sign * cnt
            total += coeff * exp / n_shots
    return total


def main():
    service = QiskitRuntimeService()
    backend = service.backend(BACKEND_NAME)

    job1 = service.job(JOB1_ID)
    job0 = service.job(JOB0_ID)
    s_circuits = [p[0] for p in job1.inputs['pubs']]
    e_circuit, e_obs_dict = job0.inputs['pubs'][0][0], job0.inputs['pubs'][0][1]
    isa_W = SparsePauliOp.from_list(list(e_obs_dict.items()))
    assert len(s_circuits) == 5 and e_circuit.num_qubits == 156 and isa_W.num_qubits == 156

    H_full = core.build_H_full(X, K, N=N, F=F)
    paulis = core.decompose_to_paulis(H_full, NQ)
    W = SparsePauliOp.from_list([(l, c) for l, c, _ in paulis])
    groups = W.group_commuting(qubit_wise=True)
    Hsub, _ = core.build_H_subspace(X, K, N=N, F=F)
    E_exact = float(eigh(Hsub, eigvals_only=True)[0])

    est_total = 2 * EST_S_QPU + 2 * EST_E_QPU
    print(f"QPU estimate (from prior actual charges): 2 x Sampler(5 circuits x {SAMPLER_SHOTS}) ~ {2*EST_S_QPU:.0f}s"
          f" + 2 x Estimator(1 pub, {ESTIMATOR_SHOTS} shots) ~ {2*EST_E_QPU:.0f}s = ~{est_total:.0f}s  (limit {BUDGET_S:.0f}s)")
    if est_total >= BUDGET_S:
        print("Estimate >= limit -- NOT submitting; asking user.")
        return

    order = ["S1", "E1", "S2", "E2"]
    subs = {}
    with Batch(backend=backend) as batch:
        for name in order:
            if name.startswith("S"):
                prim = SamplerV2(mode=batch)
                prim.options.twirling.enable_gates = False
                prim.options.twirling.enable_measure = False
                prim.options.dynamical_decoupling.enable = False
                job = prim.run(s_circuits, shots=SAMPLER_SHOTS)
            else:
                prim = EstimatorV2(mode=batch)
                prim.options.resilience_level = 0
                prim.options.twirling.enable_gates = False
                prim.options.twirling.enable_measure = False
                prim.options.dynamical_decoupling.enable = False
                prim.options.default_shots = ESTIMATOR_SHOTS
                job = prim.run([(e_circuit, isa_W)])
            subs[name] = (job, time.time())
            print(f"{name}: submitted {job.job_id()}", flush=True)

        results = {}
        for name in order:
            job, t_sub = subs[name]
            res = job.result()
            m = job.metrics()
            if name.startswith("S"):
                counts = [getattr(res[i].data, list(res[i].data.keys())[0]).get_counts() for i in range(len(res))]
                E = energy_from_counts(counts, groups)
            else:
                E = float(res[0].data.evs)
            results[name] = dict(job_id=job.job_id(), energy=E,
                                 rel_err_pct=abs(E - E_exact) / abs(E_exact) * 100,
                                 qpu_seconds=m.get('usage', {}).get('qpu_charge_time_seconds'),
                                 timestamps=m.get('timestamps'))

    print(f"\nexact = {E_exact:.6f}\n")
    for name in order:
        r = results[name]
        ts = r['timestamps']
        print(f"{name}: E={r['energy']:.6f}  rel_err={r['rel_err_pct']:.4f}%  job={r['job_id']}  "
              f"qpu_s={r['qpu_seconds']}  start={ts.get('running')}  end={ts.get('finished')}")

    with open("results/ibm_N3_L2_K0_drift_check.json", "w") as f:
        json.dump(dict(E_exact=E_exact, order=order, layout=[8, 9, 10, 11, 18, 31],
                       reference=dict(job1_sampler=-36.7279052734375, job0_estimator=-26.922119),
                       jobs=results), f, indent=2)
    print("\nSaved results/ibm_N3_L2_K0_drift_check.json")


if __name__ == "__main__":
    main()
