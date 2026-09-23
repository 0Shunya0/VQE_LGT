"""
Final hardware diagnostic: localize the Sampler-vs-Estimator gap.
  C: EstimatorV2 (resilience_level=0, no twirling/DD), Job 0's exact ISA
     circuit, W split into its 5 QWC groups as 5 observables in ONE pub.
  D: SamplerV2 8192 shots on Job 0's exact bare ISA circuit with the 5
     basis rotations appended AFTER transpilation (native rz/sx) + measure.
Both verified noiselessly (per group) before any submission. One Batch,
two jobs, submitted once.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.linalg import eigh
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, EstimatorV2, Batch

import schwinger.core as core
from ibm_taskB_step4_submit import build_groups, basis_for_group, energy_from_counts

PINNED = [8, 9, 10, 11, 18, 31]
N, K, X, F = 3, 0.0, 16.0, 2
SHOTS_S, SHOTS_E = 8192, 4096
QPU_EST = 8.0 + 13.0
QPU_LIMIT = 60.0

service = QiskitRuntimeService()
backend = service.backend("ibm_marrakesh")
j0 = service.job("daof9vdr85ps73ff6u10")
e_circ = j0.inputs["pubs"][0][0]
isa_W_job0 = SparsePauliOp.from_list(list(j0.inputs["pubs"][0][1].items()))
groups = build_groups()
Hsub, _ = core.build_H_subspace(X, K, N=N, F=F)
E_exact = float(eigh(Hsub, eigvals_only=True)[0])

# --- map groups to the 156-qubit frame; must sum to Job 0's observable
g156 = [g.apply_layout(PINNED, num_qubits=156) for g in groups]
tot156 = g156[0]
for g in g156[1:]:
    tot156 = tot156 + g
assert np.allclose((tot156 - isa_W_job0).simplify().coeffs, 0) or len((tot156 - isa_W_job0).simplify().coeffs) <= 1, \
    "group observables do not sum to Job 0's observable"
assert abs((tot156 - isa_W_job0).simplify().coeffs).max() < 1e-9

pos = {q: j for j, q in enumerate(PINNED)}


def reduce_circuit(qc156):
    red = QuantumCircuit(6)
    for ins in qc156.data:
        nm = ins.operation.name
        if nm in ("measure", "barrier"):
            continue
        qi = [qc156.find_bit(q).index for q in ins.qubits]
        assert all(q in pos for q in qi), f"instruction {nm} on non-active qubit {qi}"
        red.append(ins.operation, [pos[q] for q in qi])
    return red


# --- noiseless per-group reference on Job 0's exact circuit
red_e = reduce_circuit(e_circ)
ref_groups = [float(StatevectorEstimator().run([(red_e, g)]).result()[0].data.evs) for g in groups]
print("noiseless per-group reference:", [f"{v:.4f}" for v in ref_groups], " total", f"{sum(ref_groups):.6f}")
assert abs(sum(ref_groups) - (-43.55998866505679)) < 1e-6

# --- build D circuits: append rotations after transpilation, native rz/sx
def rotations(qc, basis):
    for i, b in enumerate(basis):          # basis[i]: physics qubit i -> physical PINNED[5-i]
        ph = PINNED[5 - i]
        if b == "X":                       # H = rz(pi/2) sx rz(pi/2)
            qc.rz(np.pi / 2, ph); qc.sx(ph); qc.rz(np.pi / 2, ph)
        elif b == "Y":                     # Sdg then H = rz(-pi/2) + H -> sx, rz(pi/2)
            qc.rz(-np.pi / 2, ph); qc.rz(np.pi / 2, ph); qc.sx(ph); qc.rz(np.pi / 2, ph)


d_circs, d_pre = [], []
for g in groups:
    basis = basis_for_group(g, 6)
    pre = e_circ.copy()
    rotations(pre, basis)
    d_pre.append(pre)
    c = pre.copy()
    creg = ClassicalRegister(6, "meas")
    c.add_register(creg)
    for j in range(6):                     # clbit j <- physical PINNED[j] (qiskit qubit j)
        c.measure(PINNED[j], creg[j])
    d_circs.append(c)

# --- noiseless check of D: probabilities -> per-group energies
d_ref = []
for g, pre in zip(groups, d_pre):
    probs = Statevector(reduce_circuit(pre)).probabilities_dict()
    d_ref.append(energy_from_counts([probs], [g], 1))
print("noiseless D per-group        :", [f"{v:.4f}" for v in d_ref], " total", f"{sum(d_ref):.6f}")
assert max(abs(a - b) for a, b in zip(d_ref, ref_groups)) < 1e-6, "D pipeline disagrees noiselessly"
print("Noiseless checks PASSED for C and D.")

print(f"QPU estimate: Estimator ~8s + Sampler ~13s = ~{QPU_EST:.0f}s (limit {QPU_LIMIT:.0f}s)")
if QPU_EST >= QPU_LIMIT:
    raise SystemExit("estimate over limit; not submitting")

with Batch(backend=backend) as batch:
    est = EstimatorV2(mode=batch)
    est.options.resilience_level = 0
    est.options.twirling.enable_gates = False
    est.options.twirling.enable_measure = False
    est.options.dynamical_decoupling.enable = False
    est.options.default_shots = SHOTS_E
    jobC = est.run([(e_circ, g156)])
    print("C submitted:", jobC.job_id(), flush=True)
    smp = SamplerV2(mode=batch)
    smp.options.twirling.enable_gates = False
    smp.options.twirling.enable_measure = False
    smp.options.dynamical_decoupling.enable = False
    jobD = smp.run(d_circs, shots=SHOTS_S)
    print("D submitted:", jobD.job_id(), flush=True)

    resC = jobC.result(); mC = jobC.metrics()
    resD = jobD.result(); mD = jobD.metrics()

evsC = [float(v) for v in np.asarray(resC[0].data.evs).ravel()]
countsD = [getattr(resD[i].data, list(resD[i].data.keys())[0]).get_counts() for i in range(5)]
evsD = [energy_from_counts([c], [g], SHOTS_S) for c, g in zip(countsD, groups)]

# B: S1 / S2 per-group
def group_e(jid):
    r = service.job(jid).result()
    cs = [getattr(r[i].data, list(r[i].data.keys())[0]).get_counts() for i in range(len(r))]
    return [energy_from_counts([c], [g], SHOTS_S) for c, g in zip(cs, groups)]

evsS1 = group_e("daofnv78gn2s739nqjlg")
evsS2 = group_e("daofnvopqrnc7399u9ig")

print(f"\nexact total = {E_exact:.6f}")
hdr = f"{'':14s}" + "".join(f"{'g'+str(i):>10s}" for i in range(5)) + f"{'total':>10s}"
print(hdr)
for name, v in [("noiseless", ref_groups), ("B: S1", evsS1), ("B: S2", evsS2),
                ("C: Estimator", evsC), ("D: Sampler", evsD)]:
    print(f"{name:14s}" + "".join(f"{x:10.4f}" for x in v) + f"{sum(v):10.4f}")
print(f"\nC job {jobC.job_id()} qpu_s={mC['usage'].get('qpu_charge_time_seconds')} {mC['timestamps']}")
print(f"D job {jobD.job_id()} qpu_s={mD['usage'].get('qpu_charge_time_seconds')} {mD['timestamps']}")

with open("results/ibm_N3_L2_K0_gap_localize.json", "w") as f:
    json.dump(dict(E_exact=E_exact, groups=[list(g.paulis.to_labels()) for g in groups],
                   noiseless=ref_groups, S1=evsS1, S2=evsS2, C_estimator=evsC, D_sampler=evsD,
                   totals=dict(noiseless=sum(ref_groups), S1=sum(evsS1), S2=sum(evsS2),
                               C=sum(evsC), D=sum(evsD)),
                   jobC=dict(id=jobC.job_id(), metrics=mC), jobD=dict(id=jobD.job_id(), metrics=mD),
                   raw_counts_D=[{k: int(v) for k, v in c.items()} for c in countsD]),
              f, indent=2, default=str)
print("Saved results/ibm_N3_L2_K0_gap_localize.json")
