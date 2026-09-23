"""
Noiseless test of the Sampler and Estimator pipelines on states that are NOT
symmetric (random theta, seeds 1-3, plus theta=0), against exact <psi|W|psi>
from core. Diagnostic only; no QPU, no saved results modified.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService

import schwinger.core as core
from schwinger.backend import build_gi_circuit, state_prep_ops
from ibm_taskB_step4_submit import (build_groups, basis_for_group, add_basis_rotation,
                                     energy_from_counts, build_measurement_circuits)

N, L, X, K, F, NQ = 3, 2, 16.0, 0.0, 2, 6
PINNED = [8, 9, 10, 11, 18, 31]
SHOTS = 8192

H_full = core.build_H_full(X, K, N=N, F=F)
paulis = core.decompose_to_paulis(H_full, NQ)
W = SparsePauliOp.from_list([(l, c) for l, c, _ in paulis])
groups = build_groups()
psi0 = core.make_psi0(N, F)
n_p = L * ((NQ - 1) + NQ)
qc, params, _ = build_gi_circuit(NQ, L)
backend = QiskitRuntimeService().backend("ibm_marrakesh")
sim = AerSimulator()


def bound_circuit(theta):
    ans = qc.assign_parameters({params[i]: theta[i] for i in range(n_p)})
    prep = QuantumCircuit(NQ)
    for q in state_prep_ops(psi0, NQ):
        prep.x(q)
    full = QuantumCircuit(NQ)
    full.compose(prep, inplace=True)
    full.compose(ans, inplace=True)
    return full


def sampler_pipeline(bound):
    circs = build_measurement_circuits(bound, groups, NQ)
    tq = [transpile(c, sim, optimization_level=1) for c in circs]
    res = sim.run(tq, shots=SHOTS, seed_simulator=7).result()
    counts = [res.get_counts(i) for i in range(len(tq))]
    E_shots = energy_from_counts(counts, groups, SHOTS)
    probs = []
    for g in groups:
        c = bound.copy()
        add_basis_rotation(c, basis_for_group(g, NQ))
        probs.append(Statevector(c).probabilities_dict())
    E_inf = energy_from_counts(probs, groups, SHOTS)
    return E_shots, E_inf


def estimator_pipeline(bound):
    tqc = transpile(bound, backend=backend, optimization_level=3, seed_transpiler=0, initial_layout=PINNED)
    assert tqc.layout.final_index_layout() == PINNED
    isa_W = W.apply_layout(tqc.layout)
    n_total = tqc.num_qubits
    pos = {q: j for j, q in enumerate(PINNED)}
    red = QuantumCircuit(6)
    for ins in tqc.data:
        qi = [tqc.find_bit(q).index for q in ins.qubits]
        red.append(ins.operation, [pos[i] for i in qi])
    terms = []
    for lbl, c in zip(isa_W.paulis.to_labels(), isa_W.coeffs):
        chars = ['I'] * 6
        for q in PINNED:
            chars[5 - pos[q]] = lbl[n_total - 1 - q]
        terms.append((''.join(chars), complex(c)))
    Wr = SparsePauliOp.from_list(terms)
    return float(StatevectorEstimator().run([(red, Wr)]).result()[0].data.evs)


thetas = {f"seed{s}": np.random.default_rng(s).uniform(-np.pi, np.pi, n_p) for s in (1, 2, 3)}
thetas["theta=0"] = np.zeros(n_p)

print(f"{'state':10s} {'exact(core)':>13s} {'circuit-sv':>12s} {'Sampler@8192':>13s} {'Sampler@inf':>12s} {'Estimator':>12s}")
for name, th in thetas.items():
    psi = core.gauge_invariant_ansatz(th, psi0, NQ, L)
    E_core = float(np.real(psi.conj() @ H_full @ psi))
    bnd = bound_circuit(th)
    E_circ = float(np.real(Statevector(bnd).expectation_value(W)))
    E_s_shots, E_s_inf = sampler_pipeline(bnd)
    E_e = estimator_pipeline(bnd)
    print(f"{name:10s} {E_core:13.5f} {E_circ:12.5f} {E_s_shots:13.5f} {E_s_inf:12.5f} {E_e:12.5f}")
