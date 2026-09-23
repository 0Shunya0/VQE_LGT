"""
Task B / Step 3: PLAN ONLY -- no submission.

Builds the transpiled circuit for ibm_marrakesh, groups W's Pauli terms into
qubit-wise-commuting measurement bases (so all-Z diagonal terms share one
basis), and prints the shot/timing budget for approval. Reuses
schwinger.core.build_H_full / decompose_to_paulis and schwinger.backend.
build_gi_circuit unmodified.

W = build_H_full(x=16, K=0, N=3, F=2) -- the same full-space operator used as
"W" throughout schwinger/backend.py's NoisyEvaluator (and hence throughout
the existing noisy-VQE production code and Step 2 of this study), so the
hardware number will be directly comparable to Step 2's noiseless/depolarizing
values computed against the same operator.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService

import schwinger.core as core
from schwinger.backend import build_gi_circuit

N, L, X, K = 3, 2, 16.0, 0.0
F = 2
NQ = N * F
BACKEND_NAME = "ibm_marrakesh"
SEED_TRANSPILER = 0
OPT_LEVEL = 3

# -- QPU-time model -----------------------------------------------------------
# No API gives an exact pre-submission execution-time number, so this uses a
# published-order-of-magnitude per-shot budget for a Heron device at this
# circuit depth: gate time (~68ns/CZ, ~32-60ns/1Q) x depth ~100 is a few
# microseconds of coherent time, but real per-shot wall time on IBM hardware
# is dominated by fixed reset + readout + between-shot overhead, documented
# by IBM as on the order of a few hundred microseconds per shot for circuits
# in this qubit/depth range. This is stated explicitly as an estimate with
# its assumption, not a queried value.
SEC_PER_SHOT = 400e-6   # 400 microseconds/shot (reset + gates + readout), order-of-magnitude
PER_CIRCUIT_OVERHEAD_S = 0.02  # ~20ms fixed load/compile overhead per distinct measurement circuit (PUB)

SHOTS_PER_BASIS = 8192
BUDGET_S = 180.0


def main():
    H_full = core.build_H_full(X, K, N=N, F=F)
    paulis = core.decompose_to_paulis(H_full, NQ)
    print(f"W has {len(paulis)} nonzero Pauli terms (nq={NQ}).")

    op = SparsePauliOp.from_list([(label, coeff) for label, coeff, _ in paulis])
    groups = op.group_commuting(qubit_wise=True)
    n_bases = len(groups)
    diag_group_sizes = [len(g) for g in groups]
    print(f"Grouped into {n_bases} qubit-wise-commuting measurement bases (sizes: {diag_group_sizes}).")
    # identify the all-Z/I group explicitly
    for i, g in enumerate(groups):
        labels = g.paulis.to_labels()
        if all(set(lbl) <= {'I', 'Z'} for lbl in labels):
            print(f"  group {i}: all-Z/I diagonal group, {len(labels)} terms share this one basis.")

    qc, params, n_cx_logical = build_gi_circuit(NQ, L)
    backend = QiskitRuntimeService().backend(BACKEND_NAME)
    tqc = transpile(qc, backend=backend, optimization_level=OPT_LEVEL, seed_transpiler=SEED_TRANSPILER)
    ops_count = tqc.count_ops()
    n_2q = ops_count.get('cz', 0) + ops_count.get('ecr', 0) + ops_count.get('cx', 0)
    depth = tqc.depth()

    total_shots = n_bases * SHOTS_PER_BASIS
    est_qpu_s = n_bases * (SHOTS_PER_BASIS * SEC_PER_SHOT + PER_CIRCUIT_OVERHEAD_S)

    queue_len = backend.status().pending_jobs

    print()
    print("=" * 60)
    print("QPU JOB PLAN (NOT SUBMITTED)")
    print("=" * 60)
    print(f"Backend: {BACKEND_NAME}  (queue length: {queue_len} pending jobs)")
    print(f"Transpiled circuit: {n_2q} CZ, depth {depth} (opt_level={OPT_LEVEL}, seed={SEED_TRANSPILER})")
    print(f"Measurement bases after grouping: {n_bases}")
    print(f"Shots per basis: {SHOTS_PER_BASIS}")
    print(f"Total shots: {total_shots}")
    print(f"Estimated QPU time: {est_qpu_s:.1f} s "
          f"({SEC_PER_SHOT*1e6:.0f} us/shot assumption + {PER_CIRCUIT_OVERHEAD_S*1000:.0f} ms/circuit overhead)")
    print(f"Budget: {BUDGET_S:.0f} s -- {'OK' if est_qpu_s <= BUDGET_S else 'OVER BUDGET, reduce shots/basis'}")
    print()
    print("STOPPED HERE per instructions. Waiting for go-ahead before any submission.")

    plan = dict(backend=BACKEND_NAME, n_bases=n_bases, shots_per_basis=SHOTS_PER_BASIS,
                total_shots=total_shots, est_qpu_s=est_qpu_s, n_cz=n_2q, depth=depth,
                queue_len=queue_len, seed_transpiler=SEED_TRANSPILER, opt_level=OPT_LEVEL)
    with open("results/ibm_N3_L2_K0_plan.json", "w") as f:
        json.dump(plan, f, indent=2)
    print("Saved results/ibm_N3_L2_K0_plan.json")


if __name__ == "__main__":
    main()
