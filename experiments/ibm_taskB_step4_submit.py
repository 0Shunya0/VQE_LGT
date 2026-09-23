"""
Task B / Step 4: submit the ONE hardware job for N=3, L=2, K=0, using the
theta already optimized (noiselessly, against build_H_subspace) and saved in
results/theta_N3_L2_K0.json. Reuses schwinger.core / schwinger.backend
unmodified.

Gate (checked before anything is submitted): the noiseless statevector
energy from ibm_taskB_step2_references.py must be within 0.01 of the exact
ground energy. If not, STOP -- that would mean the core.py fix isn't
reaching this code path.

Before trusting real hardware counts, this script first self-tests its own
classical reconstruction of <W> from grouped, basis-rotated measurement
circuits against a noiseless AerSimulator shot simulation, and checks that
reconstruction matches the known noiseless statevector energy. Only if that
self-test passes does it submit to ibm_marrakesh. One job, one submission,
no retry.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

import schwinger.core as core
from schwinger.backend import build_gi_circuit, state_prep_ops

N, L, X, K = 3, 2, 16.0, 0.0
F = 2
NQ = N * F
BACKEND_NAME = "ibm_marrakesh"
SEED_TRANSPILER = 0
OPT_LEVEL = 3
SHOTS_PER_BASIS = 8192
GATE_TOL = 0.01


def build_groups():
    H_full = core.build_H_full(X, K, N=N, F=F)
    paulis = core.decompose_to_paulis(H_full, NQ)
    op = SparsePauliOp.from_list([(label, coeff) for label, coeff, _ in paulis])
    groups = op.group_commuting(qubit_wise=True)
    return groups


def basis_for_group(group, nq):
    """Per-qubit measurement basis ('X','Y','Z') for one qubit-wise-commuting
    group: for each qubit, the unique non-identity Pauli seen across the
    group's terms (defaults to 'Z' if the qubit is I in every term)."""
    basis = ['Z'] * nq
    for label in group.paulis.to_labels():
        for i, ch in enumerate(label):
            if ch != 'I':
                basis[i] = ch
    return basis


def add_basis_rotation(qc, basis):
    """basis[i] is the measurement basis for physics-qubit i; qiskit qubit
    for physics-qubit i is (nq-1-i) (see schwinger/backend.py's qk())."""
    nq = len(basis)
    for i, b in enumerate(basis):
        q = nq - 1 - i
        if b == 'X':
            qc.h(q)
        elif b == 'Y':
            qc.sdg(q)
            qc.h(q)


def energy_from_counts(counts_by_group, groups, shots_per_basis):
    """counts_by_group[i] = {bitstring: count} for group i. Bitstring keys
    from Qiskit are in the same left-to-right convention as our Pauli
    labels (position i <-> physics-qubit i), confirmed in ibm_taskB_step3."""
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
            exp /= n_shots
            total += coeff * exp
    return total


def build_measurement_circuits(tqc_ansatz, groups, nq):
    circuits = []
    for group in groups:
        basis = basis_for_group(group, nq)
        qc = tqc_ansatz.copy()
        add_basis_rotation(qc, basis)
        qc.measure_all()
        circuits.append(qc)
    return circuits


def main():
    # --- gate -------------------------------------------------------------
    with open("results/ibm_N3_L2_K0_references.json") as f:
        refs = json.load(f)
    E_exact, E_noiseless, E_depol = refs["E_exact"], refs["E_noiseless"], refs["E_depol"]
    print(f"exact:               {E_exact:.6f}")
    print(f"noiseless stateveq:  {E_noiseless:.6f}  (rel err {abs(E_noiseless-E_exact)/abs(E_exact)*100:.4f}%)")
    print(f"depolarizing+SPAM:   {E_depol:.6f}  (rel err {abs(E_depol-E_exact)/abs(E_exact)*100:.4f}%)")
    if abs(E_noiseless - E_exact) >= GATE_TOL:
        print(f"\nGATE FAILED: |noiseless - exact| = {abs(E_noiseless-E_exact):.6f} >= {GATE_TOL}. STOPPING, not submitting.")
        return
    print(f"\nGATE PASSED: |noiseless - exact| = {abs(E_noiseless-E_exact):.6f} < {GATE_TOL}")

    with open("results/theta_N3_L2_K0.json") as f:
        saved = json.load(f)
    theta = np.array(saved["theta"])

    groups = build_groups()
    n_bases = len(groups)
    print(f"\n{n_bases} measurement groups from W's Pauli decomposition.")

    qc, params, n_cx = build_gi_circuit(NQ, L)
    ansatz_bound = qc.assign_parameters({params[i]: theta[i] for i in range(len(params))})
    psi0 = core.make_psi0(N, F)
    prep = QuantumCircuit(NQ)
    for q in state_prep_ops(psi0, NQ):
        prep.x(q)
    bound = QuantumCircuit(NQ)
    bound.compose(prep, inplace=True)
    bound.compose(ansatz_bound, inplace=True)

    # --- self-test: reconstruct <W> from noiseless shots, compare to E_noiseless
    sim = AerSimulator()
    sim_circuits = build_measurement_circuits(bound, groups, NQ)
    sim_tqc = [transpile(c, sim, optimization_level=1) for c in sim_circuits]
    sim_job = sim.run(sim_tqc, shots=SHOTS_PER_BASIS, seed_simulator=0)
    sim_counts = [sim_job.result().get_counts(i) for i in range(len(sim_tqc))]
    E_selftest = energy_from_counts(sim_counts, groups, SHOTS_PER_BASIS)
    selftest_err = abs(E_selftest - E_noiseless)
    print(f"\nSELF-TEST: reconstructed <W> from noiseless {SHOTS_PER_BASIS}-shot simulation = {E_selftest:.4f}")
    print(f"           vs statevector E_noiseless = {E_noiseless:.4f}  (diff {selftest_err:.4f}, shot noise expected at this scale)")
    if selftest_err > 1.0:
        print("SELF-TEST FAILED (diff too large for shot noise alone) -- STOPPING, not submitting to hardware.")
        return
    print("SELF-TEST PASSED -- classical reconstruction pipeline is trustworthy. Proceeding to hardware.")

    if _os.environ.get("IBM_TASKB_DRY_RUN") == "1":
        print("\nDRY RUN (IBM_TASKB_DRY_RUN=1) -- stopping before hardware submission.")
        return

    # --- real hardware ------------------------------------------------------
    service = QiskitRuntimeService()
    backend = service.backend(BACKEND_NAME)
    hw_circuits = build_measurement_circuits(bound, groups, NQ)
    hw_tqc = [transpile(c, backend=backend, optimization_level=OPT_LEVEL, seed_transpiler=SEED_TRANSPILER)
              for c in hw_circuits]
    ops = hw_tqc[0].count_ops()
    n_cz = ops.get('cz', 0) + ops.get('ecr', 0) + ops.get('cx', 0)
    depth = hw_tqc[0].depth()

    cz_errors = [p.error for p in backend.target['cz'].values() if p is not None and p.error is not None and p.error < 1.0]
    median_cz_error = float(np.median(cz_errors))

    print(f"\nSubmitting 1 job, {n_bases} circuits (one per basis), {SHOTS_PER_BASIS} shots each, to {BACKEND_NAME}...")
    sampler = SamplerV2(backend)
    job = sampler.run(hw_tqc, shots=SHOTS_PER_BASIS)
    job_id = job.job_id()
    print(f"Job ID: {job_id}  -- waiting for result...")
    result = job.result()

    hw_counts = []
    for i in range(len(hw_tqc)):
        creg_name = list(result[i].data.keys())[0]
        counts = getattr(result[i].data, creg_name).get_counts()
        hw_counts.append(counts)

    E_hardware = energy_from_counts(hw_counts, groups, SHOTS_PER_BASIS)

    energies = dict(exact=E_exact, noiseless_statevector=E_noiseless,
                     depolarizing_spam=E_depol, hardware=E_hardware)
    print("\n" + "=" * 70)
    print("FOUR ENERGIES, side by side, relative error vs exact:")
    for label, E in energies.items():
        rel = abs(E - E_exact) / abs(E_exact) * 100
        print(f"  {label:22s}: {E:12.6f}   rel_err={rel:.4f}%")

    out = dict(
        N=N, L=L, x=X, K=K, backend=BACKEND_NAME, job_id=job_id,
        shots_per_basis=SHOTS_PER_BASIS, n_bases=n_bases, n_cz=n_cz, depth=depth,
        median_cz_error_at_runtime=median_cz_error,
        energies=energies,
        raw_counts_per_basis=[{k: int(v) for k, v in c.items()} for c in hw_counts],
        pauli_groups=[list(g.paulis.to_labels()) for g in groups],
    )
    with open("results/ibm_N3_L2_K0.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/ibm_N3_L2_K0.json")


if __name__ == "__main__":
    main()
