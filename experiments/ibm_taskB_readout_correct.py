"""
Task B follow-up, Step 1: readout-correct job daoeuotr85ps73ff6a70's own raw
counts using the FULL (non-symmetric) per-qubit assignment matrices
(prob_meas1_prep0, prob_meas0_prep1 separately), tensored across its six
physical qubits, from the calibration snapshot nearest the job's creation
time. Free -- no new hardware job. Reuses schwinger.core unmodified.

Convention check (matches ibm_taskB_step4_submit.py / the verified job
reconstruction): bitstring position i (0-indexed from the left) is
physics/logical qubit i, which sits on physical qubit
job.inputs['pubs'][k][0].layout.final_index_layout()[i]. For job 1 that
mapping is [8, 9, 10, 11, 18, 31], in that order -- position 0 <-> qubit 8,
..., position 5 <-> qubit 31.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import datetime
import json

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService

import schwinger.core as core

JOB_ID = "daoeuotr85ps73ff6a70"
JOB_CREATION_UTC = "2026-09-21T08:48:35.184380+00:00"
N, L, X, K, F = 3, 2, 16.0, 0.0, 2
NQ = 6
NCX = 20
SPAM_PAPER = 0.005


def build_groups():
    H_full = core.build_H_full(X, K, N=N, F=F)
    paulis = core.decompose_to_paulis(H_full, NQ)
    op = SparsePauliOp.from_list([(label, coeff) for label, coeff, _ in paulis])
    return op.group_commuting(qubit_wise=True)


def energy_from_prob_vector(p_vec, groups, nq):
    """p_vec: length-2^nq array indexed by int(bitstring, 2), bitstring
    position i = physics/logical qubit i (matches decompose_to_paulis's own
    convention, verified in ibm_taskB_step4_submit.py's self-test)."""
    dim = 2 ** nq
    total = 0.0
    for group in groups:
        for label, coeff in zip(group.paulis.to_labels(), group.coeffs):
            coeff = float(np.real(coeff))
            exp = 0.0
            for s in range(dim):
                bitstring = format(s, f'0{nq}b')
                sign = 1
                for i, ch in enumerate(label):
                    if ch != 'I' and bitstring[i] == '1':
                        sign *= -1
                exp += sign * p_vec[s]
            total += coeff * exp
    return total


def counts_to_prob_vector(counts, nq):
    dim = 2 ** nq
    p = np.zeros(dim)
    n_shots = sum(counts.values())
    for bitstring, cnt in counts.items():
        s = int(bitstring, 2)
        p[s] = cnt / n_shots
    return p


def main():
    service = QiskitRuntimeService()
    job = service.job(JOB_ID)
    result = job.result()
    backend = job.backend()

    physical_qubits = job.inputs['pubs'][0][0].layout.final_index_layout()
    print(f"Job {JOB_ID} physical qubits (logical order, position i -> qubit): {physical_qubits}")

    dt = datetime.datetime.fromisoformat(JOB_CREATION_UTC)
    props = backend.properties(datetime=dt)
    used_historical = props is not None
    if props is None:
        props = backend.properties()

    print(f"\nCalibration snapshot: {'historical (nearest job creation)' if used_historical else 'CURRENT (fallback)'}")
    A_list = []
    for i, q in enumerate(physical_qubits):
        p10 = float(props.qubit_property(q)['prob_meas1_prep0'][0])  # P(measure 1 | prepared 0)
        p01 = float(props.qubit_property(q)['prob_meas0_prep1'][0])  # P(measure 0 | prepared 1)
        A = np.array([[1 - p10, p01], [p10, 1 - p01]])  # rows=measured, cols=true
        A_list.append(A)
        print(f"  position {i} (qubit {q}): prob_meas1_prep0={p10:.5f}  prob_meas0_prep1={p01:.5f}  "
              f"symmetric avg={0.5*(p10+p01):.5f}")

    # Full tensored assignment matrix, in the SAME bit order as our count vectors
    # (position 0 = leftmost = first kron factor, matching core.kron_list's convention).
    A_full = A_list[0]
    for A in A_list[1:]:
        A_full = np.kron(A_full, A)
    A_inv = np.linalg.inv(A_full)

    groups = build_groups()

    from scipy.linalg import eigh
    Hsub, basis = core.build_H_subspace(X, K, N=N, F=F)
    w = eigh(Hsub, eigvals_only=True)
    E_exact_val = float(w[0])
    E_mix = float(np.mean(w))

    counts_by_group = []
    for i in range(len(result)):
        creg_name = list(result[i].data.keys())[0]
        counts_by_group.append(getattr(result[i].data, creg_name).get_counts())

    # raw (uncorrected) energy, for cross-check against the saved value
    p_raw_list = [counts_to_prob_vector(c, NQ) for c in counts_by_group]
    E_raw = sum(energy_from_prob_vector(p, [g], NQ) for g, p in zip(groups, p_raw_list))

    # readout-corrected: apply A_inv to each basis's raw distribution
    p_corrected_list = [A_inv @ p for p in p_raw_list]
    neg_mass = [float(np.sum(p[p < 0])) for p in p_corrected_list]
    E_corrected_list = [energy_from_prob_vector(p, [g], NQ) for g, p in zip(groups, p_corrected_list)]
    E_corrected = sum(E_corrected_list)

    print(f"\nE_exact                 = {E_exact_val:.6f}")
    print(f"E_raw (uncorrected)      = {E_raw:.6f}   rel_err={abs(E_raw-E_exact_val)/abs(E_exact_val)*100:.4f}%")
    print(f"E_readout_corrected      = {E_corrected:.6f}   rel_err={abs(E_corrected-E_exact_val)/abs(E_exact_val)*100:.4f}%")
    print(f"(negative probability mass introduced by inversion, per basis): {[f'{n:.4f}' for n in neg_mass]}")

    err_raw_pct = abs(E_raw - E_exact_val) / abs(E_exact_val) * 100
    err_corr_pct = abs(E_corrected - E_exact_val) / abs(E_exact_val) * 100
    points_explained = err_raw_pct - err_corr_pct
    print(f"\nReadout accounts for {points_explained:.2f} percentage points of the {err_raw_pct:.2f}% raw relative error.")

    # effective two-qubit error refit
    def refit_p(E, spam):
        w_eff = (E - E_mix) / (E_exact_val - E_mix)
        w_eff_nospam = w_eff / (1 - 2 * spam)
        if w_eff_nospam <= 0:
            return float('nan')
        return 1 - w_eff_nospam ** (1 / NCX)

    p_raw_fit = refit_p(E_raw, SPAM_PAPER)
    p_corr_fit = refit_p(E_corrected, 0.0)  # readout already corrected -> assume no residual SPAM
    print(f"\nE_mix (sector mean) = {E_mix:.6f}")
    print(f"Effective per-CZ error refit from E_raw (assuming paper's eps={SPAM_PAPER}):        {p_raw_fit*100:.4f}%")
    print(f"Effective per-CZ error refit from E_readout_corrected (eps=0, already corrected): {p_corr_fit*100:.4f}%")
    print("(reference: unmitigated refit reported earlier as 1.03%)")

    out = dict(job_id=JOB_ID, physical_qubits=physical_qubits,
               readout_probs=[dict(qubit=q, prob_meas1_prep0=float(props.qubit_property(q)['prob_meas1_prep0'][0]),
                                    prob_meas0_prep1=float(props.qubit_property(q)['prob_meas0_prep1'][0]))
                               for q in physical_qubits],
               E_exact=E_exact_val, E_mix=E_mix, E_raw=E_raw, E_readout_corrected=E_corrected,
               rel_err_raw_pct=err_raw_pct, rel_err_corrected_pct=err_corr_pct,
               readout_points_explained=points_explained,
               p_eff_raw_pct=p_raw_fit * 100, p_eff_readout_corrected_pct=p_corr_fit * 100,
               negative_prob_mass_per_basis=neg_mass, used_historical_calibration=used_historical)
    with open("results/ibm_N3_L2_K0_readout_corrected.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/ibm_N3_L2_K0_readout_corrected.json")


if __name__ == "__main__":
    main()
