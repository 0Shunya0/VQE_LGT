"""
Cross-validation of noisy_sim.py:
  1. p=0, eps=0: run_circuit's final rho must match |psi><psi| from the
     existing exact statevector ansatz functions in schwinger_core.
  2. p>0: run_circuit's final rho must match qiskit-aer's density_matrix
     simulator under an equivalent NoiseModel (depolarizing_error(p,2) on cx,
     ReadoutError-equivalent bit-flip before measurement), confirming both
     the CNOT decomposition and the hand-rolled depolarizing channel are
     physically correct and match Qiskit's convention.
Also benchmarks hand-rolled vs qiskit-aer per-circuit-eval speed so the
faster backend can be picked for the production sweeps.
"""
import time
import numpy as np

import schwinger_core as core
import noisy_sim as ns


def statevector_check():
    print("=== 1. Noiseless statevector cross-check (p=0, eps=0) ===")
    rng = np.random.default_rng(0)
    for N in (2, 3):
        F, nq = 2, N * 2
        psi0 = core.make_psi0(N, F)
        n_p_gi = 2 * core.gi_params_per_layer(N, F)
        theta = rng.uniform(-np.pi, np.pi, n_p_gi)
        psi_exact = core.gauge_invariant_ansatz(theta, psi0, nq, 2)
        ops = ns.gi_ansatz_ops(theta, nq, 2)
        rho = ns.run_circuit(ops, psi0, nq, p=0.0, eps=0.0)
        rho_exact = np.outer(psi_exact, psi_exact.conj())
        err = np.max(np.abs(rho - rho_exact))
        print(f"  GI  N={N} nq={nq} n_CX={ns.count_cx(ops)} max|rho-rho_exact|={err:.2e}")
        assert err < 1e-9

        n_p_hw = 2 * (2 * nq)
        theta = rng.uniform(-np.pi, np.pi, n_p_hw)
        psi_exact = core.hw_efficient_ansatz(theta, psi0, nq, 2)
        ops = ns.hw_ansatz_ops(theta, nq, 2)
        rho = ns.run_circuit(ops, psi0, nq, p=0.0, eps=0.0)
        rho_exact = np.outer(psi_exact, psi_exact.conj())
        err = np.max(np.abs(rho - rho_exact))
        print(f"  HW  N={N} nq={nq} n_CX={ns.count_cx(ops)} max|rho-rho_exact|={err:.2e}")
        assert err < 1e-9
    print("  PASS\n")


def aer_check():
    print("=== 2. Noisy density-matrix cross-check vs qiskit-aer ===")
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    rng = np.random.default_rng(1)
    N, F, nq, L = 3, 2, 6, 2
    psi0 = core.make_psi0(N, F)
    n_p = L * core.gi_params_per_layer(N, F)
    theta = rng.uniform(-np.pi, np.pi, n_p)
    ops = ns.gi_ansatz_ops(theta, nq, L)
    p, eps = 0.02, 0.005

    rho_mine = ns.run_circuit(ops, psi0, nq, p=p, eps=0.0)

    # Build the equivalent Qiskit circuit. Qiskit's qubit 0 is the LSB
    # (rightmost) qubit; our convention is qubit q = bit (nq-1-q), i.e. our
    # qubit 0 is the MSB. Map our qubit index a -> qiskit qubit (nq-1-a) so
    # that state labelling agrees.
    def qk(q):
        return nq - 1 - q

    qc = QuantumCircuit(nq)
    # prepare psi0 (a computational basis state)
    idx = int(np.argmax(np.abs(psi0)))
    bits = format(idx, f'0{nq}b')  # bits[a] is our qubit a's value (MSB-first)
    for a, b in enumerate(bits):
        if b == '1':
            qc.x(qk(a))
    for op in ops:
        if op[0] == 'u1':
            _, q, U = op
            qc.append(UnitaryGate(U), [qk(q)])
        else:
            _, ctrl, tgt = op
            qc.cx(qk(ctrl), qk(tgt))
    qc.save_density_matrix()

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p, 2), ['cx'])
    sim = AerSimulator(method='density_matrix', noise_model=noise_model)
    result = sim.run(qc).result()
    rho_qiskit_dm = np.asarray(result.data(0)['density_matrix'])

    # Because our qubit a is placed on qiskit register index qk(a)=nq-1-a, and
    # qiskit's flat index gives register c weight 2**c, our qubit a contributes
    # weight 2**(nq-1-a) to the qiskit index -- exactly the weight it already
    # has in our own MSB-first index convention. So the flat array index is
    # literally identical between the two conventions; no permutation needed.
    rho_qiskit = rho_qiskit_dm

    err = np.max(np.abs(rho_mine - rho_qiskit))
    print(f"  after cx-depolarizing only, max|rho_mine-rho_qiskit| = {err:.2e}")
    assert err < 1e-8, "depolarizing channel mismatch vs qiskit-aer"

    # now also cross-check the bit-flip SPAM channel by applying it to both
    rho_mine_full = ns.run_circuit(ops, psi0, nq, p=p, eps=eps)
    from qiskit_aer.noise import pauli_error
    noise_model2 = NoiseModel()
    noise_model2.add_all_qubit_quantum_error(depolarizing_error(p, 2), ['cx'])
    # Apply the SPAM bit-flip as a quantum error (X with prob eps) attached to
    # an explicit id gate inserted right before save_density_matrix, since
    # ReadoutError only affects classical measurement outcomes, not the saved
    # density matrix.
    bitflip = pauli_error([('X', eps), ('I', 1 - eps)])
    noise_model2.add_all_qubit_quantum_error(bitflip, ['id'])
    qc2 = QuantumCircuit(nq)
    for a, b in enumerate(bits):
        if b == '1':
            qc2.x(qk(a))
    for op in ops:
        if op[0] == 'u1':
            _, q, U = op
            qc2.append(UnitaryGate(U), [qk(q)])
        else:
            _, ctrl, tgt = op
            qc2.cx(qk(ctrl), qk(tgt))
    for a in range(nq):
        qc2.id(qk(a))
    qc2.save_density_matrix()
    sim2 = AerSimulator(method='density_matrix', noise_model=noise_model2)
    result2 = sim2.run(qc2).result()
    rho_qiskit_dm2 = np.asarray(result2.data(0)['density_matrix'])
    rho_qiskit2 = rho_qiskit_dm2
    err2 = np.max(np.abs(rho_mine_full - rho_qiskit2))
    print(f"  after cx-depolarizing + SPAM bit-flip, max|rho_mine-rho_qiskit| = {err2:.2e}")
    assert err2 < 1e-8, "SPAM bit-flip channel mismatch vs qiskit-aer"
    print("  PASS\n")
    return qc, noise_model


def benchmark():
    print("=== 3. Speed benchmark: hand-rolled numpy vs qiskit-aer ===")
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    rng = np.random.default_rng(2)
    for N in (3, 4):
        F, nq, L = 2, N * 2, 2
        psi0 = core.make_psi0(N, F)
        n_p = L * core.gi_params_per_layer(N, F)
        theta = rng.uniform(-np.pi, np.pi, n_p)
        ops = ns.gi_ansatz_ops(theta, nq, L)
        p, eps = 0.01, 0.005

        n_rep = 20
        t0 = time.time()
        for _ in range(n_rep):
            rho = ns.run_circuit(ops, psi0, nq, p=p, eps=eps)
        t_mine = (time.time() - t0) / n_rep

        def qk(q):
            return nq - 1 - q
        idx = int(np.argmax(np.abs(psi0)))
        bits = format(idx, f'0{nq}b')
        qc = QuantumCircuit(nq)
        for a, b in enumerate(bits):
            if b == '1':
                qc.x(qk(a))
        for op in ops:
            if op[0] == 'u1':
                _, q, U = op
                qc.append(UnitaryGate(U), [qk(q)])
            else:
                _, ctrl, tgt = op
                qc.cx(qk(ctrl), qk(tgt))
        qc.save_density_matrix()
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(depolarizing_error(p, 2), ['cx'])
        sim = AerSimulator(method='density_matrix', noise_model=noise_model)
        t0 = time.time()
        for _ in range(n_rep):
            result = sim.run(qc).result()
        t_aer = (time.time() - t0) / n_rep

        print(f"  N={N} nq={nq} n_CX={ns.count_cx(ops)}: "
              f"hand-rolled {t_mine * 1000:.2f} ms/eval, qiskit-aer {t_aer * 1000:.2f} ms/eval, "
              f"speedup {t_aer / t_mine:.1f}x")


if __name__ == "__main__":
    statevector_check()
    aer_check()
    benchmark()
