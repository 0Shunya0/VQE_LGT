"""
Production noisy-circuit backend for train-noisy VQE, built on
qiskit-aer's AerSimulator(method="density_matrix").

Benchmarked against a hand-rolled NumPy density-matrix propagator
(noisy_sim.py) on this machine (24 cores, numpy 2.3 / qiskit-aer 0.17):

    nq=6 (N=3,L=2, n_CX=20): hand-rolled  8.8 ms/eval   vs aer  4.7-8.4 ms/eval
    nq=8 (N=4,L=2, n_CX=28): hand-rolled 351   ms/eval   vs aer  8-13   ms/eval

qiskit-aer wins clearly at both sizes actually run in this study (nq up to 8),
by ~2x at nq=6 and ~30-40x at nq=8 (the hand-rolled path is einsum/reshape-
overhead bound, not FLOP bound, at this scale) -- confirmed even in the
realistic "transpile once, assign_parameters + run per COBYLA iteration"
pattern used in production, not just a repeated-fixed-circuit microbenchmark.
So AerSimulator(method="density_matrix") is the evaluator used for every
optimization in this study; noisy_sim.py's hand-rolled propagator (itself
cross-validated against Aer to <1e-15) served as the independent physics
reference that this backend's circuits were checked against before being
trusted for production (see test_noisy_sim.py and test_qiskit_backend.py).

Every 2-qubit interaction is literal hardware gates: the gauge-invariant
ansatz's fused Uxy(theta) is built from its exact 2-CNOT decomposition
(the same one verified in noisy_sim.uxy_ops against Qiskit's own
XXPlusYYGate definition), and the hardware-efficient ansatz's CNOT ladder is
literal. A NoiseModel attaches depolarizing_error(p, 2) to every 'cx'; the
per-qubit SPAM bit-flip is appended directly to the transpiled circuit as a
literal QuantumError instruction on every qubit immediately before readout
(not routed through the NoiseModel on a placeholder gate name -- an early
version used 'id' gates for this and the transpiler silently dropped them,
which would have silently dropped the SPAM channel too; see
_bitflip_instruction below). So the optimizer's cost function is Tr(W rho) on
the actual noisy circuit output, not a post-hoc analytic contraction.
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import SGate, SdgGate, SXGate, SXdgGate, RYGate, RZGate
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error

import schwinger.core as core


def qk(q, nq):
    """Our qubit index q (MSB-first) -> qiskit register index (LSB-first qubit 0)."""
    return nq - 1 - q


# ===== circuit builders (native gates, Qiskit Parameters) ===================

def _uxy_block(qc, params, off, i, p1, p2, nq):
    """One Uxy(theta) via its exact 2-CNOT decomposition (see noisy_sim.uxy_ops
    for the derivation and the machine-precision check against XXPlusYYGate)."""
    a, b = qk(p1, nq), qk(p2, nq)
    th = params[off + i]
    qc.append(SdgGate(), [a]); qc.append(SXGate(), [a]); qc.append(SGate(), [a]); qc.append(SGate(), [b])
    qc.cx(a, b)
    qc.append(RYGate(-th / 2), [a]); qc.append(RYGate(-th / 2), [b])
    qc.cx(a, b)
    qc.append(SdgGate(), [b]); qc.append(SdgGate(), [a]); qc.append(SXdgGate(), [a]); qc.append(SGate(), [a])


def build_gi_circuit(nq, L):
    """Gauge-invariant ansatz. Returns (QuantumCircuit, params list, n_CX)."""
    n_per = (nq - 1) + nq
    n_p = L * n_per
    params = [Parameter(f'th{i}') for i in range(n_p)]
    qc = QuantumCircuit(nq)
    for l in range(L):
        off = l * n_per
        for q in range(nq - 1):
            _uxy_block(qc, params, off, q, q, q + 1, nq)
        for q in range(nq):
            qc.append(RZGate(params[off + (nq - 1) + q]), [qk(q, nq)])
    n_cx = 2 * L * (nq - 1)
    return qc, params, n_cx


def build_hw_circuit(nq, L):
    """Hardware-efficient ansatz. Returns (QuantumCircuit, params list, n_CX)."""
    n_per = 2 * nq
    n_p = L * n_per
    params = [Parameter(f'th{i}') for i in range(n_p)]
    qc = QuantumCircuit(nq)
    for l in range(L):
        off = l * n_per
        for q in range(nq):
            qc.append(RYGate(params[off + 2 * q]), [qk(q, nq)])
            qc.append(RZGate(params[off + 2 * q + 1]), [qk(q, nq)])
        for q in range(nq - 1):
            qc.cx(qk(q, nq), qk(q + 1, nq))
    n_cx = L * (nq - 1)
    return qc, params, n_cx


def state_prep_ops(psi0, nq):
    """X-gate prep for the computational-basis reference state (schwinger_core.make_psi0)."""
    idx = int(np.argmax(np.abs(psi0)))
    bits = format(idx, f'0{nq}b')
    return [qk(a, nq) for a, b in enumerate(bits) if b == '1']


def fold_circuit(qc, base_ncx, lam):
    """
    Per-gate CNOT folding for ZNE: replaces the first n_fold CNOTs (circuit
    order) with cx.cx.cx (=cx, since CNOT is self-inverse; each replacement
    leaves the ideal unitary unchanged but adds 2 extra noisy CNOTs), and
    leaves the remaining CNOTs and all single-qubit gates untouched.
    n_fold = base_ncx*(lam-1)/2 so that total n_CX = base_ncx + 2*n_fold = lam*base_ncx
    exactly, for lam in {1, 2, 3} (base_ncx = 2*L*(nq-1) is always even).
    Chosen over whole-circuit folding (qc . qc^-1 . qc) because it generalizes
    to lam=2 (needed for the 3-point Richardson exponential fit) without
    fractional-gate tricks; for lam=3 it triples n_CX exactly like whole-
    circuit folding does, and since only CX gates carry noise here the two
    schemes are physically equivalent for this study.
    """
    if lam not in (1, 2, 3):
        raise ValueError("only lambda in {1,2,3} implemented")
    n_fold = base_ncx * (lam - 1) // 2
    if lam == 1:
        return qc
    folded = QuantumCircuit(qc.num_qubits)
    folded_count = 0
    for instr in qc.data:
        if instr.operation.name == 'cx' and folded_count < n_fold:
            for _ in range(3):
                folded.append(instr.operation, instr.qubits, instr.clbits)
            folded_count += 1
        else:
            folded.append(instr.operation, instr.qubits, instr.clbits)
    return folded


# ===== noise model ============================================================

def build_noise_model(p):
    """CNOT depolarizing only. The SPAM bit-flip is NOT routed through the
    NoiseModel (see note below) -- it is appended directly to the circuit."""
    nm = NoiseModel()
    if p > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p, 2), ['cx'])
    return nm


def _bitflip_instruction(eps):
    """
    A per-qubit bit-flip channel as a literal circuit instruction rather than
    a NoiseModel entry keyed on a gate name: qc.id(q) is transparently dropped
    by the transpiler (confirmed empirically -- an 'id'-keyed NoiseModel entry
    would silently vanish along with it), so the SPAM channel is attached
    directly to the circuit via QuantumError.to_instruction(), which survives
    transpilation because it IS the instruction, not something matched by name.
    """
    return pauli_error([('X', eps), ('I', 1 - eps)]).to_instruction()


# ===== evaluator ==============================================================

class NoisyEvaluator:
    """
    Builds and transpiles a noisy density-matrix circuit ONCE for a given
    (N, F, L, ansatz, fold_lambda, p, eps); evaluate(theta, H) then just binds
    parameters and runs -- the fast path benchmarked above.
    """

    def __init__(self, N, F, L, ansatz='gi', fold_lambda=1, p=0.0, eps=0.0, x=16.0):
        self.N, self.F, self.L = N, F, L
        self.nq = N * F
        self.ansatz = ansatz
        self.x = x
        builder = build_gi_circuit if ansatz == 'gi' else build_hw_circuit
        base_qc, self.params, base_ncx = builder(self.nq, L)
        self.n_cx = base_ncx * fold_lambda
        self.fold_lambda = fold_lambda

        psi0 = core.make_psi0(N, F)
        self.psi0 = psi0
        prep = QuantumCircuit(self.nq)
        for q in state_prep_ops(psi0, self.nq):
            prep.x(q)

        full = QuantumCircuit(self.nq)
        full.compose(prep, inplace=True)
        full.compose(base_qc, inplace=True)

        self.noise_model = build_noise_model(p)
        self.sim = AerSimulator(method='density_matrix', noise_model=self.noise_model)
        # Transpile the UNFOLDED circuit first, then fold, then append SPAM:
        # folding before transpile is silently undone by InverseCancellation/
        # commutative-cancellation optimization passes, which collapse a
        # cx.cx.cx run right back down to a single cx (confirmed empirically
        # -- n_CX after transpile stayed 20 for every fold_lambda until this
        # ordering was fixed). Folding the already-transpiled circuit means
        # there is no later optimization pass to undo it.
        tqc = transpile(full, self.sim, optimization_level=1)
        if fold_lambda != 1:
            tqc = fold_circuit(tqc, base_ncx, fold_lambda)
        if eps > 0:
            bitflip = _bitflip_instruction(eps)
            for q in range(self.nq):
                tqc.append(bitflip, [q])
        self.tqc = tqc
        self.n_params = len(self.params)

    def _bound_circuit_with_W(self, theta, W_matrix):
        bound = self.tqc.assign_parameters({self.params[i]: theta[i] for i in range(self.n_params)})
        bound.save_expectation_value(Operator(W_matrix), list(range(self.nq)), label='W')
        return bound

    def energy(self, theta, K):
        """<W> = Tr(W rho) for the given parameters and chemical potential K."""
        H = core.build_H_full(self.x, K, N=self.N, F=self.F)
        bound = self._bound_circuit_with_W(theta, H)
        res = self.sim.run(bound).result()
        return float(np.real(res.data(0)['W']))

    def energy_and_state(self, theta, K):
        """<W> plus the noisy density matrix (for fidelity / <Q_tot^2> diagnostics)."""
        H = core.build_H_full(self.x, K, N=self.N, F=self.F)
        bound = self.tqc.assign_parameters({self.params[i]: theta[i] for i in range(self.n_params)})
        bound.save_expectation_value(Operator(H), list(range(self.nq)), label='W')
        bound.save_density_matrix(label='rho')
        res = self.sim.run(bound).result()
        data = res.data(0)
        return float(np.real(data['W'])), np.asarray(data['rho'])

    def cost_fn(self, K):
        """Return a fast theta -> float closure for a fixed K (reuses H)."""
        H = core.build_H_full(self.x, K, N=self.N, F=self.F)
        Wop = Operator(H)

        def cost(theta):
            bound = self.tqc.assign_parameters({self.params[i]: theta[i] for i in range(self.n_params)})
            bound.save_expectation_value(Wop, list(range(self.nq)), label='W')
            res = self.sim.run(bound).result()
            return float(np.real(res.data(0)['W']))

        return cost
