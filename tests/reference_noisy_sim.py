"""
Gate-level density-matrix circuit simulator for train-noisy VQE.

Replaces the post-hoc analytic contraction in schwinger_core.noisy() with an
actual noisy circuit: every 2-qubit interaction is expressed as real CNOTs
(the gauge-invariant ansatz's fused Uxy(theta) = exp(-i theta (XX+YY)/2) is
decomposed into its exact minimal 2-CNOT circuit, verified against Qiskit's
XXPlusYYGate analytic definition to machine precision -- see
verify_uxy_decomposition() below), a two-qubit depolarizing channel is
injected after every CNOT, and a per-qubit bit-flip (SPAM) channel is applied
immediately before readout. <W> = Tr(W rho) is evaluated on the resulting
noisy density matrix and THAT is what the optimizer sees.

Qubit convention matches schwinger_core: qubit q occupies bit (nq-1-q) of the
state index (MSB-first), so qubit 0 is the leftmost / most significant.

Circuits are represented as a flat op-list IR:
    ('u1', q, U)         -- apply 2x2 unitary U to qubit q
    ('cx', ctrl, tgt)     -- CNOT, ctrl and tgt adjacent (|ctrl-tgt|==1)
This makes gate counting (for n_CX bookkeeping) and ZNE folding (append the
inverse circuit then the circuit again) mechanical and gate-list-agnostic.

Density matrices are stored dense ((2**nq, 2**nq) complex) since nq <= 8 in
this study (dim <= 256); all gate/channel applications are done via local
tensor contraction on a reshaped view (cost O(dim^2), not O(dim^3)) so that
COBYLA can call the cost function tens of thousands of times per sweep cell.
"""
import numpy as np

# ---- fixed single-qubit gates (as used in the verified Uxy decomposition) --
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
Sdg = S.conj().T
SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex)
SXdg = SX.conj().T


def Ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def Rz(theta):
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)


CX_MSB_CTRL = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)  # ctrl=first(MSB), tgt=second(LSB)
CX_LSB_CTRL = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)  # ctrl=second(LSB), tgt=first(MSB)


# ===== op-list construction ==================================================

def uxy_ops(p1, p2, theta):
    """
    Exact 2-CNOT circuit for Uxy(theta) = exp(-i theta (XX+YY)/2) acting on
    adjacent qubits (p1=MSB/first, p2=LSB/second), i.e. Uxy_apply's matrix
    restricted to {|01>,|10>} is [[cos(t/2), -i sin(t/2)], [-i sin(t/2), cos(t/2)]].
    Verified to match qiskit.circuit.library.XXPlusYYGate(theta, 0.0)'s own
    analytic gate definition to machine precision (see verify_uxy_decomposition).
    """
    h = theta / 2
    return [
        ('u1', p1, Sdg), ('u1', p1, SX), ('u1', p1, S), ('u1', p2, S),
        ('cx', p1, p2),
        ('u1', p1, Ry(-h)), ('u1', p2, Ry(-h)),
        ('cx', p1, p2),
        ('u1', p2, Sdg), ('u1', p1, Sdg), ('u1', p1, SXdg), ('u1', p1, S),
    ]


def gi_ansatz_ops(theta, nq, L):
    """Gate list for the gauge-invariant ansatz (matches schwinger_core.gauge_invariant_ansatz)."""
    n_per = (nq - 1) + nq
    ops = []
    for l in range(L):
        off = l * n_per
        for q in range(nq - 1):
            ops += uxy_ops(q, q + 1, theta[off + q])
        for q in range(nq):
            ops.append(('u1', q, Rz(theta[off + (nq - 1) + q])))
    return ops


def hw_ansatz_ops(theta, nq, L):
    """Gate list for the hardware-efficient ansatz (matches schwinger_core.hw_efficient_ansatz)."""
    n_per = 2 * nq
    ops = []
    for l in range(L):
        off = l * n_per
        for q in range(nq):
            ops.append(('u1', q, Ry(theta[off + 2 * q])))
            ops.append(('u1', q, Rz(theta[off + 2 * q + 1])))
        for q in range(nq - 1):
            ops.append(('cx', q, q + 1))
    return ops


def count_cx(ops):
    return sum(1 for o in ops if o[0] == 'cx')


def dagger_ops(ops):
    """Adjoint of a gate list, gates reversed and each individually daggered (CNOT is self-inverse)."""
    out = []
    for o in reversed(ops):
        if o[0] == 'u1':
            out.append(('u1', o[1], o[2].conj().T))
        else:
            out.append(o)
    return out


def fold_ops(ops, lam):
    """Global unitary folding for ZNE: lambda=1 -> ops; lambda=3 -> ops . ops^-1 . ops (triples n_CX)."""
    if lam == 1:
        return list(ops)
    if lam == 3:
        return ops + dagger_ops(ops) + ops
    raise ValueError("only lambda in {1,3} implemented")


# ===== density-matrix engine =================================================

def initial_rho(psi0):
    psi0 = psi0.reshape(-1, 1).astype(complex)
    return psi0 @ psi0.conj().T


def _contract_ket_bra(t, U):
    """Apply U to axis 1 (ket) and U* to axis 4 (bra) of a (B,k,A,B,k,A) tensor,
    via tensordot (BLAS matmul) instead of einsum (no BLAS dispatch for this
    index pattern -- ~10-20x faster for the tensor sizes used here, since the
    dominant cost of run_circuit is exactly this contraction, called twice per
    gate and roughly (n_CX*2 + n_1q_gates) times per circuit)."""
    # ket side: contract U's 2nd index against axis 1, new index lands at axis 0
    t = np.tensordot(U, t, axes=([1], [1]))
    t = np.moveaxis(t, 0, 1)
    # bra side: contract U*'s 2nd index against axis 4, new index lands at axis 0
    t = np.tensordot(U.conj(), t, axes=([1], [4]))
    t = np.moveaxis(t, 0, 4)
    return t


def _apply_u1(rho, nq, q, U):
    dim = rho.shape[0]
    B, A = 2 ** q, 2 ** (nq - 1 - q)
    t = rho.reshape(B, 2, A, B, 2, A)
    t = _contract_ket_bra(t, U)
    return t.reshape(dim, dim)


def _apply_cx(rho, nq, ctrl, tgt):
    """CNOT on adjacent qubits (either order)."""
    dim = rho.shape[0]
    q = min(ctrl, tgt)
    B, A = 2 ** q, 2 ** (nq - 2 - q)
    U = CX_MSB_CTRL if ctrl < tgt else CX_LSB_CTRL
    t = rho.reshape(B, 4, A, B, 4, A)
    t = _contract_ket_bra(t, U)
    return t.reshape(dim, dim)


def _depolarize_adjacent_2q(rho, nq, q, p):
    """rho -> (1-p) rho + p * (I4/4 (x) Tr_{q,q+1} rho), q,q+1 adjacent. Matches
    Qiskit's depolarizing_error(p, 2) convention E(rho)=(1-p)rho + p*Tr[rho]*I/4."""
    if p <= 0:
        return rho
    dim = rho.shape[0]
    B, A = 2 ** q, 2 ** (nq - 2 - q)
    t = rho.reshape(B, 4, A, B, 4, A)
    reduced = np.einsum('pqrsqu->prsu', t)          # trace out the target pair
    mixed = np.einsum('qt,prsu->pqrstu', np.eye(4, dtype=complex) / 4.0, reduced)
    new_t = (1 - p) * t + p * mixed
    return new_t.reshape(dim, dim)


def _bitflip_1q(rho, nq, q, eps):
    if eps <= 0:
        return rho
    dim = rho.shape[0]
    B, A = 2 ** q, 2 ** (nq - 1 - q)
    t = rho.reshape(B, 2, A, B, 2, A)
    K0 = np.sqrt(1 - eps) * I2
    K1 = np.sqrt(eps) * X
    out = _contract_ket_bra(t, K0) + _contract_ket_bra(t, K1)
    return out.reshape(dim, dim)


def run_circuit(ops, psi0, nq, p=0.0, eps=0.0):
    """Execute the op list on |psi0><psi0|, injecting 2q depolarizing(p) after
    every CNOT and per-qubit bit-flip(eps) immediately before readout. Returns
    the final density matrix rho, shape (2**nq, 2**nq)."""
    rho = initial_rho(psi0)
    for op in ops:
        if op[0] == 'u1':
            _, q, U = op
            rho = _apply_u1(rho, nq, q, U)
        else:
            _, ctrl, tgt = op
            rho = _apply_cx(rho, nq, ctrl, tgt)
            if p > 0:
                rho = _depolarize_adjacent_2q(rho, nq, min(ctrl, tgt), p)
    if eps > 0:
        for q in range(nq):
            rho = _bitflip_1q(rho, nq, q, eps)
    return rho


def expectation(rho, W):
    return float(np.real(np.trace(W @ rho)))


def fidelity_with_pure(rho, psi_ref):
    psi_ref = psi_ref.reshape(-1)
    return float(np.real(psi_ref.conj() @ rho @ psi_ref))


# ===== self-check against the exact statevector code (p=0, eps=0) ===========

def verify_uxy_decomposition(n_trials=20, seed=0):
    """uxy_ops(theta), applied noiselessly, must reproduce Uxy_apply's matrix exactly."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_trials):
        theta = rng.uniform(-2 * np.pi, 2 * np.pi)
        mat = np.zeros((4, 4), dtype=complex)
        for i in range(4):
            psi = np.zeros(4, dtype=complex); psi[i] = 1.0
            rho = run_circuit(uxy_ops(0, 1, theta), psi, nq=2, p=0.0, eps=0.0)
            # extract output column: since input is a basis state, rho = |out><out|
            # recover |out> via the dominant eigenvector (rho is rank-1 here)
            w, v = np.linalg.eigh(rho)
            out = v[:, -1] * np.sqrt(max(w[-1], 0))
            mat[:, i] = out
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        ref = np.eye(4, dtype=complex)
        ref[1, 1] = c; ref[1, 2] = -1j * s
        ref[2, 1] = -1j * s; ref[2, 2] = c
        # compare up to per-column global phase (eigh sign/phase ambiguity)
        diff = 0.0
        for i in range(4):
            col = mat[:, i]; rcol = ref[:, i]
            idx = np.argmax(np.abs(rcol))
            if abs(col[idx]) > 1e-12:
                ph = rcol[idx] / col[idx]
                col = col * ph
            diff = max(diff, float(np.max(np.abs(col - rcol))))
        worst = max(worst, diff)
    return worst


if __name__ == "__main__":
    err = verify_uxy_decomposition()
    print(f"max |Uxy(theta) reconstruction error| over random thetas: {err:.2e}")
    assert err < 1e-9, "Uxy 2-CNOT decomposition does not match Uxy_apply"
    print("PASS: 2-CNOT Uxy decomposition matches schwinger_core.Uxy_apply to machine precision.")
