"""
Verbatim extraction of cell 3 ("Core code (validated)") from schwinger_vqe_paper_final.ipynb.
Not reimplemented -- copied as-is so every downstream script reuses the exact
Hamiltonian / ansatz / observable / VQE / noise code the paper's noiseless
results were produced with. Only these top imports were added since the
notebook relies on its own cell 1 for them.
"""
import numpy as np
from math import comb
from scipy.linalg import eigh
from scipy.optimize import minimize

# ===== hamiltonian.py =====
"""
Kogut-Susskind -> Jordan-Wigner spin Hamiltonian for the two-flavor Schwinger
model, in the Q_tot = 0 physical sector.

Two chemical-potential conventions are supported:

    'nu0only'  (DEFAULT, the manuscript convention)
        nu_0 = 2 sqrt(x) K   on flavor 0,   nu_1 = 0.
        Reproduces the trapped-ion benchmark of Melzer et al. (2025):
            E0(N=2) = -223.0, -30.5644, +1.0  at K = -14, 0, +10.
        N=3 first-order phase boundary at |K| ~ 5.6.

    'antisym'  (isospin chemical potential, NOT used in the paper)
        nu_0 = +2 sqrt(x) K,  nu_1 = -2 sqrt(x) K.
        N=3 boundary at |K| ~ 2.8.  Provided only for comparison.

Two well-understood bugs are fixed here and must stay fixed:

  (1) Kinetic doubling.  The hop must iterate over |01> states only
      (b1 == 0 and b2 == 1) and set H[i,j] and H[j,i] together.  Using the
      condition b1 != b2 double-counts every hop and doubles the kinetic energy.

  (2) Staggered charge background.  The Kogut-Susskind charge is
          Q_n = sum_f occ_{n,f} - (F/2)(1 - (-1)^n).
      Dropping the background term -(F/2)(1-(-1)^n) lowers the K=0 energies by
      ~1-2 units relative to Table I.

Qubit ordering note: build_H_full / build_H_subspace use the convention that
qubit p = n*F + f occupies bit (nq-1-p) of the integer state label (MSB-first,
matching the gate primitives in ansatz.py).  build_hamiltonian (the basis-indexed
builder used by the noise module) uses LSB-first labelling; the two never share
state integers, so this is internal-only.
"""



CONVENTION_DEFAULT = "nu0only"

# --- Pauli matrices / tensor infrastructure (MSB-first qubit ordering) -------

_I2 = np.eye(2, dtype=complex)
_X2 = np.array([[0, 1], [1, 0]], dtype=complex)
_Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z2 = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_list(ops):
    r = ops[0]
    for op in ops[1:]:
        r = np.kron(r, op)
    return r


def _single_Z(q, nq):
    ops = [_I2] * nq
    ops[q] = _Z2
    return kron_list(ops)


def sector_dim(N):
    """Dimension of the Q_tot = 0 sector, C(2N, N)."""
    return comb(2 * N, N)


def nu_of(K, x, convention=CONVENTION_DEFAULT):
    """Return (nu_0, nu_1) for a given convention."""
    a = 2.0 * np.sqrt(x) * K
    if convention == "nu0only":
        return a, 0.0
    if convention == "antisym":
        return a, -a
    raise ValueError("convention must be 'nu0only' or 'antisym'")


# --- Full-space (2^{2N} x 2^{2N}) operator-form Hamiltonian ------------------

def build_H_full(x, K, N=2, F=2, convention=CONVENTION_DEFAULT):
    """
    Dense Hamiltonian on the full 2^{2N} Hilbert space (operator form).

    Used for the gauge-invariant vs hardware-efficient comparison, where the
    unconstrained ansatz must be able to leave the physical sector.
    """
    nu0, nu1 = nu_of(K, x, convention)
    nu = [nu0, nu1]
    nq = N * F
    dim = 2 ** nq
    H = np.zeros((dim, dim), dtype=complex)
    Imat = np.eye(dim, dtype=complex)

    # kinetic hop
    for n in range(N - 1):
        for f in range(F):
            p1, p2 = n * F + f, (n + 1) * F + f
            z_str = list(range(p1 + 1, p2))
            for G in (_X2, _Y2):
                ops = [_I2] * nq
                ops[p1] = G
                ops[p2] = G
                for qz in z_str:
                    ops[qz] = _Z2
                H += -x / 2 * kron_list(ops)

    # chemical potential
    for n in range(N):
        for f in range(F):
            q = n * F + f
            H += nu[f] / 2 * (_single_Z(q, nq) + Imat)

    # electric field: sum_{ne} (sum_{k<=ne} Q_k)^2, with charge background
    for ne in range(N - 1):
        cumQ = np.zeros((dim, dim), dtype=complex)
        for k in range(ne + 1):
            for f in range(F):
                q = k * F + f
                cumQ += (Imat - _single_Z(q, nq)) / 2
            cumQ -= F / 2.0 * (1 - (-1) ** k) * Imat
        H += cumQ @ cumQ
    return H


# --- Physical-sector (C(2N,N) x C(2N,N)) Hamiltonian, fast ------------------

def physical_basis(N, F=2):
    """Integer labels of the Q_tot=0 sector (MSB-first), and a reverse index."""
    nq = N * F
    basis = [s for s in range(2 ** nq)
             if sum(1 - 2 * ((s >> (nq - 1 - q)) & 1) for q in range(nq)) == 0]
    return basis, {s: i for i, s in enumerate(basis)}


def build_H_subspace(x, K, N=2, F=2, convention=CONVENTION_DEFAULT):
    """
    Hamiltonian projected onto the Q_tot = 0 sector (the workhorse for scaling).

    Returns (H, basis) with H of shape (C(2N,N), C(2N,N)).  Bug-fixes (1) and (2)
    above are applied.
    """
    nu0, nu1 = nu_of(K, x, convention)
    nu = [nu0, nu1]
    nq = N * F
    basis, basis_idx = physical_basis(N, F)
    H = np.zeros((len(basis), len(basis)), dtype=complex)

    # kinetic: |01> -> |10> only, set both H[i,j] and H[j,i] (fix #1)
    for n in range(N - 1):
        for f in range(F):
            p1, p2 = n * F + f, (n + 1) * F + f
            z_str = list(range(p1 + 1, p2))
            for i, si in enumerate(basis):
                b1 = (si >> (nq - 1 - p1)) & 1
                b2 = (si >> (nq - 1 - p2)) & 1
                if b1 == 0 and b2 == 1:
                    z_sign = 1
                    for qz in z_str:
                        z_sign *= (1 - 2 * ((si >> (nq - 1 - qz)) & 1))
                    sj = si ^ (1 << (nq - 1 - p1)) ^ (1 << (nq - 1 - p2))
                    if sj in basis_idx:
                        j = basis_idx[sj]
                        H[i, j] += -x * z_sign
                        H[j, i] += -x * z_sign

    # diagonal: chemical potential + electric field with charge background (fix #2)
    for i, si in enumerate(basis):
        diag = 0.0
        for n in range(N):
            for f in range(F):
                q = n * F + f
                bq = (si >> (nq - 1 - q)) & 1
                diag += nu[f] * (1 - bq)        # (nu_f/2)(Z+1) = nu_f * (1-b)
        for ne in range(N - 1):
            cumQ = 0.0
            for k in range(ne + 1):
                for f in range(F):
                    q = k * F + f
                    cumQ += (1 - ((si >> (nq - 1 - q)) & 1))
                cumQ -= F / 2.0 * (1 - (-1) ** k)
            diag += cumQ ** 2
        H[i, i] += diag

    return H, basis


def ground_state(x, K, N=2, F=2, convention=CONVENTION_DEFAULT):
    """Physical-sector ground energy and (sector-indexed) ground vector."""
    from scipy.linalg import eigh
    H, basis = build_H_subspace(x, K, N, F, convention)
    w, v = eigh(H)
    return float(w[0]), v[:, 0], basis


# Reference values (x = 16, K = 0) from exact diagonalization -----------------

REF_E0 = {2: -30.5644, 3: -43.5660, 4: -68.5764, 5: -84.0895, 6: -107.3030}
REF_S = {2: 1.383, 3: 1.071, 4: 0.804, 5: 1.001, 6: 1.406}


# ===== ansatz.py =====
"""
Ansatz circuits as statevector maps (sparse, O(2^nq) per gate).

Gauge-invariant (GI) ansatz: each layer applies the fermionic-exchange gate
    Uxy(theta) = exp(-i theta (XX + YY) / 2)
on all nq-1 adjacent pairs, then Rz on all nq qubits.  Uxy commutes with Z_i+Z_j,
so the total excitation number (hence Q_tot) is conserved and the state never
leaves the physical sector.  Parameters per layer: (nq-1) + nq = 2 nq - 1 = 4N-1.

Hardware-efficient (HW) ansatz: Ry, Rz on every qubit then a CNOT ladder.  It has
no charge constraint and is used only as the comparison baseline that leaks out
of the physical sector.  Parameters per layer: 2 nq.

Qubit ordering is MSB-first: qubit q occupies bit (nq-1-q) of the state index,
matching hamiltonian.build_H_full / build_H_subspace.
"""



def gi_params_per_layer(N, F=2):
    nq = N * F
    return (nq - 1) + nq          # 4N - 1


def hw_params_per_layer(N, F=2):
    nq = N * F
    return 2 * nq


# --- gate primitives --------------------------------------------------------

def Uxy_apply(psi, p1, p2, theta, nq):
    """Fermionic exchange: mixes |01> <-> |10>, conserves excitation count."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    out = psi.copy()
    for idx in range(2 ** nq):
        if (idx >> (nq - 1 - p1)) & 1 == 0 and (idx >> (nq - 1 - p2)) & 1 == 1:
            j = idx ^ (1 << (nq - 1 - p1)) ^ (1 << (nq - 1 - p2))
            a01, a10 = psi[idx], psi[j]
            out[idx] = c * a01 - 1j * s * a10
            out[j] = -1j * s * a01 + c * a10
    return out


def Rz_apply(psi, q, theta, nq):
    out = psi.copy()
    for idx in range(len(psi)):
        b = (idx >> (nq - 1 - q)) & 1
        out[idx] = psi[idx] * (np.exp(-1j * theta / 2) if b == 0
                               else np.exp(1j * theta / 2))
    return out


def Ry_apply(psi, q, theta, nq):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    out = psi.copy()
    for idx in range(2 ** nq):
        if (idx >> (nq - 1 - q)) & 1 == 0:
            j = idx | (1 << (nq - 1 - q))
            a0, a1 = psi[idx], psi[j]
            out[idx] = c * a0 - s * a1
            out[j] = s * a0 + c * a1
    return out


def cnot_apply(psi, ctrl, tgt, nq):
    out = psi.copy()
    for idx in range(2 ** nq):
        if (idx >> (nq - 1 - ctrl)) & 1 == 1:
            partner = idx ^ (1 << (nq - 1 - tgt))
            if idx < partner:
                out[idx], out[partner] = psi[partner], psi[idx]
    return out


# --- ansatze ----------------------------------------------------------------

def gauge_invariant_ansatz(theta, psi0, nq, L):
    n_per = (nq - 1) + nq
    psi = psi0.copy()
    for l in range(L):
        off = l * n_per
        for q in range(nq - 1):
            psi = Uxy_apply(psi, q, q + 1, theta[off + q], nq)
        for q in range(nq):
            psi = Rz_apply(psi, q, theta[off + (nq - 1) + q], nq)
    return psi


def hw_efficient_ansatz(theta, psi0, nq, L):
    n_per = 2 * nq
    psi = psi0.copy()
    for l in range(L):
        off = l * n_per
        for q in range(nq):
            psi = Ry_apply(psi, q, theta[off + 2 * q], nq)
            psi = Rz_apply(psi, q, theta[off + 2 * q + 1], nq)
        for q in range(nq - 1):
            psi = cnot_apply(psi, q, q + 1, nq)
    return psi


def make_psi0(N, F=2):
    """Charge-neutral product reference |0101...> (Q_tot = 0)."""
    nq = N * F
    bits = [0 if f == 0 else 1 for n in range(N) for f in range(F)]
    idx = sum(b * (2 ** (nq - 1 - i)) for i, b in enumerate(bits))
    psi = np.zeros(2 ** nq, dtype=complex)
    psi[idx] = 1.0
    return psi


# ===== observables.py =====
"""
Observables for diagnosing VQE states:

  - build_operators : Q_tot, Q_tot^2, N_0, N_1 on the full 2^{2N} space.
  - measure_state   : <Q_tot>, <Q_tot^2>, <N_0>, <N_1>, fidelity with a reference.
  - entanglement_entropy : half-system von Neumann entropy via Schmidt (SVD).

<Q_tot^2> = 0 is the cheap, convention-independent signature that a state is in
the physical sector; a hardware-efficient ansatz that leaks reports
<Q_tot^2> != 0 with zero fidelity, which is how charge-sector leakage is detected.
"""




def build_operators(N=2, F=2):
    nq = N * F
    dim = 2 ** nq
    Imat = np.eye(dim, dtype=complex)
    Q_tot = sum((Imat - _single_Z(q, nq)) / 2 for q in range(nq)) - (nq // 2) * Imat
    Q2 = Q_tot @ Q_tot
    N0 = sum((Imat - _single_Z(n * F + 0, nq)) / 2 for n in range(N))
    N1 = sum((Imat - _single_Z(n * F + 1, nq)) / 2 for n in range(N))
    return Q_tot, Q2, N0, N1


def measure_state(psi, Q_tot, Q2, N0_op, N1_op, psi_ref):
    return {
        "Q_tot": float(np.real(psi.conj() @ Q_tot @ psi)),
        "Q_tot2": float(np.real(psi.conj() @ Q2 @ psi)),
        "N0": float(np.real(psi.conj() @ N0_op @ psi)),
        "N1": float(np.real(psi.conj() @ N1_op @ psi)),
        "fidelity": float(abs(psi.conj() @ psi_ref) ** 2),
    }


def entanglement_entropy(psi_full, partition_A, nq):
    """Half-system von Neumann entropy S_{A|B} (nats) via Schmidt decomposition."""
    B = [q for q in range(nq) if q not in partition_A]
    t = psi_full.reshape([2] * nq)
    t = np.transpose(t, partition_A + B)
    m = t.reshape(2 ** len(partition_A), 2 ** len(B))
    sv = np.linalg.svd(m, compute_uv=False)
    lam = sv ** 2
    lam = lam[lam > 1e-15]
    return float(-np.sum(lam * np.log(lam)))


def entropy_of_sector_state(psi_sector, basis, N, F=2):
    """Entropy for a state given as amplitudes over the physical-sector basis."""
    nq = N * F
    psi_full = np.zeros(2 ** nq, dtype=complex)
    for amp, s in zip(psi_sector, basis):
        psi_full[s] = amp
    return entanglement_entropy(psi_full, list(range(N)), nq)


# ===== vqe.py =====
"""
VQE optimization (derivative-free COBYLA) and the trainability diagnostics.

  - run_vqe          : general runner over an arbitrary ansatz on a full-space H,
                       with an optional charge penalty lambda <Q_tot^2>.
  - run_vqe_scaling  : convenience wrapper that optimizes the GI ansatz directly
                       in the physical sector (used for the p/d scaling table).
  - grad_variance    : finite-difference gradient variance vs N (barren-plateau
                       test).  The comparison that matters is against the e^{-nq}
                       prediction, ~55x suppression over nq = 4 -> 8.
  - stability_study  : final-energy distribution over many restarts.
  - convergence_trace: per-iteration energy of a single optimization.
"""




def run_vqe(ansatz_fn, H, nq, L, n_params, psi0,
            n_restarts=10, maxiter=2000, penalty_op=None, penalty_lam=0.0,
            seed=42, return_per_restart=False):
    """
    Minimize <H> (optionally <H> + lambda <penalty_op>) over an ansatz acting on
    the full 2^nq space.  The penalty term is used only for the hardware-efficient
    ansatz; the gauge-invariant ansatz needs none.
    """
    rng = np.random.default_rng(seed)
    best_E, best_psi = np.inf, None
    per_restart = []

    def cost(theta):
        psi = ansatz_fn(theta, psi0, nq, L)
        e = float(np.real(psi.conj() @ H @ psi))
        if penalty_op is not None and penalty_lam > 0:
            g = float(np.real(psi.conj() @ penalty_op @ psi))
            return e + penalty_lam * g
        return e

    for _ in range(n_restarts):
        t0 = rng.uniform(-np.pi, np.pi, n_params)
        r = minimize(cost, t0, method="COBYLA",
                     options={"maxiter": maxiter, "rhobeg": 0.5})
        psi_f = ansatz_fn(r.x, psi0, nq, L)
        e_true = float(np.real(psi_f.conj() @ H @ psi_f))   # report true energy, not penalized
        per_restart.append((e_true, psi_f.copy()))
        if e_true < best_E:
            best_E, best_psi = e_true, psi_f.copy()

    if return_per_restart:
        return best_E, best_psi, per_restart
    return best_E, best_psi


def _energy_sector(theta, Hsub, basis, nq, psi0, L):
    """GI-ansatz energy projected onto the physical sector."""
    psi = gauge_invariant_ansatz(theta, psi0, nq, L)
    psi_sub = np.array([psi[s] for s in basis])
    norm = np.linalg.norm(psi_sub)
    if norm < 1e-12:
        return 0.0
    psi_sub /= norm
    return float(np.real(psi_sub.conj() @ Hsub @ psi_sub))


def run_vqe_scaling(N, L, x=16.0, K=0.0, n_restarts=8, seed=42, maxiter=2000):
    """GI-ansatz VQE in the physical sector; returns the best energy."""
    F = 2
    nq = N * F
    Hsub, basis = build_H_subspace(x, K, N=N, F=F)
    psi0 = make_psi0(N, F)
    n_p = L * ((nq - 1) + nq)
    rng = np.random.default_rng(seed)
    best = np.inf
    for _ in range(n_restarts):
        t0 = rng.uniform(-np.pi, np.pi, n_p)
        r = minimize(_energy_sector, t0, args=(Hsub, basis, nq, psi0, L),
                     method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.5})
        best = min(best, r.fun)
    return best


def grad_variance(N, L=2, x=16.0, n_samples=20, seed=77, eps=1e-3):
    """Mean and std of mean-squared gradient over random parameter points."""
    F = 2
    nq = N * F
    Hsub, basis = build_H_subspace(x, 0.0, N=N, F=F)
    psi0 = make_psi0(N, F)
    n_p = L * ((nq - 1) + nq)
    rng = np.random.default_rng(seed)
    norms = []
    for _ in range(n_samples):
        theta = rng.uniform(-np.pi, np.pi, n_p)
        grad = np.zeros(n_p)
        for j in range(n_p):
            tp = theta.copy(); tp[j] += eps
            tm = theta.copy(); tm[j] -= eps
            grad[j] = (_energy_sector(tp, Hsub, basis, nq, psi0, L)
                       - _energy_sector(tm, Hsub, basis, nq, psi0, L)) / (2 * eps)
        norms.append(np.mean(grad ** 2))
    return float(np.mean(norms)), float(np.std(norms))


def stability_study(N, L_vals, x=16.0, n_restarts=20, seed=99, maxiter=1000):
    """Final-energy array for each L over many random restarts."""
    F = 2
    nq = N * F
    Hsub, basis = build_H_subspace(x, 0.0, N=N, F=F)
    psi0 = make_psi0(N, F)
    rng = np.random.default_rng(seed)
    results = {}
    for L in L_vals:
        n_p = L * ((nq - 1) + nq)
        finals = []
        for _ in range(n_restarts):
            t0 = rng.uniform(-np.pi, np.pi, n_p)
            r = minimize(_energy_sector, t0, args=(Hsub, basis, nq, psi0, L),
                         method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.5})
            finals.append(r.fun)
        results[L] = np.array(finals)
    return results


def convergence_trace(ansatz_fn, H, nq, L, n_params, psi0,
                      n_steps=60, n_tries=5, seed=7, maxiter_step=1):
    """
    Best-of-n_tries per-iteration energy trace (COBYLA with maxiter swept).
    Returns the energy as a function of optimizer iteration for the run that ends
    lowest.
    """
    rng = np.random.default_rng(seed)
    best_final = np.inf
    best_trace = None
    for _ in range(n_tries):
        t0 = rng.uniform(-np.pi, np.pi, n_params)
        trace = []
        for it in range(1, n_steps + 1):
            r = minimize(lambda th: float(np.real(
                            (ansatz_fn(th, psi0, nq, L)).conj()
                            @ H @ ansatz_fn(th, psi0, nq, L))),
                         t0, method="COBYLA",
                         options={"maxiter": it, "rhobeg": 0.5})
            trace.append(r.fun)
        if trace[-1] < best_final:
            best_final = trace[-1]
            best_trace = np.array(trace)
    return best_trace


# ===== noise.py =====
"""
Trapped-ion noise and zero-noise extrapolation, train-noiseless/evaluate-noisy.

A charge-conserving circuit depolarizes toward the maximally mixed state WITHIN
the physical sector, so the noisy energy is a contraction of the exact energy
toward the sector mean E_mix = Tr(H_phys)/d:

    E_noisy(K) = w_eff * E_exact(K) + (1 - w_eff) * E_mix,
    w_eff      = (1 - p)^nCX * (1 - 2 eta_SPAM).

First-order Richardson ZNE folds nCX -> 3 nCX:
    E_mit = 1.5 E(1) - 0.5 E(3).
It cannot track the exponential decay, so a residual grows with p and a ~2%
per-gate limit appears.  Fitting the physically correct form
    E(lambda) = a + b c^lambda
removes the depolarizing part and pins the residual to a p-independent SPAM floor
2 eta_SPAM |E0 - E_mix|.

DEFAULT convention is 'nu0only' (the manuscript convention): the N=3 first-order
boundary is at |K| ~ 5.6.  Running with 'antisym' instead gives |K| ~ 2.8 and is
the stale value that appeared in earlier figure panels.
"""



NCX_N3_L2 = 20         # CNOTs in the N=3, L=2 circuit
SPAM_DEFAULT = 0.005   # per-qubit readout (SPAM) error


def exact_and_mix(N, K, x=16.0, convention=CONVENTION_DEFAULT):
    """Return (E_ground, E_sectormean, ground_entropy) at chemical potential K."""
    H, basis = build_H_subspace(x, K, N=N, convention=convention)
    w, v = eigh(H)
    Emix = float(np.mean(w))                       # Tr(H_phys)/d
    S = entropy_of_sector_state(v[:, 0], basis, N)
    return float(w[0]), Emix, S


def noisy(E0, Emix, p, ncx, spam=SPAM_DEFAULT):
    """Depolarizing + SPAM contraction of the exact energy toward E_mix."""
    w = (1 - p) ** ncx
    w_eff = w * (1 - 2 * spam)
    return w_eff * E0 + (1 - w_eff) * Emix


def zne_linear(E0, Emix, p, ncx, spam=SPAM_DEFAULT):
    """First-order Richardson ZNE, lambda = 1, 3."""
    e1 = noisy(E0, Emix, p, ncx, spam)
    e3 = noisy(E0, Emix, p, 3 * ncx, spam)
    return 1.5 * e1 - 0.5 * e3


def zne_quadratic(E0, Emix, p, ncx, spam=SPAM_DEFAULT):
    """Three-point Richardson ZNE, lambda = 1, 2, 3."""
    e1 = noisy(E0, Emix, p, 1 * ncx, spam)
    e2 = noisy(E0, Emix, p, 2 * ncx, spam)
    e3 = noisy(E0, Emix, p, 3 * ncx, spam)
    return 3 * e1 - 3 * e2 + e3


def zne_exponential(E0, Emix, p, ncx, spam=SPAM_DEFAULT, lambdas=(1, 2, 3, 4, 5)):
    """
    Fit E(lambda) = a + b c^lambda and return a (the lambda -> 0 limit).
    Removes the depolarizing contribution exactly, leaving only the SPAM floor.
    """
    lam = np.array(lambdas, dtype=float)
    E = np.array([noisy(E0, Emix, p, int(round(l * ncx)), spam) for l in lam])

    def model(params):
        a, b, c = params
        return a + b * np.power(c, lam)

    def resid(params):
        return float(np.sum((model(params) - E) ** 2))

    from scipy.optimize import minimize
    # crude but stable init: a ~ E_mix-contracted limit, c in (0,1)
    best = None
    for c0 in (0.3, 0.6, 0.9):
        r = minimize(resid, [E[-1], E[0] - E[-1], c0], method="Nelder-Mead",
                     options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
        if best is None or r.fun < best.fun:
            best = r
    a, b, c = best.x
    return float(a + b)          # E(lambda=0) = a + b c^0, the zero-noise limit


def boundary_K(N=3, x=16.0, convention=CONVENTION_DEFAULT, Kmax=8.0, npts=161,
               all_crossings=False):
    """Locate the first-order phase boundary.

    Bug fix (see results/boundary_check.md): the original implementation
    returned Kv[1 + argmax(|diff(dE/dK)|)] -- the location of the SINGLE
    LARGEST slope discontinuity. That assumption silently breaks for N=4:
    every unit increase in the flavor-0 occupation <N_0> changes dE/dK by
    the same fixed amount (each site contributes the same -2*sqrt(x) slope),
    so N=4's two level crossings (<N_0>: 2->3 at K~2.5, and 3->4 at K~6.4)
    produce EXACTLY equal slope jumps (4.000000, bit-for-bit identical, not
    just numerically close). np.argmax breaks that tie by returning the
    FIRST maximum, so boundary_K(N=4) silently returned the non-terminal
    K~2.5 crossing instead of the terminal K~6.4 one that continues the
    N=2 (3.95) / N=3 (5.6) trend. N=2 and N=3 never hit this because they
    only have one crossing each in the scanned range.

    Fix: return the TERMINAL crossing (the one where the ground state
    reaches its saturating flavor-0 occupation, N_0 = N), found directly via
    <N_0> level crossings rather than the slope-jump proxy, since a level
    crossing IS an <N_0> jump (independent of the accidental degeneracy in
    slope-jump size that broke the old proxy). Verified: N=2 -> 3.95,
    N=3 -> 5.60, N=4 -> 6.40 (matches results/boundary_check.md exactly).

    all_crossings=True returns every crossing found as a list of
    (K, N0_before, N0_after) tuples, not just the terminal one -- e.g. for
    N=4, [(2.5, 2.0, 3.0), (6.4, 3.0, 4.0)].
    """
    Kv = np.linspace(0, Kmax, npts)
    Q_tot, Q2, N0_op, N1_op = build_operators(N, F=2)
    N0v = np.zeros(npts)
    for i, K in enumerate(Kv):
        _, psi = exact_ground_state_full(x, float(K), N, F=2, convention=convention)
        N0v[i] = float(np.real(psi.conj() @ N0_op @ psi))

    d = np.abs(np.diff(N0v))
    idxs = np.where(d > 0.3)[0]  # genuine level crossings, not numerical noise
    crossings = [(float(Kv[i + 1]), float(round(N0v[i])), float(round(N0v[i + 1])))
                 for i in idxs]

    if all_crossings:
        return crossings
    if not crossings:
        raise RuntimeError(f"boundary_K(N={N}): no <N_0> level crossing found in K in [0,{Kmax}]")
    terminal = [c for c in crossings if c[2] >= N]
    return (terminal[0] if terminal else crossings[-1])[0]


# ===== shotnoise.py =====
"""
Finite-shot measurement budget (Table V), evaluated on the exact ground state.

W is decomposed into Pauli strings W = sum_i c_i P_i; each P_i is measured with
n_shots samples drawn from its true +-1 distribution.  The per-term shot-noise
standard deviation is bounded by sum_i |c_i| / sqrt(n_shots).

This is statevector-only (no circuit backend): the shot noise is isolated from
optimizer and gate errors, exactly as in the manuscript.
"""



_P1Q = {"I": _I2, "X": _X2, "Y": _Y2, "Z": _Z2}


def _all_strings(n):
    return list("IXYZ") if n == 1 else [c + s for c in "IXYZ" for s in _all_strings(n - 1)]


def decompose_to_paulis(H, nq, tol=1e-10):
    """Return [(label, coeff, matrix), ...] for the nonzero Pauli terms of H."""
    dim = 2 ** nq
    out = []
    for label in _all_strings(nq):
        P = kron_list([_P1Q[c] for c in label])
        c = float(np.real(np.trace(P.conj().T @ H) / dim))
        if abs(c) > tol:
            out.append((label, c, P))
    return out


def measure_pauli(psi, P, n_shots, rng=None):
    """Sample n_shots measurements of Pauli P on psi; return the mean (+-1)."""
    rng = rng or np.random.default_rng()
    exp = float(np.real(psi.conj() @ P @ psi))
    p_plus = np.clip((1 + exp) / 2, 0.0, 1.0)
    outcomes = rng.choice([1, -1], size=n_shots, p=[p_plus, 1 - p_plus])
    return float(np.mean(outcomes))


def energy_shots(psi, paulis, n_shots=100, rng=None):
    """Shot-noisy estimate of <W> = sum_i c_i <P_i>."""
    rng = rng or np.random.default_rng()
    return sum(c * measure_pauli(psi, P, n_shots, rng) for _, c, P in paulis)


def sigma_bound(paulis, n_shots=100):
    """Loose upper bound on the shot-noise standard deviation, sum|c_i|/sqrt(n)."""
    return sum(abs(c) for _, c, _ in paulis) / np.sqrt(n_shots)


def energy_sigma(psi, paulis, n_shots=100):
    """
    Realized shot-noise standard deviation of <W>, assuming each term measured
    independently:  sqrt( sum_i c_i^2 (1 - <P_i>^2) / n_shots ).  Much smaller
    than sigma_bound when many terms have <P_i> ~ +-1 (near-product states).
    """
    var = 0.0
    for _, c, P in paulis:
        exp = float(np.real(psi.conj() @ P @ psi))
        var += c * c * (1 - exp ** 2) / n_shots
    return float(np.sqrt(var))


def exact_ground_state_full(x, K, N=2, F=2, convention="nu0only"):
    """Exact physical-sector ground state embedded in the full 2^{2N} space."""
    from scipy.linalg import eigh
    nq = N * F
    H = build_H_full(x, K, N, F, convention)
    basis = [s for s in range(2 ** nq) if bin(s).count("1") == nq // 2]
    Hsub = np.array([[H[i, j] for j in basis] for i in basis])
    w, v = eigh(Hsub)
    psi = np.zeros(2 ** nq, dtype=complex)
    for i, s in enumerate(basis):
        psi[s] = v[i, 0]
    return float(w[0]), psi