"""
Bug 2 investigation: independently locate the N=2,3,4 phase boundary via fine
exact diagonalization (not by trusting schwinger_core.boundary_K), using two
independent signatures -- the largest dE/dK discontinuity and the largest
jump in <N_0> (flavor-0 occupation; first-order transitions are level
crossings between different flavor-occupation sectors, so they show up as
near-integer jumps in <N_0> at exactly the same K, independent of the slope
estimate). Then compare against what boundary_K(N) actually returns.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np

import schwinger.core as core

X = 16.0
CONV = "nu0only"


def scan(N, Kmax=8.0, npts=161):
    Kv = np.linspace(0, Kmax, npts)
    Q_tot, Q2, N0_op, N1_op = core.build_operators(N, F=2)
    E = np.zeros(npts)
    N0v = np.zeros(npts)
    for i, K in enumerate(Kv):
        E0, psi = core.exact_ground_state_full(X, float(K), N, F=2, convention=CONV)
        E[i] = E0
        N0v[i] = float(np.real(psi.conj() @ N0_op @ psi))
    dE = np.gradient(E, Kv)
    jump_idx_slope = 1 + int(np.argmax(np.abs(np.diff(dE))))
    kb_slope = float(Kv[jump_idx_slope])
    jump_idx_N0 = 1 + int(np.argmax(np.abs(np.diff(N0v))))
    kb_N0 = float(Kv[jump_idx_N0])
    N0_jump_size = float(np.abs(np.diff(N0v))[jump_idx_N0 - 1])
    return Kv, E, N0v, dE, kb_slope, kb_N0, N0_jump_size


def all_jumps(Kv, N0v, dE, thresh=0.3):
    """Every point where <N0> jumps by more than `thresh`, with the dE/dK
    step size at that same point, so every genuine level crossing in the
    scanned range is visible, not just the single largest one."""
    d = np.abs(np.diff(N0v))
    idxs = np.where(d > thresh)[0]
    out = []
    for i in idxs:
        out.append(dict(K=float(Kv[i + 1]), N0_before=float(N0v[i]), N0_after=float(N0v[i + 1]),
                         dE_before=float(dE[i]), dE_after=float(dE[i + 1]),
                         slope_jump=float(abs(dE[i + 1] - dE[i]))))
    return out


def main():
    print(f"{'N':>3} {'kb_slope':>10} {'kb_N0':>10} {'N0_jump':>8} {'boundary_K()':>13}")
    results = {}
    for N in (2, 3, 4):
        Kv, E, N0v, dE, kb_slope, kb_N0, N0_jump = scan(N)
        kb_reported = core.boundary_K(N=N, convention=CONV)
        results[N] = dict(kb_slope=kb_slope, kb_N0=kb_N0, N0_jump=N0_jump, kb_reported=kb_reported)
        print(f"{N:>3} {kb_slope:>10.3f} {kb_N0:>10.3f} {N0_jump:>8.3f} {kb_reported:>13.3f}")

        print(f"    all N0 crossings for N={N} (max N0 = {N}):")
        for j in all_jumps(Kv, N0v, dE):
            print(f"      K~{j['K']:.3f}: N0 {j['N0_before']:.0f}->{j['N0_after']:.0f}, "
                  f"dE/dK {j['dE_before']:.3f}->{j['dE_after']:.3f} (slope jump {j['slope_jump']:.3f})")

    # dump full N0(K) trace around the region of interest for N=4 for diagnosis
    print("\nN=4 <N0> trace, K=0..8 (every 5th point):")
    Kv4, E4, N0v4, dE4, *_ = scan(4)
    for i in range(0, len(Kv4), 5):
        print(f"  K={Kv4[i]:.2f}  E={E4[i]:.4f}  N0={N0v4[i]:.4f}  dE/dK={dE4[i]:.4f}")

    return results


if __name__ == "__main__":
    main()
