"""
Tests for schwinger.core's Hamiltonian/flavor-sector/boundary-location code.

Run: python -m pytest tests/test_hamiltonian.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import schwinger.core as core

X = 16.0


def test_Q_tot_commutes_with_W():
    """The Kogut-Susskind charge Q_tot must be an exact conserved quantity
    of the bare Hamiltonian (not just of the GI ansatz's trajectory) --
    [Q_tot, H] = 0 as an operator identity, independent of any state."""
    for N in (2, 3):
        H = core.build_H_full(X, 0.0, N=N, F=2)
        Q_tot, Q2, N0, N1 = core.build_operators(N, F=2)
        comm = Q_tot @ H - H @ Q_tot
        assert np.max(np.abs(comm)) < 1e-10, f"N={N}: [Q_tot,H] != 0"


@pytest.mark.parametrize("N", [2, 3, 4, 5, 6])
def test_sector_dimension(N):
    """The Q_tot=0 physical sector has dimension C(2N,N)."""
    basis, _ = core.physical_basis(N, F=2)
    assert len(basis) == core.sector_dim(N)


@pytest.mark.slow
@pytest.mark.parametrize("N", [2, 3, 4, 5, 6])
def test_exact_energies_match_table(N):
    """Exact ground-state energy at x=16, K=0 must match REF_E0 (Table II)
    to 1e-3 (REF_E0 is itself rounded to 4 decimal places)."""
    E0, _, _ = core.ground_state(X, 0.0, N=N, F=2)
    assert abs(E0 - core.REF_E0[N]) < 1e-3


@pytest.mark.slow
def test_boundary_K():
    """boundary_K must return the TERMINAL <N_0> level crossing at every N,
    not the first of possibly several tied slope jumps (the bug: N=4 has
    two crossings, K~2.5 and K~6.4, with numerically identical slope jumps
    of exactly 4.000000 -- argmax silently picked the first)."""
    expected = {2: 3.95, 3: 5.60, 4: 6.40}
    for N, kb_expected in expected.items():
        kb = core.boundary_K(N=N, x=X, convention="nu0only")
        assert abs(kb - kb_expected) < 1e-6, f"N={N}: got {kb}, expected {kb_expected}"


@pytest.mark.slow
def test_boundary_K_all_crossings():
    """N=4 has two crossings in range: (K~2.5, N0 2->1) and (K~6.4, N0 1->0), N0 = flavor-0 particle number (occupied = bit 0).
    all_crossings=True must return both, in K order."""
    crossings = core.boundary_K(N=4, x=X, convention="nu0only", all_crossings=True)
    assert len(crossings) == 2
    (k1, a1, b1), (k2, a2, b2) = crossings
    assert abs(k1 - 2.5) < 0.05 and (a1, b1) == (2.0, 1.0)
    assert abs(k2 - 6.4) < 0.05 and (a2, b2) == (1.0, 0.0)
