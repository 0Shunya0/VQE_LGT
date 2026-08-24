"""
The in-loop noisy-circuit evaluator (schwinger.backend.NoisyEvaluator), run
at p=0, eps=0, must reproduce the existing noiseless Table numbers to 1e-3
absolute energy: N=3,L=2 -> -43.560 and N=4,L=5 -> -68.562. This is the gate
schwinger.train_noisy_vqe.validation_gate() already implements; this test is
a thin pytest wrapper around it.

Slow: builds/transpiles two Aer circuits and runs 8 + 16 COBYLA restarts at
maxiter=2500 (the N=4,L=5 case needs 16 restarts -- 8 alone landed short by
1.7e-2 on a first attempt, see schwinger.train_noisy_vqe.validation_gate's
docstring). Expect several minutes.

Run: python -m pytest tests/test_validation_gate.py -m slow
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schwinger.train_noisy_vqe import validation_gate


@pytest.mark.slow
def test_validation_gate_passes():
    assert validation_gate(seed=42, maxiter=2500, n_restarts=8) is True
