"""
schwinger: simulation code for "Disentangling Expressibility, Symmetry
Protection, and Hardware Noise in Variational Quantum Simulation of the
Two-Flavor Schwinger Model".

Modules:
    core                 Hamiltonian, ansatz, observables, noiseless VQE
                         drivers, post-hoc noise/ZNE, boundary_K, shot-noise
                         estimator (schwinger_core.py, unchanged content)
    backend              Qiskit circuit builders, noise-model setup, ZNE
                         folding, NoisyEvaluator (qiskit_backend.py, unchanged
                         content)
    parallel_worker      process-pool cell worker + thread pinning
    warmstart_worker     single-restart worker for warm-start sweeps
    kink_warmstart_core  bidirectional warm-start/hysteresis sweep driver
    train_noisy_vqe      train-noisy VQE driver and the validation gate

This package intentionally keeps schwinger_core.py's and qiskit_backend.py's
original content as single files (schwinger/core.py, schwinger/backend.py)
rather than splitting them into narrower modules: they are large working
files with dozens of downstream scripts depending on their exact behavior,
and a hand-split risks subtly changing which names are visible where an
import order matters. Only import paths changed, not behavior.
"""
from schwinger.core import *  # noqa: F401,F403
from schwinger.backend import (  # noqa: F401
    build_gi_circuit, build_hw_circuit, fold_circuit, build_noise_model,
    NoisyEvaluator,
)
