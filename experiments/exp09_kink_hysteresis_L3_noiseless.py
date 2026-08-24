"""Sweep E, descending leg: N=3, L=3, GI, p=0, eps=0, K=7.00->4.00.
Run seed_hysteresis_ascending.py first."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os

from schwinger.kink_warmstart_core import run_warmstart_sweep, RESULTS_DIR

CSV_PATH = os.path.join(RESULTS_DIR, "kink_hysteresis_N3_L3_noiseless.csv")
SEED_BASE = 150000  # distinct from every prior sweep (10000..140000)

if __name__ == "__main__":
    run_warmstart_sweep(p=0.0, eps=0.0, csv_path=CSV_PATH, seed_base=SEED_BASE, direction="down")
