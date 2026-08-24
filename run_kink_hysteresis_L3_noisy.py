"""Sweep F, descending leg: N=3, L=3, GI, p=0.01, eps=SPAM_DEFAULT, K=7.00->4.00.
Run seed_hysteresis_ascending.py first."""
import os

import schwinger_core as core
from kink_warmstart_core import run_warmstart_sweep, RESULTS_DIR

CSV_PATH = os.path.join(RESULTS_DIR, "kink_hysteresis_N3_L3_noisy.csv")
SEED_BASE = 160000  # distinct from every prior sweep (10000..150000)

if __name__ == "__main__":
    run_warmstart_sweep(p=0.01, eps=core.SPAM_DEFAULT, csv_path=CSV_PATH, seed_base=SEED_BASE,
                         direction="down")
