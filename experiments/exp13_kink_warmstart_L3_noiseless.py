"""Sweep C: warm-start continuation, N=3, L=3, GI, p=0, eps=0."""
import os

from kink_warmstart_core import run_warmstart_sweep, RESULTS_DIR

CSV_PATH = os.path.join(RESULTS_DIR, "kink_warmstart_N3_L3_noiseless.csv")
SEED_BASE = 130000  # distinct from every prior sweep (10000..120000)

if __name__ == "__main__":
    run_warmstart_sweep(p=0.0, eps=0.0, csv_path=CSV_PATH, seed_base=SEED_BASE)
