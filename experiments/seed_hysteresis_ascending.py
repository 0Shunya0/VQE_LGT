"""
Fast, synchronous step: copy the already-complete ascending warm-start sweep
(results/kink_warmstart_N3_L3_noiseless.csv / _noisy.csv) into the new
hysteresis CSVs with direction='up' added. Run once before launching the
descending passes (run_kink_hysteresis_L3_noiseless.py / _noisy.py).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os

import pandas as pd

from schwinger.kink_warmstart_core import CSV_FIELDS, RESULTS_DIR

SOURCE = {
    "noiseless": os.path.join(RESULTS_DIR, "kink_warmstart_N3_L3_noiseless.csv"),
    "noisy": os.path.join(RESULTS_DIR, "kink_warmstart_N3_L3_noisy.csv"),
}
DEST = {
    "noiseless": os.path.join(RESULTS_DIR, "kink_hysteresis_N3_L3_noiseless.csv"),
    "noisy": os.path.join(RESULTS_DIR, "kink_hysteresis_N3_L3_noisy.csv"),
}


def seed_ascending_pass(kind):
    dest = DEST[kind]
    if os.path.exists(dest):
        existing = pd.read_csv(dest)
        if "direction" in existing.columns and (existing["direction"] == "up").any():
            print(f"[{dest}] ascending pass already present, skipping copy.", flush=True)
            return
    df = pd.read_csv(SOURCE[kind])
    df.insert(df.columns.get_loc("K_index") + 1, "direction", "up")
    df = df[CSV_FIELDS]
    write_header = not os.path.exists(dest)
    df.to_csv(dest, mode="a", header=write_header, index=False)
    print(f"[{dest}] seeded with {len(df)} ascending-pass rows (direction='up') from {SOURCE[kind]}", flush=True)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for kind in ("noiseless", "noisy"):
        seed_ascending_pass(kind)
