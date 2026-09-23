"""
One-off diagnostic: compare pre-fix (results/pre_fix_backup/) vs post-fix
(results/) energy columns per file, flagging any shift that looks bigger
than a fixed-operator-offset re-measurement would predict (as opposed to a
qualitative change: sign flips, huge relative blowups, NaNs, wildly
different fidelity/leakage behavior).
"""
import os
import numpy as np
import pandas as pd

BACKUP = "results/pre_fix_backup"
RESULTS = "results"


def compare_energy(fname, key_cols, energy_col="energy"):
    old_path = os.path.join(BACKUP, fname)
    new_path = os.path.join(RESULTS, fname)
    if not (os.path.exists(old_path) and os.path.exists(new_path)):
        print(f"{fname}: MISSING (old={os.path.exists(old_path)}, new={os.path.exists(new_path)})")
        return
    old = pd.read_csv(old_path)
    new = pd.read_csv(new_path)
    merged = old.merge(new, on=key_cols, suffixes=("_old", "_new"))
    if len(merged) == 0:
        print(f"{fname}: no matching rows on keys {key_cols} (old n={len(old)}, new n={len(new)})")
        return
    d = merged[f"{energy_col}_new"] - merged[f"{energy_col}_old"]
    rel = d / merged[f"{energy_col}_old"].abs().clip(lower=1e-6)
    print(f"{fname}  (n={len(merged)}/{len(old)} matched, energy_col={energy_col}):")
    print(f"    diff:  mean={d.mean():+.4f}  std={d.std():.4f}  min={d.min():+.4f}  max={d.max():+.4f}")
    print(f"    |rel|: mean={rel.abs().mean()*100:.2f}%  max={rel.abs().max()*100:.2f}%")
    big = merged[rel.abs() > 0.20]
    if len(big):
        print(f"    ** {len(big)} rows with >20% relative shift -- inspect **")
    return d, rel


print("=" * 90)
compare_energy("noisy_inloop_N3.csv", ["p", "K", "restart"])
print()
compare_energy("kink_hysteresis_N3_L3_noiseless.csv", ["direction", "K", "restart_id"])
print()
compare_energy("kink_hysteresis_N3_L3_noisy.csv", ["direction", "K", "restart_id"])
print()
compare_energy("kink_warmstart_N3_L3_noiseless.csv", ["K", "restart_id"])
print()
compare_energy("kink_warmstart_N3_L3_noisy.csv", ["K", "restart_id"])
print()
compare_energy("inloop_zne_N3.csv", ["p", "K", "fold_lambda", "restart"])
print()
compare_energy("kink_dense_N3_L3_noiseless.csv", ["K", "restart"])
print()
compare_energy("kink_dense_N3_L3_noisy.csv", ["K", "restart"])
print()
compare_energy("noisy_N4_spot.csv", ["K_label", "restart"])
print()
compare_energy("kink_dense_N3.csv", ["K", "restart"])
print()
compare_energy("noiseless_N4_control.csv", ["K_label", "restart"])
print()

# gradvar_hw: different schema (grad_var_mean, not energy/per-restart)
print("gradvar_hw.csv:")
old = pd.read_csv(os.path.join(BACKUP, "gradvar_hw.csv"))
new = pd.read_csv(os.path.join(RESULTS, "gradvar_hw.csv"))
m = old.merge(new, on=["N", "ansatz"], suffixes=("_old", "_new"))
for _, r in m.iterrows():
    d = r["grad_var_mean_new"] - r["grad_var_mean_old"]
    rel = d / abs(r["grad_var_mean_old"]) * 100 if r["grad_var_mean_old"] else float("nan")
    print(f"  N={r['N']} {r['ansatz']}: old={r['grad_var_mean_old']:.4f} new={r['grad_var_mean_new']:.4f} "
          f"diff={d:+.4f} ({rel:+.2f}%)")

print()
print("=" * 90)
print("inloop_zne_N3_mitigated.csv (derived E_mit_linear / E_exact -- boundary-row check):")
old = pd.read_csv(os.path.join(BACKUP, "inloop_zne_N3_mitigated.csv"))
new = pd.read_csv(os.path.join(RESULTS, "inloop_zne_N3_mitigated.csv"))
m = old.merge(new, on=["p", "K"], suffixes=("_old", "_new"))
d = m["E_mit_linear_new"] - m["E_mit_linear_old"]
rel = d / m["E_mit_linear_old"].abs().clip(lower=1e-6)
print(f"  n matched={len(m)}: E_mit_linear diff mean={d.mean():+.4f} std={d.std():.4f} "
      f"max|rel|={rel.abs().max()*100:.2f}%")
dex = m["E_exact_new"] - m["E_exact_old"]
print(f"  E_exact (should be ~identical, both isospectral-safe): mean diff={dex.mean():+.6f}, max|diff|={dex.abs().max():.6f}")
