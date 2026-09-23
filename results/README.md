# results/ notes

- **INVALID hardware results (EstimatorV2):** `ibm_N3_L2_K0_mitigated.json` (Job 0, Job A, Job B) and the E1/E2 rows of
  `ibm_N3_L2_K0_drift_check.json` are invalid. The server returned 0 +/- 0 for the Y-Z-Y group on physical qubits
  (8,9,10),(10,11,18), so each energy is missing about 9.5 units. Reason and evidence: `ibm_N3_L2_K0_gap_localize.json`.
  Nothing was deleted; each file carries a top-level `"status"` field.
- **Valid hardware results (SamplerV2):** `ibm_N3_L2_K0.json` (job 1), the S1/S2 rows of `ibm_N3_L2_K0_drift_check.json`,
  and `D_sampler` in `ibm_N3_L2_K0_gap_localize.json`. Raw counts for all four Sampler jobs (daoeuotr85ps73ff6a70,
  daofnv78gn2s739nqjlg, daofnvopqrnc7399u9ig, daonh9lr85ps73ffhqjg) are in `ibm_N3_L2_K0_sampler_counts.json`.
- `pre_fix_backup/` holds every output generated before the `build_H_full` / `build_operators` sign fix.
