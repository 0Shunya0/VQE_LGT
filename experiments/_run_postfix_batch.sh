#!/bin/bash
# Post-fix rerun batch, in dependency order. Logs each script separately and
# records wall-clock time per step. Stops on first failure (set -e) so a
# crash doesn't silently continue into dependent steps.
set -e
cd "$(dirname "$0")/.."
LOGDIR="results/postfix_run_logs"
mkdir -p "$LOGDIR"
TIMES_FILE="$LOGDIR/_step_times.txt"
: > "$TIMES_FILE"

run_step() {
  name="$1"
  script="$2"
  echo "=== START $name $(date) ===" | tee -a "$TIMES_FILE"
  t0=$(date +%s)
  set +e
  python "$script" > "$LOGDIR/${name}.log" 2>&1
  rc=$?
  set -e
  t1=$(date +%s)
  dt=$((t1 - t0))
  echo "=== END $name rc=$rc elapsed=${dt}s $(date) ===" | tee -a "$TIMES_FILE"
  if [ $rc -ne 0 ]; then
    echo "STEP FAILED: $name (see $LOGDIR/${name}.log)" | tee -a "$TIMES_FILE"
    exit 1
  fi
}

run_step "01_exp13_warmstart_noiseless" "experiments/exp13_kink_warmstart_L3_noiseless.py"
run_step "02_exp14_warmstart_noisy"     "experiments/exp14_kink_warmstart_L3_noisy.py"
run_step "03_seed_hysteresis_ascending" "experiments/seed_hysteresis_ascending.py"
run_step "04_exp01_inloop_N3"           "experiments/exp01_inloop_N3.py"
run_step "05_exp09_hysteresis_noiseless" "experiments/exp09_kink_hysteresis_L3_noiseless.py"
run_step "06_exp09_hysteresis_noisy"    "experiments/exp09_kink_hysteresis_L3_noisy.py"
run_step "07_exp02_inloop_zne_N3"       "experiments/exp02_inloop_zne_N3.py"
run_step "08_exp08_kink_dense_L3_noiseless" "experiments/exp08_kink_dense_L3_noiseless.py"
run_step "09_exp08_kink_dense_L3_noisy" "experiments/exp08_kink_dense_L3_noisy.py"
run_step "10_exp03_N4_spot"             "experiments/exp03_N4_spot.py"
run_step "11_exp04_gradvar"             "experiments/exp04_gradvar.py"
run_step "12_exp05_kink_dense_L2"       "experiments/exp05_kink_dense_L2.py"
run_step "13_exp07_noiseless_N4_control" "experiments/exp07_noiseless_N4_control.py"

echo "=== BATCH COMPLETE $(date) ===" | tee -a "$TIMES_FILE"
