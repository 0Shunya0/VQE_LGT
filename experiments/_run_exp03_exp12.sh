#!/bin/bash
cd "$(dirname "$0")/.."
L=results/postfix_run_logs; T=$L/_step_times_2.txt
for s in exp03_N4_spot exp12_kink_basin_check_L3; do
  echo "START $s $(date)" >> $T; t0=$(date +%s)
  python experiments/$s.py > $L/$s.log 2>&1; rc=$?
  echo "END $s rc=$rc elapsed=$(( $(date +%s)-t0 ))s $(date)" >> $T
  [ $rc -ne 0 ] && { echo "FAILED $s" >> $T; exit 1; }
done
echo "ALL DONE $(date)" >> $T
