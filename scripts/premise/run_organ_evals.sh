#!/usr/bin/env bash
# Evaluate every trained J (both regimes, all seeds) on the organ-TARGETED sim sets
# (spleen, kidney) -- the fairer non-dominant-organ premise test.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1
WT="$(pwd)"; VENV="/Users/tohntebs/Codes/Probe_BMode_Gym/.venv/bin/python"
export PYTHONPATH="$WT/src"

for organ in spleen kidney; do
  SET="outputs/premise/organ_sets/${organ}.npz"
  for reg in crop full; do
    OUT="outputs/premise/${organ}_${reg}"; mkdir -p "$OUT"
    for s in 0 1 2; do
      cp "outputs/premise/${reg}/J_seed${s}.pt" "$OUT/J_seed${s}.pt" 2>/dev/null || continue
      echo "== $organ regime=$reg seed=$s =="
      $VENV scripts/premise/eval_premise.py --seed "$s" --sim "$SET" --jdir "$OUT" --out "$OUT" \
          > "$OUT/eval_seed${s}.log" 2>&1
      grep -A7 "per-organ IoU" "$OUT/eval_seed${s}.log" | tail -8
    done
    echo "-- aggregate $organ $reg --"
    $VENV scripts/premise/aggregate.py --dir "$OUT" --metric iou 2>/dev/null | sed -n '3,12p'
  done
done
echo "== ORGAN EVALS DONE =="
