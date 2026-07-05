#!/usr/bin/env bash
# Train + eval J for a set of seeds under one J-training regime (full-frame or fan-cropped real).
# Usage: run_seeds.sh <variant: full|crop> <outdir> <seed0> [seed1 ...]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1
WT="$(pwd)"; VENV="/Users/tohntebs/Codes/Probe_BMode_Gym/.venv/bin/python"
export PYTHONPATH="$WT/src"

VARIANT="$1"; OUT="$2"; shift 2
SEEDS=("$@")
CROP=""; [ "$VARIANT" = "crop" ] && CROP="--crop-real"
mkdir -p "$OUT" "$OUT/logs"
echo "== J regime=$VARIANT out=$OUT seeds=${SEEDS[*]} =="
for s in "${SEEDS[@]}"; do
  echo "-- seed $s: train --"
  $VENV scripts/premise/train_organ_J.py --seed "$s" $CROP --out "$OUT" --epochs 400 \
      > "$OUT/logs/J_seed${s}.log" 2>&1
  tail -1 "$OUT/logs/J_seed${s}.log"
  echo "-- seed $s: eval --"
  $VENV scripts/premise/eval_premise.py --seed "$s" --jdir "$OUT" --out "$OUT" \
      > "$OUT/logs/eval_seed${s}.log" 2>&1
  grep -A7 "per-organ IoU" "$OUT/logs/eval_seed${s}.log" || tail -3 "$OUT/logs/eval_seed${s}.log"
done
echo "== aggregate ($VARIANT) =="
$VENV scripts/premise/aggregate.py --dir "$OUT" --metric iou
echo "== DONE $VARIANT =="
