#!/usr/bin/env bash
# Clean definitive re-run after the adversarial-review fixes:
#   * fair CUT checkpoint (exp_real_cut.pt) in eval_premise.py
#   * aspect-preserving fan crop (common.py) + support-weighted J selection (train_organ_J.py)
# Retrains all J (both regimes, seeds 0-2), evals on pairs_med (liver) and the organ-targeted
# spleen/kidney sets, then aggregates IoU and precision.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1
WT="$(pwd)"; VENV="/Users/tohntebs/Codes/Probe_BMode_Gym/.venv/bin/python"
export PYTHONPATH="$WT/src"
SEEDS="0 1 2"

for reg in crop full; do
  OUT="outputs/premise/$reg"; mkdir -p "$OUT/logs"
  CROP=""; [ "$reg" = "crop" ] && CROP="--crop-real"
  for s in $SEEDS; do
    echo "== train J $reg seed $s =="
    $VENV scripts/premise/train_organ_J.py --seed "$s" $CROP --out "$OUT" --epochs 400 \
        > "$OUT/logs/J_seed${s}.log" 2>&1; tail -1 "$OUT/logs/J_seed${s}.log"
    echo "== eval(pairs_med) $reg seed $s =="
    $VENV scripts/premise/eval_premise.py --seed "$s" --jdir "$OUT" --out "$OUT" \
        > "$OUT/logs/eval_seed${s}.log" 2>&1; grep -A6 "per-organ IoU" "$OUT/logs/eval_seed${s}.log" | tail -6
  done
done

# organ-targeted eval sets (fair CUT already baked into eval_premise.py)
for organ in spleen kidney; do
  for reg in crop full; do
    OUT="outputs/premise/${organ}_${reg}"; mkdir -p "$OUT"
    for s in $SEEDS; do
      cp "outputs/premise/${reg}/J_seed${s}.pt" "$OUT/J_seed${s}.pt"
      $VENV scripts/premise/eval_premise.py --seed "$s" --sim "outputs/premise/organ_sets/${organ}.npz" \
          --jdir "$OUT" --out "$OUT" > "$OUT/eval_seed${s}.log" 2>&1
    done
  done
done

echo "############ AGGREGATES ############"
for d in crop full spleen_crop spleen_full kidney_crop kidney_full; do
  echo "===== $d (IoU) ====="; $VENV scripts/premise/aggregate.py --dir "outputs/premise/$d" --metric iou 2>/dev/null
  echo "----- $d (precision) -----"; $VENV scripts/premise/aggregate.py --dir "outputs/premise/$d" --metric precision 2>/dev/null | sed -n '3,10p'
done
echo "############ RERUN DONE ############"
