#!/usr/bin/env bash
# Organ-conditioned (liver) acquisition RL: physics / CUT / CycleGAN / SonoSPADE x 3 seeds.
# Same real-trained J reward, same warm-start budget, same held-out eval setups. Sequential to
# avoid MPS contention. Primary metric = ground-truth liver visibility (true label), reported by
# scripts/premise/agg_organ_rl.py.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1
WT="$(pwd)"; VENV="/Users/tohntebs/Codes/Probe_BMode_Gym/.venv/bin/python"
export PYTHONPATH="$WT/src"
J="outputs/premise/crop/J_seed0.pt"
OUT="outputs/organ_rl"; mkdir -p "$OUT/logs"
IMIT=200; ITERS=20; EPI=4; NSTEPS=12; EVAL=16

for seed in 0 1 2; do
  for cond in physics cut cyclegan sonospade; do
    echo "==== $cond seed $seed ===="
    $VENV scripts/train_organ_rl.py --condition "$cond" --seed "$seed" --J "$J" \
       --imitation $IMIT --iters $ITERS --episodes-per-iter $EPI --n-steps $NSTEPS \
       --eval-episodes $EVAL --out "$OUT/${cond}_s${seed}.json" \
       > "$OUT/logs/${cond}_s${seed}.log" 2>&1
    grep -E "\[FINAL\]|after imitation" "$OUT/logs/${cond}_s${seed}.log" | tail -2
  done
done
echo "#### ORGAN RL DONE ####"
