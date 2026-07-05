#!/usr/bin/env bash
# SonoSPADE REAL-data head-to-head on the Kaggle ussimandsegm (Vitale) set.
#
# Uses the 617-frame real US pool as the translators' target and the 60 annotated real frames as the
# labeled test for per-class W1 and TSTR (the genuine metric that ties texture to a downstream task).
# Run scripts/ingest_kaggle_us.py first (builds the pool + labeled test). Resumable: each translator
# is skipped if its checkpoint exists. Every translator is trained on the SAME real pool.
set -e
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=.venv/bin/python
POOL=data/real/kaggle_vitale/pool
TEST=data/eval/kaggle_vitale_test.npz
mkdir -p outputs/experiment

[ -d "$POOL" ] || { echo "run scripts/ingest_kaggle_us.py first ($POOL missing)"; exit 1; }

echo "== baseline translators (grayscale) on the Kaggle real pool =="
for V in cyclegan cut cyclegan_sc; do
  OUT=data/dataset/exp_real_$V.pt
  [ -f "$OUT" ] || $PY scripts/train_texture_gan.py --variant $V --sim-npz data/dataset/usbg.npz \
      --real-dir "$POOL" --iters 800 --out "$OUT" \
      --preview outputs/experiment/real_${V}_preview.png
done

echo "== SonoSPADE (grouped S_US + all improvements) on the Kaggle real pool =="
[ -f data/dataset/exp_real_sonospade.pt ] || $PY scripts/train_texture_gan.py --variant sonospade \
    --pairs data/dataset/pairs_med.npz --real-dir "$POOL" --s-us data/dataset/s_us_grouped.pt \
    --iters 1500 --out data/dataset/exp_real_sonospade.pt \
    --preview outputs/experiment/real_sonospade_preview.png

echo "== evaluate on the 60 labeled real frames (real TSTR + per-class W1) =="
$PY scripts/eval_texture.py --pairs data/dataset/pairs_med.npz --real-pairs "$TEST" \
    --limit 200 --tstr-iters 500 \
    --cyclegan data/dataset/exp_real_cyclegan.pt --cut data/dataset/exp_real_cut.pt \
    --cyclegan-sc data/dataset/exp_real_cyclegan_sc.pt --sonospade data/dataset/exp_real_sonospade.pt \
    --out outputs/experiment/results_real.json

echo "== done: outputs/experiment/results_real.json (fill the paper with fill_paper_results.py --results it) =="
