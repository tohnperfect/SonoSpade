#!/usr/bin/env bash
# SonoSPADE proof-of-concept experiment on the USB Gym substrate.
#
# Trains every translator on the same real pool, then evaluates them on the same held-out content
# with the per-tissue metric suite and a held-out-patient proxy-real TSTR test. Resumable: each
# stage is skipped if its output already exists. Honest scope: in the absence of the Kaggle Vitale
# labeled real set (future work), the labeled "real" test is a proxy (a known texture transform on
# unseen-patient renders), so TSTR and per-class W1 here are directional, while the per-tissue
# sensitivity, structure preservation, and speckle statistics are measured directly.
set -e
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=.venv/bin/python
mkdir -p data/eval outputs/experiment

echo "== 1. held-out-patient proxy-real labeled test set (3 unseen cases) =="
if [ ! -f data/eval/proxy_real_test.npz ]; then
  $PY scripts/make_dataset.py --cases s0519 s0522 s0591 --target liver --image-size 128 \
      --renderer physics --pairs-out data/eval/heldout_pairs.npz --pairs-n 50 --pairs-dr mild --pairs-only
  $PY - <<'PYEOF'
import numpy as np
from usbg.dataset import load_pairs_npz, write_pairs_npz
from usbg.texture_gan import proxy_real_texture
imgs, labs, sess, meta = load_pairs_npz("data/eval/heldout_pairs.npz")
imgs = imgs.astype(np.float32) / 255.0
rng = np.random.default_rng(7)
pr = np.stack([proxy_real_texture(imgs[i], rng) for i in range(len(imgs))], 0)
write_pairs_npz("data/eval/proxy_real_test.npz", (pr * 255).astype(np.uint8), labs.astype(np.uint8),
                sessions=sess, extra_meta={"note": "held-out-patient proxy real (no Kaggle Vitale yet)"})
print("proxy-real test frames:", len(pr))
PYEOF
else echo "  exists, skip"; fi

echo "== 2. baseline translators (grayscale) fresh on the real pool =="
for V in cyclegan cut cyclegan_sc; do
  OUT=data/dataset/exp_$V.pt
  if [ ! -f "$OUT" ]; then
    $PY scripts/train_texture_gan.py --variant $V --sim-npz data/dataset/usbg.npz \
        --real-dir data/real_us_crop --iters 800 --out "$OUT" \
        --preview outputs/experiment/${V}_preview.png
  else echo "  $V exists, skip"; fi
done

echo "== 3. SonoSPADE (grouped S_US + all improvements) =="
if [ ! -f data/dataset/exp_sonospade.pt ]; then
  $PY scripts/train_texture_gan.py --variant sonospade --pairs data/dataset/pairs_med.npz \
      --real-dir data/real_us_crop --s-us data/dataset/s_us_grouped.pt --iters 1500 \
      --out data/dataset/exp_sonospade.pt --preview outputs/experiment/sonospade_preview.png
else echo "  exists, skip"; fi

echo "== 4. evaluate all on the same held-out content + proxy-real TSTR =="
$PY scripts/eval_texture.py --pairs data/dataset/pairs_med.npz \
    --real-pairs data/eval/proxy_real_test.npz --limit 200 --tstr-iters 500 \
    --cyclegan data/dataset/exp_cyclegan.pt --cut data/dataset/exp_cut.pt \
    --cyclegan-sc data/dataset/exp_cyclegan_sc.pt --sonospade data/dataset/exp_sonospade.pt \
    --out outputs/experiment/results.json

echo "== done: outputs/experiment/results.json =="
