# SonoSPADE

**Real-time, GPU-free, per-tissue ultrasound texture synthesis for closed-loop acquisition.**

SonoSPADE is a physics-conditioned, *per-tissue* ultrasound texture stage: it refines a physics
B-mode render into realistic ultrasound under spatially-adaptive conditioning from the **tissue label
slice**, trained **unpaired with no real annotations**. A single generator pass runs at **15–85 fps
without a GPU**, two to three orders of magnitude faster (per frame) than a semantic-diffusion
synthesizer, so it can sit inside a closed-loop reinforcement-learning acquisition simulator, where a
new frame must be rendered at every agent step.

📄 The paper is [`paper/main_sashimi.pdf`](paper/main_sashimi.pdf) (MICCAI SASHIMI camera-ready, LNCS
format), with [`paper/supplementary.pdf`](paper/supplementary.pdf) alongside it. `paper/` is the
complete Overleaf-ready source package: see [`paper/README.txt`](paper/README.txt) to rebuild.

---

## Why per-tissue?

Global sim-to-real translators (CycleGAN, CUT) apply one appearance map to the whole image, so they
cannot guarantee that *liver reads as liver and kidney as kidney*. SonoSPADE conditions the generator
on **both** the physics render (content) **and** the label map (per-tissue modulation, via SPADE), and
supervises it with a segmenter pretrained for free on aligned simulator pairs (frozen, it pseudo-labels
the unlabeled real pool for an OASIS discriminator and enforces a per-tissue consistency loss). Relabel
a region and only *its* texture changes.

## Key results (honest)

| Property | SonoSPADE | Best global recolor | Note |
|---|---|---|---|
| **Label-swap locality** (per-tissue control) | **≈ 26** | 0 | 0 for every label-free translator by construction |
| **Per-organ intensity W₁** (↓, real match) | **0.119** | 0.246 (CUT) | best among *learned* translators |
| **Structure preservation** (edge-IoU, ↑) | **0.41** | 0.17 | 2.4× the best baseline |
| Whole-frame **FID/KID** | loses *by design* | wins | these reward the global recolor SonoSPADE does not perform |
| **Latency** (CPU / iGPU, single pass) | **66 / 12 ms** | ~same | diffusion is 133×–665× slower **per frame** at standard T |
| **Downstream TSTR** (train-on-synthetic Dice) | overtakes raw render on every seed | n/a | only after matching sector geometry + test-time adaptation; absolute Dice stays low |

All numbers are on the **public** [Kaggle *ussimandsegm* (Vitale et al.)](https://www.kaggle.com/datasets/ignaciorlando/ussimandsegm)
abdominal ultrasound set, with simulator content from open [TotalSegmentator](https://github.com/wasserthal/TotalSegmentator)
CT (see [Data](#data)).

## Repository layout

```
src/usbg/            the SonoSPADE code (historical package name, kept so imports match the release)
  texture_gan.py       SonoSPADE generator + trainer + CUT/CycleGAN baselines + inference translators
  segmenter.py         S_US: the free frozen segmenter (pseudo-labels + consistency)
  texture_eval.py      per-tissue metrics (locality, organ-W1, speckle SNR, TSTR)
  volume.py            CT → canonical tissue label volume + acoustic tables
  render_lotus.py      label slice → physics B-mode (pure torch/numpy; the SonoSPADE substrate)
  slicer.py, placement.py   probe pose → 2D label slice; organ-centred pose seating
  _vendor/             self-contained SE(3) geometry + scan geometry (vendored; no external repo needed)
scripts/             training, evaluation, and experiment drivers
paper/               SASHIMI submission package: LNCS source + compiled main_sashimi.pdf,
                     supplementary.pdf, figures, bibliography (README.txt = build steps)
docs/                SONOSPADE.md (method notes)
configs/             sonospade.yaml (hyperparameters)
data/                README with data-download / regeneration instructions (data itself not shipped)
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # core (texture synthesis + evaluation)
pip install -e ".[viz]"     # + figures / previews
```
Runs on CPU or Apple-Silicon MPS (`export PYTORCH_ENABLE_MPS_FALLBACK=1`); no discrete GPU required.
The import package is `usbg`; SonoSPADE is `usbg.texture_gan` (+ `usbg.segmenter`).

## Quickstart

```python
import numpy as np, usbg.volume as V, usbg.render_lotus as R
from usbg.texture_gan import load_translator          # dual-input aware loader

lv  = V.synthetic_phantom()                            # a labeled phantom (no download)
lab = lv.labels[lv.labels.shape[0] // 2]               # a 2D label slice
phys = R.render(np.asarray(lab), renderer="physics")   # physics B-mode render
son  = load_translator("data/dataset/exp_real_sonospade.pt")   # after training (below)
real_like = son.translate_aligned(phys, lab)           # per-tissue-textured frame
```

## Data

Everything runs on **open** datasets; no private data is required or shipped. See
[`data/README.md`](data/README.md) for one-command download/ingest of the Kaggle *ussimandsegm* real
ultrasound set and the TotalSegmentator CT content.

## Reproduce the paper

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
# one-command sim→real head-to-head (S_US -> SonoSPADE + baselines -> per-tissue metrics + FID/KID)
bash scripts/run_sonospade_experiment_real.sh
# downstream transfer (geometry remap + test-time adaptation)
python scripts/curvi_warp.py ; python scripts/eval_tta_tent.py
# latency (measured): SonoSPADE vs a representative diffusion U-Net, same hardware
python scripts/bench_latency.py ; python scripts/bench_diffusion_latency.py
```

## Closed-loop acquisition RL

The paper reports a brief in-the-loop proof of concept (an agent whose reward is a real-trained organ
recognizer). The full closed-loop RL study (method, seeds, and the multi-organ analysis) is **not
included in this release**; see [`scripts/closed_loop_rl/README.md`](scripts/closed_loop_rl/README.md).

## How to cite

If you use this code or ideas, please cite the paper and link this repository:

```bibtex
@inproceedings{sonospade2026,
  title     = {SonoSPADE: Real-Time, Per-Tissue Ultrasound Texture Synthesis for Closed-Loop Acquisition},
  author    = {Intharah, Thanapong and Gao, Zhichao and Dong, Hao},
  booktitle = {Simulation and Synthesis in Medical Imaging (SASHIMI), MICCAI Workshop},
  year      = {2026},
  note      = {Code: https://github.com/tohnperfect/SonoSpade}
}
```

## Limitations

Unpaired training matches per-tissue *statistics*, not exact speckle: this is training texture, not
diagnostic imagery. Acoustically similar soft tissues are near-inseparable in B-mode. Validation is a
single dataset/anatomy (abdominal Kaggle, 60 annotated real frames). See the paper's Limitations.

## Acknowledgements & license

Built on LOTUS-style physics rendering and [TotalSegmentator](https://github.com/wasserthal/TotalSegmentator)
CT labels; evaluated on the public [Kaggle *ussimandsegm*](https://www.kaggle.com/datasets/ignaciorlando/ussimandsegm)
(Vitale et al.) ultrasound set. `src/usbg/_vendor/` contains self-contained copies of the authors' SE(3)
and B-mode scan geometry so the release runs with no external repo. Released under the terms in
[`LICENSE`](LICENSE).
