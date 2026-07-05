# SonoSPADE

**Real-time, GPU-free, per-tissue ultrasound texture synthesis for closed-loop acquisition.**

SonoSPADE is a physics-conditioned, *per-tissue* ultrasound texture stage: it refines a physics
B-mode render into realistic ultrasound under spatially-adaptive conditioning from the **tissue label
slice**, trained **unpaired with no real annotations**. A single generator pass runs at **15–85 fps
without a GPU** — two to three orders of magnitude faster (per frame) than a semantic-diffusion
synthesizer — so it can sit inside a closed-loop reinforcement-learning acquisition simulator, where a
new frame must be rendered at every agent step.

📄 The paper (LNCS, SASHIMI-format) is in [`paper/main.pdf`](paper/main.pdf).

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
| **Downstream TSTR** (train-on-synthetic Dice) | overtakes raw render on every seed | — | only after matching sector geometry + test-time adaptation; absolute Dice stays low |
| **In-the-loop liver acquisition** (ground-truth, 3 seeds) | **highest, 3/3 seeds** | tie/below | a *small, liver-only* proof of concept — see below |

Summary figures: [`assets/premise_summary.png`](assets/premise_summary.png),
[`assets/organ_rl_summary.png`](assets/organ_rl_summary.png). Full honest write-up of the closed-loop
experiments (including what does **not** work): [`docs/PREMISE_CHECK_RESULT.md`](docs/PREMISE_CHECK_RESULT.md).

## Repository layout

```
src/usbg/            the SonoSPADE code (package name `usbg` = the closed-loop US testbed it lives in)
  texture_gan.py       SonoSPADE generator + trainer + CUT/CycleGAN baselines + inference translators
  segmenter.py         S_US: the free frozen segmenter (pseudo-labels + consistency)
  texture_eval.py      per-tissue metrics (locality, organ-W1, speckle SNR, TSTR)
  volume.py            CT → canonical tissue label volume + acoustic tables
  render_lotus.py      label slice → physics B-mode (pure torch/numpy; the SonoSPADE substrate)
  slicer.py            probe pose → 2D label slice
  _vendor/             self-contained SE(3) geometry + scan geometry (vendored; no external repo needed)
  (mujoco_env, goal_rl, placement, policy, rl ... : the closed-loop acquisition environment)
scripts/             training, evaluation, and experiment drivers
  premise/             the per-tissue reward-viability check + organ-conditioned RL experiment
paper/               LNCS source + compiled PDF (main.pdf)
docs/                SONOSPADE.md (method notes), PREMISE_CHECK_RESULT.md (closed-loop results)
configs/             sonospade.yaml (hyperparameters)
data/                README with data-download / regeneration instructions (data itself not shipped)
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                # core (texture synthesis + evaluation)
pip install -e ".[rl,viz]"      # + closed-loop env (mujoco) + figures
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

## Reproduce the paper

See [`data/README.md`](data/README.md) to obtain the Kaggle Vitale real set and TotalSegmentator CT.
Then:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
# one-command sim→real head-to-head (S_US -> SonoSPADE + baselines -> per-tissue metrics + FID/KID)
bash scripts/run_sonospade_experiment_real.sh
# downstream transfer (geometry remap + test-time adaptation)
python scripts/curvi_warp.py ; python scripts/eval_tta_tent.py
# latency (measured): SonoSPADE vs a representative diffusion U-Net, same hardware
python scripts/bench_latency.py ; python scripts/bench_diffusion_latency.py
```

## Closed-loop experiments (`scripts/premise/`, `scripts/train_organ_rl.py`)

The distinctive "does per-tissue texture *help an agent*?" study. A **real-trained** organ recognizer
`J` (trained only on real Kaggle frames — non-circular) scores an organ-acquisition RL agent by
ground-truth organ visibility.

```bash
# 1) reward-viability gate: does J recognize SonoSPADE's organ >> a global recolor's?
bash scripts/premise/rerun_all.sh
# 2) organ-conditioned liver acquisition RL: physics/CUT/CycleGAN/SonoSPADE x 3 seeds
bash scripts/premise/run_organ_rl.sh && python scripts/premise/agg_organ_rl.py
```

**Honest finding.** SonoSPADE's *liver* is recognized far better than a global recolor's (IoU
0.48 vs 0.14–0.21), and the agent reaches the highest ground-truth liver visibility in **3/3 seeds** —
a **small, seed-robust, liver-only** win. Kidney/spleen/gallbladder are **not** reward-viable: no
translator (SonoSPADE included) renders them recognizably to a real recognizer, a simulator/translator
limitation, not a recognizer one. Details and figures in
[`docs/PREMISE_CHECK_RESULT.md`](docs/PREMISE_CHECK_RESULT.md).

## Limitations

Unpaired training matches per-tissue *statistics*, not exact speckle — this is training texture, not
diagnostic imagery. Acoustically similar soft tissues are near-inseparable in B-mode. Validation is a
single dataset/anatomy (abdominal Kaggle, 60 annotated real frames). The in-the-loop win is modest and
liver-only. See the paper's Limitations section.

## Citation

```bibtex
@inproceedings{sonospade,
  title     = {SonoSPADE: Real-Time, Per-Tissue Ultrasound Texture Synthesis for Closed-Loop Acquisition},
  author    = {Anonymous},
  booktitle = {Simulation and Synthesis in Medical Imaging (SASHIMI), MICCAI Workshop},
  year      = {2026}
}
```

## Acknowledgements & license

Built on LOTUS-style physics rendering and TotalSegmentator CT labels; evaluated on the public Kaggle
*ussimandsegm* (Vitale et al.) real ultrasound set. `src/usbg/_vendor/` contains self-contained copies
of the authors' SE(3) geometry and B-mode scan geometry so the release runs with no external repo.
Code released under the terms in [`LICENSE`](LICENSE).
