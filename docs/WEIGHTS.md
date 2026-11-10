# SonoSPADE pretrained weights

Six trained checkpoints are attached to the GitHub release
[v0.2.0](https://github.com/tohnperfect/SonoSpade/releases/tag/v0.2.0), in two sets. Each set pairs
a SonoSPADE generator (the texture stage) with the frozen segmenter `S_US` it was trained against.
You only need the generator to make SonoSPADE frames; the segmenter is there for retraining,
ablations and the pseudo-label checks.

| File | Set | Kind | Geometry | Size |
|---|---|---|---|---|
| `sonospade_sashimi_linear.pt` | SASHIMI paper | generator | linear | 35.9 MB |
| `s_us_sashimi_linear.pt` | SASHIMI paper | segmenter | linear | 34.6 MB |
| `sonospade_sashimi_curvi.pt` | SASHIMI paper | generator | curvilinear | 35.9 MB |
| `s_us_sashimi_curvi.pt` | SASHIMI paper | segmenter | curvilinear | 34.6 MB |
| `sonospade_usbgym_curvi_s0.pt` | USB Gym | generator | curvilinear | 35.9 MB |
| `s_us_usbgym_curvi_s0.pt` | USB Gym | segmenter | curvilinear | 34.6 MB |

SHA-256 checksums are in [`weights/MANIFEST.json`](../weights/MANIFEST.json) and in `SHA256SUMS`
on the release page. The files are byte-for-byte the checkpoints that were trained; only their names
changed for the release.

## Which file to use

- **Reproducing or extending the SASHIMI paper:** the `sashimi` set. The linear pair is the one
  behind the paper's main comparison; the curvilinear pair is the one behind its geometry-matching
  and test-time-adaptation experiments (Sec. 4 of the paper).
- **Working in [USB Gym](https://github.com/tohnperfect/USBGym):** the `usbgym` set. It is the
  texture stage every SonoSPADE result in the USB Gym article was run with (its run records give
  the MD5 `70440a5c6167e69489835ef553bfe37c`, which is this file). Its training CTs exclude USB Gym's
  six test patients. The `sashimi` set was trained on eight CTs that include two of those test
  patients (`s0358`, `s0513`), so do not use it for USB Gym's held-out evaluation.
- **Your own simulator:** either set, chosen by scan geometry. Use `linear` for a rectangular
  linear-array frame and `curvi` for a curvilinear sector.

## Use

```bash
python scripts/fetch_weights.py --set sashimi      # or --set usbgym, or name single files
```

```python
import numpy as np, usbg.volume as V, usbg.render_lotus as R
from usbg.texture_gan import load_translator
from usbg.segmenter import Segmenter

lv   = V.synthetic_phantom()
lab  = np.asarray(lv.labels[lv.labels.shape[0] // 2])
phys = R.render(lab, renderer="physics")
son  = load_translator("weights/sonospade_sashimi_linear.pt")
frame = son.translate_aligned(phys, lab)                  # per-tissue-textured frame in [0, 1]

s_us = Segmenter.load("weights/s_us_sashimi_linear.pt")    # only for retraining / pseudo-labels
```

Input is a physics B-mode render in [0, 1] plus the tissue label map at the same size (13 canonical
classes, `usbg.volume`). The generators were trained at 128 x 128 and are fully convolutional, so
other sizes run; the closer to the training size, the closer to the training statistics. The code
loads every checkpoint with `torch.load(..., weights_only=True)`: the files hold only tensors and a
few integers.

## Checkpoint contents

- Generator: `G_spade` (the exponential-moving-average weights, used at inference), `G_raw` (the raw
  training weights), and the architecture integers `n_classes` (13), `ngf` (32), `n_blocks` (6),
  `spade_hidden` (64). 4.48 M parameters.
- Segmenter: `state` (U-Net weights), `n_classes` (9), `base` (32), `depth` (4) and `group_map`, which
  maps the 13 canonical classes to 9 by merging the soft tissues whose simulated backscatter is
  nearly identical. 8.64 M parameters.

## Training

All six were trained on an Apple-silicon Mac (MPS) with the scripts in this repository.

**Content (both sets):** physics renders of labelled abdominal CTs from the TotalSegmentator dataset,
with the label slice of each render as its aligned label. The segmenters saw only these simulated
pairs, never a real ultrasound frame.

**Real target (generators only):** the unlabelled real pool of the Kaggle *ussimandsegm* dataset
(Vitale et al.), 404 frames from its training split, prepared by `scripts/ingest_kaggle_us.py`. The
60 annotated real frames were held out for evaluation. The released files contain no images, and the
discriminator is not released.

**SASHIMI set:** 880 simulated pairs from eight CTs (`s0011`, `s0058`, `s0344`, `s0358`, `s0461`,
`s0477`, `s0511`, `s0513`; 110 views each); the curvilinear pair uses sector-warped versions of the
same views (`scripts/curvi_warp.py`). Generators: 1500 iterations, settings in
`configs/sonospade.yaml` and `configs/sonospade_curvi.yaml` (listed in the paper's supplementary
material).

**USB Gym set:** rendered with USB Gym's renderer and its anatomical probe seating, so it matches
USB Gym frames most closely.
- Generator: 4620 curvilinear pairs from eight training CTs (`s0011`, `s0058`, `s0344`, `s0461`,
  `s0477`, `s0511`, `s0522`, `s0703`), 1500 iterations, batch 4, seed 0, `configs/sonospade_curvi.yaml`.
- Segmenter: trained on five of those CTs (`s0011`, `s0058`, `s0511`, `s0522`, `s0703`) and
  validated on three (`s0344`, `s0461`, `s0477`), 2000 iterations, batch 8, learning rate 5e-4,
  seed 0.
- None of USB Gym's six test patients (`s0358`, `s0513`, `s0519`, `s0591`, `s0649`, `s0762`) was
  used.

## Intended use and limits

The weights produce training texture for ultrasound simulation, for example in closed-loop probe
guidance, reinforcement learning or segmentation pretraining. They are not diagnostic imagery and
are not for clinical use.

- Training is unpaired, so the frames match per-tissue intensity statistics, not the exact speckle
  of any real scan.
- Soft tissues with near-identical simulated backscatter (fat, bowel, pancreas, muscle, other soft
  tissue) share one segmenter class, so their textures are less distinct than those of the liver,
  kidney, spleen, gallbladder or vessels.
- One real dataset (abdominal, one scanner) defined the target appearance; other anatomy or scanners
  are out of distribution.

## License, data and citation

The weights are released by the authors under Apache-2.0, the license of this repository. The
training data keep their own terms:

- TotalSegmentator CT dataset, CC BY 4.0: Wasserthal et al., *Radiology: Artificial Intelligence*,
  2023, [doi:10.5281/zenodo.10047263](https://doi.org/10.5281/zenodo.10047263).
- Kaggle [*ussimandsegm*](https://www.kaggle.com/datasets/ignaciorlando/ussimandsegm) (generators
  only), whose page states "Data files © Original Authors" and asks users to cite Vitale, Orlando,
  Iarussi and Larrabide, *Improving realism in patient-specific abdominal ultrasound simulation using
  CycleGANs*, IJCARS 15(2):183-192 (2019). Using its frames, for example to retrain, follows the
  dataset's own terms.

If you use the weights, cite the SonoSPADE paper (see the README) and the two datasets above.
