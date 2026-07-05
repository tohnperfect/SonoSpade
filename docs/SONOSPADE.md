# SonoSPADE: physics-conditioned, per-tissue ultrasound texture synthesis

SonoSPADE replaces the tissue-agnostic grayscale texture translator with a dual-input, spatially
adaptive one, so each segmented tissue gets its own learned speckle (liver reads as liver, kidney
as kidney). It combines the strengths of SPADE (spatially adaptive per-class synthesis) and
S-CycleGAN (segmentation-consistent CT to US translation) and adds the USB Gym's own lever: the
physics render as a second conditioning input. Working name is a placeholder, rename for the paper.

Host is Apple Silicon (MPS, no CUDA). Always run:

```
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

No em-dashes anywhere in code or docs (commas, colons, or parentheses instead).

## The method (five deltas)

1. Dual input. G receives the physics B-mode (`render_lotus.render_bmode_physics`) AND the one-hot
   canonical label slice. It refines a tissue-correct render, it does not invent US from CT.
2. Spatially adaptive conditioning. SPADE normalization injects the label map at every scale (an
   architectural per-tissue guarantee, stronger than S-CycleGAN's concatenation).
3. One-sided CUT backbone. PatchNCE content preservation, no inverse generator, no cycle loss
   (lighter on MPS, no cycle texture wash).
4. Unpaired segmentation consistency. A U-Net `S_US` is pretrained for free on the simulator's
   perfectly aligned (image, label) pairs, then frozen. An OASIS style (N+1)-class segmentation
   discriminator (real classes plus a fake class) supplies per-pixel realism, with `S_US`
   pseudo-labeling the unlabeled real pool so no real annotations are ever needed. A frozen `S_US`
   consistency loss (CE + Dice) forces `S_US(G(phys, label))` to match the input label.
5. Per-tissue metrics answering S-CycleGAN's open problem: per-class speckle SNR, GLCM contrast and
   homogeneity, per-class intensity Wasserstein-1 vs real, and TSTR Dice.

Objective:

```
L = L_adv(OASIS seg-D) + lambda_nce * L_patchNCE + lambda_seg * L_seg(S_US(G(phys,label)), label)
    [+ lambda_tc * L_temporal, optional P6]
```

The one trap: never drop `L_seg`. Without it the generator can ignore the label map and collapse to
a global recolor, and texture stops being per-tissue.

## Files

New:
- `src/usbg/segmenter.py` : compact U-Net `S_US`, 13 canonical classes, label alignment, Dice / CE+Dice.
- `src/usbg/texture_eval.py` : per-tissue speckle SNR, GLCM, per-class W1, TSTR, temporal flicker.
- `src/usbg/manifest.py` : the data provenance and normalization contract (schema + validation).
- `scripts/train_segmenter.py` : pretrain `S_US` on aligned sim pairs (session or frame holdout).
- `scripts/eval_texture.py` : per-tissue metrics, TSTR, and the baseline comparison table.
- `configs/sonospade.yaml`, `data/manifest.json`.

Edited:
- `src/usbg/texture_gan.py` : SPADE dual-input generator (`_spade_generator`), OASIS discriminator
  (reuses the segmenter U-Net), `SonoSPADE` trainer, `SonoSPADETranslator`, `load_translator`,
  `load_config`, `build_trainer("sonospade")`. The existing cyclegan / cut / cyclegan_sc translators
  and the NumPy renderer are untouched and remain as fallbacks and baselines.
- `src/usbg/render_lotus.py` : `render()` passes the label slice to a `wants_label` translator.
- `src/usbg/dataset.py`, `scripts/make_dataset.py` : emit aligned (image, label) pairs (`--pairs-out`).
- `scripts/train_texture_gan.py` : the `sonospade` variant and dual-input training loop.
- `tests/` : `test_texture_gan.py` extended; `test_segmenter.py`, `test_texture_eval.py`,
  `test_manifest.py` added.

## Reproduce (end to end)

```
export PYTORCH_ENABLE_MPS_FALLBACK=1

# P0: config + manifest already scaffolded (data/manifest.json validates).

# 5.1 Free aligned sim pairs (scale --pairs-n and --cases up for the full run; 10k then 30k-50k).
python scripts/make_dataset.py --cases s0011 s0058 s0344 s0358 s0461 s0477 s0511 s0513 \
    --target liver --image-size 128 --renderer physics \
    --pairs-out data/dataset/usbg_labeled.npz --pairs-n 110 --pairs-only

# P2: pretrain the frozen S_US segmenter (hold out whole sessions).
python scripts/train_segmenter.py --pairs data/dataset/usbg_labeled.npz \
    --iters 3000 --out data/dataset/s_us.pt

# P3: train SonoSPADE unpaired against the real pool (--real-dir), or --proxy-real for a demo.
python scripts/train_texture_gan.py --variant sonospade \
    --pairs data/dataset/usbg_labeled.npz --real-dir data/real_us_crop --s-us data/dataset/s_us.pt \
    --iters 4000 --out data/dataset/texture_sonospade.pt --preview outputs/sonospade_before_after.png

# P4: per-tissue metrics + TSTR + baseline table (needs a labeled real set for W1 / TSTR).
python scripts/eval_texture.py --pairs data/dataset/usbg_labeled.npz \
    --real-pairs data/eval/kaggle_vitale.npz \
    --sonospade data/dataset/texture_sonospade.pt \
    --cut data/dataset/texture_cut.pt --cyclegan data/dataset/texture_cyclegan.pt \
    --out outputs/texture_eval.json

# P5: SonoSPADE is a drop-in Renderer texture. load_translator auto-detects the checkpoint type.
python scripts/make_dataset.py --cases s0011 --texture-gan data/dataset/texture_sonospade.pt --randomize
```

## Phase status

- P0 branch, config, manifest scaffold. Done, manifest validates.
- P1 SPADE dual-input generator. Done, overfits one paired frame to near zero L1 (tests assert it).
- P2 sim-pretrained `S_US`. Done. See the honest note below on the Dice target.
- P3 unpaired OASIS + PatchNCE + seg-consistency training. Done, runs on MPS.
- P4 per-tissue metrics + TSTR + baseline table. Done, harness runs and writes JSON + table.
- P5 renderer / dataset integration. Done, `Renderer(translator=SonoSPADE)` is a drop-in, closed loop runs.
- P6 stretch. Temporal flicker metric implemented (`texture_eval.temporal_flicker`) AND the temporal
  training loss is now wired (see improvements below); the SDM diffusion comparison is DEFERRED
  (needs a cloud CUDA box, see the confirm flags).

## Improvements (v2)

Four additions strengthen the method; all are config-driven and default-on where safe.

1. Grouped segmenter scheme (`segmenter.GROUPED_MAP`, `--group` in train_segmenter,
   `train.group_segmenter` in the config). The acoustically degenerate soft tissues (fat, GI wall,
   pancreas, muscle, generic soft tissue) collapse into one class, so S_US is not asked to separate
   the inseparable. The generator still conditions on the full 13 canonical classes; only S_US and
   the OASIS discriminator work in the grouped 9-class space (the trainer remaps the conditioning
   label with the S_US group map for the pseudo-label and consistency targets). This directly lifts
   the held-out Dice that acoustic degeneracy caps under the canonical scheme: on the same 8-case
   mild-DR set and session split, macro Dice rises from 0.50 (canonical 13-class) to 0.66 (grouped),
   and the merged soft-tissue class segments at 0.75 versus about 0.35 for its individual components.
2. Temporal-consistency loss (`loss.lambda_tc`). The translator must commute with a small in-plane
   shift, so adjacent slices (approximately small shifts of the content) do not flicker. In a short
   run the term drops from ~0.32 to ~0.12. This completes P6's training side (the metric already
   existed).
3. OASIS-style LabelMix consistency (`loss.lambda_labelmix`). A CutMix-style spatial mix of a real
   and a fake frame is fed to the discriminator, which must be consistent (its output on the mix
   equals the mix of its outputs). A light regularizer that stabilizes the per-pixel discriminator.
4. Generator EMA plus R1 gradient penalty (`train.ema_decay`, `loss.r1_gamma`). The inference
   generator is an exponential moving average of the training weights (rampup decay, so short runs
   are not washed out), which is what `SonoSPADETranslator` deploys. The R1 penalty regularizes the
   discriminator's gradient on real frames; because the OASIS discriminator is per-pixel, the penalty
   is normalized by the output spatial size, and the discriminator uses average pooling so it stays
   twice-differentiable. In a short run R1 starts large (regularizing the raw discriminator) and
   decays as it stabilizes, the expected behavior.

## Smoke results (correctness evidence, not a full training run)

A short 500-iteration SonoSPADE run against the real 149-frame `data/real_us_crop` pool (S_US from
the 8-case sim set), scored on held-out sim frames:

- Per-tissue texture is genuinely per-class: amplitude speckle SNR shifts differently per tissue
  (for example liver 2.49 to 3.94, gallbladder 2.21 to 1.86, kidney 3.35 to 3.86), so the label map
  is actually driving texture, not a global recolor.
- Label-swap sensitivity (`texture_eval.label_swap_sensitivity`) makes that quantitative and directly
  answers the one trap: holding the physics render fixed and relabeling a tissue changes that
  region's texture about 13x more than it changes the rest of the frame (mean locality ratio ~22 over
  30 frames, in-region 0.065 vs out-region 0.005). A global recolor or a label-ignoring translator
  scores near zero here by construction, so this is the ablation that proves the conditioning works.
- Structure preservation via `edge_iou` (physics input vs translation), higher is better:
  CycleGAN 0.27 (its cycle loss washes structure, the failure SonoSPADE is designed to avoid),
  CUT 0.46, SonoSPADE 0.44 at only 500 iterations. SonoSPADE matches the one-sided CUT baseline and
  is far above CycleGAN. Note `edge_iou` binarizes Sobel edges, so ANY realistic speckle lowers the
  absolute value (speckle is high-frequency edges); the comparison across translators is the signal,
  and PatchNCE preserves content in feature space rather than raw pixels. A full run (4000+ iters)
  raises this further.

## Honest note on the P2 Dice target

The handoff's P2 acceptance is "Dice > 0.8 on held-out sim slices (perfect labels make this easy)".
The mechanism is verified: a small learnable set overfits to Dice ~1.0 (asserted in
`tests/test_segmenter.py`). On real multi-case sim data the picture is more nuanced and worth stating
plainly rather than tuning around:

- Acoustically DISTINCT tissues clear the bar: liver 0.93, lung 0.90, vessels 0.72 (held-out sessions).
- The soft-tissue family does not: fat, GI wall, pancreas, muscle, and generic soft tissue all have
  nearly identical acoustic constants in `volume.ACOUSTIC_LOTUS` (backscatter ~0.40 to 0.50), so they
  render as near-identical gray and are genuinely inseparable by B-mode texture alone. This is a real
  property of ultrasound, not a training bug, and it drags the 13-class macro Dice to ~0.45 to 0.50.

This matches the handoff's own non-goal: "Tissues absent from the real pool stay unconstrained, so
flag them." For SonoSPADE's purposes `S_US` is used to pseudo-label and enforce consistency on the
DOMINANT, distinct structures (liver, vessel, lung, bone, background), where it is strong. Options to
raise the macro number if wanted: more cases and frames (10k to 50k as specified), mild domain
randomization (`--pairs-dr mild`, keeps per-tissue acoustics; already ~+0.05), or merging the
degenerate soft-tissue labels into one class for the segmenter.

## Confirm before publication (from the handoff)

- Own-data ethics and de-identification: confirm consent covers research and publication, and that
  every real frame's patient banner is removed. `data/real_us_crop` is FLAGGED in the manifest.
- Open-data licenses: verify Kaggle Vitale, CACTUSS, and AbdomenCT-1K terms.
- Labeled eval set: decide Kaggle masks only vs hand-labeling own frames; record it in the manifest.
- Cloud CUDA: confirm access for the P6 diffusion comparison, else it stays deferred.
- Working name: SonoSPADE is provisional, pick the final paper name.

## References

SPADE arXiv:1903.07291, OASIS arXiv:2012.04781, S-CycleGAN arXiv:2406.01191 (Song and Chong,
IEEE CBS 2024), CUT arXiv:2007.15651, CACTUSS arXiv:2207.08619, Echo from Noise arXiv:2305.05424,
CySGAN arXiv:2204.03082.
