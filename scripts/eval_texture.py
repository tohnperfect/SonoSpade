#!/usr/bin/env python3
"""P4: per-tissue texture evaluation and the baseline comparison table.

Runs the SonoSPADE metric suite (the numeric answer to S-CycleGAN's open problem) over a set of
translators, on a shared set of labeled physics frames plus a labeled real set, and writes a JSON
results file and a printed table. Every translator is scored with the SAME protocol:

  * per-class speckle SNR (closeness to the real per-class SNR),
  * per-class GLCM contrast / homogeneity,
  * per-class intensity Wasserstein-1 to the real pool (lower is better),
  * edge_iou structure preservation (physics input vs translation, higher is better),
  * TSTR Dice (train a segmenter on the translated frames + labels, test on real labeled frames).

Baselines (handoff): raw physics (no translation), cyclegan, cut, cyclegan_sc (the S-CycleGAN style
structure-consistent reimpl), and sonospade (proposed). SPADE (paired upper bound) is scored when a
paired checkpoint is supplied. Translators that are not available are skipped and noted.

Usage:
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  python scripts/eval_texture.py --pairs data/dataset/usbg_labeled.npz \\
      --real-pairs data/eval/kaggle_vitale.npz \\
      --sonospade data/dataset/texture_sonospade.pt \\
      --cyclegan data/dataset/texture_cyclegan.pt --cut data/dataset/texture_cut.pt \\
      --out outputs/texture_eval.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _load_pairs(path, limit=None):
    from usbg.dataset import load_pairs_npz
    imgs, labs, _sess, _meta = load_pairs_npz(path)
    imgs = imgs.astype(np.float32) / 255.0
    labs = labs.astype(np.int64)
    if limit:
        imgs, labs = imgs[:limit], labs[:limit]
    return imgs, labs


def _translate_pool(variant, ckpt, phys_imgs, labels, device):
    """Return the translated (image) pool for a variant. 'physics' is the identity (raw render)."""
    import numpy as _np
    from usbg import texture_gan as TG
    if variant == "physics":
        return [_np.asarray(im, _np.float32) for im in phys_imgs]
    if variant == "sonospade":
        tr = TG.SonoSPADETranslator(ckpt, device=device)
        # labels are already aligned to the image, so use translate_aligned (no transpose dance).
        return [tr.translate_aligned(im, lab) for im, lab in zip(phys_imgs, labels)]
    # single-channel translators (cyclegan | cut | cyclegan_sc)
    tr = TG.TextureTranslator(ckpt, device=device)
    return [tr.translate(im) for im in phys_imgs]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/dataset/usbg_labeled.npz",
                    help="aligned (physics image, label) pairs (the content to translate + labels)")
    ap.add_argument("--real-pairs", default=None,
                    help="labeled real set (image,label) for per-class W1 and TSTR (e.g. Kaggle Vitale)")
    ap.add_argument("--limit", type=int, default=200, help="cap frames per pool for a fast eval")
    ap.add_argument("--physics", action="store_true", default=True)
    ap.add_argument("--cyclegan", default=None)
    ap.add_argument("--cut", default=None)
    ap.add_argument("--cyclegan-sc", default=None)
    ap.add_argument("--sonospade", default=None)
    ap.add_argument("--spade", default=None, help="paired SPADE upper bound (optional)")
    ap.add_argument("--tstr-iters", type=int, default=600)
    ap.add_argument("--out", default="outputs/texture_eval.json")
    args = ap.parse_args(argv)

    from usbg import texture_eval as TE
    from usbg import texture_gan as TG
    from usbg.render_lotus import get_device
    from usbg.volume import N_LABELS
    device = get_device()

    phys_imgs, labels = _load_pairs(args.pairs, args.limit)
    print(f"content pool: {len(phys_imgs)} labeled physics frames ({phys_imgs.shape[1:]})")

    real_imgs = real_labs = None
    if args.real_pairs and os.path.exists(args.real_pairs):
        real_imgs, real_labs = _load_pairs(args.real_pairs, args.limit)
        print(f"real labeled pool: {len(real_imgs)} frames")
    else:
        print("no --real-pairs given: per-class W1 vs real and TSTR are skipped (FLAG: distributional "
              "metrics need a labeled real set, e.g. Kaggle Vitale)")

    variants = [("physics", None)]
    for name, ck in [("cyclegan", args.cyclegan), ("cut", args.cut),
                     ("cyclegan_sc", args.cyclegan_sc), ("sonospade", args.sonospade),
                     ("spade", args.spade)]:
        if ck and os.path.exists(ck):
            variants.append((name, ck))
        elif ck:
            print(f"  {name}: checkpoint {ck} not found, skipping")

    real_snr = (TE.summarize_per_class([TE.speckle_snr_per_class(im, lb)
                for im, lb in zip(real_imgs, real_labs)]) if real_imgs is not None else {})

    results = {"n_content": len(phys_imgs), "classes": N_LABELS, "variants": {}}
    for variant, ck in variants:
        print(f"scoring {variant} ...")
        trans = _translate_pool(variant, ck, phys_imgs, labels, device)
        # per-frame per-class metrics, then reduce across frames
        snr = TE.summarize_per_class([TE.speckle_snr_per_class(t, lb)
                                      for t, lb in zip(trans, labels)])
        glcm_c = TE.summarize_per_class([{c: v["contrast"] for c, v in
                                          TE.glcm_per_class(t, lb).items()}
                                         for t, lb in zip(trans, labels)])
        glcm_h = TE.summarize_per_class([{c: v["homogeneity"] for c, v in
                                          TE.glcm_per_class(t, lb).items()}
                                         for t, lb in zip(trans, labels)])
        edge = float(np.mean([TE.edge_iou(p, t) for p, t in zip(phys_imgs, trans)]))
        entry = {
            "speckle_snr": {str(k): v for k, v in snr.items()},
            "snr_closeness_to_real": {str(k): abs(v - real_snr[k]) for k, v in snr.items()
                                      if k in real_snr},
            "glcm_contrast": {str(k): v for k, v in glcm_c.items()},
            "glcm_homogeneity": {str(k): v for k, v in glcm_h.items()},
            "edge_iou_vs_physics": edge,
        }
        # per-tissue sensitivity: only meaningful for a dual-input (label-conditioned) translator
        if variant == "sonospade" and ck:
            tr = TG.SonoSPADETranslator(ck, device=device)
            sens = TE.summarize_per_class([
                {c: v["locality"] for c, v in
                 TE.label_swap_sensitivity(tr.translate_aligned, im, lb).items()}
                for im, lb in zip(phys_imgs[:40], labels[:40])])
            entry["label_swap_locality"] = {str(k): v for k, v in sens.items()}
            entry["label_swap_locality_mean"] = (float(np.mean(list(sens.values())))
                                                 if sens else float("nan"))
        if real_imgs is not None:
            w1 = TE.intensity_w1_per_class(trans, labels, real_imgs, real_labs)
            entry["intensity_w1_per_class"] = {str(k): v for k, v in w1.items()}
            entry["intensity_w1_mean"] = float(np.mean(list(w1.values()))) if w1 else float("nan")
            # organ-only W1 (exclude background): whole-frame W1 is dominated by the background, which
            # a global recolor trivially matches; per-tissue realism is what the method is about
            from usbg.volume import BACKGROUND
            organs = [v for k, v in w1.items() if int(k) != BACKGROUND]
            entry["intensity_w1_organ_mean"] = float(np.mean(organs)) if organs else float("nan")
            mean_dice, per_class = TE.tstr_dice(trans, labels, real_imgs, real_labs, device=device,
                                                iters=args.tstr_iters)
            entry["tstr_dice"] = mean_dice
            entry["tstr_dice_per_class"] = {str(k): v for k, v in per_class.items()}
        results["variants"][variant] = entry

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote results -> {args.out}")
    print_table(results)
    return 0


def print_table(results):
    """Compact summary table across variants. Bold-by-convention is left to the paper table
    generator; here we print the raw comparable numbers (edge IoU, mean W1, TSTR Dice)."""
    print("\n" + "=" * 64)
    print(f"{'variant':14s} {'edge_iou':>9s} {'W1_mean':>9s} {'TSTR_dice':>10s}")
    print("-" * 64)
    for v, e in results["variants"].items():
        w1 = e.get("intensity_w1_mean", float("nan"))
        dice = e.get("tstr_dice", float("nan"))
        print(f"{v:14s} {e['edge_iou_vs_physics']:9.3f} {w1:9.3f} {dice:10.3f}")
    print("=" * 64)
    print("edge_iou: higher = structure preserved. W1_mean: lower = closer to real. "
          "TSTR_dice: higher = texture transfers.")


if __name__ == "__main__":
    raise SystemExit(main())
