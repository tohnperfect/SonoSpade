#!/usr/bin/env python3
"""Slim ablation eval: Edge IoU, organ W1, and label-swap locality for a set of SonoSPADE
checkpoints (no TSTR / GLCM, so it is fast). Every checkpoint is a sonospade-variant generator with
one training term toggled off; all share the same content pool and real test set as the main table.

Usage:
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  .venv/bin/python scripts/eval_ablation.py \
      --pairs data/dataset/pairs_med.npz --real-pairs data/eval/kaggle_vitale_test.npz \
      --ckpt full=/path/ckpt_full.pt --ckpt no_seg=/path/ckpt_no_seg.pt ... \
      --out outputs/experiment/ablation_real.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_texture import _load_pairs  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/dataset/pairs_med.npz")
    ap.add_argument("--real-pairs", default="data/eval/kaggle_vitale_test.npz")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--ckpt", action="append", default=[],
                    help="name=path, repeatable (each a sonospade checkpoint)")
    ap.add_argument("--out", default="outputs/experiment/ablation_real.json")
    args = ap.parse_args(argv)

    from usbg import texture_eval as TE
    from usbg import texture_gan as TG
    from usbg.render_lotus import get_device
    from usbg.volume import BACKGROUND
    device = get_device()

    phys_imgs, labels = _load_pairs(args.pairs, args.limit)
    real_imgs, real_labs = _load_pairs(args.real_pairs, args.limit)
    print(f"content {len(phys_imgs)}, real {len(real_imgs)}")

    results = {"variants": {}}
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"  {name}: {path} missing, skipping")
            continue
        tr = TG.SonoSPADETranslator(path, device=device)
        trans = [tr.translate_aligned(im, lab) for im, lab in zip(phys_imgs, labels)]
        edge = float(np.mean([TE.edge_iou(p, t) for p, t in zip(phys_imgs, trans)]))
        w1 = TE.intensity_w1_per_class(trans, labels, real_imgs, real_labs)
        organs = [v for k, v in w1.items() if int(k) != BACKGROUND]
        w1_organ = float(np.mean(organs)) if organs else float("nan")
        sens = TE.summarize_per_class([
            {c: v["locality"] for c, v in
             TE.label_swap_sensitivity(tr.translate_aligned, im, lb).items()}
            for im, lb in zip(phys_imgs[:40], labels[:40])])
        loc = float(np.mean(list(sens.values()))) if sens else float("nan")
        results["variants"][name] = {"edge_iou": edge, "w1_organ": w1_organ, "locality": loc}
        print(f"{name:10s} edge_iou {edge:6.3f}  W1_organ {w1_organ:6.3f}  locality {loc:7.2f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
