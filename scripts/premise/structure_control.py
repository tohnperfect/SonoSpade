#!/usr/bin/env python3
"""structure_control.py: the texture-agnostic half of the double dissociation (handoff Sec 3.3).

The premise eval shows SonoSPADE's LIVER is recognized by a real recognizer better than a global
recolor's. Is that because SonoSPADE preserves anatomy better, or because its per-tissue APPEARANCE
is more real-like? This control measures STRUCTURE preservation only -- edge-IoU between each
translator's output and the physics render (texture_gan.edge_iou), which is texture/appearance
agnostic. If CUT preserves structure as well as SonoSPADE (they tie here) yet SonoSPADE wins the
organ-recognition eval, the recognition win is about per-tissue appearance, not structure (the
double dissociation). CycleGAN, which visibly scrambles anatomy, should score low here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import load_sim  # noqa: E402

CKPTS = {"cut": "data/dataset/exp_real_cut.pt",
         "cyclegan": "data/dataset/exp_real_cyclegan.pt",
         "sonospade": "data/dataset/exp_real_sonospade.pt"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="data/dataset/pairs_med.npz")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="outputs/premise/structure_control.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    from usbg.render_lotus import get_device
    from usbg.texture_gan import load_translator, edge_iou
    device = args.device or get_device()
    translators = {k: load_translator(v, device) for k, v in CKPTS.items()}
    imgs, labs_canon, _ = load_sim(args.sim)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(imgs))[:args.n]

    scores = {k: [] for k in translators}
    for i in idx:
        ph = imgs[i]
        for name, t in translators.items():
            out = t.translate_aligned(ph, labs_canon[i]) if getattr(t, "wants_label", False) else t(ph)
            scores[name].append(edge_iou(ph, out))
    summary = {name: {"edge_iou_mean": float(np.mean(v)), "edge_iou_std": float(np.std(v))}
               for name, v in scores.items()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"structure preservation (edge-IoU vs physics render, n={len(idx)}):")
    for name in ["cut", "cyclegan", "sonospade"]:
        s = summary[name]
        print(f"  {name:10s} {s['edge_iou_mean']:.3f} +- {s['edge_iou_std']:.3f}")
    print(f"wrote {args.out}")
    print("\nInterpretation: if CUT ~ SonoSPADE here but SonoSPADE wins the organ-recognition eval,\n"
          "the recognition win is about per-tissue APPEARANCE, not structure preservation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
