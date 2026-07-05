#!/usr/bin/env python3
"""Generate the SonoSPADE paper figures from a trained checkpoint.

figures/qualitative.png: rows of (physics render, SonoSPADE output, tissue label) for a few
held-out frames, plus a label-swap panel showing that relabeling a region changes its texture
locally. Written to paper/sonospade/figures.

Usage:
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  python scripts/fig_sonospade.py --ckpt data/dataset/exp_sonospade.pt --pairs data/dataset/pairs_med.npz
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/dataset/exp_sonospade.pt")
    ap.add_argument("--pairs", default="data/dataset/pairs_med.npz")
    ap.add_argument("--out", default="paper/sonospade/figures/qualitative.png")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from usbg.texture_gan import SonoSPADETranslator
    from usbg.dataset import load_pairs_npz
    from usbg.volume import LIVER, VESSEL

    tr = SonoSPADETranslator(args.ckpt)
    imgs, labs, _s, _m = load_pairs_npz(args.pairs)
    imgs = imgs.astype(np.float32) / 255.0
    labs = labs.astype(np.int64)
    sel = list(range(len(imgs) - args.n, len(imgs)))          # held-out tail

    fig, axes = plt.subplots(3, args.n + 1, figsize=(2.0 * (args.n + 1), 6.0), squeeze=False)
    for r in range(3):
        for c in range(args.n + 1):
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
    for k, i in enumerate(sel):
        out = tr.translate_aligned(imgs[i], labs[i])
        axes[0][k].imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
        axes[1][k].imshow(out, cmap="gray", vmin=0, vmax=1)
        axes[2][k].imshow(labs[i], cmap="tab20", vmin=0, vmax=13)
    axes[0][0].set_ylabel("physics", fontsize=11)
    axes[1][0].set_ylabel("SonoSPADE", fontsize=11)
    axes[2][0].set_ylabel("label", fontsize=11)

    # label-swap panel (last column): relabel the largest non-liver region to liver, show the diff
    i = sel[0]
    base = tr.translate_aligned(imgs[i], labs[i])
    lab = labs[i].copy()
    # pick a present non-liver, non-background class with the most pixels
    ids, counts = np.unique(lab, return_counts=True)
    cand = [(c, n) for c, n in zip(ids, counts) if c not in (0, LIVER)]
    swapped = lab.copy()
    if cand:
        c = max(cand, key=lambda t: t[1])[0]
        swapped[lab == c] = LIVER
    alt = tr.translate_aligned(imgs[i], swapped)
    axes[0][args.n].imshow(base, cmap="gray", vmin=0, vmax=1); axes[0][args.n].set_title("out", fontsize=9)
    axes[1][args.n].imshow(alt, cmap="gray", vmin=0, vmax=1); axes[1][args.n].set_title("label-swapped", fontsize=9)
    axes[2][args.n].imshow(np.abs(alt - base), cmap="magma"); axes[2][args.n].set_title("|diff| (local)", fontsize=9)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
