#!/usr/bin/env python3
"""Qualitative figure for Experiment A: the curvilinear pipeline.

Rows: curvilinear physics render -> SonoSPADE-curvi output -> a real Kaggle frame, showing that the
warped sim geometry now matches the clinical sector so a downstream segmenter meets the same spatial
prior. Saves paper/sonospade/figures/curvilinear.png.
"""
from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from usbg import texture_gan as TG
    from usbg.dataset import load_pairs_npz
    from usbg.render_lotus import get_device

    device = get_device()
    imgs, labs, _s, _m = load_pairs_npz("data/dataset/pairs_med_curvi.npz")
    imgs = imgs.astype(np.float32) / 255.0
    real = np.load("data/eval/kaggle_vitale_test.npz")["images"].astype(np.float32) / 255.0

    tr = TG.SonoSPADETranslator("data/dataset/exp_real_sonospade_curvi.pt", device=device)
    # pick a few content frames with visible structure (most non-background)
    idx = sorted(range(len(imgs)), key=lambda i: -(labs[i] > 0).mean())[:4]
    cols = len(idx)
    fig, ax = plt.subplots(3, cols, figsize=(2.1 * cols, 6.4))
    for c, i in enumerate(idx):
        out = tr.translate_aligned(imgs[i], labs[i])
        ax[0][c].imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
        ax[1][c].imshow(np.clip(out, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax[2][c].imshow(real[c % len(real)], cmap="gray", vmin=0, vmax=1)
        for r in range(3):
            ax[r][c].set_xticks([]); ax[r][c].set_yticks([])
    ax[0][0].set_ylabel("physics (curvi)", fontsize=11)
    ax[1][0].set_ylabel("SonoSPADE-curvi", fontsize=11)
    ax[2][0].set_ylabel("real Kaggle", fontsize=11)
    plt.tight_layout(pad=0.4)
    out_path = "paper/sonospade/figures/curvilinear.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
