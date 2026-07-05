#!/usr/bin/env python3
"""Ingest the Kaggle ussimandsegm real ultrasound set (Vitale) into the SonoSPADE pipeline.

Builds two artifacts from ~/Codes/Dataset/ussimandsegm/abdominal_US/abdominal_US/RUS:
  * an unlabeled REAL POOL (all 617 real US frames, grayscale, resized) as a folder of PNGs, the
    target distribution the translators learn (drop in via --real-dir),
  * a LABELED TEST npz (the 60 annotated frames, organ masks mapped to our canonical labels), the
    real head-to-head set for per-class W1 and TSTR.

The real US annotations color-code five organs (Kaggle legend): violet=liver, yellow=kidney,
green=gallbladder, pink=spleen, red=vessels (pancreas / adrenals / bones appear only in the simulated
set). Each pixel is mapped to the nearest canonical color, so the dark-purple background tint in one
mask and the anti-aliased boundaries resolve cleanly. All five are acoustically distinct tissues.

Usage:
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  python scripts/ingest_kaggle_us.py --src ~/Codes/Dataset/ussimandsegm --size 128 \\
      --pool-dir data/real/kaggle_vitale/pool --test-out data/eval/kaggle_vitale_test.npz \\
      --verify outputs/experiment/kaggle_ingest_verify.png
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from usbg.volume import BACKGROUND, LIVER, GALLBLADDER, VESSEL, KIDNEY, SPLEEN

# canonical annotation color (RGB in the files) -> our canonical tissue label. Background is black;
# the dark-purple (10,0,10) tint seen in one mask maps to background by nearest color.
COLOR_TO_LABEL = {
    (0, 0, 0): BACKGROUND,
    (100, 0, 100): LIVER,        # violet
    (255, 255, 0): KIDNEY,       # yellow
    (0, 255, 0): GALLBLADDER,    # green
    (255, 0, 255): SPLEEN,       # pink
    (255, 0, 0): VESSEL,         # red
}
_COLORS = np.array(list(COLOR_TO_LABEL.keys()), dtype=np.int16)          # (C,3)
_LABELS = np.array(list(COLOR_TO_LABEL.values()), dtype=np.int64)        # (C,)


def mask_to_canonical(rgb: np.ndarray) -> np.ndarray:
    """RGB mask (H,W,3) -> canonical label map (H,W) by nearest annotation color (handles the
    background tint and anti-aliased boundaries)."""
    h, w, _ = rgb.shape
    d = ((rgb[:, :, None, :].astype(np.int32) - _COLORS[None, None, :, :]) ** 2).sum(-1)  # (H,W,C)
    return _LABELS[d.argmin(-1)].reshape(h, w)


def _resize_gray(img_L, size):
    from PIL import Image
    return np.asarray(img_L.convert("L").resize((size, size), Image.BILINEAR), np.float32) / 255.0


def _robust_norm(a, lo=1.0, hi=99.0):
    p1, p99 = np.percentile(a, [lo, hi])
    return np.clip((a - p1) / (p99 - p1 + 1e-6), 0, 1).astype(np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.expanduser("~/Codes/Dataset/ussimandsegm"))
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--pool-dir", default="data/real/kaggle_vitale/pool")
    ap.add_argument("--test-out", default="data/eval/kaggle_vitale_test.npz")
    ap.add_argument("--pool-split", choices=["train", "all"], default="train",
                    help="train: the 404 real TRAIN frames (held out from the labeled test, no "
                         "leakage); all: every real frame (617, includes the labeled test frames)")
    ap.add_argument("--verify", default="outputs/experiment/kaggle_ingest_verify.png")
    args = ap.parse_args(argv)

    from PIL import Image
    from usbg.dataset import write_pairs_npz
    rus = os.path.join(args.src, "abdominal_US", "abdominal_US", "RUS")

    # 1. unlabeled real pool. Default = the TRAIN split only, so the labeled test frames are never
    # seen by the translators (leakage-free head-to-head). The labeled masks live under the TEST
    # split, so the train split is disjoint from the evaluation set by construction.
    os.makedirs(args.pool_dir, exist_ok=True)
    for old in glob.glob(os.path.join(args.pool_dir, "*.png")):
        os.remove(old)                                    # clean any prior (e.g. all-split) pool
    pool_files = sorted(glob.glob(os.path.join(rus, "images", "train", "*.jpg")))
    if args.pool_split == "all":
        pool_files += sorted(glob.glob(os.path.join(rus, "images", "test", "*.jpg")))
    for i, f in enumerate(pool_files):
        a = _robust_norm(_resize_gray(Image.open(f), args.size))
        Image.fromarray((a * 255).astype(np.uint8)).save(os.path.join(args.pool_dir, f"{i:05d}.png"))
    print(f"real pool: {len(pool_files)} frames -> {args.pool_dir}")

    # 2. labeled test: the annotated frames, image + canonical label (aligned, full frame resized)
    ann_files = sorted(glob.glob(os.path.join(rus, "annotations", "test", "*.png")))
    imgs, labs, names = [], [], []
    for a in ann_files:
        stem = os.path.basename(a)[:-4]
        img_path = os.path.join(rus, "images", "test", stem + ".jpg")
        if not os.path.exists(img_path):
            continue
        img = _robust_norm(_resize_gray(Image.open(img_path), args.size))
        m = np.asarray(Image.open(a).convert("RGB"))
        lab = mask_to_canonical(m)
        lab = np.asarray(Image.fromarray(lab.astype(np.uint8)).resize(
            (args.size, args.size), Image.NEAREST), np.int64)                # nearest: no class blend
        imgs.append((img * 255).astype(np.uint8)); labs.append(lab.astype(np.uint8)); names.append(stem)
    imgs = np.asarray(imgs); labs = np.asarray(labs)
    write_pairs_npz(args.test_out, imgs, labs, sessions=np.array(names),
                    extra_meta={"source": "kaggle ussimandsegm RUS (Vitale)",
                                "organs": "liver,kidney,gallbladder,spleen,vessel"})
    # report label coverage
    u, c = np.unique(labs, return_counts=True)
    from usbg.volume import USBG_LABEL_NAMES
    cov = {USBG_LABEL_NAMES.get(int(k), int(k)): round(100 * v / c.sum(), 2) for k, v in zip(u, c)}
    print(f"labeled test: {len(imgs)} frames -> {args.test_out}")
    print(f"  label coverage %: {cov}")

    # 3. verification overlay: image with its label boundaries, for a visual sanity check
    if args.verify:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = min(12, len(imgs))
        fig, axes = plt.subplots(2, n, figsize=(1.8 * n, 3.8), squeeze=False)
        for i in range(n):
            axes[0][i].imshow(imgs[i], cmap="gray"); axes[0][i].axis("off")
            axes[0][i].set_title(names[i], fontsize=7)
            axes[1][i].imshow(imgs[i], cmap="gray")
            axes[1][i].imshow(labs[i], cmap="tab10", vmin=0, vmax=9, alpha=0.45)
            axes[1][i].axis("off")
        axes[0][0].set_ylabel("real US", fontsize=9)
        axes[1][0].set_ylabel("label overlay", fontsize=9)
        os.makedirs(os.path.dirname(os.path.abspath(args.verify)), exist_ok=True)
        fig.suptitle("Kaggle Vitale real US: image (top) and canonical-label overlay (bottom)", fontsize=10)
        fig.tight_layout(); fig.savefig(args.verify, dpi=140); plt.close(fig)
        print(f"verification overlay -> {args.verify}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
