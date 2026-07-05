#!/usr/bin/env python3
"""Train an unpaired sim-to-real B-mode texture translator and save its generator.

Variants (--variant):
  cyclegan | cut | cyclegan_sc : the tissue-agnostic grayscale translators (single channel in/out).
        `sim` pool = physics renders (from a usbg dataset npz). `real` pool = real frames
        (--real-dir) or --proxy-real (a distinct US-like texture on a disjoint half of sim).
  sonospade : the proposed dual-input, per-tissue translator (P3). Content + conditioning come from
        aligned (image, label) pairs (--pairs, from make_dataset --pairs-out); the `real` pool is
        --real-dir or --proxy-real; a frozen S_US (--s-us, from train_segmenter.py) supplies the
        pseudo-labels for the OASIS discriminator and the segmentation-consistency loss.

Usage (proposed method):
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  python scripts/train_texture_gan.py --variant sonospade \\
      --pairs data/dataset/usbg_labeled.npz --real-dir data/real/kaggle_vitale/pool --s-us data/dataset/s_us.pt \\
      --iters 4000 --out data/dataset/texture_sonospade.pt --preview outputs/sonospade_before_after.png

Usage (existing CycleGAN, unchanged):
  python scripts/train_texture_gan.py --sim-npz data/dataset/usbg.npz --proxy-real --iters 500
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def load_real_dir(d, size):
    """Load all grayscale frames under a directory (recurses into subfolders), resized to size."""
    from PIL import Image
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    out = []
    for root, _dirs, files in os.walk(d):
        for fn in sorted(files):
            if fn.lower().endswith(exts):
                im = Image.open(os.path.join(root, fn)).convert("L").resize((size, size))
                out.append(np.asarray(im, np.float32) / 255.0)
    if not out:
        raise SystemExit(f"no images ({', '.join(exts)}) found under --real-dir {d}")
    return np.asarray(out, np.float32)


def train_grayscale(args):
    """The existing single-channel translators (cyclegan | cut | cyclegan_sc)."""
    import torch
    from usbg import texture_gan as TG
    from usbg.render_lotus import get_device

    device = get_device()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    frames = TG.frames_from_npz(args.sim_npz)
    rng.shuffle(frames)
    if args.real_dir:
        sim_pool, real_pool = frames, load_real_dir(args.real_dir, args.size)
        real_kind = f"real-dir ({len(real_pool)})"
    elif args.proxy_real:
        half = len(frames) // 2
        sim_pool = frames[:half]
        real_pool = np.stack([TG.proxy_real_texture(f, rng) for f in frames[half:]], 0)
        real_kind = "proxy-real"
    else:
        raise SystemExit("provide --real-dir or --proxy-real")
    print(f"[{args.variant}] sim {len(sim_pool)}, real {len(real_pool)} ({real_kind}), "
          f"device {device}, {args.iters} iters")

    gan = TG.build_trainer(args.variant, device, n_blocks=args.n_blocks)
    for it in range(args.iters):
        si = rng.integers(0, len(sim_pool), args.batch)
        ri = rng.integers(0, len(real_pool), args.batch)
        losses = gan.step(TG.to_batch(sim_pool, si, device), TG.to_batch(real_pool, ri, device))
        if (it + 1) % max(1, args.iters // 10) == 0:
            print("  iter %4d  " % (it + 1) + "  ".join(f"{k} {v:.3f}" for k, v in losses.items()))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    gan.save(args.out)
    print(f"saved generator -> {args.out}")

    if args.preview:
        gan.G_s2r.eval()
        show = sim_pool[:5]
        with torch.no_grad():
            trans = [gan.G_s2r(TG.to_batch(show, [i], device))[0, 0].cpu().numpy()
                     for i in range(len(show))]
        _save_before_after(show, [np.clip((t + 1) / 2, 0, 1) for t in trans], args.preview,
                           f"{args.variant} ({real_kind}, {args.iters} iters, {device})")
    return 0


def train_sonospade(args):
    """The proposed dual-input, per-tissue SonoSPADE translator (P3)."""
    import torch
    from usbg import texture_gan as TG
    from usbg import segmenter as SEG
    from usbg.dataset import load_pairs_npz
    from usbg.render_lotus import get_device

    if not args.pairs:
        raise SystemExit("sonospade needs --pairs (aligned image,label pairs from make_dataset --pairs-out)")
    if not args.s_us:
        raise SystemExit("sonospade needs --s-us (a pretrained S_US from scripts/train_segmenter.py)")

    device = get_device()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cfg = TG.load_config(args.config)
    mcfg, lcfg = cfg.get("model", {}), cfg.get("loss", {})
    K = mcfg.get("n_classes", 13)

    imgs, labs, _sess, _meta = load_pairs_npz(args.pairs)      # aligned physics image + label
    imgs = imgs.astype(np.float32) / 255.0
    labs = labs.astype(np.int64)
    if args.real_dir:
        real_pool = load_real_dir(args.real_dir, imgs.shape[1])
        real_kind = f"real-dir ({len(real_pool)})"
    elif args.proxy_real:
        real_pool = np.stack([TG.proxy_real_texture(imgs[i], rng)
                              for i in range(len(imgs))], 0)   # distinct texture on the sim frames
        real_kind = "proxy-real"
    else:
        raise SystemExit("provide --real-dir or --proxy-real for the real pool")

    s_us = SEG.Segmenter.load(args.s_us, device=device)
    print(f"[sonospade] pairs {len(imgs)} ({imgs.shape[1:]}), real {len(real_pool)} ({real_kind}), "
          f"S_US {args.s_us}, device {device}, {args.iters} iters")

    tcfg = cfg.get("train", {})
    gan = TG.build_trainer("sonospade", device, s_us=s_us, n_classes=K,
                           ngf=mcfg.get("ngf", 32), ndf=mcfg.get("ndf", 32),
                           n_blocks=args.n_blocks or mcfg.get("n_blocks", 6),
                           spade_hidden=mcfg.get("spade_hidden", 64),
                           lambda_nce=lcfg.get("lambda_nce", 1.0),
                           lambda_seg=lcfg.get("lambda_seg", 5.0),
                           lambda_seg_dice=lcfg.get("lambda_seg_dice", 1.0),
                           lambda_tc=lcfg.get("lambda_tc", 0.0),
                           lambda_lm=lcfg.get("lambda_labelmix", 0.0),
                           r1_gamma=lcfg.get("r1_gamma", 0.0),
                           ema_decay=tcfg.get("ema_decay", 0.999),
                           nce_tau=lcfg.get("nce_tau", 0.07), lr=cfg.get("optim", {}).get("lr", 2e-4))

    def phys_batch(idx):
        b = np.stack([imgs[i] for i in idx], 0)[:, None] * 2 - 1
        return torch.from_numpy(b.astype(np.float32)).to(device)

    def seg_batch(idx):
        lb = np.stack([labs[i] for i in idx], 0)
        return TG.onehot_labels(lb, K, device)

    def real_batch(idx):
        b = np.stack([real_pool[i] for i in idx], 0)[:, None] * 2 - 1
        return torch.from_numpy(b.astype(np.float32)).to(device)

    for it in range(args.iters):
        pi = rng.integers(0, len(imgs), args.batch)
        ri = rng.integers(0, len(real_pool), args.batch)
        losses = gan.step(phys_batch(pi), seg_batch(pi), real_batch(ri))
        if (it + 1) % max(1, args.iters // 10) == 0:
            print("  iter %4d  " % (it + 1) + "  ".join(f"{k} {v:.3f}" for k, v in losses.items()))

    gan.save(args.out)
    print(f"saved SonoSPADE generator -> {args.out}")

    if args.preview:
        Gp = gan.G_ema if getattr(gan, "G_ema", None) is not None else gan.G   # the deployed generator
        Gp.eval()
        show = list(range(min(5, len(imgs))))
        with torch.no_grad():
            trans = []
            for i in show:
                seg = TG.onehot_labels(labs[i][None], K, device)
                a = torch.from_numpy(imgs[i][None, None] * 2 - 1).to(device)
                trans.append(np.clip((Gp(a, seg)[0, 0].cpu().numpy() + 1) / 2, 0, 1))
        _save_before_after([imgs[i] for i in show], trans, args.preview,
                           f"SonoSPADE ({real_kind}, {args.iters} iters, {device})",
                           labels=[labs[i] for i in show])
    return 0


def _save_before_after(before, after, path, title, labels=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nrow = 3 if labels is not None else 2
    n = len(before)
    fig, axes = plt.subplots(nrow, n, figsize=(2.0 * n, 2.1 * nrow), squeeze=False)
    for i in range(n):
        axes[0][i].imshow(before[i], cmap="gray", vmin=0, vmax=1); axes[0][i].axis("off")
        axes[1][i].imshow(after[i], cmap="gray", vmin=0, vmax=1); axes[1][i].axis("off")
        if labels is not None:
            axes[2][i].imshow(labels[i], cmap="tab20", vmin=0, vmax=13); axes[2][i].axis("off")
    axes[0][0].set_title("raw physics (sim)", fontsize=9, loc="left")
    axes[1][0].set_title("translated", fontsize=9, loc="left")
    if labels is not None:
        axes[2][0].set_title("label", fontsize=9, loc="left")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"wrote before/after -> {path}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["cyclegan", "cut", "cyclegan_sc", "sonospade"],
                    default="cyclegan")
    ap.add_argument("--sim-npz", default="data/dataset/usbg.npz",
                    help="physics-render frames (grayscale variants)")
    ap.add_argument("--pairs", default=None, help="aligned image,label pairs npz (sonospade)")
    ap.add_argument("--s-us", default=None, help="pretrained S_US checkpoint (sonospade)")
    ap.add_argument("--config", default="configs/sonospade.yaml")
    ap.add_argument("--real-dir", default=None, help="dir of real B-mode frames (the real pool)")
    ap.add_argument("--proxy-real", action="store_true",
                    help="synthesize the real pool (demo without real data)")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--n-blocks", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/dataset/texture_gan.pt")
    ap.add_argument("--preview", default="outputs/texture_before_after.png")
    args = ap.parse_args(argv)

    if args.variant == "sonospade":
        return train_sonospade(args)
    return train_grayscale(args)


if __name__ == "__main__":
    raise SystemExit(main())
