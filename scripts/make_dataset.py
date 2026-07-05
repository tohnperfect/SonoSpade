#!/usr/bin/env python3
"""make_dataset.py (SonoSPADE release): render aligned (physics image, canonical label) pairs by
sweeping probe poses over CT-derived labeled volumes. These are SonoSPADE's free supervision -- the
generator's content input and the exact labels that pretrain S_US. gen_labeled_pairs() is also used
by the closed-loop organ experiments (scripts/premise/). The RL-demonstrator dataset builder from the
full testbed is omitted from this release.

Usage:
  python scripts/make_dataset.py --cases s1086 s0777 --target liver --n-per-case 200 \
      --image-size 128 --out data/dataset/pairs_med.npz
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from usbg import volume as V
from usbg import slicer as S
from usbg import render_lotus as R
from usbg.placement import seat_volume_under_probe

RAW_ROOT = "data/raw/totalseg_small"
CACHE_DIR = "data/cache"

def resolve_case(case_id: str) -> V.LabeledVolume:
    cache = os.path.join(CACHE_DIR, f"{case_id}.npz")
    if os.path.exists(cache):
        return V.load_cache(cache)
    case_dir = os.path.join(RAW_ROOT, case_id)
    ct = os.path.join(case_dir, "ct.nii.gz")
    seg = os.path.join(case_dir, "segmentations")
    print(f"  building volume for {case_id} (not cached) ...")
    lv = V.load_labeled_ct(ct, seg)
    V.save_cache(lv, cache)
    return lv


def _resize_float(img01, size):
    from PIL import Image
    a = np.clip(np.asarray(img01) * 255.0, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(a).resize((size, size), Image.BILINEAR), np.float32) / 255.0

def _mild_randomize(base, rng):
    """A gentler domain randomization for S_US pretraining: vary gain, TGC, PSF, and speckle (so
    S_US is appearance-robust) while leaving the per-tissue acoustic constants (z/mu/backscatter)
    at their defaults, so acoustically distinct tissues keep distinct brightness and stay separable.
    Full domain_randomize scrambles backscatter, which collapses the soft-tissue classes together."""
    from dataclasses import replace
    u = rng.uniform
    return replace(base, gain=float(u(0.85, 1.15)), tgc_slope=float(u(0.7, 1.3)),
                   psf_sigma_lat0=float(u(0.9, 1.4)), psf_focus_depth=float(u(0.03, 0.08)),
                   speckle_strength=float(u(0.7, 1.1)), noise_sigma=float(u(0.0, 0.03)),
                   dyn_range_db=float(u(48, 56)), seed=int(rng.integers(2 ** 31 - 1)))


def gen_labeled_pairs(cases, target_name, n_per_case, image_size, device, seed=0,
                      randomize="full", standoff_m=0.0, depth_m=0.05):
    """Render aligned (physics image, canonical label) pairs by sweeping poses around the seated
    base pose on each case. These are SonoSPADE's free supervision: the generator's content input
    and the exact labels that pretrain S_US. Returns (images uint8 (N,H,W), labels uint8 (N,H,W)).

    Each pose is a small random se(3) perturbation of the base pose (so views vary while anatomy is
    intersected), and with randomize=True each frame draws its own physics domain-randomization
    config so S_US keys on structure rather than a fixed appearance. The label is transposed and
    resized to the render orientation with segmenter.align_label_to_image so image and label align.
    """
    from usbg.placement import seat_volume_under_probe
    from usbg.contracts_bridge import pose_from_rpy
    from usbg.segmenter import align_label_to_image

    name_to_label = {v: k for k, v in V.USBG_LABEL_NAMES.items()}
    target = name_to_label.get(target_name, V.GALLBLADDER)
    rng = np.random.default_rng(seed)
    imgs, labs, sess = [], [], []
    for case in cases:
        lv = resolve_case(case)
        tgt = target if lv.has(target) else (V.LIVER if lv.has(V.LIVER) else V.SOFT_TISSUE)
        world_from_volume, T_base = seat_volume_under_probe(lv, tgt, standoff_m=standoff_m,
                                                            depth_m=depth_m)
        for _ in range(n_per_case):
            perturb = pose_from_rpy(tx=float(rng.uniform(-0.03, 0.03)),
                                    ty=float(rng.uniform(-0.03, 0.03)),
                                    tz=float(rng.uniform(-0.02, 0.02)),
                                    rx=float(rng.uniform(-12, 12)),
                                    ry=float(rng.uniform(-12, 12)),
                                    rz=float(rng.uniform(-20, 20)))
            T_wp = T_base @ perturb
            lab = S.slice_at_pose(lv, T_wp, world_from_volume)          # (N_LINES, N_SAMPLES)
            if randomize == "full" or randomize is True:
                cfg = R.domain_randomize(R.RenderConfig(), rng)
            elif randomize == "mild":
                cfg = _mild_randomize(R.RenderConfig(), rng)
            else:                                                        # "off": speckle seed only
                cfg = R.RenderConfig().with_seed(int(rng.integers(2 ** 31 - 1)))
            img = R.render(lab, "physics", device=device, cfg=cfg)      # (N_SAMPLES, N_LINES)
            img_sq = _resize_float(img, image_size)                     # (size, size) float [0,1]
            lab_sq = align_label_to_image(lab, (image_size, image_size))
            imgs.append((np.clip(img_sq, 0, 1) * 255).astype(np.uint8))
            labs.append(lab_sq.astype(np.uint8))
            sess.append(case)
    return np.asarray(imgs, np.uint8), np.asarray(labs, np.uint8), np.asarray(sess)



def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--target", default="liver")
    ap.add_argument("--n-per-case", type=int, default=200)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--randomize", default="mild", choices=["full", "mild", "off"])
    ap.add_argument("--depth-m", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    from usbg.render_lotus import get_device
    from usbg.dataset import write_pairs_npz
    dev = get_device()
    imgs, labs, sess = gen_labeled_pairs(args.cases, args.target, args.n_per_case, args.image_size,
                                         dev, seed=args.seed, randomize=args.randomize,
                                         depth_m=args.depth_m)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_pairs_npz(args.out, imgs, labs, sessions=sess,
                    extra_meta={"target": args.target, "n": int(len(imgs))})
    print(f"wrote {len(imgs)} (image,label) pairs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
