#!/usr/bin/env python3
"""FID and KID on the real head-to-head (revision).

Computes Frechet Inception Distance (FID) and Kernel Inception Distance (KID) between each
translator's outputs on the shared content pool and the real Kaggle Vitale pool, using standard
InceptionV3 (pool3, 2048-d) features. Reuses eval_texture._translate_pool so the generated pools are
identical to the ones scored in results_real.json.

Real reference: the 404-frame real train pool (data/real/kaggle_vitale/pool) -- the distribution the
translators were trained toward; larger and more stable than the 60-frame test split. FID is biased
upward at this sample size; KID's unbiased estimator is the more reliable of the two here, so we
report both and lean on KID.

Usage:
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  .venv/bin/python scripts/eval_fid_kid.py \
      --pairs data/dataset/pairs_med.npz --real-dir data/real/kaggle_vitale/pool \
      --cyclegan data/dataset/exp_real_cyclegan.pt --cut data/dataset/exp_real_cut.pt \
      --cyclegan-sc data/dataset/exp_real_cyclegan_sc.pt \
      --sonospade data/dataset/exp_real_sonospade.pt \
      --out outputs/experiment/fid_kid_real.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_texture import _load_pairs, _translate_pool  # noqa: E402

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _inception(device):
    import torch
    from torchvision.models import Inception_V3_Weights, inception_v3
    m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    m.fc = torch.nn.Identity()  # expose the 2048-d pool3 features
    m.eval().to(device)
    return m


def _features(model, imgs, device, batch=32):
    """imgs: list/array of HxW float in [0,1] (or 0..255). Returns (N, 2048) numpy features."""
    import torch
    import torch.nn.functional as F
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    feats = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch):
            chunk = imgs[i:i + batch]
            x = np.stack([np.asarray(a, np.float32) for a in chunk])
            if x.max() > 1.5:
                x = x / 255.0
            x = np.clip(x, 0.0, 1.0)
            t = torch.from_numpy(x).to(device).unsqueeze(1)  # (B,1,H,W)
            t = t.repeat(1, 3, 1, 1)
            t = F.interpolate(t, size=(299, 299), mode="bilinear", align_corners=False)
            t = (t - mean) / std
            f = model(t)
            feats.append(f.float().cpu().numpy())
    return np.concatenate(feats, 0)


def _fid(fr, fg):
    from scipy import linalg
    mu_r, mu_g = fr.mean(0), fg.mean(0)
    cr = np.cov(fr, rowvar=False)
    cg = np.cov(fg, rowvar=False)
    diff = mu_r - mu_g
    # small ridge for numerical stability of the matrix sqrt of the covariance product
    eps = 1e-6
    offset = np.eye(cr.shape[0]) * eps
    covmean = linalg.sqrtm((cr + offset) @ (cg + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cr + cg - 2.0 * covmean))


def _kid(fr, fg, subset=100, n_subsets=100, seed=0):
    """Unbiased polynomial-kernel MMD^2 (KID), averaged over random subsets. Returns (mean, std)."""
    rng = np.random.default_rng(seed)
    d = fr.shape[1]
    n = min(subset, fr.shape[0], fg.shape[0])
    vals = []

    def poly(a, b):
        return (a @ b.T / d + 1.0) ** 3

    for _ in range(n_subsets):
        r = fr[rng.choice(fr.shape[0], n, replace=False)]
        g = fg[rng.choice(fg.shape[0], n, replace=False)]
        krr = poly(r, r)
        kgg = poly(g, g)
        krg = poly(r, g)
        np.fill_diagonal(krr, 0.0)
        np.fill_diagonal(kgg, 0.0)
        mmd = (krr.sum() / (n * (n - 1)) + kgg.sum() / (n * (n - 1))
               - 2.0 * krg.mean())
        vals.append(mmd)
    return float(np.mean(vals)), float(np.std(vals))


def _load_real_dir(path, limit=None):
    from PIL import Image
    files = sorted(glob.glob(os.path.join(path, "*.png")))
    if limit:
        files = files[:limit]
    out = []
    for f in files:
        out.append(np.asarray(Image.open(f).convert("L"), np.float32) / 255.0)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/dataset/pairs_med.npz")
    ap.add_argument("--real-dir", default="data/real/kaggle_vitale/pool")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--cyclegan", default=None)
    ap.add_argument("--cut", default=None)
    ap.add_argument("--cyclegan-sc", default=None)
    ap.add_argument("--sonospade", default=None)
    ap.add_argument("--kid-subset", type=int, default=100)
    ap.add_argument("--out", default="outputs/experiment/fid_kid_real.json")
    args = ap.parse_args(argv)

    from usbg.render_lotus import get_device
    device = get_device()

    phys_imgs, labels = _load_pairs(args.pairs, args.limit)
    real_imgs = _load_real_dir(args.real_dir)
    print(f"content pool: {len(phys_imgs)} frames; real reference pool: {len(real_imgs)} frames")

    model = _inception(device)
    fr = _features(model, real_imgs, device)
    print(f"real features: {fr.shape}")

    variants = [("physics", None)]
    for name, ck in [("cyclegan", args.cyclegan), ("cut", args.cut),
                     ("cyclegan_sc", args.cyclegan_sc), ("sonospade", args.sonospade)]:
        if ck and os.path.exists(ck):
            variants.append((name, ck))
        elif ck:
            print(f"  {name}: checkpoint {ck} not found, skipping")

    results = {"real_n": len(real_imgs), "gen_n": len(phys_imgs),
               "kid_subset": args.kid_subset, "variants": {}}
    for variant, ck in variants:
        trans = _translate_pool(variant, ck, phys_imgs, labels, device)
        fg = _features(model, trans, device)
        fid = _fid(fr, fg)
        subset = min(args.kid_subset, len(real_imgs), len(trans))
        kid_mean, kid_std = _kid(fr, fg, subset=subset)
        results["variants"][variant] = {"fid": fid, "kid": kid_mean, "kid_std": kid_std}
        print(f"{variant:12s} FID {fid:8.2f}  KID {kid_mean*1e3:7.2f}e-3 +/- {kid_std*1e3:.2f}e-3")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
