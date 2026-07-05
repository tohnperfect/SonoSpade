#!/usr/bin/env python3
"""Experiment B: test-time adaptation (TENT) on the TSTR downstream segmenter.

TSTR trains a segmenter on synthetic (image, label) pairs and tests on real frames. Here we add TENT
(Wang et al., ICLR 2021): after training on synthetic, adapt only the BatchNorm affine parameters by
minimizing prediction entropy on a few UNLABELED real frames (from the 404-frame train pool, disjoint
from the 60-frame labeled test), then re-evaluate on the 60 test frames. Run for both the SonoSPADE
-curvi and the raw-physics-curvi downstream, so the comparison to raw physics is fair. Reports Dice
before and after adaptation (the delta is the TTA effect).

Usage:
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  .venv/bin/python scripts/eval_tta_tent.py \
      --pairs data/dataset/pairs_med_curvi.npz --real-pairs data/eval/kaggle_vitale_test.npz \
      --real-dir data/real/kaggle_vitale/pool --sonospade data/dataset/exp_real_sonospade_curvi.pt \
      --n-adapt 50 --out outputs/experiment/tta_curvi.json
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


def _train_downstream(Xtr, Ytr, n_classes, device, iters=500, base=24, depth=4, lr=5e-4, seed=0):
    import torch
    from usbg.segmenter import build_segmenter, seg_ce_dice_loss
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    net = build_segmenter(n_classes, base=base, depth=depth).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for _ in range(iters):
        bi = rng.integers(0, len(Xtr), size=min(8, len(Xtr)))
        x = torch.from_numpy(Xtr[bi][:, None].astype(np.float32)).to(device)
        y = torch.from_numpy(Ytr[bi].astype(np.int64)).to(device)
        opt.zero_grad(); seg_ce_dice_loss(net(x), y, n_classes=n_classes).backward(); opt.step()
    return net


def _dice(net, Xte, Yte, n_classes, device, batch=8):
    import torch
    net.eval()
    inter = np.zeros(n_classes); psum = np.zeros(n_classes); tsum = np.zeros(n_classes)
    with torch.no_grad():
        for j in range(0, len(Xte), batch):
            x = torch.from_numpy(Xte[j:j + batch][:, None].astype(np.float32)).to(device)
            pred = net(x).argmax(1).cpu().numpy(); tgt = Yte[j:j + batch]
            for c in range(n_classes):
                p = pred == c; t = tgt == c
                inter[c] += np.logical_and(p, t).sum(); psum[c] += p.sum(); tsum[c] += t.sum()
    per = {c: float((2 * inter[c] + 1e-6) / (psum[c] + tsum[c] + 1e-6))
           for c in range(1, n_classes) if tsum[c] > 0}
    return (float(np.mean(list(per.values()))) if per else float("nan")), per


def _tent_adapt(net, adapt_imgs, device, steps=40, batch=10, lr=1e-3):
    """TENT: minimize prediction entropy, updating only BatchNorm affine params (BN uses batch stats)."""
    import torch
    import torch.nn as nn
    net.train()
    for m in net.modules():                     # freeze everything except BN affine
        for p in m.parameters(recurse=False):
            p.requires_grad_(False)
    params = []
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True); m.track_running_stats = False; m.running_mean = None; m.running_var = None
            params += [m.weight, m.bias]
    opt = torch.optim.Adam(params, lr=lr)
    rng = np.random.default_rng(0)
    A = np.asarray(adapt_imgs, np.float32)
    for _ in range(steps):
        bi = rng.integers(0, len(A), size=min(batch, len(A)))
        x = torch.from_numpy(A[bi][:, None]).to(device)
        opt.zero_grad()
        logits = net(x)
        p = logits.softmax(1)
        ent = -(p * torch.log(p + 1e-8)).sum(1).mean()
        ent.backward(); opt.step()
    return net


def _load_real_dir(path, n, exclude_shape=None):
    from PIL import Image
    files = sorted(glob.glob(os.path.join(path, "*.png")))[:n]
    return [np.asarray(Image.open(f).convert("L"), np.float32) / 255.0 for f in files]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/dataset/pairs_med_curvi.npz")
    ap.add_argument("--real-pairs", default="data/eval/kaggle_vitale_test.npz")
    ap.add_argument("--real-dir", default="data/real/kaggle_vitale/pool")
    ap.add_argument("--sonospade", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--n-adapt", type=int, default=50)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--seeds", default="0", help="comma-separated downstream seeds for robustness")
    ap.add_argument("--out", default="outputs/experiment/tta_curvi.json")
    args = ap.parse_args(argv)

    from usbg.render_lotus import get_device
    from usbg.volume import N_LABELS
    device = get_device()

    phys, labels = _load_pairs(args.pairs, args.limit)
    real_te, real_te_lab = _load_pairs(args.real_pairs, args.limit)
    adapt = _load_real_dir(args.real_dir, args.n_adapt)
    print(f"synth {len(phys)}, real-test {len(real_te)}, adapt(unlabeled) {len(adapt)}")

    seeds = [int(s) for s in args.seeds.split(",")]
    results = {"n_adapt": len(adapt), "seeds": seeds, "variants": {}}
    for variant, ck in (("physics", None), ("sonospade", args.sonospade)):
        trans = np.asarray(_translate_pool(variant, ck, phys, labels, device), np.float32)
        base_runs, tent_runs = [], []
        for sd in seeds:
            net = _train_downstream(trans, labels, N_LABELS, device, iters=args.iters, seed=sd)
            b, _ = _dice(net, real_te, real_te_lab, N_LABELS, device)
            net = _tent_adapt(net, adapt, device)
            t, _ = _dice(net, real_te, real_te_lab, N_LABELS, device)
            base_runs.append(b); tent_runs.append(t)
        results["variants"][variant] = {
            "tstr": float(np.mean(base_runs)), "tstr_std": float(np.std(base_runs)),
            "tstr_tent": float(np.mean(tent_runs)), "tstr_tent_std": float(np.std(tent_runs)),
            "delta": float(np.mean(tent_runs) - np.mean(base_runs)),
            "base_runs": base_runs, "tent_runs": tent_runs}
        r = results["variants"][variant]
        print(f"{variant:10s} TSTR {r['tstr']:.3f}+/-{r['tstr_std']:.3f} -> "
              f"TENT {r['tstr_tent']:.3f}+/-{r['tstr_tent_std']:.3f} (delta {r['delta']:+.3f})  "
              f"base={[round(x,3) for x in base_runs]} tent={[round(x,3) for x in tent_runs]}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
