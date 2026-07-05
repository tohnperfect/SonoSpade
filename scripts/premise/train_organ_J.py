#!/usr/bin/env python3
"""train_organ_J.py: train the reward/premise judge J -- a compact organ segmenter -- on
REAL Kaggle Vitale frames+masks ONLY. Never sees sim or SonoSPADE output (non-circular).

One seed = one train/val split of the 60 labeled real frames + one trained J. Strong intensity
augmentation forces J to use per-tissue texture/structure, not global brightness. Saves the J
checkpoint and its held-out-real validation scores (the in-domain ceiling for the premise eval).

Run:
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  PYTHONPATH=src .venv/bin/python scripts/premise/train_organ_J.py --seed 0 --out outputs/premise
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import (N_J, ORGAN_JIDS, J_NAMES, load_real, augment, per_organ_scores)  # noqa: E402


def make_split(n: int, seed: int, val_frac: float = 0.25):
    rng = np.random.default_rng(1000 + seed)
    idx = rng.permutation(n)
    n_val = max(8, int(round(val_frac * n)))
    return idx[n_val:], idx[:n_val]                     # train_idx, val_idx


def batch_tensor(imgs, labs, idx, rng, device, augment_on):
    import torch
    from common import augment as _aug
    xs, ys = [], []
    for i in idx:
        im, lb = imgs[i], labs[i]
        if augment_on:
            im, lb = _aug(im, lb, rng)
        xs.append(im); ys.append(lb)
    x = torch.from_numpy(np.stack(xs)[:, None].astype(np.float32)).to(device)
    y = torch.from_numpy(np.stack(ys).astype(np.int64)).to(device)
    return x, y


def evaluate_real(net, imgs, labs, val_idx, device, min_pixels=40):
    """Per-organ recall/IoU of J on held-out REAL frames (the in-domain ceiling)."""
    import torch
    net.eval()
    agg = {j: {"recall": [], "iou": []} for j in ORGAN_JIDS}
    with torch.no_grad():
        for i in val_idx:
            x = torch.from_numpy(imgs[i][None, None].astype(np.float32)).to(device)
            pred = net(x)[0].argmax(0).cpu().numpy()
            sc = per_organ_scores(pred, labs[i], min_pixels=min_pixels)
            for j, v in sc.items():
                agg[j]["recall"].append(v["recall"]); agg[j]["iou"].append(v["iou"])
    out = {}
    for j in ORGAN_JIDS:
        if agg[j]["recall"]:
            out[J_NAMES[j]] = {"recall": float(np.mean(agg[j]["recall"])),
                               "iou": float(np.mean(agg[j]["iou"])),
                               "n_frames": len(agg[j]["recall"])}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/premise")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--real", default="data/eval/kaggle_vitale_test.npz")
    ap.add_argument("--crop-real", action="store_true",
                    help="crop the fan-beam border from real frames (domain-match to linear sim)")
    ap.add_argument("--balance", action="store_true",
                    help="oversample frames containing rare organs + class-weight the loss, and select "
                         "the checkpoint on the rare organs (for the kidney/spleen/gallbladder attempt)")
    ap.add_argument("--sel-organs", default="liver,gallbladder,vessel,kidney,spleen",
                    help="organs whose mean val IoU drives checkpoint selection")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    import torch
    from usbg.segmenter import build_segmenter, seg_ce_dice_loss
    from usbg.render_lotus import get_device
    device = args.device or get_device()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    imgs, labs, names = load_real(args.real, crop_fan=args.crop_real)
    n = len(imgs)
    tr_idx, val_idx = make_split(n, args.seed)
    # report organ coverage of each split so we know what J can actually learn / be scored on
    def cov(idx):
        c = {}
        for j in ORGAN_JIDS:
            c[J_NAMES[j]] = int(sum(int((labs[i] == j).sum()) >= 40 for i in idx))
        return c
    print(f"[seed {args.seed}] real n={n} train={len(tr_idx)} val={len(val_idx)}")
    print(f"[seed {args.seed}] train organ coverage (#frames>=40px): {cov(tr_idx)}")
    print(f"[seed {args.seed}] val   organ coverage (#frames>=40px): {cov(val_idx)}")

    net = build_segmenter(n_classes=N_J, base=args.base, depth=args.depth).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    rng = np.random.default_rng(args.seed)
    sel_names = [s.strip() for s in args.sel_organs.split(",") if s.strip()]

    # rare-organ balancing: per-frame oversampling weight = sum over organs present of 1/frequency,
    # and per-class loss weights inverse to global pixel frequency. Lets J spend capacity on the
    # rare kidney/spleen/gallbladder instead of being dominated by liver/background.
    frame_w, class_w = None, None
    if args.balance:
        import torch as _t
        org_frames = {j: sum(int((labs[i] == j).sum()) >= 40 for i in tr_idx) for j in ORGAN_JIDS}
        fw = []
        for i in tr_idx:
            w = 1.0
            for j in ORGAN_JIDS:
                if int((labs[i] == j).sum()) >= 40:
                    w += 1.0 / max(org_frames[j], 1)
            fw.append(w)
        frame_w = np.asarray(fw, float); frame_w /= frame_w.sum()
        px = np.bincount(np.concatenate([labs[i].ravel() for i in tr_idx]), minlength=N_J).astype(float)
        inv = (px.sum() / np.maximum(px, 1)); inv = inv / inv.mean()
        class_w = _t.tensor(np.clip(inv, 0.3, 6.0), dtype=_t.float32, device=device)
        print(f"[seed {args.seed}] BALANCE on: class_w={np.round(class_w.cpu().numpy(),2)}")

    def loss_fn(logits, y):
        import torch.nn.functional as F
        if class_w is None:
            return seg_ce_dice_loss(logits, y, n_classes=N_J, dice_weight=1.0)
        ce = F.cross_entropy(logits, y, weight=class_w)
        # soft Dice (unweighted) keeps overlap quality
        probs = F.softmax(logits, dim=1)
        tgt = F.one_hot(y.clamp(0, N_J - 1), N_J).permute(0, 3, 1, 2).float()
        inter = (probs * tgt).sum((0, 2, 3)); den = probs.sum((0, 2, 3)) + tgt.sum((0, 2, 3))
        dice = 1.0 - ((2 * inter + 1e-6) / (den + 1e-6)).mean()
        return ce + dice

    best_iou, best_state, best_val = -1.0, None, {}
    n_batches = max(1, len(tr_idx) // args.batch)
    for ep in range(args.epochs):
        net.train()
        if frame_w is not None:
            order = rng.choice(tr_idx, size=n_batches * args.batch, replace=True, p=frame_w)
        else:
            order = rng.permutation(tr_idx)
        losses = []
        for k in range(0, len(order), args.batch):
            bidx = order[k:k + args.batch]
            x, y = batch_tensor(imgs, labs, bidx, rng, device, augment_on=True)
            opt.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        sched.step()
        if (ep + 1) % 40 == 0 or ep == args.epochs - 1:
            val = evaluate_real(net, imgs, labs, val_idx, device)
            if not val:
                miou = 0.0
            elif args.balance:
                # rare-organ attempt: select on the UNWEIGHTED mean IoU over the target organs
                # (--sel-organs), so kidney/spleen/gallbladder are not drowned by liver.
                picked = [val[k]["iou"] for k in sel_names if k in val]
                miou = float(np.mean(picked)) if picked else 0.0
            else:
                # support-WEIGHTED mean IoU: a 1-frame organ cannot swing which epoch is saved.
                wsum = sum(v["n_frames"] for v in val.values())
                miou = float(sum(v["iou"] * v["n_frames"] for v in val.values()) / max(wsum, 1))
            print(f"[seed {args.seed}] ep {ep+1:3d} loss {np.mean(losses):.3f} "
                  f"val_mIoU {miou:.3f} " + " ".join(f"{k}:{v['iou']:.2f}" for k, v in val.items()))
            if miou > best_iou:
                best_iou = miou
                best_state = {kk: vv.detach().cpu().clone() for kk, vv in net.state_dict().items()}
                best_val = val

    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, f"J_seed{args.seed}.pt")
    torch.save({"state": best_state or net.state_dict(), "n_classes": N_J,
                "base": args.base, "depth": args.depth, "seed": args.seed,
                "crop_real": bool(args.crop_real),
                "val_idx": val_idx.tolist(), "train_idx": tr_idx.tolist(),
                "val_scores": best_val, "val_mIoU": best_iou}, ckpt)
    with open(os.path.join(args.out, f"J_seed{args.seed}_val.json"), "w") as fh:
        json.dump({"seed": args.seed, "val_mIoU": best_iou, "val_scores": best_val,
                   "train_cov": cov(tr_idx), "val_cov": cov(val_idx)}, fh, indent=2)
    print(f"[seed {args.seed}] saved {ckpt}  best val_mIoU={best_iou:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
