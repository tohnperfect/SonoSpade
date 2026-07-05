#!/usr/bin/env python3
"""Experiment A: curvilinear (sector) remap + view augmentation for SonoSPADE.

The Kaggle Vitale real frames are convex/sector B-mode (dark triangular corners, fan-shaped lit
region: measured apex ~28 px above the 128-frame top, centered near col 68, half-angle ~22 deg,
radial extent r in [36,128] px). Our physics substrate renders a LINEAR rectangular frame, so a
downstream segmenter trained on sim sees a different spatial prior than the real test frames, which
is a large part of why TSTR is near zero. This module warps the linear (image, label) render into a
matching sector so the geometry lines up, and can bake view-augmented copies (probe rotation and
translation) into the training pool for view variety. Warp is applied to BOTH image (bilinear) and
label (nearest, to keep integer classes); outside-sector pixels become background (image 0, label 0),
exactly as in the real frames.

Defaults match the measured Kaggle geometry; they are not tuned per result.
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.ndimage import map_coordinates

# measured on data/eval/kaggle_vitale_test.npz at 128x128 (see git log / commit message)
DEFAULT = dict(apex_x=68.0, apex_y=-28.0, half_angle_deg=23.0, r_min=36.0, r_max=128.0)


def _sector_coords(H, W, p, rot_deg=0.0, tx=0.0, ty=0.0):
    """For each output pixel return source (row, col) fractional coords into the linear render, plus
    an in-sector mask. rot/tx/ty are the view augmentation (apex rotation about vertical, translation
    in fractions of frame)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ax, ay = p["apex_x"] + tx * W, p["apex_y"] + ty * H
    dx, dy = xx - ax, yy - ay
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.degrees(np.arctan2(dx, dy)) - rot_deg          # angle from vertical, rotated
    half = p["half_angle_deg"]
    mask = (np.abs(theta) <= half) & (r >= p["r_min"]) & (r <= p["r_max"])
    # linear render: column <- normalized angle, row <- normalized radius
    src_col = (theta + half) / (2.0 * half) * (W - 1)
    src_row = (r - p["r_min"]) / (p["r_max"] - p["r_min"]) * (H - 1)
    return src_row, src_col, mask


def warp_to_sector(img, label, p=DEFAULT, rot_deg=0.0, tx=0.0, ty=0.0):
    H, W = img.shape
    src_row, src_col, mask = _sector_coords(H, W, p, rot_deg, tx, ty)
    coords = np.stack([src_row.ravel(), src_col.ravel()])
    wimg = map_coordinates(img.astype(np.float32), coords, order=1, mode="constant", cval=0.0)
    wimg = (wimg.reshape(H, W) * mask).astype(np.float32)
    wlab = map_coordinates(label.astype(np.float32), coords, order=0, mode="constant", cval=0.0)
    wlab = (wlab.reshape(H, W) * mask).astype(label.dtype)    # outside sector -> background (0)
    return wimg, wlab


def build(in_npz, out_npz, n_aug=0, rot_max=15.0, trans_max=0.10, seed=0, p=DEFAULT):
    from usbg.dataset import load_pairs_npz
    imgs, labs, sess, meta = load_pairs_npz(in_npz)
    imgs = imgs.astype(np.float32)
    if imgs.max() > 1.5:
        imgs = imgs / 255.0
    rng = np.random.default_rng(seed)
    out_i, out_l, out_s = [], [], []
    for i in range(len(imgs)):
        wi, wl = warp_to_sector(imgs[i], labs[i], p)
        out_i.append((wi * 255).astype(np.uint8)); out_l.append(wl); out_s.append(sess[i])
        for _ in range(n_aug):                                # train-only view variety
            rot = float(rng.uniform(-rot_max, rot_max))
            tx = float(rng.uniform(-trans_max, trans_max)); ty = float(rng.uniform(-trans_max, trans_max))
            ai, al = warp_to_sector(imgs[i], labs[i], p, rot_deg=rot, tx=tx, ty=ty)
            out_i.append((ai * 255).astype(np.uint8)); out_l.append(al); out_s.append(sess[i])
    I = np.stack(out_i); L = np.stack(out_l).astype(np.uint8); S = np.asarray(out_s)
    import json
    meta_out = dict(meta) if isinstance(meta, dict) else {"orig": str(meta)}
    meta_out["curvi_warp"] = f"sector n_aug={n_aug}"
    np.savez_compressed(out_npz, images=I, labels=L, sessions=S, meta=json.dumps(meta_out))
    print(f"wrote {out_npz}: {I.shape} ({n_aug} aug/frame)")
    return out_npz


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-npz", default="data/dataset/pairs_med.npz")
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--n-aug", type=int, default=0)
    ap.add_argument("--preview", default=None)
    args = ap.parse_args(argv)
    build(args.in_npz, args.out_npz, n_aug=args.n_aug)
    if args.preview:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d = np.load(args.out_npz)
        rl = np.load("data/eval/kaggle_vitale_test.npz")["images"]
        fig, ax = plt.subplots(2, 4, figsize=(10, 5))
        for j in range(4):
            ax[0][j].imshow(d["images"][j], cmap="gray"); ax[0][j].set_title(f"curvi sim {j}", fontsize=8)
            ax[1][j].imshow(rl[j], cmap="gray"); ax[1][j].set_title(f"real {j}", fontsize=8)
            ax[0][j].axis("off"); ax[1][j].axis("off")
        plt.tight_layout(); plt.savefig(args.preview, dpi=90); print(f"preview -> {args.preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
