#!/usr/bin/env python3
"""eval_premise.py: the Section-4 premise check. Load a real-trained judge J (seed) and measure
its per-organ recognition on SIM content textured by each translator:

    physics (no translation) | CUT | CycleGAN | SonoSPADE

For every sim frame and every organ present in its TRUE label, we score how well J recognizes
that organ WITHIN its true region (recall + IoU), then aggregate per organ per translator.

Premise (Section 4): SonoSPADE (per-tissue-correct texture) should be recognized by J far better
than a global recolor (CUT/CycleGAN), which scrambles per-tissue statistics. The comparison across
translators is on identical content/geometry per frame, so only texture differs -- a fair test.

Non-circular: J was trained ONLY on real Kaggle frames (train_organ_J.py); it never saw sim or
SonoSPADE output. We also report J's confusion in each organ's region (what wrong organ a global
recolor makes J see) and J's held-out-real ceiling (from the J checkpoint).

Run:
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  PYTHONPATH=src .venv/bin/python scripts/premise/eval_premise.py --seed 0 --out outputs/premise
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import (N_J, ORGAN_JIDS, J_NAMES, load_sim, per_organ_scores)  # noqa: E402

TRANSLATORS = ["physics", "cut", "cyclegan", "sonospade"]
# All three translators MUST come from the same real-Kaggle-pool training batch (exp_real_*), so
# the only thing that differs is the texture stage (honesty guardrail #3). The older
# data/dataset/texture_cut.pt was trained toward a DIFFERENT real distribution (data/real_us_crop)
# and is ~2x too bright with too little variance vs real Kaggle -- using it unfairly handicaps CUT.
CKPTS = {"cut": "data/dataset/exp_real_cut.pt",
         "cyclegan": "data/dataset/exp_real_cyclegan.pt",
         "sonospade": "data/dataset/exp_real_sonospade.pt"}


def load_J(ckpt, device):
    import torch
    from usbg.segmenter import build_segmenter
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    net = build_segmenter(n_classes=ck.get("n_classes", N_J),
                          base=ck.get("base", 32), depth=ck.get("depth", 4)).to(device)
    net.load_state_dict(ck["state"]); net.eval()
    return net, ck


def texture_frame(name, translators, img01, label_canon):
    """Apply a texture stage to a physics render. physics = identity."""
    if name == "physics":
        return img01
    t = translators[name]
    if getattr(t, "wants_label", False):
        return t.translate_aligned(img01, label_canon)      # SonoSPADE: dual input, aligned label
    return t(img01)                                          # CUT / CycleGAN: single input


def select_eval_frames(labs_J, per_organ=120, min_pixels=60, seed=0):
    """Pick a balanced-ish eval subset: for each organ, the frames where it is most present, so
    thin organs (kidney, gallbladder, vessel) are represented, not drowned by liver."""
    rng = np.random.default_rng(seed)
    chosen = set()
    for jid in ORGAN_JIDS:
        frac = (labs_J == jid).reshape(len(labs_J), -1).sum(1)
        cand = np.where(frac >= min_pixels)[0]
        if len(cand) == 0:
            continue
        cand = cand[np.argsort(-frac[cand])][:per_organ]
        chosen.update(int(i) for i in cand)
    return sorted(chosen)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/premise")
    ap.add_argument("--jdir", default=None, help="dir holding J_seed{seed}.pt (default: --out)")
    ap.add_argument("--sim", default="data/dataset/pairs_med.npz")
    ap.add_argument("--min-pixels", type=int, default=60)
    ap.add_argument("--per-organ", type=int, default=150)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    import torch
    from usbg.render_lotus import get_device
    from usbg.texture_gan import load_translator
    device = args.device or get_device()

    jdir = args.jdir or args.out
    jckpt = os.path.join(jdir, f"J_seed{args.seed}.pt")
    net, jinfo = load_J(jckpt, device)
    print(f"[seed {args.seed}] J from {jckpt}  ceiling val_mIoU={jinfo.get('val_mIoU'):.3f}")

    translators = {k: load_translator(v, device) for k, v in CKPTS.items()}
    imgs, labs_canon, labs_J = load_sim(args.sim)
    frames = select_eval_frames(labs_J, per_organ=args.per_organ, min_pixels=args.min_pixels,
                                seed=args.seed)
    print(f"[seed {args.seed}] eval frames: {len(frames)}")

    # accumulate per (translator, organ): recall/precision/iou/null across frames; plus confusion
    acc = {t: {j: {"recall": [], "precision": [], "iou": [], "null_iou": []} for j in ORGAN_JIDS}
           for t in TRANSLATORS}
    confusion = {t: {j: np.zeros(N_J, dtype=np.int64) for j in ORGAN_JIDS} for t in TRANSLATORS}

    for fi in frames:
        img = imgs[fi]; lc = labs_canon[fi]; lj = labs_J[fi]
        present = per_organ_scores(lj, lj, min_pixels=args.min_pixels)   # organs present here
        if not present:
            continue
        for t in TRANSLATORS:
            tex = texture_frame(t, translators, img, lc)
            with torch.no_grad():
                x = torch.from_numpy(tex[None, None].astype(np.float32)).to(device)
                pred = net(x)[0].argmax(0).cpu().numpy()
            sc = per_organ_scores(pred, lj, min_pixels=args.min_pixels)
            for j, v in sc.items():
                acc[t][j]["recall"].append(v["recall"]); acc[t][j]["iou"].append(v["iou"])
                acc[t][j]["precision"].append(v["precision"]); acc[t][j]["null_iou"].append(v["null_iou"])
                # what does J call this organ's true region?
                reg = pred[lj == j]
                cc = np.bincount(reg, minlength=N_J)
                confusion[t][j] += cc

    # summarize
    summary = {"seed": args.seed, "n_frames": len(frames),
               "ceiling_real_val": jinfo.get("val_scores", {}),
               "translators": {}}
    for t in TRANSLATORS:
        summary["translators"][t] = {}
        for j in ORGAN_JIDS:
            r = acc[t][j]["recall"]; io = acc[t][j]["iou"]
            if not r:
                continue
            conf = confusion[t][j]; tot = int(conf.sum())
            conf_frac = {J_NAMES[k]: round(float(conf[k]) / max(tot, 1), 3)
                         for k in range(N_J) if conf[k] > 0}
            summary["translators"][t][J_NAMES[j]] = {
                "recall": float(np.mean(r)), "precision": float(np.mean(acc[t][j]["precision"])),
                "iou": float(np.mean(io)), "null_iou": float(np.mean(acc[t][j]["null_iou"])),
                "n_frames": len(r), "region_pred_frac": conf_frac}

    os.makedirs(args.out, exist_ok=True)
    outp = os.path.join(args.out, f"premise_seed{args.seed}.json")
    with open(outp, "w") as fh:
        json.dump(summary, fh, indent=2)

    # console table (IoU)
    organs_seen = [J_NAMES[j] for j in ORGAN_JIDS
                   if any(J_NAMES[j] in summary["translators"][t] for t in TRANSLATORS)]
    print(f"\n[seed {args.seed}] per-organ IoU of real-trained J on sim content:")
    hdr = f"{'organ':12s}" + "".join(f"{t:>11s}" for t in TRANSLATORS)
    print(hdr)
    for name in organs_seen:
        row = f"{name:12s}"
        for t in TRANSLATORS:
            v = summary["translators"][t].get(name)
            row += f"{(v['iou'] if v else float('nan')):11.3f}"
        print(row)
    print(f"[seed {args.seed}] wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
