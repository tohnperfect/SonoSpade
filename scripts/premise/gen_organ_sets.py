#!/usr/bin/env python3
"""gen_organ_sets.py: render organ-TARGETED sim content (physics image + canonical label) for the
fairer premise test. Unlike pairs_med (liver-dominated, ~70% liver, spleen ~0), each set seats a
chosen non-dominant organ (kidney / spleen / gallbladder / vessel) under the probe so it is
prominent -- the case where per-tissue SPECIFICITY should matter (a global recolor puts the wrong
tissue's texture in that organ's region). Output npz matches the pairs_med schema (images uint8,
labels uint8 aligned to images), so scripts/premise/eval_premise.py runs on it via --sim.
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--organ", required=True,
                    choices=["kidney", "spleen", "gallbladder", "vessel", "liver"])
    ap.add_argument("--cases", nargs="+", default=None, help="default: all cached cases")
    ap.add_argument("--n-per-case", type=int, default=20)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--randomize", default="mild", choices=["full", "mild", "off"])
    ap.add_argument("--depth", type=float, default=0.05, help="depth_m to seat the organ at")
    ap.add_argument("--min-frac", type=float, default=0.0,
                    help="keep only frames where the target organ is >= this fraction of the frame "
                         "(so the eval set is genuinely organ-PROMINENT, not liver-dominated)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    import glob
    import json
    from usbg import render_lotus as R
    from usbg.dataset import write_pairs_npz
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from make_dataset import gen_labeled_pairs

    device = args.device or R.get_device()
    cases = args.cases or [os.path.basename(f)[:-4] for f in sorted(glob.glob("data/cache/*.npz"))]
    print(f"[{args.organ}] cases={len(cases)} n_per_case={args.n_per_case} device={device}")
    imgs, labs, sess = gen_labeled_pairs(cases, args.organ, args.n_per_case, args.size, device,
                                         seed=args.seed, randomize=args.randomize, depth_m=args.depth)
    # coverage of the target organ in the rendered set
    from usbg.volume import USBG_LABEL_NAMES
    name_to_id = {v: k for k, v in USBG_LABEL_NAMES.items()}
    tid = name_to_id[args.organ]
    frac = (labs == tid).reshape(len(labs), -1).mean(1)
    if args.min_frac > 0:
        keep = frac >= args.min_frac
        imgs, labs, sess, frac = imgs[keep], labs[keep], sess[keep], frac[keep]
        print(f"[{args.organ}] kept {int(keep.sum())}/{len(keep)} frames with organ >= {args.min_frac}")
    print(f"[{args.organ}] N={len(imgs)}  target-organ pixel-frac: "
          f"median={np.median(frac):.4f} p90={np.percentile(frac,90):.4f} "
          f"#frames>=3%={int((frac>=0.03).sum())}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_pairs_npz(args.out, imgs, labs, sessions=sess,
                    extra_meta={"target": args.organ, "randomize": args.randomize})
    print(f"[{args.organ}] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
