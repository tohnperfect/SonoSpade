#!/usr/bin/env python3
"""aggregate.py: combine per-seed premise results into mean +/- std tables and a verdict.

Reads premise_seed*.json (from eval_premise.py) in a directory and reports, per organ and per
translator, the mean +/- std IoU/recall across seeds, plus SonoSPADE-vs-CUT and SonoSPADE-vs-
CycleGAN deltas. The verdict is intentionally conservative: it names the organs where SonoSPADE
beats BOTH global recolors across all seeds (seed-robust), and flags under-powered organs.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

TRANSLATORS = ["physics", "cut", "cyclegan", "sonospade"]
ORGANS = ["liver", "gallbladder", "vessel", "kidney", "spleen"]


def load_seeds(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "premise_seed*.json"))):
        out.append(json.load(open(f)))
    return out


def collect(seeds, organ, translator, metric):
    vals = []
    for s in seeds:
        v = s["translators"].get(translator, {}).get(organ)
        if v is not None:
            vals.append(v[metric])
    return np.array(vals, dtype=float)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/premise")
    ap.add_argument("--metric", default="iou", choices=["iou", "recall", "precision"])
    args = ap.parse_args(argv)

    seeds = load_seeds(args.dir)
    if not seeds:
        print(f"no premise_seed*.json in {args.dir}"); return 1
    nseed = len(seeds)
    print(f"aggregating {nseed} seed(s) from {args.dir}  (metric={args.metric})\n")

    # ceiling (held-out real) per organ, averaged over seeds
    ceil = {o: [] for o in ORGANS}
    for s in seeds:
        for o, v in s.get("ceiling_real_val", {}).items():
            ceil[o].append(v.get(args.metric, v.get("iou")))

    hdr = f"{'organ':12s}" + "".join(f"{t:>16s}" for t in TRANSLATORS) + f"{'real-ceiling':>16s}"
    print(hdr); print("-" * len(hdr))
    verdict = {}
    for o in ORGANS:
        row = f"{o:12s}"
        means = {}
        for t in TRANSLATORS:
            a = collect(seeds, o, t, args.metric)
            if len(a):
                means[t] = a.mean()
                row += f"{a.mean():8.3f}+-{a.std():5.3f}"
            else:
                means[t] = float("nan"); row += f"{'-':>16s}"
        c = np.array(ceil[o], float)
        row += f"{(c.mean() if len(c) else float('nan')):8.3f}+-{(c.std() if len(c) else 0):5.3f}"
        # trivial 'predict organ everywhere' null IoU (floor a volume-flooding translator hits)
        nulls = collect(seeds, o, "sonospade", "null_iou")
        if len(nulls) and args.metric == "iou":
            row += f"   null={nulls.mean():.3f}"
        print(row)
        # per-seed head-to-head: does SonoSPADE beat BOTH global recolors on every seed?
        son = collect(seeds, o, "sonospade", args.metric)
        cut = collect(seeds, o, "cut", args.metric)
        cyc = collect(seeds, o, "cyclegan", args.metric)
        if len(son) == nseed and len(cut) == nseed and len(cyc) == nseed and nseed >= 1:
            beats_all = bool(np.all((son > cut) & (son > cyc)))
            verdict[o] = {"beats_both_every_seed": beats_all,
                          "d_vs_cut": float((son - cut).mean()),
                          "d_vs_cyc": float((son - cyc).mean()),
                          "son": float(son.mean()), "ceiling": float(c.mean() if len(c) else np.nan)}

    print("\nSonoSPADE vs global recolor (mean delta, and seed-robust win):")
    for o, v in verdict.items():
        tag = "WIN(all seeds)" if v["beats_both_every_seed"] else "not-robust"
        powered = "" if (v["ceiling"] == v["ceiling"] and v["ceiling"] > 0.2 and v["son"] > 0.02) \
            else "  [UNDER-POWERED: low ceiling or ~0 transfer]"
        print(f"  {o:12s} dCUT={v['d_vs_cut']:+.3f} dCyc={v['d_vs_cyc']:+.3f}  {tag}{powered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
