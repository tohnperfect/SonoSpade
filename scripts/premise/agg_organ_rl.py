#!/usr/bin/env python3
"""Aggregate organ-RL results: ground-truth liver visibility per condition (mean +- std over seeds).
Primary = final ground-truth liver fraction; also the gain over start and over the static baseline."""
from __future__ import annotations
import glob, json, os
import numpy as np

CONDS = ["physics", "cut", "cyclegan", "sonospade"]


def load(d, cond):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, f"{cond}_s*.json"))):
        rows.append(json.load(open(f)))
    return rows


def col(rows, phase, key):
    return np.array([r[phase][key] for r in rows], float)


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="outputs/organ_rl")
    a = ap.parse_args()
    print(f"Organ-conditioned (liver) acquisition RL -- ground-truth liver visibility (true label)\n")
    hdr = f"{'condition':10s} {'n':>2s} | {'start_gt':>9s} {'final_gt(FINAL)':>16s} {'gain_vs_start':>13s} " \
          f"{'gain_vs_static':>14s} | {'final_J(2nd)':>12s}"
    print(hdr); print("-" * len(hdr))
    res = {}
    for c in CONDS:
        rows = load(a.dir, c)
        if not rows:
            print(f"{c:10s}  (no runs)"); continue
        fg = col(rows, "final", "final_gt"); sg = col(rows, "final", "start_gt")
        st = col(rows, "final", "static_gt"); fj = col(rows, "final", "final_j")
        res[c] = fg
        print(f"{c:10s} {len(rows):2d} | {sg.mean():9.3f} {fg.mean():7.3f}+-{fg.std():5.3f}  "
              f"{(fg-sg).mean():+13.3f} {(fg-st).mean():+14.3f} | {fj.mean():12.3f}")
    if "sonospade" in res:
        print("\nSonoSPADE vs each (final ground-truth liver, mean delta; +=SonoSPADE better):")
        for c in ["physics", "cut", "cyclegan"]:
            if c in res:
                d = res["sonospade"].mean() - res[c].mean()
                # per-seed win count (paired if same #seeds)
                n = min(len(res["sonospade"]), len(res[c]))
                wins = int(np.sum(res["sonospade"][:n] > res[c][:n]))
                print(f"  vs {c:9s} d={d:+.3f}   (SonoSPADE higher in {wins}/{n} seeds)")


if __name__ == "__main__":
    main()
