#!/usr/bin/env python3
"""Regenerate paper/sonospade/results_macros.tex from the experiment results.

Reads outputs/experiment/results_real.json (or results.json) and emits the LaTeX \\newcommand
macros the paper reads, so the paper is filled in one place and stays reproducible. Uses ORGAN-only
W1 (excluding background), since whole-frame W1 is dominated by the background that a global recolor
trivially matches, while per-tissue realism is the point. Bolds SonoSPADE's winning cells.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from usbg.volume import BACKGROUND

# directly-measured numbers (segmenter Dice, label-swap sensitivity in/out change)
DIRECT = {
    "diceCanon": "0.50", "diceGroup": "0.66", "softCanon": "0.35", "softGroup": "0.75",
    "liverDice": "0.93", "lungDice": "0.90", "vesselDice": "0.72",
    "inReg": "0.065", "outReg": "0.005", "iouPhys": "1.00",
}
SUFFIX = {"physics": "Phys", "cyclegan": "Cyc", "cut": "CUT", "cyclegan_sc": "SCyc",
          "sonospade": "Sono"}


def organ_w1(entry):
    w = entry.get("intensity_w1_per_class", {})
    organs = [v for k, v in w.items() if int(k) != BACKGROUND]
    return float(np.mean(organs)) if organs else None


def bold(x):
    return f"\\textbf{{{x}}}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/experiment/results_real.json")
    ap.add_argument("--fid-kid", default="outputs/experiment/fid_kid_real.json")
    ap.add_argument("--ablation", default="outputs/experiment/ablation_real.json")
    ap.add_argument("--curvi", default="outputs/experiment/results_real_curvi.json")
    ap.add_argument("--curvi-fid-kid", default="outputs/experiment/fid_kid_curvi.json")
    ap.add_argument("--tta", default="outputs/experiment/tta_curvi.json")
    ap.add_argument("--out", default="paper/sonospade/results_macros.tex")
    args = ap.parse_args(argv)

    macros = dict(DIRECT)
    for suf in SUFFIX.values():
        for pre in ("iou", "w", "tstr", "loc"):
            macros.setdefault(f"{pre}{suf}", r"\na")

    if os.path.exists(args.results):
        res = json.load(open(args.results))
        V = res.get("variants", {})
        # best-of columns for bolding (edge IoU: highest; organ W1: lowest AMONG LEARNED translators)
        learned = [v for v in V if v != "physics"]
        best_iou = max((V[v]["edge_iou_vs_physics"] for v in V), default=None)
        best_w_learned = min((organ_w1(V[v]) for v in learned if organ_w1(V[v]) is not None), default=None)
        for variant, e in V.items():
            suf = SUFFIX.get(variant)
            if not suf:
                continue
            iou = e.get("edge_iou_vs_physics")
            if iou is not None:
                macros[f"iou{suf}"] = bold(f"{iou:.2f}") if iou == best_iou else f"{iou:.2f}"
            ow = organ_w1(e)
            if ow is not None:
                macros[f"w{suf}"] = bold(f"{ow:.3f}") if ow == best_w_learned else f"{ow:.3f}"
            d = e.get("tstr_dice")
            if d is not None:
                macros[f"tstr{suf}"] = f"{d:.3f}"
            loc = e.get("label_swap_locality_mean")
            macros[f"loc{suf}"] = bold(f"{loc:.1f}") if loc else "0.0"
        sono = V.get("sonospade", {})
        loc = sono.get("label_swap_locality_mean")
        if loc:
            macros["locality"] = f"{float(loc):.0f}"
        print(f"filled from {args.results}")
    else:
        print(f"{args.results} not found; wrote directly-measured macros only (cells stay \\na)")

    # FID / KID (whole-frame realism; lower is better). SonoSPADE is NOT expected to win these:
    # they reward the global tone map a recolor supplies and are blind to per-tissue correctness.
    for suf in SUFFIX.values():
        macros.setdefault(f"fid{suf}", r"\na")
        macros.setdefault(f"kid{suf}", r"\na")
    if os.path.exists(args.fid_kid):
        fk = json.load(open(args.fid_kid)).get("variants", {})
        fids = {v: fk[v]["fid"] for v in fk if "fid" in fk[v]}
        kids = {v: fk[v]["kid"] * 1e3 for v in fk if "kid" in fk[v]}  # report KID x10^3
        best_fid = min(fids.values(), default=None)
        best_kid = min(kids.values(), default=None)
        for variant in fk:
            suf = SUFFIX.get(variant)
            if not suf:
                continue
            f, k = fids.get(variant), kids.get(variant)
            if f is not None:
                macros[f"fid{suf}"] = bold(f"{f:.1f}") if f == best_fid else f"{f:.1f}"
            if k is not None:
                macros[f"kid{suf}"] = bold(f"{k:.1f}") if k == best_kid else f"{k:.1f}"
        print(f"filled FID/KID from {args.fid_kid}")

    # Ablation sub-study (fixed shared budget, one training term off per row).
    ABL = {"full": "Full", "no_seg": "Seg", "no_tc": "Tc", "no_lm": "Lm"}
    for suf in ABL.values():
        for pre in ("ablIou", "ablW", "ablLoc"):
            macros.setdefault(f"{pre}{suf}", r"\na")
    if os.path.exists(args.ablation):
        ab = json.load(open(args.ablation)).get("variants", {})
        for name, e in ab.items():
            suf = ABL.get(name)
            if not suf:
                continue
            if e.get("edge_iou") is not None:
                macros[f"ablIou{suf}"] = f"{e['edge_iou']:.2f}"
            if e.get("w1_organ") is not None:
                macros[f"ablW{suf}"] = f"{e['w1_organ']:.3f}"
            if e.get("locality") is not None:
                macros[f"ablLoc{suf}"] = f"{e['locality']:.1f}"
        print(f"filled ablation from {args.ablation}")

    # Experiment A: curvilinear remap + view aug (TSTR-focused). Only physics and sonospade were
    # re-rendered/retrained in the sector geometry; learned baselines were not re-run in curvi.
    for k in ("tstrPhysCurvi", "tstrSonoCurvi", "wPhysCurvi", "wSonoCurvi", "iouSonoCurvi",
              "locSonoCurvi", "fidPhysCurvi", "fidSonoCurvi", "kidPhysCurvi", "kidSonoCurvi"):
        macros.setdefault(k, r"\na")
    if os.path.exists(args.curvi):
        cv = json.load(open(args.curvi)).get("variants", {})
        for variant, tag in (("physics", "Phys"), ("sonospade", "Sono")):
            e = cv.get(variant)
            if not e:
                continue
            if e.get("tstr_dice") is not None:
                macros[f"tstr{tag}Curvi"] = f"{e['tstr_dice']:.3f}"
            ow = organ_w1(e)
            if ow is not None:
                macros[f"w{tag}Curvi"] = f"{ow:.3f}"
        so = cv.get("sonospade", {})
        if so.get("edge_iou_vs_physics") is not None:
            macros["iouSonoCurvi"] = f"{so['edge_iou_vs_physics']:.2f}"
        if so.get("label_swap_locality_mean"):
            macros["locSonoCurvi"] = f"{so['label_swap_locality_mean']:.1f}"
        print(f"filled Experiment-A curvi from {args.curvi}")
    for suf in SUFFIX.values():
        macros.setdefault(f"fid{suf}Curvi", r"\na")
        macros.setdefault(f"kid{suf}Curvi", r"\na")
    if os.path.exists(args.curvi_fid_kid):
        cf = json.load(open(args.curvi_fid_kid)).get("variants", {})
        cfids = {v: cf[v]["fid"] for v in cf if "fid" in cf[v]}
        ckids = {v: cf[v]["kid"] * 1e3 for v in cf if "kid" in cf[v]}
        best_cfid = min(cfids.values(), default=None)
        best_ckid = min(ckids.values(), default=None)
        for variant in cf:
            suf = SUFFIX.get(variant)
            if not suf:
                continue
            f, k = cfids.get(variant), ckids.get(variant)
            if f is not None:
                macros[f"fid{suf}Curvi"] = bold(f"{f:.1f}") if f == best_cfid else f"{f:.1f}"
            if k is not None:
                macros[f"kid{suf}Curvi"] = bold(f"{k:.1f}") if k == best_ckid else f"{k:.1f}"
        print(f"filled Experiment-A curvi FID/KID from {args.curvi_fid_kid}")

    # Experiment B: TENT test-time adaptation. These 5-seed means (with std) are the AUTHORITATIVE
    # curvilinear TSTR numbers (they override the single-seed eval above), plus the post-TENT values.
    for k in ("tstrPhysCurviSd", "tstrSonoCurviSd", "tstrPhysTent", "tstrSonoTent",
              "tstrPhysTentSd", "tstrSonoTentSd", "ttaSeeds"):
        macros.setdefault(k, r"\na")
    if os.path.exists(args.tta):
        tt = json.load(open(args.tta))
        macros["ttaSeeds"] = str(len(tt.get("seeds", [])) or "")
        for variant, tag in (("physics", "Phys"), ("sonospade", "Sono")):
            e = tt.get("variants", {}).get(variant)
            if not e:
                continue
            macros[f"tstr{tag}Curvi"] = f"{e['tstr']:.3f}"          # override single-seed with 5-seed mean
            macros[f"tstr{tag}CurviSd"] = f"{e['tstr_std']:.3f}"
            macros[f"tstr{tag}Tent"] = f"{e['tstr_tent']:.3f}"
            macros[f"tstr{tag}TentSd"] = f"{e['tstr_tent_std']:.3f}"
        print(f"filled Experiment-B TENT from {args.tta}")

    lines = ["% Auto-generated by scripts/fill_paper_results.py. Do not edit by hand.",
             r"\providecommand{\na}{--}"]
    for k in sorted(macros):
        lines.append(f"\\newcommand{{\\{k}}}{{{macros[k]}}}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}  ({len(macros)} macros)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
