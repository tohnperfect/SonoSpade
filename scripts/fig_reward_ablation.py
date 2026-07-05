#!/usr/bin/env python3
"""M3: goal-reaching reward ablation bar figure, from the aggregate (all rewards judged by the SAME
embedding metric + pose). Shows reach success and final embedding similarity for reward in
{embedding, SSIM, NCC}, mean with seed std error bars, plus the static reference."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", default="outputs/scale/aggregate.json")
    ap.add_argument("--out", default="paper/figures/reward_ablation.png")
    args = ap.parse_args(argv)

    d = json.load(open(args.aggregate))["reward_ablation"]
    goal = json.load(open(args.aggregate)).get("goal", {})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rewards = ["embed", "ssim", "ncc"]
    labels = ["embedding", "SSIM", "NCC"]
    succ = [d[r]["success_rate"][0] for r in rewards]
    succ_e = [d[r]["success_rate"][1] for r in rewards]
    fsim = [d[r]["final_sim"][0] for r in rewards]
    fsim_e = [d[r]["final_sim"][1] for r in rewards]
    static_succ = goal.get("static", {}).get("success_rate", [None])[0]

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.4))
    x = np.arange(len(rewards))
    ax[0].bar(x, succ, yerr=succ_e, color=["#37b87a", "#4aa3df", "#9b7cd6"], capsize=4, edgecolor="black")
    if static_succ is not None:
        ax[0].axhline(static_succ, color="#e0635a", ls="--", lw=1, label=f"static {static_succ:.2f}")
        ax[0].legend(fontsize=8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels); ax[0].set_ylabel("reach success rate")
    ax[0].set_title("Reach success by reward"); ax[0].grid(axis="y", alpha=0.3)
    ax[1].bar(x, fsim, yerr=fsim_e, color=["#37b87a", "#4aa3df", "#9b7cd6"], capsize=4, edgecolor="black")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels); ax[1].set_ylabel("final embedding similarity")
    ax[1].set_title("Final similarity by reward"); ax[1].grid(axis="y", alpha=0.3)
    fig.suptitle("Goal-reaching reward ablation (all judged by the trained embedding), mean +/- std over seeds",
                 fontsize=10)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out, dpi=140); plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
