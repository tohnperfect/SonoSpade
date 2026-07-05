#!/usr/bin/env python3
"""make_figure.py: summary figure for the per-tissue premise check."""
from __future__ import annotations
import glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

TR = ["physics", "cut", "cyclegan", "sonospade"]
COL = {"physics": "#888", "cut": "#e08a2b", "cyclegan": "#c0392b", "sonospade": "#2e86c1"}


def agg(d, organ, metric="iou"):
    vals, ceil, nul = [], [], []
    for f in sorted(glob.glob(os.path.join(d, "premise_seed*.json"))):
        s = json.load(open(f))
        row = {t: s["translators"].get(t, {}).get(organ) for t in TR}
        if all(row[t] is not None for t in TR):
            vals.append([row[t][metric] for t in TR])
            son = s["translators"]["sonospade"].get(organ, {})
            ceil.append(s.get("ceiling_real_val", {}).get(organ, {}).get(metric, np.nan))
            nul.append(son.get("null_iou", np.nan))
    if not vals:
        return None
    v = np.array(vals)
    return v.mean(0), v.std(0), np.nanmean(ceil), np.nanmean(nul)


def bars(ax, d, organ, title, metric="iou"):
    r = agg(d, organ, metric)
    ax.set_title(title, fontsize=10)
    if r is None:
        ax.text(0.5, 0.5, "not present\n(untestable)", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#999"); ax.set_xticks([]); return
    m, sd, ceil, nul = r
    x = np.arange(len(TR))
    ax.bar(x, m, yerr=sd, color=[COL[t] for t in TR], capsize=3, edgecolor="k", linewidth=0.4)
    if np.isfinite(ceil):
        ax.axhline(ceil, ls="--", c="green", lw=1.2, label=f"real ceiling {ceil:.2f}")
    if metric == "iou" and np.isfinite(nul):
        ax.axhline(nul, ls=":", c="purple", lw=1.0, label=f"flood null {nul:.2f}")
    ax.set_xticks(x); ax.set_xticklabels(["phys", "CUT", "CycGAN", "SonoSP"], fontsize=8, rotation=20)
    ax.set_ylim(0, max(0.75, (ceil if np.isfinite(ceil) else 0) * 1.1))
    ax.legend(fontsize=6, loc="upper left")
    ax.set_ylabel(f"J {metric}", fontsize=8)


fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
bars(axes[0], "outputs/premise/full", "liver", "LIVER (full-frame J)\nSonoSPADE wins all seeds")
bars(axes[1], "outputs/premise/crop", "liver", "LIVER (fan-crop J)\nSonoSPADE wins all seeds")
bars(axes[2], "outputs/premise/spleen_full", "spleen", "SPLEEN (prominent, full-J)\nall ~0: untestable")
bars(axes[3], "outputs/premise/kidney_full", "kidney", "KIDNEY (targeted, full-J)\nall ~0: untestable")
fig.suptitle("Premise check: real-trained organ recognizer J on translator-textured sim content "
             "(3 seeds, mean+-std). Fair CUT = exp_real_cut.pt.", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("outputs/premise/premise_summary.png", dpi=140)
print("wrote outputs/premise/premise_summary.png")
