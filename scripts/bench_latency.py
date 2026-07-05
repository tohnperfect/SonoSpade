#!/usr/bin/env python3
"""Inference-latency benchmark for the SonoSPADE texture stage.

Measures single-frame latency (the RL / interactive regime: one frame per step, batch 1) on CPU
(no GPU) and on the available accelerator, plus a diffusion-cost reference (number of network
evaluations). Reports ms/frame and frames/s. This is the evidence for the low-cost, real-time,
RL-oriented framing: a diffusion synthesizer needs tens to hundreds of denoising steps per frame,
SonoSPADE needs one generator pass.
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def bench(device_str, ckpt, pairs, n=100, warmup=15):
    import torch
    from usbg import texture_gan as TG
    from usbg.dataset import load_pairs_npz
    dev = torch.device(device_str)
    tr = TG.SonoSPADETranslator(ckpt, device=dev)
    imgs, labs, _s, _m = load_pairs_npz(pairs)
    imgs = imgs.astype(np.float32) / 255.0
    labs = labs.astype(np.int64)
    idx = list(range(min(n + warmup, len(imgs))))
    # warmup
    for i in idx[:warmup]:
        tr.translate_aligned(imgs[i % len(imgs)], labs[i % len(imgs)])
    if device_str == "mps":
        torch.mps.synchronize()
    # timed, single-frame
    ts = []
    for i in idx[warmup:]:
        t0 = time.perf_counter()
        tr.translate_aligned(imgs[i % len(imgs)], labs[i % len(imgs)])
        if device_str == "mps":
            torch.mps.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)  # ms
    ts = np.array(ts)
    return float(ts.mean()), float(ts.std()), float(np.median(ts))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/dataset/exp_real_sonospade_curvi.pt")
    ap.add_argument("--pairs", default="data/dataset/pairs_med_curvi.npz")
    ap.add_argument("--n", type=int, default=120)
    args = ap.parse_args(argv)

    import torch
    print(f"torch {torch.__version__}; threads {torch.get_num_threads()}")
    results = {}
    for dev in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
        mean, std, med = bench(dev, args.ckpt, args.pairs, n=args.n)
        fps = 1000.0 / mean
        results[dev] = dict(mean_ms=mean, std_ms=std, median_ms=med, fps=fps)
        print(f"{dev:4s}  {mean:7.2f} ms/frame (median {med:6.2f}, sd {std:5.2f})  ->  {fps:7.1f} fps")
    # diffusion reference: latency scales with the number of denoising steps (one net eval each)
    print("\nreference: a semantic-diffusion synthesizer runs T denoising steps per frame")
    for T in [50, 250, 1000]:
        eq = results["cpu"]["mean_ms"] * T
        print(f"  T={T:4d} steps -> ~{eq/1000:6.2f} s/frame at SonoSPADE's per-pass CPU cost "
              f"({1000.0/eq:.3f} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
