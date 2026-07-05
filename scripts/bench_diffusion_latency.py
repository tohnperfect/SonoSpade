#!/usr/bin/env python3
"""Measured per-denoising-pass latency of a REPRESENTATIVE semantic-diffusion U-Net, on the same
hardware and single-frame (batch-1) regime as scripts/bench_latency.py, to replace the estimated
diffusion row with a benchmark.

Diffusion inference latency is set by the denoising U-Net architecture and the number of steps T,
NOT by the trained weights, so an untrained but architecturally faithful U-Net gives the real
per-pass cost. We use an ADM/guided-diffusion-style U-Net (the family Echo-from-noise's semantic
diffusion model builds on): GroupNorm+SiLU residual blocks with a timestep embedding, self-attention
at the low resolutions, label-map conditioning by input concatenation, at SonoSPADE's own output
resolution (128x128). We deliberately size it CONSERVATIVELY (base 64, channel mult 1-2-4-8), lighter
than the base-128/256 configs common in the cited works, so the reported diffusion cost is a lower
bound and the speed gap is not inflated. Reports ms/denoising-pass and s/frame at several T.
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np


def _groups(c):
    return min(32, c) if c % min(32, c) == 0 else 1


def build_unet(base=64, mult=(1, 2, 4, 8), in_ch=14, out_ch=1, attn_res=(16, 8), res_blocks=2,
               img=128):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    temb_dim = base * 4

    def timestep_embedding(t, dim):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    class ResBlock(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.norm1 = nn.GroupNorm(_groups(cin), cin)
            self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
            self.emb = nn.Linear(temb_dim, cout)
            self.norm2 = nn.GroupNorm(_groups(cout), cout)
            self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
            self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

        def forward(self, x, t):
            h = self.conv1(F.silu(self.norm1(x)))
            h = h + self.emb(F.silu(t))[:, :, None, None]
            h = self.conv2(F.silu(self.norm2(h)))
            return h + self.skip(x)

    class Attn(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.norm = nn.GroupNorm(_groups(c), c)
            self.qkv = nn.Conv2d(c, c * 3, 1)
            self.proj = nn.Conv2d(c, c, 1)

        def forward(self, x):
            B, C, H, W = x.shape
            qkv = self.qkv(self.norm(x)).reshape(B, 3, C, H * W)
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
            attn = torch.softmax((q.transpose(1, 2) @ k) / math.sqrt(C), dim=-1)  # (B,HW,HW)
            h = (v @ attn.transpose(1, 2)).reshape(B, C, H, W)
            return x + self.proj(h)

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.temb = nn.Sequential(nn.Linear(base, temb_dim), nn.SiLU(),
                                      nn.Linear(temb_dim, temb_dim))
            self.stem = nn.Conv2d(in_ch, base, 3, padding=1)
            chs = [base]
            self.downs = nn.ModuleList()
            c = base
            res = img
            for i, m in enumerate(mult):
                cout = base * m
                for _ in range(res_blocks):
                    blk = nn.ModuleList([ResBlock(c, cout)])
                    if res in attn_res:
                        blk.append(Attn(cout))
                    self.downs.append(blk); c = cout; chs.append(c)
                if i != len(mult) - 1:
                    self.downs.append(nn.ModuleList([nn.Conv2d(c, c, 3, stride=2, padding=1)]))
                    chs.append(c); res //= 2
            self.mid = nn.ModuleList([ResBlock(c, c), Attn(c), ResBlock(c, c)])
            self.ups = nn.ModuleList()
            for i, m in reversed(list(enumerate(mult))):
                cout = base * m
                for _ in range(res_blocks + 1):
                    blk = nn.ModuleList([ResBlock(c + chs.pop(), cout)])
                    if res in attn_res:
                        blk.append(Attn(cout))
                    self.ups.append(blk); c = cout
                if i != 0:
                    self.ups.append(nn.ModuleList([nn.Upsample(scale_factor=2, mode="nearest"),
                                                   nn.Conv2d(c, c, 3, padding=1)]))
                    res *= 2
            self.out = nn.Sequential(nn.GroupNorm(_groups(c), c), nn.SiLU(),
                                     nn.Conv2d(c, out_ch, 3, padding=1))

        def forward(self, x, t):
            temb = self.temb(timestep_embedding(t, base))
            h = self.stem(x)
            hs = [h]
            for blk in self.downs:
                if isinstance(blk[0], nn.Conv2d):      # downsample
                    h = blk[0](h)
                else:
                    h = blk[0](h, temb)
                    if len(blk) > 1:
                        h = blk[1](h)
                hs.append(h)
            for mod in self.mid:
                h = mod(h, temb) if isinstance(mod, ResBlock) else mod(h)
            for blk in self.ups:
                if isinstance(blk[0], nn.Upsample):    # upsample
                    h = blk[1](blk[0](h))
                else:
                    h = torch.cat([h, hs.pop()], dim=1)
                    h = blk[0](h, temb)
                    if len(blk) > 1:
                        h = blk[1](h)
            return self.out(h)

    return UNet()


def bench(device_str, base, img=128, n=60, warmup=10):
    import torch
    dev = torch.device(device_str)
    net = build_unet(base=base, img=img).to(dev).eval()
    nparam = sum(p.numel() for p in net.parameters())
    x = torch.randn(1, 14, img, img, device=dev)
    t = torch.randint(0, 1000, (1,), device=dev)
    with torch.no_grad():
        for _ in range(warmup):
            net(x, t)
        if device_str == "mps":
            torch.mps.synchronize()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            net(x, t)
            if device_str == "mps":
                torch.mps.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts)
    return float(ts.mean()), float(ts.std()), float(np.median(ts)), nparam


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args(argv)
    import torch
    print(f"torch {torch.__version__}; threads {torch.get_num_threads()}; "
          f"representative ADM/SDM-style U-Net, base={args.base}, {args.img}x{args.img}")
    res = {}
    for dev in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
        mean, std, med, npar = bench(dev, args.base, args.img, n=args.n)
        res[dev] = mean
        print(f"{dev:4s}  {mean:8.2f} ms/denoising-pass (median {med:7.2f}, sd {std:6.2f})  "
              f"[{npar/1e6:.1f}M params]")
    print("\nper-FRAME latency = ms/pass x T denoising steps:")
    for dev in res:
        print(f"  [{dev}]")
        for T in [25, 50, 250, 1000]:
            s = res[dev] * T / 1000.0
            print(f"    T={T:4d}: {s:8.2f} s/frame ({1.0/s:.4f} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
