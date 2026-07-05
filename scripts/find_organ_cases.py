#!/usr/bin/env python3
"""Scan an (extracted) TotalSegmentator dataset for cases whose CT actually contains the
target organs (non-empty masks). Default target: liver AND gallbladder (BiTNet aligned).

Usage:
  python scripts/find_organ_cases.py <dataset_dir> [--need liver gallbladder] [--limit N]

Each subject dir is expected to hold ct.nii.gz and segmentations/<structure>.nii.gz.
Prints the matching case dirs (one per line) sorted by gallbladder voxel count, descending.
"""
from __future__ import annotations

import argparse
import os
import sys


def _mask_voxels(path: str) -> int:
    import numpy as np
    import nibabel as nib
    if not os.path.exists(path):
        return 0
    try:
        return int((np.asanyarray(nib.load(path).dataobj) > 0.5).sum())
    except Exception:
        return 0


def find_cases(dataset_dir: str, need: list[str]):
    hits = []
    for root, _dirs, files in os.walk(dataset_dir):
        if "ct.nii.gz" not in files:
            continue
        seg = os.path.join(root, "segmentations")
        if not os.path.isdir(seg):
            continue
        counts = {n: _mask_voxels(os.path.join(seg, f"{n}.nii.gz")) for n in need}
        if all(v > 0 for v in counts.values()):
            hits.append((root, counts))
    key = need[-1] if need else None
    hits.sort(key=lambda rc: rc[1].get(key, 0), reverse=True)
    return hits


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir")
    ap.add_argument("--need", nargs="+", default=["liver", "gallbladder"])
    ap.add_argument("--limit", type=int, default=0, help="print at most N cases (0 = all)")
    args = ap.parse_args(argv)

    hits = find_cases(args.dataset_dir, args.need)
    sys.stderr.write(
        f"found {len(hits)} case(s) with non-empty {' + '.join(args.need)} "
        f"under {args.dataset_dir}\n"
    )
    shown = hits if args.limit <= 0 else hits[: args.limit]
    for root, counts in shown:
        detail = " ".join(f"{k}={v}" for k, v in counts.items())
        sys.stderr.write(f"  {os.path.basename(root)}: {detail}\n")
        print(root)
    return 0 if hits else 1


if __name__ == "__main__":
    raise SystemExit(main())
