#!/usr/bin/env python3
"""fetch_weights.py: download the released SonoSPADE checkpoints and check each one.

The files are attached to the GitHub release named in weights/MANIFEST.json. Each download is
checked against the size and SHA-256 in the manifest before it is kept; a file already present with
the right hash is not downloaded again. Standard library only. See docs/WEIGHTS.md for what each
file is.

Usage:
  python scripts/fetch_weights.py                                # all six files (about 210 MB)
  python scripts/fetch_weights.py --set sashimi                  # the paper's four files
  python scripts/fetch_weights.py --set usbgym                   # the USB Gym texture stage
  python scripts/fetch_weights.py sonospade_sashimi_linear.pt    # named files only
  python scripts/fetch_weights.py --list                         # show the files, download nothing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "weights" / "MANIFEST.json"
CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def is_valid(path: Path, entry: dict) -> bool:
    return (path.is_file() and path.stat().st_size == entry["bytes"]
            and sha256_of(path) == entry["sha256"])


def download(url: str, dest: Path, entry: dict) -> None:
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "sonospade-fetch-weights"})
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
        got = 0
        while True:
            block = r.read(CHUNK)
            if not block:
                break
            f.write(block)
            got += len(block)
            print(f"\r  {dest.name}: {got / 1e6:6.1f} / {entry['bytes'] / 1e6:.1f} MB", end="",
                  flush=True)
    print()
    if not is_valid(part, entry):
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{dest.name}: size or SHA-256 does not match the manifest; removed")
    os.replace(part, dest)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("names", nargs="*", help="file names from the manifest (default: every file "
                                             "in --set)")
    ap.add_argument("--set", choices=["all", "sashimi", "usbgym"], default="all",
                    help="sashimi = the paper's checkpoints; usbgym = the texture stage USB Gym used")
    ap.add_argument("--out", default=str(REPO / "weights"), help="directory to write into")
    ap.add_argument("--base-url", default=None,
                    help="download from this URL prefix instead of the manifest's (a mirror)")
    ap.add_argument("--list", action="store_true", help="list the files and exit")
    args = ap.parse_args(argv)

    manifest = json.loads(MANIFEST.read_text())
    entries = manifest["files"]
    if args.names:
        known = {e["name"]: e for e in entries}
        unknown = [n for n in args.names if n not in known]
        if unknown:
            ap.error(f"not in the manifest: {', '.join(unknown)} "
                     f"(choose from: {', '.join(known)})")
        entries = [known[n] for n in args.names]
    elif args.set != "all":
        entries = [e for e in entries if e["set"] == args.set]

    if args.list:
        for e in entries:
            print(f"{e['name']:32s} {e['set']:8s} {e['kind']:10s} {e['geometry']:12s} "
                  f"{e['bytes'] / 1e6:5.1f} MB")
        return 0

    base = (args.base_url or manifest["base_url"]).rstrip("/")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    failed = []
    for e in entries:
        dest = out / e["name"]
        if is_valid(dest, e):
            print(f"  {e['name']}: present, SHA-256 matches")
            continue
        try:
            download(f"{base}/{e['name']}", dest, e)
            print(f"  {e['name']}: downloaded, SHA-256 matches")
        except Exception as exc:  # report every file, then fail once at the end
            print(f"  {e['name']}: FAILED ({exc})", file=sys.stderr)
            failed.append(e["name"])
    if failed:
        print(f"{len(failed)} file(s) failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"{len(entries)} file(s) ready in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
