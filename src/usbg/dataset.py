"""dataset.py: write / read the (image, pose, twist, maneuver, ...) dataset.

Two on-disk forms, both from the same in-memory ClipSample list:
  * npz  : the canonical store, object arrays keyed with the recognizer's field names
           (twist, rel6/seqs, valid, labels, next_id, ttnb) so par.eval loaders and the
           Phase 5 autolabeler consume it unchanged, plus the new `images`.
  * LeRobot-style export : per-episode frames as PNGs + a records.jsonl + info.json using
           LeRobot v2 feature names (observation.images.bmode, observation.state, action,
           task). Swap jsonl for parquet (add pyarrow) to get a drop-in LeRobot v2 dataset.

Per-frame fields (handoff section 7): image (H,W uint8), pose (4x4), twist (6, the action,
same normalization convention as par.contracts), rel6 (6), maneuver (int, CLASSES_6),
next_maneuver (int, -1 within a single-action clip), ttnb (frames to the clip's end),
instruction (str), source ("clean"/"noisy"), and trajectory / frame ids.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from .contracts_bridge import CHANNELS_TWIST, CHANNELS_REL6, CLASSES_5, SESSION_CLASSES_6, \
    DT_C, OUT_RATE_HZ, NEXT_ID_NONE, GEOMETRY_SHA256


@dataclass
class ClipSample:
    """One single-action trajectory with its rendered frames."""
    images: np.ndarray     # (N, H, W) uint8
    poses: np.ndarray      # (N, 4, 4)
    twist: np.ndarray      # (N, 6) float32
    rel6: np.ndarray       # (N, 6) float32
    valid: np.ndarray      # (N,) bool
    label: int             # clip maneuver (0..4)
    instruction: str
    source: str            # clean | noisy
    case: str              # source CT case id
    traj_id: int
    forces: np.ndarray = field(default=None)   # (N,) float32 contact force (QC), optional

    def per_frame_maneuver(self):
        return np.full(len(self.images), self.label, dtype=np.int64)

    def next_id(self):
        return np.full(len(self.images), NEXT_ID_NONE, dtype=np.int64)

    def ttnb(self):
        n = len(self.images)
        return np.arange(n - 1, -1, -1, dtype=np.int64)   # frames to end of maneuver


def resize_uint8(img_float01: np.ndarray, size: int) -> np.ndarray:
    """Resize a float [0,1] image to (size,size) uint8 for compact storage."""
    from PIL import Image
    a = np.clip(np.asarray(img_float01) * 255.0, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(a).resize((size, size), Image.BILINEAR), dtype=np.uint8)


# ======================================================================================
# npz (canonical, recognizer-compatible)
# ======================================================================================
def write_npz(path: str, clips: list[ClipSample], image_size: int, extra_meta: dict | None = None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    obj = lambda xs: np.array(xs, dtype=object)
    meta = {
        "schema_version": "usbg-1.0",
        "out_rate_hz": OUT_RATE_HZ, "dt_c": DT_C,
        "twist_channels": list(CHANNELS_TWIST), "rel6_channels": list(CHANNELS_REL6),
        "classes": list(CLASSES_5), "session_classes": list(SESSION_CLASSES_6),
        "image_size": image_size, "geometry_hash": GEOMETRY_SHA256,
        "n_clips": len(clips),
        "n_clean": sum(c.source == "clean" for c in clips),
        "n_noisy": sum(c.source == "noisy" for c in clips),
        "cases": sorted({c.case for c in clips}),
    }
    if extra_meta:
        meta.update(extra_meta)
    np.savez_compressed(
        path,
        images=obj([c.images for c in clips]),
        poses=obj([c.poses for c in clips]),
        twist=obj([c.twist.astype(np.float32) for c in clips]),
        seqs=obj([c.rel6.astype(np.float32) for c in clips]),        # rel6 under the npz key `seqs`
        valid=obj([c.valid for c in clips]),
        frame_labels=obj([c.per_frame_maneuver() for c in clips]),
        next_id=obj([c.next_id() for c in clips]),
        ttnb=obj([c.ttnb() for c in clips]),
        labels=np.array([c.label for c in clips], dtype=np.int64),
        instructions=np.array([c.instruction for c in clips]),
        sources=np.array([c.source for c in clips]),
        cases=np.array([c.case for c in clips]),
        traj_ids=np.array([c.traj_id for c in clips], dtype=np.int64),
        classes=np.array(list(CLASSES_5)),
        meta=np.array(json.dumps(meta)),
    )
    return path, meta


def load_npz(path: str) -> tuple[list[dict], dict]:
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    n = len(d["labels"])
    clips = []
    for i in range(n):
        clips.append({
            "images": d["images"][i], "poses": d["poses"][i], "twist": d["twist"][i],
            "rel6": d["seqs"][i], "valid": d["valid"][i],
            "frame_labels": d["frame_labels"][i], "next_id": d["next_id"][i],
            "ttnb": d["ttnb"][i], "label": int(d["labels"][i]),
            "instruction": str(d["instructions"][i]), "source": str(d["sources"][i]),
            "case": str(d["cases"][i]), "traj_id": int(d["traj_ids"][i]),
        })
    return clips, meta


# ======================================================================================
# aligned (image, label) pairs for SonoSPADE (free supervision: S_US pretraining + G content)
# ======================================================================================
def write_pairs_npz(path: str, images: np.ndarray, labels: np.ndarray,
                    sessions: np.ndarray | None = None, extra_meta: dict | None = None):
    """Store perfectly aligned (physics image, canonical label) pairs for SonoSPADE.

    images  : (N, H, W) uint8 physics B-mode. labels : (N, H, W) uint8 canonical label maps in the
              SAME orientation and size as the images (see segmenter.align_label_to_image). These
              are the generator's content input and the free, exact supervision that pretrains S_US.
    sessions: optional (N,) session id per frame (e.g. the source CT case), so S_US and the
              translator split by session, never by frame (the normalization contract, no leakage).
    """
    images = np.asarray(images, dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.uint8)
    if images.shape != labels.shape:
        raise ValueError(f"images {images.shape} and labels {labels.shape} must match (aligned pairs)")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    meta = {"schema_version": "usbg-pairs-1.0", "n": int(images.shape[0]),
            "shape": list(images.shape[1:])}
    if extra_meta:
        meta.update(extra_meta)
    arrays = {"images": images, "labels": labels, "meta": json.dumps(meta)}
    if sessions is not None:
        arrays["sessions"] = np.asarray(sessions)
    np.savez_compressed(path, **arrays)
    return path, meta


def load_pairs_npz(path: str):
    """Return (images uint8 (N,H,W), labels uint8 (N,H,W), sessions (N,) or None, meta dict)."""
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"])) if "meta" in d else {}
    sessions = d["sessions"] if "sessions" in d else None
    return d["images"], d["labels"], sessions, meta


# ======================================================================================
# LeRobot-style export (PNG frames + jsonl + info.json)
# ======================================================================================
def write_lerobot(out_dir: str, clips: list[ClipSample], image_size: int):
    from PIL import Image
    img_root = os.path.join(out_dir, "images", "observation.images.bmode")
    data_dir = os.path.join(out_dir, "data", "chunk-000")
    os.makedirs(img_root, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "meta"), exist_ok=True)

    total_frames = 0
    for ep, c in enumerate(clips):
        ep_img_dir = os.path.join(img_root, f"episode_{ep:06d}")
        os.makedirs(ep_img_dir, exist_ok=True)
        records = []
        man = c.per_frame_maneuver(); nid = c.next_id(); tt = c.ttnb()
        for f in range(len(c.images)):
            fn = os.path.join(ep_img_dir, f"frame_{f:06d}.png")
            Image.fromarray(c.images[f]).save(fn)
            records.append({
                "episode_index": ep, "frame_index": f,
                "timestamp": round(f * DT_C, 4),
                "observation.images.bmode": os.path.relpath(fn, out_dir),
                "observation.state": c.rel6[f].astype(float).tolist(),   # rel6 pose
                "pose": c.poses[f].astype(float).reshape(-1).tolist(),
                "action": c.twist[f].astype(float).tolist(),             # 6-DOF twist
                "maneuver": int(man[f]), "next_maneuver": int(nid[f]), "ttnb": int(tt[f]),
                "valid": bool(c.valid[f]), "source": c.source, "case": c.case,
                "task": c.instruction,
            })
        with open(os.path.join(data_dir, f"episode_{ep:06d}.jsonl"), "w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        total_frames += len(records)

    info = {
        "codebase_version": "usbg-lerobot-style-1.0", "robot_type": "sim_us_probe",
        "fps": OUT_RATE_HZ, "total_episodes": len(clips), "total_frames": total_frames,
        "features": {
            "observation.images.bmode": {"dtype": "image", "shape": [image_size, image_size, 1]},
            "observation.state": {"dtype": "float32", "shape": [6], "names": list(CHANNELS_REL6)},
            "action": {"dtype": "float32", "shape": [6], "names": list(CHANNELS_TWIST)},
            "maneuver": {"dtype": "int64", "shape": [1], "names": list(CLASSES_5)},
        },
        "note": "jsonl per episode; swap for parquet (add pyarrow) for a drop-in LeRobot v2 dataset.",
    }
    with open(os.path.join(out_dir, "meta", "info.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    return out_dir, info


# ======================================================================================
# contact sheet
# ======================================================================================
def contact_sheet(clips: list[ClipSample], out_png: str, per_clip: int = 4, max_clips: int = 8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = clips[:max_clips]
    fig, axes = plt.subplots(len(show), per_clip, figsize=(2.0 * per_clip, 2.0 * len(show)),
                             squeeze=False)
    for r, c in enumerate(show):
        idx = np.linspace(0, len(c.images) - 1, per_clip).round().astype(int)
        for k, fi in enumerate(idx):
            ax = axes[r][k]
            ax.imshow(c.images[fi], cmap="gray", vmin=0, vmax=255, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if k == 0:
                ax.set_ylabel(f"{CLASSES_5[c.label]}\n{c.source}/{c.case}", fontsize=7)
            ax.set_title(f"f{fi}", fontsize=7)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.suptitle("dataset sample clips (rows) x frames (cols)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png
