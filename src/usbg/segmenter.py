"""segmenter.py: S_US, a compact U-Net that labels a B-mode into canonical tissue classes.

S_US is the segmentation-consistency anchor of SonoSPADE. It is pretrained for FREE on the
simulator's perfectly aligned (physics image, label slice) pairs, where the label is exact, so it
reaches high Dice with no manual annotation. During translator training it is FROZEN and used two
ways: (1) a consistency loss forces S_US(G(phys, label)) to match the input label, so the generator
cannot ignore the layout and recolor globally, and (2) it pseudo-labels the unpaired real pool so
the OASIS discriminator has a per-pixel semantic target without any real annotation.

Canonical classes are volume.N_LABELS (13: background..soft_tissue). Input is a single-channel
B-mode in [0,1] at the render orientation (N_SAMPLES rows, N_LINES cols) or any square resize of it;
output is per-pixel class logits (B, n_classes, H, W).
"""
from __future__ import annotations

import numpy as np

from .volume import (N_LABELS, BACKGROUND, LIVER, GALLBLADDER, VESSEL, KIDNEY, GI, PANCREAS,
                     SPLEEN, MUSCLE, FAT, BONE, LUNG, SOFT_TISSUE)

# ======================================================================================
# Grouped label scheme (improvement 1): merge the acoustically degenerate soft tissues into one
# class so S_US is not asked to separate the inseparable. Fat, GI wall, pancreas, muscle, and generic
# soft tissue have near-equal backscatter in ACOUSTIC_LOTUS and render near-identically in B-mode, so
# a segmenter cannot tell them apart from grayscale and their per-class Dice is meaninglessly low.
# Acoustically distinct structures keep their own class. The generator still conditions on the full
# canonical scheme; only S_US and the OASIS discriminator operate on the grouped scheme.
GROUPED_MAP: dict[int, int] = {
    BACKGROUND: 0, LIVER: 1, GALLBLADDER: 2, VESSEL: 3, KIDNEY: 4, SPLEEN: 5,
    GI: 6, PANCREAS: 6, MUSCLE: 6, FAT: 6, SOFT_TISSUE: 6, BONE: 7, LUNG: 8,
}
N_GROUPED = 9
GROUPED_NAMES = {0: "background", 1: "liver", 2: "gallbladder", 3: "vessel", 4: "kidney",
                 5: "spleen", 6: "soft_tissue", 7: "bone", 8: "lung"}


def grouped_map_array() -> np.ndarray:
    """Length-N_LABELS int array mapping each canonical id to its grouped id (identity elsewhere)."""
    a = np.arange(N_LABELS, dtype=np.int64)
    for k, v in GROUPED_MAP.items():
        a[k] = v
    return a


def remap_labels(label: np.ndarray, map_array: np.ndarray) -> np.ndarray:
    """Apply a canonical -> scheme id map (from grouped_map_array) to a label array."""
    return np.asarray(map_array, np.int64)[np.asarray(label)]


def _double_conv(cin, cout):
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True))


def build_segmenter(n_classes: int = N_LABELS, base: int = 32, depth: int = 4, in_ch: int = 1,
                    pool: str = "max"):
    """Compact U-Net (encoder / bottleneck / decoder with skip connections). `depth` down/up stages,
    `base` first-stage width (doubles each stage). Returns an nn.Module: (B,in_ch,H,W) -> logits.

    pool: 'max' (default) or 'avg'. Use 'avg' when the net needs to be TWICE differentiable (the
    OASIS discriminator under an R1 gradient penalty): max pooling is not infinitely differentiable,
    average pooling is."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    Pool = nn.AvgPool2d if pool == "avg" else nn.MaxPool2d

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.depth = depth
            self.downs = nn.ModuleList()
            self.pools = nn.ModuleList()
            c = in_ch
            w = base
            for _ in range(depth):
                self.downs.append(_double_conv(c, w))
                self.pools.append(Pool(2))
                c = w
                w *= 2
            self.bottleneck = _double_conv(c, w)
            self.ups = nn.ModuleList()
            self.up_convs = nn.ModuleList()
            for _ in range(depth):
                self.ups.append(nn.Conv2d(w, w // 2, 3, padding=1))
                self.up_convs.append(_double_conv(w, w // 2))    # w//2 (up) + w//2 (skip) = w in
                w //= 2
            self.head = nn.Conv2d(w, n_classes, 1)
            self.n_classes = n_classes

        def forward(self, x):
            skips = []
            h = x
            for down, pool in zip(self.downs, self.pools):
                h = down(h)
                skips.append(h)
                h = pool(h)
            h = self.bottleneck(h)
            for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
                h = F.interpolate(h, size=skip.shape[2:], mode="nearest")
                h = conv(torch.cat([up(h), skip], dim=1))
            return self.head(h)

    return UNet()


def align_label_to_image(label_slice: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Put a slicer label (N_LINES, N_SAMPLES) into the rendered-image orientation and size.

    render_bmode_* returns the transpose of the label slice (depth down rows), so the label that
    lines up with a rendered image is label_slice.T, nearest-resized to out_hw (class ids must not
    be interpolated). Returns int64 (out_hw) so it is a drop-in target for cross entropy / Dice.
    """
    from PIL import Image
    lab = np.asarray(label_slice)
    lab_img = lab.T                                              # (N_SAMPLES, N_LINES): matches render
    h, w = out_hw
    if lab_img.shape != (h, w):
        lab_img = np.asarray(Image.fromarray(lab_img.astype(np.uint8)).resize(
            (w, h), Image.NEAREST), dtype=np.int64)
    return lab_img.astype(np.int64)


def dice_score(logits, target, n_classes: int = N_LABELS, ignore_background: bool = True,
               eps: float = 1e-6) -> float:
    """Mean Dice over the classes actually present in `target` (per-image argmax vs ground truth).

    logits: (B,C,H,W). target: (B,H,W) long. Background (class 0) is skipped by default because it
    dominates the frame and would inflate the score."""
    import torch
    pred = logits.argmax(dim=1)
    dices = []
    start = 1 if ignore_background else 0
    for cls in range(start, n_classes):
        t = (target == cls)
        if not bool(t.any()):
            continue
        p = (pred == cls)
        inter = (p & t).sum().float()
        denom = p.sum().float() + t.sum().float()
        dices.append((2 * inter + eps) / (denom + eps))
    if not dices:
        return 1.0
    return float(torch.stack(dices).mean())


def seg_ce_dice_loss(logits, target, n_classes: int = N_LABELS, dice_weight: float = 1.0):
    """Cross entropy plus soft (differentiable) Dice, the segmentation-consistency objective.
    logits: (B,C,H,W). target: (B,H,W) long. Returns a scalar tensor."""
    import torch
    import torch.nn.functional as F
    ce = F.cross_entropy(logits, target)
    probs = F.softmax(logits, dim=1)
    tgt = F.one_hot(target.clamp(0, n_classes - 1), n_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * tgt).sum(dims)
    denom = probs.sum(dims) + tgt.sum(dims)
    dice = 1.0 - ((2 * inter + 1e-6) / (denom + 1e-6)).mean()
    return ce + dice_weight * dice


class Segmenter:
    """Inference / training wrapper around the S_US U-Net (device handling, save / load)."""

    def __init__(self, device=None, n_classes: int = N_LABELS, base: int = 32, depth: int = 4,
                 group_map=None):
        import torch
        from .render_lotus import get_device
        self.device = device or get_device()
        self.n_classes = n_classes
        # canonical id -> this segmenter's class id (None means canonical == identity, 13 classes)
        self.group_map = np.asarray(group_map, np.int64) if group_map is not None else None
        self.net = build_segmenter(n_classes, base=base, depth=depth).to(self.device)
        self._cfg = {"n_classes": n_classes, "base": base, "depth": depth,
                     "group_map": (self.group_map.tolist() if self.group_map is not None else None)}

    def remap(self, canonical_label):
        """Map a canonical (13-class) label into this segmenter's class scheme (identity if canonical)."""
        if self.group_map is None:
            return np.asarray(canonical_label)
        return remap_labels(canonical_label, self.group_map)

    def predict_logits(self, img01: np.ndarray):
        """img01: HxW float [0,1] -> logits tensor (1, C, H, W) on device (no grad)."""
        import torch
        a = torch.from_numpy(np.asarray(img01, np.float32))[None, None].to(self.device)
        self.net.eval()
        with torch.no_grad():
            return self.net(a)

    def predict(self, img01: np.ndarray) -> np.ndarray:
        """img01: HxW float [0,1] -> (H,W) int64 argmax label map."""
        return self.predict_logits(img01)[0].argmax(0).cpu().numpy().astype(np.int64)

    def save(self, path: str):
        import os
        import torch
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"state": self.net.state_dict(), **self._cfg}, path)

    @classmethod
    def load(cls, path: str, device=None) -> "Segmenter":
        import torch
        from .render_lotus import get_device
        dev = device or get_device()
        ck = torch.load(path, map_location=dev, weights_only=False)
        seg = cls(device=dev, n_classes=ck.get("n_classes", N_LABELS),
                  base=ck.get("base", 32), depth=ck.get("depth", 4),
                  group_map=ck.get("group_map"))
        seg.net.load_state_dict(ck["state"])
        seg.net.eval()
        return seg


__all__ = ["build_segmenter", "align_label_to_image", "dice_score", "seg_ce_dice_loss",
           "Segmenter", "N_LABELS", "GROUPED_MAP", "N_GROUPED", "GROUPED_NAMES",
           "grouped_map_array", "remap_labels"]
