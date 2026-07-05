"""similarity.py: pluggable B-mode image-similarity metrics for goal-conditioned acquisition.

The goal RL reward is "how similar does the current shot look to the target shot", and it must be
easy to swap in a trained deep model for that judgement. Everything implements one interface:

    class Similarity:  score(cur01, target01) -> float in [0,1]  (higher = more similar)

Provided out of the box (no training, so RL runs immediately):
  * NCCSimilarity  : normalized cross-correlation on a speckle-reduced (blurred, downsampled) image
  * SSIMSimilarity : structural similarity (windowed), robust to small intensity shifts
And a learned slot:
  * EmbeddingSimilarity : cosine similarity in the feature space of ANY encoder that maps
    (B,1,H,W) -> (B,D); load a trained model via TorchScript (from_torchscript) or a state_dict
    for the reference encoder (from_checkpoint). This is where a trained B-mode similarity model
    plugs in.

load_similarity(spec) dispatches: "ncc" | "ssim" | "ts:<path>" (TorchScript) | "embed:<path>"
(reference-encoder state_dict, e.g. from scripts/train_bmode_embedding.py).
"""
from __future__ import annotations

import numpy as np


def _prep(img01, size, blur):
    """float [0,1] HxW -> (size,size) float, optionally blurred to suppress speckle."""
    from scipy.ndimage import gaussian_filter, zoom
    a = np.asarray(img01, dtype=np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    zy, zx = size / a.shape[0], size / a.shape[1]
    a = zoom(a, (zy, zx), order=1)
    if blur > 0:
        a = gaussian_filter(a, blur)
    return a


class Similarity:
    """Base: higher score = more similar, in [0,1]. Frames are float [0,1] HxW, any size."""

    def score(self, cur01, target01) -> float:
        raise NotImplementedError

    def score_batch(self, curs, target01) -> np.ndarray:
        return np.asarray([self.score(c, target01) for c in curs], dtype=np.float32)


class NCCSimilarity(Similarity):
    """Zero-mean normalized cross-correlation on a blurred, downsampled image, mapped to [0,1].
    Smooth and view-sensitive, a good default reward for visual servoing."""

    def __init__(self, size=64, blur=1.5):
        self.size, self.blur = size, blur

    def score(self, cur01, target01) -> float:
        a = _prep(cur01, self.size, self.blur); b = _prep(target01, self.size, self.blur)
        a = a - a.mean(); b = b - b.mean()
        denom = float(np.sqrt((a * a).sum() * (b * b).sum())) + 1e-8
        ncc = float((a * b).sum()) / denom                      # [-1, 1]
        return 0.5 * (ncc + 1.0)


class SSIMSimilarity(Similarity):
    """Windowed structural similarity (Wang 2004), mean over the image, mapped to [0,1]."""

    def __init__(self, size=96, blur=0.8, win=7):
        self.size, self.blur, self.win = size, blur, win

    def score(self, cur01, target01) -> float:
        from scipy.ndimage import uniform_filter
        a = _prep(cur01, self.size, self.blur); b = _prep(target01, self.size, self.blur)
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        mu_a = uniform_filter(a, self.win); mu_b = uniform_filter(b, self.win)
        va = uniform_filter(a * a, self.win) - mu_a ** 2
        vb = uniform_filter(b * b, self.win) - mu_b ** 2
        vab = uniform_filter(a * b, self.win) - mu_a * mu_b
        ssim = ((2 * mu_a * mu_b + c1) * (2 * vab + c2)) / \
               ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2) + 1e-12)
        return float(np.clip(ssim.mean(), -1, 1) * 0.5 + 0.5)


def build_bmode_encoder(embed_dim=128, in_ch=1):
    """Reference convolutional encoder (B,1,H,W) -> (B,embed_dim). Trained by
    scripts/train_bmode_embedding.py; also usable as the goal/similarity backbone."""
    import torch.nn as nn

    def blk(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1),
                             nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    return nn.Sequential(blk(in_ch, 32), blk(32, 64), blk(64, 128), blk(128, embed_dim),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten())


class EmbeddingSimilarity(Similarity):
    """Cosine similarity in a (pluggable, possibly trained) encoder's feature space."""

    def __init__(self, encoder, device=None, size=128):
        import torch
        from .render_lotus import get_device
        self.device = device or get_device()
        self.encoder = encoder.to(self.device).eval() if hasattr(encoder, "to") else encoder
        self.size = size
        self._torch = torch

    def _embed(self, imgs01):
        import numpy as _np
        batch = _np.stack([_prep(i, self.size, 0.0) for i in imgs01], 0)[:, None]
        t = self._torch.from_numpy(batch.astype("float32")).to(self.device)
        with self._torch.no_grad():
            e = self.encoder(t)
        e = e / (e.norm(dim=1, keepdim=True) + 1e-8)
        return e

    def score_batch(self, curs, target01) -> np.ndarray:
        ec = self._embed(list(curs))
        et = self._embed([target01])
        return (ec @ et.t()).squeeze(1).cpu().numpy() * 0.5 + 0.5     # cosine [-1,1] -> [0,1]

    def score(self, cur01, target01) -> float:
        return float(self.score_batch([cur01], target01)[0])

    @classmethod
    def from_torchscript(cls, path, device=None, size=128):
        import torch
        from .render_lotus import get_device
        dev = device or get_device()
        return cls(torch.jit.load(path, map_location=dev), device=dev, size=size)

    @classmethod
    def from_checkpoint(cls, path, device=None, size=128):
        import torch
        from .render_lotus import get_device
        dev = device or get_device()
        ck = torch.load(path, map_location=dev, weights_only=False)
        enc = build_bmode_encoder(embed_dim=ck.get("embed_dim", 128))
        enc.load_state_dict(ck["encoder_state"])
        return cls(enc, device=dev, size=ck.get("size", size))


def load_similarity(spec: str, device=None) -> Similarity:
    """Dispatch a similarity spec: ncc | ssim | ts:<torchscript.pt> | embed:<encoder_ckpt.pt>."""
    if spec in ("ncc", ""):
        return NCCSimilarity()
    if spec == "ssim":
        return SSIMSimilarity()
    if spec.startswith("ts:"):
        return EmbeddingSimilarity.from_torchscript(spec[3:], device=device)
    if spec.startswith("embed:"):
        return EmbeddingSimilarity.from_checkpoint(spec[6:], device=device)
    raise ValueError(f"unknown similarity spec {spec!r} (use ncc | ssim | ts:<path> | embed:<path>)")
