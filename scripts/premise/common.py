"""common.py: shared scheme, data loading, preprocessing, and augmentation for the
per-tissue premise check (Section 4 of docs/HANDOFF_per_tissue_rl_task.md).

The premise: a REAL-trained organ recognizer J should recognize the correct organ in a
SonoSPADE-textured sim frame (per-tissue-correct appearance) but fail under a global recolor
(CUT / CycleGAN), which scrambles per-tissue statistics. J is trained ONLY on real Kaggle
Vitale frames+masks -- never on sim / SonoSPADE output -- so the test is non-circular.

Both the real Kaggle labels and the sim (pairs_med) labels use usbg's canonical 13-class
scheme (volume.py). J operates on a compact 6-class ORGAN scheme covering exactly the organs
that have real masks (liver, gallbladder, vessel, kidney, spleen) plus background. Everything
not one of those five organs -- soft tissue, muscle, fat, GI, bone, lung -- is background=0,
matching how the real Kaggle annotations were made (only the five organs are colored).
"""
from __future__ import annotations

import numpy as np

from usbg.volume import (BACKGROUND, LIVER, GALLBLADDER, VESSEL, KIDNEY, SPLEEN, N_LABELS,
                         USBG_LABEL_NAMES)

# ----------------------------------------------------------------------------------------
# J's organ scheme: canonical id -> J class. Only the five real-annotated organs get a class;
# every other canonical tissue collapses to background (0), exactly as the real masks encode it.
# ----------------------------------------------------------------------------------------
J_NAMES = {0: "bg", 1: "liver", 2: "gallbladder", 3: "vessel", 4: "kidney", 5: "spleen"}
N_J = len(J_NAMES)
ORGAN_JIDS = [1, 2, 3, 4, 5]                      # the organ classes (exclude bg)

# canonical (13-class) -> J (6-class)
_CAN2J = np.zeros(N_LABELS, dtype=np.int64)
for _can, _j in {LIVER: 1, GALLBLADDER: 2, VESSEL: 3, KIDNEY: 4, SPLEEN: 5}.items():
    _CAN2J[_can] = _j                             # BACKGROUND and all other tissues stay 0


def canonical_to_J(label_canonical: np.ndarray) -> np.ndarray:
    """Map a canonical (13-class) label map into J's 6-class organ scheme."""
    lab = np.asarray(label_canonical)
    return _CAN2J[np.clip(lab, 0, N_LABELS - 1)]


def jname(jid: int) -> str:
    return J_NAMES[int(jid)]


# ----------------------------------------------------------------------------------------
# data loading (uint8 [0,255] on disk -> float [0,1]); both datasets share this convention.
# ----------------------------------------------------------------------------------------
def crop_to_content(img: np.ndarray, lab: np.ndarray, thresh: float = 0.04, size: int = 128):
    """Crop a fan-beam real frame to its content bounding box and resize to (size,size).

    The Kaggle frames are curvilinear (sector) with a large black border (~65% of the frame);
    the linear sim renders have essentially none. J otherwise learns 'organs live inside a fan
    surrounded by black' and fails to fire on borderless sim frames. Cropping the border away at
    training time makes J's domain match the sim renders. Translator-agnostic: it is applied to
    J's real training data only, identically for every translator that is later scored."""
    from PIL import Image
    m = np.asarray(img) > thresh
    if not m.any():
        return img, lab
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    ic = np.asarray(img, np.float32)[y0:y1, x0:x1]
    lc = np.asarray(lab)[y0:y1, x0:x1]
    # pad the content bbox to a SQUARE (centered, zero border) BEFORE the resize, so the resize
    # never anisotropically stretches organs (a confound the reviewer flagged: a non-square bbox
    # resized straight to 128x128 distorts aspect ratio; padding to square preserves it).
    h, w = ic.shape
    s = max(h, w)
    ic_sq = np.zeros((s, s), np.float32); lc_sq = np.zeros((s, s), lc.dtype)
    y_off, x_off = (s - h) // 2, (s - w) // 2
    ic_sq[y_off:y_off + h, x_off:x_off + w] = ic
    lc_sq[y_off:y_off + h, x_off:x_off + w] = lc
    ic = np.asarray(Image.fromarray((ic_sq * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR),
                    np.float32) / 255.0
    lc = np.asarray(Image.fromarray(lc_sq.astype(np.uint8)).resize((size, size), Image.NEAREST), np.int64)
    return ic, lc


def load_real(path: str = "data/eval/kaggle_vitale_test.npz", crop_fan: bool = False):
    """Real Kaggle Vitale: images [0,1] (N,H,W) float32, J-scheme labels (N,H,W) int64, names.
    crop_fan=True removes the sector black border (crop-to-content) so J's domain matches the
    borderless linear sim renders it will be scored on."""
    d = np.load(path, allow_pickle=True)
    imgs = d["images"].astype(np.float32) / 255.0
    labs = canonical_to_J(d["labels"])
    if crop_fan:
        cropped = [crop_to_content(imgs[i], labs[i]) for i in range(len(imgs))]
        imgs = np.stack([c[0] for c in cropped]).astype(np.float32)
        labs = np.stack([c[1] for c in cropped]).astype(np.int64)
    names = d["sessions"] if "sessions" in d else np.arange(len(imgs))
    return imgs, labs, np.asarray(names)


def load_sim(path: str = "data/dataset/pairs_med.npz"):
    """Sim content (pairs_med): physics renders [0,1] (N,H,W) float32, CANONICAL labels
    (N,H,W) int64 (kept canonical, since SonoSPADE conditions on the full 13-class scheme),
    plus the same labels mapped into J's scheme for the true-organ-region metric."""
    d = np.load(path, allow_pickle=True)
    imgs = d["images"].astype(np.float32) / 255.0
    labs_canon = d["labels"].astype(np.int64)
    labs_J = canonical_to_J(labs_canon)
    return imgs, labs_canon, labs_J


# ----------------------------------------------------------------------------------------
# augmentation for training J. Geometric transforms apply to (image, label) together (label
# nearest); intensity transforms apply to the image only. The intensity jitter is deliberately
# STRONG so J cannot recognize an organ from absolute brightness -- it must use per-tissue
# texture / structure. This removes the confound that CycleGAN happens to match real's global
# darkness, and it costs SonoSPADE its brightness offset too, so it is the honest choice.
# ----------------------------------------------------------------------------------------
def augment(img: np.ndarray, lab: np.ndarray, rng: np.random.Generator,
            intensity: bool = True, geom: bool = True) -> tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import rotate, zoom
    im = np.asarray(img, np.float32).copy()
    lb = np.asarray(lab, np.int64).copy()
    h, w = im.shape

    if geom:
        if rng.random() < 0.5:                                  # horizontal flip
            im = im[:, ::-1].copy(); lb = lb[:, ::-1].copy()
        if rng.random() < 0.8:                                  # small rotation
            ang = float(rng.uniform(-15, 15))
            im = rotate(im, ang, reshape=False, order=1, mode="reflect")
            lb = rotate(lb, ang, reshape=False, order=0, mode="reflect")
        if rng.random() < 0.8:                                  # zoom + center crop/pad
            s = float(rng.uniform(0.6, 1.5))                    # wide: bridge real/sim organ scale
            im2 = zoom(im, s, order=1); lb2 = zoom(lb, s, order=0)
            im, lb = _center_fit(im2, h, w, 0.0), _center_fit(lb2, h, w, 0)
        if rng.random() < 0.8:                                  # small translation
            dy = int(rng.integers(-10, 11)); dx = int(rng.integers(-10, 11))
            im = np.roll(im, (dy, dx), (0, 1)); lb = np.roll(lb, (dy, dx), (0, 1))
        im = np.clip(im, 0, 1).astype(np.float32)

    if intensity:
        g = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))   # gamma in [0.5, 2.0]
        im = np.clip(im, 0, 1) ** g
        a = float(rng.uniform(0.6, 1.5)); b = float(rng.uniform(-0.15, 0.15))  # contrast/brightness
        im = np.clip(a * (im - 0.5) + 0.5 + b, 0, 1)
        if rng.random() < 0.5:                                  # additive speckle-ish noise
            im = np.clip(im + rng.normal(0, float(rng.uniform(0.01, 0.06)), im.shape), 0, 1)
        im = im.astype(np.float32)

    return im, lb.astype(np.int64)


def _center_fit(a: np.ndarray, h: int, w: int, pad_val):
    """Center-crop or center-pad a to (h, w)."""
    ah, aw = a.shape
    out = np.full((h, w), pad_val, dtype=a.dtype)
    y0s = max((ah - h) // 2, 0); x0s = max((aw - w) // 2, 0)
    y0d = max((h - ah) // 2, 0); x0d = max((w - aw) // 2, 0)
    ch = min(h, ah); cw = min(w, aw)
    out[y0d:y0d + ch, x0d:x0d + cw] = a[y0s:y0s + ch, x0s:x0s + cw]
    return out


# ----------------------------------------------------------------------------------------
# per-organ recognition metrics (used by both the premise eval and J validation)
# ----------------------------------------------------------------------------------------
def per_organ_scores(pred_J: np.ndarray, true_J: np.ndarray, min_pixels: int = 40) -> dict:
    """For each organ present in true_J with >= min_pixels, return recall (fraction of the organ's
    true pixels that J labels as that organ), precision (of the pixels J calls X, fraction truly X),
    IoU (pred==X vs true==X), and null_iou (the trivial 'predict X everywhere' IoU = |true X| /
    |frame|, a floor that a liver-flooding translator would hit). Organs absent/too small omitted.
    precision separates per-tissue specificity from mere volume: a translator that makes J flood a
    dominant organ gets high recall but low precision."""
    pred = np.asarray(pred_J); true = np.asarray(true_J)
    frame = true.size
    out = {}
    for jid in ORGAN_JIDS:
        t = (true == jid)
        nt = int(t.sum())
        if nt < min_pixels:
            continue
        p = (pred == jid)
        npred = int(p.sum())
        inter = int((p & t).sum())
        union = int((p | t).sum())
        out[jid] = {"recall": inter / nt, "precision": inter / max(npred, 1),
                    "iou": inter / max(union, 1), "n_true": nt,
                    "null_iou": nt / frame}
    return out
