"""texture_eval.py: per-tissue texture metrics and the TSTR harness for SonoSPADE.

S-CycleGAN's own paper names the missing piece: a numeric metric for ultrasound synthesis quality.
This module supplies one PER TISSUE, which is the contribution. Everything works on float32 [0,1]
frames plus an aligned canonical label map (segmenter.align_label_to_image), so a metric can be run
inside each tissue region.

Metrics:
  * speckle_snr / speckle_snr_per_class : mean over standard deviation inside a region. Fully
    developed speckle has a characteristic SNR near 1.91, so closeness to the real per-class SNR is
    a texture-realism signal.
  * glcm_per_class : Haralick contrast and homogeneity per tissue (a compact NumPy GLCM, no skimage
    dependency), versus real.
  * intensity_w1_per_class : the existing texture_gan.intensity_w1 run inside each label region.
  * edge_iou (re-exported) : structure preservation between the physics input and the translation.
  * tstr_dice : train a segmenter on SYNTHETIC (image, label) pairs, test on REAL labeled frames.
    The headline metric: higher means the per-tissue texture transfers to a downstream task.
  * temporal_flicker : mean absolute difference across adjacent frames (P6 stretch), lower = steadier.
"""
from __future__ import annotations

import numpy as np

from .texture_gan import edge_iou, intensity_w1
from .volume import N_LABELS, USBG_LABEL_NAMES

# Fully developed (Rayleigh) speckle has amplitude SNR = mean/std ~ 1.91; report closeness to this
# and to the real per-class SNR rather than an absolute target.
FULLY_DEVELOPED_SNR = 1.91


def speckle_snr(img01: np.ndarray, mask: np.ndarray | None = None, min_px: int = 32,
                min_std: float = 1e-3) -> float:
    """Amplitude speckle SNR = mean / standard deviation over a region (all pixels if mask is None).

    Returns NaN when the region is too small OR (nearly) constant. The constant guard matters: a flat
    nonzero region (padding, a saturated bright area, a near-anechoic fill) has std ~ 0, and without
    the guard mean / (std + eps) would return a huge FINITE value (for example 5e5), which np.nanmean
    would then silently fold into the per-class aggregate and destroy the realism signal. A region
    with essentially no variance carries no speckle, so NaN ("not measurable") is the correct answer;
    developed speckle has std well above min_std."""
    a = np.asarray(img01, np.float32)
    vals = a[mask] if mask is not None else a.ravel()
    if vals.size < min_px:
        return float("nan")
    sd = float(vals.std())
    if sd < min_std:                                     # (near) constant region: no speckle to measure
        return float("nan")
    return float(vals.mean()) / sd


def speckle_snr_per_class(img01: np.ndarray, label: np.ndarray, n_classes: int = N_LABELS,
                          min_px: int = 32) -> dict:
    """Per-class amplitude SNR: {class_id: snr} for classes with enough pixels in this frame."""
    lab = np.asarray(label)
    out = {}
    for c in range(n_classes):
        m = lab == c
        if int(m.sum()) >= min_px:
            snr = speckle_snr(img01, m, min_px=min_px)
            if not np.isnan(snr):                        # skip (near) constant regions, no speckle
                out[c] = snr
    return out


def _glcm_contrast_homogeneity(img01, mask, levels=16, offsets=((0, 1), (1, 0))):
    """Compact gray-level co-occurrence over pixel pairs whose BOTH endpoints fall in `mask`.
    Returns (contrast, homogeneity), averaged over the offsets. NaN if too few pairs."""
    q = np.clip(np.round(np.asarray(img01, np.float32) * (levels - 1)), 0, levels - 1).astype(np.int64)
    m = np.asarray(mask, bool)
    P = np.zeros((levels, levels), np.float64)
    for dr, dc in offsets:
        h, w = q.shape
        a = q[:h - dr, :w - dc]
        b = q[dr:, dc:]
        ma = m[:h - dr, :w - dc]
        mb = m[dr:, dc:]
        valid = ma & mb
        if not valid.any():
            continue
        ii = a[valid]; jj = b[valid]
        np.add.at(P, (ii, jj), 1.0)
        np.add.at(P, (jj, ii), 1.0)          # symmetric GLCM
    total = P.sum()
    if total < 8:
        return float("nan"), float("nan")
    P /= total
    idx = np.arange(levels)
    di = (idx[:, None] - idx[None, :]) ** 2
    contrast = float((P * di).sum())
    homogeneity = float((P / (1.0 + di)).sum())
    return contrast, homogeneity


def glcm_per_class(img01: np.ndarray, label: np.ndarray, n_classes: int = N_LABELS,
                   levels: int = 16, min_px: int = 64) -> dict:
    """Per-class GLCM texture: {class_id: {"contrast":.., "homogeneity":..}}."""
    lab = np.asarray(label)
    out = {}
    for c in range(n_classes):
        m = lab == c
        if int(m.sum()) >= min_px:
            con, hom = _glcm_contrast_homogeneity(img01, m, levels=levels)
            out[c] = {"contrast": con, "homogeneity": hom}
    return out


def intensity_w1_per_class(fake_pool, fake_labels, real_pool, real_labels,
                           n_classes: int = N_LABELS, bins: int = 128, min_px: int = 200) -> dict:
    """Per-class intensity Wasserstein-1 between the fake and real pools: pool all pixels of a class
    across frames, then compare the two 1D intensity distributions. {class_id: w1}. Classes absent
    from either pool are skipped (unconstrained, flag them upstream)."""
    def gather(pool, labels, c):
        vals = []
        for img, lab in zip(pool, labels):
            m = np.asarray(lab) == c
            if m.any():
                vals.append(np.asarray(img, np.float32)[m])
        return np.concatenate(vals) if vals else np.empty(0, np.float32)

    out = {}
    for c in range(n_classes):
        fv = gather(fake_pool, fake_labels, c)
        rv = gather(real_pool, real_labels, c)
        if fv.size >= min_px and rv.size >= min_px:
            out[c] = intensity_w1(fv, rv, bins=bins)
    return out


def label_swap_sensitivity(translate_aligned, phys_img01, aligned_label, n_classes: int = N_LABELS,
                           ref_class: int = 1, min_px: int = 200):
    """Quantify that texture is genuinely PER-TISSUE, not a global recolor (the handoff's one trap).

    Holding the physics render fixed, relabel each tissue region to a reference tissue and measure
    how much the translated texture changes INSIDE that region (it should change: the label controls
    its texture) versus OUTSIDE it (it should barely change: the effect is local). A global-recolor
    translator, or one that ignores the label, scores near zero in-region. A true per-tissue
    translator scores high in-region and low out-region, so the locality ratio is large.

    translate_aligned : fn(img01, aligned_label_HxW) -> img01 (e.g. SonoSPADETranslator.translate_aligned).
    Returns {class_id: {"in_region", "out_region", "locality"}} over classes with enough pixels.
    """
    import numpy as _np
    img = _np.asarray(phys_img01, _np.float32)
    lab = _np.asarray(aligned_label)
    base = _np.asarray(translate_aligned(img, lab), _np.float32)
    out = {}
    for c in range(n_classes):
        m = lab == c
        if c == ref_class or int(m.sum()) < min_px or not (~m).any():
            continue
        swapped = lab.copy()
        swapped[m] = ref_class                              # this region now claims the reference tissue
        alt = _np.asarray(translate_aligned(img, swapped), _np.float32)
        diff = _np.abs(alt - base)
        in_region = float(diff[m].mean())
        out_region = float(diff[~m].mean())
        out[c] = {"in_region": in_region, "out_region": out_region,
                  "locality": in_region / (out_region + 1e-6)}
    return out


def temporal_flicker(frames) -> float:
    """Mean absolute intensity difference between adjacent frames (P6 stretch). Lower is steadier;
    a translator that flickers frame to frame scores high. `frames` is an ordered (T,H,W) sequence."""
    f = np.asarray(frames, np.float32)
    if f.ndim != 3 or f.shape[0] < 2:
        return float("nan")
    return float(np.abs(f[1:] - f[:-1]).mean())


def summarize_per_class(metric_dicts, reducer=np.nanmean) -> dict:
    """Reduce a list of per-frame {class: value} dicts into {class: reduced_value} across frames."""
    acc: dict[int, list] = {}
    for d in metric_dicts:
        for c, v in d.items():
            acc.setdefault(c, []).append(v)
    return {c: float(reducer(vs)) for c, vs in acc.items()}


def class_name(c: int) -> str:
    return USBG_LABEL_NAMES.get(int(c), str(int(c)))


# ======================================================================================
# TSTR: train a segmenter on synthetic frames, test on real labeled frames (the headline metric)
# ======================================================================================
def tstr_dice(train_imgs, train_labels, test_imgs, test_labels, device=None,
              n_classes: int = N_LABELS, base: int = 24, depth: int = 4, iters: int = 800,
              batch: int = 8, lr: float = 5e-4, seed: int = 0, ignore_background: bool = True):
    """Train S_synth on (train_imgs, train_labels) then evaluate Dice on (test_imgs, test_labels).

    Images are float32 [0,1] (N,H,W), labels int (N,H,W) aligned to the images. Higher Dice means
    the synthetic per-tissue texture is realistic enough for a segmenter to transfer to real US.
    Returns (mean_dice, per_class_dice_dict)."""
    import torch
    from .segmenter import build_segmenter, seg_ce_dice_loss
    from .render_lotus import get_device
    dev = device or get_device()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    Xtr = np.asarray(train_imgs, np.float32)
    Ytr = np.asarray(train_labels, np.int64)
    net = build_segmenter(n_classes, base=base, depth=depth).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for _ in range(iters):
        bi = rng.integers(0, len(Xtr), size=min(batch, len(Xtr)))
        x = torch.from_numpy(Xtr[bi][:, None]).to(dev)
        y = torch.from_numpy(Ytr[bi]).to(dev)
        opt.zero_grad()
        loss = seg_ce_dice_loss(net(x), y, n_classes=n_classes)
        loss.backward(); opt.step()

    net.eval()
    Xte = np.asarray(test_imgs, np.float32)
    Yte = np.asarray(test_labels, np.int64)
    inter = np.zeros(n_classes); psum = np.zeros(n_classes); tsum = np.zeros(n_classes)
    with torch.no_grad():
        for j in range(0, len(Xte), batch):
            x = torch.from_numpy(Xte[j:j + batch][:, None]).to(dev)
            pred = net(x).argmax(1).cpu().numpy()
            tgt = Yte[j:j + batch]
            for c in range(n_classes):
                p = pred == c; t = tgt == c
                inter[c] += np.logical_and(p, t).sum()
                psum[c] += p.sum(); tsum[c] += t.sum()
    per_class = {}
    start = 1 if ignore_background else 0
    for c in range(start, n_classes):
        if tsum[c] > 0:
            per_class[c] = float((2 * inter[c] + 1e-6) / (psum[c] + tsum[c] + 1e-6))
    mean_dice = float(np.mean(list(per_class.values()))) if per_class else float("nan")
    return mean_dice, per_class


__all__ = ["FULLY_DEVELOPED_SNR", "speckle_snr", "speckle_snr_per_class", "glcm_per_class",
           "intensity_w1_per_class", "label_swap_sensitivity", "temporal_flicker",
           "summarize_per_class", "class_name", "tstr_dice", "edge_iou", "intensity_w1"]
