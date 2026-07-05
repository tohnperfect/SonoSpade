"""volume.py: build a canonical labeled anatomy volume for the slicer and renderers.

Input: a TotalSegmentator case (ct.nii.gz + segmentations/<structure>.nii.gz, one binary
mask per structure). Output: one integer label volume in usbg's canonical semantic scheme,
plus the voxel-to-world affine in METRES, cached to .npz so it is built once.

Two design points that differ from the handoff's starter, on purpose:

  1. Canonical semantic labels (not LOTUS's integers). Gallbladder gets its OWN label so the
     Phase 7 target-visibility reward and the dataset can find it, even though acoustically it
     renders anechoic (like a vessel lumen). The renderers map canonical labels through the
     acoustic tables below, so LOTUS's fixed 9-label dict is not a constraint.

  2. Body interior defaults to soft tissue (from Hounsfield units), then organ masks overlay,
     then bone overlays last (brightest). Without this, gaps between organs would render as
     water (anechoic), which is not what a real abdominal B-mode looks like. This mirrors
     us_render.make_phantom (soft tissue everywhere, structures overlaid).

The acoustic tables (Z, attenuation, mu_0, mu_1, sigma_0) are seeded from LOTUS's published
defaults (models/us_rendering_model.py) mapped onto the canonical labels, so Phase 2 renders
directly from a canonical slice with no relabeling.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# ======================================================================================
# Canonical semantic label scheme
# ======================================================================================
BACKGROUND = 0     # air / coupling gel / outside the body (anechoic)
LIVER = 1
GALLBLADDER = 2    # BiTNet biliary target; distinct label, anechoic acoustics
VESSEL = 3         # aorta, IVC, portal / splenic vein
KIDNEY = 4
GI = 5             # stomach, duodenum, small bowel, colon, esophagus
PANCREAS = 6
SPLEEN = 7
MUSCLE = 8         # autochthon, iliopsoas, gluteals, etc.
FAT = 9
BONE = 10          # vertebrae, ribs, hips, sacrum, sternum, etc.
LUNG = 11
SOFT_TISSUE = 12   # generic body interior (HU based), the default fill

USBG_LABEL_NAMES = {
    BACKGROUND: "background", LIVER: "liver", GALLBLADDER: "gallbladder", VESSEL: "vessel",
    KIDNEY: "kidney", GI: "gi", PANCREAS: "pancreas", SPLEEN: "spleen", MUSCLE: "muscle",
    FAT: "fat", BONE: "bone", LUNG: "lung", SOFT_TISSUE: "soft_tissue",
}
N_LABELS = len(USBG_LABEL_NAMES)

# TotalSegmentator v2 structure stem -> canonical label. Missing structures are skipped, so
# this is safe across body regions. Order does not matter here (overlay order is fixed below).
TOTALSEG_MAP: dict[str, int] = {
    "liver": LIVER,
    "gallbladder": GALLBLADDER,
    # vessels
    "aorta": VESSEL, "inferior_vena_cava": VESSEL,
    "portal_vein_and_splenic_vein": VESSEL,
    "pulmonary_vein": VESSEL, "iliac_artery_left": VESSEL, "iliac_artery_right": VESSEL,
    "iliac_vena_left": VESSEL, "iliac_vena_right": VESSEL,
    # kidneys / urinary
    "kidney_left": KIDNEY, "kidney_right": KIDNEY, "urinary_bladder": KIDNEY,
    # GI tract
    "stomach": GI, "duodenum": GI, "small_bowel": GI, "colon": GI, "esophagus": GI,
    # solid organs
    "pancreas": PANCREAS, "spleen": SPLEEN,
    # muscle
    "autochthon_left": MUSCLE, "autochthon_right": MUSCLE,
    "iliopsoas_left": MUSCLE, "iliopsoas_right": MUSCLE,
    "gluteus_maximus_left": MUSCLE, "gluteus_maximus_right": MUSCLE,
    "gluteus_medius_left": MUSCLE, "gluteus_medius_right": MUSCLE,
    "gluteus_minimus_left": MUSCLE, "gluteus_minimus_right": MUSCLE,
    # lungs
    "lung_upper_lobe_left": LUNG, "lung_lower_lobe_left": LUNG,
    "lung_upper_lobe_right": LUNG, "lung_middle_lobe_right": LUNG, "lung_lower_lobe_right": LUNG,
}
# Any structure stem starting with one of these prefixes is bone (vertebrae_*, rib_*, ...).
BONE_PREFIXES = ("vertebrae", "rib", "hip", "sacrum", "femur", "humerus",
                 "scapula", "clavicula", "sternum", "costal_cartilages", "skull")

# Fixed overlay order: soft tissue (HU) first, organs next, bone last (bone wins).
_ORGAN_OVERLAY = [LIVER, SPLEEN, PANCREAS, KIDNEY, GI, VESSEL, GALLBLADDER, LUNG, MUSCLE]

# ======================================================================================
# Acoustic tables (seeded from LOTUS defaults), canonical label -> params
#   Z: acoustic impedance (MRayl), att: attenuation (dB cm^-1 at 1 MHz),
#   mu_0: mean scatterer brightness, mu_1: scatterer density, sigma_0: brightness std.
# LOTUS reference (models/us_rendering_model.py): lung .0004/1.64, fat 1.38/0.63,
#   vessel 1.61/0.18, kidney 1.62/1.0, muscle 1.62/1.09, background 0.3/0.54, liver 1.65/0.7,
#   soft tissue 1.63/0.54, bone 7.8/5.0.
# ======================================================================================
ACOUSTIC_LOTUS: dict[int, tuple] = {
    #            Z       att    mu_0   mu_1   sigma_0
    BACKGROUND: (0.0004, 1.64,  0.30,  0.20,  0.00),   # air / gel (near-anechoic)
    LIVER:      (1.65,   0.70,  0.40,  0.80,  0.14),
    GALLBLADDER:(1.53,   0.10,  0.02,  0.05,  0.01),   # anechoic fluid sac
    VESSEL:     (1.61,   0.18,  0.001, 0.00,  0.01),   # anechoic lumen
    KIDNEY:     (1.62,   1.00,  0.45,  0.60,  0.30),
    GI:         (1.60,   0.70,  0.45,  0.60,  0.20),
    PANCREAS:   (1.62,   0.70,  0.45,  0.55,  0.20),
    SPLEEN:     (1.62,   0.80,  0.42,  0.62,  0.18),
    MUSCLE:     (1.62,   1.09,  0.45,  0.64,  0.10),
    FAT:        (1.38,   0.63,  0.50,  0.50,  0.00),
    BONE:       (7.80,   5.00,  0.78,  0.56,  0.10),   # bright interface + shadow
    LUNG:       (0.0004, 1.64,  0.78,  0.56,  0.10),   # air-filled, bright speckle + shadow
    SOFT_TISSUE:(1.63,   0.54,  0.45,  0.64,  0.10),
}

# Canonical label -> us_render.py TISSUES scheme (0 water, 1 soft, 2 vessel, 3 bone/gas, 4 cyst)
# for the NumPy fallback renderer.
USBG_TO_USRENDER: dict[int, int] = {
    BACKGROUND: 0, LIVER: 1, GALLBLADDER: 4, VESSEL: 2, KIDNEY: 1, GI: 1, PANCREAS: 1,
    SPLEEN: 1, MUSCLE: 1, FAT: 1, BONE: 3, LUNG: 3, SOFT_TISSUE: 1,
}

# Hounsfield thresholds for the generic (non-mask) fill.
HU_AIR = -200.0     # below this, outside body / air -> background
HU_FAT = -30.0      # [-200, -30) -> fat
HU_BONE = 300.0     # above this -> bone (if not already an organ)


@dataclass
class LabeledVolume:
    """A canonical labeled volume ready for the slicer.

    labels    : (I, J, K) uint8 canonical labels
    affine_m  : (4, 4) voxel -> world (V frame) in METRES
    spacing_m : (3,) metres per voxel along each axis (abs of affine diagonal)
    origin_m  : (3,) world coord (metres) of voxel (0, 0, 0)
    meta      : provenance dict (case id, source, label counts)
    """
    labels: np.ndarray
    affine_m: np.ndarray
    spacing_m: np.ndarray
    origin_m: np.ndarray
    meta: dict

    def label_counts(self) -> dict:
        u, c = np.unique(self.labels, return_counts=True)
        return {USBG_LABEL_NAMES.get(int(k), str(int(k))): int(v) for k, v in zip(u, c)}

    def has(self, label: int) -> bool:
        return bool(np.any(self.labels == label))


def _affine_to_meters(affine_mm: np.ndarray) -> np.ndarray:
    """nibabel affine is voxel -> world in mm; scale the linear+translation parts to metres."""
    a = np.asarray(affine_mm, dtype=np.float64).copy()
    a[:3, :] = a[:3, :] / 1000.0
    return a


def load_labeled_ct(ct_path: str, seg_dir: str) -> LabeledVolume:
    """Merge a TotalSegmentator case into one canonical label volume.

    ct_path : path to ct.nii.gz. seg_dir : dir of per-structure binary masks (<stem>.nii.gz).
    Reorients to closest canonical (RAS) so the affine is near axis-aligned, then fills soft
    tissue / fat / air by HU, overlays organ masks, overlays bone last.
    """
    import nibabel as nib

    ct = nib.as_closest_canonical(nib.load(ct_path))
    hu = np.asanyarray(ct.dataobj).astype(np.float32)
    affine_mm = ct.affine
    shape = hu.shape

    labels = np.full(shape, SOFT_TISSUE, dtype=np.uint8)
    labels[hu < HU_AIR] = BACKGROUND
    labels[(hu >= HU_AIR) & (hu < HU_FAT)] = FAT

    def _load_mask(stem: str) -> np.ndarray | None:
        p = os.path.join(seg_dir, f"{stem}.nii.gz")
        if not os.path.exists(p):
            return None
        m = nib.as_closest_canonical(nib.load(p))
        return np.asanyarray(m.dataobj) > 0.5

    # organ overlays (fixed order; later wins within organs)
    present = []
    by_label: dict[int, np.ndarray] = {}
    for stem, lab in TOTALSEG_MAP.items():
        mask = _load_mask(stem)
        if mask is None or not mask.any():
            continue
        by_label.setdefault(lab, np.zeros(shape, dtype=bool))
        by_label[lab] |= mask
        present.append(stem)
    for lab in _ORGAN_OVERLAY:
        if lab in by_label:
            labels[by_label[lab]] = lab

    # bone overlays last: explicit bone masks OR high HU inside the body
    bone_mask = np.zeros(shape, dtype=bool)
    if os.path.isdir(seg_dir):
        for fn in os.listdir(seg_dir):
            stem = fn[:-7] if fn.endswith(".nii.gz") else os.path.splitext(fn)[0]
            if stem.startswith(BONE_PREFIXES):
                m = _load_mask(stem)
                if m is not None:
                    bone_mask |= m
    bone_mask |= (hu >= HU_BONE) & (labels != BACKGROUND)
    labels[bone_mask] = BONE

    affine_m = _affine_to_meters(affine_mm)
    spacing_m = np.abs(np.diag(affine_m)[:3]).astype(np.float64)
    origin_m = affine_m[:3, 3].astype(np.float64)
    meta = {
        "source": "totalsegmentator",
        "ct_path": os.path.abspath(ct_path),
        "structures_present": sorted(present),
        "shape": list(shape),
    }
    lv = LabeledVolume(labels=labels, affine_m=affine_m, spacing_m=spacing_m,
                       origin_m=origin_m, meta=meta)
    meta["label_counts"] = lv.label_counts()
    return lv


def save_cache(lv: LabeledVolume, path: str) -> str:
    """Cache a built label volume to .npz so it is built once."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    import json
    np.savez_compressed(path, labels=lv.labels, affine_m=lv.affine_m,
                        spacing_m=lv.spacing_m, origin_m=lv.origin_m,
                        meta=json.dumps(lv.meta))
    return path


def load_cache(path: str) -> LabeledVolume:
    import json
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"])) if "meta" in d else {}
    return LabeledVolume(labels=d["labels"], affine_m=d["affine_m"],
                         spacing_m=d["spacing_m"], origin_m=d["origin_m"], meta=meta)


def synthetic_phantom() -> LabeledVolume:
    """A labeled phantom (no data download) in the same LabeledVolume shape, for tests and as
    a rendering sanity baseline. Reuses us_render.make_phantom, remapped to canonical labels."""
    from ._reuse import us_render
    vol, spacing, origin = us_render.make_phantom()   # labels {1 soft,2 vessel,3 bone,4 cyst}
    remap = {0: BACKGROUND, 1: SOFT_TISSUE, 2: VESSEL, 3: BONE, 4: GALLBLADDER}
    labels = np.zeros_like(vol, dtype=np.uint8)
    for src, dst in remap.items():
        labels[vol == src] = dst
    spacing_m = np.array([float(spacing)] * 3, dtype=np.float64)   # already metres
    origin_m = np.asarray(origin, dtype=np.float64)
    affine_m = np.eye(4, dtype=np.float64)
    affine_m[:3, :3] = np.diag(spacing_m)
    affine_m[:3, 3] = origin_m
    meta = {"source": "synthetic_phantom"}
    lv = LabeledVolume(labels=labels, affine_m=affine_m, spacing_m=spacing_m,
                       origin_m=origin_m, meta=meta)
    meta["label_counts"] = lv.label_counts()
    return lv
