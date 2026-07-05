"""placement.py: put the probe where it will actually see the target anatomy.

Phase 1/2 use these to aim the imaging plane at an organ so slices and renders are meaningful.
Phase 3 (mujoco_env) reuses target_centroid_m to seat the CT under the MuJoCo probe surface.
"""
from __future__ import annotations

import numpy as np

from .contracts_bridge import pose_from_rpy


def target_centroid_m(lv, label: int) -> np.ndarray:
    """World (volume-frame) centroid in metres of a canonical label, via the volume affine."""
    ijk = np.argwhere(lv.labels == label)
    if len(ijk) == 0:
        from .volume import USBG_LABEL_NAMES
        raise ValueError(f"label {label} ({USBG_LABEL_NAMES.get(label, label)}) absent in volume")
    c_vox = ijk.mean(axis=0)
    return (lv.affine_m[:3, :3] @ c_vox) + lv.affine_m[:3, 3]


def base_pose_over(lv, label: int, depth_m: float = 0.03) -> np.ndarray:
    """A probe pose whose central scanline passes through the target at ~depth_m.

    With world_from_volume identity (volume frame == world), the probe origin sits at the
    target centroid shifted back along the axial (+z) axis by depth_m, so scanlines run +z
    into the organ and it appears at roughly depth_m.
    """
    c = target_centroid_m(lv, label)
    return pose_from_rpy(tx=float(c[0]), ty=float(c[1]), tz=float(c[2] - depth_m))


def seat_volume_under_probe(lv, label: int, standoff_m: float = 0.005,
                            depth_m: float = 0.05):
    """Place the CT under a MuJoCo skin surface at z=0 so the probe images the target.

    Returns (world_from_volume, T0_world_from_probe). The probe starts at (0, 0, standoff_m)
    above the skin, oriented so its imaging axis (+z probe) points DOWN into the patient
    (-z world), and world_from_volume (axis-aligned, translation only) is solved so the target
    centroid sits at depth_m along that central scanline. Axis-aligned start; refine later.
    """
    # probe looking down into the tissue: rotate 180 deg about x so +z_probe -> -z_world
    T0 = pose_from_rpy(tx=0.0, ty=0.0, tz=standoff_m, rx=180.0)
    # central scanline point at depth_m in the world:  T0 @ (0,0,depth_m)
    p_world = (T0[:3, :3] @ np.array([0.0, 0.0, depth_m])) + T0[:3, 3]
    centroid = target_centroid_m(lv, label)
    world_from_volume = np.eye(4, dtype=np.float64)          # identity rotation (axis-aligned)
    world_from_volume[:3, 3] = p_world - centroid            # translation only
    return world_from_volume, T0
