"""slicer.py: pose T -> 2D label slice, reusing the recognizer's us_render geometry.

The probe imaging plane and its scan-point grid come from us_render._scan_grid (convex or
linear), imported (not copied) via usbg._reuse. Given a probe pose expressed in the volume
frame (T_volume_from_probe, from contracts_bridge.volume_from_probe), we transform the
probe-frame scan points into the volume frame, convert to voxel indices through the volume's
inverse affine (so anisotropic real CT and the axis-aligned phantom use one code path), and
look up nearest-neighbour labels.

Output orientation is (N_LINES, N_SAMPLES): rows are scanlines (lateral), columns are samples
(axial / depth). This is exactly what LOTUS's UltrasoundRendering.forward expects as its
ct_slice (W=lateral rows, H=depth columns), so Phase 2 feeds the slice with no transpose.
"""
from __future__ import annotations

import numpy as np

from ._reuse import us_render, N_LINES, N_SAMPLES
from .contracts_bridge import volume_from_probe


def scan_grid_probe():
    """Return (pts_probe [N_LINES, N_SAMPLES, 3] metres, depth [N_SAMPLES]) in the probe frame."""
    return us_render._scan_grid()


def _sample_labels_affine(labels: np.ndarray, inv_affine_m: np.ndarray,
                          pts_world_m: np.ndarray) -> np.ndarray:
    """Nearest-neighbour label lookup at volume-frame points (N,3) metres. Out-of-volume -> 0.

    Uses the linear (3,3) form of the inverse affine via einsum rather than a homogeneous
    (4,4) matmul. On Apple Silicon (numpy 2.x + Accelerate) the zero-heavy 4x4 matmul raises
    spurious overflow/divide-by-zero FP flags even though the result is finite and correct;
    the einsum path gives the identical result without tripping them.
    """
    n = pts_world_m.shape[0]
    lin = inv_affine_m[:3, :3]                                             # (3,3)
    t = inv_affine_m[:3, 3]                                                # (3,)
    vox = np.einsum("ij,nj->ni", lin, pts_world_m) + t                     # (N,3) voxel coords
    idx = np.round(vox).astype(np.int64)
    shape = np.asarray(labels.shape)
    inb = np.all((idx >= 0) & (idx < shape), axis=1)
    out = np.zeros(n, dtype=np.uint8)                                      # 0 = background outside
    ii = idx[inb]
    out[inb] = labels[ii[:, 0], ii[:, 1], ii[:, 2]]
    return out


def slice_labels(lv, T_volume_from_probe: np.ndarray) -> np.ndarray:
    """Sample the 2D label slice for a probe pose already expressed in the volume frame.

    lv                  : a volume.LabeledVolume (labels + affine_m).
    T_volume_from_probe : 4x4, probe frame -> volume frame (metres).
    Returns uint8 (N_LINES, N_SAMPLES): rows = scanlines (lateral), cols = samples (depth).
    """
    pts_probe, _depth = scan_grid_probe()                                  # (L,S,3), metres
    L, S = pts_probe.shape[:2]
    flat = pts_probe.reshape(-1, 3).astype(np.float64)
    T = np.asarray(T_volume_from_probe, dtype=np.float64)
    pts_vol = np.einsum("ij,nj->ni", T[:3, :3], flat) + T[:3, 3]           # (N,3) volume frame
    inv_affine = np.linalg.inv(np.asarray(lv.affine_m, dtype=np.float64))
    lab = _sample_labels_affine(lv.labels, inv_affine, pts_vol)
    return lab.reshape(L, S)


def slice_at_pose(lv, T_world_from_probe: np.ndarray,
                  T_world_from_volume: np.ndarray) -> np.ndarray:
    """Convenience: slice given a MuJoCo world probe pose and the CT placement in the world.

    Composes contracts_bridge.volume_from_probe then slice_labels. Returns (N_LINES, N_SAMPLES).
    """
    T_vp = volume_from_probe(T_world_from_probe, T_world_from_volume)
    return slice_labels(lv, T_vp)


# convenience re-exports
__all__ = ["scan_grid_probe", "slice_labels", "slice_at_pose", "N_LINES", "N_SAMPLES"]
