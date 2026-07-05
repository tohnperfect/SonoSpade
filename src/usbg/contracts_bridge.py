"""contracts_bridge.py: the single import surface for the cross-repo contract.

us_bmode_gym never redefines channels, classes, timing, or SE(3) math. It imports them
from the sibling `par` package (probe_action_recognizer) and re-exports them here so the
rest of usbg has one place to look. This module also:

  1. Guards against se3 copy-drift by calling par.contracts.assert_geometry_parity().
  2. Defines the two frame transforms that are usbg's own responsibility: placing the CT
     label volume in the MuJoCo world (world_from_volume) and mapping a MuJoCo probe pose
     into the volume frame the slicer samples in (volume_from_probe).

Frame conventions (kept identical to scripts/us_render.py in the recognizer repo):
  probe frame  : origin at the transducer face centre; imaging plane is probe x (lateral)
                 by z (axial), with +z pointing INTO the tissue (depth). +Z along the beam.
  volume frame : axes aligned with the CT voxel grid; origin at the world coordinate of
                 voxel (0, 0, 0); units in METRES. volume.py returns (spacing_m, origin_m)
                 consistent with this, so (pts_m - origin_m) / spacing_m gives voxel indices.
  world frame  : the MuJoCo world, metres. world_from_volume seats the CT under the probe.

Sampling a slice at MuJoCo probe pose T_world_from_probe:
  T_volume_from_probe = volume_from_probe(T_world_from_probe, T_world_from_volume)
  then slicer transforms probe-frame scan points by T_volume_from_probe and looks up labels.
"""
from __future__ import annotations

import numpy as np

# re-export the contract verbatim from par (do not redefine any of these)
from ._vendor import contracts as _C
from ._vendor.contracts import (  # noqa: F401  (re-exported for downstream import convenience)
    CHANNELS_TWIST,
    CHANNELS_REL6,
    CHANNELS_INPUT,
    IN_CHANNELS,
    TARGET_CHANNELS,
    CLASSES_5,
    SESSION_CLASSES_6,
    N_CLASSES,
    HOLD_IDX,
    DT_C,
    OUT_RATE_HZ,
    DOMINANT_AXIS,
    GEOMETRY_SHA256,
    CLIP_SIGMA,
    NEXT_ID_NONE,
    TTNB_UNITS,
    LATENT_D,
    NormStats,
    class_names,
)

# re-export the exact SE(3) geometry from par.se3
from ._vendor import se3 as _se3
from ._vendor.se3 import (  # noqa: F401
    so3_log,
    so3_exp,
    se3_log,
    se3_exp,
    se3_interpolate,
    invert_T,
)

__all__ = [
    # contract constants
    "CHANNELS_TWIST", "CHANNELS_REL6", "CHANNELS_INPUT", "IN_CHANNELS", "TARGET_CHANNELS",
    "CLASSES_5", "SESSION_CLASSES_6", "N_CLASSES", "HOLD_IDX", "DT_C", "OUT_RATE_HZ",
    "DOMINANT_AXIS", "GEOMETRY_SHA256", "CLIP_SIGMA", "NEXT_ID_NONE", "TTNB_UNITS",
    "LATENT_D", "NormStats", "class_names",
    # se3
    "so3_log", "so3_exp", "se3_log", "se3_exp", "se3_interpolate", "invert_T",
    # usbg frame transforms
    "assert_geometry_parity", "pose_from_rpy", "make_world_from_volume", "volume_from_probe",
    "summary",
]


def assert_geometry_parity() -> None:
    """Fail loudly if the vendored se3.py drifted from the pinned GEOMETRY_SHA256.

    Delegates to par.contracts, which hashes par/se3.py and compares to GEOMETRY_SHA256.
    The sim corpus stores this same hash in its meta, so parity here means the poses we
    render are in the exact geometry the recognizer and the demonstrator were built on.
    """
    _C.assert_geometry_parity()


def pose_from_rpy(tx=0.0, ty=0.0, tz=0.0, rx=0.0, ry=0.0, rz=0.0) -> np.ndarray:
    """Build a 4x4 SE(3) pose from a translation (metres) and roll/pitch/yaw (degrees).

    Rotation order is Rz @ Ry @ Rx, matching scripts/us_render.py::pose exactly so the
    slicer geometry we reuse stays parity-correct. Returns float64 (4, 4).
    """
    rx, ry, rz = np.deg2rad([rx, ry, rz])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = [tx, ty, tz]
    return T


def make_world_from_volume(t=(0.0, 0.0, 0.0), rpy_deg=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Seat the CT volume frame in the MuJoCo world. Returns 4x4 world_from_volume (metres).

    Phase 0/1 start axis-aligned and coincident (identity by default). Phase 3 refines this
    so the probe on the MuJoCo body surface samples plausible anatomy (translate the CT so
    its skin surface meets the probe face, rotate to the desired imaging orientation).
    """
    return pose_from_rpy(tx=t[0], ty=t[1], tz=t[2],
                         rx=rpy_deg[0], ry=rpy_deg[1], rz=rpy_deg[2])


def volume_from_probe(T_world_from_probe: np.ndarray,
                      T_world_from_volume: np.ndarray) -> np.ndarray:
    """Map a MuJoCo probe pose into the volume frame the slicer samples in.

    T_volume_from_probe = inv(T_world_from_volume) @ T_world_from_probe, using the exact
    SE(3) inverse from par.se3. The slicer then transforms probe-frame scan points by this
    matrix to get volume-frame metres, and (pts - origin) / spacing gives voxel indices.
    """
    T_wp = np.asarray(T_world_from_probe, dtype=np.float64)
    T_wv = np.asarray(T_world_from_volume, dtype=np.float64)
    return invert_T(T_wv) @ T_wp


def summary() -> dict:
    """Return (and, when run as a script, print) the reused contract values (Phase 0 accept)."""
    assert_geometry_parity()
    info = {
        "CHANNELS_TWIST": list(CHANNELS_TWIST),
        "CHANNELS_REL6": list(CHANNELS_REL6),
        "CHANNELS_INPUT": list(CHANNELS_INPUT),
        "IN_CHANNELS": IN_CHANNELS,
        "TARGET_CHANNELS": TARGET_CHANNELS,
        "CLASSES_5": list(CLASSES_5),
        "SESSION_CLASSES_6": list(SESSION_CLASSES_6),
        "N_CLASSES": N_CLASSES,
        "HOLD_IDX": HOLD_IDX,
        "DT_C": DT_C,
        "OUT_RATE_HZ": OUT_RATE_HZ,
        "DOMINANT_AXIS": dict(DOMINANT_AXIS),
        "GEOMETRY_SHA256": GEOMETRY_SHA256,
    }
    return info


def _print_summary() -> None:
    info = summary()
    print("usbg.contracts_bridge: reused par contract (geometry parity OK)")
    print(f"  channels (input, {info['IN_CHANNELS']}): {info['CHANNELS_INPUT']}")
    print(f"  twist target ({info['TARGET_CHANNELS']}): {info['CHANNELS_TWIST']}")
    print(f"  classes (5 train): {info['CLASSES_5']}")
    print(f"  classes (6 session): {info['SESSION_CLASSES_6']}  (hold idx {info['HOLD_IDX']})")
    print(f"  timing: DT_C={info['DT_C']} s ({info['OUT_RATE_HZ']} Hz)")
    print(f"  dominant axis per action: {info['DOMINANT_AXIS']}")
    print(f"  se3 geometry sha256: {info['GEOMETRY_SHA256'][:16]}... (parity asserted)")


if __name__ == "__main__":
    _print_summary()
