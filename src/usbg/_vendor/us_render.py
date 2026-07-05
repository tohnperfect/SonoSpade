#!/usr/bin/env python3
"""us_render.py -- minimal pose-conditioned B-mode ultrasound renderer (Mac-native, NumPy only).

WHY THIS FILE
  You already simulate the probe's 6-DOF motion in MuJoCo and compute poses T_k in SE(3).
  The one missing modality for a downstream VLA is the B-mode IMAGE the probe would see at
  each pose. This is the smallest thing that adds that modality and runs today on a MacBook
  (Apple Silicon, no CUDA): a convolutional ray model (Salehi 2015 / LOTUS-style) that slices
  a labeled 3D volume along the probe's imaging plane and applies a simple acoustic appearance
  model (boundary reflection + depth attenuation + speckle + beam blur + log compression).

  Pure NumPy so it runs anywhere. On your Mac you can later move the hot loop to torch with
  device="mps" for speed; the math is identical.

CONVENTION (match this to your contracts.py probe frame; it is a one-line change below)
  Input pose T is a 4x4 world_from_probe matrix. The imaging plane is the probe-local
  x (lateral) - z (axial) plane, with +z pointing INTO the tissue (depth). A curvilinear
  (convex) probe fans scanlines in that plane; set PROBE="linear" for a linear array.

  Feed T_k straight from your sim. Rendering along a maneuver trajectory (slide/rock/fan/
  rotate/compress) yields a B-mode video, i.e. the (image, action) pairs a VLA needs.

USAGE
  python scripts/us_render.py            # renders a phantom at 3 poses + a fan sweep -> PNGs
"""
from __future__ import annotations

import os
import numpy as np

# ----------------------------------------------------------------------------- geometry
IMG_DEPTH_M = 0.12          # axial depth imaged (m)
IMG_WIDTH_M = 0.08          # lateral extent at the face (m), used for linear probe
N_LINES = 160               # scanlines (lateral samples)
N_SAMPLES = 512             # samples per scanline (axial)
PROBE = "convex"            # "convex" (curvilinear) or "linear"
FOV_DEG = 60.0              # convex field of view
APEX_M = 0.04               # convex virtual apex behind the face (m)
FREQ_ATTEN = 0.9            # lumped attenuation strength (higher = darker with depth)
SPECKLE = 0.6               # multiplicative speckle strength
BEAM_LAT_MM = 1.4           # lateral beam width (PSF), mm
PULSE_AX_MM = 0.7           # axial pulse length (PSF), mm

# ------------------------------------------------------------------- tissue acoustic table
# label -> (impedance [MRayl-ish, arbitrary units], attenuation, backscatter amplitude)
TISSUES = {
    0: ("water/gel",   1.48, 0.02, 0.02),   # anechoic coupling medium
    1: ("soft tissue", 1.63, 0.60, 0.55),   # liver-like parenchyma (speckly mid-grey)
    2: ("vessel lumen",1.53, 0.05, 0.05),   # hypo/anechoic tube (blood)
    3: ("bone/gas",    6.00, 3.00, 0.90),   # very bright interface + strong shadow
    4: ("cyst",        1.50, 0.03, 0.03),   # anechoic sphere
}
LABELS = sorted(TISSUES)
Z_OF = np.array([TISSUES[k][1] for k in LABELS], dtype=np.float32)     # impedance
A_OF = np.array([TISSUES[k][2] for k in LABELS], dtype=np.float32)     # attenuation
S_OF = np.array([TISSUES[k][3] for k in LABELS], dtype=np.float32)     # backscatter


def make_phantom(n=220, spacing=0.5e-3):
    """A labeled 3D volume (analytic, no data download). Returns (vol, spacing, origin)."""
    vol = np.ones((n, n, n), dtype=np.uint8)  # 1 = soft tissue everywhere
    ax = (np.arange(n) - n / 2) * spacing
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    # a vessel: cylinder along Y at (x=+1cm, z=6cm)
    vessel = ((X - 0.010) ** 2 + (Z - 0.060) ** 2) < (0.008 ** 2)
    vol[vessel] = 2
    # a cyst: sphere at (x=-1.5cm, z=4cm)
    cyst = ((X + 0.015) ** 2 + Y ** 2 + (Z - 0.040) ** 2) < (0.009 ** 2)
    vol[cyst] = 4
    # a bright bone/gas interface: slab at z ~ 9cm
    bone = (np.abs(Z - 0.090) < 0.004)
    vol[bone] = 3
    origin = np.array([-n / 2 * spacing] * 3, dtype=np.float32)  # world coord of vol[0,0,0]
    return vol, np.float32(spacing), origin


def _sample_labels(vol, spacing, origin, pts_world):
    """Nearest-neighbour label lookup at world points (N,3). Out-of-volume -> water(0)."""
    idx = np.round((pts_world - origin) / spacing).astype(np.int64)   # (N,3)
    n = vol.shape[0]
    inb = np.all((idx >= 0) & (idx < n), axis=1)
    out = np.zeros(len(pts_world), dtype=np.uint8)                    # 0 = water outside
    ii = idx[inb]
    out[inb] = vol[ii[:, 0], ii[:, 1], ii[:, 2]]
    return out


def _scan_grid():
    """Sample points in the PROBE frame: returns (pts_probe [L,S,3], depth [S])."""
    depth = np.linspace(0.0, IMG_DEPTH_M, N_SAMPLES).astype(np.float32)
    if PROBE == "linear":
        xs = np.linspace(-IMG_WIDTH_M / 2, IMG_WIDTH_M / 2, N_LINES).astype(np.float32)
        X = np.repeat(xs[:, None], N_SAMPLES, axis=1)          # (L,S) lateral
        Zc = np.repeat(depth[None, :], N_LINES, axis=0)        # (L,S) axial
    else:  # convex: scanlines fan from a virtual apex behind the face
        angs = np.deg2rad(np.linspace(-FOV_DEG / 2, FOV_DEG / 2, N_LINES)).astype(np.float32)
        r = APEX_M + depth                                     # distance from apex
        X = np.sin(angs)[:, None] * r[None, :]
        Zc = np.cos(angs)[:, None] * r[None, :] - APEX_M
    Y = np.zeros_like(X)
    pts = np.stack([X, Y, Zc], axis=-1)                        # (L,S,3) in probe frame
    return pts, depth


def _gauss1d(sigma):
    k = max(1, int(round(sigma * 3)))
    x = np.arange(-k, k + 1, dtype=np.float32)
    g = np.exp(-0.5 * (x / max(sigma, 1e-3)) ** 2)
    return g / g.sum()


def _sepconv(img, gx, gz):
    """Separable convolution (lateral gx over axis0, axial gz over axis1)."""
    from numpy import convolve
    out = np.apply_along_axis(lambda m: convolve(m, gz, mode="same"), 1, img)
    out = np.apply_along_axis(lambda m: convolve(m, gx, mode="same"), 0, out)
    return out


def render_bmode(T_world_from_probe, vol, spacing, origin, rng=None):
    """Render one B-mode image (uint8, shape [N_SAMPLES, N_LINES]) for a probe pose."""
    rng = rng or np.random.default_rng(0)
    pts_probe, depth = _scan_grid()                            # (L,S,3), (S,)
    L, S = pts_probe.shape[:2]
    flat = pts_probe.reshape(-1, 3)
    world = (T_world_from_probe[:3, :3] @ flat.T).T + T_world_from_probe[:3, 3]
    lab = _sample_labels(vol, spacing, origin, world).reshape(L, S)
    Z = Z_OF[lab]; A = A_OF[lab]; Sc = S_OF[lab]               # (L,S) acoustic props

    # boundary reflection = normalized impedance gradient along depth
    dZ = np.abs(np.diff(Z, axis=1, prepend=Z[:, :1]))
    refl = dZ / (Z + 1e-3)
    # round-trip attenuation (cumulative along depth)
    dz = depth[1] - depth[0]
    atten = np.exp(-2.0 * FREQ_ATTEN * np.cumsum(A * dz * 100.0, axis=1))
    # backscatter speckle (multiplicative, Rayleigh-like)
    speck = Sc * np.abs(rng.normal(size=(L, S)).astype(np.float32))
    echo = (refl * 4.0 + speck * SPECKLE) * atten              # (L,S)

    # point spread: axial pulse + lateral beam
    gx = _gauss1d(BEAM_LAT_MM / (IMG_WIDTH_M * 1000 / L))
    gz = _gauss1d(PULSE_AX_MM / (IMG_DEPTH_M * 1000 / S))
    echo = _sepconv(echo, gx, gz)

    # log compression + 8-bit
    img = 20.0 * np.log10(echo + 1e-4)
    img = np.clip((img - img.min()) / (np.ptp(img) + 1e-6), 0, 1)
    img = (img ** 1.1 * 255).astype(np.uint8)
    return img.T                                               # (S,L): depth down, lateral across


def pose(tx=0.0, ty=0.0, tz=0.0, rx=0.0, ry=0.0, rz=0.0):
    """Build a 4x4 world_from_probe pose. Angles in degrees, translations in metres.
    Probe sits above the phantom (z into tissue); tune to your sim frame."""
    rx, ry, rz = np.deg2rad([rx, ry, rz])
    cx, sx = np.cos(rx), np.sin(rx); cy, sy = np.cos(ry), np.sin(ry); cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = (Rz @ Ry @ Rx).astype(np.float32)
    T[:3, 3] = [tx, ty, tz]
    return T


def main():
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs_us")
    os.makedirs(out, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vol, spacing, origin = make_phantom()
    rng = np.random.default_rng(1)

    # three poses that mimic maneuvers: straight, fanned (tilt about x), rotated in-plane
    poses = {"straight": pose(), "fan_15deg": pose(rx=15), "slide_x_+2cm": pose(tx=0.02)}
    fig, axes = plt.subplots(1, len(poses), figsize=(3.0 * len(poses), 3.4))
    for ax, (name, T) in zip(axes, poses.items()):
        img = render_bmode(T, vol, spacing, origin, rng)
        ax.imshow(img, cmap="gray", aspect="auto"); ax.set_title(name, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Pose-conditioned B-mode (synthetic phantom): vessel, cyst, bright interface", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out, "bmode_poses.png"), dpi=130); plt.close(fig)

    # a fan sweep = one maneuver -> a short B-mode "video" (the VLA training signal)
    fig, axes = plt.subplots(1, 6, figsize=(12, 2.6))
    for ax, ang in zip(axes, np.linspace(-20, 20, 6)):
        img = render_bmode(pose(rx=float(ang)), vol, spacing, origin, rng)
        ax.imshow(img, cmap="gray", aspect="auto"); ax.set_title(f"fan {ang:+.0f} deg", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(out, "bmode_fan_sweep.png"), dpi=130); plt.close(fig)
    print("wrote", os.path.join(out, "bmode_poses.png"), "and bmode_fan_sweep.png")


if __name__ == "__main__":
    main()
