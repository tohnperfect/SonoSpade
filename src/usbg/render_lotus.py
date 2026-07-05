"""render_lotus.py: label slice -> B-mode image, on MPS, with physics cues and a NumPy fallback.

Three renderers behind one dispatcher:
  render(label_slice, renderer="physics", cfg=RenderConfig) -> upgraded physics renderer (default)
  render(label_slice, renderer="lotus")  -> the original LOTUS-style reimplementation
  render(label_slice, renderer="numpy")  -> the recognizer's us_render appearance model (reused)

The "physics" renderer (this file's upgrade, spec: physics cues + domain randomization for
goal-conditioned RL) adds the view-discriminability cues a goal-image policy keys on, while
staying a PURE, DETERMINISTIC, differentiable, MPS/CPU function render(pose, volume, cfg, seed):

  * Acoustic shadowing (U1): cumulative transmission through interfaces (Fresnel reflection
    R = ((Z2-Z1)/(Z2+Z1))^2, cumprod of 1-R) times round-trip attenuation (exp of -2 cumsum mu dz).
    Behind bone / gas the envelope collapses toward zero, so shadows appear and move with the probe.
  * Depth-varying PSF, TGC, correlated speckle (U2): a beam that widens away from the focus (a few
    focal-zone bands, each with its own lateral Gaussian), fully developed speckle (envelope of the
    PSF convolved with a complex white field, so grain = resolution cell, not per-pixel noise), a
    time-gain curve that brightens the far field, and log compression.

LOTUS ships a differentiable convolutional ray renderer but hardcodes CUDA at import, so the
physics here is reimplemented device-agnostically from usbg's own acoustic tables
(volume.ACOUSTIC_LOTUS). All ops are torch ops; noise is drawn from a fixed CPU generator seeded by
cfg.seed so a given (pose, cfg, seed) is bit-reproducible on CPU (and stable on MPS within
tolerance), which the goal-conditioned reward depends on.

Input is a canonical label slice (N_LINES, N_SAMPLES) from slicer: rows are scanlines (lateral),
columns are samples (axial / depth). Output is float32 (N_SAMPLES, N_LINES) in [0,1] (depth down
rows, lateral across columns), or the warped curvilinear sector if warp=True.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ._reuse import us_render
from .volume import ACOUSTIC_LOTUS, USBG_TO_USRENDER

# LOTUS default hyperparameters (models/us_rendering_model.py), used by the "lotus" renderer
ALPHA_BOUNDARY = 0.1     # boundary sharpness in -exp(-dZ^2 / alpha) + 1
BETA_SCATTER = 10.0      # sigmoid sharpness in the differentiable scatter approximation
TGC_DEFAULT = 8.0        # time-gain compensation (brightens with depth)
DIST_SCALE = 0.006       # per-sample axial step for the attenuation exponent (tuned for depth)

# physics-renderer tuning constants (calibrated so bone/gas shadow strongly while the soft-tissue
# far field stays visible after TGC, and the intensity distribution moves toward real ultrasound)
ATTEN_COEF = 0.9         # attenuation strength (multiplies mu*dz in the round-trip exponent)
REFL_GAIN = 3.5          # specular-reflection brightness relative to scattering
TGC_REF_MU = 0.55        # reference attenuation the time-gain curve compensates (soft tissue),
                         # so a normal-tissue field is flat and bone's larger loss stays a shadow
GAMMA = 1.3              # display gamma on the log-compressed image (moderate brightness/contrast)
TOP_PCTILE = 0.99        # robust white level (percentile, not max) so a few bright specular pixels
                         # do not crush the rest of the image (content-stable brightness)
IMG_DEPTH_M = float(us_render.IMG_DEPTH_M)   # axial depth imaged (m)
_LABELS = tuple(sorted(ACOUSTIC_LOTUS))


def get_device(prefer: str = "mps") -> str:
    import torch
    if prefer == "mps" and torch.backends.mps.is_available():
        return "mps"
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ======================================================================================
# RenderConfig: every physics/appearance knob in one place (determinism + domain randomization)
# ======================================================================================
@dataclass(frozen=True)
class RenderConfig:
    """All renderer knobs in one frozen struct, so determinism, differentiability, and the
    same-pipeline rule are enforced in one place. Per-label multiplier dicts (keyed by canonical
    label id) default to 1.0 for any missing label. `translator` names the sim-to-real stage that
    the env/goal pipeline applies identically to target and observation.
    """
    z_scale: dict = field(default_factory=dict)      # per-label impedance multiplier
    mu_scale: dict = field(default_factory=dict)     # per-label attenuation multiplier
    bs_scale: dict = field(default_factory=dict)     # per-label backscatter (echogenicity) mult.
    gain: float = 1.0                                 # overall brightness
    tgc_slope: float = 1.0                            # depth brightness profile (time-gain)
    dyn_range_db: float = 50.0                        # log-compression window (dB)
    psf_sigma_ax: float = 1.0                         # axial PSF sigma (pixels)
    psf_sigma_lat0: float = 1.1                       # lateral PSF sigma at the focus (pixels)
    psf_focus_depth: float = 0.05                     # focal depth (m)
    focal_scale: float = 0.04                         # how fast the beam widens off-focus (m)
    speckle_strength: float = 0.9                     # texture contrast
    noise_sigma: float = 0.004                        # electronic noise (low floor -> dark shadows)
    shadow_gain: float = 1.0                          # strengthen/weaken reflection shadows
    translator: str = "none"                          # {none, cyclegan, cut, cyclegan_sc}
    seed: int = 0                                     # drives speckle + noise; deterministic

    @property
    def seed_family(self) -> int:
        """Identifier of the per-episode randomization draw (everything but the per-step seed)."""
        return hash((tuple(sorted(self.z_scale.items())), tuple(sorted(self.mu_scale.items())),
                     tuple(sorted(self.bs_scale.items())), self.gain, self.tgc_slope,
                     self.dyn_range_db, self.psf_sigma_ax, self.psf_sigma_lat0,
                     self.psf_focus_depth, self.focal_scale, self.speckle_strength,
                     self.noise_sigma, self.shadow_gain, self.translator))

    def with_seed(self, seed: int) -> "RenderConfig":
        return replace(self, seed=int(seed))


def default_config(**overrides) -> RenderConfig:
    return replace(RenderConfig(), **overrides)


def domain_randomize(base: RenderConfig, rng) -> RenderConfig:
    """Sample a per-episode config: acoustic params, gain, PSF, and noise vary while the anatomy
    (labels) does not, forcing structure over appearance. Sample ONCE per episode, hold fixed, and
    render target and observation with the SAME draw (the same-pipeline rule)."""
    u = rng.uniform
    return replace(
        base,
        z_scale={L: float(u(0.90, 1.10)) for L in _LABELS},
        mu_scale={L: float(u(0.70, 1.40)) for L in _LABELS},
        bs_scale={L: float(u(0.70, 1.40)) for L in _LABELS},
        gain=float(u(0.80, 1.25)), tgc_slope=float(u(0.6, 1.4)),
        psf_sigma_lat0=float(u(0.8, 1.6)), psf_focus_depth=float(u(0.03, 0.08)),
        speckle_strength=float(u(0.6, 1.2)), noise_sigma=float(u(0.0, 0.05)),
        dyn_range_db=float(u(45, 60)), seed=int(rng.integers(2 ** 31 - 1)))


# ======================================================================================
# NumPy fallback: reuse us_render's appearance model on a canonical label slice
# ======================================================================================
def render_bmode_numpy(label_slice: np.ndarray, rng=None) -> np.ndarray:
    """Render a canonical (N_LINES, N_SAMPLES) label slice with us_render's appearance model.
    Returns float32 (N_SAMPLES, N_LINES) in [0,1]."""
    rng = rng or np.random.default_rng(0)
    lab = np.asarray(label_slice)
    lab_us = np.zeros_like(lab, dtype=np.int64)
    for c, u in USBG_TO_USRENDER.items():
        lab_us[lab == c] = u
    L, S = lab_us.shape
    Z = us_render.Z_OF[lab_us]
    A = us_render.A_OF[lab_us]
    Sc = us_render.S_OF[lab_us]

    dZ = np.abs(np.diff(Z, axis=1, prepend=Z[:, :1]))
    refl = dZ / (Z + 1e-3)
    depth = np.linspace(0.0, us_render.IMG_DEPTH_M, S).astype(np.float32)
    dz = depth[1] - depth[0]
    atten = np.exp(-2.0 * us_render.FREQ_ATTEN * np.cumsum(A * dz * 100.0, axis=1))
    speck = Sc * np.abs(rng.normal(size=(L, S)).astype(np.float32))
    echo = (refl * 4.0 + speck * us_render.SPECKLE) * atten

    gx = us_render._gauss1d(us_render.BEAM_LAT_MM / (us_render.IMG_WIDTH_M * 1000 / L))
    gz = us_render._gauss1d(us_render.PULSE_AX_MM / (us_render.IMG_DEPTH_M * 1000 / S))
    echo = us_render._sepconv(echo, gx, gz)

    img = 20.0 * np.log10(echo + 1e-4)
    img = np.clip((img - img.min()) / (np.ptp(img) + 1e-6), 0, 1)
    img = (img ** 1.1).astype(np.float32)
    return img.T                                   # (S, L): depth rows, lateral cols


# ======================================================================================
# physics renderer (torch, MPS-native): shadowing + depth-varying PSF + correlated speckle + TGC
# ======================================================================================
def _acoustic_maps(lab, cfg, device):
    """Per-pixel (Z impedance, MU attenuation, BS backscatter) with cfg per-label multipliers."""
    import torch
    maxid = max(_LABELS)
    Z0 = np.zeros(maxid + 1, np.float32); MU0 = np.zeros(maxid + 1, np.float32)
    BS0 = np.zeros(maxid + 1, np.float32)
    zs = np.ones(maxid + 1, np.float32); ms = np.ones(maxid + 1, np.float32)
    bs = np.ones(maxid + 1, np.float32)
    for L, (z, a, m0, m1, s0) in ACOUSTIC_LOTUS.items():
        Z0[L], MU0[L], BS0[L] = z, a, m0        # impedance, attenuation, mean backscatter
        zs[L] = cfg.z_scale.get(L, 1.0); ms[L] = cfg.mu_scale.get(L, 1.0)
        bs[L] = cfg.bs_scale.get(L, 1.0)
    t = lambda x: torch.as_tensor(x, device=device)
    Zt = t(Z0 * zs)[lab]; MUt = t(MU0 * ms)[lab]; BSt = t(BS0 * bs)[lab]
    return Zt, MUt, BSt


def _gauss1d_kernel(sigma, device):
    import torch
    sigma = max(float(sigma), 1e-3)
    r = max(1, int(round(3 * sigma)))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return (k / k.sum())


def _conv_axis(img, kernel, axis):
    """Separable 1D convolution of a 2D (H,W) tensor along axis 0 (rows) or 1 (cols), padded same."""
    import torch
    import torch.nn.functional as F
    k = kernel.numel()
    pad = k // 2
    x = img[None, None]
    if axis == 0:                                    # along rows (lateral)
        w = kernel.view(1, 1, k, 1)
        x = F.pad(x, (0, 0, pad, pad), mode="reflect")
        return F.conv2d(x, w)[0, 0]
    w = kernel.view(1, 1, 1, k)                       # along cols (axial / depth)
    x = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    return F.conv2d(x, w)[0, 0]


def _lateral_depthvarying(img, cfg, depth_m, device, n_bands=3):
    """Convolve laterally (axis 0) with a Gaussian whose sigma grows away from the focus, applied
    in a few depth bands (cheap approximation to a spatially varying beam width)."""
    import torch
    S = img.shape[1]
    out = torch.empty_like(img)
    edges = np.linspace(0, S, n_bands + 1).astype(int)
    for k in range(n_bands):
        c0, c1 = edges[k], edges[k + 1]
        if c1 <= c0:
            continue
        dmid = float(depth_m[(c0 + c1) // 2])
        sigma = cfg.psf_sigma_lat0 * (1.0 + abs(dmid - cfg.psf_focus_depth) / cfg.focal_scale)
        blurred = _conv_axis(img, _gauss1d_kernel(sigma, device), axis=0)
        out[:, c0:c1] = blurred[:, c0:c1]
    return out


def render_bmode_physics(label_slice: np.ndarray, cfg: RenderConfig | None = None,
                         device: str | None = None, *, warp: bool = False) -> np.ndarray:
    """Physics renderer with shadowing, depth-varying PSF, correlated speckle, TGC, log-compress.
    Deterministic given (label_slice, cfg, cfg.seed). Returns float32 (N_SAMPLES, N_LINES) in [0,1].
    """
    import torch

    cfg = cfg or RenderConfig()
    device = device or get_device()
    lab = torch.as_tensor(np.ascontiguousarray(label_slice), dtype=torch.long, device=device)
    L, S = lab.shape                                          # rows=scanlines, cols=depth
    Z, MU, BS = _acoustic_maps(lab, cfg, device)             # (L,S) each
    dz = IMG_DEPTH_M / S
    dz_cm = dz * 100.0
    depth = np.linspace(0.0, IMG_DEPTH_M, S, dtype=np.float32)
    depth_t = torch.as_tensor(depth, device=device)[None, :]  # (1,S)

    # U1: acoustic shadow = cumulative transmission (Fresnel) * round-trip attenuation, per scanline
    eps = 1e-6
    Rc = torch.zeros_like(Z)
    dZ = Z[:, 1:] - Z[:, :-1]
    Rc[:, 1:] = (dZ / (Z[:, 1:] + Z[:, :-1] + eps)) ** 2      # intensity reflection [0,1]
    Rc = torch.clamp(Rc * cfg.shadow_gain, 0.0, 0.999)
    T_cum = torch.cumprod(1.0 - Rc, dim=1)                    # transmitted energy, ~0 behind bone/gas
    A_cum = torch.exp(-2.0 * ATTEN_COEF * torch.cumsum(MU * dz_cm, dim=1))
    shadow = T_cum * A_cum                                    # (L,S) remaining energy in [0,1]

    # U2: fully developed (correlated) speckle = envelope of PSF-convolved complex white noise
    g = torch.Generator().manual_seed(int(cfg.seed))          # CPU generator -> determinism
    w_re = torch.randn(L, S, generator=g).to(device)
    w_im = torch.randn(L, S, generator=g).to(device)
    kz = _gauss1d_kernel(cfg.psf_sigma_ax, device)
    env_re = _lateral_depthvarying(_conv_axis(w_re, kz, 1), cfg, depth, device)
    env_im = _lateral_depthvarying(_conv_axis(w_im, kz, 1), cfg, depth, device)
    env = torch.sqrt(env_re ** 2 + env_im ** 2 + eps)         # Rayleigh-like correlated envelope

    scat = BS * cfg.speckle_strength * env * shadow           # scattering echo (energy-weighted)
    spec = REFL_GAIN * Rc * shadow                            # specular echo at interfaces
    echo = _conv_axis(scat + spec, kz, 1)                     # apply axial PSF to the summed echo
    echo = _lateral_depthvarying(echo, cfg, depth, device)    # and the depth-varying lateral PSF

    # TGC: a depth-only display gain set to compensate the reference (soft-tissue) attenuation, so
    # a normal-tissue field is flat with depth while bone/gas (much larger mu) stay dark. TGC cannot
    # undo shadowing (a lateral, structure-dependent loss), which is what keeps it a usable cue.
    depth_cm = depth_t * 100.0
    tgc = torch.exp(2.0 * ATTEN_COEF * TGC_REF_MU * cfg.tgc_slope * depth_cm)
    echo = echo * tgc * cfg.gain
    echo = echo + cfg.noise_sigma * torch.randn(L, S, generator=g).to(device)
    echo = torch.clamp(echo, min=0.0)

    # log compression into a dynamic-range window -> [0,1], then a mild gamma for a moderate
    # brightness (the sim-to-real translator handles final intensity/contrast matching). The white
    # level is a high percentile (not the max) so a few bright specular pixels do not crush the rest.
    db = 20.0 * torch.log10(echo + 1e-4)
    top = torch.quantile(db.flatten(), TOP_PCTILE)
    img01 = torch.clamp((db - (top - cfg.dyn_range_db)) / cfg.dyn_range_db, 0.0, 1.0)
    img01 = img01 ** GAMMA

    img = img01.t().contiguous().cpu().numpy().astype(np.float32)   # (S,L)
    if warp:
        img = warp_curvilinear(img)
    return img


# ======================================================================================
# LOTUS-style device-agnostic renderer (original reimplementation, kept for comparison)
# ======================================================================================
def _acoustic_tables(device):
    import torch
    maxid = max(ACOUSTIC_LOTUS)
    Z = np.zeros(maxid + 1, np.float32); ATT = np.zeros(maxid + 1, np.float32)
    MU0 = np.zeros(maxid + 1, np.float32); MU1 = np.zeros(maxid + 1, np.float32)
    SIG0 = np.zeros(maxid + 1, np.float32)
    for lab, (z, a, m0, m1, s0) in ACOUSTIC_LOTUS.items():
        Z[lab], ATT[lab], MU0[lab], MU1[lab], SIG0[lab] = z, a, m0, m1, s0
    t = lambda x: torch.as_tensor(x, device=device)
    return t(Z), t(ATT), t(MU0), t(MU1), t(SIG0)


def _gauss2d_kernel(device, size: int = 3, std: float = 0.5):
    import torch
    d1 = torch.distributions.Normal(0.0, std)
    d2 = torch.distributions.Normal(0.0, std * 3)
    ax = torch.arange(-size, size + 1, dtype=torch.float32)
    vx = d1.log_prob(ax).exp(); vy = d2.log_prob(ax).exp()
    k = torch.outer(vx, vy); k = k / k.sum()
    return k[None, None].to(device)


def render_bmode_lotus(label_slice: np.ndarray, device: str | None = None, *,
                       warp: bool = False, tgc: float = TGC_DEFAULT,
                       dist_scale: float = DIST_SCALE, seed: int = 0) -> np.ndarray:
    """Original LOTUS-style physics reimplementation. Returns float32 (N_SAMPLES, N_LINES) in [0,1]."""
    import torch
    import torch.nn.functional as F

    device = device or get_device()
    torch.manual_seed(seed)
    lab = torch.as_tensor(np.ascontiguousarray(label_slice), dtype=torch.long, device=device)
    Zt, ATTt, MU0t, MU1t, SIG0t = _acoustic_tables(device)
    Z = Zt[lab]; ATT = ATTt[lab]; MU0 = MU0t[lab]; MU1 = MU1t[lab]; SIG0 = SIG0t[lab]

    dZ = torch.zeros_like(Z); dZ[:, 1:] = Z[:, 1:] - Z[:, :-1]
    boundary = 1.0 - torch.exp(-(dZ ** 2) / ALPHA_BOUNDARY)
    denom = torch.zeros_like(Z); denom[:, 1:] = Z[:, 1:] + Z[:, :-1]
    ratio = torch.zeros_like(Z); ratio[:, 1:] = dZ[:, 1:] / torch.clamp(denom[:, 1:], min=1e-6)
    refl = torch.sigmoid(ratio ** 2)

    atten = torch.exp(-ATT * dist_scale)
    atten_total = torch.cumprod(atten, dim=1)
    gain = torch.linspace(1.0, tgc, Z.shape[1], device=device)[None, :]
    atten_total = atten_total * gain
    reflection_total = torch.cumprod(1.0 - refl * boundary, dim=1)

    noise = torch.randn(Z.shape, device=device)
    scatter_prob = torch.randn(Z.shape, device=device)
    gate = torch.sigmoid(BETA_SCATTER * (MU1 - scatter_prob))
    scatterers = gate * (noise * SIG0 + MU0)

    kernel = _gauss2d_kernel(device)
    scat_conv = F.conv2d(scatterers[None, None], kernel, padding="same")[0, 0]
    bound_conv = F.conv2d(boundary[None, None], kernel, padding="same")[0, 0]
    b = atten_total * scat_conv
    r = atten_total * reflection_total * refl * bound_conv
    intensity = torch.clamp(b + r, 0.0, 1.0)

    img = intensity.t().contiguous().cpu().numpy().astype(np.float32)
    if warp:
        img = warp_curvilinear(img)
    return img


def warp_curvilinear(img_sl: np.ndarray, out_hw=(360, 480), fov_deg=None,
                     apex_frac: float = 0.35) -> np.ndarray:
    """Map a linear (S=depth, L=lateral) image to a convex sector for display (pure NumPy)."""
    S, L = img_sl.shape
    fov = np.deg2rad(us_render.FOV_DEG if fov_deg is None else fov_deg)
    out_h, out_w = out_hw
    apex = apex_frac * out_h
    yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    cx = out_w / 2.0
    dx = xx - cx; dy = yy + apex
    ang = np.arctan2(dx, dy)
    rad = np.sqrt(dx ** 2 + dy ** 2) - apex
    col = (ang / (fov / 2.0) * 0.5 + 0.5) * (L - 1)
    row = rad / (out_h - apex) * (S - 1)
    valid = (col >= 0) & (col <= L - 1) & (row >= 0) & (row <= S - 1)
    ci = np.clip(np.round(col).astype(np.int64), 0, L - 1)
    ri = np.clip(np.round(row).astype(np.int64), 0, S - 1)
    out = np.zeros((out_h, out_w), np.float32)
    out[valid] = img_sl[ri[valid], ci[valid]]
    return out


# ======================================================================================
# dispatcher
# ======================================================================================
@dataclass
class Renderer:
    """A bound (kind, cfg, translator, device) so one instance renders BOTH the goal target and
    every observation through the identical pipeline (the same-pipeline rule). `mode` is a signature
    the goal code asserts on, so a translated observation can never be compared to a raw / differently
    drawn target."""
    kind: str = "physics"
    cfg: "RenderConfig | None" = None
    translator: object = None                 # callable img01 -> img01, or None
    device: str | None = None

    def render(self, label_slice):
        return render(label_slice, self.kind, device=self.device, cfg=self.cfg,
                      texture=self.translator)

    def render_at(self, lv, T, world_from_volume):
        from . import slicer as _S
        return self.render(_S.slice_at_pose(lv, T, world_from_volume))

    @property
    def mode(self):
        fam = self.cfg.seed_family if (self.kind == "physics" and self.cfg is not None) else None
        return (self.kind, fam, self.translator is not None)


def render(label_slice: np.ndarray, renderer: str = "physics", device: str | None = None,
           texture=None, cfg: RenderConfig | None = None, **kw) -> np.ndarray:
    """Dispatch to the physics (default), LOTUS-style, or NumPy renderer. Returns float32 [0,1].

    If `texture` is a callable (e.g. a TextureTranslator.translate), it is applied as the sim-to-real
    post-stage (label slice -> physics -> texture -> B-mode). For the goal reward's same-pipeline
    rule, pass the identical `cfg` and `texture` to the target and to every observation.

    A dual-input translator (SonoSPADE) sets `wants_label = True`; render then passes the label slice
    as a second argument so texture can be per-tissue, while single-input translators are unchanged.
    """
    if renderer == "numpy":
        img = render_bmode_numpy(label_slice, rng=kw.get("rng"))
    elif renderer == "lotus":
        img = render_bmode_lotus(label_slice, device=device, warp=kw.get("warp", False),
                                 tgc=kw.get("tgc", TGC_DEFAULT),
                                 dist_scale=kw.get("dist_scale", DIST_SCALE),
                                 seed=kw.get("seed", 0))
    elif renderer == "physics":
        c = cfg or RenderConfig()
        if "seed" in kw:
            c = c.with_seed(kw["seed"])
        img = render_bmode_physics(label_slice, c, device=device, warp=kw.get("warp", False))
    else:
        raise ValueError(f"unknown renderer {renderer!r} (expected 'physics', 'lotus' or 'numpy')")
    if texture is not None:
        if getattr(texture, "wants_label", False):
            img = np.asarray(texture(img, label_slice), dtype=np.float32)   # dual input (SonoSPADE)
        else:
            img = np.asarray(texture(img), dtype=np.float32)
    return img


__all__ = ["render", "Renderer", "render_bmode_physics", "render_bmode_lotus", "render_bmode_numpy",
           "warp_curvilinear", "get_device", "RenderConfig", "default_config", "domain_randomize",
           "TGC_DEFAULT", "DIST_SCALE", "IMG_DEPTH_M"]
