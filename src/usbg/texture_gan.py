"""texture_gan.py: unpaired sim-to-real B-mode texture translation (CycleGAN), MPS-native.

The main sim-to-real lever (handoff Phase 2b). A LOTUS render has the right anatomy but a
"clean" synthetic texture; a real B-mode has characteristic speckle, shadowing, and blur. We
learn an UNPAIRED translator sim -> real with a compact CycleGAN (ResNet generator + PatchGAN
discriminator, LSGAN + cycle-consistency + identity), everything device-agnostic so it trains
and runs on MPS. CUT (the CACTUSS choice) is the lighter alternative; CycleGAN is implemented
here because it is self-contained and easy to control.

The generator trains on two pools of unpaired frames: a `sim` pool (LOTUS renders) and a `real`
pool. Real, label-free, pose-free B-mode clips in the target anatomy are the intended `real`
pool (a separate track); for a demonstration without them, proxy_real_texture synthesizes a
distinct US-like texture so the pipeline is exercised end to end and is drop-in for real frames.

At inference, TextureTranslator applies the trained G_sim2real as a post-stage on a rendered
frame; render_lotus and make_dataset expose it (with a domain-randomization mix).
"""
from __future__ import annotations

import numpy as np


# ======================================================================================
# proxy "real" texture (numpy) for demonstration without real US frames
# ======================================================================================
def proxy_real_texture(img01: np.ndarray, rng) -> np.ndarray:
    """Turn a clean LOTUS render into a distinct, more US-like texture (a stand-in target).

    Finer correlated speckle, depth-cumulative acoustic shadowing, mild blur and gamma. This is
    a KNOWN transform used only to give the unpaired GAN a target distribution when no real
    frames are available; the real pipeline points --real at actual B-mode clips instead.
    """
    from scipy.ndimage import gaussian_filter
    img = np.asarray(img01, dtype=np.float32)
    h, w = img.shape
    speckle = gaussian_filter(rng.rayleigh(0.5, size=(h, w)).astype(np.float32), sigma=0.8)
    speckle = speckle / (speckle.mean() + 1e-6)
    out = img * (0.55 + 0.75 * speckle)                      # granular multiplicative speckle
    shadow = np.exp(-1.2 * np.cumsum(img, axis=0) / h)       # darken below bright interfaces
    out = out * shadow
    out = gaussian_filter(out, sigma=0.7)                    # transducer PSF blur
    out = np.clip(out, 0, 1) ** 1.25                         # dynamic-range compression
    m = out.max()
    return (out / m if m > 0 else out).astype(np.float32)


# ======================================================================================
# networks (device-agnostic)
# ======================================================================================
def _resnet_generator(ch=1, ngf=32, n_blocks=6):
    import torch.nn as nn

    def block(dim):
        return nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim),
            nn.ReLU(True), nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim))

    class ResnetG(nn.Module):
        def __init__(self):
            super().__init__()
            layers = [nn.ReflectionPad2d(3), nn.Conv2d(ch, ngf, 7),
                      nn.InstanceNorm2d(ngf), nn.ReLU(True)]
            c = ngf
            for _ in range(2):                                # downsample
                layers += [nn.Conv2d(c, c * 2, 3, stride=2, padding=1),
                           nn.InstanceNorm2d(c * 2), nn.ReLU(True)]
                c *= 2
            self.res = nn.ModuleList([block(c) for _ in range(n_blocks)])
            self.enc = nn.Sequential(*layers)
            self.feat_channels = (ngf, c, c)                  # early conv, encoded, mid-res
            up = []
            for _ in range(2):                                # upsample (resize-conv avoids the
                up += [nn.Upsample(scale_factor=2, mode="nearest"),   # checkerboard artifact that
                       nn.Conv2d(c, c // 2, 3, padding=1),            # transposed conv produces)
                       nn.InstanceNorm2d(c // 2), nn.ReLU(True)]
                c //= 2
            up += [nn.ReflectionPad2d(3), nn.Conv2d(c, ch, 7), nn.Tanh()]
            self.dec = nn.Sequential(*up)

        def forward(self, x):
            h = self.enc(x)
            for b in self.res:
                h = h + b(h)
            return self.dec(h)

        def forward_features(self, x):
            """Encoder features at three depths (for CUT's PatchNCE). Channels = self.feat_channels."""
            e = self.enc
            h = x
            for i in range(4):                                # first conv + IN + ReLU
                h = e[i](h)
            f_early = h
            for i in range(4, len(e)):
                h = e[i](h)
            f_enc = h
            for j, b in enumerate(self.res):
                h = h + b(h)
                if j == len(self.res) // 2:
                    f_mid = h
            return [f_early, f_enc, f_mid]

    return ResnetG()


def sobel_edges(x):
    """Edge magnitude of a (B,1,H,W) tensor in [-1,1] or [0,1]; used for the structure term/metric."""
    import torch
    import torch.nn.functional as F
    kx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


def _patch_discriminator(ch=1, ndf=32, n_layers=3):
    import torch.nn as nn
    layers = [nn.Conv2d(ch, ndf, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
    c = ndf
    for i in range(1, n_layers):
        layers += [nn.Conv2d(c, c * 2, 4, stride=2, padding=1),
                   nn.InstanceNorm2d(c * 2), nn.LeakyReLU(0.2, True)]
        c *= 2
    layers += [nn.Conv2d(c, c * 2, 4, stride=1, padding=1), nn.InstanceNorm2d(c * 2),
               nn.LeakyReLU(0.2, True), nn.Conv2d(c * 2, 1, 4, stride=1, padding=1)]
    return nn.Sequential(*layers)


# ======================================================================================
# SonoSPADE networks: spatially-adaptive (SPADE) dual-input generator (phys B-mode + one-hot label)
# ======================================================================================
def onehot_labels(labels, n_classes, device=None):
    """(B,H,W) or (H,W) int label map -> (B,n_classes,H,W) float one-hot, on `device`.
    Values are clamped into [0, n_classes-1] so an unexpected id never indexes out of range."""
    import torch
    import torch.nn.functional as F
    t = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    if t.ndim == 2:
        t = t[None]
    t = t.clamp_(0, n_classes - 1)
    oh = F.one_hot(t, n_classes).permute(0, 3, 1, 2).to(torch.float32)
    return oh.to(device) if device is not None else oh


class _SPADE:
    """Factory for a SPADE normalization module (Park 2019): a parameter-free InstanceNorm whose
    per-pixel scale (gamma) and bias (beta) are predicted from the resized label map, so the
    semantic layout modulates activations at every scale (the per-tissue architectural guarantee)."""

    def __new__(cls, norm_nc, label_nc, hidden=64, ks=3):
        import torch.nn as nn
        import torch.nn.functional as F

        class SPADE(nn.Module):
            def __init__(self):
                super().__init__()
                self.param_free = nn.InstanceNorm2d(norm_nc, affine=False)
                pw = ks // 2
                self.shared = nn.Sequential(nn.Conv2d(label_nc, hidden, ks, padding=pw), nn.ReLU(True))
                self.to_gamma = nn.Conv2d(hidden, norm_nc, ks, padding=pw)
                self.to_beta = nn.Conv2d(hidden, norm_nc, ks, padding=pw)

            def forward(self, x, seg):
                normed = self.param_free(x)
                seg = F.interpolate(seg, size=x.shape[2:], mode="nearest")
                a = self.shared(seg)
                return normed * (1 + self.to_gamma(a)) + self.to_beta(a)

        return SPADE()


class _SPADEResBlock:
    """Factory for a SPADE residual block: two SPADE-normalized 3x3 convs with a (learned when the
    channel count changes) SPADE-normalized shortcut. LeakyReLU activations, as in SPADE/OASIS."""

    def __new__(cls, fin, fout, label_nc, hidden=64):
        import torch.nn as nn
        import torch.nn.functional as F
        fmid = min(fin, fout)
        learned = fin != fout

        class SPADEResBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv0 = nn.Conv2d(fin, fmid, 3, padding=1)
                self.conv1 = nn.Conv2d(fmid, fout, 3, padding=1)
                self.norm0 = _SPADE(fin, label_nc, hidden)
                self.norm1 = _SPADE(fmid, label_nc, hidden)
                self.learned = learned
                if learned:
                    self.conv_s = nn.Conv2d(fin, fout, 1, bias=False)
                    self.norm_s = _SPADE(fin, label_nc, hidden)

            def _act(self, x):
                return F.leaky_relu(x, 0.2)

            def forward(self, x, seg):
                x_s = self.conv_s(self.norm_s(x, seg)) if self.learned else x
                dx = self.conv0(self._act(self.norm0(x, seg)))
                dx = self.conv1(self._act(self.norm1(dx, seg)))
                return x_s + dx

        return SPADEResBlock()


def _spade_generator(n_classes, ngf=32, n_blocks=6, spade_hidden=64):
    """Dual-input SPADE generator: refines a physics B-mode (content) under label-map modulation.

    Input:  phys (B,1,H,W) in [-1,1]  and  seg (B,n_classes,H,W) one-hot.
    Output: fake real US (B,1,H,W) in [-1,1] (Tanh). The content flows through a plain
    (InstanceNorm) encoder / decoder while every residual and upsample block is SPADE-modulated by
    the label map, so texture is set per tissue without the net being free to ignore the layout.
    Exposes forward_features(phys, seg) and feat_channels for reuse of CUT's PatchNCE head.
    """
    import torch.nn as nn

    class SPADEGen(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(1, ngf, 7),
                                      nn.InstanceNorm2d(ngf), nn.ReLU(True))
            c = ngf
            down = []
            for _ in range(2):
                down += [nn.Conv2d(c, c * 2, 3, stride=2, padding=1),
                         nn.InstanceNorm2d(c * 2), nn.ReLU(True)]
                c *= 2
            self.down = nn.Sequential(*down)
            self.feat_channels = (ngf, c, c)                 # early conv, encoded, mid-res (PatchNCE)
            self.res = nn.ModuleList([_SPADEResBlock(c, c, n_classes, spade_hidden)
                                      for _ in range(n_blocks)])
            self.up = nn.ModuleList()
            self.up_blocks = nn.ModuleList()
            for _ in range(2):
                self.up.append(nn.Upsample(scale_factor=2, mode="nearest"))
                self.up_blocks.append(_SPADEResBlock(c, c // 2, n_classes, spade_hidden))
                c //= 2
            self.out = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(c, 1, 7), nn.Tanh())

        def forward(self, phys, seg):
            h = self.down(self.stem(phys))
            for b in self.res:
                h = b(h, seg)
            for up, blk in zip(self.up, self.up_blocks):
                h = blk(up(h), seg)
            return self.out(h)

        def forward_features(self, phys, seg):
            """Content-path features at three depths for PatchNCE (seg only shapes the res blocks)."""
            f_early = self.stem(phys)
            f_enc = self.down(f_early)
            h = f_enc
            f_mid = f_enc
            for j, b in enumerate(self.res):
                h = b(h, seg)
                if j == len(self.res) // 2:
                    f_mid = h
            return [f_early, f_enc, f_mid]

    return SPADEGen()


# ======================================================================================
# CycleGAN trainer
# ======================================================================================
class CycleGAN:
    """Compact CycleGAN for unpaired sim<->real texture. LSGAN + cycle + identity."""

    def __init__(self, device, ngf=32, ndf=32, n_blocks=6, lr=2e-4,
                 lambda_cyc=10.0, lambda_id=5.0, lambda_struct=0.0):
        import torch
        self.device = device
        self.G_s2r = _resnet_generator(ngf=ngf, n_blocks=n_blocks).to(device)
        self.G_r2s = _resnet_generator(ngf=ngf, n_blocks=n_blocks).to(device)
        self.D_r = _patch_discriminator(ndf=ndf).to(device)
        self.D_s = _patch_discriminator(ndf=ndf).to(device)
        self.lambda_cyc, self.lambda_id, self.lambda_struct = lambda_cyc, lambda_id, lambda_struct
        self.opt_g = torch.optim.Adam(
            list(self.G_s2r.parameters()) + list(self.G_r2s.parameters()), lr=lr, betas=(0.5, 0.999))
        self.opt_d = torch.optim.Adam(
            list(self.D_r.parameters()) + list(self.D_s.parameters()), lr=lr, betas=(0.5, 0.999))

    def _adv(self, d_out, target_is_real):
        import torch
        t = torch.ones_like(d_out) if target_is_real else torch.zeros_like(d_out)
        return ((d_out - t) ** 2).mean()

    def step(self, sim, real):
        """One optimization step on a (sim, real) minibatch in [-1,1]. Returns a loss dict."""
        import torch
        # generators
        self.opt_g.zero_grad()
        fake_r = self.G_s2r(sim)
        fake_s = self.G_r2s(real)
        loss_g_adv = self._adv(self.D_r(fake_r), True) + self._adv(self.D_s(fake_s), True)
        loss_cyc = ((self.G_r2s(fake_r) - sim).abs().mean()
                    + (self.G_s2r(fake_s) - real).abs().mean()) * self.lambda_cyc
        loss_id = ((self.G_s2r(real) - real).abs().mean()
                   + (self.G_r2s(sim) - sim).abs().mean()) * self.lambda_id
        # structure-consistency: edges of the sim->real translation must match the sim's edges, so
        # the translator changes texture without shifting or hallucinating anatomy (variant iii)
        loss_struct = torch.zeros((), device=self.device)
        if self.lambda_struct > 0:
            loss_struct = (sobel_edges(fake_r) - sobel_edges(sim)).abs().mean() * self.lambda_struct
        loss_g = loss_g_adv + loss_cyc + loss_id + loss_struct
        loss_g.backward()
        self.opt_g.step()
        # discriminators
        self.opt_d.zero_grad()
        loss_d = (self._adv(self.D_r(real), True) + self._adv(self.D_r(fake_r.detach()), False)
                  + self._adv(self.D_s(sim), True) + self._adv(self.D_s(fake_s.detach()), False)) * 0.5
        loss_d.backward()
        self.opt_d.step()
        return {"g": loss_g.item(), "d": loss_d.item(), "cyc": loss_cyc.item(),
                "id": loss_id.item(), "struct": float(loss_struct.detach())}

    def save(self, path):
        import torch
        torch.save({"G_s2r": self.G_s2r.state_dict()}, path)


class _PatchSampleF:
    """CUT's projection head H: sample num_patches locations per feature layer, project via a small
    MLP, L2-normalize. Built lazily once feature channels are known."""

    def __init__(self, device, nc=256):
        import torch.nn as nn
        self.device = device
        self.nc = nc
        self.mlps = None
        self._nn = nn

    def build(self, channels):
        import torch
        self.mlps = [self._nn.Sequential(self._nn.Linear(c, self.nc), self._nn.ReLU(),
                                         self._nn.Linear(self.nc, self.nc)).to(self.device)
                     for c in channels]
        self.params = [p for m in self.mlps for p in m.parameters()]
        return self

    def __call__(self, feats, num_patches=256, patch_ids=None):
        import torch
        outs, ids = [], []
        for i, f in enumerate(feats):
            B, C, H, W = f.shape
            ff = f.permute(0, 2, 3, 1).reshape(B, H * W, C)
            if patch_ids is not None:
                pid = patch_ids[i]
            else:
                pid = torch.randperm(H * W, device=f.device)[:num_patches]
            samp = ff[:, pid, :].reshape(-1, C)
            samp = self.mlps[i](samp)
            outs.append(samp / (samp.norm(dim=1, keepdim=True) + 1e-7))
            ids.append(pid)
        return outs, ids


class CUT:
    """Contrastive Unpaired Translation (Park 2020): one-sided sim->real generator with a PatchGAN
    discriminator and multilayer PatchNCE (a translated patch must match its source patch and differ
    from other patches). Lighter than CycleGAN (no inverse generator or cycle). Saves G as G_s2r so
    the inference TextureTranslator loads it unchanged."""

    def __init__(self, device, ngf=32, ndf=32, n_blocks=6, lr=2e-4, lambda_nce=1.0, nce_tau=0.07):
        import torch
        self.device = device
        self.lambda_nce, self.tau = lambda_nce, nce_tau
        self.G_s2r = _resnet_generator(ngf=ngf, n_blocks=n_blocks).to(device)
        self.D_r = _patch_discriminator(ndf=ndf).to(device)
        self.H = _PatchSampleF(device).build(self.G_s2r.feat_channels)
        self.opt_g = torch.optim.Adam(self.G_s2r.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_d = torch.optim.Adam(self.D_r.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_h = torch.optim.Adam(self.H.params, lr=lr, betas=(0.5, 0.999))

    def _adv(self, d_out, real):
        import torch
        t = torch.ones_like(d_out) if real else torch.zeros_like(d_out)
        return ((d_out - t) ** 2).mean()

    def _patchnce(self, feat_q, feat_k):
        import torch
        import torch.nn.functional as F
        loss = 0.0
        for q, k in zip(feat_q, feat_k):
            n = q.shape[0]
            logits = (q @ k.t()) / self.tau                    # (n,n): diagonal = positive
            labels = torch.arange(n, device=q.device)
            loss = loss + F.cross_entropy(logits, labels)
        return loss / max(1, len(feat_q))

    def step(self, sim, real):
        import torch
        fake = self.G_s2r(sim)
        # discriminator
        self.opt_d.zero_grad()
        loss_d = 0.5 * (self._adv(self.D_r(real), True) + self._adv(self.D_r(fake.detach()), False))
        loss_d.backward(); self.opt_d.step()
        # generator + projection head
        self.opt_g.zero_grad(); self.opt_h.zero_grad()
        loss_g_adv = self._adv(self.D_r(fake), True)
        fk = self.G_s2r.forward_features(fake)
        fq = self.G_s2r.forward_features(sim)
        q, ids = self.H(fk); k, _ = self.H(fq, patch_ids=ids)
        loss_nce = self._patchnce(q, k)
        idt = self.G_s2r(real)                                  # identity NCE stabilizes structure
        fk2 = self.G_s2r.forward_features(idt); fq2 = self.G_s2r.forward_features(real)
        q2, ids2 = self.H(fk2); k2, _ = self.H(fq2, patch_ids=ids2)
        loss_nce = 0.5 * (loss_nce + self._patchnce(q2, k2))
        loss_g = loss_g_adv + self.lambda_nce * loss_nce
        loss_g.backward(); self.opt_g.step(); self.opt_h.step()
        return {"g": float(loss_g.detach()), "d": float(loss_d.detach()),
                "nce": float(loss_nce.detach())}

    def save(self, path):
        import torch
        torch.save({"G_s2r": self.G_s2r.state_dict()}, path)


# ======================================================================================
# SonoSPADE trainer: dual-input SPADE G + OASIS seg-D + PatchNCE + frozen S_US seg-consistency
# ======================================================================================
def _patchnce_loss(feat_q, feat_k, tau):
    """Multilayer PatchNCE (Park 2020): a translated patch (q) must match its source patch (k, the
    positive on the diagonal) and differ from the other sampled patches. Reused by CUT and SonoSPADE."""
    import torch
    import torch.nn.functional as F
    loss = 0.0
    for q, k in zip(feat_q, feat_k):
        n = q.shape[0]
        logits = (q @ k.t()) / tau                     # (n,n): diagonal = positive pair
        labels = torch.arange(n, device=q.device)
        loss = loss + F.cross_entropy(logits, labels)
    return loss / max(1, len(feat_q))


class SonoSPADE:
    """Physics-conditioned, per-tissue ultrasound texture translator (the proposed method).

    G is the dual-input SPADE generator (physics B-mode content + one-hot label modulation). It is
    trained UNPAIRED against a real pool with three forces, exactly the handoff objective:
      * L_adv:  an OASIS style per-pixel (N real + 1 fake) segmentation discriminator. The real pool
                has no labels, so a frozen S_US pseudo-labels it, giving D a semantic target for free.
                G wins when D classifies each fake pixel as its conditioning tissue class, which is a
                per-pixel (hence per-tissue) realism signal, far stronger than a global real/fake bit.
      * L_nce:  PatchNCE content preservation, reusing CUT.forward_features and _PatchSampleF so the
                translated frame keeps the physics render's anatomy (no cycle, no texture wash).
      * L_seg:  a frozen S_US consistency loss (CE + Dice): S_US(G(phys, label)) must match the input
                label, so G cannot ignore the layout and collapse to a global recolor (the one trap).

    S_US is pretrained on the free aligned sim pairs (scripts/train_segmenter.py) and frozen here (v1).
    G is one-sided (no inverse generator, no cycle), so it is light enough to train on MPS.
    """

    def __init__(self, device, s_us=None, s_us_ckpt=None, n_classes=13, ngf=32, ndf=32, n_blocks=6,
                 spade_hidden=64, d_depth=4, lr=2e-4, lambda_nce=1.0, lambda_seg=5.0,
                 lambda_seg_dice=1.0, lambda_tc=0.0, lambda_lm=0.0, r1_gamma=0.0, ema_decay=0.999,
                 nce_tau=0.07, beta1=0.5, beta2=0.999):
        import copy
        import torch
        from .segmenter import Segmenter, build_segmenter
        self.device = device
        self.n_classes = n_classes                     # generator conditions on the full canonical scheme
        self._ngf, self._n_blocks, self._spade_hidden = ngf, n_blocks, spade_hidden
        self.lambda_nce, self.lambda_seg, self.lambda_seg_dice = lambda_nce, lambda_seg, lambda_seg_dice
        self.lambda_tc, self.lambda_lm, self.r1_gamma = lambda_tc, lambda_lm, r1_gamma  # improvements 2,3,4
        self.tau = nce_tau
        self.G = _spade_generator(n_classes, ngf=ngf, n_blocks=n_blocks,
                                  spade_hidden=spade_hidden).to(device)
        if s_us is None:
            if s_us_ckpt is None:
                raise ValueError("sonospade needs a pretrained S_US: pass s_us= or s_us_ckpt=")
            s_us = Segmenter.load(s_us_ckpt, device=device)
        self.S = s_us
        # S_US may use a grouped scheme (improvement 1): the discriminator and the consistency /
        # pseudo-label targets live in the segmenter's class space, not the 13-class conditioning space
        self.n_seg = self.S.n_classes
        gm = getattr(self.S, "group_map", None)
        self.gmap_t = (torch.as_tensor(gm, dtype=torch.long, device=device)
                       if gm is not None else None)   # canonical id -> segmenter class id
        # OASIS discriminator = a per-pixel (n_seg + 1)-way classifier; reuse the segmenter U-Net
        # (in_ch=1, avg pooling so it stays twice-differentiable under the optional R1 penalty)
        self.D = build_segmenter(self.n_seg + 1, base=ndf, depth=d_depth, in_ch=1, pool="avg").to(device)
        self.H = _PatchSampleF(device).build(self.G.feat_channels)
        self.S.net.eval()
        for p in self.S.net.parameters():
            p.requires_grad_(False)                    # frozen; gradients still flow through to G
        # generator EMA (improvement 4): a smoothed weight copy used for inference (rampup decay so a
        # short run is not washed out by an under-updated average)
        self.ema_decay = ema_decay
        if ema_decay and ema_decay > 0:
            self.G_ema = _spade_generator(n_classes, ngf=ngf, n_blocks=n_blocks,
                                          spade_hidden=spade_hidden).to(device)
            self.G_ema.load_state_dict(self.G.state_dict())
            for p in self.G_ema.parameters():
                p.requires_grad_(False)
            self._ema_steps = 0
        else:
            self.G_ema = None
        self.opt_g = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(beta1, beta2))
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(beta1, beta2))
        self.opt_h = torch.optim.Adam(self.H.params, lr=lr, betas=(beta1, beta2))

    @staticmethod
    def _to01(x):
        return (x + 1.0) * 0.5                          # [-1,1] (G / D convention) -> [0,1] (S_US)

    def _seg_target(self, target_lab):
        """Map a canonical (13-class) conditioning label into the segmenter's class scheme (identity
        unless S_US uses the grouped scheme, in which case degenerate soft tissues collapse to one)."""
        return self.gmap_t[target_lab] if self.gmap_t is not None else target_lab

    def _balanced_ce(self, logits, target):
        """Class-balanced per-pixel cross entropy (OASIS): weight each class inversely to its pixel
        frequency in the target, so rare tissues are not ignored. Absent classes get zero weight."""
        import torch
        import torch.nn.functional as F
        n = logits.shape[1]
        counts = torch.bincount(target.flatten().to("cpu"), minlength=n).float()  # CPU: MPS-safe
        present = counts > 0
        total = counts.sum().clamp(min=1)
        w = torch.zeros(n)
        w[present] = total / (present.sum().float() * counts[present])
        return F.cross_entropy(logits, target, weight=w.to(logits.device))

    @staticmethod
    def _shift(x, dy, dx):
        import torch
        return torch.roll(x, shifts=(dy, dx), dims=(2, 3))

    def generator_losses(self, phys, label_onehot, fake):
        """The generator-side loss TERMS as tensors, for a given fake. step() and the tests share this
        exact code path, so the seg-consistency gradient path into G (through the frozen S_US) is
        actually exercised. Includes the optional temporal-consistency term (improvement 2) when
        lambda_tc > 0: the translator must commute with a small shift, so adjacent slices (which are
        approximately small in-plane shifts of the content) do not flicker."""
        from .segmenter import seg_ce_dice_loss
        target_lab = label_onehot.argmax(dim=1)                  # canonical conditioning classes
        seg_target = self._seg_target(target_lab)                # in the segmenter's class scheme
        loss_g_adv = self._balanced_ce(self.D(fake), seg_target)
        fk = self.G.forward_features(fake, label_onehot)
        fq = self.G.forward_features(phys, label_onehot)
        q, ids = self.H(fk); k, _ = self.H(fq, patch_ids=ids)
        loss_nce = _patchnce_loss(q, k, self.tau)
        seg_logits = self.S.net(self._to01(fake))                # frozen S_US, grad must flow into G
        loss_seg = seg_ce_dice_loss(seg_logits, seg_target, n_classes=self.n_seg,
                                    dice_weight=self.lambda_seg_dice)
        out = {"adv": loss_g_adv, "nce": loss_nce, "seg": loss_seg}
        if self.lambda_tc and self.lambda_tc > 0:
            dy, dx = 2, 2
            shifted = self.G(self._shift(phys, dy, dx), self._shift(label_onehot, dy, dx))
            out["tc"] = (shifted - self._shift(fake, dy, dx)).abs().mean()
        return out

    def _labelmix_mask(self, b, h, w):
        """A random rectangle (CutMix-style) binary mask, per sample (b,1,h,w). Used by the LabelMix
        consistency regularizer (improvement 3)."""
        import torch
        m = torch.zeros(b, 1, h, w, device=self.device)
        for i in range(b):
            hh = int(torch.randint(h // 4, max(h // 4 + 1, 3 * h // 4), (1,)).item())
            ww = int(torch.randint(w // 4, max(w // 4 + 1, 3 * w // 4), (1,)).item())
            y0 = int(torch.randint(0, max(1, h - hh + 1), (1,)).item())
            x0 = int(torch.randint(0, max(1, w - ww + 1), (1,)).item())
            m[i, 0, y0:y0 + hh, x0:x0 + ww] = 1.0
        return m

    def _update_ema(self):
        import torch
        if self.G_ema is None:
            return
        self._ema_steps += 1
        d = min(self.ema_decay, (1.0 + self._ema_steps) / (10.0 + self._ema_steps))  # rampup
        with torch.no_grad():
            for pe, p in zip(self.G_ema.parameters(), self.G.parameters()):
                pe.mul_(d).add_(p.detach(), alpha=1.0 - d)
            for be, b in zip(self.G_ema.buffers(), self.G.buffers()):
                be.copy_(b)

    def step(self, phys, label_onehot, real):
        """One optimization step. phys (B,1,H,W) in [-1,1], label_onehot (B,n_classes,H,W) float,
        real (B,1,H,W) in [-1,1]. Returns a loss dict. The three core losses are always on (never
        drop the seg-consistency term, or texture stops being per-tissue); temporal, LabelMix, R1,
        and EMA activate from their weights (0 = off)."""
        import torch
        target_lab = label_onehot.argmax(dim=1)
        fake = self.G(phys, label_onehot)                        # (B,1,H,W) in [-1,1]
        B, _, H, W = fake.shape

        # discriminator: real -> S_US pseudo classes, fake -> the fake class (index n_seg)
        self.opt_d.zero_grad()
        with torch.no_grad():
            pseudo = self.S.net(self._to01(real)).argmax(dim=1)  # already in the segmenter scheme
        use_r1 = self.r1_gamma and self.r1_gamma > 0
        real_in = real.detach().requires_grad_(True) if use_r1 else real
        d_real = self.D(real_in)
        loss_d_real = self._balanced_ce(d_real, pseudo)
        d_fake = self.D(fake.detach())
        fake_tgt = torch.full_like(target_lab, self.n_seg)       # per-pixel "fake" class
        loss_d_fake = self._balanced_ce(d_fake, fake_tgt)
        loss_d = loss_d_real + loss_d_fake
        extra = {}
        if use_r1:                                               # R1 gradient penalty (improvement 4)
            grad = torch.autograd.grad(d_real.sum(), real_in, create_graph=True)[0]
            # the OASIS D is per-pixel, so normalize by the output spatial size to keep the penalty
            # resolution-independent and in the usual O(1..10) range (else it swamps the D loss)
            loss_r1 = grad.pow(2).flatten(1).sum(1).mean() / float(H * W)
            loss_d = loss_d + 0.5 * self.r1_gamma * loss_r1
            extra["r1"] = float(loss_r1.detach())
        if self.lambda_lm and self.lambda_lm > 0:               # LabelMix consistency (improvement 3)
            m = self._labelmix_mask(B, H, W)
            mixed = m * real.detach() + (1.0 - m) * fake.detach()
            target_mix = m * d_real.detach() + (1.0 - m) * d_fake.detach()
            loss_lm = (self.D(mixed) - target_mix).pow(2).mean()
            loss_d = loss_d + self.lambda_lm * loss_lm
            extra["lm"] = float(loss_lm.detach())
        loss_d.backward(); self.opt_d.step()

        # generator: adv + NCE + seg-consistency (+ optional temporal)
        self.opt_g.zero_grad(); self.opt_h.zero_grad()
        g = self.generator_losses(phys, label_onehot, fake)
        loss_g = g["adv"] + self.lambda_nce * g["nce"] + self.lambda_seg * g["seg"]
        if "tc" in g:
            loss_g = loss_g + self.lambda_tc * g["tc"]
        loss_g.backward(); self.opt_g.step(); self.opt_h.step()
        self._update_ema()
        out = {"g": float(loss_g.detach()), "d": float(loss_d.detach()),
               "adv": float(g["adv"].detach()), "nce": float(g["nce"].detach()),
               "seg": float(g["seg"].detach())}
        if "tc" in g:
            out["tc"] = float(g["tc"].detach())
        out.update(extra)
        return out

    def save(self, path):
        import os
        import torch
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # deploy the EMA generator when available (smoother inference); keep the raw weights too
        deployed = self.G_ema if self.G_ema is not None else self.G
        torch.save({"G_spade": deployed.state_dict(), "G_raw": self.G.state_dict(),
                    "n_classes": self.n_classes, "ngf": self._ngf, "n_blocks": self._n_blocks,
                    "spade_hidden": self._spade_hidden}, path)


def load_config(path: str = "configs/sonospade.yaml") -> dict:
    """Load the SonoSPADE YAML config into a plain nested dict (pyyaml). Kept dependency-light so
    P0's 'config loads' check is just import usbg.texture_gan then load_config()."""
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh)


def build_trainer(variant: str, device, **kw):
    """Registry of sim-to-real translators: cyclegan | cut | cyclegan_sc | sonospade.

    sonospade is the dual-input SPADE generator with an OASIS segmentation discriminator, PatchNCE
    content preservation, and a frozen S_US segmentation-consistency loss (handoff method). It needs
    a pretrained S_US segmenter, passed as s_us= (a usbg.segmenter U-Net) or s_us_ckpt= (a path)."""
    if variant == "cyclegan":
        return CycleGAN(device, **kw)
    if variant == "cyclegan_sc":
        return CycleGAN(device, lambda_struct=kw.pop("lambda_struct", 8.0), **kw)
    if variant == "cut":
        return CUT(device, **kw)
    if variant == "sonospade":
        return SonoSPADE(device, **kw)
    raise ValueError(f"unknown translator variant {variant!r} "
                     "(cyclegan | cut | cyclegan_sc | sonospade)")


# ======================================================================================
# comparison metrics: realism (intensity Wasserstein-1) and structure preservation (edge IoU)
# ======================================================================================
def intensity_w1(pool_a: np.ndarray, pool_b: np.ndarray, bins=256) -> float:
    """1D Wasserstein-1 distance between two pixel-intensity distributions (frames in [0,1])."""
    a = np.sort(np.asarray(pool_a, np.float32).ravel())
    b = np.sort(np.asarray(pool_b, np.float32).ravel())
    q = np.linspace(0, 1, bins, dtype=np.float32)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def edge_iou(src01: np.ndarray, trans01: np.ndarray, thresh=0.15) -> float:
    """Structure-preservation proxy: IoU of binarized Sobel edges of source vs translated frame.
    High IoU means the translator kept anatomy edges in place (did not shift/hallucinate)."""
    from scipy.ndimage import sobel
    def edges(x):
        gx = sobel(x.astype(np.float32), axis=0); gy = sobel(x.astype(np.float32), axis=1)
        e = np.sqrt(gx ** 2 + gy ** 2)
        e = e / (e.max() + 1e-8)
        return e > thresh
    ea, eb = edges(np.asarray(src01)), edges(np.asarray(trans01))
    inter = np.logical_and(ea, eb).sum(); union = np.logical_or(ea, eb).sum()
    return float(inter / (union + 1e-8))


# ======================================================================================
# inference-time translator (the render_lotus / make_dataset post-stage)
# ======================================================================================
class TextureTranslator:
    """Apply a trained G_sim2real to a rendered frame. Loads only the generator (inference)."""

    def __init__(self, ckpt_path, device=None, ngf=32, n_blocks=6):
        import torch
        from .render_lotus import get_device
        self.device = device or get_device()
        self.G = _resnet_generator(ngf=ngf, n_blocks=n_blocks).to(self.device)
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.G.load_state_dict(state["G_s2r"] if "G_s2r" in state else state)
        self.G.eval()

    def translate(self, img01: np.ndarray) -> np.ndarray:
        """img01: HxW float [0,1] -> textured HxW float [0,1] (same shape)."""
        import torch
        a = torch.from_numpy(np.asarray(img01, np.float32) * 2 - 1)[None, None].to(self.device)
        with torch.no_grad():
            out = self.G(a)[0, 0].cpu().numpy()
        return np.clip((out + 1) / 2, 0, 1).astype(np.float32)

    __call__ = translate                              # callable, so it is a drop-in render texture


class SonoSPADETranslator:
    """Inference-time SonoSPADE post-stage: refine a physics render with per-tissue texture, given
    the label slice. Dual input, so render_lotus.render passes the label slice too (wants_label).

    translate(img01, label_slice): img01 is the physics render (N_SAMPLES, N_LINES) in [0,1];
    label_slice is the canonical (N_LINES, N_SAMPLES) slicer output, transposed and resized to the
    image internally (segmenter.align_label_to_image) so image and label align."""

    wants_label = True

    def __init__(self, ckpt_path, device=None):
        import torch
        from .render_lotus import get_device
        self.device = device or get_device()
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.n_classes = state.get("n_classes", 13)
        self.G = _spade_generator(self.n_classes, ngf=state.get("ngf", 32),
                                  n_blocks=state.get("n_blocks", 6),
                                  spade_hidden=state.get("spade_hidden", 64)).to(self.device)
        self.G.load_state_dict(state["G_spade"] if "G_spade" in state else state)
        self.G.eval()

    def translate_aligned(self, img01: np.ndarray, aligned_label: np.ndarray) -> np.ndarray:
        """Textured output given a label ALREADY in the image orientation and size (H,W int ids).
        No transpose. Use this when you already hold an aligned label (evaluation, ablations)."""
        import torch
        img = np.asarray(img01, np.float32)
        h, w = img.shape
        lab = np.asarray(aligned_label)
        if lab.shape != (h, w):                                         # nearest-resize class ids
            from PIL import Image
            lab = np.asarray(Image.fromarray(lab.astype(np.uint8)).resize((w, h), Image.NEAREST))
        seg = onehot_labels(lab, self.n_classes, self.device)
        a = torch.from_numpy(img * 2 - 1)[None, None].to(self.device)
        with torch.no_grad():
            out = self.G(a, seg)[0, 0].cpu().numpy()
        out = np.clip((out + 1) / 2, 0, 1).astype(np.float32)
        if out.shape != img.shape:                                      # guard non-multiple-of-4 sizes
            from PIL import Image
            out = np.asarray(Image.fromarray((out * 255).astype(np.uint8)).resize(
                (w, h), Image.BILINEAR), np.float32) / 255.0
        return out

    def translate(self, img01: np.ndarray, label_slice: np.ndarray) -> np.ndarray:
        """Textured output given a slicer-orientation label (N_LINES, N_SAMPLES); transposes and
        resizes it to the render, then applies the generator. This is what render_lotus calls."""
        from .segmenter import align_label_to_image
        img = np.asarray(img01, np.float32)
        return self.translate_aligned(img, align_label_to_image(label_slice, img.shape))

    __call__ = translate


def load_translator(ckpt_path, device=None):
    """Return the right inference translator for a checkpoint, so any consumer (render_lotus,
    make_dataset, the env) is a drop-in regardless of variant. A SonoSPADE checkpoint stores its
    generator under 'G_spade' (dual input, wants_label); the grayscale variants store 'G_s2r'."""
    import torch
    from .render_lotus import get_device
    dev = device or get_device()
    state = torch.load(ckpt_path, map_location=dev, weights_only=False)
    if "G_spade" in state:
        return SonoSPADETranslator(ckpt_path, device=dev)
    return TextureTranslator(ckpt_path, device=dev)


# ======================================================================================
# frame pools for training
# ======================================================================================
def frames_from_npz(path: str, limit: int | None = None) -> np.ndarray:
    """Flatten all rendered frames from a usbg dataset npz into (N, H, W) float [0,1]."""
    d = np.load(path, allow_pickle=True)
    imgs = []
    for clip in d["images"]:
        for f in clip:
            imgs.append(np.asarray(f, np.float32) / 255.0)
    arr = np.asarray(imgs, np.float32)
    if limit:
        arr = arr[:limit]
    return arr


def to_batch(frames, idx, device):
    import torch
    b = np.stack([frames[i] for i in idx], 0)[:, None] * 2 - 1     # (B,1,H,W) in [-1,1]
    return torch.from_numpy(b.astype(np.float32)).to(device)
