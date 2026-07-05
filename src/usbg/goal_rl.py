"""goal_rl.py: goal-conditioned RL that reaches a target B-mode shot by image similarity.

The objective is purely visual: given a target shot, the policy servos the probe so the current
shot looks as similar as possible to the target. Pose is never used as a goal or a reward, only
as an offline diagnostic (does high image similarity recover the target angle?). The similarity
is pluggable (usbg.similarity), so a trained B-mode similarity model can replace the default.

Same-pipeline rule (critical for the reward): within an episode the target and every observation
are rendered by the identical pipeline and the identical per-episode config draw (same physics
renderer, same domain-randomization cfg, same sim-to-real translator). The target is therefore
rendered through the env (env.render_at), never separately, and imitation pairs share one cfg draw.

Setup is local visual servoing: the target is a pose over the organ, the probe starts from a small
random perturbation of that pose (in-plane slide, axial press, fan; roll about the beam axis is
nearly invisible so it is barely perturbed), and the policy (current B-mode history + target B-mode
-> bounded 6-DoF twist) is trained with REINFORCE to maximize the per-step similarity.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .contracts_bridge import pose_from_rpy, invert_T, se3_log, DT_C
from .placement import seat_volume_under_probe
from .eval import _prep
from . import render_lotus as R
from .rl import discounted_returns

MAX_TWIST = np.array([0.03, 0.03, 0.03, 0.5, 0.5, 0.5], dtype=np.float64)   # bound per channel


@dataclass
class GoalRender:
    """How the goal pipeline renders (kept identical for target and observation). `randomize` draws
    a per-episode domain-randomization cfg; `translator` is the sim-to-real stage (a callable or
    None); `kind` selects the physics/lotus/numpy renderer."""
    kind: str = "physics"
    translator: object = None
    randomize: bool = False
    device: str | None = None

    def episode_cfg(self, rng):
        if self.kind != "physics":
            return None
        return R.domain_randomize(R.RenderConfig(), rng) if self.randomize \
            else R.RenderConfig(seed=int(rng.integers(2 ** 31 - 1)))

    def renderer(self, cfg):
        return R.Renderer(self.kind, cfg, self.translator, self.device)


def assert_same_pipeline(a: R.Renderer, b: R.Renderer):
    """Guard the same-pipeline rule: a translated / differently-drawn observation must never be
    compared against a raw or differently-drawn target."""
    if a.mode != b.mode:
        raise AssertionError(f"same-pipeline violated: target {a.mode} != observation {b.mode}")


def build_goal_gaussian_policy(net, max_twist=MAX_TWIST, explore_frac=0.5):
    """Wrap a GoalPolicy net as a bounded Gaussian policy over SI twist (mean = max*tanh(head)).
    Exploration std is per-channel, a fraction of each channel's range (the channels differ by an
    order of magnitude, so a single scalar std would saturate the linear channels)."""
    import torch
    import torch.nn as nn

    init_log_std = np.log(explore_frac * np.asarray(max_twist, dtype=np.float64))

    class _GP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net
            self.log_std = nn.Parameter(torch.as_tensor(init_log_std, dtype=torch.float32))
            self.register_buffer("max_twist", torch.as_tensor(max_twist, dtype=torch.float32))

        def mean(self, cur, tgt):
            return self.max_twist * torch.tanh(self.net(cur, tgt))

        def act(self, cur, tgt, deterministic=False):
            m = self.mean(cur, tgt)
            if deterministic:
                return m, None
            std = self.log_std.exp()
            a = (m + std * torch.randn_like(m)).detach()
            a = torch.clamp(a, -self.max_twist, self.max_twist)
            logp = (-0.5 * (((a - m) / std) ** 2 + 2 * self.log_std + np.log(2 * np.pi))).sum(-1)
            return a, logp

    return _GP()


def sample_setup(lv, target_label, rng, lateral_m=0.02, axial_m=0.008, fan_deg=15.0,
                 roll_deg=4.0):
    """Return (world_from_volume, T_target, T_start). Start is a small random perturbation of the
    target pose along the image-observable axes (in-plane slide, axial press, fan); the beam-axis
    roll is barely perturbed since it is nearly invisible and need not be corrected to match a view.
    The target IMAGE is rendered by the caller through the env, so it shares the observation's
    pipeline (same-pipeline rule)."""
    Wfv, T_tgt = seat_volume_under_probe(lv, target_label, standoff_m=0.0, depth_m=0.05)
    d = rng.uniform
    perturb = pose_from_rpy(tx=d(-lateral_m, lateral_m), ty=d(-lateral_m, lateral_m),
                            tz=d(-axial_m, axial_m), rx=d(-fan_deg, fan_deg),
                            ry=d(-fan_deg, fan_deg), rz=d(-roll_deg, roll_deg))
    return Wfv, T_tgt, T_tgt @ perturb


def analytic_twist(T_cur, T_tgt):
    """Privileged corrective twist that moves T_cur toward T_tgt (body frame), clipped to bounds.
    Used only as a TRAINING signal for imitation warm-start (pose available in sim now, from the
    AprilTag rig later); the policy itself never sees the pose."""
    xi = se3_log(invert_T(np.asarray(T_cur)) @ np.asarray(T_tgt))
    return np.clip(xi / DT_C, -MAX_TWIST, MAX_TWIST)


def imitation_batch(lv, target_label, rng, history, image_size, batch, grender: GoalRender = None,
                    **setup_kw):
    """(cur_stacks (B,history,H,W), targets (B,1,H,W), corrective_twists (B,6)) for warm-start.
    Each pair (current, target) is rendered through ONE per-sample cfg draw (same-pipeline)."""
    grender = grender or GoalRender()
    curs, tgts, tws = [], [], []
    for _ in range(batch):
        Wfv, T_tgt, T_cur = sample_setup(lv, target_label, rng, **setup_kw)
        rend = grender.renderer(grender.episode_cfg(rng))
        tgt_shot = rend.render_at(lv, T_tgt, Wfv)
        cur_shot = rend.render_at(lv, T_cur, Wfv)
        curs.append(np.stack([_prep(cur_shot, image_size)] * history, 0))
        tgts.append(_prep(tgt_shot, image_size)[None])
        tws.append(analytic_twist(T_cur, T_tgt))
    return (np.asarray(curs, np.float32), np.asarray(tgts, np.float32),
            np.asarray(tws, np.float32))


def _pose_error(T_cur, T_tgt):
    rel = se3_log(invert_T(np.asarray(T_tgt)) @ np.asarray(T_cur))
    return float(np.linalg.norm(rel[:3]) * 1000.0), float(np.degrees(np.linalg.norm(rel[3:])))


def rollout(env, policy, similarity, T_tgt, history, image_size, device, n_steps=16,
            deterministic=False, effort=0.005):
    """Reward is the per-step similarity GAIN (s_t - s_{t-1}) minus a small effort penalty, so the
    return telescopes to the net improvement over the start and credits corrective actions. The
    target is rendered through the env (same pipeline as the observations)."""
    import torch
    obs, info = env.reset()
    target_shot = env.render_at(T_tgt)                          # SAME cfg draw as obs
    hist = deque([_prep(obs, image_size)] * history, maxlen=history)
    tgt_t = torch.from_numpy(_prep(target_shot, image_size)[None, None]).to(device)
    s_prev = similarity.score(obs, target_shot)
    logps, rewards, sims = [], [], [s_prev]
    for _ in range(n_steps):
        cur = torch.from_numpy(np.stack(hist, 0)[None]).to(device)
        a, logp = policy.act(cur, tgt_t, deterministic=deterministic)
        a_np = a[0].detach().cpu().numpy()
        obs, info = env.step(a_np)
        hist.append(_prep(obs, image_size))
        s = similarity.score(obs, target_shot)
        rewards.append((s - s_prev) - effort * float(a_np @ a_np))
        sims.append(s); s_prev = s
        if logp is not None:
            logps.append(logp[0])
    return {"log_probs": logps, "rewards": np.asarray(rewards), "sims": np.asarray(sims),
            "v0": sims[0], "v_final": sims[-1], "pose": env.world_from_probe}


def _make_env(lv, Wfv, T_start, grender: GoalRender, rng):
    from .mujoco_env import USProbeEnv
    return USProbeEnv(lv, Wfv, T_start, renderer=grender.kind, device=grender.device,
                      translator=grender.translator, randomize=grender.randomize, rng=rng)


def evaluate(lv, act_fns: dict, similarity, history, image_size, device, grender: GoalRender = None,
             n_episodes=12, n_steps=16, seed=999, **setup_kw):
    """Evaluate several act_fns (name -> fn(cur_stack, target_shot)->twist) on the SAME held-out
    target setups. The target is rendered through each env (same pipeline as its observations).
    Returns per-name start/final similarity, success (final >= start + 0.02), and the pose/angle
    error to the target (a diagnostic; pose is not the objective)."""
    grender = grender or GoalRender()
    acc = {k: {"start": [], "final": [], "succ": [], "pos": [], "ang": []} for k in act_fns}
    for ep in range(n_episodes):
        rng = np.random.default_rng(seed + ep)                  # same setup for every policy
        Wfv, T_tgt, T_start = sample_setup(lv, _target_of(lv), rng, **setup_kw)
        for name, fn in act_fns.items():
            env = _make_env(lv, Wfv, T_start, grender, np.random.default_rng(seed + ep))
            obs, _i = env.reset()
            tgt_shot = env.render_at(T_tgt)                     # SAME cfg draw as obs
            hist = deque([_prep(obs, image_size)] * history, maxlen=history)
            s0 = similarity.score(obs, tgt_shot); s = s0
            for _ in range(n_steps):
                obs, _i = env.step(fn(np.stack(hist, 0), tgt_shot))
                hist.append(_prep(obs, image_size))
                s = similarity.score(obs, tgt_shot)
            pe, ae = _pose_error(env.world_from_probe, T_tgt)
            env.close()
            d = acc[name]
            d["start"].append(s0); d["final"].append(s); d["succ"].append(s >= s0 + 0.02)
            d["pos"].append(pe); d["ang"].append(ae)
    return {name: {"start_sim": float(np.mean(d["start"])), "final_sim": float(np.mean(d["final"])),
                   "success_rate": float(np.mean(d["succ"])), "pos_err_mm": float(np.mean(d["pos"])),
                   "ang_err_deg": float(np.mean(d["ang"]))} for name, d in acc.items()}


def _target_of(lv):
    from . import volume as V
    return V.GALLBLADDER if lv.has(V.GALLBLADDER) else (V.LIVER if lv.has(V.LIVER) else V.SOFT_TISSUE)


def load_goal_policy(ckpt_path, device=None):
    """Load a trained goal policy; return (act_fn(cur_stack, target_shot)->twist, history, size)."""
    import torch
    from .policy import GoalPolicy
    from .render_lotus import get_device
    dev = device or get_device()
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    net = GoalPolicy.build(history=ck["history"]).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    mt = torch.as_tensor(MAX_TWIST, dtype=torch.float32, device=dev)
    size = ck["image_size"]

    def act(hist_stack, target_shot):
        cur = torch.from_numpy(hist_stack[None]).to(dev)
        tgt = torch.from_numpy(_prep(target_shot, size)[None, None]).to(dev)
        with torch.no_grad():
            return (mt * torch.tanh(net(cur, tgt))[0]).cpu().numpy()

    return act, ck["history"], size
