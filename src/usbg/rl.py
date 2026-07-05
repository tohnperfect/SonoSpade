"""rl.py: RL fine-tuning of the BC image policy toward a target-visibility reward.

The reward is dense and free in sim: the fraction of the imaging plane occupied by the target
organ, read from the label slice (no B-mode render needed for the reward). We fine-tune the BC
policy (which reproduces open-loop maneuvers) into a target-SEEKING policy: initialize from the
BC weights, wrap it as a Gaussian policy (BC twist = mean, learnable log-std for exploration),
roll out on randomized anatomy placements, and update with REINFORCE + a normalized-advantage
baseline. Rollouts use the fast NumPy renderer since only the label slice drives the reward.

Frames/actions stay in the BC convention: the net predicts NORMALIZED twist; we denormalize
(TwistNorm) before stepping the env.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from . import slicer as S
from .contracts_bridge import pose_from_rpy
from .placement import seat_volume_under_probe
from .eval import _prep


def target_visibility(env, target_label) -> float:
    lab = S.slice_at_pose(env.lv, env.world_from_probe, env.world_from_volume)
    return float((lab == target_label).mean())


def random_placement(lv, target_label, rng, lateral_m=0.02, depth_m=0.05):
    """A seating with a random lateral offset so the target starts partly out of view."""
    Wfv, T0 = seat_volume_under_probe(lv, target_label, standoff_m=-0.003, depth_m=depth_m)
    dx, dy = rng.uniform(-lateral_m, lateral_m, size=2)
    T0j = T0 @ pose_from_rpy(tx=float(dx), ty=float(dy))
    return Wfv, T0j


def build_gaussian_policy(bc_net, init_log_std: float = -1.5):
    """Wrap a BC net as a Gaussian policy over NORMALIZED twist (mean = BC output)."""
    import torch
    import torch.nn as nn

    class _GaussPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = bc_net
            self.log_std = nn.Parameter(torch.full((6,), float(init_log_std)))

        def forward(self, img, man):
            return self.net(img, man)["twist"], self.log_std

        def act(self, img, man, deterministic=False):
            mean, log_std = self(img, man)
            if deterministic:
                return mean, None
            std = log_std.exp()
            a = (mean + std * torch.randn_like(mean)).detach()   # sampled action is fixed
            logp = (-0.5 * (((a - mean) / std) ** 2 + 2 * log_std
                           + np.log(2 * np.pi))).sum(-1)          # grad flows via mean, log_std
            return a, logp

    return _GaussPolicy()


def rollout(env, policy, tn, target_label, history, image_size, device, man_id,
            n_steps=16, deterministic=False, contact_bonus=0.02, smooth_cost=0.0005,
            force_thresh=0.5):
    """One episode. Returns dict with log_probs, rewards, and visibility trace.

    reward_t = visibility_t + contact_bonus * 1[in contact] - smooth_cost * ||action||^2
    """
    import torch
    obs, info = env.reset()
    hist = deque([_prep(obs, image_size)] * history, maxlen=history)
    v0 = target_visibility(env, target_label)
    logps, rewards, vis = [], [], [v0]
    for _ in range(n_steps):
        x = torch.from_numpy(np.stack(hist, 0)[None]).to(device)
        m = torch.tensor([man_id], dtype=torch.long, device=device)
        a, logp = policy.act(x, m, deterministic=deterministic)
        a_np = a[0].detach().cpu().numpy()
        tw = tn.denormalize(a_np)
        obs, info = env.step(tw)
        hist.append(_prep(obs, image_size))
        v = target_visibility(env, target_label)
        r = v + contact_bonus * (info["contact_force"] > force_thresh) - smooth_cost * float(a_np @ a_np)
        vis.append(v); rewards.append(r)
        if logp is not None:
            logps.append(logp[0])
    return {"log_probs": logps, "rewards": np.asarray(rewards), "visibility": np.asarray(vis),
            "v0": v0, "v_final": vis[-1], "v_mean": float(np.mean(vis))}


def discounted_returns(rewards, gamma=0.95):
    R = 0.0
    out = np.zeros_like(rewards, dtype=np.float64)
    for t in range(len(rewards) - 1, -1, -1):
        R = rewards[t] + gamma * R
        out[t] = R
    return out


def evaluate_success(env_factory, act_fn, target_label, history, image_size, device, man_id,
                     n_episodes=12, n_steps=16, seed=0):
    """Deterministic eval: mean visibility and success (final vis >= start vis) over placements."""
    from collections import deque as dq
    import numpy as np
    rng = np.random.default_rng(seed)
    finals, starts, means, succ = [], [], [], []
    for _ in range(n_episodes):
        env = env_factory(rng)
        obs, info = env.reset()
        hist = dq([_prep(obs, image_size)] * history, maxlen=history)
        v0 = target_visibility(env, target_label)
        vs = [v0]
        for _ in range(n_steps):
            tw = act_fn(np.stack(hist, 0), man_id)
            obs, info = env.step(tw)
            hist.append(_prep(obs, image_size))
            vs.append(target_visibility(env, target_label))
        env.close()
        starts.append(v0); finals.append(vs[-1]); means.append(float(np.mean(vs)))
        succ.append(vs[-1] >= v0)
    return {"mean_final_vis": float(np.mean(finals)), "mean_vis": float(np.mean(means)),
            "mean_start_vis": float(np.mean(starts)), "success_rate": float(np.mean(succ))}
