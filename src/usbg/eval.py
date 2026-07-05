"""eval.py: closed-loop rollout of the BC image policy in the MuJoCo env.

Rolls the policy forward from a base pose: at each step it stacks the last `history` rendered
B-mode frames, predicts a 6-DOF twist, and steps the env (which integrates the pose and renders
the next frame). We compare the BC policy against a random-twist and a static (hold) baseline on:

  contact_rate      : fraction of steps in contact (holds the probe on the patient)
  recognizer_agree  : does the recognizer, run on the rollout kinematics, call it the commanded
                      maneuver (i.e. did the policy actually reproduce the maneuver)
  target_visibility : mean fraction of the target organ in view along the rollout

Acceptance (handoff): rolled out, the BC policy holds contact and reproduces the maneuver more
than a random or static baseline.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from . import slicer as S
from . import render_lotus as R
from .contracts_bridge import CLASSES_5, DT_C


def _prep(obs_float01, image_size):
    from PIL import Image
    a = np.clip(np.asarray(obs_float01) * 255.0, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(a).resize((image_size, image_size), Image.BILINEAR),
                      dtype=np.float32) / 255.0


def target_visibility(env, target_label) -> float:
    lab = S.slice_at_pose(env.lv, env.world_from_probe, env.world_from_volume)
    return float((lab == target_label).mean())


def load_bc_policy(ckpt_path, device=None):
    import torch
    from .policy import BModePolicy, TwistNorm
    device = device or R.get_device()
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    net = BModePolicy.build(history=ck["history"]).to(device)
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net, TwistNorm.from_dict(ck["twist_norm"]), ck["history"], ck["image_size"], device


def bc_policy_fn(net, tn, device):
    import torch

    def fn(hist_stack, maneuver_id):
        x = torch.from_numpy(hist_stack[None]).to(device)                 # (1,h,H,W)
        m = torch.tensor([maneuver_id], dtype=torch.long, device=device)
        with torch.no_grad():
            z = net(x, m)["twist"][0].cpu().numpy()
        return tn.denormalize(z)                                          # SI twist
    return fn


def rollout(env, maneuver_id, policy_fn, n_steps, history, image_size, target_label):
    obs, info = env.reset()
    frame = _prep(obs, image_size)
    hist = deque([frame] * history, maxlen=history)
    twists, forces, vis, poses = [], [info["contact_force"]], [target_visibility(env, target_label)], []
    for _ in range(n_steps):
        tw = np.asarray(policy_fn(np.stack(hist, 0), maneuver_id), dtype=np.float64).reshape(6)
        obs, info = env.step(tw)
        hist.append(_prep(obs, image_size))
        twists.append(tw); forces.append(info["contact_force"])
        vis.append(target_visibility(env, target_label)); poses.append(env.world_from_probe)
    return {"twists": np.asarray(twists), "forces": np.asarray(forces),
            "visibility": np.asarray(vis), "poses": np.asarray(poses)}

# (recognizer-agreement / policy-comparison helpers that depended on the sibling
# probe_action_recognizer and probe_action_sim_rl repos are omitted from this standalone release.)
