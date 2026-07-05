#!/usr/bin/env python3
"""Organ-conditioned acquisition RL (the per-tissue RL task; handoff Section 6).

The probe servos to bring the TARGET organ (liver, the one organ a real recognizer transfers to --
see docs/PREMISE_CHECK_RESULT.md) into view. The per-step reward is a FROZEN real-trained organ
recognizer J's confidence that the organ is present in the CURRENT textured view -- NOT image
similarity to a target (that is the trap the handoff warns against). Each condition renders the env
through a different sim-to-real translator (physics / CUT / CycleGAN / SonoSPADE), so the agent
learns from a translator-specific reward. If SonoSPADE's per-tissue-correct texture gives a reward J
can actually drive learning from, the SonoSPADE agent should reach higher GROUND-TRUTH organ
visibility (from the true label slice, texture-independent) on held-out patients.

Honesty:
  * J is trained ONLY on real Kaggle frames (scripts/premise/train_organ_J.py) -- never on sim.
  * Primary metric = ground-truth liver fraction in the acquired view (true label), not J-reward.
  * Same J, same warm-start budget, same eval setups across all four translator conditions; >=3 seeds.

Usage:
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  PYTHONPATH=src .venv/bin/python scripts/train_organ_rl.py --condition sonospade --seed 0 \\
      --J outputs/premise/crop/J_seed0.pt --out outputs/organ_rl/sonospade_s0.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

TRAIN_CASES = ["s0011", "s0058", "s0344", "s0461", "s0477", "s0511", "s0522", "s0703",
               "s0720", "s0727", "s0777", "s1086", "s1373"]
EVAL_CASES = ["s0358", "s0513", "s0519", "s0591", "s0649", "s0762"]
CKPTS = {"physics": None, "cut": "data/dataset/exp_real_cut.pt",
         "cyclegan": "data/dataset/exp_real_cyclegan.pt",
         "sonospade": "data/dataset/exp_real_sonospade.pt"}
LIVER = 1  # canonical + J-scheme id for liver


def resolve_case(cid):
    from usbg import volume as V
    return V.load_cache(os.path.join("data/cache", f"{cid}.npz"))


class LiverReward:
    """Frozen real-trained J -> scalar 'liver in view' reward = mean liver-class probability over the
    (resized) current view. Same J for every condition, so only the translator differs."""

    def __init__(self, jckpt, device):
        import torch
        from usbg.segmenter import build_segmenter
        self.torch = torch
        self.device = device
        ck = torch.load(jckpt, map_location=device, weights_only=False)
        self.net = build_segmenter(n_classes=ck.get("n_classes", 6), base=ck.get("base", 32),
                                   depth=ck.get("depth", 4)).to(device)
        self.net.load_state_dict(ck["state"]); self.net.eval()

    def __call__(self, obs01):
        from usbg.eval import _prep
        x = _prep(obs01, 128)
        t = self.torch.from_numpy(x[None, None].astype(np.float32)).to(self.device)
        with self.torch.no_grad():
            p = self.net(t).softmax(1)[0, LIVER]
        return float(p.mean())


def gt_liver_frac(lv, T, Wfv):
    """Ground-truth liver fraction in the view at pose T (from the TRUE label slice; texture-free)."""
    from usbg import slicer as S
    from usbg.segmenter import align_label_to_image
    lab = S.slice_at_pose(lv, np.asarray(T, np.float64), Wfv)
    return float((align_label_to_image(lab, (128, 128)) == LIVER).mean())


_SEAT_CACHE = {}  # id(lv) -> (Wfv, T_tgt): the liver seat only depends on the case, not the perturbation


def _liver_seat(lv):
    from usbg.placement import seat_volume_under_probe
    k = id(lv)
    if k not in _SEAT_CACHE:
        _SEAT_CACHE[k] = seat_volume_under_probe(lv, LIVER, standoff_m=0.0, depth_m=0.05)
    return _SEAT_CACHE[k]


def sample_lowliver_setup(lv, rng, setup_kw, max_liver=0.40, min_liver=0.12, tries=60):
    """Liver is so dominant that a pose either sees it fully (trivial) or, if it is out of view, gives
    NO cue which way to servo (partial observability -> unlearnable). So we rejection-sample a START
    where liver is PARTIALLY in view -- fraction in [min_liver, max_liver] -- so the current image is
    informative about the direction to servo AND there is headroom to reach the liver-centered seat
    (~0.7). Returns (Wfv, T_tgt, T_start). Seat cached per case; only the perturbation is resampled."""
    from usbg.contracts_bridge import pose_from_rpy
    Wfv, T_tgt = _liver_seat(lv)
    lat = setup_kw.get("lateral_m", 0.06); ax = setup_kw.get("axial_m", 0.015)
    fan = setup_kw.get("fan_deg", 32.0); roll = 4.0
    best = None
    for _ in range(tries):
        perturb = pose_from_rpy(tx=rng.uniform(-lat, lat), ty=rng.uniform(-lat, lat),
                                tz=rng.uniform(-ax, ax), rx=rng.uniform(-fan, fan),
                                ry=rng.uniform(-fan, fan), rz=rng.uniform(-roll, roll))
        T_start = T_tgt @ perturb
        lf = gt_liver_frac(lv, T_start, Wfv)
        # prefer a start inside the band; else keep the one closest to the band centre
        target_c = 0.5 * (min_liver + max_liver)
        score = abs(lf - target_c) if not (min_liver <= lf <= max_liver) else -1.0
        if best is None or score < best[0]:
            best = (score, Wfv, T_tgt, T_start)
        if min_liver <= lf <= max_liver:
            break
    return best[1], best[2], best[3]


def imitation_batch_lowliver(lv, rng, history, image_size, batch, grender, setup_kw, max_liver):
    """Warm-start batch: low-liver START pose + privileged corrective twist toward the liver-centered
    pose. Rendered through the condition's translator; the policy conditions on the current view only
    (no target image), so the corrective twist is the sole supervision."""
    from usbg.eval import _prep
    from usbg import goal_rl as G
    curs, tws = [], []
    for _ in range(batch):
        Wfv, T_tgt, T_cur = sample_lowliver_setup(lv, rng, setup_kw, max_liver)
        rend = grender.renderer(grender.episode_cfg(rng))
        cur_shot = rend.render_at(lv, T_cur, Wfv)
        curs.append(np.stack([_prep(cur_shot, image_size)] * history, 0))
        tws.append(G.analytic_twist(T_cur, T_tgt))
    return np.asarray(curs, np.float32), np.asarray(tws, np.float32)


def organ_rollout(env, policy, reward_fn, zero_tgt, history, image_size, device, n_steps, effort=0.004,
                  deterministic=False):
    """REINFORCE rollout. Reward = per-step GAIN in J-liver confidence minus an effort penalty (the
    return telescopes to net improvement, crediting corrective actions)."""
    import torch
    from collections import deque
    from usbg.eval import _prep
    obs, _ = env.reset()
    hist = deque([_prep(obs, image_size)] * history, maxlen=history)
    s_prev = reward_fn(obs)
    logps, rewards = [], []
    for _ in range(n_steps):
        cur = torch.from_numpy(np.stack(hist, 0)[None]).to(device)
        a, logp = policy.act(cur, zero_tgt, deterministic=deterministic)
        obs, _ = env.step(a[0].detach().cpu().numpy())
        hist.append(_prep(obs, image_size))
        s = reward_fn(obs)
        rewards.append((s - s_prev) - effort * float(a[0].detach().cpu().numpy() @ a[0].detach().cpu().numpy()))
        s_prev = s
        if logp is not None:
            logps.append(logp[0])
    return {"log_probs": logps, "rewards": np.asarray(rewards), "j_final": s_prev}


def evaluate(policy, reward_fn, translator, grender_kind, device, history, image_size, zero_tgt,
             n_episodes, n_steps, setup_kw, max_liver, seed=999):
    """Ground-truth liver visibility (primary) + J-reward (secondary) on held-out setups, vs a
    static (no-move) baseline on the SAME setups."""
    import torch
    from collections import deque
    from usbg import goal_rl as G
    from usbg.eval import _prep
    out = {"start_gt": [], "final_gt": [], "static_gt": [], "final_j": [], "start_j": []}
    for ep in range(n_episodes):
        cid = EVAL_CASES[ep % len(EVAL_CASES)]
        lv = resolve_case(cid)
        rng = np.random.default_rng(seed + ep)
        Wfv, T_tgt, T_start = sample_lowliver_setup(lv, rng, setup_kw, max_liver)
        env = G._make_env(lv, Wfv, T_start, G.GoalRender(kind=grender_kind, translator=translator,
                                                         device=device), np.random.default_rng(seed + ep))
        obs, _ = env.reset()
        out["start_gt"].append(gt_liver_frac(lv, env.world_from_probe, Wfv))
        out["start_j"].append(reward_fn(obs))
        # static baseline: never move
        out["static_gt"].append(gt_liver_frac(lv, env.world_from_probe, Wfv))
        hist = deque([_prep(obs, image_size)] * history, maxlen=history)
        for _ in range(n_steps):
            cur = torch.from_numpy(np.stack(hist, 0)[None]).to(device)
            a, _ = policy.act(cur, zero_tgt, deterministic=True)
            obs, _ = env.step(a[0].detach().cpu().numpy())
            hist.append(_prep(obs, image_size))
        out["final_gt"].append(gt_liver_frac(lv, env.world_from_probe, Wfv))
        out["final_j"].append(reward_fn(obs))
        env.close()
    return {k: float(np.mean(v)) for k, v in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=list(CKPTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--J", default="outputs/premise/crop/J_seed0.pt")
    ap.add_argument("--imitation", type=int, default=250)
    ap.add_argument("--imitation-batch", type=int, default=12)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--episodes-per-iter", type=int, default=6)
    ap.add_argument("--n-steps", type=int, default=12)
    ap.add_argument("--history", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rl-lr", type=float, default=1e-4)
    ap.add_argument("--bc-anchor", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--lateral-m", type=float, default=0.06)
    ap.add_argument("--axial-m", type=float, default=0.015)
    ap.add_argument("--fan-deg", type=float, default=32.0)
    ap.add_argument("--start-max-liver", type=float, default=0.40,
                    help="rejection-sample START poses with liver fraction in [0.12, this] -- partially "
                         "in view so the image is informative AND there is headroom to acquire it")
    ap.add_argument("--eval-episodes", type=int, default=18)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    import torch
    from usbg import goal_rl as G
    from usbg.policy import GoalPolicy
    from usbg.texture_gan import load_translator
    from usbg.render_lotus import get_device
    device = args.device or get_device()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    translator = None if CKPTS[args.condition] is None else load_translator(CKPTS[args.condition], device)
    grender = G.GoalRender(kind="physics", translator=translator, device=device)
    reward_fn = LiverReward(args.J, device)
    setup_kw = dict(lateral_m=args.lateral_m, axial_m=args.axial_m, fan_deg=args.fan_deg)
    lv_train = [resolve_case(c) for c in TRAIN_CASES]

    net = GoalPolicy.build(history=args.history)
    policy = G.build_goal_gaussian_policy(net).to(device)
    zero_tgt = torch.zeros(1, 1, args.image_size, args.image_size, device=device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    print(f"[{args.condition} s{args.seed}] J={args.J} translator={'on' if translator else 'off'} "
          f"device={device}")

    # imitation warm-start: privileged corrective twist toward the liver-centered pose (target images
    # are zeroed -- the policy conditions on the current view only, not on a target shot).
    for it in range(args.imitation):
        lv = lv_train[it % len(lv_train)]
        cur, tw = imitation_batch_lowliver(lv, rng, args.history, args.image_size,
                                           args.imitation_batch, grender, setup_kw, args.start_max_liver)
        zt = torch.zeros(cur.shape[0], 1, args.image_size, args.image_size, device=device)
        pred = policy.mean(torch.from_numpy(cur).to(device), zt)
        loss = ((pred - torch.from_numpy(tw).to(device)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % max(1, args.imitation // 3) == 0:
            print(f"  imitation {it+1}/{args.imitation} twistMSE {loss.item():.4f}")

    ev_after_im = evaluate(policy, reward_fn, translator, "physics", device, args.history,
                           args.image_size, zero_tgt, args.eval_episodes, args.n_steps, setup_kw,
                           args.start_max_liver)
    print(f"  [after imitation] gt_liver start {ev_after_im['start_gt']:.3f} -> "
          f"final {ev_after_im['final_gt']:.3f} (static {ev_after_im['static_gt']:.3f}); "
          f"J start {ev_after_im['start_j']:.3f} -> final {ev_after_im['final_j']:.3f}")

    # RL fine-tune on the translator-specific J-liver reward (REINFORCE + BC anchor, fixed std).
    rl_opt = torch.optim.Adam(policy.net.parameters(), lr=args.rl_lr)
    rl_curve = []
    for it in range(args.iters):
        logps, advs, jfin = [], [], []
        for ei in range(args.episodes_per_iter):
            lv = lv_train[(it + ei) % len(lv_train)]
            Wfv, T_tgt, T_start = sample_lowliver_setup(lv, rng, setup_kw, args.start_max_liver)
            env = G._make_env(lv, Wfv, T_start, grender, np.random.default_rng(args.seed + it * 97 + ei))
            roll = organ_rollout(env, policy, reward_fn, zero_tgt, args.history, args.image_size,
                                 device, args.n_steps)
            env.close()
            if not roll["log_probs"]:
                continue
            ret = G.discounted_returns(roll["rewards"], args.gamma)
            logps.extend(roll["log_probs"]); advs.extend(ret.tolist()); jfin.append(roll["j_final"])
        if not logps:
            continue
        adv = torch.tensor(advs, dtype=torch.float32, device=device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        bc = torch.zeros((), device=device)
        if args.bc_anchor > 0:
            lv = lv_train[it % len(lv_train)]
            cur_b, tw_b = imitation_batch_lowliver(lv, rng, args.history, args.image_size,
                                                   args.imitation_batch, grender, setup_kw,
                                                   args.start_max_liver)
            zt = torch.zeros(cur_b.shape[0], 1, args.image_size, args.image_size, device=device)
            pred = policy.mean(torch.from_numpy(cur_b).to(device), zt)
            bc = ((pred - torch.from_numpy(tw_b).to(device)) ** 2).mean()
        loss = -(torch.stack(logps) * adv).mean() + args.bc_anchor * bc
        rl_opt.zero_grad(); loss.backward(); rl_opt.step()
        rl_curve.append(float(np.mean(jfin)) if jfin else float("nan"))
        if (it + 1) % max(1, args.iters // 5) == 0:
            print(f"  iter {it+1}/{args.iters} trainJ {np.mean(jfin):.3f} bc {float(bc.detach()):.4f}")

    ev_final = evaluate(policy, reward_fn, translator, "physics", device, args.history,
                        args.image_size, zero_tgt, args.eval_episodes, args.n_steps, setup_kw,
                        args.start_max_liver)
    print(f"  [FINAL] gt_liver start {ev_final['start_gt']:.3f} -> final {ev_final['final_gt']:.3f} "
          f"(static {ev_final['static_gt']:.3f}); J final {ev_final['final_j']:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"condition": args.condition, "seed": args.seed, "J": args.J,
               "after_imitation": ev_after_im, "final": ev_final, "rl_curve": rl_curve,
               "setup": setup_kw, "config": vars(args)}, open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
