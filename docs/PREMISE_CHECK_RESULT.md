# Per-tissue RL premise check — result (Section 4 of the handoff)

**Question (one-shot, non-RL de-risk):** does a **real-trained** organ recognizer `J` recognize the
correct organ in a sim frame textured by **SonoSPADE** (per-tissue-conditioned) much better than in a
frame textured by a **global recolor** (CUT / CycleGAN)? If yes, the organ-conditioned RL task is
de-risked; if no, don't build it.

Harness: `scripts/premise/` (`common.py`, `train_organ_J.py`, `eval_premise.py`, `aggregate.py`,
`gen_organ_sets.py`, `structure_control.py`, `make_figure.py`). Run with
`PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=<worktree>/src .venv/bin/python`.

## Design (honesty guardrails from Section 3)
- `J` = compact organ U-Net trained **only** on the 60 labeled real Kaggle Vitale frames
  (`data/eval/kaggle_vitale_test.npz`), 6 classes {bg, liver, gallbladder, vessel, kidney, spleen}.
  It never sees sim or SonoSPADE output → **non-circular**. Strong intensity augmentation so it judges
  per-tissue texture, not global brightness.
- Primary metric = per-organ IoU/recall/precision **inside the organ's true label region** on held-out
  sim content; the comparison across translators is on identical content/geometry per frame (only the
  texture stage differs). Held-out **real** IoU is the in-domain ceiling; the "predict-organ-everywhere"
  IoU is the flooding null.
- ≥3 seeds; two J-training regimes (full-frame real; fan-cropped real to bridge the sector-border gap).

## Critical fix found by adversarial review
An adversarial-review workflow (4 independent lenses + verification) found that the CUT arm was being
scored with the **wrong checkpoint** `data/dataset/texture_cut.pt` (trained toward a *different* real
distribution, `data/real_us_crop`; ~2× too bright, ~2.4× too little variance vs real Kaggle) while
CycleGAN/SonoSPADE used the coherent Kaggle-pool `exp_real_*` batch — violating guardrail #3. **The
handoff doc itself (lines 74, 124) names `texture_cut.pt`.** Fixed to `data/dataset/exp_real_cut.pt`.
This removed a spurious "CUT beats SonoSPADE" flip. All numbers below use the fair CUT.

## Result (3 seeds, mean ± std) — see `outputs/premise/premise_summary.png`

**LIVER — SonoSPADE robustly wins (every seed, both regimes, IoU and precision):**

| regime | metric | physics | CUT | CycleGAN | **SonoSPADE** | real ceiling |
|---|---|---|---|---|---|---|
| full-frame J | IoU | 0.132 | 0.069 | 0.146 | **0.270 ± 0.052** | 0.487 |
| full-frame J | precision | 0.673 | 0.555 | 0.686 | **0.787** | — |
| fan-crop J | IoU | 0.270 | 0.214 | 0.138 | **0.477 ± 0.119** | 0.511 |
| fan-crop J | precision | 0.660 | 0.646 | 0.659 | **0.759** | — |

SonoSPADE approaches the real ceiling and beats both recolors on **precision** too (so it is not a
liver-flooding artifact; all arms sit far below the flood null 0.69).

**NON-DOMINANT ORGANS (kidney, spleen, vessel, gallbladder) — untestable:** ~0 IoU for **every**
translator including SonoSPADE, even on a deliberately prominent spleen (28–40 % of frame, rendered via
`gen_organ_sets.py`). Real ceilings are 0.2–0.7, so `J` *can* recognize these organs on real data — it
just does not transfer to any sim rendering of them. This is a **J/data limitation, not a
SonoSPADE-specific failure**, driven by: (i) 4–10 real training frames per rare organ; (ii) tiny sim
organs + fan-vs-linear geometry gap; (iii) the physics acoustic model renders liver (μ₀=0.40) and spleen
(μ₀=0.42) near-identically, so no translator can produce a spleen a real recognizer distinguishes.

**Structure control (`structure_control.py`):** edge-IoU vs the physics render — SonoSPADE 0.41, CUT 0.19,
CycleGAN 0.13. SonoSPADE preserves anatomy *best* (its SPADE label-conditioning anchors structure), so
the clean "tie on structure, diverge on appearance" double dissociation does **not** hold — SonoSPADE's
liver advantage is structure **and** appearance entangled, not pure per-tissue appearance.

## Can we add kidney / spleen / gallbladder? (reward-viability gate)

To strengthen a paper with more than one organ, each organ must first clear the same gate liver
passed: a REAL-trained recognizer must identify SonoSPADE's rendering of it far better than a global
recolor's. We attacked this hard: rendered organ-PROMINENT sim sets (`gen_organ_sets.py --min-frac`)
and trained a rare-organ-BALANCED `J` (`train_organ_J.py --balance`, oversampling + class weights).
The balanced `J` recognizes these organs **well on REAL data**, yet **~0 on sim for every translator**:

| organ | balanced-J REAL ceiling (IoU) | prominent-sim best translator | SonoSPADE (sim) | reward-viable? |
|---|---|---|---|---|
| **liver** | 0.56 | **SonoSPADE 0.27–0.48** | 0.27–0.48 | **YES** |
| kidney | 0.32 | cyclegan 0.003 | 0.000 | no |
| spleen | 0.77 | cut 0.038 | 0.025 | no |
| gallbladder | 0.75 | physics 0.030 | 0.009 | no |

The balanced `J` proves the recognizer is not the bottleneck (it identifies real kidney/spleen/
gallbladder fine). The bottleneck is the **simulator + translator**: LOTUS physics + the learned
translators do not reproduce real-recognizable kidney/spleen/gallbladder appearance, even when the
organ fills 10–28 % of the frame and `J` is optimized for it. Organ-specific reasons: kidney needs
internal cortico-medullary structure none of the translators synthesize; spleen is acoustically
near-identical to liver (μ₀ 0.42 vs 0.40); gallbladder is anechoic (dark) and the translators
brighten it, so SonoSPADE is actually *worse* than raw physics there. **Conclusion: kidney, spleen,
and gallbladder are NOT reward-viable — running RL for them would train on a dead (uninformative)
reward and cannot show a SonoSPADE win. Liver is the only reward-viable organ.**

## Liver acquisition RL (organ-conditioned task; `scripts/train_organ_rl.py`)

Reward = the real-trained `J`'s liver confidence on the current *textured* view; primary metric =
ground-truth liver fraction from the true label. Same `J`, same warm-start, same held-out setups
across conditions. The task needs a rejection-sampled PARTIAL-view start band (liver 12–40 % in view):
liver is so dominant that starts are otherwise either trivial (liver already in view) or unlearnable
(liver out of view gives no directional cue — partial observability).

Result (3 seeds, mean ± std; `outputs/organ_rl/`, aggregate via `agg_organ_rl.py`):

| condition | final ground-truth liver | J-reward (2nd) |
|---|---|---|
| physics | 0.142 ± 0.005 | 0.198 |
| CUT | 0.144 ± 0.001 | 0.178 |
| CycleGAN | 0.142 ± 0.003 | 0.163 |
| **SonoSPADE** | **0.152 ± 0.004** | **0.323** |

SonoSPADE reaches the highest ground-truth liver acquisition and does so **seed-robustly — higher in
3/3 seeds** vs each baseline — but by a **small margin** (+0.008–0.010, ~1 % of the frame), while its
reward signal is ~2× stronger (0.323 vs 0.16–0.20). The honest reading: the per-tissue-correct reward
*is* a much cleaner training signal, but on this task the gain is small because the privileged
imitation warm-start plus liver dominance already nearly solve it, leaving little headroom for the
reward-quality difference to act. This is a modest, seed-robust positive — not a dramatic win.

## Multi-organ evidence: lean on the paper's existing per-organ texture metrics

The RL demonstration is single-organ (liver) by necessity. The MULTI-organ per-tissue claim is already
carried, honestly, by the paper's existing per-organ texture metrics (`paper/sonospade/results_macros.tex`,
reproduced by `scripts/eval_texture.py` / `run_sonospade_experiment_real.sh`) — these need no recognition
or RL and hold across the label scheme:
- **Label-swap locality** — SonoSPADE **26.3** vs **0.0** for every label-free translator (physics, CUT,
  CycleGAN, CycleGAN-SC). This is the clean per-tissue proof: only SonoSPADE changes an organ's texture
  when you swap its label, i.e. it synthesizes per-tissue-DISTINCT appearance across organs.
- **Organ-W1** (per-organ intensity-distribution distance to real) — SonoSPADE 0.119, best among the
  learned translators (CUT 0.246, CycleGAN 0.266).
- **Structure preservation** (edge-IoU) — SonoSPADE 0.41 vs CUT 0.17, CycleGAN 0.11.

## Verdict
- **Per-tissue synthesis is real and multi-organ** — carried by locality (26.3 vs 0) + organ-W1 across
  organs (existing metrics).
- **In-the-loop utility is demonstrated for liver:** with a real-trained reward judge, SonoSPADE's
  per-tissue-correct liver is recognized robustly (IoU 0.27–0.48 vs global-recolor 0.07–0.21) and yields
  the highest ground-truth liver acquisition, seed-robustly (3/3), though by a modest margin on this
  imitation-dominated task.
- **It does NOT extend to kidney/spleen/gallbladder as RL tasks:** a real recognizer cannot identify any
  translator's rendering of them on sim (reward-viability table above), so the reward is dead. This is a
  simulator/translator limitation (not the recognizer — a balanced J identifies them fine on real data),
  and is stated as a scoped limitation / future work, not hidden.
- **Bottom line:** honest single-organ in-the-loop win (liver) + robust multi-organ per-tissue-control
  metrics (existing) — no fabricated multi-organ RL sweep.
