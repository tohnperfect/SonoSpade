"""contracts.py: the single source of truth for the cross-repo bridge.

Pins every interface value in one no-torch module so the loaders, model, training, and
eval all speak the same representation. Enforced by tests/ before model/train are trusted.

Hard rules encoded structurally here (see plan/implementation_plan.html):
  HR1  never freeze normalization on sim   -> NORM_POLICY, NormStats.source, load_norm_stats
  HR2  never apply P during sim training    -> apply_P(domain) raises unless domain == "real"
  HR4  mask valid/NaN, never zero-fill      -> valid_eff is derived in data.py, not here
  HR5  tilt == fan; 5 train / 6 session     -> CLASSES_5 / SESSION_CLASSES_6, DOMINANT_AXIS['tilt']
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ======================================================================================
# Channels, classes, timing
# ======================================================================================
CHANNELS_TWIST = ["vx", "vy", "vz", "wx", "wy", "wz"]   # frozen 6-ch target order
CHANNELS_REL6 = ["x", "y", "z", "rx", "ry", "rz"]       # aux input (npz key `seqs`)
CHANNELS_INPUT = CHANNELS_TWIST + CHANNELS_REL6         # 12-ch model input order
IN_CHANNELS = 12                                        # z(twist 6) (+) z(rel6 6)
TARGET_CHANNELS = 6                                     # twist only

CLASSES_5 = ["slide", "rock", "tilt", "rotate", "compress"]   # tilt == fan; single-action npz
SESSION_CLASSES_6 = CLASSES_5 + ["hold"]                       # sessions / recognizer output
N_CLASSES = 6                                                  # model is ALWAYS 6-way
HOLD_IDX = 5

DT_C = 0.1                                              # 10 Hz canonical step (seconds)
OUT_RATE_HZ = 10.0

# dominant_axis keyed by literal 'tilt' (NOT 'fan'); matches the on-disk meta exactly (HR5)
DOMINANT_AXIS = {"slide": "vx", "rock": "wy", "tilt": "wx", "rotate": "wz", "compress": "vz"}
SWAP_ROCK_FAN = True

# sha256 of the vendored se3.py; equals the sim data's meta.geometry_hash (the sim vendored
# probe_pose/geometry.py verbatim and stored its file hash). Guards copy-drift AND provenance.
GEOMETRY_SHA256 = "bc784f6ec5edeaaae5a480ca4518f9587bebab4a36edda21dd274bb43cb7cb72"

# Normalization policy (HR1): provisional sim z-score now; rig freezes robust median/MAD later.
NORM_POLICY = "provisional_sim_now__frozen_real_later"
CLIP_SIGMA = 5.0                                        # clamp normalized inputs to +/- this

# Anticipation / eval constants
TTNB_UNITS = "frames"                                   # multiply by DT_C for seconds
NEXT_ID_NONE = -1                                       # next_maneuver_id sentinel (no next)
ANTICIPATION_TAUS_S = (0.25, 0.5, 1.0)                  # seconds (=> 2 or 3, 5, 10 frames)
F1_KS = (10, 25, 50)                                    # segmental F1 IoU thresholds (%)
ECE_BINS = 15
ECE_TARGET = 0.05

# Teacher / Phase-2 export (heads exist from day one; export itself is deferred)
LATENT_D = 64                                           # penultimate latent dim (in [64,128])
KD_TEMP_RANGE = (2.0, 4.0)                              # distillation temperature (NOT T_cal)
H_FRAMES = (5, 10)                                      # action-chunk horizons (~0.5 s, 1.0 s)

# ======================================================================================
# Paths (expanded at call time; never hardcode a user home)
# ======================================================================================
_SIM_DIR = "~/Codes/probe_action_sim_rl/experiments/action_recognition"
SIM_TRAIN_NPZ = os.path.join(_SIM_DIR, "action_train.npz")
SIM_SESSIONS_NPZ = os.path.join(_SIM_DIR, "action_test_sessions.npz")
SIM_META_JSON = os.path.join(_SIM_DIR, "action_dataset_meta.json")
# Rig's P contract (real-only). NOTE: our teacher export writes teacher_feature_spec.json,
# a DIFFERENT file, so we never clobber this one.
RIG_FEATURE_SPEC = "~/Codes/probe_pose_from_apriltags/feature_spec.json"
RIG_NORM_STATS = "~/Codes/probe_pose_from_apriltags/twist_norm_stats.json"  # may not exist yet

# Real training CSV (probe_6dof_training.csv) column schema, current 24 cols + PENDING.
REAL_CSV_COLUMNS = [
    "frame_index", "timestamp", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw",
    "roll_deg", "pitch_deg", "yaw_deg", "vx", "vy", "vz", "wx", "wy", "wz",
    "n_tags", "cameras_used", "source", "disagree_mm", "disagree_deg", "lost",
]
REAL_CSV_PENDING_COLUMNS = ["recovery_jump", "reinitialized", "usable_flag", "trust_weight"]


def _expand(p: str) -> str:
    return os.path.expanduser(str(p))


# ======================================================================================
# se3 parity / geometry hash
# ======================================================================================
def se3_source_sha256() -> str:
    """sha256 of the vendored se3.py on disk."""
    path = Path(__file__).with_name("se3.py")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_geometry_parity() -> None:
    """Guard against se3 copy-drift: sha256(se3.py) must equal GEOMETRY_SHA256."""
    got = se3_source_sha256()
    if got != GEOMETRY_SHA256:
        raise AssertionError(
            f"se3.py sha256 {got} != GEOMETRY_SHA256 {GEOMETRY_SHA256}; vendored copy drifted."
        )


# ======================================================================================
# Meta loading (asserts the data was produced with the expected geometry/timing)
# ======================================================================================
def load_meta(src) -> dict:
    """Load the dataset meta from an npz path, an npz 'meta' entry, or a JSON path.

    Accepts: a path to action_dataset_meta.json, a path to an .npz (reads its 'meta'),
    or an already-loaded 0-d/str meta object. Asserts the cross-repo invariants.
    """
    if isinstance(src, (str, os.PathLike)):
        p = _expand(src)
        if str(p).endswith(".npz"):
            raw = np.load(p, allow_pickle=True)["meta"]
            meta = json.loads(raw.item() if hasattr(raw, "item") else str(raw))
        else:
            with open(p) as f:
                meta = json.load(f)
    elif isinstance(src, np.ndarray):
        meta = json.loads(src.item())
    elif isinstance(src, dict):
        meta = src
    else:
        meta = json.loads(str(src))

    assert meta.get("geometry_hash") == GEOMETRY_SHA256, (
        f"meta.geometry_hash {meta.get('geometry_hash')} != {GEOMETRY_SHA256}"
    )
    assert float(meta.get("dt_c", DT_C)) == DT_C, f"meta.dt_c {meta.get('dt_c')} != {DT_C}"
    assert float(meta.get("out_rate_hz", OUT_RATE_HZ)) == OUT_RATE_HZ
    assert bool(meta.get("swap_rock_fan", True)) is True, "expected swap_rock_fan True"
    # R_align identity and NOT applied on disk (raw sim frame): P handles real later.
    assert meta.get("R_align_applied", False) is False, "sim data should be in the raw sim frame"
    return meta


# ======================================================================================
# Axis map P (rig -> sim), REAL DATA ONLY (HR2)
# ======================================================================================
def load_P(path: str = RIG_FEATURE_SPEC):
    """Load the signed axis permutation R_rig_to_sim. REAL DATA ONLY.

    Returns (P (3,3) float64, spec dict). Asserts det == +1 (proper rotation). P is applied
    to REAL twist/rel6 so real data lands in the sim frame; it is NEVER touched on sim (HR2).
    Signs are provisional (frozen_for_phase2 == False) but the permutation is firm, which is
    all Phase-1 recognition needs.
    """
    with open(_expand(path)) as f:
        spec = json.load(f)
    P = np.asarray(spec["R_rig_to_sim"], dtype=np.float64)
    assert P.shape == (3, 3), f"R_rig_to_sim must be 3x3, got {P.shape}"
    det = float(np.linalg.det(P))
    assert abs(det - 1.0) < 1e-9, f"P det {det} != +1 (must be a proper rotation)"
    return P, spec


def apply_P(arr: np.ndarray, P: np.ndarray, *, domain: str) -> np.ndarray:
    """Apply the rig->sim axis map to a twist (..,6) or rel6 (..,6) array. REAL DATA ONLY.

    Convention (feature_spec): v_sim = P @ v_rig, w_sim = P @ w_rig; same P for both halves.
    For row-vector arrays this is arr[..., :3] @ P.T and arr[..., 3:] @ P.T (matches twist6's
    R_align). Raises unless domain == 'real' (HR2: P must never be applied to sim).
    """
    if domain != "real":
        raise ValueError(
            f"apply_P called with domain={domain!r}; P is REAL-ONLY (HR2). Sim trains in its "
            "own frame and must never have P applied."
        )
    a = np.asarray(arr, dtype=np.float64)
    assert a.shape[-1] == 6, f"expected last dim 6 (twist/rel6), got {a.shape}"
    out = a.copy()
    out[..., :3] = a[..., :3] @ P.T
    out[..., 3:] = a[..., 3:] @ P.T
    return out


# ======================================================================================
# Normalization stats (provisional sim now; frozen real median/MAD later)
# ======================================================================================
@dataclass
class NormStats:
    """Per-channel normalization over the 12 input channels [twist(6) (+) rel6(6)].

    Tolerates BOTH conventions so the frozen real file (robust median/MAD) loads cleanly:
      method == 'zscore' -> center=mean,   scale=std
      method == 'mad'    -> center=median, scale=1.4826*MAD  (robust z-score)
    """
    center: np.ndarray                 # (12,)
    scale: np.ndarray                  # (12,)  (already includes the 1.4826 factor for MAD)
    method: str = "zscore"             # 'zscore' | 'mad'
    source: str = "sim_provisional"    # provenance; the rig freezes source='real_frozen'
    clip_sigma: float = CLIP_SIGMA
    channels: list = field(default_factory=lambda: list(CHANNELS_INPUT))

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=np.float64).reshape(IN_CHANNELS)
        self.scale = np.asarray(self.scale, dtype=np.float64).reshape(IN_CHANNELS)
        # guard against divide-by-zero on a dead channel
        self.scale = np.where(np.abs(self.scale) < 1e-8, 1.0, self.scale)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """z-normalize x (.., 12); NaN passes through as NaN (the loader masks/imputes it)."""
        return (np.asarray(x, dtype=np.float64) - self.center) / self.scale

    def invert(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=np.float64) * self.scale + self.center

    def to_json(self, path: str | None = None) -> dict:
        d = {
            "method": self.method,
            "source": self.source,
            "clip_sigma": float(self.clip_sigma),
            "channels": list(self.channels),
        }
        if self.method == "mad":
            d["median"] = self.center.tolist()
            d["mad"] = (self.scale / 1.4826).tolist()
        else:
            d["mean"] = self.center.tolist()
            d["std"] = self.scale.tolist()
        if path is not None:
            with open(_expand(path), "w") as f:
                json.dump(d, f, indent=2)
        return d

    @classmethod
    def from_json(cls, src) -> "NormStats":
        """Parse a stats dict/path tolerant of mean/std OR median/MAD (the real file)."""
        if isinstance(src, (str, os.PathLike)):
            with open(_expand(src)) as f:
                d = json.load(f)
        else:
            d = dict(src)
        method = d.get("method")
        if "median" in d and "mad" in d:
            center = np.asarray(d["median"], dtype=np.float64)
            scale = np.asarray(d["mad"], dtype=np.float64) * 1.4826
            method = method or "mad"
        elif "mean" in d and "std" in d:
            center = np.asarray(d["mean"], dtype=np.float64)
            scale = np.asarray(d["std"], dtype=np.float64)
            method = method or "zscore"
        elif "center" in d and "scale" in d:
            center = np.asarray(d["center"], dtype=np.float64)
            scale = np.asarray(d["scale"], dtype=np.float64)
            method = method or "zscore"
        else:
            raise ValueError(f"norm stats dict has no recognized keys: {list(d)}")
        return cls(
            center=center, scale=scale, method=method,
            source=d.get("source", "unknown"),
            clip_sigma=float(d.get("clip_sigma", CLIP_SIGMA)),
            channels=d.get("channels", list(CHANNELS_INPUT)),
        )


def load_norm_stats(path: str = RIG_NORM_STATS) -> dict | None:
    """Return the rig's frozen real stats dict, or None if the file does not exist yet (HR1).

    Graceful by design: the real twist_norm_stats.json is created only after rig recalibration.
    """
    p = _expand(path)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


# ======================================================================================
# Convenience
# ======================================================================================
def class_names(domain: str = "session") -> list:
    """'clip'/'train' -> the 5 single-action classes; 'session'/'recognizer' -> 6 (incl hold)."""
    if domain in ("clip", "train", "single"):
        return list(CLASSES_5)
    return list(SESSION_CLASSES_6)
