"""mujoco_env.py: a probe pressed against a patient surface, rendering the B-mode it sees.

A minimal MuJoCo (CPU) environment for Phase 3. The probe is a dynamic body welded to a mocap
target (the standard teleoperation pattern): we drive the mocap each step with a 6-DOF twist
action, the weld pulls the probe there, and contact with the patient surface pushes back so the
normal force is a real, sane signal (a purely kinematic mocap probe against a static box would
report zero force, since neither body has DOFs). The observation is the LOTUS B-mode rendered
at the probe's ACTUAL pose (which lags the target when pressing), so image and force are
consistent.

Frames: the patient skin is a box whose top face is at world z = 0; tissue is below. The probe
starts just above the skin, imaging axis (+z probe) pointing down into the patient. The CT is
seated under the probe by placement.seat_volume_under_probe (axis-aligned start).
"""
from __future__ import annotations

import numpy as np

from . import slicer as S
from . import render_lotus as R
from .contracts_bridge import se3_exp, DT_C


def _mjcf(patient_half=(0.12, 0.15, 0.08), probe_half=(0.008, 0.008, 0.012),
          weld_solref=(0.02, 1.0)) -> str:
    """Build the model XML. Patient box top face at z=0; probe welded to a mocap target."""
    px, py, pz = patient_half
    bx, by, bz = probe_half
    return f"""
<mujoco model="us_probe_gym">
  <option timestep="0.002" gravity="0 0 0" integrator="implicitfast"/>
  <compiler autolimits="true"/>
  <default>
    <geom friction="1 0.05 0.001" solref="0.01 1" solimp="0.9 0.95 0.001"/>
  </default>
  <worldbody>
    <light pos="0 0 0.5" dir="0 0 -1"/>
    <!-- patient: static box, top (skin) at z=0, tissue below -->
    <geom name="patient" type="box" pos="0 0 {-pz}" size="{px} {py} {pz}"
          rgba="0.8 0.6 0.55 1" solref="0.02 1" solimp="0.8 0.9 0.001"/>
    <!-- mocap target the probe is welded to -->
    <body name="target" mocap="true" pos="0 0 0.02">
      <geom type="box" size="0.002 0.002 0.002" contype="0" conaffinity="0"
            rgba="0 1 0 0.3"/>
    </body>
    <!-- probe: dynamic body with a free joint. The imaging face is at the body origin
         (transducer face centre, +z probe images into tissue); the housing geom sits BEHIND
         the face (local -z) so the face just touches the skin at contact, no pre-penetration. -->
    <body name="probe" pos="0 0 0.02">
      <freejoint name="probe_free"/>
      <inertial pos="0 0 0" mass="0.05" diaginertia="1e-5 1e-5 1e-5"/>
      <geom name="probe_tip" type="box" size="{bx} {by} {bz}" pos="0 0 {-bz}"
            rgba="0.2 0.4 0.9 1"/>
    </body>
  </worldbody>
  <equality>
    <weld name="probe_weld" body1="probe" body2="target"
          solref="{weld_solref[0]} {weld_solref[1]}" solimp="0.9 0.95 0.001"/>
  </equality>
</mujoco>
"""


def _mat_from_T(T):
    return np.asarray(T[:3, :3], dtype=np.float64).reshape(9)


def _quat_from_mat(mat9):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat9, dtype=np.float64))
    return q                                   # (w, x, y, z)


class USProbeEnv:
    """Probe-vs-patient environment. obs = LOTUS B-mode at the probe's actual pose.

    Parameters:
      lv                : volume.LabeledVolume to image.
      world_from_volume : 4x4 placing the CT in the world (see placement.seat_volume_under_probe).
      T0                : initial world_from_probe (probe above skin, imaging down).
      renderer          : "lotus" (MPS) or "numpy".
      settle_steps      : mj_step iterations per env step (let the weld and contact settle).
    """

    def __init__(self, lv, world_from_volume, T0, *, renderer="physics", device=None,
                 settle_steps=10, cfg=None, translator=None, randomize=False, rng=None):
        import mujoco
        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(_mjcf())
        self.data = mujoco.MjData(self.model)
        self.lv = lv
        self.world_from_volume = np.asarray(world_from_volume, dtype=np.float64)
        self.T0 = np.asarray(T0, dtype=np.float64)
        self.renderer = renderer
        self.device = device or (R.get_device() if renderer in ("physics", "lotus") else "cpu")
        self.settle_steps = settle_steps
        # sim-to-real texture is a same-pipeline stage: one Renderer renders every observation (and,
        # for the goal reward, the target via render_at) through the identical cfg + translator draw.
        self._base_cfg = cfg if cfg is not None else (R.RenderConfig() if renderer == "physics"
                                                      else None)
        self.translator = translator
        self.randomize = randomize
        self._rng = rng if rng is not None else np.random.default_rng(0)
        self._renderer = R.Renderer(kind=renderer, cfg=self._base_cfg, translator=translator,
                                    device=self.device)

        self._probe_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "probe")
        self._probe_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "probe_tip")
        self._patient_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "patient")
        self._target = None            # commanded target pose (world_from_probe)
        self.reset()

    # pose helpers
    def _set_mocap(self, T):
        self.data.mocap_pos[0] = T[:3, 3]
        self.data.mocap_quat[0] = _quat_from_mat(_mat_from_T(T))

    def _place_probe(self, T):
        """Hard-set the free joint to T (used on reset so we start already at the pose)."""
        self.data.qpos[:3] = T[:3, 3]
        self.data.qpos[3:7] = _quat_from_mat(_mat_from_T(T))
        self.data.qvel[:] = 0.0

    @property
    def world_from_probe(self) -> np.ndarray:
        """The probe body's ACTUAL pose in the world (metres)."""
        T = np.eye(4)
        T[:3, 3] = self.data.xpos[self._probe_bid]
        T[:3, :3] = self.data.xmat[self._probe_bid].reshape(3, 3)
        return T

    # contact
    def contact_force(self) -> float:
        """Total normal contact force (N) between the probe tip and the patient surface."""
        mujoco = self._mj
        total = 0.0
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if self._probe_gid in (c.geom1, c.geom2) and self._patient_gid in (c.geom1, c.geom2):
                mujoco.mj_contactForce(self.model, self.data, i, f6)
                total += abs(f6[0])                    # normal component
        return float(total)

    # observation
    @property
    def cfg_ep(self):
        """The render config in force this episode (the per-episode domain-randomization draw)."""
        return self._renderer.cfg

    def render_at(self, T_world_from_probe) -> np.ndarray:
        """Render a B-mode at an arbitrary world pose through the SAME pipeline as the observation
        (same renderer, cfg draw, and translator). Used to render the goal target consistently."""
        lab = S.slice_at_pose(self.lv, np.asarray(T_world_from_probe, dtype=np.float64),
                              self.world_from_volume)
        return self._renderer.render(lab)

    def render_obs(self, **kw) -> np.ndarray:
        return self.render_at(self.world_from_probe)

    def _obs_info(self):
        return {"pose": self.world_from_probe, "target": self._target.copy(),
                "contact_force": self.contact_force()}

    # gym-style API
    def reset(self, T0=None):
        self._mj.mj_resetData(self.model, self.data)
        # draw one per-episode render config (domain randomization), held fixed for the whole
        # episode so target and every observation share it (same-pipeline rule)
        if self.randomize and self.renderer == "physics":
            self._renderer.cfg = R.domain_randomize(self._base_cfg or R.RenderConfig(), self._rng)
        T = np.asarray(T0, dtype=np.float64) if T0 is not None else self.T0
        self._target = T.copy()
        self._set_mocap(T)
        self._place_probe(T)
        self._mj.mj_forward(self.model, self.data)
        return self.render_obs(), self._obs_info()

    def step(self, twist):
        """Advance by a body-frame 6-DOF twist [vx,vy,vz,wx,wy,wz] over one DT_C.

        Integrates the commanded target pose T <- T @ exp(twist*DT_C), drives the mocap, steps
        the physics so the weld and contact settle, then renders the B-mode at the actual pose.
        """
        twist = np.asarray(twist, dtype=np.float64).reshape(6)
        self._target = self._target @ se3_exp(twist * DT_C)
        self._set_mocap(self._target)
        for _ in range(self.settle_steps):
            self._mj.mj_step(self.model, self.data)
        obs = self.render_obs()
        return obs, self._obs_info()

    def step_to_pose(self, T_world_from_probe):
        """Drive the probe toward an absolute world pose (used by scripted / demonstrator poses)."""
        self._target = np.asarray(T_world_from_probe, dtype=np.float64).copy()
        self._set_mocap(self._target)
        for _ in range(self.settle_steps):
            self._mj.mj_step(self.model, self.data)
        return self.render_obs(), self._obs_info()

    def force_at_pose(self, T_world_from_probe) -> float:
        """Hard-set the probe to a pose and return the contact force, without stepping dynamics.

        The probe has a free joint (DOFs), so mj_forward's constraint solver computes a nonzero
        normal force from any penetration. Used by dataset generation to attach a QC force to a
        kinematically rendered frame at the exact commanded pose.
        """
        self._place_probe(np.asarray(T_world_from_probe, dtype=np.float64))
        self._set_mocap(np.asarray(T_world_from_probe, dtype=np.float64))
        self._mj.mj_forward(self.model, self.data)
        return self.contact_force()

    def close(self):
        pass
