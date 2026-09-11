"""Stand up from belly contact; the actor sees only attitude and command history."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from mjbatch import Batch
from scipy.optimize import least_squares


@dataclass(frozen=True)
class StandUpConfig:
    prestep: bool = False  # optional scripted foot placement before the learned rise
    target_height: float = 0.08  # root origin above plane, metres
    sim_dt: float = 0.002
    ctrl_dt: float = 0.02
    episode_seconds: float = 6.0
    max_speed: float = np.pi / 3
    max_acceleration: float = 4.0
    max_torque: float = 0.34
    friction: float = 1.5  # dimensionless sliding coefficient, foot-floor only
    angle_noise: float = 0.002
    hold_seconds: float = 0.5
    lateral_tolerance: float = 0.005  # free XY corridor around each initial foothold


def attitude(quat):
    """WT901-like XYZ Euler angles in radians, MuJoCo wxyz quaternion."""
    w, x, y, z = quat.T
    return np.stack(
        (
            np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)),
            np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
        ),
        axis=-1,
    )


class CubebotStandUp:
    observation_size = 15
    action_size = 12

    def __init__(self, num_envs=64, config=None, seed=0, num_threads=0):
        self.config = c = config or StandUpConfig()
        if not (0.04 <= c.target_height <= 0.085):
            raise ValueError(
                "target_height must be 0.04..0.085 m for this model's fixed footholds"
            )
        if (
            min(
                c.sim_dt,
                c.ctrl_dt,
                c.episode_seconds,
                c.max_speed,
                c.max_acceleration,
                c.friction,
                c.max_torque,
                c.hold_seconds,
                c.lateral_tolerance,
            )
            <= 0
            or c.angle_noise < 0
        ):
            raise ValueError("Timing, servo limits and friction must be positive")
        self.substeps = round(c.ctrl_dt / c.sim_dt)
        if self.substeps < 1 or not np.isclose(self.substeps * c.sim_dt, c.ctrl_dt):
            raise ValueError("ctrl_dt must be an integer multiple of sim_dt")
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.rng = np.random.default_rng(seed)
        # Absolute include path also resolves mesh assets in installed packages.
        robot = Path(__file__).resolve().parents[1] / "cubebot.xml"
        foot_names = tuple(
            n + "_foot_collision"
            for n in ("left_back", "right_back", "right_front", "left_front")
        )
        # Explicit pairs override geom mixing: sliding plus torsional friction.
        contacts = "".join(
            f'<pair name="{name}_floor" geom1="floor" geom2="{name}" '
            f'condim="4" friction="{c.friction} {c.friction} 0.02 0.0001 0.0001" '
            'solref="0.01 1" solimp="0.9 0.95 0.001"/>'
            for name in foot_names
        )
        self.model = m = mujoco.MjModel.from_xml_string(
            f'<mujoco><include file="{robot}"/><option timestep="{c.sim_dt}" '
            'integrator="implicitfast" cone="elliptic" iterations="80" impratio="1"/>'
            '<worldbody><geom name="floor" type="plane" size="2 2 .1" '
            'rgba=".7 .75 .8 1"/></worldbody>'
            f"<contact>{contacts}</contact></mujoco>"
        )
        m.actuator_forcerange[:] = [-c.max_torque, c.max_torque]
        self.joints = m.actuator_trnid[:, 0].copy()
        self.qadr = m.jnt_qposadr[self.joints]
        self.dadr = m.jnt_dofadr[self.joints]
        self.low = np.where(
            m.jnt_limited[self.joints], m.jnt_range[self.joints, 0], -np.pi
        )
        self.high = np.where(
            m.jnt_limited[self.joints], m.jnt_range[self.joints, 1], np.pi
        )
        m.actuator_ctrllimited[:] = True
        m.actuator_ctrlrange[:] = np.stack((self.low, self.high), axis=-1)
        self.feet = np.array(
            [
                m.geom(n + "_foot_collision").id
                for n in ("left_back", "right_back", "right_front", "left_front")
            ]
        )
        # Preserve the previous body/leg friction; stronger grip is foot-specific.
        m.geom_friction[:, :] = [1.2, 0.02, 0.002]
        m.geom_condim[:] = 4
        m.geom_solref[:] = [0.01, 1]
        m.opt.noslip_iterations = 5
        self.footholds = np.array(
            [
                [0.123, -0.123, 0.008],
                [0.123, 0.123, 0.008],
                [-0.123, 0.123, 0.008],
                [-0.123, -0.123, 0.008],
            ]
        )
        self.initial_height = 0.0224  # bottom collision box touches z=0
        self.initial_pose = self.solve_pose(self.initial_height)
        self.prestep_commands = np.empty((0, 12))
        self.prestep_feet = np.empty((0, 4, 3))
        self.prestep_swing = np.empty((0, 4), dtype=bool)
        self.standing_footholds = self.footholds.copy()
        if c.prestep:
            self._build_prestep()
        self.standing_pose = self.solve_pose(c.target_height, self.standing_footholds)
        poses = np.vstack(
            (self.initial_pose, self.standing_pose, self.prestep_commands)
        )
        # Command envelope around the feasible rise. In particular, hip yaw
        # cannot wander through its full range and sweep the feet sideways.
        self.low = np.maximum(self.low, poses.min(axis=0) - 0.08)
        self.high = np.minimum(self.high, poses.max(axis=0) + 0.08)
        self.batch = Batch(m, num_sims=num_envs, num_threads=num_threads)
        self.qpos = self.batch.bind("qpos")
        self.qvel = self.batch.bind("qvel")
        self.ctrl = self.batch.bind("ctrl")
        self.geom_xpos = self.batch.bind("geom_xpos")
        self.command = np.zeros((num_envs, 12))
        self.velocity = np.zeros_like(self.command)
        self.steps = np.zeros(num_envs, dtype=int)
        self.hold = np.zeros(num_envs, dtype=int)
        self.reset()

    def solve_pose(self, height, footholds=None):
        """IK only at construction; never passed to the actor as sensor data."""
        footholds = self.footholds if footholds is None else footholds
        d = mujoco.MjData(self.model)
        d.qpos[:3] = [0, 0, height]
        start = np.array(
            [0.835, 1, 1.9, -0.835, 1, 1.9, 0.835, 1, -1.9, -0.835, -1, -1.9]
        )

        def residual(q):
            d.qpos[self.qadr] = q
            mujoco.mj_forward(self.model, d)
            return (d.geom_xpos[self.feet] - footholds).ravel()

        result = least_squares(
            residual,
            start,
            bounds=(self.low, self.high),
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=500,
        )
        if np.max(np.abs(residual(result.x))) > 1e-5:
            raise ValueError(f"Unreachable pose at height {height}")
        if any(contact.dist < -1e-5 for contact in d.contact):
            raise ValueError(
                f"Pose at height {height} has penetrating collision geometry"
            )
        return result.x

    def _build_prestep(self):
        """One foot at a time, while the belly supports the body; no state teleporting."""
        commands, targets, swings = [], [], []
        placed = self.footholds.copy()
        # One second per foot, diagonally alternating, followed by landing settle.
        for foot in (2, 0, 3, 1):
            origin = placed[foot].copy()
            destination = origin.copy()
            destination[:2] += np.sign(origin[:2]) * 0.006
            for phase in np.linspace(0, 1, max(2, round(1.0 / self.config.ctrl_dt))):
                smooth = phase**3 * (10 - 15 * phase + 6 * phase**2)
                target = placed.copy()
                target[foot] = origin + smooth * (destination - origin)
                target[foot, 2] += 0.005 * np.sin(np.pi * smooth) ** 2
                commands.append(self.solve_pose(self.initial_height, target))
                targets.append(target)
                swing = np.zeros(4, dtype=bool)
                swing[foot] = 0 < phase < 1
                swings.append(swing)
            placed[foot] = destination
        for _ in range(max(1, round(0.4 / self.config.ctrl_dt))):
            commands.append(commands[-1].copy())
            targets.append(placed.copy())
            swings.append(np.zeros(4, dtype=bool))
        self.prestep_commands = np.asarray(commands)
        self.prestep_feet = np.asarray(targets)
        self.prestep_swing = np.asarray(swings)
        self.standing_footholds = placed

    @property
    def preparing(self):
        return self.steps < len(self.prestep_commands)

    def reset(self, indices=None):
        ids = (
            np.arange(self.num_envs)
            if indices is None
            else np.asarray(indices, dtype=np.int32)
        )
        self.batch.reset(ids)
        self.qpos[ids] = self.model.qpos0
        self.qpos[ids, :3] = [0, 0, self.initial_height]
        self.qpos[np.ix_(ids, self.qadr)] = self.initial_pose
        self.command[ids] = self.initial_pose
        self.ctrl[ids] = self.initial_pose
        self.velocity[ids] = 0
        self.steps[ids] = 0
        self.hold[ids] = 0
        self.batch.forward()
        return self.observation()

    def observation(self):
        angles = attitude(self.qpos[:, 3:7])
        angles += self.rng.normal(0, self.config.angle_noise, angles.shape)
        # No encoders: these are the last commanded angles, not qpos.
        return np.concatenate((angles / np.pi, self.command / np.pi), axis=-1).astype(
            np.float32
        )

    def step(self, action):
        c = self.config
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.command.shape or not np.isfinite(action).all():
            raise ValueError(f"Expected finite actions with shape {self.command.shape}")
        preparing = self.preparing.copy()
        desired_feet = np.broadcast_to(
            self.standing_footholds, (self.num_envs, 4, 3)
        ).copy()
        swing = np.zeros((self.num_envs, 4), dtype=bool)
        if preparing.any():
            frame = self.steps[preparing]
            desired_feet[preparing] = self.prestep_feet[frame]
            swing[preparing] = self.prestep_swing[frame]
            action = action.copy()
            action[preparing] = (
                self.prestep_commands[frame] - self.command[preparing]
            ) / (c.ctrl_dt * c.max_speed)
        old_feet = self.geom_xpos[:, self.feet].copy()
        old_velocity = self.velocity.copy()
        desired_velocity = np.clip(action, -1, 1) * c.max_speed
        # Bound both velocity and acceleration, including braking at joint limits.
        braking = np.sqrt(
            2
            * c.max_acceleration
            * np.maximum(
                np.where(
                    desired_velocity >= 0,
                    self.high - self.command,
                    self.command - self.low,
                ),
                0,
            )
        )
        desired_velocity = np.clip(desired_velocity, -braking, braking)
        self.velocity += np.clip(
            desired_velocity - self.velocity,
            -c.max_acceleration * c.ctrl_dt,
            c.max_acceleration * c.ctrl_dt,
        )
        old_command = self.command.copy()
        self.command[:] = np.clip(
            self.command + self.velocity * c.ctrl_dt, self.low, self.high
        )
        self.velocity[:] = (self.command - old_command) / c.ctrl_dt
        # Interpolate setpoints at physics frequency, not a staircase at 50 Hz.
        for k in range(self.substeps):
            self.ctrl[:] = (
                old_command + (self.command - old_command) * (k + 1) / self.substeps
            )
            self.batch.step()
        self.batch.forward()
        self.steps += 1
        height = self.qpos[:, 2].copy()
        angles = attitude(self.qpos[:, 3:7])
        feet = self.geom_xpos[:, self.feet]
        contact = feet[:, :, 2] <= 0.009
        slip = np.linalg.norm(
            (feet[:, :, :2] - old_feet[:, :, :2]) / c.ctrl_dt, axis=-1
        )
        drift = np.linalg.norm(feet[:, :, :2] - desired_feet[:, :, :2], axis=-1)
        upright = np.exp(-8 * np.sum(angles[:, :2] ** 2, axis=-1))
        height_score = np.exp(-(((height - c.target_height) / 0.02) ** 2))
        progress = np.clip(
            (height - self.initial_height) / (c.target_height - self.initial_height),
            0,
            1,
        )
        stable = (
            (~preparing)
            & (abs(height - c.target_height) < 0.008)
            & (np.max(abs(angles[:, :2]), axis=-1) < 0.12)
            & contact.all(axis=-1)
            & (np.max(slip, axis=-1) < 0.02)
            & (np.linalg.norm(self.qvel[:, :6], axis=-1) < 0.15)
        )
        self.hold = np.where(stable, self.hold + 1, 0)
        success = self.hold * c.ctrl_dt >= c.hold_seconds
        # A soft vertical corridor: penalize XY drift even while a foot is airborne.
        # Huber cost keeps growing outside the corridor; there is no flat escape region.
        excess = np.maximum(drift - c.lateral_tolerance, 0) / 0.01
        lateral_cost = np.mean(
            np.where(excess < 1, 0.5 * excess**2, excess - 0.5), axis=-1
        )
        reward = (
            3 * height_score * upright
            + progress * upright
            + stable.astype(float)
            - 2 * np.mean((slip / 0.1) ** 2 * contact, axis=-1)
            - lateral_cost
            - 0.5 * np.mean(~contact & ~swing, axis=-1)
            - 0.03 * np.mean(self.velocity**2, axis=-1)
            - 0.01 * np.mean(((self.velocity - old_velocity) / c.ctrl_dt) ** 2, axis=-1)
            - 0.02 * np.mean(self.qvel[:, self.dadr] ** 2, axis=-1)
            - 0.2 * np.sum(self.qvel[:, :2] ** 2, axis=-1)
        )
        # Preparation is a controller phase; reward its foot placement instead of height.
        placement_reward = 1.0 - np.mean(
            np.sum((feet - desired_feet) ** 2, axis=-1) / 0.01**2, axis=-1
        )
        reward = np.where(preparing, placement_reward, reward)
        terminated = (
            (np.max(abs(angles[:, :2]), axis=-1) > 1.0)
            | (height < 0.01)
            | ((~preparing) & (np.max(drift, axis=-1) > 0.04))
        )
        truncated = (
            self.steps - len(self.prestep_commands)
        ) * c.ctrl_dt >= c.episode_seconds
        reward -= 5 * terminated
        info = {
            "preparing": preparing,
            "height": height,
            "success": success,
            "stable": stable,
            "slip": np.mean(slip * contact, axis=-1),
            "contact_count": contact.sum(axis=-1),
            "lateral_drift": np.mean(drift, axis=-1),
            "lateral_cost": lateral_cost,
        }
        # Explicit reset lets PPO bootstrap timeouts from the terminal observation.
        return (
            self.observation(),
            reward.astype(np.float32),
            terminated,
            truncated,
            info,
        )
