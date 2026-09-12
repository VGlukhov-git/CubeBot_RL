"""Stand up from belly contact; the actor sees only attitude and command history."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from mjbatch import Batch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class StandUpConfig:
    height_agnostic: bool = True
    min_standing_height: float = 0.045
    world_level: bool = True
    max_foot_inward: float = 0.03  # allowance along the inward direction, metres
    start_airborne: bool = True
    slope_roll_degrees: float = 15.0
    slope_pitch_degrees: float = 15.0
    dynamic_slope: bool = True
    slope_change_start: float = 2.5  # seconds after simultaneous landing
    slope_change_duration: float = 4.0
    prestep: bool = False  # optional scripted foot placement before the learned rise
    target_height: float = 0.08  # root origin above plane, metres
    command_height_offset: float = (
        0.003  # geometric setpoint margin for servo deflection
    )
    sim_dt: float = 0.001
    ctrl_dt: float = 0.02
    episode_seconds: float = 8.0
    max_speed: float = np.pi / 3
    max_acceleration: float = 4.0
    max_torque: float = 0.34
    friction: float = 1.5  # dimensionless sliding coefficient, foot-floor only
    angle_noise: float = 0.002
    hold_seconds: float = 0.5
    lateral_tolerance: float = 0.005  # free XY corridor around each initial foothold


def attitude(quat):
    """WT901-like roll/pitch in radians, from a MuJoCo wxyz quaternion."""
    w, x, y, z = quat.T
    return np.stack(
        (
            np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)),
        ),
        axis=-1,
    )


class CubebotStandUp:
    observation_size = 14
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
                c.slope_change_start,
                c.slope_change_duration,
            )
            <= 0
            or c.angle_noise < 0
        ):
            raise ValueError("Timing, servo limits and friction must be positive")
        if (
            not np.isfinite(c.command_height_offset)
            or not 0 <= c.command_height_offset <= 0.005
        ):
            raise ValueError("command_height_offset must be in 0..0.005 m")
        if not all(
            np.isfinite(v) and 0 <= v <= 25
            for v in (c.slope_roll_degrees, c.slope_pitch_degrees)
        ):
            raise ValueError("Slope limits must be finite and in 0..25 degrees")
        if not np.isfinite(c.max_foot_inward) or not 0 <= c.max_foot_inward <= 0.03:
            raise ValueError("max_foot_inward must be in 0..0.03 m")
        if (
            not np.isfinite(c.min_standing_height)
            or not 0.03 <= c.min_standing_height <= 0.07
        ):
            raise ValueError("min_standing_height must be in 0.03..0.07 m")
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
        if c.start_airborne:
            self.footholds[:, :2] += np.sign(self.footholds[:, :2]) * 0.003
            self.footholds[:, 2] += 0.005
        self.initial_height = 0.0224  # bottom collision box touches z=0
        self.initial_pose = self.solve_pose(self.initial_height)
        self.prestep_commands = np.empty((0, 12))
        self.prestep_feet = np.empty((0, 4, 3))
        self.prestep_swing = np.empty((0, 4), dtype=bool)
        self.standing_footholds = self.footholds.copy()
        if c.start_airborne:
            self._build_landing()
        elif c.prestep:
            self._build_prestep()
        posture_feet = self.standing_footholds.copy()
        if c.world_level:
            posture_feet[:, :2] -= (
                (2.0 / 3.0)
                * c.max_foot_inward
                * posture_feet[:, :2]
                / np.linalg.norm(posture_feet[:, :2], axis=-1)[:, None]
            )
        self.standing_pose = self.solve_pose(
            c.target_height + c.command_height_offset, posture_feet
        )
        self.leveling_matrix = (
            self._leveling_matrix(posture_feet) if c.world_level else None
        )
        poses = np.vstack(
            (self.initial_pose, self.standing_pose, self.prestep_commands)
        )
        # Command envelope around the feasible rise. In particular, hip yaw
        # cannot wander through its full range and sweep the feet sideways.
        # Leveling needs differential extension well beyond the old narrow corridor.
        margin = 0.8 if c.world_level else 0.08
        self.low = np.maximum(self.low, poses.min(axis=0) - margin)
        self.high = np.minimum(self.high, poses.max(axis=0) + margin)
        self.batch = Batch(m, num_sims=num_envs, num_threads=num_threads)
        self.floor = m.geom("floor").id
        self.floor_quats = self.batch.expand("geom_quat")
        self.slope_degrees = np.zeros((num_envs, 2))
        self.slope_start_degrees = np.zeros((num_envs, 2))
        self.slope_target_degrees = np.zeros((num_envs, 2))
        self.plane_rotation = np.broadcast_to(np.eye(3), (num_envs, 3, 3)).copy()
        self.qpos = self.batch.bind("qpos")
        self.qvel = self.batch.bind("qvel")
        self.ctrl = self.batch.bind("ctrl")
        self.geom_xpos = self.batch.bind("geom_xpos")
        self.command = np.zeros((num_envs, 12))
        self.velocity = np.zeros_like(self.command)
        self.steps = np.zeros(num_envs, dtype=int)
        self.hold = np.zeros(num_envs, dtype=int)
        self.reset()

    def _leveling_matrix(self, posture_feet):
        """IMU roll/pitch -> differential leg extension, from nominal geometry.

        Construction uses the model; runtime feedback uses only measured attitude
        and previous commands, not joint encoders or the true plane angle.
        """
        data = mujoco.MjData(self.model)
        data.qpos[self.qadr] = self.standing_pose
        mujoco.mj_forward(self.model, data)
        jacobian = np.zeros((4, 12))
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros_like(jacp)
        for i, geom in enumerate(self.feet):
            mujoco.mj_jac(
                self.model,
                data,
                jacp,
                jacr,
                data.geom_xpos[geom],
                self.model.geom_bodyid[geom],
            )
            jacobian[i] = jacp[2, self.dadr]
        return (
            3.0
            * np.linalg.pinv(jacobian)
            @ np.stack((posture_feet[:, 1], -posture_feet[:, 0]), axis=-1)
        )

    def foothold_error(self, feet, anchors, inward):
        """Penalize travel beyond the inward region, allowing different inward directions.

        A foot may move by at most `inward` metres and must not increase its
        distance from the nominal body centre. Contact slip is penalized separately.
        """
        displacement = np.linalg.norm(feet[..., :2] - anchors[..., :2], axis=-1)
        old_radius = np.linalg.norm(anchors[..., :2], axis=-1)
        new_radius = np.linalg.norm(feet[..., :2], axis=-1)
        travel = old_radius - new_radius
        excess = np.maximum(displacement - inward, 0.0)
        outward = np.maximum(-travel, 0.0)
        return np.hypot(excess, outward), travel

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
            np.clip(start, self.low + 1e-8, self.high - 1e-8),
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

    def _build_landing(self):
        """All four feet start in air and move together to the support surface."""
        destination = self.footholds.copy()
        destination[:, :2] += np.sign(destination[:, :2]) * 0.003
        destination[:, 2] = 0.008
        commands, targets, swings = [], [], []
        for phase in np.linspace(0, 1, max(2, round(1.0 / self.config.ctrl_dt))):
            smooth = phase**3 * (10 - 15 * phase + 6 * phase**2)
            target = self.footholds + smooth * (destination - self.footholds)
            commands.append(self.solve_pose(self.initial_height, target))
            targets.append(target)
            swings.append(np.full(4, phase < 1, dtype=bool))
        for _ in range(max(1, round(0.4 / self.config.ctrl_dt))):
            commands.append(commands[-1].copy())
            targets.append(destination.copy())
            swings.append(np.zeros(4, dtype=bool))
        self.prestep_commands = np.asarray(commands)
        self.prestep_feet = np.asarray(targets)
        self.prestep_swing = np.asarray(swings)
        self.standing_footholds = destination

    def plane_points(self, points):
        """World coordinates -> each simulation's local support plane coordinates."""
        return np.einsum("nfi,nij->nfj", points, self.plane_rotation)

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

    def _set_surface_slopes(self, ids, slopes):
        """Update each selected model's static plane orientation."""
        rotation = Rotation.from_euler("xy", slopes, degrees=True)
        quats = rotation.as_quat()[:, [3, 0, 1, 2]]
        self.plane_rotation[ids] = rotation.as_matrix()
        self.slope_degrees[ids] = slopes
        self.floor_quats[ids, self.floor] = quats
        self.batch.set_const(ids)
        self.batch.forward(ids)
        return quats

    def _advance_surfaces(self):
        """Smoothly rotate the support after the robot has had time to stand."""
        if not self.config.dynamic_slope:
            return
        elapsed = (self.steps - len(self.prestep_commands)) * self.config.ctrl_dt
        phase = np.clip(
            (elapsed - self.config.slope_change_start)
            / self.config.slope_change_duration,
            0.0,
            1.0,
        )
        moving = (phase > 0) & (phase < 1)
        finishing = (phase >= 1) & np.any(
            self.slope_degrees != self.slope_target_degrees, axis=-1
        )
        ids = np.flatnonzero(moving | finishing)
        if not len(ids):
            return
        smooth = phase[ids] ** 3 * (10 - 15 * phase[ids] + 6 * phase[ids] ** 2)
        slopes = self.slope_start_degrees[ids] + smooth[:, None] * (
            self.slope_target_degrees[ids] - self.slope_start_degrees[ids]
        )
        self._set_surface_slopes(ids, slopes)

    def reset(self, indices=None, slopes=None, slope_targets=None):
        ids = (
            np.arange(self.num_envs)
            if indices is None
            else np.asarray(indices, dtype=np.int32)
        )
        if slopes is None:
            limits = [self.config.slope_roll_degrees, self.config.slope_pitch_degrees]
            slopes = self.rng.uniform(-np.asarray(limits), limits, (len(ids), 2))
        slopes = np.asarray(slopes, dtype=float)
        if (
            slopes.shape != (len(ids), 2)
            or not np.isfinite(slopes).all()
            or (abs(slopes) > 25).any()
        ):
            raise ValueError(
                "slopes must be (selected_envs, 2), roll/pitch in ±25 degrees"
            )
        if slope_targets is None:
            limits = [self.config.slope_roll_degrees, self.config.slope_pitch_degrees]
            slope_targets = self.rng.uniform(-np.asarray(limits), limits, (len(ids), 2))
        slope_targets = np.asarray(slope_targets, dtype=float)
        if (
            slope_targets.shape != (len(ids), 2)
            or not np.isfinite(slope_targets).all()
            or (abs(slope_targets) > 25).any()
        ):
            raise ValueError(
                "slope_targets must be (selected_envs, 2), roll/pitch in ±25 degrees"
            )
        self.slope_start_degrees[ids] = slopes
        self.slope_target_degrees[ids] = slope_targets
        quats = self._set_surface_slopes(ids, slopes)
        self.batch.reset(ids)
        self.qpos[ids] = self.model.qpos0
        self.qpos[ids, :3] = self.plane_rotation[ids, :, 2] * self.initial_height
        self.qpos[ids, 3:7] = quats
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
        self._advance_surfaces()
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
        old_feet = self.plane_points(self.geom_xpos[:, self.feet])
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
        height = np.einsum("ni,ni->n", self.qpos[:, :3], self.plane_rotation[:, :, 2])
        body_rotation = Rotation.from_quat(self.qpos[:, [4, 5, 6, 3]]).as_matrix()
        relative_rotation = np.einsum(
            "nji,njk->nik", self.plane_rotation, body_rotation
        )
        relative_quat = Rotation.from_matrix(relative_rotation).as_quat()[
            :, [3, 0, 1, 2]
        ]
        world_angles = attitude(self.qpos[:, 3:7])
        angles = world_angles if c.world_level else attitude(relative_quat)
        feet = self.plane_points(self.geom_xpos[:, self.feet])
        contact = feet[:, :, 2] <= 0.009
        slip = np.linalg.norm(
            (feet[:, :, :2] - old_feet[:, :, :2]) / c.ctrl_dt, axis=-1
        )
        drift = np.linalg.norm(feet[:, :, :2] - desired_feet[:, :, :2], axis=-1)
        allowance = np.where(
            preparing, 0.0, c.max_foot_inward if c.world_level else 0.0
        )[:, None]
        corridor_error, inward_travel = self.foothold_error(
            feet, desired_feet, allowance
        )
        upright = np.exp(-8 * np.sum(angles**2, axis=-1))
        height_score = np.exp(-(((height - c.target_height) / 0.02) ** 2))
        progress = np.clip(
            (height - self.initial_height) / (c.target_height - self.initial_height),
            0,
            1,
        )
        stable = (
            (~preparing)
            & (
                (height >= c.min_standing_height)
                if c.height_agnostic
                else (abs(height - c.target_height) < 0.008)
            )
            & (np.max(abs(angles), axis=-1) < 0.12)
            & contact.all(axis=-1)
            & (
                (np.max(corridor_error, axis=-1) <= c.lateral_tolerance)
                | (not c.world_level)
            )
            & (np.max(slip, axis=-1) < 0.02)
            & (np.linalg.norm(self.qvel[:, :6], axis=-1) < 0.15)
        )
        self.hold = np.where(stable, self.hold + 1, 0)
        success = self.hold * c.ctrl_dt >= c.hold_seconds
        # A soft vertical corridor: penalize XY drift even while a foot is airborne.
        # Huber cost keeps growing outside the corridor; there is no flat escape region.
        excess = np.maximum(corridor_error - c.lateral_tolerance, 0) / 0.01
        lateral_cost = np.mean(
            np.where(excess < 1, 0.5 * excess**2, excess - 0.5), axis=-1
        )
        raised = np.clip(
            (height - self.initial_height)
            / (c.min_standing_height - self.initial_height),
            0.0,
            1.0,
        )
        posture_reward = (
            (3.0 * raised + 0.2 * height_score) * upright
            if c.height_agnostic
            else 3 * height_score * upright + progress * upright
        )
        reward = (
            posture_reward
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
            | ((~preparing) & (np.max(corridor_error, axis=-1) > 0.04))
        )
        truncated = (
            self.steps - len(self.prestep_commands)
        ) * c.ctrl_dt >= c.episode_seconds
        reward -= 5 * terminated
        info = {
            "slope_degrees": self.slope_degrees.copy(),
            "slope_target_degrees": self.slope_target_degrees.copy(),
            "preparing": preparing,
            "world_tilt_degrees": np.rad2deg(np.max(abs(world_angles), axis=-1)),
            "corridor_error": np.mean(corridor_error, axis=-1),
            "foot_inward": np.mean(inward_travel, axis=-1),
            "height": height,
            "success": success,
            "stable": stable,
            "upright_reward": upright,
            "posture_reward": posture_reward,
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
