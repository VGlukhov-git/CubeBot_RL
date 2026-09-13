"""GPU stand-up training: MJX physics + a CUDA PyTorch PPO policy.

This keeps the deployable ``.pt`` checkpoint format used by ``hardware.py``
and ``play.py`` while moving both simulation and neural-network work to CUDA.
The first curriculum stage intentionally uses a flat, static floor.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict
from pathlib import Path

# JAX and PyTorch share the same 12 GiB device.  Let both allocate on demand.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import torch
from jax import dlpack as jax_dlpack
from mujoco import mjx
from mujoco_playground._src import mjx_env
from torch import nn

from .environment import CubebotStandUp, StandUpConfig, attitude
from .policy import Policy


def _to_torch(value: jax.Array) -> torch.Tensor:
    """Zero-copy JAX CUDA array -> PyTorch CUDA tensor."""
    return torch.utils.dlpack.from_dlpack(value)


def _to_jax(value: torch.Tensor) -> jax.Array:
    """Zero-copy PyTorch CUDA tensor -> JAX CUDA array."""
    return jax_dlpack.from_dlpack(value.detach().contiguous())


def _roll_pitch(quat: jax.Array) -> jax.Array:
    """WT901-like world roll/pitch from a MuJoCo wxyz quaternion."""
    w, x, y, z = quat
    roll = jp.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = jp.arcsin(jp.clip(2 * (w * y - z * x), -1.0, 1.0))
    return jp.asarray([roll, pitch])


class StandUpMjx:
    """Flat-floor GPU environment with hardware-compatible observations."""

    observation_size = 14
    action_size = 12

    def __init__(self, config: StandUpConfig):
        # Reuse the tested host-side IK once at construction.  Rollouts below
        # use only MJX and never call mjbatch.
        host = CubebotStandUp(1, config, seed=0, num_threads=1)
        self.config = config
        self.model = host.model
        self.initial_pose = jp.asarray(host.initial_pose)
        self.standing_pose = jp.asarray(host.standing_pose)
        self.initial_height = float(host.initial_height)
        self.low = jp.asarray(host.low)
        self.high = jp.asarray(host.high)
        self.leveling_matrix = np.asarray(host.leveling_matrix)
        self.joints = np.asarray(host.joints)
        self.qadr = jp.asarray(host.qadr, dtype=jp.int32)
        self.dadr = jp.asarray(host.dadr, dtype=jp.int32)
        self.joint_names = [self.model.joint(int(j)).name for j in self.joints]
        self.root_qadr = int(
            self.model.jnt_qposadr[
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, "root_freejoint"
                )
            ]
        )
        self.root_dadr = int(
            self.model.jnt_dofadr[
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, "root_freejoint"
                )
            ]
        )
        self.mjx_model = mjx.put_model(self.model)
        self.n_substeps = round(config.ctrl_dt / config.sim_dt)
        self.episode_steps = round(config.episode_seconds / config.ctrl_dt)
        if self.n_substeps < 1:
            raise ValueError("ctrl_dt must be at least sim_dt")

    def reset(self, rng: jax.Array) -> mjx_env.State:
        del rng
        qpos = jp.asarray(self.model.qpos0)
        qpos = qpos.at[self.root_qadr : self.root_qadr + 3].set(
            jp.asarray([0.0, 0.0, self.initial_height])
        )
        qpos = qpos.at[self.root_qadr + 3 : self.root_qadr + 7].set(
            jp.asarray([1.0, 0.0, 0.0, 0.0])
        )
        qpos = qpos.at[self.qadr].set(self.initial_pose)
        qvel = jp.zeros(self.model.nv)
        data = mjx_env.make_data(
            self.model, qpos=qpos, qvel=qvel, ctrl=self.initial_pose
        )
        data = mjx.forward(self.mjx_model, data)
        info = {
            "command": self.initial_pose,
            "command_velocity": jp.zeros(self.action_size),
            "previous_action": jp.zeros(self.action_size),
            "step_count": jp.asarray(0, dtype=jp.int32),
        }
        metrics = {
            "height": qpos[self.root_qadr + 2],
            "tilt": jp.asarray(0.0),
            "stable": jp.asarray(0.0),
        }
        return mjx_env.State(
            data=data,
            obs=self._observation(data, info),
            reward=jp.asarray(0.0),
            done=jp.asarray(0.0),
            metrics=metrics,
            info=info,
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        c = self.config
        action = jp.clip(action, -1.0, 1.0)
        command = state.info["command"]
        old_velocity = state.info["command_velocity"]
        desired_velocity = (
            1.5 * (self.standing_pose - command) + 0.05 * action
        )
        desired_velocity = jp.clip(desired_velocity, -c.max_speed, c.max_speed)
        braking = jp.sqrt(
            2
            * c.max_acceleration
            * jp.maximum(
                jp.where(
                    desired_velocity >= 0,
                    self.high - command,
                    command - self.low,
                ),
                0.0,
            )
        )
        desired_velocity = jp.clip(desired_velocity, -braking, braking)
        velocity = old_velocity + jp.clip(
            desired_velocity - old_velocity,
            -c.max_acceleration * c.ctrl_dt,
            c.max_acceleration * c.ctrl_dt,
        )
        new_command = jp.clip(
            command + velocity * c.ctrl_dt, self.low, self.high
        )
        velocity = (new_command - command) / c.ctrl_dt
        data = mjx_env.step(
            self.mjx_model, state.data, new_command, self.n_substeps
        )

        angles = _roll_pitch(
            data.qpos[self.root_qadr + 3 : self.root_qadr + 7]
        )
        height = data.qpos[self.root_qadr + 2]
        tilt = jp.max(jp.abs(angles))
        upright = jp.exp(-8.0 * jp.sum(jp.square(angles)))
        raised = jp.clip(
            (height - self.initial_height)
            / (c.min_standing_height - self.initial_height),
            0.0,
            1.0,
        )
        height_score = jp.exp(-jp.square((height - c.target_height) / 0.02))
        stable = (
            (height >= c.min_standing_height)
            & (tilt < 0.12)
            & (jp.linalg.norm(data.qvel[:6]) < 0.15)
        )
        root_velocity_cost = jp.sum(jp.square(data.qvel[:6]))
        command_acceleration = (velocity - old_velocity) / c.ctrl_dt
        reward = (
            (3.0 * raised + 0.2 * height_score) * upright
            + stable.astype(jp.float32)
            - 0.03 * jp.mean(jp.square(velocity))
            - 0.005 * jp.mean(jp.square(command_acceleration))
            - 0.10 * root_velocity_cost
            - 0.02 * jp.mean(jp.square(action))
        )
        count = state.info["step_count"] + 1
        fallen = (tilt > 1.0) | (height < 0.01)
        timeout = count >= self.episode_steps
        done = fallen | timeout
        reward = reward - 5.0 * fallen.astype(jp.float32)
        info = {
            "command": new_command,
            "command_velocity": velocity,
            "previous_action": action,
            "step_count": count,
        }
        metrics = {
            "height": height,
            "tilt": tilt,
            "stable": stable.astype(jp.float32),
        }
        return state.replace(
            data=data,
            obs=self._observation(data, info),
            reward=reward,
            done=done.astype(jp.float32),
            metrics=metrics,
            info=info,
        )

    def _observation(self, data: mjx.Data, info: dict) -> jax.Array:
        angles = _roll_pitch(
            data.qpos[self.root_qadr + 3 : self.root_qadr + 7]
        )
        return jp.concatenate((angles / jp.pi, info["command"] / jp.pi)).astype(
            jp.float32
        )


class GpuBatch:
    """Jitted vectorization and automatic resets for StandUpMjx."""

    def __init__(self, environment: StandUpMjx, num_envs: int, seed: int):
        self.environment = environment
        self.num_envs = num_envs
        self.key = jax.random.key(seed)
        self._reset = jax.jit(jax.vmap(environment.reset))
        self._step = jax.jit(jax.vmap(environment.step))
        self.state = self._new_states()

    def _new_states(self):
        self.key, reset_key = jax.random.split(self.key)
        return self._reset(jax.random.split(reset_key, self.num_envs))

    def reset(self):
        self.state = self._new_states()
        return self.state.obs

    def step(self, action: jax.Array):
        stepped = self._step(self.state, action)
        reward = stepped.reward
        done = stepped.done.astype(bool)
        metrics = stepped.metrics
        reset = self._new_states()

        def select(old, fresh):
            mask = done.reshape((self.num_envs,) + (1,) * (old.ndim - 1))
            return jp.where(mask, fresh, old)

        self.state = jax.tree.map(select, stepped, reset)
        return self.state.obs, reward, done, metrics


@torch.no_grad()
def evaluate(policy: Policy, env: GpuBatch):
    obs = _to_torch(env.reset())
    returns = torch.zeros(env.num_envs, device=obs.device)
    active = torch.ones(env.num_envs, dtype=torch.bool, device=obs.device)
    stable_tail = torch.zeros(env.num_envs, device=obs.device)
    final_height = torch.zeros(env.num_envs, device=obs.device)
    final_tilt = torch.zeros(env.num_envs, device=obs.device)
    for _ in range(env.environment.episode_steps):
        action = policy.action(obs)
        obs_jax, reward_jax, done_jax, metrics = env.step(_to_jax(action))
        reward = _to_torch(reward_jax)
        done = _to_torch(done_jax)
        stable = _to_torch(metrics["stable"])
        height = _to_torch(metrics["height"])
        tilt = _to_torch(metrics["tilt"])
        returns += reward * active
        stable_tail += stable * active
        final_height = torch.where(active, height, final_height)
        final_tilt = torch.where(active, tilt, final_tilt)
        active &= ~done
        obs = _to_torch(obs_jax)
    success = (final_height >= env.environment.config.min_standing_height) & (
        final_tilt < 0.12
    )
    return {
        "reward": returns.mean().item(),
        "success": success.float().mean().item(),
        "height": final_height.mean().item(),
        "tilt_deg": torch.rad2deg(final_tilt).mean().item(),
        "stable_fraction": (
            stable_tail / env.environment.episode_steps
        ).mean().item(),
    }


def save_checkpoint(path, policy, environment, update, seed, metrics):
    state = {name: value.detach().cpu() for name, value in policy.state_dict().items()}
    torch.save(
        {
            "format_version": 3,
            "policy": state,
            "config": asdict(environment.config),
            "update": update,
            "seed": seed,
            "controller": {
                "reference": policy.reference.detach().cpu().tolist(),
                "max_speed": policy.max_speed,
                "feedback_gain": policy.feedback_gain,
                "leveling_matrix": policy.leveling_matrix.detach().cpu().tolist(),
                "initial_reference": policy.initial_reference.detach().cpu().tolist(),
                "residual_scale": policy.residual_scale,
            },
            "evaluation": metrics,
            "observation": "roll_pitch/pi, previous_commands/pi",
            "joint_names": environment.joint_names,
            "hardware": {
                "ctrl_dt": environment.config.ctrl_dt,
                "max_speed": environment.config.max_speed,
                "max_acceleration": environment.config.max_acceleration,
                "joint_low": np.asarray(environment.low).tolist(),
                "joint_high": np.asarray(environment.high).tolist(),
                "preparation_commands": [],
            },
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--eval-envs", type=int, default=32)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cubebot_playground/checkpoints/standup/horizontal_1kg_gpu.pt"),
    )
    args = parser.parse_args()
    if min(
        args.num_envs,
        args.eval_envs,
        args.updates,
        args.rollout,
        args.epochs,
        args.eval_every,
    ) < 1:
        parser.error("all sizes must be positive")
    if jax.default_backend() != "gpu" or not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is required: JAX backend={jax.default_backend()}, "
            f"torch.cuda.is_available()={torch.cuda.is_available()}"
        )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    config = StandUpConfig(
        world_level=True,
        start_airborne=False,
        prestep=False,
        slope_roll_degrees=0.0,
        slope_pitch_degrees=0.0,
        dynamic_slope=False,
        max_foot_inward=0.04,
        target_height=0.08,
        max_torque=0.34,
    )
    environment = StandUpMjx(config)
    train_env = GpuBatch(environment, args.num_envs, args.seed)
    eval_env = GpuBatch(environment, args.eval_envs, args.seed + 10000)
    policy = Policy(
        np.array(environment.standing_pose, copy=True),
        config.max_speed,
        0.05,
        feedback_gain=1.5,
        leveling_matrix=np.array(environment.leveling_matrix, copy=True),
        initial_reference=np.array(environment.initial_pose, copy=True),
    ).to(device)
    actor_optimizer = torch.optim.Adam(
        [*policy.actor.parameters(), policy.log_std], lr=3e-5
    )
    critic_optimizer = torch.optim.Adam(policy.critic.parameters(), lr=3e-4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    last_path = args.output.with_name(args.output.stem + ".last" + args.output.suffix)
    best_score = (-float("inf"), -float("inf"))

    print(
        f"GPU training: JAX={jax.devices()[0]}, PyTorch={torch.cuda.get_device_name(0)}, "
        f"envs={args.num_envs}",
        flush=True,
    )
    start = time.monotonic()
    obs = _to_torch(train_env.reset())
    for update in range(1, args.updates + 1):
        observations, actions, old_logs, values, deltas, dones = (
            [] for _ in range(6)
        )
        for _ in range(args.rollout):
            with torch.no_grad():
                distribution = policy.distribution(obs)
                raw = distribution.sample()
                action = policy.action(obs, raw)
                value = policy.value(obs)
                old_log = distribution.log_prob(raw).sum(-1)
            next_obs_jax, reward_jax, done_jax, _ = train_env.step(_to_jax(action))
            next_obs = _to_torch(next_obs_jax)
            reward = _to_torch(reward_jax)
            done = _to_torch(done_jax)
            with torch.no_grad():
                next_value = policy.value(next_obs)
            observations.append(obs)
            actions.append(raw)
            old_logs.append(old_log)
            values.append(value)
            deltas.append(reward + 0.99 * next_value * ~done - value)
            dones.append(done)
            obs = next_obs

        advantage = torch.zeros(args.num_envs, device=device)
        advantages = []
        for index in reversed(range(args.rollout)):
            advantage = deltas[index] + 0.99 * 0.95 * ~dones[index] * advantage
            advantages.append(advantage)
        advantages = torch.stack(advantages[::-1]).flatten()
        returns = advantages + torch.stack(values).flatten()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        batch_obs = torch.stack(observations).reshape(-1, environment.observation_size)
        batch_action = torch.stack(actions).reshape(-1, environment.action_size)
        old_log = torch.stack(old_logs).flatten()

        stop_actor = False
        for _ in range(args.epochs):
            for ids in torch.randperm(len(returns), device=device).split(1024):
                critic_loss = 0.5 * (
                    policy.value(batch_obs[ids]) - returns[ids]
                ).square().mean()
                critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(policy.critic.parameters(), 1.0)
                critic_optimizer.step()
                if stop_actor:
                    continue
                distribution = policy.distribution(batch_obs[ids])
                log_ratio = (
                    distribution.log_prob(batch_action[ids]).sum(-1) - old_log[ids]
                )
                ratio = log_ratio.exp()
                if ((ratio - 1) - log_ratio).mean().item() > 0.01:
                    stop_actor = True
                    continue
                surrogate = torch.minimum(
                    ratio * advantages[ids],
                    ratio.clamp(0.8, 1.2) * advantages[ids],
                )
                anchor = torch.tanh(distribution.mean).square().mean()
                actor_loss = -surrogate.mean() + 0.1 * anchor
                actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    [*policy.actor.parameters(), policy.log_std], 0.5
                )
                actor_optimizer.step()

        if update == 1 or update % args.eval_every == 0 or update == args.updates:
            metrics = evaluate(policy, eval_env)
            score = (metrics["success"], metrics["stable_fraction"])
            save_checkpoint(last_path, policy, environment, update, args.seed, metrics)
            if score > best_score:
                best_score = score
                save_checkpoint(args.output, policy, environment, update, args.seed, metrics)
            steps = update * args.rollout * args.num_envs
            print(
                f"update={update} steps={steps} reward={metrics['reward']:.1f} "
                f"success={metrics['success']:.3f} height={metrics['height']:.4f} "
                f"tilt_deg={metrics['tilt_deg']:.2f} "
                f"stable={metrics['stable_fraction']:.3f} "
                f"steps/s={steps / (time.monotonic() - start):.0f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
