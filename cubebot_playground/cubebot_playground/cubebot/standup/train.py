"""Residual PPO: preserve the feasible rise and learn bounded corrections."""

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .environment import CubebotStandUp, StandUpConfig


class Policy(nn.Module):
    def __init__(
        self, reference=None, max_speed=1.0, residual_scale=0.01, feedback_gain=1.0
    ):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(15, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 12),
        )
        self.critic = nn.Sequential(
            nn.Linear(15, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.log_std = nn.Parameter(torch.full((12,), -1.0))
        self.register_buffer(
            "reference",
            None
            if reference is None
            else torch.as_tensor(reference, dtype=torch.float32),
        )
        self.max_speed = max_speed
        self.feedback_gain = feedback_gain
        self.residual_scale = (
            residual_scale  # rad/s, before normalization to env action
        )
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

    def distribution(self, obs):
        return Normal(self.actor(obs), self.log_std.clamp(-4, -0.5).exp())

    def action(self, obs, raw=None):
        raw = self.actor(obs) if raw is None else raw
        if self.reference is None:  # Read-only compatibility with old checkpoints.
            return torch.tanh(raw)
        command = obs[:, 3:] * torch.pi
        baseline_velocity = self.feedback_gain * (
            self.reference - command
        )  # no encoder input
        correction = self.residual_scale * torch.tanh(raw)
        return ((baseline_velocity + correction) / self.max_speed).clamp(-0.95, 0.95)

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    @classmethod
    def from_checkpoint(cls, saved):
        settings = saved.get("controller", {})
        policy = cls(
            reference=settings.get("reference"),
            max_speed=settings.get("max_speed", 1.0),
            residual_scale=settings.get("residual_scale", 0.01),
            feedback_gain=settings.get("feedback_gain", 1.0),
        )
        policy.load_state_dict(saved["policy"])
        return policy


@torch.no_grad()
def evaluate(policy, env):
    """Fixed-seed complete episodes, including preparation, no action noise."""
    env.rng = np.random.default_rng(10000)
    obs = env.reset()
    active = np.ones(env.num_envs, dtype=bool)
    success = np.zeros(env.num_envs, dtype=bool)
    final_height = np.zeros(env.num_envs)
    rewards, slips, drifts, stable_tail = [], [], [], []
    length = len(env.prestep_commands) + int(
        np.ceil(env.config.episode_seconds / env.config.ctrl_dt)
    )
    for step in range(length):
        action = policy.action(torch.from_numpy(obs)).numpy()
        obs, reward, terminated, truncated, info = env.step(action)
        rise = active & ~info["preparing"]
        if rise.any():
            rewards.extend(reward[rise].tolist())
            slips.extend(info["slip"][rise].tolist())
            drifts.extend(info["lateral_drift"][rise].tolist())
        if step >= length - round(1 / env.config.ctrl_dt):
            stable_tail.extend((info["stable"] & active).tolist())
        done = active & (terminated | truncated)
        final_height[done] = info["height"][done]
        success[done] = info["success"][done] & ~terminated[done]
        active[done] = False
        if not active.any():
            break
    return {
        "success": float(success.mean()),
        "height": float(final_height.mean()),
        "rise_reward": float(np.mean(rewards)) if rewards else 0.0,
        "slip": float(np.mean(slips)) if slips else 0.0,
        "drift": float(np.mean(drifts)) if drifts else 0.0,
        "stable_tail": float(np.mean(stable_tail)) if stable_tail else 0.0,
    }


def save_checkpoint(path, policy, env, update, seed, metrics):
    torch.save(
        {
            "format_version": 2,
            "policy": policy.state_dict(),
            "config": asdict(env.config),
            "update": update,
            "seed": seed,
            "controller": {
                "reference": policy.reference.tolist(),
                "max_speed": policy.max_speed,
                "feedback_gain": policy.feedback_gain,
                "residual_scale": policy.residual_scale,
            },
            "evaluation": metrics,
            "observation": "rpy/pi, previous_commands/pi",
            "joint_names": [env.model.joint(int(j)).name for j in env.joints],
        },
        path,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--num-threads", type=int, default=0)
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--prestep", action="store_true")
    p.add_argument("--height", type=float, default=0.08)
    p.add_argument("--torque", type=float, default=0.34)
    p.add_argument("--friction", type=float, default=1.5)
    p.add_argument(
        "--residual-scale",
        type=float,
        default=0.01,
        help="Maximum learned velocity correction, rad/s",
    )
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output", type=Path, default=Path("checkpoints/standup/residual.pt")
    )
    args = p.parse_args()
    if (
        min(
            args.num_envs,
            args.updates,
            args.rollout,
            args.epochs,
            args.eval_every,
            args.eval_envs,
        )
        < 1
        or args.num_envs * args.rollout < 2
    ):
        p.error("positive sizes required, with at least two rollout samples")
    if not 0 < args.residual_scale <= 0.05:
        p.error("residual-scale must be in (0, .05] rad/s")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    config = StandUpConfig(
        prestep=args.prestep,
        target_height=args.height,
        max_torque=args.torque,
        friction=args.friction,
    )
    env = CubebotStandUp(args.num_envs, config, args.seed, args.num_threads)
    eval_env = CubebotStandUp(args.eval_envs, config, 10000, args.num_threads)
    policy = Policy(
        env.standing_pose,
        config.max_speed,
        args.residual_scale,
        feedback_gain=1.0 if config.prestep else 0.8,
    )
    actor_optimizer = torch.optim.Adam(
        [*policy.actor.parameters(), policy.log_std], lr=3e-5
    )
    critic_optimizer = torch.optim.Adam(policy.critic.parameters(), lr=3e-4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    last_path = args.output.with_name(args.output.stem + ".last" + args.output.suffix)
    best_score = (-float("inf"),) * 3

    def check_and_save(update):
        nonlocal best_score
        metrics = evaluate(policy, eval_env)
        # Prefer reliable final standing, then stable last-second behavior, then slip.
        score = (metrics["success"], metrics["stable_tail"], -metrics["slip"])
        save_checkpoint(last_path, policy, env, update, args.seed, metrics)
        if score > best_score:
            best_score = score
            save_checkpoint(args.output, policy, env, update, args.seed, metrics)
        print(
            f"eval update={update} success={metrics['success']:.3f} "
            f"height={metrics['height']:.4f} rise_slip={metrics['slip']:.5f} "
            f"drift={metrics['drift']:.4f} stable_tail={metrics['stable_tail']:.3f} "
            f"best_success={best_score[0]:.3f}",
            flush=True,
        )

    check_and_save(0)  # A measured baseline is saved before any PPO updates.
    obs = torch.from_numpy(env.observation())
    recent_successes = []
    phase_stats = {"prep": [], "rise": []}
    start = time.monotonic()
    for update in range(1, args.updates + 1):
        observations, actions, log_probs, values, deltas, dones, masks = (
            [] for _ in range(7)
        )
        for _ in range(args.rollout):
            with torch.no_grad():
                dist = policy.distribution(obs)
                raw = dist.sample()
                value = policy.value(obs)
                log_prob = dist.log_prob(raw).sum(-1)
                action = policy.action(obs, raw).numpy()
            nxt, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            mask = ~info["preparing"]
            with torch.no_grad():
                next_value = policy.value(torch.from_numpy(nxt))
            observations.append(obs)
            actions.append(raw)
            log_probs.append(log_prob)
            values.append(value)
            deltas.append(
                torch.from_numpy(reward)
                + 0.99 * next_value * torch.from_numpy(~terminated)
                - value
            )
            dones.append(torch.from_numpy(done))
            masks.append(torch.from_numpy(mask))
            recent_successes.extend(info["success"][done].tolist())
            recent_successes = recent_successes[-256:]
            for name, selection in [("prep", ~mask), ("rise", mask)]:
                if selection.any():
                    phase_stats[name].append(
                        [
                            selection.sum(),
                            reward[selection].sum(),
                            info["height"][selection].sum(),
                            info["slip"][selection].sum(),
                            info["lateral_drift"][selection].sum(),
                        ]
                    )
            if done.any():
                nxt = env.reset(np.flatnonzero(done))
            obs = torch.from_numpy(nxt)
        advantage = torch.zeros(args.num_envs)
        advantages = []
        for t in reversed(range(args.rollout)):
            advantage = deltas[t] + 0.99 * 0.95 * (~dones[t]) * advantage
            advantages.append(advantage)
        advantages = torch.stack(advantages[::-1]).flatten()
        returns = advantages + torch.stack(values).flatten()
        controlled = torch.stack(masks).flatten()
        if controlled.sum() > 1:
            advantages = (advantages - advantages[controlled].mean()) / (
                advantages[controlled].std() + 1e-8
            )
        batch_obs = torch.stack(observations).reshape(-1, 15)
        batch_action = torch.stack(actions).reshape(-1, 12)
        old_log = torch.stack(log_probs).flatten()
        stop_actor = False
        for _ in range(args.epochs):
            for ids in torch.randperm(len(returns)).split(512):
                critic_loss = (
                    0.5 * (policy.value(batch_obs[ids]) - returns[ids]).square().mean()
                )
                critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(policy.critic.parameters(), 1.0)
                critic_optimizer.step()
                actor_ids = ids[controlled[ids]]
                if stop_actor or len(actor_ids) < 2:
                    continue
                dist = policy.distribution(batch_obs[actor_ids])
                log_ratio = (
                    dist.log_prob(batch_action[actor_ids]).sum(-1) - old_log[actor_ids]
                )
                ratio = log_ratio.exp()
                if ((ratio - 1) - log_ratio).mean().item() > 0.01:
                    stop_actor = True
                    continue
                surrogate = torch.minimum(
                    ratio * advantages[actor_ids],
                    ratio.clamp(0.8, 1.2) * advantages[actor_ids],
                )
                # Keep small corrections unless the task reward justifies a departure.
                anchor = torch.tanh(dist.mean).square().mean()
                actor_loss = -surrogate.mean() + 0.1 * anchor
                actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    [*policy.actor.parameters(), policy.log_std], 0.5
                )
                actor_optimizer.step()
        if update == 1 or update % args.eval_every == 0 or update == args.updates:
            steps = update * args.rollout * args.num_envs
            print(
                f"update={update} steps={steps} episode_success="
                f"{np.mean(recent_successes) if recent_successes else float('nan'):.3f} "
                f"steps/s={steps / (time.monotonic() - start):.0f}",
                flush=True,
            )
            for name, records in phase_stats.items():
                if records:
                    totals = np.sum(records, axis=0)
                    print(
                        f"  {name}: samples={int(totals[0])} reward={totals[1] / totals[0]:.3f} "
                        f"height={totals[2] / totals[0]:.4f} slip={totals[3] / totals[0]:.5f} "
                        f"drift={totals[4] / totals[0]:.4f}",
                        flush=True,
                    )
                records.clear()
            check_and_save(update)


if __name__ == "__main__":
    main()
