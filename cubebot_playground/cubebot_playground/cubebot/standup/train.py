"""Residual PPO: preserve the feasible rise and learn bounded corrections."""

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .environment import CubebotStandUp, StandUpConfig
from .policy import Policy


@torch.no_grad()
def evaluate(policy, env):
    """Fixed-seed complete episodes, including preparation, no action noise."""
    env.rng = np.random.default_rng(10000)
    roll = env.config.slope_roll_degrees
    pitch = env.config.slope_pitch_degrees
    grid = np.array([(r, p) for r in (-roll, 0, roll) for p in (-pitch, 0, pitch)])
    targets = grid[np.arange(env.num_envs) % len(grid)]
    slopes = np.zeros_like(targets) if env.config.dynamic_slope else targets
    obs = env.reset(slopes=slopes, slope_targets=targets)
    active = np.ones(env.num_envs, dtype=bool)
    success = np.zeros(env.num_envs, dtype=bool)
    final_height = np.zeros(env.num_envs)
    final_tilt = np.zeros(env.num_envs)
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
        final_tilt[done] = info["world_tilt_degrees"][done]
        success[done] = info["success"][done] & ~terminated[done]
        active[done] = False
        if not active.any():
            break
        if done.any():
            obs = env.reset(np.flatnonzero(done))
    return {
        "success": float(success.mean()),
        "cases": [
            {
                "roll": float(r),
                "pitch": float(p),
                "start_roll": float(sr),
                "start_pitch": float(sp),
                "success": bool(ok),
                "height": float(h),
                "world_tilt_degrees": float(tilt),
            }
            for (r, p), (sr, sp), ok, h, tilt in zip(
                targets, slopes, success, final_height, final_tilt
            )
        ],
        "height": float(final_height.mean()),
        "world_tilt_degrees": float(final_tilt.mean()),
        "level_fraction": float((final_tilt < np.rad2deg(0.12)).mean()),
        "rise_reward": float(np.mean(rewards)) if rewards else 0.0,
        "slip": float(np.mean(slips)) if slips else 0.0,
        "drift": float(np.mean(drifts)) if drifts else 0.0,
        "stable_tail": float(np.mean(stable_tail)) if stable_tail else 0.0,
    }


def save_checkpoint(path, policy, env, update, seed, metrics):
    torch.save(
        {
            "format_version": 3,
            "policy": policy.state_dict(),
            "config": asdict(env.config),
            "update": update,
            "seed": seed,
            "controller": {
                "reference": policy.reference.tolist(),
                "max_speed": policy.max_speed,
                "feedback_gain": policy.feedback_gain,
                "leveling_matrix": None
                if policy.leveling_matrix is None
                else policy.leveling_matrix.tolist(),
                "initial_reference": None
                if policy.initial_reference is None
                else policy.initial_reference.tolist(),
                "residual_scale": policy.residual_scale,
            },
            "evaluation": metrics,
            "observation": "roll_pitch/pi, previous_commands/pi",
            "joint_names": [env.model.joint(int(j)).name for j in env.joints],
            "hardware": {
                "ctrl_dt": env.config.ctrl_dt,
                "max_speed": env.config.max_speed,
                "max_acceleration": env.config.max_acceleration,
                "joint_low": env.low.tolist(),
                "joint_high": env.high.tolist(),
                "preparation_commands": env.prestep_commands.tolist(),
            },
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
    p.add_argument(
        "--prestep",
        action="store_true",
        help="Compatibility alias; simultaneous landing is now always enabled",
    )
    p.add_argument(
        "--foot-inward",
        type=float,
        default=0.03,
        help="Allowed inward foot travel in metres, 0..0.03",
    )
    p.add_argument(
        "--slope-roll",
        type=float,
        default=15.0,
        help="Random plane roll range ±degrees, 0..15",
    )
    p.add_argument(
        "--slope-pitch",
        type=float,
        default=15.0,
        help="Random plane pitch range ±degrees, 0..15",
    )
    p.add_argument(
        "--static-slope",
        action="store_true",
        help="Disable the smooth post-rise change of the support plane",
    )
    p.add_argument(
        "--slope-change-start",
        type=float,
        default=2.5,
        help="Seconds after landing before the support begins to tilt",
    )
    p.add_argument(
        "--slope-change-duration",
        type=float,
        default=4.0,
        help="Duration of the smooth support tilt in seconds",
    )
    p.add_argument(
        "--height",
        type=float,
        default=0.08,
        help="Reference pose height, not an exact success target",
    )
    p.add_argument(
        "--min-standing-height",
        type=float,
        default=0.045,
        help="Minimum raised-body clearance for success, metres",
    )
    p.add_argument("--torque", type=float, default=0.34)
    p.add_argument("--friction", type=float, default=1.5)
    p.add_argument(
        "--residual-scale",
        type=float,
        default=0.05,
        help="Maximum learned velocity correction, rad/s",
    )
    p.add_argument(
        "--rise-gain",
        type=float,
        default=1.5,
        help="Reference convergence gain in 1/s; servo limits remain unchanged",
    )
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-envs", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output", type=Path, default=Path("checkpoints/standup/horizontal.pt")
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
    if not 0 < args.rise_gain <= 6:
        p.error("rise-gain must be in (0, 6] 1/s")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    config = StandUpConfig(
        start_airborne=True,
        max_foot_inward=args.foot_inward,
        min_standing_height=args.min_standing_height,
        slope_roll_degrees=args.slope_roll,
        slope_pitch_degrees=args.slope_pitch,
        dynamic_slope=not args.static_slope,
        slope_change_start=args.slope_change_start,
        slope_change_duration=args.slope_change_duration,
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
        feedback_gain=args.rise_gain,
        leveling_matrix=env.leveling_matrix,
        initial_reference=env.initial_pose if config.world_level else None,
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
            f"height={metrics['height']:.4f} tilt_deg={metrics['world_tilt_degrees']:.2f} rise_slip={metrics['slip']:.5f} "
            f"drift={metrics['drift']:.4f} stable_tail={metrics['stable_tail']:.3f} "
            f"level_fraction={metrics['level_fraction']:.3f} best_success={best_score[0]:.3f}",
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
        batch_obs = torch.stack(observations).reshape(-1, env.observation_size)
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
