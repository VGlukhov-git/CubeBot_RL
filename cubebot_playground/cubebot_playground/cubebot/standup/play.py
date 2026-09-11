"""Evaluate a checkpoint headlessly, or view it (mjpython on macOS)."""

import argparse
import time
from contextlib import nullcontext

import mujoco
import numpy as np
import torch

from .environment import CubebotStandUp, StandUpConfig
from .train import Policy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--episodes", type=int, default=5)
    args = p.parse_args()
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    env = CubebotStandUp(1, StandUpConfig(**saved["config"]))
    policy = Policy.from_checkpoint(saved).eval()
    data = mujoco.MjData(env.model)
    context = nullcontext(None)
    if args.viewer:
        from mujoco import viewer as mj_viewer

        context = mj_viewer.launch_passive(env.model, data)
    with context as viewer:
        for episode in range(args.episodes):
            obs = env.reset()
            total, slip = 0.0, []
            for _ in range(
                len(env.prestep_commands)
                + int(np.ceil(env.config.episode_seconds / env.config.ctrl_dt))
            ):
                start = time.monotonic()
                with torch.no_grad():
                    action = policy.action(torch.from_numpy(obs)).numpy()
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward[0]
                if not info["preparing"][0]:
                    slip.append(info["slip"][0])
                if viewer:
                    if not viewer.is_running():
                        return
                    data.qpos[:] = env.qpos[0]
                    data.qvel[:] = env.qvel[0]
                    mujoco.mj_forward(env.model, data)
                    viewer.sync()
                    time.sleep(max(0, env.config.ctrl_dt - (time.monotonic() - start)))
                if terminated[0] or truncated[0]:
                    break
            print(
                f"episode={episode + 1} reward={total:.2f} height={info['height'][0]:.4f} "
                f"success={bool(info['success'][0])} rise_slip={np.mean(slip) if slip else 0.0:.5f} m/s"
            )


if __name__ == "__main__":
    main()
