"""Add the trajectory and command limits required by the hardware runtime."""

import argparse
from pathlib import Path

import torch

from .environment import CubebotStandUp, StandUpConfig
from .policy import Policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    env = CubebotStandUp(1, StandUpConfig(**saved["config"]))
    # Materialize legacy migration so the exported file itself has no yaw input.
    saved["policy"] = Policy.from_checkpoint(saved).state_dict()
    saved["format_version"] = 3
    saved["observation"] = "roll_pitch/pi, previous_commands/pi"
    saved["hardware"] = {
        "ctrl_dt": env.config.ctrl_dt,
        "max_speed": env.config.max_speed,
        "max_acceleration": env.config.max_acceleration,
        "joint_low": env.low.tolist(),
        "joint_high": env.high.tolist(),
        "preparation_commands": env.prestep_commands.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, args.output)
    print(f"saved {args.output} ({len(env.prestep_commands)} preparation frames)")


if __name__ == "__main__":
    main()
