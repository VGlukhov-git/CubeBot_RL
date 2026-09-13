"""Run a stand-up checkpoint without MuJoCo, using JSON Lines for hardware I/O."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .policy import Policy


class HardwareController:
    """Stateful 50 Hz controller with the same command dynamics as training."""

    def __init__(self, saved):
        if "hardware" not in saved:
            raise ValueError(
                "Checkpoint has no hardware metadata; run export_hardware first"
            )
        self.names = tuple(saved["joint_names"])
        if len(self.names) != 12:
            raise ValueError("Expected exactly 12 joint names")
        hw = saved["hardware"]
        self.dt = float(hw["ctrl_dt"])
        self.max_speed = float(hw["max_speed"])
        self.max_acceleration = float(hw["max_acceleration"])
        self.low = np.asarray(hw["joint_low"], dtype=np.float64)
        self.high = np.asarray(hw["joint_high"], dtype=np.float64)
        self.preparation = np.asarray(
            hw["preparation_commands"], dtype=np.float64
        ).reshape(-1, 12)
        self.initial = np.asarray(
            saved["controller"]["initial_reference"], dtype=np.float64
        )
        if any(array.shape != (12,) for array in (self.low, self.high, self.initial)):
            raise ValueError("Hardware command arrays must contain 12 values")
        self.policy = Policy.from_checkpoint(saved).eval()
        self.reset()

    def reset(self):
        self.command = self.initial.copy()
        self.velocity = np.zeros(12, dtype=np.float64)
        self.frame = 0

    @property
    def preparing(self):
        return self.frame < len(self.preparation)

    @torch.inference_mode()
    def step(self, roll_pitch_radians):
        roll_pitch = np.asarray(roll_pitch_radians, dtype=np.float64)
        if roll_pitch.shape != (2,) or not np.isfinite(roll_pitch).all():
            raise ValueError("roll_pitch_radians must contain two finite values")
        preparing = self.preparing
        if preparing:
            action = (self.preparation[self.frame] - self.command) / (
                self.dt * self.max_speed
            )
        else:
            observation = np.concatenate((roll_pitch / np.pi, self.command / np.pi))
            action = self.policy.action(
                torch.from_numpy(observation[None]).to(torch.float32)
            ).numpy()[0]

        desired_velocity = np.clip(action, -1, 1) * self.max_speed
        braking = np.sqrt(
            2
            * self.max_acceleration
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
            -self.max_acceleration * self.dt,
            self.max_acceleration * self.dt,
        )
        self.command = np.clip(
            self.command + self.velocity * self.dt, self.low, self.high
        )
        self.frame += 1
        return self.command.copy(), preparing


def _read_tilt(message, degrees):
    if isinstance(message, dict):
        tilt = [message[name] for name in ("roll", "pitch")]
    else:
        tilt = message
    tilt = np.asarray(tilt, dtype=np.float64)
    return np.deg2rad(tilt) if degrees else tilt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--input-degrees", action="store_true", help="Treat incoming RPY as degrees"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    controller = HardwareController(saved)

    for line in sys.stdin:
        if not line.strip():
            continue
        tilt = _read_tilt(json.loads(line), args.input_degrees)
        command, preparing = controller.step(tilt)
        output = {
            "preparing": preparing,
            "positions_rad": dict(zip(controller.names, command.tolist())),
            "positions_deg": dict(
                zip(controller.names, np.rad2deg(command).tolist())
            ),
        }
        print(json.dumps(output, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
