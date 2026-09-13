"""Standalone JSON Lines runner for a self-contained CubeBot ONNX model."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


def load_state(path, command, velocity, frame):
    state = json.loads(path.read_text())
    restored_command = np.asarray(state["command"], dtype=np.float32)
    restored_velocity = np.asarray(state["velocity"], dtype=np.float32)
    restored_frame = int(state["frame"])
    if (
        restored_command.shape != command.shape
        or restored_velocity.shape != velocity.shape
        or not np.isfinite(restored_command).all()
        or not np.isfinite(restored_velocity).all()
        or restored_frame < 0
    ):
        raise ValueError("Invalid saved controller state")
    return restored_command, restored_velocity, restored_frame


def save_state(path, command, velocity, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "command": command.tolist(),
                "velocity": velocity.tolist(),
                "frame": frame,
            },
            separators=(",", ":"),
        )
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--input-degrees", action="store_true")
    parser.add_argument(
        "--state-file", type=Path, help="Save controller state after every command"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Restore state from --state-file"
    )
    args = parser.parse_args()
    if args.resume and args.state_file is None:
        parser.error("--resume requires --state-file")

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    metadata = json.loads(session.get_modelmeta().custom_metadata_map["cubebot"])
    hardware = metadata["hardware"]
    names = metadata["joint_names"]
    dt = float(hardware["ctrl_dt"])
    max_speed = float(hardware["max_speed"])
    max_acceleration = float(hardware["max_acceleration"])
    low = np.asarray(hardware["joint_low"], dtype=np.float32)
    high = np.asarray(hardware["joint_high"], dtype=np.float32)
    preparation = np.asarray(hardware["preparation_commands"], dtype=np.float32)
    command = np.asarray(metadata["initial_command"], dtype=np.float32)
    velocity = np.zeros(12, dtype=np.float32)
    frame = 0
    if args.resume:
        command, velocity, frame = load_state(
            args.state_file, command, velocity, frame
        )

    for line in sys.stdin:
        if not line.strip():
            continue
        message = json.loads(line)
        if isinstance(message, dict):
            tilt = np.asarray([message["roll"], message["pitch"]], dtype=np.float32)
        else:
            tilt = np.asarray(message, dtype=np.float32)
        if tilt.shape != (2,) or not np.isfinite(tilt).all():
            raise ValueError("Input must contain finite roll and pitch")
        if args.input_degrees:
            tilt = np.deg2rad(tilt)

        preparing = frame < len(preparation)
        if preparing:
            action = (preparation[frame] - command) / (dt * max_speed)
        else:
            observation = np.concatenate((tilt / np.pi, command / np.pi))[None]
            action = session.run(None, {"observation": observation})[0][0]

        desired_velocity = np.clip(action, -1, 1) * max_speed
        braking = np.sqrt(
            2
            * max_acceleration
            * np.maximum(
                np.where(desired_velocity >= 0, high - command, command - low), 0
            )
        )
        desired_velocity = np.clip(desired_velocity, -braking, braking)
        velocity += np.clip(
            desired_velocity - velocity,
            -max_acceleration * dt,
            max_acceleration * dt,
        )
        command = np.clip(command + velocity * dt, low, high)
        frame += 1
        if args.state_file is not None:
            save_state(args.state_file, command, velocity, frame)
        output = {
            "preparing": preparing,
            "positions_rad": dict(zip(names, command.tolist())),
            "positions_deg": dict(zip(names, np.rad2deg(command).tolist())),
        }
        print(json.dumps(output, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
