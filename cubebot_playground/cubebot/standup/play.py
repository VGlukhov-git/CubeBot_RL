"""Evaluate a checkpoint headlessly, or view it (mjpython on macOS)."""

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import mujoco
import numpy as np
import torch

from .environment import CubebotStandUp, StandUpConfig
from .policy import Policy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--roll", type=float, help="Fixed floor roll in degrees, -15..15")
    p.add_argument("--pitch", type=float, help="Fixed floor pitch in degrees, -15..15")
    p.add_argument("--target-roll", type=float, help="Roll after the post-rise tilt")
    p.add_argument("--target-pitch", type=float, help="Pitch after the post-rise tilt")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--episodes", type=int, default=5)
    args = p.parse_args()
    torch.set_num_threads(1)
    checkpoint = Path(args.checkpoint)
    if checkpoint.suffix.lower() == ".onnx":
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(checkpoint), providers=["CPUExecutionProvider"]
        )
        metadata_map = session.get_modelmeta().custom_metadata_map
        if "cubebot" not in metadata_map:
            raise ValueError("ONNX model has no CubeBot metadata")
        metadata = json.loads(metadata_map["cubebot"])
        config = metadata.get("config")
        if config is None:
            # ONNX models exported by older versions did not embed the full
            # simulation configuration.  Use the matching PT checkpoint when
            # it is available so the Viewer reproduces the trained setup.
            source_checkpoint = checkpoint.with_suffix(".pt")
            if not source_checkpoint.is_file():
                raise ValueError(
                    "This older ONNX model has no embedded environment config "
                    f"and {source_checkpoint.name} was not found. Re-export it "
                    "with the current export_onnx.py."
                )
            source = torch.load(
                source_checkpoint, map_location="cpu", weights_only=True
            )
            config = dict(source["config"])
            print(f"ONNX environment config: {source_checkpoint}")
        else:
            config = dict(config)

        input_name = session.get_inputs()[0].name

        def policy_action(observation):
            return session.run(
                None,
                {input_name: np.asarray(observation, dtype=np.float32)},
            )[0]

        runtime_name = "ONNX Runtime"
    elif checkpoint.suffix.lower() in (".pt", ".pth"):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = dict(saved["config"])
        policy = Policy.from_checkpoint(saved).eval()

        def policy_action(observation):
            with torch.no_grad():
                return policy.action(torch.from_numpy(observation)).numpy()

        runtime_name = "PyTorch"
    else:
        raise ValueError("Checkpoint must have a .pt, .pth, or .onnx extension")

    # Old checkpoints retain their original flat/sequential initialization.
    config.setdefault("height_agnostic", False)
    config.setdefault("world_level", False)
    config.setdefault("max_foot_inward", 0.0)
    config.setdefault("command_height_offset", 0.0)
    config.setdefault("start_airborne", False)
    config.setdefault("slope_roll_degrees", 0.0)
    config.setdefault("slope_pitch_degrees", 0.0)
    config.setdefault("dynamic_slope", False)
    config.setdefault("slope_change_start", 2.5)
    config.setdefault("slope_change_duration", 4.0)
    env = CubebotStandUp(1, StandUpConfig(**config))
    print(f"Policy runtime: {runtime_name} ({checkpoint})")
    data = mujoco.MjData(env.model)
    context = nullcontext(None)
    if args.viewer:
        from mujoco import viewer as mj_viewer

        context = mj_viewer.launch_passive(env.model, data)
    with context as viewer:
        for episode in range(args.episodes):
            slopes = (
                None
                if args.roll is None and args.pitch is None
                else np.array([[args.roll or 0.0, args.pitch or 0.0]])
            )
            targets = (
                None
                if args.target_roll is None and args.target_pitch is None
                else np.array([[args.target_roll or 0.0, args.target_pitch or 0.0]])
            )
            if targets is not None and slopes is None:
                slopes = np.zeros((1, 2))
            obs = env.reset(slopes=slopes, slope_targets=targets)
            env.model.geom_quat[env.floor] = env.floor_quats[0, env.floor]
            mujoco.mj_setConst(env.model, data)
            total, slip = 0.0, []
            for _ in range(
                len(env.prestep_commands)
                + int(np.ceil(env.config.episode_seconds / env.config.ctrl_dt))
            ):
                start = time.monotonic()
                action = policy_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward[0]
                if not info["preparing"][0]:
                    slip.append(info["slip"][0])
                if viewer:
                    if not viewer.is_running():
                        return
                    # mjbatch owns a model per simulation. Mirror its changing
                    # floor orientation into the separate interactive model.
                    env.model.geom_quat[env.floor] = env.floor_quats[0, env.floor]
                    mujoco.mj_setConst(env.model, data)
                    data.qpos[:] = env.qpos[0]
                    data.qvel[:] = env.qvel[0]
                    mujoco.mj_forward(env.model, data)
                    viewer.sync()
                    time.sleep(max(0, env.config.ctrl_dt - (time.monotonic() - start)))
                if terminated[0] or truncated[0]:
                    break
            print(
                f"episode={episode + 1} slope={env.slope_degrees[0].tolist()} reward={total:.2f} height={info['height'][0]:.4f} "
                f"tilt_deg={info['world_tilt_degrees'][0]:.2f} success={bool(info['success'][0])} rise_slip={np.mean(slip) if slip else 0.0:.5f} m/s"
            )


if __name__ == "__main__":
    main()
