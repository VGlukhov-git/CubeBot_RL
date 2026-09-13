"""Export the deterministic 14-input stand-up policy to one ONNX file."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .policy import Policy


class OnnxPolicy(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        return self.policy.action(observation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "hardware" not in saved:
        raise ValueError("Export hardware metadata into the checkpoint before ONNX")
    wrapper = OnnxPolicy(Policy.from_checkpoint(saved).eval()).eval()
    example = torch.zeros((1, 14), dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (example,),
        args.output,
        input_names=["observation"],
        output_names=["normalized_velocity"],
        opset_version=18,
        dynamo=True,
    )

    import onnx
    import onnxruntime as ort

    model = onnx.load(args.output)
    metadata = {
        "observation": "[roll/pi,pitch/pi,12 previous_commands/pi]",
        "normalized_velocity": "12 desired command velocities divided by max_speed",
        "joint_names": saved["joint_names"],
        "ctrl_dt": saved["config"]["ctrl_dt"],
        "max_speed": saved["config"]["max_speed"],
        "max_acceleration": saved["config"]["max_acceleration"],
        "initial_command": saved["controller"]["initial_reference"],
        "hardware": saved["hardware"],
        "config": saved["config"],
    }
    entry = model.metadata_props.add()
    entry.key = "cubebot"
    entry.value = json.dumps(metadata, separators=(",", ":"))
    onnx.save(model, args.output)

    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    expected = wrapper(example).detach().numpy()
    actual = session.run(None, {"observation": example.numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    print(
        f"saved and verified {args.output}: observation (1, 14) "
        "-> normalized_velocity (1, 12)"
    )


if __name__ == "__main__":
    main()
