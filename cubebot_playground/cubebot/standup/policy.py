"""Torch policy shared by training and the lightweight hardware runtime."""

import torch
from torch import nn
from torch.distributions import Normal


class Policy(nn.Module):
    def __init__(
        self,
        reference=None,
        max_speed=1.0,
        residual_scale=0.01,
        feedback_gain=5.0,
        leveling_matrix=None,
        initial_reference=None,
    ):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(14, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 12),
        )
        self.critic = nn.Sequential(
            nn.Linear(14, 128),
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
        self.register_buffer(
            "leveling_matrix",
            None
            if leveling_matrix is None
            else torch.as_tensor(leveling_matrix, dtype=torch.float32),
        )
        self.register_buffer(
            "initial_reference",
            None
            if initial_reference is None
            else torch.as_tensor(initial_reference, dtype=torch.float32),
        )
        self.max_speed = max_speed
        self.feedback_gain = feedback_gain
        self.residual_scale = residual_scale
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

    def distribution(self, obs):
        return Normal(self.actor(obs), self.log_std.clamp(-4, -0.5).exp())

    def action(self, obs, raw=None):
        raw = self.actor(obs) if raw is None else raw
        if self.reference is None:
            return torch.tanh(raw)
        command = obs[:, 2:] * torch.pi
        desired = self.reference
        if self.leveling_matrix is not None:
            progress = torch.linalg.vector_norm(
                command - self.initial_reference, dim=-1
            ) / torch.linalg.vector_norm(
                self.reference - self.initial_reference
            ).clamp_min(1e-6)
            gate = ((progress - 0.25) / 0.5).clamp(0, 1)
            tilt = (obs[:, :2] * torch.pi).clamp(-0.35, 0.35)
            desired = desired + gate[:, None] * (tilt @ self.leveling_matrix.T).clamp(
                -0.8, 0.8
            )
        baseline_velocity = self.feedback_gain * (desired - command)
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
            leveling_matrix=settings.get("leveling_matrix"),
            initial_reference=settings.get("initial_reference"),
        )
        state = dict(saved["policy"])
        # Version 2 checkpoints used [roll, pitch, yaw, 12 commands]. Dropping
        # column 2 preserves their exact behavior for yaw=0 and makes yaw inert.
        for key in ("actor.0.weight", "critic.0.weight"):
            weight = state[key]
            if weight.shape[1] == 15:
                state[key] = torch.cat((weight[:, :2], weight[:, 3:]), dim=1)
            elif weight.shape[1] != 14:
                raise ValueError(f"Unsupported observation width in {key}")
        policy.load_state_dict(state)
        return policy
