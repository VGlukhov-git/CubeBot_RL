from __future__ import annotations

import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


# Brax 0.14.2 still uses this API, which was removed in JAX 0.11.
# This is the drop-in replacement recommended by the JAX migration guide.
if not hasattr(jax, "device_put_replicated"):
    def _device_put_replicated(value, devices):
        mesh = jax.sharding.Mesh(np.array(devices), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.P("x"))
        return jax.tree.map(
            lambda leaf: jax.device_put(
                jnp.stack([leaf] * len(devices)),
                sharding,
            ),
            value,
        )

    jax.device_put_replicated = _device_put_replicated

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from mujoco_playground import wrapper

from cubebot_playground.cubebot.balance import CubebotBalance


def main():
    env = CubebotBalance(
        config_overrides={"episode_length": 200},
    )

    checkpoint_dir = Path("checkpoints/cubebot_balance")
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,

        policy_hidden_layer_sizes=(
            64,
            64,
        ),

        value_hidden_layer_sizes=(
            64,
            64,
        ),
    )

    train_fn = functools.partial(
        ppo.train,

        num_timesteps=100_000,

        num_evals=5,

        reward_scaling=1.0,

        episode_length=env.episode_length,

        normalize_observations=True,

        action_repeat=1,

        unroll_length=20,

        num_minibatches=4,

        num_updates_per_batch=2,

        discounting=0.97,

        learning_rate=3e-4,

        entropy_cost=1e-3,

        num_envs=4,
        num_eval_envs=4,
        batch_size=64,

        seed=0,

        network_factory=network_factory,

        save_checkpoint_path=checkpoint_dir,

        wrap_env_fn=wrapper.wrap_for_brax_training,
    )

    def progress(num_steps, metrics):
        reward = metrics.get(
            "eval/episode_reward"
        )

        if reward is not None:
            print(
                f"{num_steps:,} steps"
                f" | reward = {reward:.3f}",
                flush=True,
            )

    print("Starting CubebotBalance training", flush=True)

    make_inference_fn, params, metrics = train_fn(
        environment=env,
        progress_fn=progress,
    )

    print("Training finished", flush=True)

    return (
        make_inference_fn,
        params,
        metrics,
    )


if __name__ == "__main__":
    main()
