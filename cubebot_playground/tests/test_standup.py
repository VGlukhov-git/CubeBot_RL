import mujoco
import numpy as np
import pytest
from cubebot_playground.cubebot.standup import CubebotStandUp, StandUpConfig


def test_reset_geometry_and_partial_reset():
    env = CubebotStandUp(3, StandUpConfig(angle_noise=0), num_threads=2)
    np.testing.assert_allclose(env.geom_xpos[:, env.feet, 2], 0.008, atol=1e-6)
    assert env.observation().shape == (3, 15)
    assert env.model.body_mass.sum() == pytest.approx(0.8)
    data = mujoco.MjData(env.model)
    data.qpos[:] = env.qpos[0]
    mujoco.mj_forward(env.model, data)
    assert min(c.dist for c in data.contact) >= -1e-6
    assert env.qpos[0, 2] == pytest.approx(0.0224)
    for _ in range(5):
        env.step(np.ones((3, 12)) * 0.1)
    before = env.qpos.copy()
    command = env.command.copy()
    env.reset([1])
    np.testing.assert_array_equal(env.qpos[[0, 2]], before[[0, 2]])
    np.testing.assert_array_equal(env.command[[0, 2]], command[[0, 2]])
    np.testing.assert_allclose(env.command[1], env.initial_pose)
    assert env.steps.tolist() == [5, 0, 5]


def test_observation_has_no_encoder_or_height_leak():
    env = CubebotStandUp(1, StandUpConfig(angle_noise=0))
    before = env.observation()
    env.qpos[:, env.qadr] += 0.1
    env.qpos[:, 2] += 0.01
    env.qvel[:] = 3
    np.testing.assert_array_equal(env.observation(), before)
    env.command += 0.01
    assert not np.array_equal(env.observation()[:, 3:], before[:, 3:])


def test_step_limits_timeouts_and_parallel_equivalence():
    config = StandUpConfig(angle_noise=0, episode_seconds=0.2)
    env = CubebotStandUp(2, config, num_threads=2)
    single = CubebotStandUp(1, config, num_threads=1)
    for step in range(10):
        previous = env.command.copy()
        old_velocity = env.velocity.copy()
        action = np.full((2, 12), 0.2 if step < 5 else -0.2)
        obs, reward, terminated, truncated, info = env.step(action)
        single.step(action[:1])
        assert np.isfinite(obs).all() and np.isfinite(reward).all()
        assert (
            np.max(abs(env.command - previous))
            <= config.max_speed * config.ctrl_dt + 1e-10
        )
        assert (
            np.max(abs(env.velocity - old_velocity))
            <= config.max_acceleration * config.ctrl_dt + 1e-10
        )
        assert not terminated.any()
        assert truncated.all() == (step == 9)
        np.testing.assert_allclose(env.qpos[0], single.qpos[0], atol=1e-12)
    assert (env.model.geom_friction[env.feet, 0] > 0).all()
    assert not info["success"].any()
    with pytest.raises(ValueError):
        env.step(np.full((2, 12), np.nan))


def test_reachable_height_validation():
    for height in [0.04, 0.085]:
        CubebotStandUp(1, StandUpConfig(target_height=height))
    with pytest.raises(ValueError):
        CubebotStandUp(config=StandUpConfig(target_height=0.13))


def test_smooth_teacher_can_stand_at_hardware_mass_and_torque():
    env = CubebotStandUp(1, StandUpConfig(angle_noise=0))
    for _ in range(300):
        action = np.clip(
            (env.standing_pose - env.command) / env.config.max_speed, -0.95, 0.95
        )
        _, _, terminated, _, info = env.step(action)
        assert not terminated[0]
    assert info["success"][0]
    assert abs(info["height"][0] - 0.08) < 0.008
    assert info["lateral_drift"][0] < 0.005


def test_command_corridor_prevents_large_sideways_hip_sweeps():
    env = CubebotStandUp(1, StandUpConfig(angle_noise=0))
    hips = np.array([0, 3, 6, 9])
    for _ in range(80):
        env.step(np.ones((1, 12)))
        assert np.all(env.command >= env.low - 1e-12)
        assert np.all(env.command <= env.high + 1e-12)
        assert np.max(abs(env.command[0, hips] - env.initial_pose[hips])) < 0.081


@pytest.mark.parametrize("friction", [0.7, 1.5])
def test_foot_floor_contact_uses_explicit_physical_friction(friction):
    env = CubebotStandUp(1, StandUpConfig(friction=friction, angle_noise=0))
    # Real generated contacts, not only configuration arrays.
    data = mujoco.MjData(env.model)
    data.qpos[:] = env.qpos[0]
    data.qpos[2] -= 1e-5
    mujoco.mj_forward(env.model, data)
    floor = env.model.geom("floor").id
    found = set()
    for contact in data.contact:
        pair = {contact.geom1, contact.geom2}
        if floor in pair and pair.intersection(env.feet):
            found.update(pair.intersection(env.feet))
            assert contact.dim == 4
            np.testing.assert_allclose(
                contact.friction, [friction, friction, 0.02, 0.0001, 0.0001]
            )
    assert found == set(env.feet)


def test_prestep_lifts_each_foot_then_allows_rise_and_full_episode():
    env = CubebotStandUp(2, StandUpConfig(prestep=True, angle_noise=0), num_threads=2)
    peak = np.zeros(4)
    for i in range(len(env.prestep_commands)):
        # The deterministic preparation controller ignores the policy's actions.
        _, _, terminated, truncated, info = env.step(np.array([[1] * 12, [-1] * 12]))
        assert info["preparing"].all()
        assert not terminated.any() and not truncated.any()
        assert not info["success"].any()
        assert env.prestep_swing[i].sum() <= 1
        np.testing.assert_allclose(env.qpos[0], env.qpos[1], atol=1e-12)
        peak = np.maximum(peak, env.geom_xpos[0, env.feet, 2] - 0.008)
    assert (peak > 0.004).all()
    np.testing.assert_allclose(
        env.geom_xpos[0, env.feet, :2], env.standing_footholds[:, :2], atol=0.001
    )
    assert not env.preparing.any()
    for i in range(300):
        action = np.clip(
            (env.standing_pose - env.command) / env.config.max_speed, -0.95, 0.95
        )
        obs, _, terminated, truncated, info = env.step(action)
        assert not terminated.any()
        assert truncated.all() == (i == 299)
    assert info["success"].all()
    assert obs.shape == (2, 15)
    before = env.qpos[1].copy()
    env.reset([0])
    assert env.preparing.tolist() == [True, False]
    np.testing.assert_array_equal(env.qpos[1], before)


def test_residual_policy_preserves_baseline_and_bounds_corrections():
    import torch
    from cubebot_playground.cubebot.standup.train import Policy

    env = CubebotStandUp(1, StandUpConfig(angle_noise=0))
    policy = Policy(env.standing_pose, env.config.max_speed, residual_scale=0.01)
    obs = torch.from_numpy(env.observation())
    baseline = np.clip(
        (env.standing_pose - env.command) / env.config.max_speed, -0.95, 0.95
    )
    np.testing.assert_allclose(policy.action(obs).detach().numpy(), baseline, atol=1e-7)
    for value in [-100.0, 100.0]:
        action = policy.action(obs, torch.full((1, 12), value)).detach().numpy()
        assert np.max(abs(action - baseline)) <= 0.01 / env.config.max_speed + 1e-7


def test_residual_checkpoint_roundtrip_and_legacy_loading(tmp_path):
    import torch
    from cubebot_playground.cubebot.standup.train import Policy, save_checkpoint

    env = CubebotStandUp(1)
    policy = Policy(env.standing_pose, env.config.max_speed, 0.012)
    obs = torch.from_numpy(env.observation())
    path = tmp_path / "policy.pt"
    save_checkpoint(path, policy, env, 0, 0, {})
    restored = Policy.from_checkpoint(torch.load(path, weights_only=True))
    torch.testing.assert_close(policy.action(obs), restored.action(obs))
    legacy = Policy()
    loaded_legacy = Policy.from_checkpoint({"policy": legacy.state_dict()})
    torch.testing.assert_close(loaded_legacy.action(obs), torch.tanh(legacy.actor(obs)))


@pytest.mark.parametrize("prestep", [False, True])
def test_zero_residual_evaluation_completes_prestep_and_stands(prestep):
    import torch
    from cubebot_playground.cubebot.standup.train import Policy, evaluate

    torch.set_num_threads(1)
    env = CubebotStandUp(2, StandUpConfig(prestep=prestep), num_threads=2)
    policy = Policy(
        env.standing_pose, env.config.max_speed, feedback_gain=1.0 if prestep else 0.8
    )
    metrics = evaluate(policy, env)
    assert metrics["success"] == 1.0
    assert metrics["stable_tail"] > 0.5
    assert metrics["slip"] < 0.01
    assert abs(metrics["height"] - 0.08) < 0.008


def test_large_foothold_drift_is_penalized_and_terminates_rise():
    env = CubebotStandUp(1, StandUpConfig(angle_noise=0))
    env.qpos[:, 0] += 0.08
    env.batch.forward()
    _, _, terminated, _, info = env.step(np.zeros((1, 12)))
    assert terminated[0]
    assert info["lateral_cost"][0] > 2.0  # old capped penalty had an escape plateau
