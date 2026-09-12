import mujoco
import numpy as np
import pytest
from cubebot_playground.cubebot.standup import CubebotStandUp, StandUpConfig


def legacy_config(**kwargs):
    kwargs.setdefault("episode_seconds", 6.0)
    return StandUpConfig(
        height_agnostic=False,
        world_level=False,
        start_airborne=False,
        slope_roll_degrees=0,
        slope_pitch_degrees=0,
        command_height_offset=0,
        sim_dt=0.002,
        **kwargs,
    )


def test_reset_geometry_and_partial_reset():
    env = CubebotStandUp(3, legacy_config(angle_noise=0), num_threads=2)
    np.testing.assert_allclose(env.geom_xpos[:, env.feet, 2], 0.008, atol=1e-6)
    assert env.observation().shape == (3, 14)
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
    env = CubebotStandUp(1, legacy_config(angle_noise=0))
    before = env.observation()
    env.qpos[:, env.qadr] += 0.1
    env.qpos[:, 2] += 0.01
    env.qvel[:] = 3
    np.testing.assert_array_equal(env.observation(), before)
    env.qpos[:, 3:7] = [np.cos(0.2), 0, 0, np.sin(0.2)]
    np.testing.assert_array_equal(env.observation(), before)
    env.command += 0.01
    assert not np.array_equal(env.observation()[:, 2:], before[:, 2:])


def test_step_limits_timeouts_and_parallel_equivalence():
    config = legacy_config(angle_noise=0, episode_seconds=0.2)
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
        CubebotStandUp(1, legacy_config(target_height=height))
    with pytest.raises(ValueError):
        CubebotStandUp(config=legacy_config(target_height=0.13))


def test_smooth_teacher_can_stand_at_hardware_mass_and_torque():
    env = CubebotStandUp(1, legacy_config(angle_noise=0))
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
    env = CubebotStandUp(1, legacy_config(angle_noise=0))
    hips = np.array([0, 3, 6, 9])
    for _ in range(80):
        env.step(np.ones((1, 12)))
        assert np.all(env.command >= env.low - 1e-12)
        assert np.all(env.command <= env.high + 1e-12)
        assert np.max(abs(env.command[0, hips] - env.initial_pose[hips])) < 0.081


@pytest.mark.parametrize("friction", [0.7, 1.5])
def test_foot_floor_contact_uses_explicit_physical_friction(friction):
    env = CubebotStandUp(1, legacy_config(friction=friction, angle_noise=0))
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
    env = CubebotStandUp(2, legacy_config(prestep=True, angle_noise=0), num_threads=2)
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
    assert obs.shape == (2, 14)
    before = env.qpos[1].copy()
    env.reset([0])
    assert env.preparing.tolist() == [True, False]
    np.testing.assert_array_equal(env.qpos[1], before)


def test_residual_policy_preserves_baseline_and_bounds_corrections():
    import torch
    from cubebot_playground.cubebot.standup.train import Policy

    env = CubebotStandUp(1, legacy_config(angle_noise=0))
    policy = Policy(
        env.standing_pose,
        env.config.max_speed,
        residual_scale=0.01,
        feedback_gain=1.0,
    )
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

    env = CubebotStandUp(1, legacy_config())
    policy = Policy(env.standing_pose, env.config.max_speed, 0.012)
    obs = torch.from_numpy(env.observation())
    path = tmp_path / "policy.pt"
    save_checkpoint(path, policy, env, 0, 0, {})
    saved = torch.load(path, weights_only=True)
    restored = Policy.from_checkpoint(saved)
    torch.testing.assert_close(policy.action(obs), restored.action(obs))
    for key in ("actor.0.weight", "critic.0.weight"):
        weight = saved["policy"][key]
        yaw_column = torch.randn((len(weight), 1))
        saved["policy"][key] = torch.cat(
            (weight[:, :2], yaw_column, weight[:, 2:]), dim=1
        )
    migrated = Policy.from_checkpoint(saved)
    torch.testing.assert_close(policy.action(obs), migrated.action(obs))
    legacy = Policy()
    loaded_legacy = Policy.from_checkpoint({"policy": legacy.state_dict()})
    torch.testing.assert_close(loaded_legacy.action(obs), torch.tanh(legacy.actor(obs)))


@pytest.mark.parametrize("prestep", [False, True])
def test_zero_residual_evaluation_completes_prestep_and_stands(prestep):
    import torch
    from cubebot_playground.cubebot.standup.train import Policy, evaluate

    torch.set_num_threads(1)
    env = CubebotStandUp(2, legacy_config(prestep=prestep), num_threads=2)
    policy = Policy(
        env.standing_pose, env.config.max_speed, feedback_gain=1.0 if prestep else 0.8
    )
    metrics = evaluate(policy, env)
    assert metrics["success"] == 1.0
    assert metrics["stable_tail"] > 0.5
    assert metrics["slip"] < 0.01
    assert abs(metrics["height"] - 0.08) < 0.008


def test_large_foothold_drift_is_penalized_and_terminates_rise():
    env = CubebotStandUp(1, legacy_config(angle_noise=0))
    env.qpos[:, 0] += 0.08
    env.batch.forward()
    _, _, terminated, _, info = env.step(np.zeros((1, 12)))
    assert terminated[0]
    assert info["lateral_cost"][0] > 2.0  # old capped penalty had an escape plateau


def test_sloped_reset_has_airborne_feet_and_independent_planes():
    env = CubebotStandUp(4, StandUpConfig(angle_noise=0), num_threads=2)
    slopes = np.array([[-15.0, -15.0], [15.0, 15.0], [0.0, 0.0], [-8.0, 12.0]])
    env.reset(slopes=slopes)
    floor_matrices = env.batch.bind("geom_xmat")
    env.batch.forward()
    np.testing.assert_allclose(
        floor_matrices[:, env.floor].reshape(-1, 3, 3), env.plane_rotation, atol=1e-12
    )
    local_feet = env.plane_points(env.geom_xpos[:, env.feet])
    np.testing.assert_allclose(local_feet[:, :, 2] - 0.008, 0.005, atol=1e-6)
    height = np.einsum("ni,ni->n", env.qpos[:, :3], env.plane_rotation[:, :, 2])
    np.testing.assert_allclose(height, 0.0224, atol=1e-12)
    np.testing.assert_array_equal(env.model.opt.gravity, [0, 0, -9.81])
    assert env.observation().shape == (4, 14)
    assert not np.allclose(env.observation()[0, :3], env.observation()[1, :3])
    old_qpos = env.qpos.copy()
    old_quats = env.floor_quats.copy()
    env.reset([1], slopes=np.array([[4.0, -5.0]]))
    keep = [0, 2, 3]
    np.testing.assert_array_equal(env.qpos[keep], old_qpos[keep])
    np.testing.assert_array_equal(env.floor_quats[keep], old_quats[keep])
    np.testing.assert_array_equal(env.slope_degrees[keep], slopes[keep])


def test_simultaneous_landing_and_rise_on_all_extreme_slopes():
    import torch
    from cubebot_playground.cubebot.standup.train import Policy, evaluate

    torch.set_num_threads(1)
    env = CubebotStandUp(
        9,
        StandUpConfig(
            height_agnostic=False,
            world_level=False,
            dynamic_slope=False,
            episode_seconds=6.0,
            angle_noise=0,
        ),
        num_threads=4,
    )
    assert len(env.prestep_commands) == 70
    assert env.prestep_swing[10].all()
    for leg in range(4):
        assert (
            np.linalg.norm(
                env.prestep_commands[20, 3 * leg : 3 * leg + 3]
                - env.initial_pose[3 * leg : 3 * leg + 3]
            )
            > 0.001
        )
    policy = Policy(env.standing_pose, env.config.max_speed, feedback_gain=1.0)
    metrics = evaluate(policy, env)
    assert metrics["success"] == 1.0
    assert len(metrics["cases"]) == 9
    assert {(c["roll"], c["pitch"]) for c in metrics["cases"]} == {
        (r, p) for r in [-15.0, 0.0, 15.0] for p in [-15.0, 0.0, 15.0]
    }
    assert metrics["slip"] < 0.005
    assert metrics["stable_tail"] > 0.9


def test_random_slope_limits_and_validation():
    env = CubebotStandUp(
        16, StandUpConfig(slope_roll_degrees=7, slope_pitch_degrees=12), seed=42
    )
    assert (abs(env.slope_degrees[:, 0]) <= 7).all()
    assert (abs(env.slope_degrees[:, 1]) <= 12).all()
    assert np.unique(env.slope_degrees, axis=0).shape[0] == 16
    before = env.slope_degrees.copy()
    env.reset()
    assert not np.array_equal(before, env.slope_degrees)
    with pytest.raises(ValueError):
        CubebotStandUp(1, StandUpConfig(slope_roll_degrees=26))
    with pytest.raises(ValueError):
        env.reset([0], slopes=np.array([[0.0, np.nan]]))


def test_inward_foot_region_allows_thirty_mm_but_penalizes_outward_travel():
    env = CubebotStandUp(1)
    anchors = env.standing_footholds[None].copy()
    direction = anchors[..., :2] / np.linalg.norm(
        anchors[..., :2], axis=-1, keepdims=True
    )
    inside = anchors.copy()
    inside[..., :2] -= 0.03 * direction
    error, travel = env.foothold_error(inside, anchors, 0.03)
    np.testing.assert_allclose(error, 0, atol=1e-12)
    np.testing.assert_allclose(travel, 0.03)
    outside = anchors.copy()
    outside[..., :2] += 0.03 * direction
    assert (env.foothold_error(outside, anchors, 0.03)[0] > 0.029).all()
    sideways = anchors.copy()
    sideways[..., :2] += 0.03 * direction[..., ::-1] * [1, -1]
    assert (env.foothold_error(sideways, anchors, 0.03)[0] > 0).all()
    # In the landing phase no inward allowance is supplied.
    np.testing.assert_allclose(env.foothold_error(inside, anchors, 0)[0], 0.03)


def test_horizontal_controller_uses_imu_and_roundtrips(tmp_path):
    import torch
    from cubebot_playground.cubebot.standup.train import Policy, save_checkpoint

    env = CubebotStandUp(1)
    policy = Policy(
        env.standing_pose,
        env.config.max_speed,
        leveling_matrix=env.leveling_matrix,
        initial_reference=env.initial_pose,
    )
    obs = torch.zeros((1, 14))
    obs[:, 2:] = torch.tensor(env.standing_pose / np.pi)
    flat = policy.action(obs).detach()
    obs[:, 0] = 0.1 / np.pi
    tilted = policy.action(obs).detach()
    assert not torch.allclose(flat, tilted)
    path = tmp_path / "level.pt"
    save_checkpoint(path, policy, env, 0, 0, {})
    restored = Policy.from_checkpoint(torch.load(path, weights_only=True))
    torch.testing.assert_close(restored.action(obs), policy.action(obs))


def test_world_horizontal_goal_does_not_reward_parallel_sloped_body_as_upright():
    env = CubebotStandUp(1, StandUpConfig(angle_noise=0))
    env.reset(slopes=np.array([[15.0, 0.0]]))
    _, _, _, _, info = env.step(np.zeros((1, 12)))
    assert info["world_tilt_degrees"][0] > 14
    assert info["upright_reward"][0] < 0.7
    assert not info["stable"][0]


def test_stable_lower_stance_succeeds_without_matching_eight_cm():
    env = CubebotStandUp(
        1, StandUpConfig(slope_roll_degrees=0, slope_pitch_degrees=0, angle_noise=0)
    )
    feet = env.standing_footholds.copy()
    feet[:, :2] -= (
        0.02 * feet[:, :2] / np.linalg.norm(feet[:, :2], axis=-1, keepdims=True)
    )
    lower_pose = env.solve_pose(0.066, feet)
    for _ in range(len(env.prestep_commands) + 400):
        action = np.clip((lower_pose - env.command) / env.config.max_speed, -0.95, 0.95)
        _, _, terminated, _, info = env.step(action)
        assert not terminated[0]
    assert 0.045 <= info["height"][0] < 0.072
    assert info["success"][0]


def test_lying_is_not_a_success_with_relaxed_height_goal():
    env = CubebotStandUp(1, StandUpConfig(slope_roll_degrees=0, slope_pitch_degrees=0))
    _, _, _, _, info = env.step(np.zeros((1, 12)))
    assert not info["stable"][0]
    assert info["posture_reward"][0] < 0.2


def test_faster_rise_keeps_servo_limits_and_reaches_stance_earlier():
    import torch
    from cubebot_playground.cubebot.standup.train import Policy

    config = StandUpConfig(
        dynamic_slope=False,
        slope_roll_degrees=0,
        slope_pitch_degrees=0,
        angle_noise=0,
    )

    def rise_time(gain):
        env = CubebotStandUp(1, config)
        policy = Policy(
            env.standing_pose,
            env.config.max_speed,
            feedback_gain=gain,
            leveling_matrix=env.leveling_matrix,
            initial_reference=env.initial_pose,
        )
        obs = env.reset(slopes=np.zeros((1, 2)), slope_targets=np.zeros((1, 2)))
        previous = env.command.copy()
        old_velocity = env.velocity.copy()
        reached = None
        for step in range(len(env.prestep_commands) + 100):
            action = policy.action(torch.from_numpy(obs)).detach().numpy()
            obs, _, _, _, info = env.step(action)
            assert (
                np.max(abs(env.command - previous))
                <= config.max_speed * config.ctrl_dt + 1e-10
            )
            assert (
                np.max(abs(env.velocity - old_velocity))
                <= config.max_acceleration * config.ctrl_dt + 1e-10
            )
            previous = env.command.copy()
            old_velocity = env.velocity.copy()
            if (
                not info["preparing"][0]
                and info["height"][0] >= config.min_standing_height
            ):
                reached = (step + 1 - len(env.prestep_commands)) * config.ctrl_dt
                break
        return reached

    assert rise_time(1.5) < rise_time(1.0)


def test_surface_changes_smoothly_only_after_robot_has_risen():
    env = CubebotStandUp(
        1,
        StandUpConfig(
            slope_roll_degrees=15,
            slope_pitch_degrees=15,
            slope_change_start=0.5,
            slope_change_duration=1.0,
            angle_noise=0,
        ),
    )
    env.reset(slopes=np.zeros((1, 2)), slope_targets=np.array([[15.0, -10.0]]))
    for _ in range(len(env.prestep_commands) + 25):
        env.step(np.zeros((1, 12)))
    np.testing.assert_allclose(env.slope_degrees, 0, atol=1e-12)
    samples = []
    for _ in range(51):
        env.step(np.zeros((1, 12)))
        samples.append(env.slope_degrees[0].copy())
    samples = np.asarray(samples)
    assert np.all(np.diff(samples[:, 0]) >= 0)
    assert np.all(np.diff(samples[:, 1]) <= 0)
    np.testing.assert_allclose(samples[-1], [15, -10], atol=1e-12)
    from scipy.spatial.transform import Rotation

    floor_quat_wxyz = env.floor_quats[0, env.floor]
    np.testing.assert_allclose(
        Rotation.from_quat(floor_quat_wxyz[[1, 2, 3, 0]]).as_matrix(),
        env.plane_rotation[0],
        atol=1e-12,
    )


def test_controller_reacts_to_post_rise_surface_tilt():
    from cubebot_playground.cubebot.standup.train import Policy, evaluate

    env = CubebotStandUp(9, StandUpConfig(angle_noise=0), num_threads=4)
    policy = Policy(
        env.standing_pose,
        env.config.max_speed,
        residual_scale=0.05,
        feedback_gain=1.5,
        leveling_matrix=env.leveling_matrix,
        initial_reference=env.initial_pose,
    )
    metrics = evaluate(policy, env)
    assert metrics["level_fraction"] >= 7 / 9
    assert metrics["world_tilt_degrees"] < 5
    assert metrics["slip"] < 0.01


def test_hardware_runtime_reproduces_environment_command_filter(tmp_path):
    import torch
    from cubebot_playground.cubebot.standup.hardware import HardwareController
    from cubebot_playground.cubebot.standup.train import Policy, save_checkpoint

    config = StandUpConfig(
        dynamic_slope=False,
        slope_roll_degrees=0,
        slope_pitch_degrees=0,
        angle_noise=0,
    )
    env = CubebotStandUp(1, config)
    policy = Policy(
        env.standing_pose,
        env.config.max_speed,
        residual_scale=0.05,
        feedback_gain=1.5,
        leveling_matrix=env.leveling_matrix,
        initial_reference=env.initial_pose,
    )
    checkpoint = tmp_path / "hardware.pt"
    save_checkpoint(checkpoint, policy, env, 0, 0, {})
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    hardware = HardwareController(saved)
    obs = env.reset(slopes=np.zeros((1, 2)), slope_targets=np.zeros((1, 2)))

    for _ in range(len(env.prestep_commands) + 10):
        roll_pitch = obs[0, :2] * np.pi
        command, _ = hardware.step(roll_pitch)
        action = policy.action(torch.from_numpy(obs)).detach().numpy()
        obs, _, _, _, _ = env.step(action)
        np.testing.assert_allclose(command, env.command[0], atol=2e-7)
