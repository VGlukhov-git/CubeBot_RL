from ml_collections import config_dict
from . import constants
def default_config():
    return config_dict.create(
        ctrl_dt=constants.CTRL_DT, sim_dt=constants.SIM_DT, episode_length=1000, action_repeat=1, vision=False,
        action_scale=constants.ACTION_SCALE_RAD, max_servo_speed=constants.MAX_SERVO_SPEED_RAD_S,
        height_min=constants.HEIGHT_MIN_M, height_max=constants.HEIGHT_MAX_M,
        obs_ang_vel_scale=0.25,
        reward_upright=3.0, reward_height=1.0, penalty_ang_vel=0.10, penalty_joint_vel=0.01,
        penalty_pose=0.05, penalty_action_rate=0.02, upright_sigma=6.0, height_sigma=500.0,
        min_body_height=0.04, min_projected_gravity_z=-0.50,
    )
