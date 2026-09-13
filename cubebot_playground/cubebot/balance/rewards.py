import jax.numpy as jp
def compute_rewards(*, projected_gravity, body_height, desired_height, body_ang_vel, joint_pos, default_joint_pos, joint_vel, action, previous_action, config):
    upright_error = jp.sum(jp.square(projected_gravity[:2]))
    upright = jp.exp(-config.upright_sigma*upright_error)
    height = jp.exp(-config.height_sigma*jp.square(body_height-desired_height))
    ang = jp.sum(jp.square(body_ang_vel[:2]))
    jvel = jp.mean(jp.square(joint_vel))
    pose = jp.mean(jp.square(joint_pos-default_joint_pos))
    arate = jp.mean(jp.square(action-previous_action))
    total = (config.reward_upright*upright + config.reward_height*height - config.penalty_ang_vel*ang - config.penalty_joint_vel*jvel - config.penalty_pose*pose - config.penalty_action_rate*arate)
    return total, {"upright":upright,"height":height,"ang_vel_penalty":ang,"joint_vel_penalty":jvel,"pose_penalty":pose,"action_rate_penalty":arate}
