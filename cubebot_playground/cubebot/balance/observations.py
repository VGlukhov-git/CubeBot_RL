import jax.numpy as jp
from .math_utils import quat_rotate_inverse
def normalized_height_command(desired_height, height_min, height_max):
    mid = 0.5*(height_min+height_max); half = 0.5*(height_max-height_min)
    return jp.clip((desired_height-mid)/half, -1.0, 1.0)
def build_observation(*, root_quat, root_ang_vel_world, servo_command, default_servo_command, action_scale, desired_height, height_min, height_max, ang_vel_scale):
    projected_gravity = quat_rotate_inverse(root_quat, jp.array([0.0,0.0,-1.0]))
    body_ang_vel = quat_rotate_inverse(root_quat, root_ang_vel_world)
    normalized_servo_command = jp.clip(
        (servo_command-default_servo_command)/action_scale,
        -1.0,
        1.0,
    )
    return jp.concatenate([
        projected_gravity,
        body_ang_vel*ang_vel_scale,
        normalized_servo_command,
        jp.array([normalized_height_command(desired_height,height_min,height_max)]),
    ])
