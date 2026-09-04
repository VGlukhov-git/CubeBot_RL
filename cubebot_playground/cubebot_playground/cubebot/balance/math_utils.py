import jax.numpy as jp
def quat_rotate_inverse(q, v):
    w = q[0]; xyz = q[1:4]
    t = 2.0 * jp.cross(xyz, v)
    return v - w * t + jp.cross(xyz, t)
