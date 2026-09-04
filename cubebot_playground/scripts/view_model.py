import math
import time

import mujoco
import mujoco.viewer

# from cubebot_playground.cubebot.constants import MODEL_PATH

MODEL_PATH = "scene.xml"

MAX_SERVO_SPEED_DEG_S = 60.0
MAX_SERVO_SPEED_RAD_S = math.radians(MAX_SERVO_SPEED_DEG_S)


model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

# disable servos
model.actuator_gainprm[:] = 0
model.actuator_biasprm[:] = 0


def joint_qpos_adr(joint_name: str) -> int:
    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )
    if jid < 0:
        raise ValueError(f"Joint not found: {joint_name}")
    return model.jnt_qposadr[jid]


def set_initial_joint_position(joint_name: str, position_rad: float) -> None:
    data.qpos[joint_qpos_adr(joint_name)] = position_rad


# Initial pose.
set_initial_joint_position("revolute_1", -1.57)
set_initial_joint_position("revolute_2", -1.57)
set_initial_joint_position("revolute_3", 1.57)
set_initial_joint_position("revolute_4", 1.57)

set_initial_joint_position("revolute_9", -1.57 / 2)  # FL
set_initial_joint_position("revolute_7", 1.57 / 2)  # FR
set_initial_joint_position("revolute_5", -1.57 / 2)  # BR
set_initial_joint_position("revolute_10", 1.57 / 2)  # BL

mujoco.mj_forward(model, data)


# actuator name -> actuator id
actuator_ids = {}
for actuator_id in range(model.nu):
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_id,
    )
    actuator_ids[name] = actuator_id


# Desired target angle for each servo.
# This can change instantly from your gait/IK code.
desired_targets = {}

# Actual command sent to MuJoCo.
# This is rate-limited to 60 deg/s.
commanded_targets = {}


for actuator_id in range(model.nu):
    joint_id = model.actuator_trnid[actuator_id, 0]
    qpos_adr = model.jnt_qposadr[joint_id]
    current = float(data.qpos[qpos_adr])

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_id,
    )

    desired_targets[name] = current
    commanded_targets[name] = current
    data.ctrl[actuator_id] = current


def set_servo_target_deg(actuator_name: str, angle_deg: float) -> None:
    """Set desired servo position. The 60 deg/s speed limit is applied later."""
    if actuator_name not in actuator_ids:
        raise ValueError(f"Actuator not found: {actuator_name}")

    desired_targets[actuator_name] = math.radians(angle_deg)


def update_servo_commands(dt: float) -> None:
    """Move commands toward desired targets at no more than 60 deg/s."""
    max_step = MAX_SERVO_SPEED_RAD_S * dt

    for name, actuator_id in actuator_ids.items():
        current_cmd = commanded_targets[name]
        desired = desired_targets[name]

        delta = desired - current_cmd
        delta = max(-max_step, min(max_step, delta))

        current_cmd += delta

        commanded_targets[name] = current_cmd
        data.ctrl[actuator_id] = current_cmd


# Example:
# set_servo_target_deg("servo_revolute_10", 20.0)


with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = model.stat.center
    viewer.cam.distance = 0.4
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20

    while viewer.is_running():
        step_start = time.time()

        update_servo_commands(model.opt.timestep)

        mujoco.mj_step(model, data)
        viewer.sync()

        elapsed = time.time() - step_start
        sleep_time = model.opt.timestep - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
