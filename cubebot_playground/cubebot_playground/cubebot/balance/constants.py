from __future__ import annotations
import math
OBSERVATION_SIZE = 19
ACTION_SIZE = 12
SIM_DT = 0.002
CTRL_DT = 0.02
MAX_SERVO_TORQUE_NM = 0.34
MAX_SERVO_SPEED_DEG_S = 60.0
MAX_SERVO_SPEED_RAD_S = math.radians(MAX_SERVO_SPEED_DEG_S)
ACTION_SCALE_DEG = 20.0
ACTION_SCALE_RAD = math.radians(ACTION_SCALE_DEG)
HEIGHT_MIN_M = 0.08
HEIGHT_MAX_M = 0.13
JOINT_NAMES = (
    "revolute_10", "revolute_12", "revolute_3",
    "revolute_5", "revolute_6", "revolute_4",
    "revolute_7", "revolute_8", "revolute_1",
    "revolute_9", "revolute_11", "revolute_2",
)
DEFAULT_JOINT_POS = {
    "revolute_1": -1.57, "revolute_2": -1.57,
    "revolute_3": 1.57, "revolute_4": 1.57,
    "revolute_9": -1.57/2, "revolute_7": 1.57/2,
    "revolute_5": -1.57/2, "revolute_10": 1.57/2,
    "revolute_12": 0.0, "revolute_6": 0.0,
    "revolute_8": 0.0, "revolute_11": 0.0,
}
