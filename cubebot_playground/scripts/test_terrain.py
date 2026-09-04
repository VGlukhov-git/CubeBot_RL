from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PACKAGE_ROOT = PROJECT_ROOT / "cubebot_playground"
CUBEBOT_DIR = PACKAGE_ROOT / "cubebot"

CUBEBOT_XML = CUBEBOT_DIR / "cubebot.xml"

sys.path.insert(0, str(PROJECT_ROOT))


from cubebot_playground.cubebot.terrains.rough import (
    generate_rough_terrain_xml,
)


# ============================================================
# TERRAIN CONFIG
# ============================================================

SEED = 10

ROWS = 8
COLS = 8

TILE_SIZE = 0.06

MIN_HEIGHT = -0.005
MAX_HEIGHT = 0.025
TERRAIN_ROLL_DEG = 15.0
TERRAIN_PITCH_DEG = 15.0
FRICTION = 1.5


# ============================================================
# GENERATE TERRAIN
# ============================================================

terrain_xml = generate_rough_terrain_xml(
    seed=SEED,
    rows=ROWS,
    cols=COLS,
    tile_size=TILE_SIZE,
    min_height=MIN_HEIGHT,
    max_height=MAX_HEIGHT,

    terrain_roll_deg=TERRAIN_ROLL_DEG,
    terrain_pitch_deg=TERRAIN_PITCH_DEG,

    friction=FRICTION,
)


# ============================================================
# CREATE TEST SCENE
# ============================================================

#
# Important:
#
# cubebot.xml remains the source of truth for the robot.
# We do not duplicate the robot XML here.
#

scene_xml = f"""
<mujoco model="cubebot_terrain_test">

    <compiler angle="radian"/>

    <option
        timestep="0.002"
        gravity="0 0 -9.81"
    />

    <visual>
        <headlight
            diffuse="0.7 0.7 0.7"
            ambient="0.3 0.3 0.3"
            specular="0.1 0.1 0.1"
        />
    </visual>


    <include file="{CUBEBOT_XML.as_posix()}"/>


    <worldbody>

        <light
            name="main_light"
            pos="0 0 2"
            dir="0 0 -1"
        />

        {terrain_xml}

    </worldbody>

</mujoco>
"""


# ============================================================
# WRITE GENERATED SCENE
# ============================================================

#
# Put the generated XML inside the project instead of /tmp.
#
# This makes mesh/include paths much easier to debug.
#

GENERATED_DIR = PROJECT_ROOT / ".generated"

GENERATED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SCENE_PATH = GENERATED_DIR / "terrain_test.xml"

SCENE_PATH.write_text(
    scene_xml,
    encoding="utf-8",
)


print("Scene:", SCENE_PATH)
print("Robot:", CUBEBOT_XML)

print()
print("Terrain")
print("  seed:", SEED)
print("  size:", f"{ROWS} x {COLS}")
print("  tile:", TILE_SIZE)
print("  min height:", MIN_HEIGHT)
print("  max height:", MAX_HEIGHT)


# ============================================================
# LOAD MUJOCO
# ============================================================

model = mujoco.MjModel.from_xml_path(
    str(SCENE_PATH)
)

data = mujoco.MjData(model)


print()
print("Model")
print("  bodies:", model.nbody)
print("  joints:", model.njnt)
print("  actuators:", model.nu)
print("  geoms:", model.ngeom)
print("  nq:", model.nq)
print("  nv:", model.nv)


# ============================================================
# JOINT HELPERS
# ============================================================

def joint_id(name: str) -> int:

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid == -1:
        raise ValueError(
            f"Joint not found: {name}"
        )

    return jid


def set_joint_position(
    name: str,
    position: float,
) -> None:

    jid = joint_id(name)

    qpos_adr = model.jnt_qposadr[jid]

    data.qpos[qpos_adr] = position


# ============================================================
# INITIAL STANDING POSE
# ============================================================

#
# These are the same angles we were already using
# in the standalone robot test.
#

# Lower joints

set_joint_position(
    "revolute_1",
    -1.57,
)

set_joint_position(
    "revolute_2",
    -1.57,
)

set_joint_position(
    "revolute_3",
    1.57,
)

set_joint_position(
    "revolute_4",
    1.57,
)


# Hip joints

set_joint_position(
    "revolute_9",
    -1.57 / 2,
)  # FL

set_joint_position(
    "revolute_7",
    1.57 / 2,
)  # FR

set_joint_position(
    "revolute_5",
    -1.57 / 2,
)  # BR

set_joint_position(
    "revolute_10",
    1.57 / 2,
)  # BL


# ============================================================
# PLACE ROBOT ABOVE TERRAIN
# ============================================================

root_joint = joint_id(
    "root_freejoint"
)

root_qpos = model.jnt_qposadr[
    root_joint
]


#
# Freejoint qpos:
#
# 0 X
# 1 Y
# 2 Z
# 3 qw
# 4 qx
# 5 qy
# 6 qz
#

data.qpos[root_qpos + 0] = 0.0
data.qpos[root_qpos + 1] = 0.0


#
# Terrain can reach MAX_HEIGHT = 0.025 m.
#
# Our robot standing root height is roughly 0.1025 m
# above a flat surface.
#
# Therefore:
#
# 0.1025 + 0.025 + small clearance
#

ROBOT_STAND_HEIGHT = 0.1025

START_CLEARANCE = 0.005

data.qpos[root_qpos + 2] = (
    ROBOT_STAND_HEIGHT
    + MAX_HEIGHT
    + START_CLEARANCE
)


# Identity quaternion

data.qpos[root_qpos + 3] = 1.0
data.qpos[root_qpos + 4] = 0.0
data.qpos[root_qpos + 5] = 0.0
data.qpos[root_qpos + 6] = 0.0


# ============================================================
# FORWARD KINEMATICS
# ============================================================

mujoco.mj_forward(
    model,
    data,
)


# ============================================================
# INITIALIZE SERVO TARGETS
# ============================================================

#
# Without this every position actuator starts with ctrl=0,
# which would immediately pull the robot toward the zero pose.
#

for actuator_id in range(
    model.nu
):

    joint = model.actuator_trnid[
        actuator_id,
        0,
    ]

    if joint < 0:
        continue

    qpos_adr = model.jnt_qposadr[
        joint
    ]

    data.ctrl[actuator_id] = (
        data.qpos[qpos_adr]
    )


# ============================================================
# DEBUG INITIAL STATE
# ============================================================

print()
print(
    "Robot start Z:",
    data.qpos[root_qpos + 2],
)

print(
    "Initial contacts:",
    data.ncon,
)


# ============================================================
# VIEWER
# ============================================================

with mujoco.viewer.launch_passive(
    model,
    data,
) as viewer:

    viewer.cam.lookat[:] = [
        0,
        0,
        0.05,
    ]

    viewer.cam.distance = 0.7
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -25


    # --------------------------------------------------------
    # SIMULATION LOOP
    # --------------------------------------------------------

    while viewer.is_running():

        step_start = time.time()


        mujoco.mj_step(
            model,
            data,
        )


        viewer.sync()


        # Run approximately in real time.

        elapsed = (
            time.time()
            - step_start
        )

        sleep_time = (
            model.opt.timestep
            - elapsed
        )

        if sleep_time > 0:

            time.sleep(
                sleep_time
            )