import mujoco
import mujoco.viewer
import time
# Load the robot
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# Disable gravity for visual check
# model.opt.gravity[:] = [0, 0, 0]

# Print model info
print("Bodies:", model.nbody)
print("Joints:", model.njnt)
print("Geoms:", model.ngeom)
print("Meshes:", model.nmesh)
print("Model extent:", model.stat.extent)
print("Model center:", model.stat.center)

def set_init_joint_positions(model, name, position):
    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name
    )
    adr = model.jnt_qposadr[jid]
    data.qpos[adr] = position
    mujoco.mj_forward(model, data)
    
# A
set_init_joint_positions(model, "revolute_1", -1.57)
set_init_joint_positions(model, "revolute_2", -1.57)
set_init_joint_positions(model, "revolute_3", 1.57)
set_init_joint_positions(model, "revolute_4", 1.57)

#C
set_init_joint_positions(model, "revolute_9", -1.57/2) #FL
set_init_joint_positions(model, "revolute_7", 1.57/2) #FR
set_init_joint_positions(model, "revolute_5", -1.57/2) #BR
set_init_joint_positions(model, "revolute_10", 1.57/2) #BL


with mujoco.viewer.launch_passive(model, data) as viewer:
    # Point camera at robot
    viewer.cam.lookat[:] = model.stat.center
    viewer.cam.distance = 0.4
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)