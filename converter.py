import mujoco

model = mujoco.MjModel.from_xml_path("/Users/vglukhov/PROJECTS/sim/CubeBot_Playground/cubebot/urdf/cubebot_mesh_collision.urdf")

xml = mujoco.mj_saveLastXML("robot.xml", model)