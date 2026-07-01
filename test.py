import mujoco

model = mujoco.MjModel.from_xml_string("""
<mujoco>
    <worldbody>
        <geom type="plane" size="5 5 .1"/>
        <body pos="0 0 1">
            <freejoint/>
            <geom type="sphere" size="0.1"/>
        </body>
    </worldbody>
</mujoco>
""")

data = mujoco.MjData(model)

viewer = mujoco.viewer.launch_passive(model, data)

while viewer.is_running():
    mujoco.mj_step(model, data)
    viewer.sync()