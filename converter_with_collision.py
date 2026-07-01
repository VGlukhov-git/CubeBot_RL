from pathlib import Path
import xml.etree.ElementTree as ET

src = Path("cubebot/urdf/cubebot.urdf")
dst = Path("cubebot/urdf/cubebot_mesh_collision.urdf")

tree = ET.parse(src)
root = tree.getroot()

for link in root.findall("link"):
    visual = link.find("visual")
    if visual is None:
        continue

    # remove old collision blocks
    for old_collision in list(link.findall("collision")):
        link.remove(old_collision)

    collision = ET.fromstring(ET.tostring(visual))
    collision.tag = "collision"

    material = collision.find("material")
    if material is not None:
        collision.remove(material)

    link.append(collision)

tree.write(dst, encoding="utf-8", xml_declaration=True)
print(f"Saved {dst}")