from pathlib import Path
import xml.etree.ElementTree as ET

src = Path("cubebot/urdf/cubebot.urdf")
dst = Path("cubebot/urdf/cubebot_view.urdf")

tree = ET.parse(src)
root = tree.getroot()

for link in root.findall("link"):
    if link.find("collision") is not None:
        continue

    visual = link.find("visual")
    if visual is None:
        continue

    collision = ET.fromstring(ET.tostring(visual))
    collision.tag = "collision"

    # remove material from collision
    material = collision.find("material")
    if material is not None:
        collision.remove(material)

    link.append(collision)

tree.write(dst, encoding="utf-8", xml_declaration=True)
print(f"Saved {dst}")