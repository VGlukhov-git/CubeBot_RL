from __future__ import annotations

import math
import random


def generate_rough_terrain_xml(
    *,
    seed: int = 1,
    rows: int = 8,
    cols: int = 8,
    tile_size: float = 0.06,
    min_height: float = -0.005,
    max_height: float = 0.025,
    terrain_roll_deg: float = 0.0,
    terrain_pitch_deg: float = 0.0,
    friction: float = 1.5,
) -> str:

    rng = random.Random(seed)

    geoms: list[str] = []

    x_offset = (rows - 1) * tile_size / 2
    y_offset = (cols - 1) * tile_size / 2

    half_z = 0.03

    for row in range(rows):
        for col in range(cols):

            x = row * tile_size - x_offset
            y = col * tile_size - y_offset

            surface_z = rng.uniform(
                min_height,
                max_height,
            )

            center_z = surface_z - half_z

            geom = f"""
        <geom
            name="terrain_{row}_{col}"
            type="box"

            pos="{x:.6f} {y:.6f} {center_z:.6f}"

            size="
                {tile_size / 2:.6f}
                {tile_size / 2:.6f}
                {half_z:.6f}
            "

            friction="{friction:.3f} 0.005 0.0001"

            rgba="0.35 0.35 0.35 1"
        />
"""

            geoms.append(geom)

    roll = math.radians(
        terrain_roll_deg
    )

    pitch = math.radians(
        terrain_pitch_deg
    )

    return f"""
    <body
        name="terrain_root"
        euler="{roll:.6f} {pitch:.6f} 0">

        {''.join(geoms)}

    </body>
"""
