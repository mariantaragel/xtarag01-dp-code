"""
Created by Marián Tarageľ (xtarag01)
"""

import json
import numpy as np
import polyscope as ps
from pathlib import Path

json_paths = [
    "data/26aa1f6ba53d4daa0c03.pts.json",
    "data/e21ece4e117357b70426.pts.json",
]

colors = [
    (0xE3 / 255, 0x9C / 255, 0x1C / 255),  # #E39C1C
    (0x1C / 255, 0xE3 / 255, 0x38 / 255),  # #1CE338
]

ps.init()
ps.set_up_dir("z_up")
ps.set_ground_plane_mode("shadow_only")

for json_path, color in zip(json_paths, colors):
    with open(json_path, "r") as f:
        data = json.load(f)

    point_cloud = np.array(data, dtype=np.float32)
    point_cloud[:, 0] = -point_cloud[:, 0]

    # Normalize to unit sphere
    centroid = point_cloud.mean(axis=0)
    point_cloud -= centroid
    max_dist = np.linalg.norm(point_cloud, axis=1).max()
    point_cloud /= max_dist

    name = Path(json_path).stem
    pc = ps.register_point_cloud(name, point_cloud)
    pc.set_color(color)

ps.show()