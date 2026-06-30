"""
Created by Marián Tarageľ (xtarag01)
"""

import polyscope as ps
import trimesh
import numpy as np
import json
from scipy.spatial import cKDTree

mesh = trimesh.load("data/0005/ori/U_Ori.stl")

segmentation_file = "data/0005/final/U_Final.json"
with open(segmentation_file, "r") as f:
	segmentation = json.load(f)

tree = cKDTree(mesh.vertices)
vertices_to_remove = np.array(segmentation["segmentation"]["22"]["vertices"])
_, indices = tree.query(vertices_to_remove, k=1)

mask = np.full(mesh.vertices.shape[0], True)
mask[indices] = False

new_mesh = mesh.copy()
new_mesh.update_vertices(mask)

ps.init()

ps.register_surface_mesh("ori", mesh.vertices, mesh.faces, smooth_shade=True)
ps.register_surface_mesh("final", new_mesh.vertices, new_mesh.faces, smooth_shade=True)

ps.register_surface_mesh("ori 1", mesh.vertices, mesh.faces, smooth_shade=True)
ps.register_surface_mesh("final 1", new_mesh.vertices, new_mesh.faces, smooth_shade=True)

ps.show()