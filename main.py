import trimesh
import json
import numpy as np
from scipy.spatial import cKDTree
import random

NUMBER_OF_TOOTH_TO_REMOVE = 4
NUMBER_OF_POINTS = 8192

def downsample_mesh(mesh, points):
    indices = random.sample(list(np.arange(len(mesh.vertices))), points)
    mask = np.full(mesh.vertices.shape[0], False)
    mask[indices] = True

    new_mesh = mesh.copy()
    new_mesh.update_vertices(mask)
    return new_mesh

mesh_file = "../../Downloads/0001/0001/ori/L_Ori.stl"
json_file = "../../Downloads/0001/0001/ori/L_Ori.json"

mesh = trimesh.load(mesh_file)

with open(json_file, "r") as f:
    mesh_data = json.load(f)
    tooth = list(mesh_data["segmentation"].keys())
    tooth = random.sample(tooth, NUMBER_OF_TOOTH_TO_REMOVE)
    for teeth in tooth:
        vertices_to_remove = np.array(mesh_data["segmentation"][teeth]["vertices"])
        tree = cKDTree(mesh.vertices)
        distances, indices = tree.query(vertices_to_remove, k=1)

        mask = np.full(mesh.vertices.shape[0], True)
        mask[indices] = False

        new_mesh = mesh.copy()
        new_mesh.update_vertices(mask)

        orig_mesh = downsample_mesh(mesh, NUMBER_OF_POINTS)
        final_mesh = downsample_mesh(new_mesh, NUMBER_OF_POINTS)

        np.savez_compressed(f"../../Downloads/orig-{teeth}.npz", pointcloud=orig_mesh.vertices)
        np.savez_compressed(f"../../Downloads/final-{teeth}.npz", pointcloud=final_mesh.vertices)

        print(f"Removing teeth: {teeth}")
        print(f"Number of vertices orig: {orig_mesh.vertices.shape[0]}")
        print(f"Number of vertices final: {final_mesh.vertices.shape[0]}")
