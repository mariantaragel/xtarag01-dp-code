import trimesh
import json
import numpy as np
from scipy.spatial import cKDTree
import random
from glob import glob
import os
from pathlib import Path

NUMBER_OF_TOOTH_TO_REMOVE = 4
NUMBER_OF_POINTS = 8192
NUMBER_OF_MESHES = 75

def downsample_mesh(mesh, points):
    indices = random.sample(list(np.arange(len(mesh.vertices))), points)
    mask = np.full(mesh.vertices.shape[0], False)
    mask[indices] = True

    new_mesh = mesh.copy()
    new_mesh.update_vertices(mask)
    return new_mesh

if __name__ == "__main__":
    meshes = []
    for root, dirs, files in os.walk("../data/Orthodontic_dental_dataset/", topdown=True):
        meshes += [os.path.join(root, f) for f in files if f.endswith(".stl")]

    meshes = random.sample(meshes, NUMBER_OF_MESHES)

    for index, mesh_file in enumerate(meshes):
        json_file = mesh_file[:-3] + "json"

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

                Path(f"../small-dataset/{index}").mkdir(parents=True, exist_ok=True)
                np.savez_compressed(f"../small-dataset/{index}/orig-{teeth}.npz", pointcloud=orig_mesh.vertices)
                np.savez_compressed(f"../small-dataset/{index}/final-{teeth}.npz", pointcloud=final_mesh.vertices)

        print(f"Preccesed mesh: {index}")
