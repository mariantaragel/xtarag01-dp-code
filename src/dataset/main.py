import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree

NUMBER_OF_TOOTH_TO_REMOVE = 5
NUMBER_OF_POINTS = 8192
NUMBER_OF_MESHES = 150


def downsample_vertices(points, n_samples):
    replace = False
    if n_samples > len(points):
        replace = True
    idx = np.random.choice(len(points), n_samples, replace=replace)

    return points[idx]


if __name__ == "__main__":
    path = "/home/marian/DP/data/Orthodontic_dental_dataset/"
    meshes = [f.path for f in os.scandir(path) if f.is_dir()]

    source_file_names = []
    target_file_names = []
    utterances = []
    object_classes = []

    for mesh_index, mesh_file in enumerate(meshes):
        json_file = mesh_file[:-3] + "json"
        orig_file_name_parts = mesh_file[:-4].split("/")
        orig_name = orig_file_name_parts[-1]
        orig_index = orig_file_name_parts[-3]

        mesh = trimesh.load(mesh_file)

        with open(json_file, "r") as f:
            mesh_data = json.load(f)
            teeth = list(mesh_data["segmentation"].keys())
            teeth = random.sample(teeth, NUMBER_OF_TOOTH_TO_REMOVE)
            for tooth_index, tooth in enumerate(teeth):
                vertices_to_remove = np.array(
                    mesh_data["segmentation"][tooth]["vertices"]
                )
                tree = cKDTree(mesh.vertices)
                distances, indices = tree.query(vertices_to_remove, k=1)

                mask = np.full(mesh.vertices.shape[0], True)
                mask[indices] = False

                new_mesh = mesh.copy()
                new_mesh.update_vertices(mask)

                source_vertices = downsample_vertices(mesh.vertices, NUMBER_OF_POINTS)
                target_vertices = downsample_vertices(
                    new_mesh.vertices, NUMBER_OF_POINTS
                )

                index = mesh_index * NUMBER_OF_TOOTH_TO_REMOVE + tooth_index
                file_name_prefix = "../small-dataset-1"
                Path(file_name_prefix + f"/{index}").mkdir(parents=True, exist_ok=True)

                source_file_name = f"/{index}/{orig_index}_{orig_name}_source.npz"
                target_file_name = f"/{index}/{orig_index}_{orig_name}_target.npz"
                utterance = f"remove tooth {tooth}"
                object_class = "dentition"

                np.savez_compressed(
                    file_name_prefix + source_file_name, pointcloud=source_vertices
                )
                np.savez_compressed(
                    file_name_prefix + target_file_name, pointcloud=target_vertices
                )

                source_file_names.append(source_file_name)
                target_file_names.append(target_file_name)
                utterances.append(utterance)
                object_classes.append(object_class)

                print(f"Precessed mesh: {index}")

    total_data_count = NUMBER_OF_MESHES * NUMBER_OF_TOOTH_TO_REMOVE
    rnd_numbers = np.random.randint(1, 100, total_data_count)
    mask1 = rnd_numbers <= 70
    mask2 = (rnd_numbers > 70) & (rnd_numbers <= 85)
    mask3 = rnd_numbers > 85

    splits = np.empty(rnd_numbers.shape, dtype="<U5")
    splits[mask1] = "train"
    splits[mask2] = "test"
    splits[mask3] = "val"

    df = pd.DataFrame(
        {
            "split": splits,
            "source_file_name": source_file_names,
            "target_file_name": target_file_names,
            "utterance": utterances,
            "object_class": object_classes,
        }
    )

    df.to_csv("../split.csv", index=False)
