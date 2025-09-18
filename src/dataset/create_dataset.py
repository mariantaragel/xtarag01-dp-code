import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree
from tqdm import tqdm

N_PC_POINTS = 8192


def downsample_vertices(points, n_samples):
    replace = False
    if n_samples > len(points):
        replace = True
    idx = np.random.choice(len(points), n_samples, replace=replace)

    return points[idx]


def remove_tooth(mesh_file, tooth, index):
    json_file = mesh_file[:-3] + "json"

    mesh = trimesh.load(mesh_file)

    with open(json_file, "r") as f:
        mesh_data = json.load(f)

        if tooth in mesh_data["segmentation"].keys():
            vertices_to_remove = np.array(mesh_data["segmentation"][tooth]["vertices"])
        else:
            return
        tree = cKDTree(mesh.vertices)
        distances, indices = tree.query(vertices_to_remove, k=1)

        mask = np.full(mesh.vertices.shape[0], True)
        mask[indices] = False

        new_mesh = mesh.copy()
        new_mesh.update_vertices(mask)

        source_vertices = downsample_vertices(mesh.vertices, N_PC_POINTS)
        target_vertices = downsample_vertices(new_mesh.vertices, N_PC_POINTS)

        file_name_prefix = "/home/marian/DP/removed-front-teeth/point-clouds"
        Path(file_name_prefix + f"/{index}").mkdir(parents=True, exist_ok=True)

        orig_file_name_parts = mesh_file[:-4].split("/")
        orig_name = orig_file_name_parts[-1]
        orig_index = orig_file_name_parts[-3]

        source_file_name = f"/{index}/{orig_index}_{orig_name}_source.npz"
        target_file_name = f"/{index}/{orig_index}_{orig_name}_target.npz"

        np.savez_compressed(
            file_name_prefix + source_file_name, pointcloud=source_vertices
        )
        np.savez_compressed(
            file_name_prefix + target_file_name, pointcloud=target_vertices
        )

        utterance = f"remove tooth {tooth}"
        if "U" in orig_name:
            object_class = "upper_jaw"
        elif "L" in orig_name:
            object_class = "lower_jaw"

        return source_file_name, target_file_name, utterance, object_class, orig_index


if __name__ == "__main__":
    path = "/home/marian/DP/data/Orthodontic_dental_dataset/"
    meshes = [f.path for f in os.scandir(path) if f.is_dir()][:150]

    with open("/home/marian/DP/data/train-test-split.json", "r") as f:
        split_data = json.load(f)
    train_indices = split_data["train"]
    other_indices = split_data["test"]
    test_indices = other_indices[: len(other_indices) // 2]
    val_indices = other_indices[len(other_indices) // 2 :]

    source_uids = []
    target_uids = []
    utterances = []
    object_classes = []
    splits = []

    index = 1

    for mesh_folder in tqdm(meshes):
        mesh_ori = mesh_folder + "/ori/U_Ori.stl"
        mesh_final = mesh_folder + "/final/U_Final.stl"
        if mesh_folder == "/home/marian/DP/data/Orthodontic_dental_dataset/0903":
            continue
        for mesh_file in [mesh_ori, mesh_final]:
            for tooth in ["11", "21"]:
                source_uid, target_uid, utterance, object_class, orig_index = (
                    remove_tooth(mesh_file, tooth, index)
                )
                index += 1
                source_uids.append(source_uid)
                target_uids.append(target_uid)
                utterances.append(utterance)
                object_classes.append(object_class)
                if orig_index in train_indices:
                    splits.append("train")
                elif orig_index in test_indices:
                    splits.append("test")
                elif orig_index in val_indices:
                    splits.append("val")

    df = pd.DataFrame(
        {
            "source_uid": source_uids,
            "target_uid": target_uids,
            "utterance": utterances,
            "object_class": object_classes,
            "source_unary_split": splits,
            "target_unary_split": splits,
            "listening_split": splits,
            "changeit_split": splits,
        }
    )

    split_folder = "/home/marian/DP/removed-front-teeth/splits"
    Path(split_folder).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{split_folder}/removed-front-teeth-split.csv", index=False)
