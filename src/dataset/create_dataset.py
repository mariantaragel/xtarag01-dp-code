import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree
from tqdm import tqdm

N_PC_POINTS = 4096


def downsample_vertices(points, n_samples):
    replace = False
    if n_samples > len(points):
        replace = True
    idx = np.random.choice(len(points), n_samples, replace=replace)

    return points[idx]


def remove_tooth(mesh_file):
    json_file = mesh_file[:-3] + "json"

    mesh = trimesh.load(mesh_file)
    source_vertices = downsample_vertices(mesh.vertices, N_PC_POINTS)

    file_name_prefix = "/home/marian/DP/removed-front-teeth-v3/point-clouds"
    Path(file_name_prefix).mkdir(parents=True, exist_ok=True)

    orig_file_name_parts = mesh_file[:-4].split("/")
    orig_name = orig_file_name_parts[-1]
    orig_index = orig_file_name_parts[-3]

    source_file_name = f"/{orig_index}_{orig_name}_source.npz"
    np.savez_compressed(file_name_prefix + source_file_name, pointcloud=source_vertices)

    source_file_names = []
    target_file_names = []
    utterances = []
    object_classes = []
    splits = []

    with open(json_file, "r") as f:
        mesh_data = json.load(f)

    with open("../../../data/train-test-split.json", "r") as f:
        split_data = json.load(f)
    train_indices = split_data["train"]
    other_indices = split_data["test"]
    test_indices = other_indices[: len(other_indices) // 2]
    val_indices = other_indices[len(other_indices) // 2 :]

    teeth = mesh_data["segmentation"].keys()

    for tooth in teeth:
        vertices_to_remove = np.array(mesh_data["segmentation"][tooth]["vertices"])
        tree = cKDTree(mesh.vertices)
        distances, indices = tree.query(vertices_to_remove, k=1)

        mask = np.full(mesh.vertices.shape[0], True)
        mask[indices] = False

        new_mesh = mesh.copy()
        new_mesh.update_vertices(mask)

        target_vertices = downsample_vertices(new_mesh.vertices, N_PC_POINTS)

        target_file_name = f"/{orig_index}_{orig_name}_{tooth}_target.npz"

        np.savez_compressed(
            file_name_prefix + target_file_name, pointcloud=target_vertices
        )

        utterance = f"remove tooth {tooth}"
        if "U" in orig_name:
            object_class = "upper_jaw"
        elif "L" in orig_name:
            object_class = "lower_jaw"

        source_file_names.append(source_file_name)
        target_file_names.append(target_file_name)
        utterances.append(utterance)
        object_classes.append(object_class)

        if orig_index in train_indices:
            splits.append("train")
        elif orig_index in test_indices:
            splits.append("test")
        elif orig_index in val_indices:
            splits.append("val")
        else:
            splits.append("unassigned")

    return (source_file_names, target_file_names, utterances, object_classes, splits)


if __name__ == "__main__":
    path = "/home/marian/DP/data/Orthodontic_dental_dataset/"
    meshes = [f.path for f in os.scandir(path) if f.is_dir()][:130]

    source_uids = []
    target_uids = []
    utterances = []
    object_classes = []
    splits = []

    for mesh_folder in tqdm(meshes):
        mesh_ori_u = mesh_folder + "/ori/U_Ori.stl"
        mesh_ori_l = mesh_folder + "/ori/L_Ori.stl"
        mesh_final_u = mesh_folder + "/final/U_Final.stl"
        mesh_final_l = mesh_folder + "/final/L_Final.stl"

        meshes_to_process = [mesh_ori_u, mesh_ori_l, mesh_final_u, mesh_final_l]

        for mesh_file in meshes_to_process:
            source_uid, target_uid, utterance, object_class, split = remove_tooth(
                mesh_file
            )

            source_uids += source_uid
            target_uids += target_uid
            utterances += utterance
            object_classes += object_class
            splits += split

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

    split_folder = "/home/marian/DP/removed-front-teeth-v3/splits"
    Path(split_folder).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{split_folder}/removed-front-teeth-split.csv", index=False)
