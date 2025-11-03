import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree
from tqdm import tqdm

N_PC_POINTS = 4096


def filter_meshes(mesh_files, must_included_teeth):
    filtered_mesh_files = []

    for mesh_file in mesh_files:
        json_file = mesh_file[:-3] + "json"
        with open(json_file, "r") as f:
            mesh_data = json.load(f)
            teeth = mesh_data["segmentation"].keys()
            if set(must_included_teeth).issubset(teeth):
                filtered_mesh_files.append(mesh_file)

    return filtered_mesh_files


def is_flipped(mesh):
    average_normal = np.mean(mesh.face_normals, axis=0)
    average_normal_normalized = average_normal / np.linalg.norm(average_normal)
    avg_x, avg_y, avg_z = average_normal_normalized
    return avg_x, avg_y, avg_z


def downsample_vertices(points, n_samples):
    replace = False
    if n_samples > len(points):
        replace = True
    idx = np.random.choice(len(points), n_samples, replace=replace)

    return points[idx]


def process_mesh(mesh, orig_index, file_name):
    vertices = downsample_vertices(mesh.vertices, N_PC_POINTS)
    if int(orig_index) > 277:
        vertices[:, 1] = -vertices[:, 1] + 20
        vertices[:, 2] = -vertices[:, 2]

    np.savez_compressed(file_name, pointcloud=vertices)


def remove_tooth(mesh_file, dataset_name, args, teeth_to_remove):
    segmentation_file = mesh_file[:-3] + "json"

    mesh_file_splitted = mesh_file[:-4].split("/")
    orig_name = mesh_file_splitted[-1]
    orig_index = mesh_file_splitted[-3]

    with open(segmentation_file, "r") as f:
        segmentation = json.load(f)

    with open(f"{args.data_dir}/train-test-split.json", "r") as f:
        split = json.load(f)

    file_name_prefix = f"{args.save_dir}/{dataset_name}/point-clouds"
    Path(file_name_prefix).mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(mesh_file)
    source_file_name = f"{file_name_prefix}/{orig_index}_{orig_name}_source.npz"
    process_mesh(mesh, orig_index, source_file_name)

    source_file_names = []
    target_file_names = []
    utterances = []
    object_classes = []
    splits = []
    source_object_classes = []
    target_object_classes = []

    train_indices = split["train"]
    other_indices = split["test"]
    test_indices = other_indices[: len(other_indices) // 2]
    val_indices = other_indices[len(other_indices) // 2 :]

    teeth = list(
        set(teeth_to_remove).intersection(set(segmentation["segmentation"].keys()))
    )

    tree = cKDTree(mesh.vertices)
    for tooth in teeth:
        vertices_to_remove = np.array(segmentation["segmentation"][tooth]["vertices"])
        distances, indices = tree.query(vertices_to_remove, k=1)

        mask = np.full(mesh.vertices.shape[0], True)
        mask[indices] = False

        new_mesh = mesh.copy()
        new_mesh.update_vertices(mask)

        target_file_name = (
            f"{file_name_prefix}/{orig_index}_{orig_name}_{tooth}_target.npz"
        )
        process_mesh(mesh, orig_index, target_file_name)

        utterance = f"remove tooth {tooth}"
        object_class = "upper jaw"
        source_object_class = "[]"
        target_object_class = f"[{tooth}]"

        source_file_names.append(source_file_name)
        target_file_names.append(target_file_name)
        utterances.append(utterance)
        object_classes.append(object_class)
        source_object_classes.append(source_object_class)
        target_object_classes.append(target_object_class)

        if orig_index in train_indices:
            splits.append("train")
        elif orig_index in test_indices:
            splits.append("test")
        elif orig_index in val_indices:
            splits.append("val")
        else:
            splits.append("train")

    return (
        source_file_names,
        target_file_names,
        utterances,
        object_classes,
        source_object_classes,
        target_object_classes,
        splits,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Path to point cloud data")
    parser.add_argument("--dataset_name", help="Name of the dataset")
    parser.add_argument("--save_dir", help="Where to store created dataset")
    args = parser.parse_args()

    teeth_to_remove = ["11", "21"]
    dataset_name = args.dataset_name
    path = args.data_dir + "/Orthodontic_dental_dataset"

    meshes = [f.path for f in os.scandir(path) if f.is_dir()]
    meshes_upper_ori = [m + "/ori/U_Ori.stl" for m in meshes]
    meshes_upper_final = [m + "/final/U_Final.stl" for m in meshes]

    meshes_to_process = sorted(meshes_upper_final + meshes_upper_ori)

    source_uids = []
    target_uids = []
    utterances = []
    object_classes = []
    splits = []
    source_object_classes = []
    target_object_classes = []

    for mesh_file in tqdm(meshes_to_process):
        (
            source_uid,
            target_uid,
            utterance,
            object_class,
            source_object_class,
            target_object_class,
            split,
        ) = remove_tooth(mesh_file, dataset_name, args, teeth_to_remove)

        source_uids += source_uid
        target_uids += target_uid
        utterances += utterance
        object_classes += object_class
        source_object_classes += source_object_class
        target_object_classes += target_object_class
        splits += split

    df = pd.DataFrame(
        {
            "source_uid": source_uids,
            "target_uid": target_uids,
            "utterance": utterances,
            "object_class": object_classes,
            "source_object_class": source_object_classes,
            "target_object_class": target_object_classes,
            "source_unary_split": splits,
            "target_unary_split": splits,
            "listening_split": splits,
            "changeit_split": splits,
        }
    )

    split_folder = f"{args.save_dir}/{dataset_name}/splits"
    Path(split_folder).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{split_folder}/raw-split.csv", index=False)
