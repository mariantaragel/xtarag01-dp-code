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


def downsample_vertices(points, n_samples):
    replace = False
    if n_samples > len(points):
        replace = True
    idx = np.random.choice(len(points), n_samples, replace=replace)

    return points[idx]


def flip_vertices_by_yz(vertices, orig_index):
    if int(orig_index) > 277:
        vertices[:, 1] = -vertices[:, 1]
        vertices[:, 2] = -vertices[:, 2]

    return vertices


def process_mesh(mesh, orig_index, segmentation, file_name):
    vertices_tooth_11 = np.array(segmentation["segmentation"]["11"]["vertices"])
    vertices_tooth_11 = flip_vertices_by_yz(vertices_tooth_11, orig_index)
    mean_11_x, mean_11_y, mean_11_z = np.mean(vertices_tooth_11, axis=0)

    vertices_tooth_21 = np.array(segmentation["segmentation"]["21"]["vertices"])
    vertices_tooth_21 = flip_vertices_by_yz(vertices_tooth_21, orig_index)
    mean_21_x, mean_21_y, mean_21_z = np.mean(vertices_tooth_21, axis=0)

    mean_x = (mean_11_x + mean_21_x) / 2
    mean_y = (mean_11_y + mean_21_y) / 2
    mean_z = (mean_11_z + mean_21_z) / 2

    print(orig_index, mean_x, mean_y, mean_z)

    vertices = downsample_vertices(mesh.vertices, N_PC_POINTS)
    vertices = flip_vertices_by_yz(vertices, orig_index)

    vertices[:, 0] = vertices[:, 0] - mean_x
    vertices[:, 1] = vertices[:, 1] - mean_y
    vertices[:, 2] = vertices[:, 2] - mean_z

    np.savez_compressed(file_name, pointcloud=vertices)


def get_class(act):
    upper_right_teeth = [str(i) for i in range(11, 19)]
    upper_left_teeth = [str(i) for i in range(21, 29)]
    upper_teeth = set(upper_left_teeth + upper_right_teeth)
    miss_teeth = upper_teeth - act
    cls = "miss_" + "_".join(sorted(miss_teeth))
    return cls


def filter_meshes(mesh_files):
    filtered_mesh_files = []

    for mesh_file in mesh_files:
        json_file = mesh_file[:-3] + "json"
        with open(json_file, "r") as f:
            mesh_data = json.load(f)
            teeth = mesh_data["segmentation"].keys()
            if set(["11", "21"]).issubset(teeth):
                filtered_mesh_files.append(mesh_file)

    return filtered_mesh_files


def remove_tooth(mesh_file, args):
    segmentation_file = mesh_file[:-3] + "json"

    mesh_file_splitted = mesh_file[:-4].split("/")
    orig_name = mesh_file_splitted[-1]
    orig_index = mesh_file_splitted[-3]

    with open(segmentation_file, "r") as f:
        segmentation = json.load(f)

    with open(args.split_file, "r") as f:
        split = json.load(f)

    file_name_prefix = f"{args.save_dir}/{args.dataset_name}/point-clouds"
    Path(file_name_prefix).mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(mesh_file)
    source_file_name = f"/{orig_index}_{orig_name}_source.npz"
    source_file_name_path = file_name_prefix + source_file_name
    process_mesh(mesh, orig_index, segmentation, source_file_name_path)

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

    patient_teeth = set(segmentation["segmentation"].keys())
    teeth = list(set(args.teeth_to_remove).intersection(patient_teeth))

    tree = cKDTree(mesh.vertices)
    for tooth in teeth:
        vertices_to_remove = np.array(segmentation["segmentation"][tooth]["vertices"])
        distances, indices = tree.query(vertices_to_remove, k=1)

        mask = np.full(mesh.vertices.shape[0], True)
        mask[indices] = False

        new_mesh = mesh.copy()
        new_mesh.update_vertices(mask)

        target_file_name = f"/{orig_index}_{orig_name}_{tooth}_target.npz"
        target_file_name_path = file_name_prefix + target_file_name
        process_mesh(new_mesh, orig_index, segmentation, target_file_name_path)

        utterance = f"remove tooth {tooth}"
        object_class = "upper jaw"
        source_object_class = "baseline"
        target_object_class = f"miss_{tooth}"

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
            print("NOT in splits:", orig_index)  # 0736, 0818
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
    parser.add_argument("--split_file", help="Path to split file")
    parser.add_argument(
        "--teeth_to_remove",
        type=str,
        nargs="*",
        default=["11", "12", "21", "22"],
        help="Which teeth will be removed from original shape",
    )
    args = parser.parse_args()

    meshes = [f.path for f in os.scandir(args.data_dir) if f.is_dir()]
    meshes_upper_final = [m + "/final/U_Final.stl" for m in meshes]
    meshes_upper_final = filter_meshes(meshes_upper_final)
    meshes_upper_ori = [m + "/ori/U_Ori.stl" for m in meshes]
    meshes_upper_ori = filter_meshes(meshes_upper_ori)
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
        ) = remove_tooth(mesh_file, args)

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

    split_folder = f"{args.save_dir}/{args.dataset_name}/splits"
    Path(split_folder).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{split_folder}/raw-split.csv", index=False)
