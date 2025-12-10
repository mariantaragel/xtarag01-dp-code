import argparse
import json
import os
import random
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


def process_mesh(mesh, orig_index, file_name):
    vertices = downsample_vertices(mesh.vertices, N_PC_POINTS)
    vertices = flip_vertices_by_yz(vertices, orig_index)
    mean_x, mean_y, mean_z = np.mean(vertices, axis=0)

    vertices[:, 0] -= mean_x
    vertices[:, 1] -= mean_y
    vertices[:, 2] -= mean_z

    np.savez_compressed(file_name, pointcloud=vertices)


def get_utterance(removed_tooth):
    if removed_tooth == "11":
        tooth_texts = ["8 universal", "right 1", "11 FDI", "right central incisor"]
    elif removed_tooth == "12":
        tooth_texts = ["7 universal", "right 2", "12 FDI", "right lateral incisor"]
    elif removed_tooth == "13":
        tooth_texts = ["6 universal", "right 3", "13 FDI", "right cuspid"]
    elif removed_tooth == "14":
        tooth_texts = ["5 universal", "right 4", "14 FDI", "right first bicuspid"]
    elif removed_tooth == "15":
        tooth_texts = ["4 universal", "right 5", "15 FDI", "right second bicuspid"]
    elif removed_tooth == "16":
        tooth_texts = ["3 universal", "right 6", "16 FDI", "right first molar"]
    elif removed_tooth == "17":
        tooth_texts = ["2 universal", "right 7", "17 FDI", "right second molar"]
    elif removed_tooth == "18":
        tooth_texts = ["1 universal", "right 8", "18 FDI", "right third molar"]
    elif removed_tooth == "21":
        tooth_texts = ["9 universal", "left 1", "21 FDI", "left central incisor"]
    elif removed_tooth == "22":
        tooth_texts = ["10 universal", "left 2", "22 FDI", "left lateral incisor"]
    elif removed_tooth == "23":
        tooth_texts = ["11 universal", "left 3", "23 FDI", "left cuspid"]
    elif removed_tooth == "24":
        tooth_texts = ["12 universal", "left 4", "24 FDI", "left first bicuspid"]
    elif removed_tooth == "25":
        tooth_texts = ["13 universal", "left 5", "25 FDI", "left second bicuspid"]
    elif removed_tooth == "26":
        tooth_texts = ["14 universal", "left 6", "26 FDI", "left first molar"]
    elif removed_tooth == "27":
        tooth_texts = ["15 universal", "left 7", "27 FDI", "left second molar"]
    elif removed_tooth == "28":
        tooth_texts = ["16 universal", "left 8", "28 FDI", "left third molar"]

    verbs = ["remove", "extract", "delete", "pull"]

    verb = random.choice(verbs)
    tooth_text = random.choice(tooth_texts)
    if verb == "remove" or verb == "delete":
        template = random.choice([
            f"{verb} {tooth_text}",
            f"Please {verb} {tooth_text}",
            f"{tooth_text} needs to be {verb}d"
        ])
    else:
        template = random.choice([
            f"{verb} {tooth_text}",
            f"Please {verb} {tooth_text}",
            f"{tooth_text} needs to be {verb}ed"
        ])

    return template


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
    process_mesh(mesh, orig_index, source_file_name_path)

    source_file_names = []
    target_file_names = []
    utterances = []
    object_classes = []
    splits = []
    source_object_classes = []
    target_object_classes = []

    train_indices = split["train"]
    test_indices = split["test"]
    val_indices = split["val"]

    patient_teeth = set(segmentation["segmentation"].keys())
    teeth = list(set(args.teeth_to_remove).intersection(patient_teeth))

    tree = cKDTree(mesh.vertices)
    for tooth in teeth:
        vertices_to_remove = np.array(segmentation["segmentation"][tooth]["vertices"])
        _, indices = tree.query(vertices_to_remove, k=1)

        mask = np.full(mesh.vertices.shape[0], True)
        mask[indices] = False

        new_mesh = mesh.copy()
        new_mesh.update_vertices(mask)

        target_file_name = f"/{orig_index}_{orig_name}_{tooth}_target.npz"
        target_file_name_path = file_name_prefix + target_file_name
        process_mesh(new_mesh, orig_index, target_file_name_path)

        utterance = get_utterance(tooth)
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
    meshes_upper_ori = [m + "/ori/U_Ori.stl" for m in meshes]
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
