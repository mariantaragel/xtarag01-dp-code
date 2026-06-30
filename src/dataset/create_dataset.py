"""
Created by Marián Tarageľ (xtarag01)
"""

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
    vertices -= np.mean(vertices, axis=0)

    max_dist = np.max(np.sqrt(np.sum(vertices**2, axis=1)))
    if max_dist > 0:
        vertices /= max_dist

    np.savez_compressed(file_name, pointcloud=vertices)


def get_utterance(op, removed_tooth, args):
    tooth_map = {
        # Upper Right
        "11": ["8 universal", "right 1", "11 FDI", "upper right central incisor"],
        "12": ["7 universal", "right 2", "12 FDI", "upper right lateral incisor"],
        "13": ["6 universal", "right 3", "13 FDI", "upper right canine"],
        "14": ["5 universal", "right 4", "14 FDI", "upper right first premolar"],
        "15": ["4 universal", "right 5", "15 FDI", "upper right second premolar"],
        "16": ["3 universal", "right 6", "16 FDI", "upper right first molar"],
        "17": ["2 universal", "right 7", "17 FDI", "upper right second molar"],
        "18": ["1 universal", "right 8", "18 FDI", "upper right third molar"],
        # Upper Left
        "21": ["9 universal", "left 1", "21 FDI", "upper left central incisor"],
        "22": ["10 universal", "left 2", "22 FDI", "upper left lateral incisor"],
        "23": ["11 universal", "left 3", "23 FDI", "upper left canine"],
        "24": ["12 universal", "left 4", "24 FDI", "upper left first premolar"],
        "25": ["13 universal", "left 5", "25 FDI", "upper left second premolar"],
        "26": ["14 universal", "left 6", "26 FDI", "upper left first molar"],
        "27": ["15 universal", "left 7", "27 FDI", "upper left second molar"],
        "28": ["16 universal", "left 8", "28 FDI", "upper left third molar"],
        # Lower Left
        "31": ["24 universal", "left 1", "31 FDI", "lower left central incisor"],
        "32": ["23 universal", "left 2", "32 FDI", "lower left lateral incisor"],
        "33": ["22 universal", "left 3", "33 FDI", "lower left canine"],
        "34": ["21 universal", "left 4", "34 FDI", "lower left first premolar"],
        "35": ["20 universal", "left 5", "35 FDI", "lower left second premolar"],
        "36": ["19 universal", "left 6", "36 FDI", "lower left first molar"],
        "37": ["18 universal", "left 7", "37 FDI", "lower left second molar"],
        "38": ["17 universal", "left 8", "38 FDI", "lower left third molar"],
        # Lower Right
        "41": ["25 universal", "right 1", "41 FDI", "lower right central incisor"],
        "42": ["26 universal", "right 2", "42 FDI", "lower right lateral incisor"],
        "43": ["27 universal", "right 3", "43 FDI", "lower right canine"],
        "44": ["28 universal", "right 4", "44 FDI", "lower right first premolar"],
        "45": ["29 universal", "right 5", "45 FDI", "lower right second premolar"],
        "46": ["30 universal", "right 6", "46 FDI", "lower right first molar"],
        "47": ["31 universal", "right 7", "47 FDI", "lower right second molar"],
        "48": ["32 universal", "right 8", "48 FDI", "lower right third molar"],
    }

    if removed_tooth not in tooth_map and op != "align":
        return "align the dental arch"

    tooth_text = tooth_map[removed_tooth][args.notation] if op != "align" else ""

    if op == "extract":
        templates = [
            f"Please remove {tooth_text}",
            f"Extraction of {tooth_text} is required",
            f"Delete the {tooth_text} from the model",
            f"I need you to pull the {tooth_text}",
            f"Can you remove {tooth_text}",
            f"The patient needs {tooth_text} removed"
        ]
    elif op == "replace":
        templates = [
            f"Add a new tooth at {tooth_text}",
            f"Insert an implant for {tooth_text}",
            f"Replace the missing {tooth_text}",
            f"Place a restoration in the {tooth_text} position",
            f"We need to put {tooth_text} back",
            f"Fill the gap where {tooth_text} was"
        ]
    elif op == "align":
        templates = [
            "Align the teeth",
            "Please perform a standard alignment on the arch",
            "Correct the positioning of the teeth",
            "Straighten the dental arch",
            "The arch needs to be perfectly aligned"
        ]

    return random.choice(templates)


def process_patient_sample(mesh_path_ori, mesh_path_final, args):
    with open(args.split_file, "r") as f:
        split_json = json.load(f)

    index = mesh_path_ori.split("/")[-3]
    if index in split_json["train"]:
        split = "train"
    elif index in split_json["test"]:
        split = "test"
    elif index in split_json["val"]:
        split = "val"

    ori_name = mesh_path_ori.split("/")[-1][:-4]
    final_name = mesh_path_final.split("/")[-1][:-4]

    file_name_prefix = f"{args.save_dir}/{args.dataset_name}/point-clouds"
    Path(file_name_prefix).mkdir(parents=True, exist_ok=True)

    mesh_ori = trimesh.load(mesh_path_ori)
    mesh_final = trimesh.load(mesh_path_final)

    results = []

    # 1. Operation: Teeth Alignment
    ori_file = f"/{index}_{ori_name}_source.npz"
    process_mesh(mesh_ori, index, file_name_prefix + ori_file)
    # final_file = f"/{index}_{final_name}_align_target.npz"
    # process_mesh(mesh_final, index, file_name_prefix + final_file)

    # results.append({
    #     "source_uid": ori_file, "target_uid": final_file, 
    #     "utterance": get_utterance("align", None, args),
    #     "split": split, "object_class": "dental_arch",
    #     "source_object_class": "miss_aligned", "target_object_class": "aligned"
    # })

    # 2. Operation: Tooth Extraction & Replacement
    segmentation_file = mesh_path_ori[:-3] + "json"
    with open(segmentation_file, "r") as f:
        segmentation = json.load(f)

    tree = cKDTree(mesh_ori.vertices)
    for tooth in args.teeth_to_remove:
        if tooth in segmentation["segmentation"]:
            vertices_to_remove = np.array(segmentation["segmentation"][tooth]["vertices"])
            _, indices = tree.query(vertices_to_remove, k=1)

            mask = np.full(mesh_ori.vertices.shape[0], True)
            mask[indices] = False

            new_mesh = mesh_ori.copy()
            new_mesh.update_vertices(mask)

            target_file = f"/{index}_{ori_name}_{tooth}_extracted.npz"
            process_mesh(new_mesh, index, file_name_prefix + target_file)

            results.append({
                "source_uid": ori_file, "target_uid": target_file,
                "utterance": get_utterance("extract", tooth, args),
                "split": split, "object_class": "dental_arch",
                "source_object_class": "miss_aligned", "target_object_class": f"miss_{tooth}"
            })

            # results.append({
            #     "source_uid": target_file, "target_uid": ori_file,
            #     "utterance": get_utterance("replace", tooth, args),
            #     "split": split, "object_class": "dental_arch",
            #     "source_object_class": f"miss_{tooth}", "target_object_class": "miss_aligned"
            # })

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Path to point cloud data")
    parser.add_argument("--dataset_name", help="Name of the dataset")
    parser.add_argument("--save_dir", help="Where to store created dataset")
    parser.add_argument("--split_file", help="Path to split file")
    parser.add_argument("--notation", type=int, default=2)
    parser.add_argument("--teeth_to_remove", type=str, nargs="*", default=["11", "21"],
                        help="Which teeth will be removed from original shape")
    args = parser.parse_args()

    meshes = [f.path for f in os.scandir(args.data_dir) if f.is_dir()]
    meshes_ori = sorted([m + "/ori/U_Ori.stl" for m in meshes] + [m + "/ori/L_Ori.stl" for m in meshes])
    meshes_final = sorted([m + "/final/U_Final.stl" for m in meshes] + [m + "/final/L_Final.stl" for m in meshes])
    meshes_to_process = zip(meshes_ori, meshes_final)

    rows = []
    for mesh_path_ori, mesh_path_final in tqdm(meshes_to_process):
        results = process_patient_sample(mesh_path_ori, mesh_path_final, args)
        rows += results

    df = pd.DataFrame(rows)
    split_folder = f"{args.save_dir}/{args.dataset_name}/splits"
    Path(split_folder).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{split_folder}/raw-split.csv", index=False)
