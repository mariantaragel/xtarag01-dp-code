"""
Created by Marián Tarageľ (xtarag01)
"""

import argparse

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", help="Name of the dataset")
    parser.add_argument("--save_dir", help="Where to store created dataset")
    args = parser.parse_args()

    dataset_name = args.dataset_name

    df = pd.read_csv(f"{args.save_dir}/{dataset_name}/splits/raw-split.csv")

    df_source = df[["source_uid", "source_object_class", "split"]].rename(
        columns={
            "source_uid": "file_name",
            "source_object_class": "object_class"
        }
    )
    df_target = df[["target_uid", "target_object_class", "split"]].rename(
        columns={
            "target_uid": "file_name",
            "target_object_class": "object_class"
        }
    )

    df_all = pd.concat([df_source, df_target])
    df_all = df_all.drop_duplicates().reset_index(drop=True)
    df_all["model_uid"] = df_all["file_name"]

    df_all.to_csv(f"{args.save_dir}/{dataset_name}/splits/unary-split.csv", index=False)
