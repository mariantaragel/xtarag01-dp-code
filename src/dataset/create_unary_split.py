import pandas as pd

dataset_name = "removed-front-teeth-v5"

df = pd.read_csv(f"/home/marian/DP/{dataset_name}/splits/removed-front-teeth-split.csv")

df_source = df[["source_uid", "object_class", "source_unary_split"]].rename(
    columns={"source_uid": "file_name", "source_unary_split": "split"}
)
df_target = df[["target_uid", "object_class", "target_unary_split"]].rename(
    columns={"target_uid": "file_name", "target_unary_split": "split"}
)

df_all = pd.concat([df_source, df_target])
df_all = df_all.drop_duplicates().reset_index(drop=True)
df_all["model_uid"] = df_all["file_name"]

df_all.to_csv(f"/home/marian/DP/{dataset_name}/splits/unary-split.csv", index=False)
