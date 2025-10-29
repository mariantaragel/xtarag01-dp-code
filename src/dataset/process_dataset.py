import argparse
from pathlib import Path

import pandas as pd

from language.vocabulary import build_vocab

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", help="Name of the dataset")
    parser.add_argument("--save_dir", help="Where to store created dataset")
    args = parser.parse_args()

    dataset_name = args.dataset_name

    df = pd.read_csv(f"{args.save_dir}/{dataset_name}/splits/raw-split.csv")

    df["assignmentid"] = "DP"
    df["saliency"] = 0
    df["tokens"] = df.utterance.str.split()
    df["hard_context"] = False
    df["target_object_class"] = df["target_object_class"]
    df["source_object_class"] = df["source_object_class"]
    df["source_dataset"] = "Orthodontic dental datatset"
    df["target_dataset"] = "Orthodontic dental datatset"

    token_list = df.tokens
    vocab = build_vocab(token_list, 0)

    df["tokens_encoded"] = df["tokens"].apply(vocab.encode)

    Path(f"{args.save_dir}/{dataset_name}/vocabulary/").mkdir(
        parents=True, exist_ok=True
    )

    df.to_csv(f"{args.save_dir}/{dataset_name}/splits/processed-split.csv", index=False)
    vocab.save(f"{args.save_dir}/{dataset_name}/vocabulary/vocabulary.pkl")
