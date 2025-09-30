import pandas as pd

from language.vocabulary import build_vocab

df = pd.read_csv(
    "/home/marian/DP/removed-front-teeth-v2/splits/removed-front-teeth-split.csv"
)

df["assignmentid"] = "DP"
df["saliency"] = 0
df["tokens"] = df.utterance.str.split()
df["hard_context"] = False
df["target_object_class"] = df["object_class"]
df["source_object_class"] = df["object_class"]
df["source_dataset"] = "Orthodontic dental datatset"
df["target_dataset"] = "Orthodontic dental datatset"

token_list = df.tokens
vocab = build_vocab(token_list, 0)

df["tokens_encoded"] = df["tokens"].apply(vocab.encode)

print(df.head())

df.to_csv(
    "/home/marian/DP/removed-front-teeth-v2/splits/removed-front-teeth-split-processed.csv"
)

vocab.save("vocabulary.pkl")
