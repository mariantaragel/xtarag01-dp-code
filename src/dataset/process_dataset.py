import pandas as pd

from language.vocabulary import build_vocab

df = pd.read_csv(
    "/home/marian/DP/removed-front-teeth-v2/splits/removed-front-teeth-split.csv"
)

df["assignmentid"] = "DP"
df["tokens"] = df["utterance"].apply(list)

token_list = df.utterance
vocab = build_vocab(token_list, 0)

df["tokens_encoded"] = df["tokens"].apply(vocab.encode)

print(df.head())

df.to_csv(
    "/home/marian/DP/removed-front-teeth-v2/splits/removed-front-teeth-split-processed.csv"
)

vocab.save("vocabulary.pkl")
