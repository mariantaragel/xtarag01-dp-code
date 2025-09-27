import pandas as pd

from language.vocabulary import build_vocab

df = pd.read_csv(
    "/home/marian/DP/removed-front-teeth-v2/splits/removed-front-teeth-split.csv"
)

df["tokens"] = df["utterance"].apply(list)

token_list = df.utterance
vocab = build_vocab(token_list, 0)
print(vocab.word2idx)

df["encoded_tokens"] = df["tokens"].apply(vocab.encode)

print(df.head())

df.to_csv(
    "/home/marian/DP/removed-front-teeth-v2/splits/removed-front-teeth-split-processed.csv"
)

vocab.save("vocabulary.pkl")
