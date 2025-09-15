import torch
import torch.nn as nn


class Embedding(nn.Module):
    word_to_idx = {
        "remove tooth 11": 0,
        "remove tooth 12": 1,
        "remove tooth 13": 2,
        "remove tooth 14": 3,
        "remove tooth 15": 4,
        "remove tooth 16": 5,
        "remove tooth 17": 6,
        "remove tooth 18": 7,
        "remove tooth 21": 8,
        "remove tooth 22": 9,
        "remove tooth 23": 10,
        "remove tooth 24": 11,
        "remove tooth 25": 12,
        "remove tooth 26": 13,
        "remove tooth 27": 14,
        "remove tooth 28": 15,
        "remove tooth 31": 16,
        "remove tooth 32": 17,
        "remove tooth 33": 18,
        "remove tooth 34": 19,
        "remove tooth 35": 20,
        "remove tooth 36": 21,
        "remove tooth 37": 22,
        "remove tooth 38": 23,
        "remove tooth 41": 24,
        "remove tooth 42": 25,
        "remove tooth 43": 26,
        "remove tooth 44": 27,
        "remove tooth 45": 28,
        "remove tooth 46": 29,
        "remove tooth 47": 30,
        "remove tooth 48": 31,
    }

    def __init__(self, num_embeddings=32, embedding_dim=64):
        super(Embedding, self).__init__()

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def __call__(self, x):
        lookup_tensor = torch.tensor(
            list(map(self.word_to_idx.get, x)), dtype=torch.long, device="cuda:0"
        )
        return self.embedding(lookup_tensor)
