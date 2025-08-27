import torch.nn as nn
import torch

class Embedding(nn.Module):
    word_to_idx = {"remove teeth 11": 0,
                   "remove teeth 12": 1,
                   "remove teeth 13": 2,
                   "remove teeth 14": 3,
                   "remove teeth 15": 4,
                   "remove teeth 16": 5,
                   "remove teeth 17": 6,
                   "remove teeth 18": 7,
                   "remove teeth 21": 8,
                   "remove teeth 22": 9,
                   "remove teeth 23": 10,
                   "remove teeth 24": 11,
                   "remove teeth 25": 12,
                   "remove teeth 26": 13,
                   "remove teeth 27": 14,
                   "remove teeth 28": 15,
                   "remove teeth 31": 16,
                   "remove teeth 32": 17,
                   "remove teeth 33": 18,
                   "remove teeth 34": 19,
                   "remove teeth 35": 20,
                   "remove teeth 36": 21,
                   "remove teeth 37": 22,
                   "remove teeth 38": 23,
                   "remove teeth 41": 24,
                   "remove teeth 42": 25,
                   "remove teeth 43": 26,
                   "remove teeth 44": 27,
                   "remove teeth 45": 28,
                   "remove teeth 46": 29,
                   "remove teeth 47": 30,
                   "remove teeth 48": 31}

    def __init__(self, num_embeddings=32, embedding_dim=64):
        super(Embedding).__init__()

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def __call__(self, x):
        lookup_tensor = torch.tensor([self.word_to_idx[x]], dtype=torch.long)
        return self.embedding(lookup_tensor)
