"""
Created by Marián Tarageľ (xtarag01)
"""

from torch import nn

class EmbeddingLangEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.encoder = nn.Embedding(vocab_size, embed_dim)

    def __call__(self, tokens):
        x = self.encoder(tokens)
        return x.mean(dim=1)