"""
Created by Marián Tarageľ (xtarag01)
"""

from torch import nn


def cross_entropy(logits, classes):
    loss = nn.CrossEntropyLoss()
    return loss(logits, classes)
