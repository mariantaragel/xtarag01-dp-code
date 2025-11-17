from torch import nn


def cross_entropy(logits, classes):
    loss = nn.CrossEntropyLoss()
    return loss(logits, classes)
