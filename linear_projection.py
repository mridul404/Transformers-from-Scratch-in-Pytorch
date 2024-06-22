import torch
import torch.nn as nn


class LinearProjection(nn.Module):
    def __init__(self, embedding_size, vocab_size):
        super().__init__()
        self.linear = nn.Linear(embedding_size, vocab_size)

    def forward(self, x):
        return self.linear(x)
