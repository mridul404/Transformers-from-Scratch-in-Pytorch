import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, embedding_size, expansion=4):
        super(FeedForward, self).__init__()

        self.embedding_size = embedding_size
        self.expansion = expansion
        self.linear1 = nn.Linear(self.embedding_size, self.embedding_size*self.expansion)
        self.linear2 = nn.Linear(self.embedding_size*self.expansion, self.embedding_size)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))
