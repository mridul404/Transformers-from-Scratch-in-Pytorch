import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(InputEmbedding, self).__init__()

        self.input_embedding = nn.Embedding(vocab_size, embedding_size)

    def forward(self, x):
        return self.input_embedding(x)


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, embedding_size, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_encoding = nn.Embedding(max_len, embedding_size)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        pos = torch.arange(0, seq_len).expand(batch_size, seq_len)
        return self.dropout(x + self.pos_encoding(pos))


inp = torch.rand(1, 1000).long()
trg = torch.rand(1, 1000).long()