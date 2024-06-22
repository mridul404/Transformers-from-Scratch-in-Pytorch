import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_head = num_head
        assert self.d_model % self.num_head == 0, "Embedding size is not divisible by number of head"
        self.d_k = self.d_model // num_head
        self.dropout = nn.Dropout(0.1)

        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)

        self.w_o = nn.Linear(self.d_model, self.d_model)

    def forward(self, q, k, v, mask=None):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = rearrange(query, 'b s (h d_k)-> b h s d_k', h=self.num_head)
        key = rearrange(key, 'b s (h d_k)-> b h s d_k', h=self.num_head)
        value = rearrange(value, 'b s (h d_k)-> b h s d_k', h=self.num_head)

        attention_score = torch.matmul(query, key.transpose(-1, -2)) // math.sqrt(self.d_k)
        if mask is not None:
            attention_score = attention_score.masked_fill_(mask == 0, -torch.inf)
        attention_score = F.softmax(attention_score, dim=-1)
        attention_score = self.dropout(attention_score)
        attention = attention_score @ value

        attention = rearrange(attention, 'b h s d_k -> b s (h d_k)')

        attention = self.w_o(attention)

        return attention


# t = torch.rand(3, 512).unsqueeze(0)
#
# model = MultiHeadAttention(512, 8)
#
# out = model(t, t, t)
#
# print(out.shape)
#

