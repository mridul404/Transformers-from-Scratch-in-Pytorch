import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention
from feedforward import FeedForward


class EncoderBlock(nn.Module):
    def __init__(self, embedding_size, num_head):
        super(EncoderBlock, self).__init__()
        self.d_model = embedding_size
        self.num_head = num_head
        self.layer_norm1 = nn.LayerNorm(self.d_model)
        self.layer_norm2 = nn.LayerNorm(self.d_model)
        self.self_attention = MultiHeadAttention(self.d_model, self.num_head)
        self.feed_forward = FeedForward(self.d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):
        residual1 = x
        x = self.self_attention(x, x, x, mask)
        x += residual1
        x = self.layer_norm1(x)
        residual2 = x
        x = self.feed_forward(x)
        x += residual2
        x = self.layer_norm2(self.dropout(x))

        return x


class Encoder(nn.Module):
    def __init__(self, embedding_size, num_head, num_layer):
        super(Encoder, self).__init__()
        self.d_model = embedding_size
        self.num_head = num_head
        self.encoder_list = nn.ModuleList([EncoderBlock(self.d_model, self.num_head) for _ in range(num_layer)])

    def forward(self, x, mask):
        for encoder in self.encoder_list:
            x = encoder(x, mask)
        return x

