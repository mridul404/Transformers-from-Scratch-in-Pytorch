import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention
from feedforward import FeedForward


class DecoderBlock(nn.Module):
    def __init__(self, embedding_size, num_head, dropout=0.1):
        super().__init__()

        self.d_model = embedding_size
        self.num_head = num_head

        self.layer_norm = nn.LayerNorm(self.d_model)
        self.attention = MultiHeadAttention(self.d_model, self.num_head)
        self.feed_forward = FeedForward(self.d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, trg_mask):
        residual = x
        x = self.attention(x, x, x, trg_mask)
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        residual = x

        x = self.attention(x, encoder_output, encoder_output, src_mask)
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        residual = x

        x = self.feed_forward(x)
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)

        return x


class Decoder(nn.Module):
    def __init__(self, embedding_size, num_head, num_layers, dropout=0.1):
        super().__init__()
        self.decoder_layers = nn.ModuleList(
            [
                DecoderBlock(embedding_size, num_head, dropout) for _ in range(num_layers)
            ]
        )

    def forward(self, x, encoder_output, src_mask, trg_mask):
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, trg_mask)
        return x


if __name__ == '__main__':
    t = torch.rand(1, 10, 512, device='cuda')
    enc = torch.rand(1, 10, 512, device='cuda')

    s_m = torch.ones(1, 1, 1, 10, device='cuda')
    t_m = torch.ones(1, 1, 1, 10, device='cuda')

    model = Decoder(512, 8, 6).to('cuda')
    out = model(t, enc, s_m, t_m)
    print(out.shape)
