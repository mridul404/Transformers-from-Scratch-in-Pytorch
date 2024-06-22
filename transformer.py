import torch
import torch.nn as nn
import torch.nn.functional as F
from encoding import InputEmbedding, PositionalEncoding
from encoder import Encoder
from decoder import Decoder
from linear_projection import LinearProjection


class Transformer(nn.Module):
    def __init__(self, embedding_size, num_head, num_layer, vocab_size, max_len):
        super().__init__()

        self.input_embedding = InputEmbedding(vocab_size, embedding_size)
        self.pos_emb = PositionalEncoding(max_len, embedding_size)
        self.encoder = Encoder(embedding_size, num_head, num_layer)
        self.decoder = Decoder(embedding_size, num_head, num_layer)
        self.linear_proj = LinearProjection(embedding_size, vocab_size)

    def forward(self, src_token, trg_token, src_mask, trg_mask):
        x = self.input_embedding(src_token)
        x = self.pos_emb(x)
        x = self.encoder(x, src_mask)
        x = self.decoder(x, trg_token, src_mask, trg_mask)
        x = self.linear_proj(x)

        return F.softmax(x, dim=-1)


if __name__ == '__main__':
    inp = torch.rand(1, 1000).long()
    trg = torch.rand(1, 1000).long()

    s_m = torch.ones(1, 1, 1, 10)
    t_m = torch.ones(1, 1, 1, 10)

    model = Transformer(512, 8, 6, 1000, 10)
    out = model(inp, trg, s_m, t_m)
    print(out.shape)
