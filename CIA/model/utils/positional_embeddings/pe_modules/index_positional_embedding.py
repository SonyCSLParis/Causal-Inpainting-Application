import torch
import torch.nn as nn


class IndexPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        # const = torch.zeros(1)
        # freq = torch.cat((freq, const))
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, freq)
        # emb = torch.stack((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer('emb', emb)

    def forward(self, pe_input):
        return self.emb[None, :pe_input.shape[1]].to(pe_input)
