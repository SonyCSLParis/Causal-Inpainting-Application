import torch
import torch.nn as nn


class Rototor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        freq = 1. / (10000 ** (torch.arange(0, 2 * dim, 2).float() / dim))
        self.register_buffer('freq', freq)

    def forward(self, pe_input, offset):
        batch_size = pe_input.shape[0]
        freq = torch.stack(batch_size*[self.freq])
        if offset is not None:
            pe_input = pe_input[:, None, :, None] + offset
        else:
            pe_input = pe_input[:, None, :, None]
        sinusoid_inp = pe_input * freq[:, None, None, :]
        emb = torch.stack((sinusoid_inp.cos(), sinusoid_inp.sin()), dim=-1)
        return emb.to(pe_input)
