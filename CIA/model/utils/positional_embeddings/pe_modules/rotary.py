import torch
import torch.nn as nn


class Rotary(nn.Module):
    def __init__(self, dim, n_heads, fix, init_type):
        super().__init__()
        # TODO: changer l'init ??
        if init_type == 'elapsed':
            freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        elif init_type == 'index':
            freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        freq_heads = torch.stack(n_heads*[freq], dim=0)
        self.freqs = nn.Parameter(freq_heads, requires_grad=(not fix))

    def forward(self, pe_input):
        sinusoid_inp = pe_input[:, None, :, None] * self.freqs[None, :, None, :]
        emb = torch.stack((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb.to(pe_input)
