import torch
import math
import torch.nn as nn
import numpy as np


class ElapsedRototor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        log_freqs = torch.linspace(start=np.log(2.*math.pi/100), end=np.log(2.*math.pi/0.01), steps=dim).float()
        self.log_freqs = nn.Parameter(log_freqs)

    def forward(self, pe_input, offset):
        batch_size = pe_input.shape[0]
        freqs = torch.stack(batch_size*[
            torch.exp(self.log_freqs)
            ])
        if offset is not None:
            pe_input = pe_input[:, None, :, None] + offset
        else:
            pe_input = pe_input[:, None, :, None]
        sinusoid_inp = pe_input * freqs[:, None, None, :]
        emb = torch.stack((sinusoid_inp.cos(), sinusoid_inp.sin()), dim=-1)
        return emb.to(pe_input)
