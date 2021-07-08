import torch
import math
import torch.nn as nn
import numpy as np


class ElapsedRototor(nn.Module):
    def __init__(self, dim, fix):
        super().__init__()
        log_periods = torch.linspace(start=np.log(0.01), end=np.log(100), steps=dim).float()
        if fix:
            self.log_periods = log_periods
        else:
            self.log_periods = nn.Parameter(log_periods)

    def forward(self, pe_input, offset):
        batch_size = pe_input.shape[0]
        periods = torch.stack(batch_size*[
            torch.exp(self.log_periods)
            ]).to(pe_input.device)
        if offset is not None:
            pe_input = pe_input[:, None, :, None] + offset
        else:
            pe_input = pe_input[:, None, :, None]
        sinusoid_inp = 2 * math.pi * pe_input / periods[:, None, None, :]
        emb = torch.stack((sinusoid_inp.cos(), sinusoid_inp.sin()), dim=-1)
        return emb.to(pe_input)
