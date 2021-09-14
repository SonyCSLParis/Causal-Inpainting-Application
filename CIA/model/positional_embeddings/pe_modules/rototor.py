import torch
import math
import torch.nn as nn
import numpy as np


class Rototor(nn.Module):
    def __init__(self, dim, n_heads, fix, init_type):
        super().__init__()
        if init_type == 'elapsed':
            log_periods = torch.linspace(start=np.log(
                0.01), end=np.log(100), steps=dim).float()
        elif init_type == 'index':
            log_periods = torch.linspace(start=np.log(
                1), end=np.log(1024), steps=dim).float()
        log_periods_heads = torch.stack(n_heads*[log_periods], dim=0)
        self.log_periods = nn.Parameter(
            log_periods_heads, requires_grad=(not fix))

    def forward(self, pe_input, offset):
        batch_size = pe_input.shape[0]
        periods = torch.stack(batch_size*[
            torch.exp(self.log_periods)
        ])
        if offset is not None:
            pe_input = pe_input[:, None, :, None] + offset
        else:
            pe_input = pe_input[:, None, :, None]
        sinusoid_inp = 2 * math.pi * pe_input / periods[:, :, None, :]
        # import matplotlib.pyplot
        # n, bins, patches = plt.hist((torch.reshape(pe_input, (-1,))).detach().cpu().numpy(),
        # 50, density=True, facecolor='g', alpha=0.75)
        # plt.savefig('pe_input.pdf')
        # plt.clf()
        # n, bins, patches = plt.hist((torch.reshape(periods, (-1,))).detach().cpu().numpy(),
        # 50, density=True, facecolor='g', alpha=0.75)
        # plt.savefig('periods.pdf')
        emb = torch.stack((sinusoid_inp.cos(), sinusoid_inp.sin()), dim=-1)
        # self.plot(periods)
        return emb.to(pe_input)

    # def plot(self, periods):
    #     for lay in range(periods.size(2)):
    #         for head in range(periods.size(0)):
    #             self.writer.add_histogram(f'periods_{lay}_{head}',
    #                                       torch.exp(periods[head, :, lay]),
    #                                       epoch_id)
