import torch
import torch.nn as nn


class ElapsedPositionalEmbedding(nn.Module):
    def __init__(self, dim, dataloader_generator):
        super().__init__()
        # number of frequency component
        self.dim = dim
        self.dataloader_generator = dataloader_generator

    def forward(self, pe_input):
        # TODO: Make this a learnable parameter???
        batch_size = x_embed.size(0)
        inv_freq = 1. / (10000 ** (torch.arange(0, self.dim, 2).float() / self.dim)).unsqueeze(0).repeat(batch_size, 1)\
            .to(x_embed)
        elapsed_time = self.compute_elapsed_time(x_embed, h, metadata_dict)
        sinusoid_inp = torch.einsum("bi,bj->bij", elapsed_time, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb
