import torch
import torch.nn as nn


class ElapsedPositionalEmbedding(nn.Module):
    def __init__(self, dim, dataloader_generator):
        super().__init__()
        # number of frequency component
        self.dim = dim
        self.dataloader_generator = dataloader_generator

    def forward(self, x_embed, h, metadata_dict):
        # TODO: Make this a learnable parameter???
        batch_size = x_embed.size(0)
        inv_freq = 1. / (10000 ** (torch.arange(0, self.dim, 2).float() / self.dim)).unsqueeze(0).repeat(batch_size, 1)\
            .to(x_embed)
        elapsed_time = self.compute_elapsed_time(x_embed, h, metadata_dict)
        sinusoid_inp = torch.einsum("bi,bj->bij", elapsed_time, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb

    def compute_elapsed_time(self, x_embed, h, metadata_dict):
        if h is None:
            h = torch.zeros((x_embed.size(0),)).to(x_embed.device)
        # Original sequence is in prefix order!
        x = metadata_dict['original_sequence']
        _, _, num_channels = x.size()
        elapsed_time = self.dataloader_generator.get_elapsed_time(x)
        h = elapsed_time[:, -1]
        # if prefix mode
        h = h - elapsed_time[:, metadata_dict['decoding_start'] - 1]
        # add zeros
        elapsed_time = torch.cat(
            [
                torch.zeros_like(elapsed_time)[:, :1],
                elapsed_time[:, :-1]
            ],
            dim=1
        )
        if elapsed_time.size(1) > metadata_dict['decoding_start']:
            # we need to have an offset for the generated inpainted region
            elapsed_time[:, metadata_dict['decoding_start']:] = (
                elapsed_time[:, metadata_dict['decoding_start']:] -
                elapsed_time[:, metadata_dict['decoding_start']].unsqueeze(1)
            )
        # TODO scale?! only 10?!
        elapsed_time = elapsed_time * 100
        h = h * 100
        elapsed_time_channelized = elapsed_time.repeat_interleave(num_channels, dim=1)
        return elapsed_time_channelized
