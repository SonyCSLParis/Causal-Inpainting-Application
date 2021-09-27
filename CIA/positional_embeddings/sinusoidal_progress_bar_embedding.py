from CIA.positional_embeddings.positional_embedding import BasePositionalEmbedding
from torch import nn
from CIA.utils import flatten
import torch
import math


class SinusoidalProgressBarEmbedding(BasePositionalEmbedding):
    def __init__(self, positional_embedding_size, num_channels,
                 dataloader_generator, data_processor, dropout, expand_channels, **kwargs):
        super(SinusoidalProgressBarEmbedding,
              self).__init__(expand_channels=expand_channels)
        assert positional_embedding_size % 2 == 0
        if type(data_processor).__name__ == 'PianoPrefixEndDataProcessor':
            raise NotImplementedError
        self.data_processor = data_processor
        self.dataloader_generator = dataloader_generator
        self.positional_embedding_size = positional_embedding_size

        self.dropout = torch.nn.Dropout(p=dropout)
        self.num_channels = num_channels

    def forward(self, x_embed, i, h, metadata_dict):
        """[summary]

        Args:
            x_embed ([type]): (batch_size, num_tokens, embedding_dim)
            i (int, optional): [description]. Defaults to 0.
            h ([type], optional): [description]. Defaults to None.
            metadata_dict (dict, optional): [description]. Defaults to {}.

        Returns:
            [type]: [description]
        """
        assert i == 0
        if h is None:
            h = torch.zeros_like(x_embed[:, 0, 0])
        assert 'original_sequence' in metadata_dict, (
            'Dictionnary metadata_dict must contain entry "original_sequence" in order to compute the elapsed time'
        )
        assert 'placeholder_duration' in metadata_dict
        assert 'decoding_start' in metadata_dict
        placeholder_duration = metadata_dict['placeholder_duration']

        x = metadata_dict['original_sequence']
        batch_size, num_events, num_channels = x.size()
        batch_size, num_tokens, _ = x_embed.size()

        # check that expand_channels is correctly set:
        if self.expand_channels:
            assert num_tokens == num_events * num_channels
        else:
            assert num_tokens == num_events

        elapsed_time = self.dataloader_generator.get_elapsed_time(x)

        zeros_location = (placeholder_duration < 0.01)

        h = elapsed_time[:, -1]

        if elapsed_time.size(1) >= metadata_dict['decoding_start']:
            h = h - elapsed_time[:, metadata_dict['decoding_start'] - 1]
        else:
            h = torch.zeros_like(h)

        h = h / placeholder_duration * 100

        h[zeros_location] = 100

        # add zeros
        elapsed_time = torch.cat(
            [torch.zeros_like(elapsed_time)[:, :1], elapsed_time[:, :-1]],
            dim=1)
        if metadata_dict['decoding_start'] < elapsed_time.size(1):
            elapsed_time[:, metadata_dict['decoding_start']:] = (
                elapsed_time[:, metadata_dict['decoding_start']:] -
                elapsed_time[:, metadata_dict['decoding_start']].unsqueeze(1)
            ) / placeholder_duration.unsqueeze(1) * 100

        # TODO no progress bar for prefixes?!
        elapsed_time[:, :metadata_dict['decoding_start']] = 0
        elapsed_time[zeros_location, metadata_dict['decoding_start']:] = 100

        # add embedding_dim to elapsed time
        elapsed_time = elapsed_time.unsqueeze(2)

        pe = torch.zeros(batch_size, num_events,
                         self.positional_embedding_size)
        pe = pe.to(device=x.device)

        div_term = torch.exp(
            torch.arange(0, self.positional_embedding_size, 2).float() *
            (-math.log(10000.0) / self.positional_embedding_size))
        div_term = div_term.to(device=x.device)
        div_term = div_term.unsqueeze(0).unsqueeze(0)
        pe[:, :, 0::2] = torch.sin(elapsed_time * div_term)
        pe[:, :, 1::2] = torch.cos(elapsed_time * div_term)

        if self.expand_channels:
            pos_embedding = pe.repeat_interleave(self.num_channels, dim=1)
        else:
            pos_embedding = pe

        pos_embedding = self.dropout(pos_embedding)
        x_embed = torch.cat([x_embed, pos_embedding], dim=2)
        return x_embed, None

    def forward_step(self, x, i=0, h=None, metadata_dict={}):
        if not self.expand_channels:
            raise NotImplementedError

        assert 'decoding_start' in metadata_dict
        # time_shift must be the last feature
        assert self.dataloader_generator.features.index('time_shift') == len(
            self.dataloader_generator.features) - 1

        assert 'original_token' in metadata_dict, (
            'Dictionnary metadata_dict must contain entry "original_token" in order to compute the elapsed time'
        )

        batch_size = x.size(0)
        # h represents the progress in %
        if h is None:
            h = torch.zeros((batch_size, )).to(x.device)

        elapsed_time = h.unsqueeze(1)

        pe = torch.zeros(batch_size, self.positional_embedding_size)
        pe = pe.to(device=x.device)

        div_term = torch.exp(
            torch.arange(0, self.positional_embedding_size, 2).float() *
            (-math.log(10000.0) / self.positional_embedding_size))
        div_term = div_term.to(device=x.device)
        div_term = div_term.unsqueeze(0)
        pe[:, 0::2] = torch.sin(elapsed_time * div_term)
        pe[:, 1::2] = torch.cos(elapsed_time * div_term)

        # dropout only on pe
        pe = self.dropout(pe)
        x_embed = torch.cat([x, pe], dim=1)

        # update h if the current token is a time_shift:
        if i % self.num_channels == self.num_channels - 1:
            # add fake features so that we can call get_elapsed_time
            target = metadata_dict['original_token']
            target = target.unsqueeze(1).unsqueeze(1)
            target = target.repeat(1, 1, self.num_channels)
            elapsed_time = self.dataloader_generator.get_elapsed_time(target)
            elapsed_time = elapsed_time.squeeze(1)
            # TODO check
            # scale by correct amount
            placeholder_duration = metadata_dict['placeholder_duration']
            zeros_location = (placeholder_duration < 0.01)

            # if i // self.num_channels < metadata_dict['decoding_start']:
            #     elapsed_time = torch.zeros_like(elapsed_time)

            if i // self.num_channels < metadata_dict['decoding_start']:
                h = torch.zeros_like(elapsed_time)
            else:
                elapsed_time = elapsed_time / placeholder_duration * 100
                h = h + elapsed_time
                h[zeros_location] = 100

        return x_embed, h