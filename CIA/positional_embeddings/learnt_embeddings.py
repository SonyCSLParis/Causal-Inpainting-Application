from CIA.positional_embeddings.positional_embedding import BasePositionalEmbedding
from torch import nn
import torch


class LearntEmbeddings(BasePositionalEmbedding):

    def __init__(self,
                 positional_embedding_size,
                 num_channels,
                 **kwargs
                 ):
        super(LearntEmbeddings, self).__init__()
        assert positional_embedding_size % 2 == 0
        self.num_channels = num_channels
        self.num_tokens_max = kwargs['num_tokens_max']

        self.time_embedding = nn.Parameter(
            torch.randn(
                1, self.num_tokens_max // self.num_channels + 1,
                   positional_embedding_size // 2
            )
        )

        self.pe_0 = nn.Parameter(
            torch.randn(
                1, num_channels, positional_embedding_size // 2
            )
        )

    def forward(self, x, i=0, h=None, target=None):
        batch_size, num_tokens, _ = x.size()
        assert i < self.num_channels
        # create init sequence
        num_events = num_tokens // self.num_channels
        channel_embeddings = self.pe_0.repeat(batch_size, num_events + 1, 1)

        offset = i % self.num_channels
        # slice so that we have the correct offset
        channel_embeddings = channel_embeddings[:, offset: offset + num_tokens]

        time_embeddings = self.time_embedding.repeat(batch_size, 1, 1)
        time_embeddings = time_embeddings.repeat_interleave(self.num_channels,
                                                                dim=1)
        time_embeddings = time_embeddings[:, offset: offset + num_tokens]

        x = torch.cat([
            x, channel_embeddings, time_embeddings
        ], dim=2)

        return x, h

    def forward_step(self, x, i=0, h=None):
        """

        :param x: (
        batch_size,
        d_model - positional_embedding_size
        )
        :param i:
        :return:
        """
        # TODO can be done better
        batch_size, _ = x.size()
        x = x.unsqueeze(1)
        batch_size, num_tokens, _ = x.size()
        # create init sequence

        positional_embeddings = self.pe_0.repeat(batch_size, 1, 1)
        offset = i % self.num_channels
        # slice so that we have the correct offset
        positional_embeddings = positional_embeddings[:, offset: offset + 1]

        time_embeddings = self.time_embedding.repeat(batch_size, 1, 1)
        offset = i // self.num_channels
        time_embeddings = time_embeddings[:, offset: offset+1]


        x = torch.cat([
            x, positional_embeddings, time_embeddings
        ], dim=2)

        x = x[:, 0]
        return x, h
