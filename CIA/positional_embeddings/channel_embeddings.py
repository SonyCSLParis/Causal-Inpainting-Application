from CIA.positional_embeddings.positional_embedding import BasePositionalEmbedding
from torch import nn
import torch


class ChannelEmbeddings(BasePositionalEmbedding):

    def __init__(self,
                 positional_embedding_size,
                 num_channels,
                 ):
        super(ChannelEmbeddings, self).__init__()
        self.num_channels = num_channels
        self.positional_embedding_size = positional_embedding_size
        self.pe_0 = nn.Parameter(
            torch.randn(
                1, num_channels, positional_embedding_size
            )
        )

    def forward(self, x, i=0, h=None, metadata_dict={}):
        """

        :param x: (
        batch_size,
        num_tokens
        d_model - positional_embedding_size
        )
        :param i:
        :return:
        """
        batch_size, num_tokens, _ = x.size()

        # create init sequence
        num_events = num_tokens // self.num_channels + 1
        positional_embeddings = self.pe_0.repeat(batch_size, num_events, 1)
        offset = i % self.num_channels

        # slice so that we have the correct offset
        positional_embeddings = positional_embeddings[:, offset: offset + num_tokens]

        x = torch.cat([
            x, positional_embeddings
        ], dim=2)

        return x, h

    def forward_step(self, x, i=0, h=None, metadata_dict={}):
        """

        :param x: (
        batch_size,
        d_model - positional_embedding_size
        )
        :param i:
        :return:
        """
        batch_size, _ = x.size()

        offset = i % self.num_channels
        positional_embeddings = self.pe_0.repeat(batch_size, 1, 1)[:,offset]


        x = torch.cat([
            x, positional_embeddings
        ], dim=1)

        return x, h


