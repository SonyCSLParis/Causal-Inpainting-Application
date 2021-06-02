
from CIA.positional_embeddings import PositionalEmbedding
from torch import nn
from CIA.data_processors import DataProcessor
from CIA.dataloaders.dataloader import DataloaderGenerator
from CIA.utils import flatten, categorical_crossentropy
import torch


class CausalModel(nn.Module):
    def __init__(self,
                 data_processor: DataProcessor,
                 dataloader_generator: DataloaderGenerator,
                 positional_embedding: PositionalEmbedding,
                 sos_embedding,
                 d_model,
                 num_channels_decoder,
                 num_events_decoder,
                 label_smoothing,
                 transformer,
                 layer_pos_emb):
        super(CausalModel, self).__init__()
        self.data_processor = data_processor
        # can be useful
        self.dataloader_generator = dataloader_generator

        # Compute num_tokens for source and target
        self.num_tokens_per_channel = self.data_processor.num_tokens_per_channel
        self.num_channels_target = len(self.num_tokens_per_channel)
        assert self.num_channels_target == num_channels_decoder
        self.d_model = d_model
        self.num_tokens_target = self.data_processor.num_tokens

        assert self.num_tokens_target == num_channels_decoder * num_events_decoder

        ######################################################
        # Embeddings
        self.positional_embedding = positional_embedding

        self.layer_pos_emb = layer_pos_emb

        self.linear_target = nn.Linear(
            self.data_processor.embedding_size +
            self.positional_embedding.positional_embedding_size,
            self.d_model)

        ########################################################
        # Start of sentence
        self.sos_embedding = sos_embedding

        ######################################################
        # self.transformer = AutoregressiveWrapper(transformer, ignore_index=ignore_index, pad_value=pad_value)
        self.transformer = transformer
        self.label_smoothing = label_smoothing

        ######################################################
        # Output dimension adjustment
        self.pre_softmaxes = nn.ModuleList([nn.Linear(self.d_model, num_tokens_of_channel)
                                            for num_tokens_of_channel in self.num_tokens_per_channel
                                            ]
                                           )

    def __repr__(self) -> str:
        return 'CausalPrefixDecoder'

    def prepare_sequence(self, target_seq, metadata_dict, h_pe_init):
        # add input positional embeddings
        target_seq, h_pe = self.positional_embedding(
            target_seq, i=0, h=h_pe_init, metadata_dict=metadata_dict)
        target_seq = self.linear_target(target_seq)

        # compute layer positional embeddings
        if self.layer_pos_emb is not None:
            layer_pos_emb = self.layer_pos_emb(x_embed=target_seq, i=0, h=h_pe_init, metadata_dict=metadata_dict)

        # shift target_seq by one
        dummy_input_target = self.sos_embedding(metadata_dict).unsqueeze(1)
        target_seq = torch.cat(
            [
                dummy_input_target,
                target_seq
            ],
            dim=1)
        target_seq = target_seq[:, :-1]

        if self.layer_pos_emb is not None:
            # For dummy input on layer positional attention, we can simply repeat the first embedding
            # which corresponds either to position 0 or elapsed time 0
            layer_pos_emb = torch.cat(
                [
                    layer_pos_emb[:, 0, :].unsqueeze(1),
                    layer_pos_emb
                ],
                dim=1)
            layer_pos_emb = layer_pos_emb[:, :-1]
        return target_seq, layer_pos_emb, h_pe

    def forward(self, target, metadata_dict, h_pe_init=None):
        """
        :param target: sequence of tokens (batch_size, num_events, num_channels)
        :return:
        """
        batch_size, _, _ = target.size()
        target_embedded = self.data_processor.embed(target)
        target_seq = flatten(target_embedded)
        target_seq, layer_pos_emb, h_pe = self.prepare_sequence(target_seq, metadata_dict, h_pe_init)

        # forward pass
        output = self.transformer(target_seq, layer_pos_emb=layer_pos_emb)
        output = output.view(batch_size,
                             -1,
                             self.num_channels_target,
                             self.d_model)
        weights_per_category = [
            pre_softmax(t[:, :, 0, :])
            for t, pre_softmax in zip(output.split(1, 2), self.pre_softmaxes)
        ]

        # we can change loss mask
        if 'loss_mask' in metadata_dict:
            loss_mask = (1 - metadata_dict['loss_mask'].long())
        else:
            loss_mask = torch.ones_like(target)

        # If prefix mode, we keep track of the two separate losses
        if 'decoding_start' in metadata_dict:
            decoding_start = metadata_dict['decoding_start']
            weights_prefix = [
                weight[:, :decoding_start]
                for weight in weights_per_category]
            target_prefix = target[:, :decoding_start]
            loss_mask_prefix = loss_mask[:, :decoding_start]
            loss_prefix = categorical_crossentropy(
                value=weights_prefix,
                target=target_prefix,
                mask=loss_mask_prefix,
                label_smoothing=self.label_smoothing
            )

            weights_inpainting = [
                weight[:, decoding_start:]
                for weight in weights_per_category]
            target_inpainting = target[:, decoding_start:]
            loss_mask_inpainting = loss_mask[:, decoding_start:]
            loss_inpainting = categorical_crossentropy(
                value=weights_inpainting,
                target=target_inpainting,
                mask=loss_mask_inpainting,
                label_smoothing=self.label_smoothing
            )

            num_tokens_prefix = loss_mask_prefix.sum()
            num_tokens_inpainting = loss_mask_inpainting.sum()

            loss = (loss_prefix * num_tokens_prefix + loss_inpainting * num_tokens_inpainting) / \
                (num_tokens_prefix + num_tokens_inpainting)

            return {
                'loss':                 loss,
                'h_pe':                 h_pe,
                'weights_per_category': weights_per_category,
                'monitored_quantities': {
                    'loss': loss.item(),
                    'loss_prefix': loss_prefix.item(),
                    'loss_inpainting': loss_inpainting.item()
                }
            }

        else:
            loss = categorical_crossentropy(
                value=weights_per_category,
                target=target,
                mask=loss_mask,
                label_smoothing=self.label_smoothing
            )

            return {
                'loss':                 loss,
                'h_pe':                 h_pe,
                'weights_per_category': weights_per_category,
                'monitored_quantities': {
                    'loss': loss.item()
                }
            }

    def forward_step(self, target, metadata_dict, state, i, h_pe):
        """
        if i == 0, target is not used: SOS instead
        :param target: sequence of tokens (batch_size,)
        :param i:
        :param h_pe:
        :return:
        """
        target_embedded = self.data_processor.embed(target)
        target_seq = flatten(target_embedded)
        target_seq, layer_pos_emb, h_pe = self.prepare_sequence(target_seq, metadata_dict, h_pe)

        # output = self.transformer.generate(seq_out_start, seq_len, context = encodings, **{**dec_kwargs, **kwargs})
        output = self.transformer(target_seq, layer_pos_emb=layer_pos_emb)[:, i, :]

        channel_index_output = i % self.num_channels_target

        weights = self.pre_softmaxes[channel_index_output](output)

        # no need for a loss
        return {
            'loss':    None,
            'state':   state,
            'h_pe':    h_pe,
            'weights': weights,
        }
