from CIA.model.causal_model import CausalModel
from performer_pytorch.autoregressive_wrapper import AutoregressiveWrapper
from CIA.positional_embeddings import PositionalEmbedding
from CIA.data_processors import DataProcessor
from CIA.dataloaders.dataloader import DataloaderGenerator
from CIA.utils import flatten


class CausalMixedInferenceModel(CausalModel):
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
        super(CausalMixedInferenceModel, self).__init__(
            data_processor, dataloader_generator, positional_embedding, sos_embedding, d_model, num_channels_decoder,
            num_events_decoder, label_smoothing, transformer, layer_pos_emb)
        # Just add a wrapper for autoregressive generation
        self.transformer = AutoregressiveWrapper(
            self.transformer, ignore_index=0, pad_value=0)

    def __repr__(self) -> str:
        return 'CausalMixedInferencePrefixDecoder'

    # def prepare_sequence(self, target_seq, metadata_dict, h_pe_init):

    # def forward(self, target, metadata_dict, h_pe_init=None):

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
        target_seq, layer_pos_emb, h_pe = self.prepare_sequence(
            target_seq, metadata_dict, h_pe)

        # output = self.transformer.generate(seq_out_start, seq_len, context = encodings, **{**dec_kwargs, **kwargs})
        output = self.transformer(
            target_seq, layer_pos_emb=layer_pos_emb)[:, i, :]

        channel_index_output = i % self.num_channels_target

        weights = self.pre_softmaxes[channel_index_output](output)

        # no need for a loss
        return {
            'loss':    None,
            'state':   state,
            'h_pe':    h_pe,
            'weights': weights,
        }
