from CIA.positional_embeddings.sinusoidal_remaining_time_embedding import (
    SinusoidalRemainingTimeEmbedding,
)
from CIA.data_processors.piano_prefixEnd_data_processor import (
    PianoPrefixEndDataProcessor,
)
from CIA.model.transformer.catformer import Catformer
from CIA.model.causal_events_model import CausalEventsModel
from CIA.model.causal_events_model_full_cat import CausalEventsModelFullCat
from torch import nn
from CIA.model.transformer.performer import Performer_
from CIA.model.causal_model import CausalModel
from CIA.dataloaders import PianoDataloaderGenerator
from CIA.data_processors import (
    MaskedPianoSourceTargetDataProcessor,
    PianoPrefixDataProcessor,
    MaskedBachSourceTargetDataProcessor,
)
from CIA.start_of_sequence_embeddings import (
    SOSEmbedding,
    BaseSOSEmbedding,
    LearntSOSEmbedding,
)
from CIA.positional_embeddings import (
    ChannelEmbeddings,
    BasePositionalEmbedding,
    PositionalEmbedding,
    SinusoidalElapsedTimeEmbedding,
    SinusoidalPositionalEmbedding,
    SinusoidalProgressBarEmbedding,
)
from .handlers import DecoderEventsHandler, DecoderPrefixHandler


def get_dataloader_generator(dataset, dataloader_generator_kwargs):
    if dataset.lower() == "piano":
        return PianoDataloaderGenerator(
            sequences_size=dataloader_generator_kwargs["sequences_size"],
            transformations=dataloader_generator_kwargs["transformations"],
            offset_beginning=dataloader_generator_kwargs["offset_beginning"],
            offset_end=dataloader_generator_kwargs["offset_end"],
            num_elements=None,
        )
    else:
        raise NotImplementedError


def get_data_processor(
    dataloader_generator, data_processor_type, data_processor_kwargs
):
    num_events = dataloader_generator.sequences_size
    value2index = dataloader_generator.dataset.value2index
    num_tokens_per_channel = [
        len(value2index[feature]) for feature in dataloader_generator.features
    ]
    if data_processor_type == "piano_prefix":
        data_processor = PianoPrefixDataProcessor(
            dataloader_generator=dataloader_generator,
            embedding_size=data_processor_kwargs["embedding_size"],
            num_events=num_events,
            num_events_context=data_processor_kwargs["num_events_context"],
            num_tokens_per_channel=num_tokens_per_channel,
        )
    elif data_processor_type == "piano_prefixEnd":
        data_processor = PianoPrefixEndDataProcessor(
            dataloader_generator=dataloader_generator,
            embedding_size=data_processor_kwargs["embedding_size"],
            num_events=num_events,
            num_events_local_window=data_processor_kwargs["num_events_local_window"],
            num_events_context=data_processor_kwargs["num_events_context"],
            num_tokens_per_channel=num_tokens_per_channel,
            reverse_prefix=data_processor_kwargs["reverse_prefix"],
        )
    else:
        raise NotImplementedError

    return data_processor


def get_source_target_data_processor(
    dataloader_generator, data_processor_type, data_processor_kwargs
):

    if data_processor_type == "masked_bach":
        num_events = dataloader_generator.sequences_size
        value2index = dataloader_generator.dataset.note2index_dicts
        num_tokens_per_channel = [
            len(value2index[feature]) for feature in dataloader_generator.features
        ]
        data_processor = MaskedBachSourceTargetDataProcessor(
            num_tokens_per_channel=num_tokens_per_channel,
            num_events=num_events,
            dataloader_generator=dataloader_generator,
            embedding_size=data_processor_kwargs["embedding_size"],
        )

    elif data_processor_type == "masked_piano":
        num_events = dataloader_generator.dataset.sequence_size
        value2index = dataloader_generator.dataset.value2index
        num_tokens_per_channel = [
            len(value2index[feature]) for feature in dataloader_generator.features
        ]
        data_processor = MaskedPianoSourceTargetDataProcessor(
            dataloader_generator=dataloader_generator,
            embedding_size=data_processor_kwargs["embedding_size"],
            num_events=num_events,
            num_tokens_per_channel=num_tokens_per_channel,
        )
    else:
        raise NotImplementedError

    return data_processor


def get_positional_embedding(
    dataloader_generator, data_processor, positional_embedding_dict
) -> PositionalEmbedding:
    base_positional_embedding_list = []
    for pe_name, pe_kwargs in positional_embedding_dict.items():
        if pe_name == "sinusoidal_embedding":
            # compute num_tokens_max:
            num_tokens_max = dataloader_generator.sequences_size + 1

            base_pe: BasePositionalEmbedding = SinusoidalPositionalEmbedding(
                positional_embedding_size=pe_kwargs["positional_embedding_size"],
                num_tokens_max=num_tokens_max,
                num_channels=pe_kwargs["num_channels"],
                dropout=pe_kwargs["dropout"],
                expand_channels=pe_kwargs["expand_channels"],
            )
        elif pe_name == "channel_embedding":
            base_pe = ChannelEmbeddings(**pe_kwargs)
        elif pe_name == "sinusoidal_elapsed_time_embedding":
            base_pe: BasePositionalEmbedding = SinusoidalElapsedTimeEmbedding(
                dataloader_generator=dataloader_generator,
                data_processor=data_processor,
                **pe_kwargs
            )
        elif pe_name == "sinusoidal_progress_bar_embedding":
            base_pe: BasePositionalEmbedding = SinusoidalProgressBarEmbedding(
                dataloader_generator=dataloader_generator,
                data_processor=data_processor,
                **pe_kwargs
            )
        elif pe_name == "sinusoidal_remaining_time_embedding":
            base_pe: BasePositionalEmbedding = SinusoidalRemainingTimeEmbedding(
                dataloader_generator=dataloader_generator,
                data_processor=data_processor,
                **pe_kwargs
            )
        else:
            raise NotImplementedError
        base_positional_embedding_list.append(base_pe)

    return PositionalEmbedding(
        base_positional_embedding_list=base_positional_embedding_list
    )


# todo write Decoder base class
def get_decoder(
    data_processor,
    dataloader_generator,
    positional_embedding,
    sos_embedding,
    decoder_kwargs,
    training_phase,
    handler_type,
):
    num_channels_decoder = data_processor.num_channels
    num_events_decoder = data_processor.num_events
    max_seq_len = data_processor.num_tokens
    features = decoder_kwargs["features"]
    layer_pe = decoder_kwargs["layer_pe"]
    if layer_pe is not None:
        layer_pe_args = layer_pe["args"]
        post_phi_layerPE = layer_pe_args["post_phi_layerPE"]
        if post_phi_layerPE and (features["type"] == "favor"):
            dim_layerPE = features["args"]["n_features"]
        else:
            dim_layerPE = decoder_kwargs["d_model"] // decoder_kwargs["n_head"]
        layer_pe["args"]["input_dim"] = dim_layerPE
        pe_input_type = layer_pe["input"]
    else:
        pe_input_type = None

    if decoder_kwargs["type"] == "performer":
        # TODO max_sequence_length is WRONG when channels are not expanded
        transformer = Performer_(
            max_seq_len=max_seq_len,  # max sequence length
            dim=decoder_kwargs["d_model"],  # dimension
            depth=decoder_kwargs["num_decoder_layers"],  # layers
            heads=decoder_kwargs["n_head"],  # heads
            causal=True,  # auto-regressive or not
            features=features,
            # how frequently to redraw the projection matrix, the more frequent, the slower the training
            feature_redraw_interval=100000,
            # defaults to softmax approximation, but can be set to True for generalized attention
            generalized_attention=False,
            # the kernel function to be used, if generalized attention is turned on, defaults to Relu
            kernel_fn=nn.ReLU(),
            # 'reversible' (Reformer paper), 'gated' (Stabilizing T for RL) or 'residual'
            execute_type=decoder_kwargs["execute_type"],
            ff_chunks=10,  # chunk feedforward layer, from Reformer paper
            use_scalenorm=False,  # use scale norm, from 'Transformers without Tears' paper
            use_rezero=False,  # use rezero, from 'Rezero is all you need' paper
            ff_glu=True,  # use GLU variant for feedforward
            emb_dropout=decoder_kwargs["dropout"],  # embedding dropout
            # feedforward dropout
            ff_dropout=decoder_kwargs["dropout"],
            attn_dropout=decoder_kwargs["dropout"],  # post-attn dropout
            # No local attention. With: decoder_kwargs['n_head']//2 ??
            local_attn_heads=decoder_kwargs["local_attn_heads"],
            local_window_size=decoder_kwargs[
                "local_window_size"
            ],  # window size of local attention,
            fast_local_attn=decoder_kwargs["fast_local_attn"],
            layer_pe=layer_pe,
            dataloader_generator=dataloader_generator,
        )
    elif decoder_kwargs["type"] == "catformer":
        transformer = Catformer(
            dim_first_layer=decoder_kwargs["d_model"],  # dimension
            expansion_factor_attn=2,  # http://proceedings.mlr.press/v139/davis21a/davis21a-supp.pdf
            expansion_factor_ff=4,
            depth=decoder_kwargs["num_decoder_layers"],  # layers
            heads=decoder_kwargs["n_head"],  # heads
            features=features,
            ff_chunks=10,  # chunk feedforward layer, from Reformer paper
            ff_glu=True,  # use GLU variant for feedforward
            emb_dropout=decoder_kwargs["dropout"],  # embedding dropout
            ff_dropout=decoder_kwargs["dropout"],  # feedforward dropout
            attn_dropout=decoder_kwargs["dropout"],  # post-attn dropout
            local_attn_heads=decoder_kwargs["local_attn_heads"],
            local_window_size=decoder_kwargs[
                "local_window_size"
            ],  # window size of local attention,
            fast_local_attn=decoder_kwargs["fast_local_attn"],
            layer_pe=layer_pe,
            dataloader_generator=dataloader_generator,
        )
    else:
        raise NotImplementedError

    if handler_type == "channel":
        decoder = CausalModel(
            data_processor=data_processor,
            dataloader_generator=dataloader_generator,
            positional_embedding=positional_embedding,
            sos_embedding=sos_embedding,
            d_model=decoder_kwargs["d_model"],
            num_channels_decoder=num_channels_decoder,
            num_events_decoder=num_events_decoder,
            label_smoothing=decoder_kwargs["label_smoothing"],
            transformer=transformer,
            pe_input_type=pe_input_type,
        )
    elif handler_type == "event":
        autoregressive_decoding_type = decoder_kwargs["autoregressive_decoding"]
        if autoregressive_decoding_type == "fullcat":
            decoder = CausalEventsModelFullCat(
                data_processor=data_processor,
                dataloader_generator=dataloader_generator,
                positional_embedding=positional_embedding,
                sos_embedding=sos_embedding,
                d_model=decoder_kwargs["d_model"],
                num_channels_decoder=num_channels_decoder,
                num_events_decoder=num_events_decoder,
                label_smoothing=decoder_kwargs["label_smoothing"],
                transformer=transformer,
                pe_input_type=pe_input_type,
            )
        elif autoregressive_decoding_type == "mlp":
            decoder = CausalEventsModel(
                data_processor=data_processor,
                dataloader_generator=dataloader_generator,
                positional_embedding=positional_embedding,
                sos_embedding=sos_embedding,
                d_model=decoder_kwargs["d_model"],
                num_channels_decoder=num_channels_decoder,
                num_events_decoder=num_events_decoder,
                label_smoothing=decoder_kwargs["label_smoothing"],
                transformer=transformer,
                pe_input_type=pe_input_type,
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return decoder


def get_sos_embedding(dataloader_generator, sos_embedding_dict) -> SOSEmbedding:
    base_sos_embedding_list = []
    for sos_name, sos_kwargs in sos_embedding_dict.items():
        if sos_name == "learnt_sos_embedding":
            base_sos: BaseSOSEmbedding = LearntSOSEmbedding(
                embedding_size=sos_kwargs["embedding_size"]
            )
        else:
            raise NotImplementedError
        base_sos_embedding_list.append(base_sos)

    return SOSEmbedding(base_sos_embedding_list=base_sos_embedding_list)


def get_handler(handler_type, decoder, model_dir, dataloader_generator):
    if handler_type == "event":
        return DecoderEventsHandler(
            model=decoder,
            model_dir=model_dir,
            dataloader_generator=dataloader_generator,
        )
    elif handler_type == "channel":
        return DecoderPrefixHandler(
            model=decoder,
            model_dir=model_dir,
            dataloader_generator=dataloader_generator,
        )
    else:
        raise NotImplementedError
