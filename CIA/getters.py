from torch import nn
from CIA.models.performer import PerformerStandard
from CIA.models.elapsed_performer import ElapsedPerformer
from CIA.models.causal_model import CausalModel
from CIA.dataloaders import BachDataloaderGenerator, PianoDataloaderGenerator
from CIA.data_processors import BachDataProcessor, MaskedPianoSourceTargetDataProcessor, PianoDataProcessor, \
    PianoPrefixDataProcessor, MaskedBachSourceTargetDataProcessor
from CIA.dataloaders.nes_dataloader import NESDataloader
from CIA.start_of_sequence_embeddings import SOSEmbedding, BaseSOSEmbedding, LearntSOSEmbedding
from CIA.positional_embeddings import ChannelEmbeddings, BasePositionalEmbedding, PositionalEmbedding, \
    SinusoidalElapsedTimeEmbedding, SinusoidalPositionalEmbedding, SinusoidalProgressBarEmbedding


def get_dataloader_generator(dataset, dataloader_generator_kwargs):
    if dataset.lower() == 'bach':
        return BachDataloaderGenerator(
            sequences_size=dataloader_generator_kwargs['sequences_size'])
    elif dataset.lower() == 'piano':
        return PianoDataloaderGenerator(
            sequences_size=dataloader_generator_kwargs['sequences_size'],
            transformations=dataloader_generator_kwargs['transformations'],
            pad_before=dataloader_generator_kwargs['pad_before']
        )
    elif dataset.lower() == 'nes':
        return NESDataloader(
            sequences_size=dataloader_generator_kwargs['sequences_size'])
    else:
        raise NotImplementedError


def get_data_processor(dataloader_generator, data_processor_type,
                       data_processor_kwargs):

    if data_processor_type == 'bach':
        # compute num_events num_tokens_per_channel
        dataset = dataloader_generator.dataset
        num_events = dataset.sequences_size * dataset.subdivision
        num_tokens_per_channel = [len(d) for d in dataset.index2note_dicts]
        data_processor = BachDataProcessor(
            embedding_size=data_processor_kwargs['embedding_size'],
            num_events=num_events,
            num_tokens_per_channel=num_tokens_per_channel)
    elif data_processor_type == 'piano':
        num_events = dataloader_generator.sequences_size
        value2index = dataloader_generator.dataset.value2index
        num_tokens_per_channel = [
            len(value2index[feature])
            for feature in dataloader_generator.features
        ]
        data_processor = PianoDataProcessor(
            dataloader_generator=dataloader_generator,
            embedding_size=data_processor_kwargs['embedding_size'],
            num_events=num_events,
            num_tokens_per_channel=num_tokens_per_channel)
    elif data_processor_type == 'piano_prefix':
        num_events = dataloader_generator.sequences_size
        value2index = dataloader_generator.dataset.value2index
        num_tokens_per_channel = [
            len(value2index[feature])
            for feature in dataloader_generator.features
        ]

        data_processor = PianoPrefixDataProcessor(
            dataloader_generator=dataloader_generator,
            embedding_size=data_processor_kwargs['embedding_size'],
            num_events=num_events,
            num_events_before=data_processor_kwargs['num_events_before'],
            num_events_after=data_processor_kwargs['num_events_after'],
            num_tokens_per_channel=num_tokens_per_channel)
    else:
        raise NotImplementedError

    return data_processor


def get_source_target_data_processor(dataloader_generator, data_processor_type,
                                     data_processor_kwargs):

    if data_processor_type == 'masked_bach':
        num_events = dataloader_generator.sequences_size
        value2index = dataloader_generator.dataset.note2index_dicts
        num_tokens_per_channel = [
            len(value2index[feature])
            for feature in dataloader_generator.features
        ]
        data_processor = MaskedBachSourceTargetDataProcessor(
            num_tokens_per_channel=num_tokens_per_channel,
            num_events=num_events,
            dataloader_generator=dataloader_generator,
            embedding_size=data_processor_kwargs['embedding_size'],
        )

    elif data_processor_type == 'masked_piano':
        num_events = dataloader_generator.dataset.sequence_size
        value2index = dataloader_generator.dataset.value2index
        num_tokens_per_channel = [
            len(value2index[feature])
            for feature in dataloader_generator.features
        ]
        data_processor = MaskedPianoSourceTargetDataProcessor(
            dataloader_generator=dataloader_generator,
            embedding_size=data_processor_kwargs['embedding_size'],
            num_events=num_events,
            num_tokens_per_channel=num_tokens_per_channel)
    else:
        raise NotImplementedError

    return data_processor


def get_positional_embedding(dataloader_generator,
                             positional_embedding_dict) -> PositionalEmbedding:
    base_positional_embedding_list = []
    for pe_name, pe_kwargs in positional_embedding_dict.items():
        if pe_name == 'sinusoidal_embedding':
            num_tokens_max = (dataloader_generator.sequences_size *
                              dataloader_generator.num_channels)
            base_pe: BasePositionalEmbedding = SinusoidalPositionalEmbedding(
                positional_embedding_size=pe_kwargs[
                    'positional_embedding_size'],
                num_tokens_max=num_tokens_max,
                num_channels=pe_kwargs['num_channels'],
                dropout=pe_kwargs['dropout'])
        elif pe_name == 'channel_embedding':
            base_pe = ChannelEmbeddings(**pe_kwargs)
        elif pe_name == 'sinusoidal_elapsed_time_embedding':
            base_pe: BasePositionalEmbedding = SinusoidalElapsedTimeEmbedding(
                dataloader_generator=dataloader_generator, **pe_kwargs)
        elif pe_name == 'sinusoidal_progress_bar_embedding':
            base_pe: BasePositionalEmbedding = SinusoidalProgressBarEmbedding(
                dataloader_generator=dataloader_generator, **pe_kwargs)
        else:
            raise NotImplementedError
        base_positional_embedding_list.append(base_pe)

    return PositionalEmbedding(
        base_positional_embedding_list=base_positional_embedding_list)


# todo write Decoder base class
def get_decoder(data_processor, dataloader_generator, positional_embedding,
                sos_embedding, decoder_type, decoder_kwargs, training_phase):
    num_channels_decoder = data_processor.num_channels
    num_events_decoder = data_processor.num_events
    num_tokens_target = data_processor.num_tokens

    if decoder_type == 'performer':
        transformer = PerformerStandard(
            # size of token dict (in our case we need an extra softmax)
            num_tokens=decoder_kwargs['d_model'],
            max_seq_len=num_tokens_target,    # max sequence length
            dim=decoder_kwargs['d_model'],                  # dimension
            depth=decoder_kwargs['num_decoder_layers'],     # layers
            heads=decoder_kwargs['n_head'],                 # heads
            causal=True,                  # auto-regressive or not
            # number of random features, if not set, will default to (d * log(d)), where d is the dimension of each head
            nb_features=decoder_kwargs['nb_features'],
            # how frequently to redraw the projection matrix, the more frequent, the slower the training
            feature_redraw_interval=1000,
            # defaults to softmax approximation, but can be set to True for generalized attention
            generalized_attention=False,
            # the kernel function to be used, if generalized attention is turned on, defaults to Relu
            kernel_fn=nn.ReLU(),
            reversible=True,              # reversible layers, from Reformer paper
            ff_chunks=10,                 # chunk feedforward layer, from Reformer paper
            use_scalenorm=False,          # use scale norm, from 'Transformers without Tears' paper
            use_rezero=False,             # use rezero, from 'Rezero is all you need' paper
            # multiply final embeddings with token weights for logits, like gpt decoder
            tie_embed=False,
            ff_glu=True,                  # use GLU variant for feedforward
            emb_dropout=decoder_kwargs['dropout'],          # embedding dropout
            ff_dropout=decoder_kwargs['dropout'],           # feedforward dropout
            attn_dropout=decoder_kwargs['dropout'],         # post-attn dropout
            # 4 heads are local attention, 4 others are global performers
            local_attn_heads=4,
            local_window_size=256,        # window size of local attention
            rotary_position_emb=True      # use rotary positional embedding, which endows linear attention with relative positional encoding with no learned parameters. should always be turned on unless if you want to go back to old absolute positional encoding
        )
    elif decoder_type == 'elapsed_transformer':
        transformer = ElapsedPerformer(
            # size of token dict (in our case we need an extra softmax)
            num_tokens=decoder_kwargs['d_model'],
            max_seq_len=num_tokens_target,    # max sequence length
            dim=decoder_kwargs['d_model'],                  # dimension
            depth=decoder_kwargs['num_decoder_layers'],     # layers
            heads=decoder_kwargs['n_head'],                 # heads
            causal=True,                  # auto-regressive or not
            # number of random features, if not set, will default to (d * log(d)), where d is the dimension of each head
            nb_features=decoder_kwargs['nb_features'],
            # how frequently to redraw the projection matrix, the more frequent, the slower the training
            feature_redraw_interval=1000,
            # defaults to softmax approximation, but can be set to True for generalized attention
            generalized_attention=False,
            # the kernel function to be used, if generalized attention is turned on, defaults to Relu
            kernel_fn=nn.ReLU(),
            reversible=True,              # reversible layers, from Reformer paper
            ff_chunks=10,                 # chunk feedforward layer, from Reformer paper
            use_scalenorm=False,          # use scale norm, from 'Transformers without Tears' paper
            use_rezero=False,             # use rezero, from 'Rezero is all you need' paper
            # multiply final embeddings with token weights for logits, like gpt decoder
            tie_embed=False,
            ff_glu=True,                  # use GLU variant for feedforward
            emb_dropout=decoder_kwargs['dropout'],          # embedding dropout
            ff_dropout=decoder_kwargs['dropout'],           # feedforward dropout
            attn_dropout=decoder_kwargs['dropout'],         # post-attn dropout
            # 4 heads are local attention, 4 others are global performers
            local_attn_heads=4,
            local_window_size=256,        # window size of local attention
            rotary_position_emb=True      # use rotary positional embedding, which endows linear attention with relative positional encoding with no learned parameters. should always be turned on unless if you want to go back to old absolute positional encoding
        )
    else:
        raise NotImplementedError

    decoder = CausalModel(
        data_processor=data_processor,
        dataloader_generator=dataloader_generator,
        positional_embedding=positional_embedding,
        sos_embedding=sos_embedding,
        d_model=decoder_kwargs['d_model'],
        num_channels_decoder=num_channels_decoder,
        num_events_decoder=num_events_decoder,
        label_smoothing=decoder_kwargs['label_smoothing'],
        transformer=transformer)

    return decoder


def get_encoder_decoder(data_processor, dataloader_generator,
                        positional_embedding_source,
                        positional_embedding_target, sos_embedding,
                        encoder_decoder_type, encoder_decoder_kwargs,
                        training_phase):

    if encoder_decoder_type == 'linear_transformer':
        decoder = EncoderDecoder(
            data_processor=data_processor,
            dataloader_generator=dataloader_generator,
            positional_embedding_target=positional_embedding_target,
            positional_embedding_source=positional_embedding_source,
            sos_embedding=sos_embedding,
            d_model_encoder=encoder_decoder_kwargs['d_model_encoder'],
            d_model_decoder=encoder_decoder_kwargs['d_model_decoder'],
            num_layers_decoder=encoder_decoder_kwargs['num_layers_decoder'],
            num_layers_encoder=encoder_decoder_kwargs['num_layers_encoder'],
            n_head_encoder=encoder_decoder_kwargs['n_head_encoder'],
            n_head_decoder=encoder_decoder_kwargs['n_head_decoder'],
            dim_feedforward_encoder=encoder_decoder_kwargs[
                'dim_feedforward_encoder'],
            dim_feedforward_decoder=encoder_decoder_kwargs[
                'dim_feedforward_decoder'],
            dropout=encoder_decoder_kwargs['dropout'],
            num_channels_target=data_processor.num_channels_target,
            num_channels_source=data_processor.num_channels_source,
            num_events_target=data_processor.num_events_target,
            num_events_source=data_processor.num_events_source,
            label_smoothing=encoder_decoder_kwargs['label_smoothing'],
            recurrent=not training_phase)
    else:
        raise NotImplementedError

    return decoder


def get_sos_embedding(dataloader_generator,
                      sos_embedding_dict) -> SOSEmbedding:
    base_sos_embedding_list = []
    for sos_name, sos_kwargs in sos_embedding_dict.items():
        if sos_name == 'learnt_sos_embedding':
            base_sos: BaseSOSEmbedding = LearntSOSEmbedding(
                embedding_size=sos_kwargs['embedding_size'])
        else:
            raise NotImplementedError
        base_sos_embedding_list.append(base_sos)

    return SOSEmbedding(base_sos_embedding_list=base_sos_embedding_list)
