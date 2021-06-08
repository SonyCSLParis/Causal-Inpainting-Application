from pathlib import Path

config = {
    'training_method':             'decoder',
    'dataset':                     'piano',

    # --- Dataloader ---
    'dataloader_generator_kwargs': dict(
        sequences_size=1024,
        transformations={
            'time_dilation':  True,
            'velocity_shift': True,
            'transposition':  True
        },
        pad_before=True,
    ),  # Can be different from the encoder's data loader

    # --- DataProcessor ---
    'data_processor_type':         'piano_prefix',  # can be used to filter out some channels
    'data_processor_kwargs':       dict(
        embedding_size=64,
        num_events_before=256,
        num_events_after=256
    ),  # Can be different from the encoder's data processor

    # --- Positional Embedding ---
    'positional_embedding_dict': dict(
        # sinusoidal_embedding=dict(
        #     positional_embedding_size=128,
        #     num_channels=4,
        #     dropout=0.
        # ),
        # sinusoidal_elapsed_time_embedding=dict(
        #     positional_embedding_size=128,
        #     num_channels=4,
        #     dropout=0.,
        #     mask_positions=False
        # ),
        channel_embedding=dict(
            positional_embedding_size=12,
            num_channels=4
        ),
        sinusoidal_progress_bar_embedding=dict(
            positional_embedding_size=128,
            num_channels=4,
            dropout=0.,
        )
    ),

    # --- Start Of Sequence embeddings
    'sos_embedding_dict': dict(
        learnt_sos_embedding=dict(
            embedding_size=512  # sum must be equal to d_model_decoder
        )
    ),

    # --- Decoder ---
    'decoder_type':                'elapsed_performer',
    'decoder_kwargs':              dict(
        d_model=512,
        n_head=8,
        local_attn_heads=4,
        num_decoder_layers=16,
        dropout=0.1,
        label_smoothing=False,
        nb_features=256,
        execute_type='reversible'  # 'reversible' (Reformer paper), 'gated' (Stabilizing T for RL) or 'residual'
    ),
    # ======== Training ========
    'lr':                          1e-4,
    'batch_size':                  16,
    'num_batches':                 256,
    'num_epochs':                  2000,

    # ======== model ID ========
    'timestamp':                   None,
    'savename':                    Path(__file__).stem,
}
