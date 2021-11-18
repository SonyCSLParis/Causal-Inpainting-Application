from pathlib import Path

local_window_size = 64
num_events_context = 256
config = {
    "training_method": "decoder",
    "dataset": "piano",  # 'piano', 'piano_test'
    # --- Dataloader ---
    "dataloader_generator_kwargs": dict(
        sequences_size=1024,
        transformations={
            "time_dilation": True,
            "velocity_shift": True,
            "transposition": True,
        },
        offset_beginning=-(local_window_size - 1),
        offset_end=-local_window_size,
    ),  # Can be different from the encoder's data loader
    # --- DataProcessor ---
    # can be used to filter out some channels
    "data_processor_type": "piano_prefixEnd",
    "data_processor_kwargs": dict(
        embedding_size=64,
        num_events_local_window=local_window_size,
        num_events_context=num_events_context,
        reverse_prefix=False,  # only for prefixEnd
    ),  # Can be different from the encoder's data processor
    # --- Positional Embedding ---
    "positional_embedding_dict": dict(
        sinusoidal_embedding=dict(
            positional_embedding_size=128,
            num_channels=4,
            dropout=0.0,
            expand_channels=False,
        ),
        sinusoidal_elapsed_time_embedding=dict(
            positional_embedding_size=128,
            num_channels=4,
            dropout=0.0,
            mask_positions=False,
            expand_channels=False,
        ),
        sinusoidal_remaining_time_embedding=dict(
            positional_embedding_size=128,
            num_channels=4,
            dropout=0.0,
            expand_channels=False,
        ),
    ),
    # --- Start Of Sequence embeddings
    "sos_embedding_dict": dict(
        learnt_sos_embedding=dict(
            embedding_size=512  # sum must be equal to d_model_decoder
        )
    ),
    # --- Handler type ---
    "handler_type": "event",  # event | channel
    # --- Decoder ---
    "decoder_kwargs": dict(
        type="perceiverio",
        autoregressive_decoding="fullcat",  # fullcat | mlp | None
        d_model=512,
        n_head=8,
        # local_attn_heads=4,
        # fast_local_attn=False,
        # local_window_size=local_window_size,  # works with batch_size = 8
        num_decoder_layers=14,
        dropout=0.1,
        downscaling=16,
        label_smoothing=False,
        features=None,  # not used for perceiver
        execute_type=None,  # not used for perceiver
        layer_pe=None,  # not used for perceiver
    ),
    # ======== Training ========
    "lr": 1e-4,
    "batch_size": 2,
    "num_batches": 32,
    "num_epochs": 1500000,
    # ======== model ID ========
    "timestamp": None,
    "savename": Path(__file__).stem,
}
