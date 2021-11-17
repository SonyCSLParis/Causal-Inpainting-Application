from pathlib import Path

num_events_context = 256
local_window_size = 64
config = {
    "dataset": "piano",  # 'piano', 'piano_test'
    # --- Dataloader ---
    "dataloader_generator_kwargs": dict(
        sequences_size=1024,
        transformations={
            "time_dilation": True,
            "velocity_shift": True,
            "transposition": True,
        },
        offset_beginning=(local_window_size - 1),
        offset_end=-local_window_size,
    ),  # Can be different from the encoder's data loader
    # --- DataProcessor ---
    "data_processor_type": "piano_prefixEnd",
    "data_processor_kwargs": dict(
        embedding_size=64,
        num_events_local_window=local_window_size,
        num_events_context=num_events_context,
        reverse_prefix=True,
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
            mask_positions=False,
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
        # autoregressive_decoding only needed if handler_type == 'event
        autoregressive_decoding="fullcat",  # fullcat | mlp | None
        type="performer",
        d_model=512,
        n_head=8,
        local_attn_heads=4,
        fast_local_attn=False,
        # previous "default" values
        # local_window_size=256,
        # num_decoder_layers=16,
        local_window_size=local_window_size,  # works with batch_size = 8
        num_decoder_layers=10,
        dropout=0.1,
        label_smoothing=False,
        features={
            "type": "elu",  # 'favor', 'elu', None is Transformer
            # 'args': dict(n_features=256),  # 'favor args
            "args": dict(),  # elu args
        },
        execute_type="gated",  # 'reversible' (Reformer paper), 'gated'
        # execute_type='reversible',  # 'reversible' (Reformer paper), 'gated' (Stabilizing T for RL) or 'residual'
        layer_pe=None
        # layer_pe=dict(
        #     type='rototor',  # 'rotary', 'spe', 'rototor', 'rototor_fix'
        #     input='elapsed',  # 'index', 'elapsed'
        #     args=dict(
        #         gated_layerSPE=False,
        #         post_phi_layerPE=True,
        #         theta_q=False,
        #     )
        # )
    ),
    # ======== Training ========
    "lr": 1e-4,
    "batch_size": 4,
    "num_batches": 512,
    "num_epochs": 3000000,
    # ======== model ID ========
    "timestamp": None,
    "savename": Path(__file__).stem,
}
