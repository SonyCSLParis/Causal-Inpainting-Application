from pathlib import Path

config = {
    'training_method':             'decoder',
    'dataset':                     'nes',

    # --- Dataloader ---
    'dataloader_generator_kwargs': dict(
        sequences_size=512,
    ),  # Can be different from the encoder's data loader

    # --- DataProcessor ---
    'data_processor_type':         'masked_piano',  # can be used to filter out some channels
    'data_processor_kwargs':       dict(
        embedding_size=32,
    ),  # Can be different from the encoder's data processor

    # --- Positional Embedding ---
    'positional_embedding_source_dict': dict(
        # 'channel_embedding': dict()
        sinusoidal_embedding= dict(
            positional_embedding_size=128,
            num_channels=4,
            dropout=0.
        ),
        sinusoidal_elapsed_time_embedding=dict(
            positional_embedding_size=128,
            num_channels=4,
            dropout=0.,
            mask_positions=True
        ),
        channel_embedding=dict(
            positional_embedding_size=12,
            num_channels=4
        )
    ),
    
    'positional_embedding_target_dict': dict(
        sinusoidal_embedding= dict(
            positional_embedding_size=128,
            num_channels=4,
            dropout=0.
        ),
        sinusoidal_elapsed_time_embedding=dict(
            positional_embedding_size=128,
            num_channels=4,
            dropout=0.,
            mask_positions=False,
        ),
        channel_embedding=dict(
            positional_embedding_size=12,
            num_channels=4
        )
    ),
    
        # --- Start Of Sequence embeddings
    'sos_embedding_dict': dict(
        learnt_sos_embedding=dict(
            embedding_size=512 # sum must be equal to d_model_decoder
        )
    ),
        
    # --- Decoder ---
    'encoder_decoder_type':                'linear_transformer',
    'encoder_decoder_kwargs':              dict(
        d_model_encoder=512,
        d_model_decoder=512,
        n_head_encoder=8,
        n_head_decoder=8,
        num_layers_encoder=8,
        num_layers_decoder=8,
        dim_feedforward_encoder=1024,
        dim_feedforward_decoder=1024,
        dropout=0.1,
        label_smoothing=False
    ),
    # ======== Training ========
    'lr':                          1e-4,
    'batch_size':                  3,
    'num_batches':                 256,
    'num_epochs':                  2000,

    # ======== model ID ========
    'timestamp':                   None,
    'savename':                    Path(__file__).stem,
}
