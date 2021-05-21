from pathlib import Path

config = {
    'dataset':                     'bach',

    # --- Dataloader ---
    'dataloader_generator_kwargs': dict(
        sequences_size=64
    ),  # Can be different from the encoder's data loader

    # --- DataProcessor ---
    'data_processor_type':         'bach',  # can be used to filter out some channels
    'data_processor_kwargs':       dict(
        embedding_size=32
    ),  # Can be different from the encoder's data processor

    # --- Decoder ---
    # 'fast_transformer'
    'decoder_type':                'linear_transformer',
    'decoder_kwargs':              dict(
        d_model=512,
        n_head=8,
        num_decoder_layers=8,
        dim_feedforward=2048,
        positional_embedding_size=32,
        dropout=0.1
    ),
    # ======== Training ========
    'lr':                          1e-4,
    'batch_size':                  4,
    'num_batches':                 32,
    'num_epochs':                  2000,

    # ======== model ID ========
    'timestamp':                   None,
    'savename':                    Path(__file__).stem,
}
