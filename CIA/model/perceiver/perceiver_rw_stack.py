from CIA.model.perceiver.perceiver_rw import PerceiverReadWrite


class PerceiverReadWriteStack(PerceiverReadWrite):
    def __init__(
        self,
        dim,
        num_layers,
        num_heads,
        dropout,
        local_window_size,
        num_events,
        downscaling,
    ):
        super(PerceiverReadWriteStack, self).__init__(
            dim,
            num_layers,
            num_heads,
            dropout,
            local_window_size,
            num_events,
            downscaling,
        )

    # only difference is that you don't process latents at each layer of the stack
    def _get_process_l(self):
        return None

    def _get_last_layer_norm(self):
        return None
