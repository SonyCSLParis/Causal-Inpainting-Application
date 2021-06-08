import torch.nn as nn

from performer_pytorch.reversible import route_args


class SequentialSequence_(nn.Module):
    def __init__(self, layers, args_route={}):
        super().__init__()
        assert all(len(route) == len(layers) for route in args_route.values(
        )), 'each argument route map must have the same depth as the number of sequential layers'
        self.layers = layers
        self.args_route = args_route

    def forward(self, x, **kwargs):
        args = route_args(self.args_route, kwargs, len(self.layers))
        layers_and_args = list(zip(self.layers, args))

        for layer_ind, ((f, g), (f_args, g_args)) in enumerate(layers_and_args):
            # extract the states for the current layer
            f_args_layer = {k: (dict(Zs=v['Zs'][:, :, :, layer_ind], Ss=v['Ss'][:, :, :, :, layer_ind]) if
                                (k == 'states' and v is not None) else v) for k, v in f_args.items()}
            x = x + f(x, **f_args_layer)
            x = x + g(x, **g_args)
        return x
