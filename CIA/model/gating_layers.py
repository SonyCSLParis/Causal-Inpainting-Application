from performer_pytorch.reversible import route_args
import torch
from torch import nn


class GatedSequence_(nn.Module):
    def __init__(self, layers, d_model, args_route={}):
        super().__init__()
        assert all(len(route) == len(layers) for route in args_route.values()), 'each argument route map must have the same depth as the number of sequential layers'
        self.layers = layers
        self.args_route = args_route
        self.Wr = nn.Linear(d_model, d_model, bias=False)
        self.Ur = nn.Linear(d_model, d_model, bias=False)
        self.Wz = nn.Linear(d_model, d_model, bias=False)
        self.Uz = nn.Linear(d_model, d_model, bias=False)
        self.Wg = nn.Linear(d_model, d_model, bias=False)
        self.Ug = nn.Linear(d_model, d_model, bias=False)
        self.bg = nn.Parameter(torch.randn((d_model,), requires_grad=True) + 1, requires_grad=True)

    def forward(self, x, **kwargs):
        args = route_args(self.args_route, kwargs, len(self.layers))
        layers_and_args = list(zip(self.layers, args))

        for (f, g), (f_args, g_args) in layers_and_args:
            x = self.gating(x, f(x, **f_args))
            x = self.gating(x, g(x, **g_args))
        return x

    def gating(self, x, y):
        """
        :param x:
        :param y: output from the attention or ff layer
        :return:
        """
        r = torch.sigmoid(self.Wr(y) + self.Ur(x))
        z = torch.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = torch.tanh(self.Wg(y) + self.Ug(r * x))
        return (1 - z) * x + z * h
