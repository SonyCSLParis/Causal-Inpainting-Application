import torch
import torch.nn as nn
from performer_pytorch.reversible import route_args


class Gating(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()
        self.Wr = nn.Linear(d_model, d_model, bias=False)
        self.Ur = nn.Linear(d_model, d_model, bias=False)
        self.Wz = nn.Linear(d_model, d_model, bias=False)
        self.Uz = nn.Linear(d_model, d_model, bias=False)
        self.Wg = nn.Linear(d_model, d_model, bias=False)
        self.Ug = nn.Linear(d_model, d_model, bias=False)
        self.bg = nn.Parameter(torch.randn(
            (d_model, ), requires_grad=True) + 1,
                               requires_grad=True)

    def forward(self, x, y):
        """
        :param x:
        :param y: output from the attention or ff layer
        :return:
        """
        r = torch.sigmoid(self.Wr(y) + self.Ur(x))
        z = torch.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = torch.tanh(self.Wg(y) + self.Ug(r * x))
        return (1 - z) * x + z * h


class GatedSequence_(nn.Module):
    def __init__(self, layers, d_model, args_route={}):
        super().__init__()
        assert all(
            len(route) == len(layers) for route in args_route.values()
        ), 'each argument route map must have the same depth as the number of sequential layers'
        self.layers = layers
        self.args_route = args_route

        self.gatings_attn = nn.ModuleList(
            [Gating(d_model=d_model) for _ in layers])
        self.gatings_ff = nn.ModuleList(
            [Gating(d_model=d_model) for _ in layers])

    def forward(self, x, **kwargs):
        args = route_args(self.args_route, kwargs, len(self.layers))
        layers_and_args_and_gatings = list(
            zip(self.layers, args, self.gatings_attn, self.gatings_ff)
        )
        states = []
        for layer_ind, ((f, g), (f_args, g_args), gating_attn,
                        gating_ff) in enumerate(layers_and_args_and_gatings):
            f_args_layer = {
                k: (dict(Zs=v['Zs'][:, :, :, layer_ind],
                         Ss=v['Ss'][:, :, :, :, layer_ind]) if
                    (k == 'states' and v is not None) else v)
                for k, v in f_args.items()
            }
            f_x, state = f(x, **f_args_layer)

            x = gating_attn(x, f_x)
            x = gating_ff(x, g(x, **g_args))

            if state is not None:
                states.append(state)
        if len(states) > 0:
            Zs = torch.stack([st['Z'] for st in states], dim=-1)
            Ss = torch.stack([st['S'] for st in states], dim=-1)
        else:
            Zs = torch.zeros_like(x)
            Ss = torch.zeros_like(x)

        if kwargs['inferring_states']:
            # TODO Zs_rot and Ss_rot are missing            
            raise NotImplementedError       
            return x, Zs, Ss, None, None
        else:
            return x
