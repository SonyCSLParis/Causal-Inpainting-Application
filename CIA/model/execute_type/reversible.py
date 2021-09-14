import torch
import torch.nn as nn
from torch.autograd.function import Function
from performer_pytorch.reversible import Deterministic, route_args


class ReversibleBlock_(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

    def forward(self, x, f_args={}, g_args={}):
        x1, x2 = torch.chunk(x, 2, dim=2)
        y1, y2 = None, None

        with torch.no_grad():
            f_x2, states = self.f(x2, record_rng=self.training, **f_args)
            y1 = x1 + f_x2
            y2 = x2 + self.g(y1, record_rng=self.training, **g_args)
        return torch.cat([y1, y2], dim=2), states

    def backward_pass(self, y, dy, f_args={}, g_args={}):
        y1, y2 = torch.chunk(y, 2, dim=2)
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim=2)
        del dy

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True, **g_args)
            torch.autograd.backward(gy1, dy2)

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            fx2, _ = self.f(x2, set_rng=True, **f_args)
            torch.autograd.backward(fx2, dx1, retain_graph=True)

        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2

            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            x = torch.cat([x1, x2.detach()], dim=2)
            dx = torch.cat([dx1, dx2], dim=2)

        return x, dx


class _ReversibleFunction_(Function):
    @staticmethod
    def forward(ctx, x, blocks, args):
        ctx.args = args
        for block, kwarg in zip(blocks, args):
            # extract the states for the current layer
            kwargs_layer = dict(f_args=kwarg['f_args'], g_args=kwarg['g_args'])
            x, _ = block(x, **kwargs_layer)
        ctx.y = x.detach()
        ctx.blocks = blocks
        # can't return a list in torch Function
        return x

    @staticmethod
    def backward(ctx, dy):
        y = ctx.y
        args = ctx.args
        for block, kwargs in zip(ctx.blocks[::-1], args[::-1]):
            y, dy = block.backward_pass(y, dy, **kwargs)
        return dy, None, None

    @staticmethod
    def forward_with_states(x, blocks, args):
        # ctx.args = args
        states = []
        for layer_ind, (block, kwarg) in enumerate(zip(blocks, args)):
            # extract the states for the current layer
            f_args_layer = {k: (dict(Zs=v['Zs'][:, :, :, layer_ind], Ss=v['Ss'][:, :, :, :, layer_ind],
                                     Zs_rot=v['Zs_rot'][:, :, :, layer_ind],
                                     Ss_rot=v['Ss_rot'][:, :, :, :, layer_ind]) if
                                (k == 'states' and v is not None) else v) for k, v in kwarg['f_args'].items()}
            kwargs_layer = dict(f_args=f_args_layer, g_args=kwarg['g_args'])
            x, state = block(x, **kwargs_layer)
            if state is not None:
                states.append(state)
        # ctx.y = x.detach()
        # ctx.blocks = blocks
        # can't return a list in torch Function
        Zs = torch.stack([st['Z'] for st in states], dim=-1)
        Ss = torch.stack([st['S'] for st in states], dim=-1)
        Zs_rot = torch.stack([st['Z_rot'] for st in states], dim=-1)
        Ss_rot = torch.stack([st['S_rot'] for st in states], dim=-1)
        return x, Zs, Ss, Zs_rot, Ss_rot


class ReversibleSequence_(nn.Module):
    def __init__(self, blocks, args_route={}):
        super().__init__()
        self.args_route = args_route
        self.blocks = nn.ModuleList(
            [ReversibleBlock_(f=f, g=g) for f, g in blocks])

    def forward(self, x, **kwargs):
        x = torch.cat([x, x], dim=-1)
        blocks = self.blocks
        args = route_args(self.args_route, kwargs, len(blocks))
        args = list(map(lambda x: {'f_args': x[0], 'g_args': x[1]}, args))
        if kwargs['inferring_states']:
            x, Zs, Ss, Zs_rot, Ss_rot = _ReversibleFunction_.forward_with_states(x, blocks, args)
            x = torch.stack(x.chunk(2, dim=-1)).sum(dim=0)
            return x, Zs, Ss, Zs_rot, Ss_rot
        else:
            x = _ReversibleFunction_.apply(x, blocks, args)
            x = torch.stack(x.chunk(2, dim=-1)).sum(dim=0)
            return x
