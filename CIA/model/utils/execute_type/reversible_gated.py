import torch
import torch.nn as nn
from torch.autograd.function import Function
from performer_pytorch.reversible import Deterministic, route_args


class ReversibleGatedBlock_(nn.Module):
    def __init__(self, f, g, d_model):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)
        # gating for attention
        self.Wfr = nn.Linear(d_model, d_model, bias=False)
        self.Ufr = nn.Linear(d_model, d_model, bias=False)
        self.Wfz = nn.Linear(d_model, d_model, bias=False)
        self.Ufz = nn.Linear(d_model, d_model, bias=False)
        self.Wfg = nn.Linear(d_model, d_model, bias=False)
        self.Ufg = nn.Linear(d_model, d_model, bias=False)
        self.bfg = nn.Parameter(torch.randn(
            (d_model,), requires_grad=True) + 1, requires_grad=True)

        # gating for ff
        self.Wgr = nn.Linear(d_model, d_model, bias=False)
        self.Ugr = nn.Linear(d_model, d_model, bias=False)
        self.Wgz = nn.Linear(d_model, d_model, bias=False)
        self.Ugz = nn.Linear(d_model, d_model, bias=False)
        self.Wgg = nn.Linear(d_model, d_model, bias=False)
        self.Ugg = nn.Linear(d_model, d_model, bias=False)
        self.bgg = nn.Parameter(torch.randn(
            (d_model,), requires_grad=True) + 1, requires_grad=True)

    def forward(self, x, f_args={}, g_args={}):
        x1, x2 = torch.chunk(x, 2, dim=2)
        y1, y2 = None, None

        with torch.no_grad():
            # attention layer
            y1 = x1
            fx1, states = self.f(x1, record_rng=self.training, **f_args)
            r = torch.sigmoid(self.Wfr(fx1) + self.Ufr(x1))
            z = torch.sigmoid(self.Wfz(fx1) + self.Ufz(x1) - self.bgg)
            h = torch.tanh(self.Wfg(fx1) + self.Ufg(r * x1))
            y2 = (1 - z) * x2 + z * h

            # ff layer
            out1 = y1
            gy1 = self.g(y1, record_rng=self.training, **g_args)
            r = torch.sigmoid(self.Wgr(gy1) + self.Ugr(y1))
            z = torch.sigmoid(self.Wgz(gy1) + self.Ugz(y1) - self.bgg)
            h = torch.tanh(self.Wgg(gy1) + self.Ugg(r * y1))
            out2 = (1 - z) * y2 + z * h
            out = torch.cat([out1, out2], dim=2)
        return out, states

    def backward_pass(self, out, dout, f_args={}, g_args={}):
        out1, out2 = torch.chunk(out, 2, dim=2)
        del out

        dout1, dout2 = torch.chunk(dout, 2, dim=2)
        del dout

        # FF layer
        with torch.no_grad():
            y1 = out1
            gy1 = self.g(y1, record_rng=self.training, **g_args)
            r = torch.sigmoid(self.Wgr(gy1) + self.Ugr(y1))
            z = torch.sigmoid(self.Wgz(gy1) + self.Ugz(y1) - self.bgg)
            h = torch.tanh(self.Wgg(gy1) + self.Ugg(r * y1))
            # z = output of a sigmoid, so never 0, but numerically stable ??????
            y2 = (out2 - (1 - z) * h) * 1 / z
            y2.grad = None

        with torch.enable_grad():
            y2.requires_grad = True
            y1.requires_grad = True
            gy1 = self.g(out1, record_rng=self.training, **g_args)
            r = torch.sigmoid(self.Wgr(gy1) + self.Ugr(out1))
            z = torch.sigmoid(self.Wgz(gy1) + self.Ugz(out1) - self.bgg)
            h = torch.tanh(self.Wgg(gy1) + self.Ugg(r * out1))
            yout2 = (1 - z) * y2 + z * h
            torch.autograd.backward(yout2, dout2)

        with torch.no_grad():
            dy1 = dout1 + y1.grad
            dy2 = y2.grad
            x = torch.cat([y1, y2.detach()], dim=2)
            dx = torch.cat([dy1, dy2], dim=2)

        # Attention layer
        with torch.no_grad():
            x1 = y1
            fx1, _ = self.f(x1, record_rng=self.training, **f_args)
            r = torch.sigmoid(self.Wfr(fx1) + self.Ufr(x1))
            z = torch.sigmoid(self.Wfz(fx1) + self.Ufz(x1) - self.bfg)
            h = torch.tanh(self.Wfg(fx1) + self.Ufg(r * x1))
            # z = output of a sigmoid, so never 0, but numerically stable ??????
            x2 = (y2 - (1 - z) * h) * 1 / z
            x2.grad = None

        with torch.enable_grad():
            y1.requires_grad = True
            fx1, _ = self.f(y1, record_rng=self.training, **f_args)
            r = torch.sigmoid(self.Wfr(fx1) + self.Ufr(y1))
            z = torch.sigmoid(self.Wfz(fx1) + self.Ufz(y1) - self.bfg)
            h = torch.tanh(self.Wfg(fx1) + self.Ufg(r * y1))
            yy2 = (1 - z) * x2 + z * h
            torch.autograd.backward(yy2, dy2)

        with torch.no_grad():
            dx1 = dy1 + yy2.grad
            dx2 = yy2.grad
            x = torch.cat([x1, x2.detach()], dim=2)
            dx = torch.cat([dx1, dx2], dim=2)

        return x, dx


class _ReversibleFunction_(Function):
    @staticmethod
    def forward(ctx, x, blocks, args):
        ctx.args = args
        states = []
        for layer_ind, (block, kwarg) in enumerate(zip(blocks, args)):
            # extract the states for the current layer
            f_args_layer = {k: (dict(Zs=v['Zs'][:, :, :, layer_ind], Ss=v['Ss'][:, :, :, :, layer_ind]) if
                                (k == 'states' and v is not None) else v) for k, v in kwarg['f_args'].items()}
            kwargs_layer = dict(f_args=f_args_layer, g_args=kwarg['g_args'])
            x, state = block(x, **kwargs_layer)
            if state is not None:
                states.append(state)
        ctx.y = x.detach()
        ctx.blocks = blocks
        # can't return a list in torch Function
        if len(states) > 0:
            Zs = torch.stack([st['Z'] for st in states], dim=-1)
            Ss = torch.stack([st['S'] for st in states], dim=-1)
        else:
            Zs = torch.zeros_like(x)
            Ss = torch.zeros_like(x)
        return x, Zs, Ss

    @staticmethod
    def backward(ctx, dy, dz, ds):
        y = ctx.y
        args = ctx.args
        for block, kwargs in zip(ctx.blocks[::-1], args[::-1]):
            y, dy = block.backward_pass(y, dy, **kwargs)
        return dy, None, None


class ReversibleGatedSequence_(nn.Module):
    def __init__(self, blocks, args_route={}):
        super().__init__()
        self.args_route = args_route
        self.blocks = nn.ModuleList(
            [ReversibleGatedBlock_(f=f, g=g) for f, g in blocks])

    def forward(self, x, **kwargs):
        x = torch.cat([x, x], dim=-1)
        blocks = self.blocks
        args = route_args(self.args_route, kwargs, len(blocks))
        args = list(map(lambda x: {'f_args': x[0], 'g_args': x[1]}, args))
        x, Zs, Ss = _ReversibleFunction_.apply(x, blocks, args)
        x = torch.stack(x.chunk(2, dim=-1)).sum(dim=0)
        return x, Zs, Ss


# if __name__ == "__main__":
#     model = ReversibleGatedSequence_
#     print("gradCheck :", torch.autograd.gradcheck((model, (b_x,))))