import torch
import torch.nn as nn
from einops import rearrange, repeat
from local_attention import LocalAttention

from performer_pytorch.performer_pytorch import FastAttention, default, empty, exists


class Attention_(nn.Module):
    def __init__(
        self,
        dim,
        causal=False,
        heads=8,
        dim_head=64,
        local_heads=0,
        local_window_size=256,
        nb_features=None,
        feature_redraw_interval=1000,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        dropout=0.,
        no_projection=False,
        qkv_bias=False,
        attn_out_bias=True
    ):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.fast_attention = FastAttention(
            dim_head, nb_features, causal=causal, generalized_attention=generalized_attention,
            kernel_fn=kernel_fn, no_projection=no_projection)

        self.heads = heads
        self.global_heads = heads - local_heads
        self.local_attn = LocalAttention(window_size=local_window_size, causal=causal, autopad=True,
                                         dropout=dropout, look_forward=int(not causal),
                                         rel_pos_emb_config=(dim_head, local_heads)) if local_heads > 0 else None

        self.to_q = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=attn_out_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb=None, context=None, mask=None, context_mask=None, **kwargs):
        b, n, _, h, gh = *x.shape, self.heads, self.global_heads

        cross_attend = exists(context)

        context = default(context, x)
        context_mask = default(
            context_mask, mask) if not cross_attend else context_mask

        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        # split between global and local heads
        (q, lq), (k, lk), (v, lv) = map(
            lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))

        attn_outs = []

        if not empty(q):
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]
                v.masked_fill_(~global_mask, 0.)

            if exists(pos_emb) and not cross_attend:
                q, k = apply_rotary_pos_emb_(q, k, pos_emb)

            out = self.fast_attention(q, k, v)
            attn_outs.append(out)

        if not empty(lq):
            assert not cross_attend, 'local attention is not compatible with cross attention'
            out = self.local_attn(lq, lk, lv, input_mask=mask)
            attn_outs.append(out)

        out = torch.cat(attn_outs, dim=1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)


class SelfAttention_(Attention_):
    def forward(self, *args, context=None, **kwargs):
        assert not exists(context), 'self attention should not receive context'
        return super().forward(*args, **kwargs)


class CrossAttention_(Attention_):
    def forward(self, *args, context=None, **kwargs):
        assert exists(context), 'cross attention should receive context'
        return super().forward(*args, context=context, **kwargs)


def rotate_every_two_(x):
    x = rearrange(x, '... (d j) -> ... d j', j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb_(q, k, sinu_pos):
    sinu_pos = rearrange(sinu_pos, 'b n (j d) -> b n j d', j=2)
    sin, cos = sinu_pos.unbind(dim=-2)
    sin, cos = map(lambda t: repeat(t, 'b t n -> b t (n j)', j=2), (sin, cos))
    # TODO: use same positional embeddings for all heads ?? perhaps can be changed when parametrising thetas
    sin_heads, cos_heads = map(lambda t: t.unsqueeze(1), (sin, cos))  # unsqueeze for head dim
    q, k = map(lambda t: (t * cos_heads) + (rotate_every_two_(t) * sin_heads), (q, k))
    return q, k
