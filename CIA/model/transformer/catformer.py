from performer_pytorch.performer_pytorch import Chunk, FeedForward, PreLayerNorm, PreScaleNorm, ReZero, cast_tuple
from performer_pytorch.reversible import route_args
import torch
from CIA.model.attentions.attentions import SelfAttention_
import torch.nn as nn
from functools import partial


class Catformer(nn.Module):
    def __init__(
        self,
        dim_first_layer,
        expansion_factor_attn,
        expansion_factor_ff,
        depth,
        heads,
        local_attn_heads,
        local_window_size,
        fast_local_attn,
        features,
        ff_chunks,
        ff_glu,
        emb_dropout,
        ff_dropout,
        attn_dropout,
        layer_pe,
        dataloader_generator,
        use_scalenorm=False,
        use_rezero=False,
        cross_attend=False,
        no_projection=False,
        qkv_bias=False,
    ):
        super().__init__()
        local_attn_heads = cast_tuple(local_attn_heads)
        self.dropout = nn.Dropout(emb_dropout)
        self.dim_last_layer = dim_first_layer * (2 * depth + 1)
        self.norm = nn.LayerNorm(self.dim_last_layer)

        layers = nn.ModuleList([])
        local_attn_heads = cast_tuple(local_attn_heads)
        local_attn_heads = local_attn_heads * \
            depth if len(local_attn_heads) == 1 else local_attn_heads
        assert len(
            local_attn_heads) == depth, 'tuple specifying number of local attention heads per depth \
                must be equal to the total depth'
        assert all(map(lambda n: n >= 0 and n <= heads, local_attn_heads)
                   ), 'local attention head value must be less than the total number of heads'

        for lay_index, local_heads in zip(range(depth), local_attn_heads):
            dim_attn = 2 * (lay_index+1) * dim_first_layer - dim_first_layer
            dim_ff = 2 * (lay_index+1) * dim_first_layer
            if use_scalenorm:
                wrapper_fn_attn = partial(PreLayerNorm, dim_attn)
                wrapper_fn_ff = partial(PreLayerNorm, dim_ff)
            elif use_rezero:
                wrapper_fn_attn = ReZero
                wrapper_fn_ff = ReZero
            else:
                wrapper_fn_attn = partial(PreLayerNorm, dim_attn)
                wrapper_fn_ff = partial(PreLayerNorm, dim_ff)

            layers.append(nn.ModuleList([
                wrapper_fn_attn(SelfAttention_(input_dim=dim_attn, output_dim=dim_first_layer, causal=True, heads=heads, local_heads=local_heads,
                                          fast_local_attn=fast_local_attn,
                                          local_window_size=local_window_size, features=features,
                                          expansion_factor_attn=expansion_factor_attn,
                                          dropout=attn_dropout, no_projection=False,
                                          qkv_bias=False, attn_out_bias=False,
                                          layer_pe=layer_pe,
                                          dataloader_generator=dataloader_generator)),
                wrapper_fn_ff(Chunk(ff_chunks, FeedForwardCat(
                    input_dim=dim_ff, output_dim=dim_first_layer, mult=expansion_factor_ff, dropout=ff_dropout, glu=ff_glu), along_dim=1))
            ]))
        self.layers = layers

        route_attn = ((True, False),) * depth * (2 if cross_attend else 1)
        route_context = ((False, False), (True, False)) * depth
        attn_route_map = {'mask': route_attn, 'pos_emb_input': route_attn,
                          'inferring_states': route_attn, 'states': route_attn}
        context_route_map = {'context': route_context,
                             'context_mask': route_context} if cross_attend else {}
        self.args_route = {**attn_route_map, **context_route_map}

    def forward(self, x, **kwargs):
        # input dropout
        x = self.dropout(x)

        if kwargs['inferring_states']:
            x, Zs, Ss, Zs_rot, Ss_rot = self.net(x, **kwargs)
            out = dict(x=x, Zs=Zs, Ss=Ss, Zs_rot=Zs_rot, Ss_rot=Ss_rot)
        else:
            args = route_args(self.args_route, kwargs, len(self.layers))
            layers_and_args_and_gatings = list(zip(self.layers, args))
            states = []
            for layer_ind, ((f, g), (f_args, g_args)) in enumerate(layers_and_args_and_gatings):
                f_args_layer = {
                    k: (dict(Zs=v['Zs'][:, :, :, layer_ind],
                            Ss=v['Ss'][:, :, :, :, layer_ind]) if
                        (k == 'states' and v is not None) else v)
                    for k, v in f_args.items()
                }

                # attention
                f_x, state = f(x, **f_args_layer)
                x = torch.cat((x, f_x), dim=-1)

                # feed-forward
                g_x = g(x, **g_args)
                x = torch.cat((x, g_x), dim=-1)

                if state is not None:
                    states.append(state)
            if len(states) > 0:
                Zs = torch.stack([st['Z'] for st in states], dim=-1)
                Ss = torch.stack([st['S'] for st in states], dim=-1)
            else:
                Zs = torch.zeros_like(x)
                Ss = torch.zeros_like(x)

        # pre-softmax norm (improve training stability)
        x = self.norm(x)

        return dict(x=x)


class FeedForwardCat(nn.Module):
    def __init__(self, input_dim, output_dim, mult, dropout, glu):
        super().__init__()
        self.glu = glu
        self.w1 = nn.Linear(input_dim, output_dim * mult * (2 if glu else 1))
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(output_dim * mult, output_dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x