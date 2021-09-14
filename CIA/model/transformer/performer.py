from numpy import log
from CIA.dataloaders import dataloader
from CIA.model.execute_type.reversible_gated import ReversibleGatedSequence_
from CIA.model.execute_type.gated import GatedSequence_
from CIA.model.execute_type.reversible import ReversibleSequence_
from CIA.model.attentions.attentions import CrossAttention_, SelfAttention_
import torch.nn as nn
from functools import partial

from performer_pytorch.performer_pytorch import Chunk, FeedForward, PreLayerNorm, PreScaleNorm, ProjectionUpdater,\
    ReZero, cast_tuple


class Performer_(nn.Module):
    def __init__(
        self,
        *,
        max_seq_len,
        dim,
        depth,
        heads,
        local_attn_heads=0,
        local_window_size=256,
        fast_local_attn=None,
        causal=False,
        ff_mult=4,
        features=None,
        feature_redraw_interval=100000,
        execute_type=None,
        ff_chunks=1,
        ff_glu=False,
        emb_dropout=0.,
        ff_dropout=0.,
        attn_dropout=0.,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        use_scalenorm=False,
        use_rezero=False,
        cross_attend=False,
        no_projection=False,
        auto_check_redraw=True,
        qkv_bias=False,
        attn_out_bias=False,
        layer_pe=None,
        dataloader_generator=None
    ):
        super().__init__()
        local_attn_heads = cast_tuple(local_attn_heads)

        self.max_seq_len = max_seq_len
        self.dim_last_layer = dim

        self.dropout = nn.Dropout(emb_dropout)
        self.norm = nn.LayerNorm(dim)

        self.performer = _Performer_(dim, depth, heads, local_attn_heads, local_window_size, fast_local_attn,
                                     causal, ff_mult,
                                     features, feature_redraw_interval, execute_type, ff_chunks,
                                     generalized_attention, kernel_fn, use_scalenorm, use_rezero, ff_glu, ff_dropout,
                                     attn_dropout, cross_attend, no_projection, auto_check_redraw,
                                     qkv_bias, attn_out_bias, layer_pe, max_seq_len, dataloader_generator)

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(self, x, pos_emb_input, **kwargs):
        b, n, d = x.shape
        assert n <= self.max_seq_len, f'sequence length {n} must be less than \
            the max sequence length {self.max_seq_len}'
        x = self.dropout(x)
        out = self.performer(x, pos_emb_input=pos_emb_input, **kwargs)
        # pre-softmax norm (improve training stability)
        out['x'] = self.norm(out['x'])
        return out


class _Performer_(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        local_attn_heads=0,
        local_window_size=256,
        fast_local_attn=None,
        causal=False,
        ff_mult=4,
        features=None,
        feature_redraw_interval=100000,
        execute_type=None,
        ff_chunks=1,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        use_scalenorm=False,
        use_rezero=False,
        ff_glu=False,
        ff_dropout=0.,
        attn_dropout=0.,
        cross_attend=False,
        no_projection=False,
        auto_check_redraw=True,
        qkv_bias=True,
        attn_out_bias=True,
        layer_pe=None,
        max_seq_len=None,
        dataloader_generator=None
    ):
        super().__init__()
        dim_head = dim // heads
        layers = nn.ModuleList([])
        local_attn_heads = cast_tuple(local_attn_heads)
        local_attn_heads = local_attn_heads * \
            depth if len(local_attn_heads) == 1 else local_attn_heads
        assert len(
            local_attn_heads) == depth, 'tuple specifying number of local attention heads per depth \
                must be equal to the total depth'
        assert all(map(lambda n: n >= 0 and n <= heads, local_attn_heads)
                   ), 'local attention head value must be less than the total number of heads'

        if use_scalenorm:
            wrapper_fn = partial(PreScaleNorm, dim)
        elif use_rezero:
            wrapper_fn = ReZero
        else:
            wrapper_fn = partial(PreLayerNorm, dim)

        for _, local_heads in zip(range(depth), local_attn_heads):
            layers.append(nn.ModuleList([
                wrapper_fn(SelfAttention_(input_dim=dim, output_dim=dim, causal=causal, heads=heads, local_heads=local_heads,
                                          fast_local_attn=fast_local_attn,
                                          local_window_size=local_window_size, features=features,
                                          generalized_attention=generalized_attention, kernel_fn=kernel_fn,
                                          dropout=attn_dropout, no_projection=no_projection,
                                          qkv_bias=qkv_bias, attn_out_bias=attn_out_bias,
                                          layer_pe=layer_pe,
                                          dataloader_generator=dataloader_generator)),
                wrapper_fn(Chunk(ff_chunks, FeedForward(
                    dim, mult=ff_mult, dropout=ff_dropout, glu=ff_glu), along_dim=1))
            ]))

            if not cross_attend:
                continue

            layers.append(nn.ModuleList([
                wrapper_fn(CrossAttention_(dim, heads=heads, dim_head=dim_head, features=features,
                                           generalized_attention=generalized_attention, kernel_fn=kernel_fn,
                                           dropout=attn_dropout, no_projection=no_projection, qkv_bias=qkv_bias,
                                           attn_out_bias=attn_out_bias,
                                           layer_pe=layer_pe)),
                wrapper_fn(Chunk(ff_chunks, FeedForward(
                    dim, mult=ff_mult, dropout=ff_dropout, glu=ff_glu), along_dim=1))
            ]))

        execute_type_ = execute_type
        if execute_type == 'reversible':
            execute_type = ReversibleSequence_
        elif execute_type == 'gated':
            execute_type = GatedSequence_
        elif execute_type == 'reversible_gated':
            execute_type = ReversibleGatedSequence_
        # elif execute_type == 'residual':
        #     execute_type = SequentialSequence_
        else:
            raise NotImplementedError

        route_attn = ((True, False),) * depth * (2 if cross_attend else 1)
        route_context = ((False, False), (True, False)) * depth
        attn_route_map = {'mask': route_attn, 'pos_emb_input': route_attn,
                          'inferring_states': route_attn, 'states': route_attn}
        context_route_map = {'context': route_context,
                             'context_mask': route_context} if cross_attend else {}
        if execute_type_ == 'gated':
            self.net = execute_type(
                layers, args_route={**attn_route_map, **context_route_map}, d_model=dim)
        else:
            self.net = execute_type(
                layers, args_route={**attn_route_map, **context_route_map})

        # keeping track of when to redraw projections for all attention layers
        self.auto_check_redraw = auto_check_redraw
        self.proj_updater = ProjectionUpdater(
            self.net, feature_redraw_interval)

    def fix_projection_matrices_(self):
        self.proj_updater.feature_redraw_interval = None

    def forward(self, x, **kwargs):
        if self.auto_check_redraw:
            self.proj_updater.redraw_projections()
        if kwargs['inferring_states']:
            x, Zs, Ss, Zs_rot, Ss_rot = self.net(x, **kwargs)
            return dict(x=x, Zs=Zs, Ss=Ss, Zs_rot=Zs_rot, Ss_rot=Ss_rot)
        else:
            x = self.net(x, **kwargs)
            return dict(x=x)
