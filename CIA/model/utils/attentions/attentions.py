from CIA.model.utils.positional_embeddings.apply_pe import apply_rotary_pos_emb_, apply_spe_pos_emb_
import math
import torch
import torch.nn as nn
from einops import rearrange
from local_attention import LocalAttention
from functools import partial

from performer_pytorch.performer_pytorch import causal_linear_attention, causal_linear_attention_noncuda, default,\
    empty, exists, gaussian_orthogonal_random_matrix, generalized_kernel, linear_attention, softmax_kernel


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
        attn_out_bias=True,
        layer_pos_enc=None
    ):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        self.dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.feature_map = FeatureMap(dim_head, nb_features, generalized_attention=generalized_attention,
                                      kernel_fn=kernel_fn, no_projection=no_projection)
        self.fast_attention = FastAttention_(causal)

        self.heads = heads
        # assert local_heads == 0, 'Dont use local attention, incompatible with recursive transfofos'
        self.global_heads = heads - local_heads
        self.local_heads = local_heads
        self.local_attn = LocalAttention(window_size=local_window_size, causal=causal, autopad=True,
                                         dropout=dropout, look_forward=int(
                                             not causal),
                                         rel_pos_emb_config=(dim_head, local_heads)) if local_heads > 0 else None

        self.to_q = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=attn_out_bias)
        self.dropout = nn.Dropout(dropout)

        self.local_layerPE = layer_pos_enc['local_layerPE']
        self.post_phi_layerPE = layer_pos_enc['post_phi_layerPE']
        self.PE_type = layer_pos_enc['PE_type']
        self.input_dim_global = layer_pos_enc['input_dim_global']
        self.input_dim_local = layer_pos_enc['input_dim_local']
        self.gated = layer_pos_enc['gated_layerSPE']
        if self.gated and (self.PE_type == 'spe'):
            self.gate_global = nn.Parameter(torch.randn(self.global_heads, self.input_dim_global))
            self.gate_local = nn.Parameter(torch.randn(self.local_heads, self.input_dim_local))

    def forward(self, x, pos_emb=None, local_pos_emb=None, context=None, mask=None, context_mask=None, **kwargs):
        batch_size, length, _, h, gh = *x.shape, self.heads, self.global_heads

        # cross-attention
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

            if self.post_phi_layerPE:
                q, k = self.feature_map(q, k)

            if exists(pos_emb) and not cross_attend:
                if self.PE_type == 'rotary':
                    q, k = apply_rotary_pos_emb_(q, k, pos_emb)
                elif self.PE_type == 'spe':
                    qbar, kbar = torch.split(pos_emb, pos_emb.size(2)//2, dim=2)
                    qbar = torch.reshape(
                        qbar, (batch_size, length, gh, self.input_dim_global, -1))
                    kbar = torch.reshape(
                        kbar, (batch_size, length, gh, self.input_dim_global, -1))
                    code_shape = (gh, self.input_dim_global)
                    if self.gated:
                        ggate = self.gate_global
                    else:
                        ggate = None
                    q, k = apply_spe_pos_emb_(
                        q, k, qbar, kbar, code_shape, ggate)
                else:
                    raise NotImplementedError

            if not self.post_phi_layerPE:
                q, k = self.feature_map(q, k)

            out, state = self.fast_attention(
                q, k, v, kwargs['states'], kwargs['inferring_states'])
            attn_outs.append(out)

        if not empty(lq):
            assert not cross_attend, 'local attention is not compatible with cross attention'

            # Apply layer PE to local attention ? Not implemented in original implem or performer, but why ?
            if self.local_layerPE:
                if self.PE_type == 'rotary':
                    lq, lk = apply_rotary_pos_emb_(lq, lk, local_pos_emb)
                elif self.PE_type == 'spe':
                    qbar, kbar = torch.split(local_pos_emb, local_pos_emb.size(2)//2, dim=2)
                    lqbar = torch.reshape(
                        qbar, (batch_size, length, self.local_heads, self.input_dim_local, -1))
                    lkbar = torch.reshape(
                        kbar, (batch_size, length, self.local_heads, self.input_dim_local, -1))
                    code_shape = (self.local_heads, self.input_dim_local)
                    if self.gated:
                        lgate = self.gate_local
                    else:
                        lgate = None
                    lq, lk = apply_spe_pos_emb_(
                        lq, lk, lqbar, lkbar, code_shape, lgate)
                else:
                    raise NotImplementedError

            out = self.local_attn(lq, lk, lv, input_mask=mask)
            attn_outs.append(out)

        out = torch.cat(attn_outs, dim=1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out), state


class SelfAttention_(Attention_):
    def forward(self, *args, context=None, **kwargs):
        assert not exists(context), 'self attention should not receive context'
        return super().forward(*args, **kwargs)


class CrossAttention_(Attention_):
    def forward(self, *args, context=None, **kwargs):
        assert exists(context), 'cross attention should receive context'
        return super().forward(*args, context=context, **kwargs)


class FeatureMap(nn.Module):
    def __init__(self, dim_heads, nb_features=None, ortho_scaling=0, causal=False, generalized_attention=False,
                 kernel_fn=nn.ReLU(), no_projection=False):
        super().__init__()
        nb_features = default(nb_features, int(
            dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(
            gaussian_orthogonal_random_matrix, nb_rows=self.nb_features, nb_columns=dim_heads, scaling=ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k):
        device = q.device
        if self.no_projection:
            q = q.softmax(dim=-1)
            k = torch.exp(k) if self.causal else k.softmax(dim=-2)
        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn=self.kernel_fn,
                                    projection_matrix=self.projection_matrix, device=device)
            q, k = map(create_kernel, (q, k))
        else:
            create_kernel = partial(
                softmax_kernel, projection_matrix=self.projection_matrix, device=device)
            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)
        return q, k


class FastAttention_(nn.Module):
    def __init__(self, causal=False):
        super().__init__()
        self.causal = causal
        if causal:
            try:
                import fast_transformers.causal_product.causal_product_cuda
                self.causal_linear_fn = partial(causal_linear_attention)
            except ImportError:
                print('unable to import cuda code for auto-regressive Performer. will default to the memory inefficient non-cuda version')
                self.causal_linear_fn = causal_linear_attention_noncuda

    def forward(self, q, k, v, states, inferrring_states):
        """
        inputs are already feature mapped
        """
        if states is not None:
            assert q.size(
                2) == 1, 'recurrent inference can only be applied to sequences of len 1'
            out, states = recursive_attention_step(q, k, v, states)
        else:
            if inferrring_states:
                out, states = infer_hidden_states(q, k, v)
            else:
                attn_fn = linear_attention if not self.causal else self.causal_linear_fn
                out = attn_fn(q, k, v)
                states = None
        return out, states


# inefficient causal linear attention, without cuda code,
# (used for parallel inference of hidden states in recurrent mode)
def infer_hidden_states(q, k, v, chunk_size=128, eps=1e-6):
    last_k_cumsum = 0
    last_context_cumsum = 0
    outs = []
    num_chunks = q.size(2) // chunk_size
    for q, k, v in zip(*map(lambda t: t.chunk(num_chunks, dim=-2), (q, k, v))):
        k_cumsum = last_k_cumsum + k.cumsum(dim=-2)
        D_inv = 1. / torch.einsum('...nd,...nd->...n',
                                  q, k_cumsum.type_as(q) + eps)
        context = torch.einsum('...nd,...ne->...nde', k, v)
        context_cumsum = last_context_cumsum + context.cumsum(dim=-3)
        out = torch.einsum('...nde,...nd,...n->...ne',
                           context_cumsum, q, D_inv)
        last_k_cumsum = k_cumsum[:, :, -1:]
        last_context_cumsum = context_cumsum[:, :, -1:]
        outs.append(out)

    out = torch.cat(outs, dim=-2)
    states = dict(Z=last_k_cumsum.squeeze(2), S=last_context_cumsum.squeeze(2))
    return out, states


def recursive_attention_step(q, k, v, states, eps=1e-6):
    k_cumsum = states['Zs'].unsqueeze(2) + k
    D_inv = 1. / torch.einsum('...nd,...nd->...n', q,
                              k_cumsum.type_as(q) + eps)
    context = torch.einsum('...nd,...ne->...nde', k, v)
    context_cumsum = states['Ss'].unsqueeze(2) + context
    out = torch.einsum('...nde,...nd,...n->...ne',
                       context_cumsum, q, D_inv)
    last_k_cumsum = k_cumsum[:, :, 0]
    last_context_cumsum = context_cumsum[:, :, 0]
    states = dict(Z=last_k_cumsum, S=last_context_cumsum)
    return out, states
