from CIA.model.utils.positional_embeddings.pe_modules.elapsed_rototor import ElapsedRototor
from CIA.model.utils.positional_embeddings.pe_modules.index_rototor import Rototor
from CIA.model.utils.positional_embeddings.pe_modules.index_spe_factorized import SineSPEFactorized
from CIA.model.utils.positional_embeddings.pe_modules.index_spe import SineSPE
from CIA.model.utils.positional_embeddings.pe_modules.elapsed_positional_embedding import ElapsedPositionalEmbedding
from CIA.model.utils.positional_embeddings.pe_modules.index_positional_embedding import IndexPositionalEmbedding
from performer_pytorch.performer_pytorch import APEX_AVAILABLE, default,\
    empty, exists, gaussian_orthogonal_random_matrix, generalized_kernel, linear_attention, null_context, softmax_kernel
from torch.cuda.amp import autocast
from functools import partial
from local_attention import LocalAttention
from einops import rearrange
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from CIA.model.utils.positional_embeddings.apply_pe import apply_rotary_pos_emb_, apply_rototor_pos_emb_, \
    apply_spe_pos_emb_, apply_spe_pos_emb_factorised
try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False


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
        layer_pe_type=None,
        layer_pos_enc=None,
        max_seq_len=None,
        dataloader_generator=None,
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

        # linear mapping to q, k, v
        self.to_q = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=attn_out_bias)
        self.dropout = nn.Dropout(dropout)

        # positional encodings
        self.local_layerPE = layer_pos_enc['local_layerPE']
        self.post_phi_layerPE = layer_pos_enc['post_phi_layerPE']
        self.input_dim_global = layer_pos_enc['input_dim_global']
        self.input_dim_local = layer_pos_enc['input_dim_local']
        self.gated = layer_pos_enc['gated_layerSPE']
        self.upsampled_layerPE = layer_pos_enc['upsampled_layerPE']
        layer_pe_args = layer_pos_enc['layer_pe_args']
        if self.gated and (self.PE_type in ['spe', 'spe_factorized']):
            self.gate_global = nn.Parameter(torch.randn(
                self.global_heads, self.input_dim_global))
            self.gate_local = nn.Parameter(torch.randn(
                self.local_heads, self.input_dim_local))
        self.to_theta_q = nn.Linear(
            dim, self.input_dim_global*self.global_heads, bias=qkv_bias)

        self.layer_pos_emb, self.layer_pos_emb_local, PE_type = \
            get_pes(layer_pe_type, n_global_heads=self.heads, n_local_heads=self.local_heads,
                    dim_layerPE=self.input_dim_global, dim_layerPE_local=self.input_dim_local,
                    max_seq_len=max_seq_len, dataloader_generator=dataloader_generator,
                    layer_pe_args=layer_pe_args)
        self.PE_type = PE_type

    def forward(self, x, pos_emb_input=None, context=None, mask=None, context_mask=None,
                **kwargs):
        batch_size, length, _, h, gh = *x.shape, self.heads, self.global_heads

        # compute layer positional embeddings. pos_emb represents the input to the pe module, i.e. indices or
        # elapsed time
        # if self.layer_pos_emb is not None:

        # if self.layer_pos_emb_local is not None:
        #     local_pos_emb = self.layer_pos_emb_local(
        #         pos_emb_input=pos_emb_input)

        # cross-attention
        cross_attend = exists(context)
        context = default(context, x)
        context_mask = default(
            context_mask, mask) if not cross_attend else context_mask

        if self.to_theta_q is not None:
            q, k, v, theta_q = self.to_q(x), self.to_k(
                context), self.to_v(context), self.to_theta_q(x)
        else:
            q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        q, k, v, theta_q = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=h) if t is not None else None, (q, k, v, theta_q))
        # split between global and local heads
        (q, lq), (k, lk), (v, lv), (theta_q, l_theta_q) = map(
            lambda t: (t[:, :gh], t[:, gh:]) if t is not None else None, (q, k, v, theta_q))

        attn_outs = []

        if not empty(q):
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]
                v.masked_fill_(~global_mask, 0.)

            if self.post_phi_layerPE:
                # q, k = self.feature_map(q, k)
                q, k = map(lambda t: F.elu(t) + 1, (q, k))

            if pos_emb_input is not None:
                if self.PE_type == 'rototor':
                    pos_emb_q = self.layer_pos_emb(
                        pe_input=pos_emb_input, offset=theta_q)
                    pos_emb_k = self.layer_pos_emb(
                        pe_input=pos_emb_input, offset=None)
                    q_rot = apply_rototor_pos_emb_(q, pos_emb_q)
                    k_rot = apply_rototor_pos_emb_(k, pos_emb_k)
                # if self.PE_type == 'rotary':
                #     q, k = apply_rotary_pos_emb_(q, k, pos_emb)
                # elif self.PE_type in ['spe', 'spe_factorized']:
                #     if self.upsampled_layerPE:
                #         raise NotImplementedError
                #     qbar, kbar = torch.split(
                #         pos_emb, pos_emb.size(2)//2, dim=2)
                #     if self.PE_type == 'spe_factorized':
                #         qbar = torch.reshape(
                #             qbar, (batch_size, length, gh, -1)).permute(0, 2, 1, 3)
                #         kbar = torch.reshape(
                #             kbar, (batch_size, length, gh, -1)).permute(0, 2, 1, 3)
                #         code_shape = (gh, self.input_dim_global)
                #         if self.gated:
                #             ggate = self.gate_global
                #         else:
                #             ggate = None
                #         q, k = apply_spe_pos_emb_factorised(
                #             q, k, qbar, kbar, code_shape, ggate)
                #         q, k = map(lambda t: t.permute(0, 1, 4, 2, 3), (q, k))
                #         b, h, r_pe, l, r_f = q.shape
                #         q, k = map(lambda t: t.reshape(
                #             b, h * r_pe, l, r_f), (q, k))
                #     else:
                #         qbar = torch.reshape(
                #             qbar, (batch_size, length, gh, self.input_dim_global, -1))
                #         kbar = torch.reshape(
                #             kbar, (batch_size, length, gh, self.input_dim_global, -1))
                #         code_shape = (gh, self.input_dim_global)
                #         if self.gated:
                #             ggate = self.gate_global
                #         else:
                #             ggate = None
                #         q, k = apply_spe_pos_emb_(
                #             q, k, qbar, kbar, code_shape, ggate)
                # else:
                #     raise NotImplementedError

            if torch.any(torch.isnan(q)):
                print('stop')
            if torch.any(torch.isnan(k)):
                print('stop')

            if not self.post_phi_layerPE:
                q, k = self.feature_map(q, k)

            if torch.any(torch.isnan(q)):
                print('stop')
            if torch.any(torch.isnan(k)):
                print('stop')

            out, state = self.fast_attention(
                q, k, q_rot, k_rot, v, kwargs['states'], kwargs['inferring_states'])
            attn_outs.append(out)

        if not empty(lq):
            assert not cross_attend, 'local attention is not compatible with cross attention'
            # Apply layer PE to local attention ? Not implemented in original implem or performer, but why ?
            if self.local_layerPE:
                if self.PE_type == 'rotary':
                    lq, lk = apply_rotary_pos_emb_(lq, lk, local_pos_emb)
                elif self.PE_type == 'spe':
                    qbar, kbar = torch.split(
                        local_pos_emb, local_pos_emb.size(2)//2, dim=2)
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
            self.causal_linear_fn = partial(causal_linear_attention)

    def forward(self, q, k, q_rot, k_rot, v, states, inferrring_states):
        """
        inputs are already feature mapped
        """
        if states is not None:
            assert q.size(
                2) == 1, 'recurrent inference can only be applied to sequences of len 1'
            out, states = recursive_attention_step(q, k, q_rot, k_rot, v, states)
        else:
            if inferrring_states:
                out, states = infer_hidden_states(q, k, q_rot, k_rot, v)
            else:
                attn_fn = linear_attention if not self.causal else self.causal_linear_fn
                out = attn_fn(q, k, q_rot, k_rot, v)
                states = None
        return out, states


# inefficient causal linear attention, without cuda code,
# (used for parallel inference of hidden states in recurrent mode)
def infer_hidden_states(q, k, q_rot, k_rot, v, chunk_size=128, eps=1e-6):
    last_k_cumsum = 0
    last_context_cumsum = 0
    last_k_cumsum_rot = 0
    last_context_cumsum_rot = 0
    outs = []
    num_chunks = q.size(2) // chunk_size
    for q, k, q_rot, k_rot, v in zip(*map(lambda t: t.chunk(num_chunks, dim=-2), (q, k, q_rot, k_rot, v))):
        k_cumsum = last_k_cumsum + k.cumsum(dim=-2)
        D = torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q))
        k_cumsum_rot = last_k_cumsum_rot + k_rot.cumsum(dim=-2)
        D_rot = torch.einsum('...nd,...nd->...n', q_rot,
                             k_cumsum_rot.type_as(q_rot))
        D_inv = 1. / (D + D_rot + eps)

        context = torch.einsum('...nd,...ne->...nde', k, v)
        context_cumsum = last_context_cumsum + context.cumsum(dim=-3)
        context_rot = torch.einsum('...nd,...ne->...nde', k_rot, v)
        context_cumsum_rot = last_context_cumsum_rot + \
            context_rot.cumsum(dim=-3)

        out = torch.einsum('...nde,...nd,...n->...ne',
                           context_cumsum, q, D_inv)
        out_rot = torch.einsum('...nde,...nd,...n->...ne',
                               context_cumsum_rot, q_rot, D_inv)
        out = out + out_rot

        last_k_cumsum = k_cumsum[:, :, -1:]
        last_context_cumsum = context_cumsum[:, :, -1:]
        last_k_cumsum_rot = k_cumsum_rot[:, :, -1:]
        last_context_cumsum_rot = context_cumsum_rot[:, :, -1:]
        outs.append(out)

    out = torch.cat(outs, dim=-2)
    states = dict(Z=last_k_cumsum.squeeze(2), S=last_context_cumsum.squeeze(2),
                  Z_rot=last_k_cumsum_rot.squeeze(2), S_rot=last_context_cumsum_rot.squeeze(2))
    return out, states


def recursive_attention_step(q, k, q_rot, k_rot, v, states, eps=1e-6):
    k_cumsum = states['Zs'].unsqueeze(2) + k
    k_cumsum_rot = states['Zs_rot'].unsqueeze(2) + k_rot
    D = torch.einsum('...nd,...nd->...n', q,
                     k_cumsum.type_as(q))
    D_rot = torch.einsum('...nd,...nd->...n', q_rot,
                         k_cumsum_rot.type_as(q_rot))
    D_inv = 1. / (D + D_rot + eps)

    context = torch.einsum('...nd,...ne->...nde', k, v)
    context_cumsum = states['Ss'].unsqueeze(2) + context
    context_rot = torch.einsum('...nd,...ne->...nde', k_rot, v)
    context_cumsum_rot = states['Ss_rot'].unsqueeze(2) + context_rot

    out = torch.einsum('...nde,...nd,...n->...ne',
                       context_cumsum, q, D_inv)
    out_rot = torch.einsum('...nde,...nd,...n->...ne',
                           context_cumsum_rot, q_rot, D_inv)
    out = out_rot + out
    last_k_cumsum = k_cumsum[:, :, 0]
    last_k_cumsum_rot = k_cumsum_rot[:, :, 0]
    last_context_cumsum = context_cumsum[:, :, 0]
    last_context_cumsum_rot = context_cumsum_rot[:, :, 0]
    states = dict(Z=last_k_cumsum, S=last_context_cumsum,
                  Z_rot=last_k_cumsum_rot, S_rot=last_context_cumsum_rot)
    return out, states


def get_D(q, k):
    k_cumsum = k.cumsum(dim=-2)
    D = torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q))
    return D


def get_N(q, k, v):
    from fast_transformers.causal_product import CausalDotProduct
    autocast_enabled = torch.is_autocast_enabled()
    is_half = isinstance(q, torch.cuda.HalfTensor)
    assert not is_half or APEX_AVAILABLE, 'half tensors can only be used if nvidia apex is available'
    cuda_context = null_context if not autocast_enabled else partial(
        autocast, enabled=False)
    causal_dot_product_fn = amp.float_function(
        CausalDotProduct.apply) if is_half else CausalDotProduct.apply
    with cuda_context():
        if autocast_enabled:
            q, k, v = map(
                lambda t: t.float(), (q, k, v))
        N = causal_dot_product_fn(q, k, v)
    return N


def causal_linear_attention(q, k, q_rot, k_rot, v, eps=1e-10):
    N = (get_N(q, k, v) + get_N(q_rot, k_rot, v))
    D_inv = 1. / (get_D(q, k) + get_D(q_rot, k_rot) + eps)
    out = torch.einsum('...nd,...n->...nd', N, D_inv)
    return out


def get_pes(layer_pe_type, n_global_heads, n_local_heads, dim_layerPE, dim_layerPE_local, max_seq_len,
            dataloader_generator, layer_pe_args):
    layer_pos_emb_local = None
    if layer_pe_type == 'index_rototor':
        layer_pos_emb = Rototor(
            dim=dim_layerPE)
        if n_local_heads > 0:
            layer_pos_emb_local = Rototor(
                dim=dim_layerPE_local)
        PE_type = 'rototor'
    elif layer_pe_type == 'elapsed_rototor':
        layer_pos_emb = ElapsedRototor(
            dim=dim_layerPE)
        if n_local_heads > 0:
            layer_pos_emb_local = ElapsedRototor(
                dim=dim_layerPE_local)
        PE_type = 'rototor'
    elif layer_pe_type == 'index_rotary':
        raise NotImplementedError
        layer_pos_emb = IndexPositionalEmbedding(
            dim=dim_layerPE, max_seq_len=max_seq_len)
        if n_local_heads > 0:
            layer_pos_emb_local = IndexPositionalEmbedding(
                dim=dim_layerPE_local, max_seq_len=max_seq_len)
        PE_type = 'rotary'
    elif layer_pe_type == 'elapsed_rotary':
        raise NotImplementedError
        layer_pos_emb = ElapsedPositionalEmbedding(
            dim=dim_layerPE, dataloader_generator=dataloader_generator)
        if n_local_heads > 0:
            layer_pos_emb_local = ElapsedPositionalEmbedding(
                dim=dim_layerPE_local, dataloader_generator=dataloader_generator)
        PE_type = 'rotary'
    elif layer_pe_type == 'index_spe':
        raise NotImplementedError
        num_sines = layer_pe_args['n_sines']
        num_realizations = layer_pe_args['n_realizations']
        num_local_head = n_local_heads
        poscoder = SineSPE(num_heads=n_global_heads, in_features=dim_layerPE,
                           num_sines=num_sines, num_realizations=num_realizations)
        layer_pos_emb = poscoder
        if n_local_heads > 0:
            poscoder_local = SineSPE(num_heads=num_local_head, in_features=dim_layerPE_local,
                                     num_sines=num_sines, num_realizations=num_realizations)
            layer_pos_emb_local = poscoder_local
        PE_type = 'spe'
    elif layer_pe_type == 'index_spe_factorized':
        raise NotImplementedError
        layer_pe_args = layer_pe_args
        num_sines = layer_pe_args['n_sines']
        num_realizations = layer_pe_args['n_realizations']
        num_local_head = n_local_heads
        poscoder = SineSPEFactorized(
            num_heads=n_global_heads, num_sines=num_sines, num_realizations=num_realizations)
        layer_pos_emb = poscoder
        if n_local_heads > 0:
            poscoder_local = SineSPEFactorized(
                num_heads=num_local_head, num_sines=num_sines, num_realizations=num_realizations)
            layer_pos_emb_local = poscoder_local
        PE_type = 'spe_factorized'
    else:
        raise NotImplementedError
    return layer_pos_emb, layer_pos_emb_local, PE_type
