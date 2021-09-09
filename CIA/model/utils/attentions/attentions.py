from numpy import inner
from CIA.model.utils.positional_embeddings.pe_modules.rotary import Rotary
from CIA.model.utils.positional_embeddings.pe_modules.rototor import Rototor
from CIA.model.utils.positional_embeddings.pe_modules.index_spe import SineSPE
from performer_pytorch.performer_pytorch import default,\
    empty, exists, gaussian_orthogonal_random_matrix, generalized_kernel, linear_attention, null_context, softmax_kernel
from torch.cuda.amp import autocast
from functools import partial
from local_attention import LocalAttention
from einops import rearrange
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from CIA.model.utils.positional_embeddings.apply_pe import apply_rotary_pos_emb_, apply_rototor_pos_emb_
try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False


class Attention_(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        causal=False,
        heads=8,
        local_heads=0,
        fast_local_attn=None,
        local_window_size=256,
        expansion_factor_attn=None,
        features=None,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        dropout=0.,
        no_projection=False,
        qkv_bias=False,
        attn_out_bias=True,
        layer_pe=None,
        dataloader_generator=None,
    ):
        super().__init__()
        assert input_dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = input_dim // heads
        if expansion_factor_attn is not None:
            inner_dim = expansion_factor_attn * output_dim
        else:
            inner_dim = dim_head * heads
        self.features_type = features['type']
        if self.features_type is None:
            pass
        elif self.features_type == 'favor':
            nb_features = features['args']['n_features']
            self.feature_map = FeatureMap(dim_head, nb_features, generalized_attention=generalized_attention,
                                          kernel_fn=kernel_fn, no_projection=no_projection)
        elif self.features_type == 'elu':
            pass
        else:
            raise NotImplementedError

        if self.features_type is not None:
            self.fast_attention = FastAttention_(causal)

        self.heads = heads
        # assert local_heads == 0, 'Dont use local attention, incompatible with recursive transfofos'
        self.global_heads = heads - local_heads
        self.local_heads = local_heads
        self.fast_local_attn = fast_local_attn
        if fast_local_attn:
            self.local_attn = FastAttention_(
                causal) if local_heads > 0 else None
        else:
            # rel_pos_emb_config=(dim_head, local_heads)  # LEGACY
            rel_pos_emb_config= None
            self.local_attn = LocalAttention(window_size=local_window_size, causal=causal, autopad=True,
                                             dropout=dropout, look_forward=int(
                                                 not causal),
                                             rel_pos_emb_config=rel_pos_emb_config) if local_heads > 0 else None

        # linear mapping to q, k, v
        self.to_q = nn.Linear(input_dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(input_dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(input_dim, inner_dim, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, output_dim, bias=attn_out_bias)
        self.dropout = nn.Dropout(dropout)

        # positional encodings
        if layer_pe is not None:
            layer_pe_args = layer_pe['args']
            self.post_phi_layerPE = layer_pe_args['post_phi_layerPE']
            self.input_dim_layerPE = layer_pe_args['input_dim']
            self.gated = layer_pe_args['gated_layerSPE']
            if self.gated:
                assert layer_pe['type'] in [
                    'spe', 'spe_factorized'], 'Not sure Gating works with anything except spe'
                self.gate_global = nn.Parameter(torch.randn(
                    self.global_heads, self.input_dim_layerPE))
                self.gate_local = nn.Parameter(torch.randn(
                    self.local_heads, self.input_dim_layerPE))
            if layer_pe_args['theta_q']:
                self.to_theta_q = nn.Linear(
                    dim, self.input_dim_layerPE*self.global_heads, bias=qkv_bias)
            else:
                self.to_theta_q = None

            n_global_heads = self.heads - self.local_heads
            if not fast_local_attn:
                n_local_heads = 0
            else:
                n_local_heads = self.local_heads
            self.layer_pos_emb, self.layer_pos_emb_local = \
                get_pes(layer_pe=layer_pe, n_global_heads=n_global_heads, n_local_heads=n_local_heads,
                        dataloader_generator=dataloader_generator)
            self.layer_pe_type = layer_pe['type']
        else:
            self.post_phi_layerPE = True
            self.to_theta_q = None

    def forward(self, x, pos_emb_input=None, context=None, mask=None, context_mask=None,
                **kwargs):
        _, _, _, h, gh = *x.shape, self.heads, self.global_heads

        # cross-attention
        cross_attend = exists(context)
        context = default(context, x)
        context_mask = default(
            context_mask, mask) if not cross_attend else context_mask

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        if self.to_theta_q is not None:
            theta_q = self.to_theta_q(x)
        else:
            theta_q = None
        q, k, v, theta_q = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=h) if t is not None else None,
            (q, k, v, theta_q))
        # split between global and local heads
        (q, lq), (k, lk), (v, lv), (theta_q, ltheta_q) = map(
            lambda t: (t[:, :gh], t[:, gh:]
                       ) if t is not None else (None, None),
            (q, k, v, theta_q))

        attn_outs = []

        if not empty(q):
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]
                v.masked_fill_(~global_mask, 0.)

            if self.post_phi_layerPE:
                if self.features_type == 'favor':
                    q, k = self.feature_map(q, k)
                elif self.features_type == 'elu':
                    q, k = map(lambda t: F.elu(t) + 1, (q, k))

            if pos_emb_input is not None:
                if self.layer_pe_type in ['rototor', 'rototor_fix']:
                    pos_emb_q = self.layer_pos_emb(
                        pe_input=pos_emb_input, offset=theta_q)
                    pos_emb_k = self.layer_pos_emb(
                        pe_input=pos_emb_input, offset=None)
                    q_rot = apply_rototor_pos_emb_(q, pos_emb_q)
                    k_rot = apply_rototor_pos_emb_(k, pos_emb_k)
                elif self.layer_pe_type == 'rotary':
                    pos_emb = self.layer_pos_emb(pe_input=pos_emb_input)
                    q, k = apply_rotary_pos_emb_(q, k, pos_emb)
                    q_rot = None
                    k_rot = None
                # elif self.PE_type in ['spe', 'spe_factorized']:
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
            else:
                q_rot = None
                k_rot = None

            if not self.post_phi_layerPE:
                if self.features_type == 'favor':
                    q, k = self.feature_map(q, k)
                    if q_rot is not None:
                        raise NotImplementedError
                elif self.features_type == 'elu':
                    q, k = map(lambda t: F.elu(t) + 1, (q, k))
                    if q_rot is not None:
                        raise NotImplementedError
            if self.features_type is None:
                _, _, time, dim = q.shape
                qv = torch.einsum('bhid,bhjd->bhij', q, k) * (dim ** -0.5)
                causal_mask = torch.triu(-float('inf') *
                                         torch.ones(time, time), diagonal=1).to(qv.device)
                qv_masked = causal_mask[None, None, :, :] + qv
                attn = qv_masked.softmax(dim=-1)
                attn = self.dropout(attn)
                out = torch.einsum('bhij,bhje->bhie', attn, v)
                state = None
            else:
                out, state = self.fast_attention(
                    q, k, q_rot, k_rot, v, kwargs['states'], kwargs['inferring_states'], horizon=None)
            attn_outs.append(out)

        if not empty(lq):
            assert not cross_attend, 'local attention is not compatible with cross attention'

            if (self.features_type is not None) and self.fast_local_attn:
                if self.post_phi_layerPE:
                    if self.features_type == 'favor':
                        lq, lk = self.feature_map(lq, lk)
                    elif self.features_type == 'elu':
                        lq, lk = map(lambda t: F.elu(t) + 1, (lq, lk))

                if pos_emb_input is not None:
                    if self.layer_pe_type in ['rototor', 'rototor_fix']:
                        lpos_emb_q = self.layer_pos_emb_local(
                            pe_input=pos_emb_input, offset=ltheta_q)
                        lpos_emb_k = self.layer_pos_emb_local(
                            pe_input=pos_emb_input, offset=None)
                        lq_rot = apply_rototor_pos_emb_(lq, lpos_emb_q)
                        lk_rot = apply_rototor_pos_emb_(lk, lpos_emb_k)
                    elif self.layer_pe_type == 'rotary':

                        pos_emb = self.layer_pos_emb_local(pe_input=pos_emb_input)
                        lq, lk = apply_rotary_pos_emb_(lq, lk, pos_emb)
                        lq_rot = None
                        lk_rot = None
                else:
                    lq_rot = None
                    lk_rot = None

                if self.features_type is None:
                    _, _, time, dim = lq.shape
                    lqv = torch.einsum('bhid,bhjd->bhij',
                                       lq, lk) * (dim ** -0.5)
                    causal_mask = torch.triu(-float('inf') *
                                             torch.ones(time, time), diagonal=1).to(lqv.device)
                    lqv_masked = causal_mask[None, None, :, :] + lqv
                    attn = lqv_masked.softmax(dim=-1)
                    attn = self.dropout(attn)
                    out = torch.einsum('bhij,bhje->bhie', attn, lv)
                    state_local = None
                else:
                    out, state_local = self.local_attn(lq, lk, lq_rot, lk_rot, lv,
                                                       kwargs['states'], kwargs['inferring_states'], horizon=256)
            else:
                out = self.local_attn(lq, lk, lv, input_mask=mask)
                state = None
            attn_outs.append(out)

        out = torch.cat(attn_outs, dim=1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        # TODO do something with state_local
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

    def forward(self, q, k, q_rot, k_rot, v, states, inferring_states, horizon):
        """
        inputs are already feature mapped
        """
        if states is not None:
            assert q.size(
                2) == 1, 'recurrent inference can only be applied to sequences of len 1'
            out, states = recursive_attention_step(
                q, k, q_rot, k_rot, v, states)
        else:
            if inferring_states:
                out, states = infer_hidden_states(
                    q, k, q_rot, k_rot, v, horizon)
            else:
                attn_fn = linear_attention if not self.causal else self.causal_linear_fn
                out = attn_fn(q, k, q_rot, k_rot, v, local=horizon)
                states = None
        return out, states


class LocalAttentionLinear(nn.Module):
    def __init__(self):
        super().__init__()
    # TODO hardcoded spans
    def forward(self, q, k, q_rot, k_rot, v, eps=1e-12):
        k_cumsum = k.cumsum(dim=-2)
        k_cumsum[:, :, 256:] = k_cumsum[:, :, 256:] - k_cumsum[:, :, :-256]
        D = torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q))
        if q_rot is not None:
            k_cumsum_rot = k_rot.cumsum(dim=-2)
            k_cumsum_rot[:, :, 256:] = k_cumsum_rot[:,
                                                    :, 256:] - k_cumsum_rot[:, :, :-256]
            D_rot = torch.einsum('...nd,...nd->...n',
                                 q_rot, k_cumsum_rot.type_as(q))
            D = D + D_rot
        D_inv = 1. / (D + eps)

        context = torch.einsum('...nd,...ne->...nde', k, v)
        context_cumsum = context.cumsum(dim=-3)
        context_cumsum[:, :, 256:] = context_cumsum[:,
                                                    :, 256:] - context_cumsum[:, :, :-256]

        out = torch.einsum('...nde,...nd,...n->...ne',
                           context_cumsum, q, D_inv)
        if q_rot is not None:
            context_rot = torch.einsum('...nd,...ne->...nde', k_rot, v)
            context_rot_cumsum = context_rot.cumsum(dim=-3)
            context_rot_cumsum[:, :, :256] = context_rot_cumsum[:, :, 256:] -\
                context_rot_cumsum[:, :, :-256]
            out_rot = torch.einsum('...nde,...nd,...n->...ne',
                                   context_rot_cumsum, q_rot, D_inv)
            out = out + out_rot
        return out


# inefficient causal linear attention, without cuda code,
# (used for parallel inference of hidden states in recurrent mode)
def infer_hidden_states(q, k, q_rot, k_rot, v, chunk_size=128, eps=1e-12):
    last_k_cumsum = 0
    last_context_cumsum = 0
    last_k_cumsum_rot = 0
    last_context_cumsum_rot = 0
    outs = []
    num_chunks = q.size(2) // chunk_size
    for q, k, q_rot, k_rot, v in zip(*map(lambda t: t.chunk(num_chunks, dim=-2), (q, k, q_rot, k_rot, v))):
        k_cumsum = last_k_cumsum + k.cumsum(dim=-2)
        D = torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q))
        context = torch.einsum('...nd,...ne->...nde', k, v)
        context_cumsum = last_context_cumsum + context.cumsum(dim=-3)
        if q_rot is not None:
            for q_rot, k_rot in zip(*map(lambda t: t.chunk(num_chunks, dim=-2), (q_rot, k_rot))):
                k_cumsum_rot = last_k_cumsum_rot + k_rot.cumsum(dim=-2)
                D_rot = torch.einsum('...nd,...nd->...n', q_rot,
                                     k_cumsum_rot.type_as(q_rot))
                context_rot = torch.einsum('...nd,...ne->...nde', k_rot, v)
                context_cumsum_rot = last_context_cumsum_rot + \
                    context_rot.cumsum(dim=-3)
            D_inv = 1. / (D + D_rot + eps)
        else:
            D_inv = 1. / (D + eps)

        out = torch.einsum('...nde,...nd,...n->...ne',
                           context_cumsum, q, D_inv)
        if q_rot is not None:
            out_rot = torch.einsum('...nde,...nd,...n->...ne',
                                   context_cumsum_rot, q_rot, D_inv)
            out = out + out_rot

        last_k_cumsum = k_cumsum[:, :, -1:]
        last_context_cumsum = context_cumsum[:, :, -1:]
        if q_rot is not None:
            last_k_cumsum_rot = k_cumsum_rot[:, :, -1:]
            last_context_cumsum_rot = context_cumsum_rot[:, :, -1:]
        outs.append(out)

    out = torch.cat(outs, dim=-2)
    if q_rot is None:
        last_k_cumsum_rot = None
        last_context_cumsum_rot = None
    states = dict(Z=last_k_cumsum.squeeze(2), S=last_context_cumsum.squeeze(2),
                  Z_rot=last_k_cumsum_rot.squeeze(2), S_rot=last_context_cumsum_rot.squeeze(2))
    return out, states


def recursive_attention_step(q, k, q_rot, k_rot, v, states, eps=1e-12):
    k_cumsum = states['Zs'].unsqueeze(2) + k
    k_cumsum_rot = states['Zs_rot'].unsqueeze(2) + k_rot
    D = torch.einsum('...nd,...nd->...n', q,
                     k_cumsum.type_as(q))
    if q_rot is not None:
        D_rot = torch.einsum('...nd,...nd->...n', q_rot,
                             k_cumsum_rot.type_as(q_rot))
        D_inv = 1. / (D + D_rot + eps)
    else:
        D_inv = 1. / (D + eps)

    context = torch.einsum('...nd,...ne->...nde', k, v)
    context_cumsum = states['Ss'].unsqueeze(2) + context
    if k_rot is not None:
        context_rot = torch.einsum('...nd,...ne->...nde', k_rot, v)
        context_cumsum_rot = states['Ss_rot'].unsqueeze(2) + context_rot

    out = torch.einsum('...nde,...nd,...n->...ne',
                       context_cumsum, q, D_inv)
    if k_rot is not None:
        out_rot = torch.einsum('...nde,...nd,...n->...ne',
                               context_cumsum_rot, q_rot, D_inv)
        out = out_rot + out

    last_k_cumsum = k_cumsum[:, :, 0]
    last_context_cumsum = context_cumsum[:, :, 0]
    if q_rot is not None:
        last_k_cumsum_rot = k_cumsum_rot[:, :, 0]
        last_context_cumsum_rot = context_cumsum_rot[:, :, 0]
    else:
        last_k_cumsum_rot = None
        last_context_cumsum_rot = None
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


def causal_linear_attention(q, k, q_rot, k_rot, v, local=None, eps=1e-12):
    if local is not None:
        # take beginning of k and v
        k_local = k[:, :, :-local]
        v_local = v[:, :, :-local]
        if k_rot is not None:
            k_rot_local = k_rot[:, :, :-local]
        # end of q
        q_local = q[:, :, local:]
        if q_rot is not None:
            q_rot_local = q_rot[:, :, local:]

    if q_rot is None:
        N = get_N(q, k, v)
        D = get_D(q, k)
        if local is not None:
            N_shifted = get_N(q_local, k_local, v_local)
            N[:, :, local:] = N[:, :, local:] - N_shifted
            D_shifted = get_D(q_local, k_local)
            D[:, :, local:] = D[:, :, local:] - D_shifted
        D_inv = 1. / (D + eps)
    else:
        N = (get_N(q, k, v) + get_N(q_rot, k_rot, v))
        D = get_D(q, k) + get_D(q_rot, k_rot)
        if local is not None:
            N_shifted = get_N(q_local, k_local, v_local) + \
                get_N(q_rot_local, k_rot_local, v_local)
            N[:, :, local:] = N[:, :, local:] - N_shifted
            D_shifted = get_D(q_local, k_local) + \
                get_D(q_rot_local, k_rot_local)
            D[:, :, local:] = D[:, :, local:] - D_shifted
        if not torch.all(D > 0):
            raise Exception('D > 0')
        D_inv = 1. / (D + eps)

    # *2 pour être sûr ??
    # N = (2*get_N(q, k, v) + get_N(q_rot, k_rot, v))
    # D_inv = 1. / (2*get_D(q, k) + get_D(q_rot, k_rot) + eps)
    out = torch.einsum('...nd,...n->...nd', N, D_inv)
    if torch.any(torch.isnan(out)):
        raise Exception('NaN in out')
    return out


def get_pes(layer_pe, n_global_heads, n_local_heads, dataloader_generator):
    layer_pe_type = layer_pe['type']
    layer_pe_input = layer_pe['input']
    layer_pe_args = layer_pe['args']
    dim_layerPE = layer_pe_args['input_dim']
    layer_pos_emb_local = None
    if layer_pe_type == 'rototor':
        layer_pos_emb = Rototor(
            dim=dim_layerPE,
            n_heads=n_global_heads,
            fix=False,
            init_type=layer_pe_input)
        if n_local_heads > 0:
            layer_pos_emb_local = Rototor(
                dim=dim_layerPE,
                n_heads=n_local_heads,
                fix=False,
                init_type=layer_pe_input)
    elif layer_pe_type == 'rototor_fix':
        layer_pos_emb = Rototor(
            dim=dim_layerPE,
            n_heads=n_global_heads,
            fix=True,
            init_type=layer_pe_input)
        if n_local_heads > 0:
            layer_pos_emb_local = Rototor(
                dim=dim_layerPE,
                n_heads=n_local_heads,
                fix=True,
                init_type=layer_pe_input)
    elif layer_pe_type == 'rotary':
        layer_pos_emb = Rotary(
            dim=dim_layerPE,
            n_heads=n_global_heads,
            fix=True,
            init_type=layer_pe_input
        )
        if n_local_heads > 0:
            layer_pos_emb = Rotary(
                dim=dim_layerPE,
                n_heads=n_local_heads,
                fix=True,
                init_type=layer_pe_input
            )
    elif layer_pe_type == 'spe':
        if layer_pe_input == 'index':
            num_sines = layer_pe_args['n_sines']
            num_realizations = layer_pe_args['n_realizations']
            num_local_head = n_local_heads
            poscoder = SineSPE(num_heads=n_global_heads, in_features=dim_layerPE,
                               num_sines=num_sines, num_realizations=num_realizations)
            layer_pos_emb = poscoder
            if n_local_heads > 0:
                poscoder_local = SineSPE(num_heads=num_local_head, in_features=dim_layerPE,
                                         num_sines=num_sines, num_realizations=num_realizations)
                layer_pos_emb_local = poscoder_local
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return layer_pos_emb, layer_pos_emb_local
