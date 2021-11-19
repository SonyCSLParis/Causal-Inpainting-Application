from CIA.model.positional_embeddings.pe_modules.rotary import Rotary
from CIA.model.positional_embeddings.pe_modules.rototor import Rototor
from CIA.model.positional_embeddings.pe_modules.index_spe import SineSPE

from performer_pytorch.performer_pytorch import (
    default,
    empty,
    exists,
    gaussian_orthogonal_random_matrix,
    generalized_kernel,
    softmax_kernel,
)
from functools import partial
from einops import rearrange
import torch.nn as nn
import torch
import torch.nn.functional as F
import math

from CIA.model.positional_embeddings.apply_pe import (
    apply_rotary_pos_emb_,
    apply_rototor_pos_emb_,
)
from CIA.model.attentions.local_attention import LocalAttention_
from CIA.model.attentions.fast_attention import FastAttention_


class DepthwiseConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=3, padding=2, groups=dim
        )

    def forward(self, x):
        batch_size, length, features = x.size()
        x = x.transpose(1, 2)

        x = self.conv(x)
        x = x.transpose(1, 2)
        return x[:, :length]


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
        dropout=0.0,
        no_projection=False,
        qkv_bias=False,
        attn_out_bias=True,
        layer_pe=None,
        dataloader_generator=None,
    ):
        super().__init__()

        # dimensions
        assert input_dim % heads == 0, "dimension must be divisible by number of heads"
        dim_head = input_dim // heads
        if expansion_factor_attn is not None:
            inner_dim = expansion_factor_attn * output_dim
        else:
            inner_dim = dim_head * heads
        self.heads = heads
        self.global_heads = heads - local_heads
        self.local_heads = local_heads

        # features map (softmax kernel if None)
        self.features_type = features["type"]
        if self.features_type is None:
            pass
        elif self.features_type == "favor":
            nb_features = features["args"]["n_features"]
            self.feature_map = FeatureMap(
                dim_head,
                nb_features,
                generalized_attention=generalized_attention,
                kernel_fn=kernel_fn,
                no_projection=no_projection,
            )
        elif self.features_type == "elu":
            pass
        else:
            raise NotImplementedError

        # global attention (if self.feature_type was set to None, softmax kernel is used)
        if self.features_type is not None:
            self.global_attention = FastAttention_(window_size=None)

        # local attention
        self.fast_local_attn = fast_local_attn
        if fast_local_attn:
            self.local_attn = (
                FastAttention_(window_size=local_window_size)
                if local_heads > 0
                else None
            )
        else:
            # rel_pos_emb_config=(dim_head, local_heads)  # LEGACY
            self.local_attn = (
                LocalAttention_(
                    window_size=local_window_size, autopad=True, dropout=dropout
                )
                if local_heads > 0
                else None
            )

        # linear mapping to q, k, v
        self.to_q = nn.Linear(input_dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(input_dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(input_dim, inner_dim, bias=qkv_bias)
        # TODO TEST with depth-wise Conv
        # self.to_q = nn.Sequential(nn.Linear(dim, inner_dim, bias=qkv_bias),
        #                           DepthwiseConv(inner_dim)
        #                           )
        # self.to_k = nn.Sequential(nn.Linear(dim, inner_dim, bias=qkv_bias),
        #                           DepthwiseConv(inner_dim)
        #                           )
        # self.to_v = nn.Sequential(nn.Linear(dim, inner_dim, bias=qkv_bias),
        #                           DepthwiseConv(inner_dim)
        #                           )
        self.to_out = nn.Linear(inner_dim, output_dim, bias=attn_out_bias)
        self.dropout = nn.Dropout(dropout)

        # positional encodings
        if layer_pe is not None:
            layer_pe_args = layer_pe["args"]
            post_phi_layerPE = layer_pe_args["post_phi_layerPE"]
            self.input_dim_layerPE = layer_pe_args["input_dim"]
            self.gated = layer_pe_args["gated_layerSPE"]
            if self.gated:
                assert layer_pe["type"] in [
                    "spe",
                    "spe_factorized",
                ], "Not sure Gating works with anything except spe"
                self.gate_global = nn.Parameter(
                    torch.randn(self.global_heads, self.input_dim_layerPE)
                )
                self.gate_local = nn.Parameter(
                    torch.randn(self.local_heads, self.input_dim_layerPE)
                )
            if layer_pe_args["theta_q"]:
                self.to_theta_q = nn.Linear(
                    input_dim, self.input_dim_layerPE * self.global_heads, bias=qkv_bias
                )
            else:
                self.to_theta_q = None

            n_global_heads = self.heads - self.local_heads
            n_local_heads = self.local_heads
            self.layer_pos_emb, self.layer_pos_emb_local = get_pes(
                layer_pe=layer_pe,
                n_global_heads=n_global_heads,
                n_local_heads=n_local_heads,
                dataloader_generator=dataloader_generator,
            )
            self.layer_pe_type = layer_pe["type"]
        else:
            post_phi_layerPE = True
            self.to_theta_q = None

        # qu'est-ce qu'on calcule ?
        self.compute_features_global = dict()
        self.compute_features_global["before_pe"] = (
            self.features_type is not None
        ) and (post_phi_layerPE)
        self.compute_features_global["after_pe"] = (
            self.features_type is not None
        ) and (not post_phi_layerPE)
        self.compute_features_local = dict()
        self.compute_features_local["before_pe"] = (
            (self.features_type is not None) and fast_local_attn and post_phi_layerPE
        )
        self.compute_features_local["after_pe"] = (
            (self.features_type is not None)
            and fast_local_attn
            and (not post_phi_layerPE)
        )

    def forward(
        self,
        x,
        pos_emb_input=None,
        context=None,
        mask=None,
        context_mask=None,
        **kwargs
    ):
        _, _, _, h, gh = *x.shape, self.heads, self.global_heads

        # cross-attention
        cross_attend = exists(context)
        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        if self.to_theta_q is not None:
            theta_q = self.to_theta_q(x)
        else:
            theta_q = None
        q, k, v, theta_q = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=h)
            if t is not None
            else None,
            (q, k, v, theta_q),
        )
        # split between global and local heads
        (q, lq), (k, lk), (v, lv), (theta_q, ltheta_q) = map(
            lambda t: (t[:, :gh], t[:, gh:]) if t is not None else (None, None),
            (q, k, v, theta_q),
        )

        attn_outs = []

        if not empty(q):
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]
                v.masked_fill_(~global_mask, 0.0)

            if self.compute_features_global["before_pe"]:
                if self.features_type == "favor":
                    q, k = self.feature_map(q, k)
                elif self.features_type == "elu":
                    q, k = map(lambda t: F.elu(t) + 1, (q, k))

            if pos_emb_input is not None:
                if self.layer_pe_type in ["rototor", "rototor_fix"]:
                    pos_emb_q = self.layer_pos_emb(
                        pe_input=pos_emb_input, offset=theta_q
                    )
                    pos_emb_k = self.layer_pos_emb(pe_input=pos_emb_input, offset=None)
                    q_rot = apply_rototor_pos_emb_(q, pos_emb_q)
                    k_rot = apply_rototor_pos_emb_(k, pos_emb_k)
                elif self.layer_pe_type == "rotary":
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

            if self.compute_features_global["after_pe"]:
                if self.features_type == "favor":
                    q, k = self.feature_map(q, k)
                    if q_rot is not None:
                        raise NotImplementedError
                elif self.features_type == "elu":
                    q, k = map(lambda t: F.elu(t) + 1, (q, k))
                    if q_rot is not None:
                        raise NotImplementedError

            if self.features_type is None:
                _, _, time, dim = q.shape
                qv = torch.einsum("bhid,bhjd->bhij", q, k) * (dim ** -0.5)
                causal_mask = torch.triu(
                    -float("inf") * torch.ones(time, time), diagonal=1
                ).to(qv.device)
                qv_masked = causal_mask[None, None, :, :] + qv
                attn = qv_masked.softmax(dim=-1)
                attn = self.dropout(attn)
                out = torch.einsum("bhij,bhje->bhie", attn, v)
                state = None
            else:
                out, state = self.global_attention(
                    q, k, q_rot, k_rot, v, kwargs["states"], kwargs["inferring_states"]
                )
            attn_outs.append(out)
        if not empty(lq):
            if self.compute_features_local["before_pe"]:
                if self.features_type == "favor":
                    lq, lk = self.feature_map(lq, lk)
                elif self.features_type == "elu":
                    lq, lk = map(lambda t: F.elu(t) + 1, (lq, lk))

            if pos_emb_input is not None:
                if self.layer_pe_type in ["rototor", "rototor_fix"]:
                    lpos_emb_q = self.layer_pos_emb_local(
                        pe_input=pos_emb_input, offset=ltheta_q
                    )
                    lpos_emb_k = self.layer_pos_emb_local(
                        pe_input=pos_emb_input, offset=None
                    )
                    lq_rot = apply_rototor_pos_emb_(lq, lpos_emb_q)
                    lk_rot = apply_rototor_pos_emb_(lk, lpos_emb_k)
                elif self.layer_pe_type == "rotary":
                    pos_emb = self.layer_pos_emb_local(pe_input=pos_emb_input)
                    lq, lk = apply_rotary_pos_emb_(lq, lk, pos_emb)
                    lq_rot = None
                    lk_rot = None
            else:
                lq_rot = None
                lk_rot = None

            if self.compute_features_local["after_pe"]:
                if self.features_type == "favor":
                    lq, lk = self.feature_map(lq, lk)
                    if lq_rot is not None:
                        raise NotImplementedError
                elif self.features_type == "elu":
                    lq, lk = map(lambda t: F.elu(t) + 1, (lq, lk))
                    if lq_rot is not None:
                        raise NotImplementedError

            out, state_local = self.local_attn(
                lq, lk, lq_rot, lk_rot, lv, kwargs["states"], kwargs["inferring_states"]
            )

            attn_outs.append(out)

        out = torch.cat(attn_outs, dim=1)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        # TODO do something with state_local
        return self.dropout(out), state


class SelfAttention_(Attention_):
    def forward(self, *args, context=None, **kwargs):
        assert not exists(context), "self attention should not receive context"
        return super().forward(*args, **kwargs)


class CrossAttention_(Attention_):
    def forward(self, *args, context=None, **kwargs):
        assert exists(context), "cross attention should receive context"
        return super().forward(*args, context=context, **kwargs)


class FeatureMap(nn.Module):
    def __init__(
        self,
        dim_heads,
        nb_features=None,
        ortho_scaling=0,
        causal=False,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        no_projection=False,
    ):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(
            gaussian_orthogonal_random_matrix,
            nb_rows=self.nb_features,
            nb_columns=dim_heads,
            scaling=ortho_scaling,
        )
        projection_matrix = self.create_projection()
        self.register_buffer("projection_matrix", projection_matrix)

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
            create_kernel = partial(
                generalized_kernel,
                kernel_fn=self.kernel_fn,
                projection_matrix=self.projection_matrix,
                device=device,
            )
            q, k = map(create_kernel, (q, k))
        else:
            create_kernel = partial(
                softmax_kernel, projection_matrix=self.projection_matrix, device=device
            )
            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)
        return q, k


def get_pes(layer_pe, n_global_heads, n_local_heads, dataloader_generator):
    layer_pe_type = layer_pe["type"]
    layer_pe_input = layer_pe["input"]
    layer_pe_args = layer_pe["args"]
    dim_layerPE = layer_pe_args["input_dim"]
    layer_pos_emb_local = None
    if layer_pe_type == "rototor":
        layer_pos_emb = Rototor(
            dim=dim_layerPE, n_heads=n_global_heads, fix=False, init_type=layer_pe_input
        )
        if n_local_heads > 0:
            layer_pos_emb_local = Rototor(
                dim=dim_layerPE,
                n_heads=n_local_heads,
                fix=False,
                init_type=layer_pe_input,
            )
    elif layer_pe_type == "rototor_fix":
        layer_pos_emb = Rototor(
            dim=dim_layerPE, n_heads=n_global_heads, fix=True, init_type=layer_pe_input
        )
        if n_local_heads > 0:
            layer_pos_emb_local = Rototor(
                dim=dim_layerPE,
                n_heads=n_local_heads,
                fix=True,
                init_type=layer_pe_input,
            )
    elif layer_pe_type == "rotary":
        layer_pos_emb = Rotary(
            dim=dim_layerPE, n_heads=n_global_heads, fix=True, init_type=layer_pe_input
        )
        if n_local_heads > 0:
            layer_pos_emb = Rotary(
                dim=dim_layerPE,
                n_heads=n_local_heads,
                fix=True,
                init_type=layer_pe_input,
            )
    elif layer_pe_type == "spe":
        if layer_pe_input == "index":
            num_sines = layer_pe_args["n_sines"]
            num_realizations = layer_pe_args["n_realizations"]
            num_local_head = n_local_heads
            poscoder = SineSPE(
                num_heads=n_global_heads,
                in_features=dim_layerPE,
                num_sines=num_sines,
                num_realizations=num_realizations,
            )
            layer_pos_emb = poscoder
            if n_local_heads > 0:
                poscoder_local = SineSPE(
                    num_heads=num_local_head,
                    in_features=dim_layerPE,
                    num_sines=num_sines,
                    num_realizations=num_realizations,
                )
                layer_pos_emb_local = poscoder_local
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return layer_pos_emb, layer_pos_emb_local
