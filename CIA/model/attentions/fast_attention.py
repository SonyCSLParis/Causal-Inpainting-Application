from functools import partial
import torch
import torch.nn as nn
from performer_pytorch.performer_pytorch import null_context
from torch.cuda.amp import autocast
try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False


class FastAttention_(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size

    def forward(self, q, k, q_rot, k_rot, v, states, inferring_states):
        """
        inputs are already feature mapped
        """
        if states is not None:
            assert q.size(
                2
            ) == 1, 'recurrent inference can only be applied to sequences of len 1'
            out, states = recursive_attention_step(q, k, q_rot, k_rot, v,
                                                   states)
        else:
            if inferring_states:
                # TODO(leo): horizon not taken into account!!!!
                # Pb with default parameters!!!
                raise NotImplementedError
                out, states = infer_hidden_states(q, k, q_rot, k_rot, v)
            else:
                out = causal_linear_attention(q, k, q_rot, k_rot, v, local=self.window_size)
                states = None
        return out, states


def causal_linear_attention(q, k, q_rot, k_rot, v, local=None, eps=1e-6):
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



# inefficient causal linear attention, without cuda code,
# (used for parallel inference of hidden states in recurrent mode)
def infer_hidden_states(q, k, q_rot, k_rot, v, chunk_size=128, eps=1e-6):
    last_k_cumsum = 0
    last_context_cumsum = 0
    last_k_cumsum_rot = 0
    last_context_cumsum_rot = 0
    outs = []
    num_chunks = q.size(2) // chunk_size
    for q, k, q_rot, k_rot, v in zip(
            *map(lambda t: t.chunk(num_chunks, dim=-2), (q, k, q_rot, k_rot,
                                                         v))):
        k_cumsum = last_k_cumsum + k.cumsum(dim=-2)
        D = torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q))
        context = torch.einsum('...nd,...ne->...nde', k, v)
        context_cumsum = last_context_cumsum + context.cumsum(dim=-3)
        if q_rot is not None:
            for q_rot, k_rot in zip(
                    *map(lambda t: t.chunk(num_chunks, dim=-2), (q_rot,
                                                                 k_rot))):
                k_cumsum_rot = last_k_cumsum_rot + k_rot.cumsum(dim=-2)
                D_rot = torch.einsum('...nd,...nd->...n', q_rot,
                                     k_cumsum_rot.type_as(q_rot))
                context_rot = torch.einsum('...nd,...ne->...nde', k_rot, v)
                context_cumsum_rot = last_context_cumsum_rot + \
                    context_rot.cumsum(dim=-3)
            D_inv = 1. / (D + D_rot + eps)
        else:
            D_inv = 1. / (D + eps)

        out = torch.einsum('...nde,...nd,...n->...ne', context_cumsum, q,
                           D_inv)
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
    states = dict(Z=last_k_cumsum.squeeze(2),
                  S=last_context_cumsum.squeeze(2),
                  Z_rot=last_k_cumsum_rot.squeeze(2),
                  S_rot=last_context_cumsum_rot.squeeze(2))
    return out, states

def recursive_attention_step(q, k, q_rot, k_rot, v, states, eps=1e-12):
    k_cumsum = states['Zs'].unsqueeze(2) + k
    k_cumsum_rot = states['Zs_rot'].unsqueeze(2) + k_rot
    D = torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q))
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

    out = torch.einsum('...nde,...nd,...n->...ne', context_cumsum, q, D_inv)
    if k_rot is not None:
        out_rot = torch.einsum('...nde,...nd,...n->...ne', context_cumsum_rot,
                               q_rot, D_inv)
        out = out_rot + out

    last_k_cumsum = k_cumsum[:, :, 0]
    last_context_cumsum = context_cumsum[:, :, 0]
    if q_rot is not None:
        last_k_cumsum_rot = k_cumsum_rot[:, :, 0]
        last_context_cumsum_rot = context_cumsum_rot[:, :, 0]
    else:
        last_k_cumsum_rot = None
        last_context_cumsum_rot = None
    states = dict(Z=last_k_cumsum,
                  S=last_context_cumsum,
                  Z_rot=last_k_cumsum_rot,
                  S_rot=last_context_cumsum_rot)
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
            q, k, v = map(lambda t: t.float(), (q, k, v))
        N = causal_dot_product_fn(q, k, v)
    return N
