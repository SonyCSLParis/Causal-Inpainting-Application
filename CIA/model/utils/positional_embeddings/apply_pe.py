import math
from einops import rearrange, repeat
import torch


def rotate_every_two_(x):
    x = rearrange(x, '... (d j) -> ... d j', j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d j -> ... (d j)')


def apply_rototor_pos_emb_(x, pos_emb):
    cos, sin = pos_emb.unbind(dim=-1)
    cos_emb, sin_emb = map(lambda t: t * x, (cos, sin))
    b, h, l, d = sin_emb.shape
    # x_embed represents rotated x grouped by 2. Hence dim(x_embed) = 2 * dim(x)
    x_embed = torch.stack((cos_emb, sin_emb), dim=-1).view(b, h, l, d * 2)
    return x_embed


# def apply_rotary_pos_emb_(q, k, sinu_pos):
#     sin, cos = sinu_pos.unbind(dim=-1)
#     sin_heads, cos_heads = map(lambda t: t.unsqueeze(
#         1), (sin, cos))  # unsqueeze for head dim
#     # use the first angle with theta = 0 to normalise
#     d = sin.size(-1)
#     sum_sqrt_lambda = math.sqrt(d-1)
#     cos_heads[:, :, :, -1] = cos_heads[:, :, :, -1] * sum_sqrt_lambda
#     # sin_heads, cos_heads: (b, h, l, n_sines)
#     # q, k: (b, h, l, r)
#     q_cos, k_cos = map(lambda t: (t[:, :, :, :, None] * cos_heads[:, :, :, None, :]), (q, k))
#     q_sin, k_sin = map(lambda t: (t[:, :, :, :, None] * sin_heads[:, :, :, None, :]), (q, k))
#     q_rot = torch.stack([q_cos, q_sin], dim=-1)
#     k_rot = torch.stack([k_cos, k_sin], dim=-1)
#     return q_rot, k_rot

def apply_rotary_pos_emb_(q, k, sinu_pos):
    sin, cos = sinu_pos.unbind(dim=-1)
    sin, cos = map(lambda t: repeat(t, 'b h t n -> b h t (n j)', j=2), (sin, cos))
    q, k = map(lambda t: (t * cos) + (rotate_every_two_(t) * sin), (q, k))
    return q, k


def apply_rotary_pos_emb_upsampled(q, k, sinu_pos):
    sinu_pos = rearrange(sinu_pos, 'b n (j d) -> b n j d', j=2)
    sin, cos = sinu_pos.unbind(dim=-2)
    sin, cos = map(lambda t: repeat(t, 'b t n -> b t (n j)', j=2), (sin, cos))
    # TODO: use same positional embeddings for all heads ?? perhaps can be changed when parametrising thetas
    sin_heads, cos_heads = map(lambda t: t.unsqueeze(
        1), (sin, cos))  # unsqueeze for head dim
    # upsample q and k
    q_up, k_up = map(lambda t: rearrange(
        [t, torch.zeros_like(t)], 't b h l d -> b h l (d t)'), (q, k))
    q_rot, k_rot = map(lambda t: (t * cos_heads) +
                       (rotate_every_two_(t) * sin_heads), (q_up, k_up))
    return q_rot, k_rot


def apply_spe_pos_emb_factorised(queries, keys, qbar, kbar, code_shape, gate):
    # apply gate if required
    if gate is not None:
        # incorporate the constant bias for Pd if required. First draw noise
        # such that noise noise^T = 1, for each head, feature, realization.
        # qbar is : (1, *shape, num_heads, keys_dim, num_realizations)
        in_features = qbar.shape[-2]
        num_realizations = qbar.shape[-1]
        gating_noise = torch.randn(
            code_shape+(num_realizations,),
            device=queries.device) / (in_features * num_realizations)**0.25
        # normalize it so that it's an additive 1 to Pd
        # gating_noise = gating_noise / gating_noise.norm(dim=2, keepdim=True)

        # constrain the gate parameter to be in [0 1]
        gate = torch.sigmoid(gate[..., None])

        # qbar is (1, *shape, num_heads, keys_dim, num_realizations)
        # gating noise is (num_heads, keys_dim, num_realizations)
        # gate is (num_heads, keys_dim, 1)
        # import ipdb; ipdb.set_trace()
        qbar = torch.sqrt(1.-gate) * qbar + torch.sqrt(gate) * gating_noise
        kbar = torch.sqrt(1.-gate) * kbar + torch.sqrt(gate) * gating_noise

    # sum over d after multiplying by queries and keys
    # qbar/kbar are (1, *shape, num_heads, keys_dim, num_realizations)
    # queries/keys  (batchsize, *shape, num_heads, keys_dim)
    qhat = (qbar[:, :, :, None, :] * queries[..., None])
    khat = (kbar[:, :, :, None, :] * keys[..., None])
    # qhat = qhat.sum(axis=-2)
    # khat = khat.sum(axis=-2)

    # result is (batchsize, ..., num_heads, n_features, n_realisations)
    return qhat, khat


def apply_spe_pos_emb_(queries, keys, qbar, kbar, code_shape, gate):
    # check that codes have the shape we are expecting
    if qbar.shape[-3:-1] != code_shape:
        raise ValueError(
            f'The inner shape of codes is {qbar.shape[-3:-1]}, '
            f'but expected {code_shape}')

    # check shapes: size of codes should be bigger than queries, keys
    code_size = qbar.shape[1:-3]
    query_size = queries.shape[1:-2]
    if (len(code_size) != len(query_size)
        or torch.any(
            torch.tensor(code_size) < torch.tensor(query_size)
    )):
        raise ValueError(f'Keys/queries have length {query_size}, '
                         f'but expected at most {code_size}')
    if qbar.shape[-3:-1] != queries.shape[-2:]:
        raise ValueError(f'shape mismatch. codes have shape {qbar.shape}, '
                         f'but queries are {queries.shape}')

    # truncate qbar and kbar for matching current queries and keys,
    # but only if we need to
    for dim in range(len(query_size)):
        if code_size[dim] > query_size[dim]:
            indices = [slice(1), *[slice(qbar.shape[1+k]) for k in range(dim)],
                       slice(query_size[dim])]
            qbar = qbar[indices]
            kbar = kbar[indices]

    # apply gate if required
    if gate is not None:
        # incorporate the constant bias for Pd if required. First draw noise
        # such that noise noise^T = 1, for each head, feature, realization.
        # qbar is : (1, *shape, num_heads, keys_dim, num_realizations)
        in_features = qbar.shape[-2]
        num_realizations = qbar.shape[-1]
        gating_noise = torch.randn(
            code_shape+(num_realizations,),
            device=queries.device) / (in_features * num_realizations)**0.25
        # normalize it so that it's an additive 1 to Pd
        # gating_noise = gating_noise / gating_noise.norm(dim=2, keepdim=True)

        # constrain the gate parameter to be in [0 1]
        gate = torch.sigmoid(gate[..., None])

        # qbar is (1, *shape, num_heads, keys_dim, num_realizations)
        # gating noise is (num_heads, keys_dim, num_realizations)
        # gate is (num_heads, keys_dim, 1)
        # import ipdb; ipdb.set_trace()
        qbar = torch.sqrt(1.-gate) * qbar + torch.sqrt(gate) * gating_noise
        kbar = torch.sqrt(1.-gate) * kbar + torch.sqrt(gate) * gating_noise

    # sum over d after multiplying by queries and keys
    # qbar/kbar are (1, *shape, num_heads, keys_dim, num_realizations)
    # queries/keys  (batchsize, *shape, num_heads, keys_dim)
    qhat = (qbar * queries[..., None]).sum(axis=-2)
    khat = (kbar * keys[..., None]).sum(axis=-2)

    # result is (batchsize, ..., num_heads, num_realizations)
    return qhat, khat
