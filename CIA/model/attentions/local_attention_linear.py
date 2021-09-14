import torch
import torch.nn as nn


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
            k_cumsum_rot[:, :,
                         256:] = k_cumsum_rot[:, :,
                                              256:] - k_cumsum_rot[:, :, :-256]
            D_rot = torch.einsum('...nd,...nd->...n', q_rot,
                                 k_cumsum_rot.type_as(q))
            D = D + D_rot
        D_inv = 1. / (D + eps)

        context = torch.einsum('...nd,...ne->...nde', k, v)
        context_cumsum = context.cumsum(dim=-3)
        context_cumsum[:, :,
                       256:] = context_cumsum[:, :,
                                              256:] - context_cumsum[:, :, :
                                                                     -256]

        out = torch.einsum('...nde,...nd,...n->...ne', context_cumsum, q,
                           D_inv)
        if q_rot is not None:
            context_rot = torch.einsum('...nd,...ne->...nde', k_rot, v)
            context_rot_cumsum = context_rot.cumsum(dim=-3)
            context_rot_cumsum[:, :, :256] = context_rot_cumsum[:, :, 256:] -\
                context_rot_cumsum[:, :, :-256]
            out_rot = torch.einsum('...nde,...nd,...n->...ne',
                                   context_rot_cumsum, q_rot, D_inv)
            out = out + out_rot
        return out
