from functools import partial
from local_attention.local_attention import default
from performer_pytorch.performer_pytorch import gaussian_orthogonal_random_matrix
import torch
from torch import nn
import math
from performer_pytorch.performer_pytorch import causal_linear_attention, causal_linear_attention_noncuda, default,\
    gaussian_orthogonal_random_matrix, generalized_kernel, linear_attention, softmax_kernel


class FastLocalAttention_(nn.Module):
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

        self.causal = causal
        if causal:
            try:
                import fast_transformers.causal_product.causal_product_cuda
                self.causal_linear_fn = partial(causal_linear_attention)
            except ImportError:
                print('unable to import cuda code for auto-regressive Performer. will default to the memory inefficient non-cuda version')
                self.causal_linear_fn = causal_linear_attention_noncuda

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v, states, inferrring_states):
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
