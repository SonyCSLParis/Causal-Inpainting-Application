from local_attention.local_attention import TOKEN_SELF_ATTN_VALUE, expand_dim, look_around, max_neg_value, merge_dims, pad_to_multiple
from performer_pytorch.performer_pytorch import default
from torch import nn
import torch
import torch.nn.functional as F


class LocalAttention_(nn.Module):
    def __init__(
        self,
        window_size,
        dropout=0.,
        autopad=False,
        exact_windowsize=False
    ):
        super().__init__()
        self.window_size = window_size
        self.look_backward = 1
        self.look_forward = 0
        self.exact_windowsize = exact_windowsize
        self.autopad = autopad
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, q_rot, k_rot, v, states, inferring_states):
        shape = q.shape

        merge_into_batch = lambda t: t.reshape(-1, *t.shape[-2:]) if t is not None else None
        q, k, q_rot, k_rot, v = map(merge_into_batch, (q, k, q_rot, k_rot, v))

        if self.autopad:
            orig_t = q.shape[1]
            q, k, q_rot, k_rot, v = map(lambda t: pad_to_multiple(t, self.window_size, dim=-2) if t is not None else None, (q, k, q_rot, k_rot, v))

        window_size, look_backward, look_forward = self.window_size, self.look_backward, self.look_forward
        b, t, e, device, dtype = *q.shape, q.device, q.dtype
        assert (t % window_size) == 0, f'sequence length {t} must be divisible by window size {window_size} for local attention'

        windows = t // window_size

        ticker = torch.arange(t, device=device, dtype=dtype)[None, :]
        b_t = ticker.reshape(1, windows, window_size)

        bucket_fn = lambda t: t.reshape(b, windows, window_size, -1) if t is not None else None
        bq, bk, bq_rot, bk_rot, bv = map(bucket_fn, (q, k, q_rot, k_rot, v))

        look_around_kwargs = {'backward': look_backward, 'forward': look_forward}
        bk = look_around(bk, **look_around_kwargs)
        if bk_rot is not None:
            bk_rot = look_around(bk_rot, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)

        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)

        dots = torch.einsum('bhie,bhje->bhij', bq, bk) * (e ** -0.5)
        if q_rot is not None:
            dots_rot = torch.einsum('bhie,bhje->bhij', bq_rot, bk_rot) * (e ** -0.5)

        mask_value = max_neg_value(dots)
        if q_rot is not None:
            mask_value_rot = max_neg_value(dots_rot)

        mask = bq_t[:, :, :, None] < bq_k[:, :, None, :]
        if self.exact_windowsize:
            max_causal_window_size = (self.window_size * self.look_backward)
            mask = mask | (bq_t[:, :, :, None] > (bq_k[:, :, None, :] + max_causal_window_size))
        dots.masked_fill_(mask, mask_value)
        if q_rot is not None:
            dots_rot.masked_fill_(mask, mask_value_rot)
        del mask

        mask = bq_k[:, :, None, :] == -1
        dots.masked_fill_(mask, mask_value)
        if q_rot is not None:
            dots_rot.masked_fill_(mask, mask_value_rot)
        del mask

        if q_rot is None:
            attn = dots.softmax(dim=-1)
        else:
            attn = (dots + dots_rot).softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhij,bhje->bhie', attn, bv)
        out = out.reshape(-1, t, e)

        if self.autopad:
            out = out[:, :orig_t, :]

        return out.reshape(*shape), None
