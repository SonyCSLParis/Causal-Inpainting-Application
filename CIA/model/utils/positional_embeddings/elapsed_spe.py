import math
import torch
from torch import nn


class ElapsedSineSPE(nn.Module):
    """
    code generator for sinusoidal stochastic positional encoding.
    Args:
        num_heads: The number of attention heads.
        in_features: The number of input features per attention head.
        num_realizations: The number of realizations of the stochastic
            process (R).
        num_sines: The number of sin and cos components (K).
    """

    def __init__(
        self,
        num_heads: int = 8,
        in_features: int = 64,
        num_realizations: int = 256,
        num_sines: int = 1
    ):
        super(ElapsedSineSPE, self).__init__()

        # saving dimensions
        self.num_heads = num_heads
        self.in_features = in_features
        self.num_sines = num_sines
        self.num_realizations = num_realizations

        # register the parameter
        for param in ['freqs', 'offsets', 'gains']:
            self.register_parameter(
                param,
                nn.Parameter(
                    torch.randn(
                        num_heads,
                        in_features,
                        num_sines
                    )
                )
            )

        # normalize the gains
        self.gains.data[...] /= torch.sqrt(
            self.gains.norm(dim=-1, keepdim=True)) / 2.

        # bias initial frequencies to low values for long term range
        self.freqs.data[...] -= 4.

        self.code_shape = (num_heads, in_features)

    def forward(self, x_embed, h, metadata_dict):
        """
        Generate the code, composed of a random QBar and Kbar,
        depending on the parameters, and return them for use with a
        SPE module to actually encode queries and keys.
        Args:
            shape: The outer shape of the inputs: (batchsize, *size)
        """
        # get shape of the queries. Here it's only 1d
        max_len = x_embed.size(1)

        # build omega_q and omega_k,
        # with shape (num_heads, keys_dim, length, 2*num_sines)
        TODO ICI REMPLACER INDICES PAR LES ELAPSED TIME
        indices = torch.linspace(
            0, max_len-1, max_len, device=self.freqs.device)

        # making sure the frequencies are in [0, 0.5]
        freqs = torch.sigmoid(self.freqs[:, :, None, :])/2.

        phases_q = (
            2 * math.pi
            * freqs * indices[None, None, :, None]
            + self.offsets[:, :, None, :]
        )
        omega_q = torch.stack([torch.cos(phases_q), torch.sin(phases_q)], dim=-1).view(
            1, self.num_heads, self.in_features, max_len, 2*self.num_sines
        )

        phases_k = (
            2 * math.pi
            * freqs * indices[None, None, :, None]
        )
        omega_k = torch.stack([torch.cos(phases_k), torch.sin(phases_k)], dim=-1).view(
            1, self.num_heads, self.in_features, max_len, 2*self.num_sines
        )

        # gains is (num_heads, keys_dim, num_sines). Making then nonnegative with softplus
        gains = nn.functional.softplus(self.gains)

        # now upsample it
        gains = torch.stack(
            (gains, gains), dim=-1).view(
                self.num_heads, self.in_features, 2*self.num_sines)

        # draw noise of appropriate shape on the right device
        z = torch.randn(
            1, self.num_heads, self.in_features, 2 * self.num_sines,
            self.num_realizations,
            device=self.freqs.device) / math.sqrt(self.num_sines * 2)

        # scale each of the 2*num_sines by the appropriate gain
        # z is still (1, num_heads, keys_dim, 2*num_sines, num_realizations)
        z = z * gains[None, ..., None]

        # computing the sum over the sines.
        # gets (1, num_heads, keys_dim, length, num_realizations)
        qbar = torch.matmul(omega_q, z)
        kbar = torch.matmul(omega_k, z)

        # permuting them to be (1, length, num_heads, keys_dim, num_realizations)
        qbar = qbar.permute(0, 3, 1, 2, 4)
        kbar = kbar.permute(0, 3, 1, 2, 4)

        # final scaling
        scale = (self.num_realizations * self.in_features)**0.25
        return (qbar/scale, kbar/scale)

    def get_posattn_matrix(self, max_len=2048):
        indices = torch.linspace(
            0, max_len-1, max_len, device=self.freqs.device)

        # making sure the frequencies are in [0, 0.5]
        freqs = torch.sigmoid(self.freqs[:, :, None, :])/2.

        phases_q = (
            2 * math.pi
            * freqs * indices[None, None, :, None]
            + self.offsets[:, :, None, :]
        )
        omega_q = torch.stack([torch.cos(phases_q), torch.sin(phases_q)], dim=-1).view(
            1, self.num_heads, self.in_features, max_len, 2*self.num_sines
        )

        phases_k = (
            2 * math.pi
            * freqs * indices[None, None, :, None]
        )
        omega_k = torch.stack([torch.cos(phases_k), torch.sin(phases_k)], dim=-1).view(
            1, self.num_heads, self.in_features, max_len, 2*self.num_sines
        )

        # gains is (num_heads, keys_dim, 2*num_sines). Making then nonnegative with softplus
        gains = nn.functional.softplus(self.gains)
        # gains = gains / torch.sqrt(gains.norm(dim=-1, keepdim=True))
        gains = torch.stack(
            (gains, gains), dim=-1).view(
                self.num_heads, self.in_features, 2*self.num_sines)

        gains_squared_diag = torch.diag_embed(gains ** 2)

        print('[get posattn matrix] Omega_q: {}, lambda: {}, Omega_k: {}'.format(
            omega_q.size(), gains_squared_diag.size(), omega_k.size()
        ))
        # print (gains_squared_diag[0, 0])

        # get (1, num_heads, keys_dim) attention maps, each of size (max_len, max_len)
        omega_q_mult_gains_squared_diag = torch.einsum(
            'ihdmk, hdku -> ihdmu',
            omega_q, gains_squared_diag
        )
        pos_attn_matrices = torch.einsum(
            'ihdmk, ihdnk -> ihdmn',
            omega_q_mult_gains_squared_diag, omega_k
        )
        print('[get posattn matrix] pos_attn: {}'.format(
            pos_attn_matrices.size()
        ))

        return pos_attn_matrices
