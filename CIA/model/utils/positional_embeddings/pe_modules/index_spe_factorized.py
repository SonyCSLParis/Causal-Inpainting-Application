import math
import torch
from torch import nn


class SineSPEFactorized(nn.Module):
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
        num_heads,
        num_realizations,
        num_sines
    ):
        super(SineSPEFactorized, self).__init__()

        # saving dimensions
        self.num_heads = num_heads
        self.num_sines = num_sines
        self.num_realizations = num_realizations

        # register the parameter
        for param in ['freqs', 'offsets', 'gains']:
            self.register_parameter(
                param,
                nn.Parameter(
                    torch.randn(
                        num_heads,
                        num_sines
                    )
                )
            )

        # normalize the gains
        self.gains.data[...] /= torch.sqrt(
            self.gains.norm(dim=-1, keepdim=True)) / 2.

        # bias initial frequencies to low values for long term range
        self.freqs.data[...] -= 4.

        # self.code_shape = (num_heads, in_features)

    def forward(self, pe_input):
        """
        Generate the code, composed of a random QBar and Kbar,
        depending on the parameters, and return them for use with a
        SPE module to actually encode queries and keys.
        Args:
            shape: The outer shape of the inputs: (batchsize, *size)
        """
        batch_size, length = pe_input.shape
        # making sure the frequencies are in [0, 0.5]
        freqs = torch.sigmoid(self.freqs[None, :, None, :])/2.

        phases_q = (
            2 * math.pi
            * freqs * pe_input[:, None, :, None]
            + self.offsets[None, :, None, :]
        )
        omega_q = torch.stack([torch.cos(phases_q), torch.sin(phases_q)], dim=-1).view(
            batch_size, self.num_heads, length, 2*self.num_sines
        )

        phases_k = (
            2 * math.pi
            * freqs * pe_input[:, None, :, None]
        )
        omega_k = torch.stack([torch.cos(phases_k), torch.sin(phases_k)], dim=-1).view(
            batch_size, self.num_heads, length, 2*self.num_sines
        )

        # gains is (num_heads, keys_dim, num_sines). Making then nonnegative with softplus
        gains = nn.functional.softplus(self.gains)

        # now upsample it
        gains = torch.stack(
            (gains, gains), dim=-1).view(
                self.num_heads, 2*self.num_sines)

        # draw noise of appropriate shape on the right device
        z = torch.randn(
            batch_size, self.num_heads, 2 * self.num_sines,
            self.num_realizations,
            device=pe_input.device) / math.sqrt(self.num_sines * 2)

        # scale each of the 2*num_sines by the appropriate gain
        # z is still (1, num_heads, keys_dim, 2*num_sines, num_realizations)
        z = z * gains[None, ..., None]

        # computing the sum over the sines.
        # gets (1, num_heads, keys_dim, length, num_realizations)
        qbar = torch.matmul(omega_q, z)
        kbar = torch.matmul(omega_k, z)

        # permuting them to be (batch, length, num_heads, num_realizations)
        qbar = qbar.permute(0, 2, 1, 3)
        kbar = kbar.permute(0, 2, 1, 3)

        # final scaling
        scale = (self.num_realizations)**0.25
        qbar = torch.reshape(qbar/scale, (batch_size, length, -1))
        kbar = torch.reshape(kbar/scale, (batch_size, length, -1))
        qk_bar = torch.cat((qbar, kbar), -1)
        return qk_bar
