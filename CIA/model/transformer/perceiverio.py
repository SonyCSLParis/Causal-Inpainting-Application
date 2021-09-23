import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange


        
class Encode(nn.Module):
    def __init__(self, dim,
                 num_heads,
                 downscaling,
                 hidden_dim):
        super().__init__()
        self.dim = dim
        self.downscaling = downscaling
        
        assert dim % num_heads == 0
        self.dim_heads = dim // num_heads
        # Bias?!        
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
               
        self.num_heads = num_heads
        
        
        
    def forward(self, x, l):
        """[summary]

        Args:
            x (M, C): [description]
            l (N, ): [description]
        """        
        
        dim = self.dim_heads
        num_latents = l.size(1)
        num_tokens = x.size(1)
        
        assert num_tokens == num_latents * self.downscaling
                
        q = self.to_q(l)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        
        qk = torch.einsum('bhid,bhjd->bhij', q, k) * (dim**-0.5)
        causal_mask = torch.triu(-float('inf') *
                                         torch.ones(num_latents, num_latents),
                                         diagonal=1).to(qk.device)
        
        causal_mask = causal_mask.repeat_interleave(self.downscaling, dim=1)
        qv_masked = causal_mask[None, None, :, :] + qk
        
        # attn = qv_masked.softmax(dim=-2)
        attn = qv_masked.softmax(dim=-1)
        
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out        
        
class Process(nn.Module):
    def __init__(self, dim,
                 num_heads,
                 hidden_dim):
        super().__init__()
        self.self_attn = Encode(dim=dim, num_heads=num_heads,downscaling=1, hidden_dim=hidden_dim)
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )        
        
    def forward(self, l):        
        out = l
        out = self.norm(out)
        out = self.self_attn(out, out)
        l = l + out
        
        out = self.norm2(l)
        out = self.mlp(out)
        out = l + out
        return out
        


class PerceiverIO(nn.Module):
    def __init__(self, dim, num_layers, **kwargs):
        super().__init__()
        self.dim_last_layer = dim
        self.downscaling = 16
        
        self.l_init = nn.Parameter(torch.randn(1, 64, 512))
        self.encode = Encode(dim=512,
                             num_heads=8,
                             downscaling=self.downscaling,
                             hidden_dim=512)
        self.process = nn.Sequential(
            *[
            Process(dim, num_heads=8, hidden_dim=512)
            for _ in range(num_layers)
        ])
        
        
        # small transformer
        self.post_process = nn.Sequential(
            *[
            Process(dim, num_heads=8, hidden_dim=512)
            for _ in range(4)
        ])
        
        self.dummy_latent = nn.Parameter(torch.randn(1, 1, 512))
        
    def forward(self, x, **kwargs):
        batch_size, num_tokens, feature_dim = x.size()
        _, num_latents, latent_dim = self.l_init.size()
        
        # intialise the tower of latents
        l = self.l_init.expand(batch_size, num_latents, latent_dim)
        l = self.encode(x, l)
        
        l = self.process(l)
                
        # offset on l        
        l = torch.cat([self.dummy_latent.repeat(batch_size, 1, 1)
                       , l],
                      dim=1)[:, :-1]        
        l = l.reshape(batch_size * num_latents, 1, latent_dim)
        
        x = x.reshape(batch_size *
                      num_tokens // self.downscaling,
                      self.downscaling, feature_dim)
        y = torch.cat([l, x], dim=1)
        
        y = self.post_process(y)
        y = y[:, 1:]
        y = y.reshape(batch_size, num_tokens, latent_dim)
        return dict(x=y)
        
        
        