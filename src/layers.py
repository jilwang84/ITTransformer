# Some code taken from tab-transformer-pytorch: https://github.com/lucidrains/tab-transformer-pytorch
# Some code taken from Torch-RecHub: https://github.com/datawhalechina/torch-rechub
# Modified by Jialiang Wang
# Copyright (c) 2023-Current Jialiang Wang <jilwang804@gmail.com>
# License: TBD

import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange


# Classical MLP
class MLP(nn.Module):

    def __init__(self, dims, act=None):
        super(MLP, self).__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            #is_last = ind >= (len(dims) - 1)
            is_last = ind + 1 >= (len(dims) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue
            if act == 'DLLP':
                layers.append(nn.BatchNorm1d(dim_out))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.0))
            elif act is not None:
                layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# Two layers MLP
class Simple_MLP(nn.Module):

    def __init__(self, dims):
        super(Simple_MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2])
        )

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


# Feature-wise MLP
class Sep_MLP(nn.Module):

    def __init__(self, dim, len_feats, categories):
        super(Sep_MLP, self).__init__()
        self.len_feats = len_feats
        self.layers = nn.ModuleList([])
        for i in range(len_feats):
            self.layers.append(Simple_MLP([dim, 5 * dim, categories[i]]))

    def forward(self, x):
        y_pred = list([])
        for i in range(self.len_feats):
            x_i = x[:, i, :]
            pred = self.layers[i](x_i)
            y_pred.append(pred)
        return y_pred


# Residual module: skip connection
class Residual(nn.Module):

    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


# Layer normalization module
class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# GEGLU: a GLU variant https://arxiv.org/pdf/2002.05202.pdf
class GEGLU(nn.Module):

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


# Fully-connected feed-forward layers module
class FeedForward(nn.Module):

    def __init__(self, dim, mult=4, dropout=0.):
        super(FeedForward, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.layers(x)


# Attention module
class Attention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        dropped_attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


class NumericalEmbedder(nn.Module):

    def __init__(self, num_numerical_types, dim):
        super(NumericalEmbedder, self).__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases

