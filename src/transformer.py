# Some code taken from tab-transformer-pytorch: https://github.com/lucidrains/tab-transformer-pytorch
# Some code taken from saint: https://github.com/somepago/saint
# Modified by Jialiang Wang
# Copyright (c) 2023-Current Jialiang Wang <jilwang804@gmail.com>
# License: TBD

import torch
import torch.nn as nn
from einops import rearrange
from .layers import PreNorm, Residual, Attention, FeedForward


class RowColTransformer(nn.Module):
    def __init__(self, dim, n_feats, depth, heads, dim_head, attn_dropout, ff_dropout, style='col'):
        super(RowColTransformer, self).__init__()
        self.layers = nn.ModuleList([])
        self.style = style

        if self.style == 'colrow':
            #layer_structure = [
            #    PreNorm(dim, Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
            #    PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout))),
            #    PreNorm(dim * n_feats, Residual(Attention(dim * n_feats, heads=heads, dim_head=64, dropout=attn_dropout))),
            #    PreNorm(dim * n_feats, Residual(FeedForward(dim * n_feats, dropout=ff_dropout)))
            #]
            layer_structure = [
                PreNorm(dim, Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout)))
            ]
            self.last_layers = nn.ModuleList([nn.ModuleList([
                PreNorm(dim * n_feats, Residual(Attention(dim * n_feats, heads=heads, dim_head=64, dropout=attn_dropout))),
                PreNorm(dim * n_feats, Residual(FeedForward(dim * n_feats, dropout=ff_dropout)))
            ])])
        elif self.style == 'row':
            layer_structure = [
                PreNorm(dim * n_feats, Residual(Attention(dim * n_feats, heads=heads, dim_head=64, dropout=attn_dropout))),
                PreNorm(dim * n_feats, Residual(FeedForward(dim * n_feats, dropout=ff_dropout)))
            ]
        elif self.style == 'col':
            layer_structure = [
                PreNorm(dim, Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout)))
            ]
        else:
            print('Unknown transformer style. Expected styles: [colrow, row, col]')
            raise NotImplementedError
        self.layers = nn.ModuleList([nn.ModuleList(layer_structure) for _ in range(depth)])

    def forward(self, x):
        _, n, _ = x.shape
        if self.style == 'colrow':
            #for attn1, ff1, attn2, ff2 in self.layers:
            #    x = attn1(x)
            #    x = ff1(x)
            #    x = rearrange(x, 'b n d -> 1 b (n d)')
            #    x = attn2(x)
            #    x = ff2(x)
            #    x = rearrange(x, '1 b (n d) -> b n d', n=n)
            for attn1, ff1 in self.layers:
                x = attn1(x)
                x = ff1(x)
            for attn1, ff1 in self.last_layers:
                x = rearrange(x, 'b n d -> 1 b (n d)')
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, '1 b (n d) -> b n d', n=n)
        elif self.style == 'row':
            for attn1, ff1 in self.layers:
                x = rearrange(x, 'b n d -> 1 b (n d)')
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, '1 b (n d) -> b n d', n=n)
        else:
            for attn1, ff1 in self.layers:
                x = attn1(x)
                x = ff1(x)
        return x

