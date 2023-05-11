# Some code taken from tab-transformer-pytorch: https://github.com/lucidrains/tab-transformer-pytorch
# Modified by Jialiang Wang
# Copyright (c) 2023-Current Jialiang Wang <jilwang804@gmail.com>
# License: TBD

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import NumericalEmbedder, MLP
from .transformer import RowColTransformer


class TabularTransformer(nn.Module):
    def __init__(
            self,
            *,
            categories,
            num_continuous,
            dim,
            depth,
            heads,
            dim_head=16,
            dim_out=1,
            mlp_hidden_mults=(4, 2),
            mlp_act=None,
            attn_dropout=0.,
            ff_dropout=0.,
            cont_embeddings='None',
            attention_type='col',
            cls_only=False
    ):
        super(TabularTransformer, self).__init__()
        assert all(map(lambda n: n > 0, categories)), 'Number of each category must be positive.'

        self.dim = dim
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.total_tokens = self.num_unique_categories + 1
        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=1)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)
            self.categorical_embeds = nn.Embedding(self.total_tokens, dim)

            cat_mask_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value=0)
            cat_mask_offset = cat_mask_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('cat_mask_offset', cat_mask_offset)
            self.mask_categorical_embeds = nn.Embedding(self.num_categories * 2, self.dim)

        self.num_continuous = num_continuous
        self.cont_embeddings = cont_embeddings
        if self.num_continuous > 0:
            if self.cont_embeddings == 'MLP':
                self.numerical_embeds = NumericalEmbedder(self.num_continuous, dim)
                con_mask_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value=0)
                con_mask_offset = con_mask_offset.cumsum(dim=-1)[:-1]
                self.register_buffer('con_mask_offset', con_mask_offset)
                self.mask_numerical_embeds = nn.Embedding(self.num_continuous * 2, self.dim)

                input_size = (dim * self.num_categories) + (dim * num_continuous)
                n_feats = self.num_categories + num_continuous
            else:
                self.norm = nn.LayerNorm(self.num_continuous)
                input_size = (dim * self.num_categories) + num_continuous
                n_feats = self.num_categories
        else:
            input_size = dim * self.num_categories
            n_feats = self.num_categories

        self.attention_type = attention_type
        self.transformer = RowColTransformer(
            dim=dim,
            n_feats=n_feats,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            style=attention_type
        )

        self.cls_only = cls_only
        if cls_only:
            if cont_embeddings == 'None':
                print('cont_embeddings=None doesn\'t support cls_only=True.')
                raise NotImplementedError

            self.mlp = nn.Sequential(
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Linear(dim, dim_out)
            )
        else:
            l = input_size // 8
            hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
            all_dimensions = [input_size, *hidden_dimensions, dim_out]
            self.mlp = MLP(all_dimensions, act=mlp_act)

    def forward(self, x_categ, x_numer, x_categ_mask, x_numer_mask):

        xs = []
        device = x_categ.device
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset.type_as(x_categ)
            x_categ = self.categorical_embeds(x_categ)

            x_categ_mask_temp = x_categ_mask + self.cat_mask_offset.type_as(x_categ_mask)
            x_categ_mask_temp = self.mask_categorical_embeds(x_categ_mask_temp)

            x_categ[x_categ_mask == 0] = x_categ_mask_temp[x_categ_mask == 0]
            xs.append(x_categ)

        if self.num_continuous > 0 and self.cont_embeddings == 'MLP':
            x_numer = self.numerical_embeds(x_numer)
            x_numer_mask_temp = x_numer_mask + self.con_mask_offset.type_as(x_numer_mask)
            x_numer_mask_temp = self.mask_numerical_embeds(x_numer_mask_temp)
            x_numer[x_numer_mask == 0] = x_numer_mask_temp[x_numer_mask == 0]
            xs.append(x_numer)

        x = torch.cat(xs, dim=1).to(device)
        latent_x = self.transformer(x)

        if self.num_continuous > 0:
            if self.cls_only:
                latent_x = latent_x[:, 0, :]
            else:
                latent_x = latent_x.flatten(1)

            if self.cont_embeddings == 'None':
                x_numer = self.norm(x_numer)
                latent_x = torch.cat([latent_x, x_numer], dim=-1).to(device)

        logits = self.mlp(latent_x)

        return logits

