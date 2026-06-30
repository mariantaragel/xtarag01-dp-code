"""
Created by Marián Tarageľ (xtarag01)
"""

import math

import torch
import torch.nn as nn

from .mlp import MLP


class FoldingNet(nn.Module):
    def __init__(self, latent_dim=1024, num_points=2048) -> None:
        super(FoldingNet, self).__init__()

        self.grid_dim = int(math.sqrt(num_points))
        self.actual_points = self.grid_dim**2

        self.fold_1 = MLP(
            in_feat_dims=latent_dim + 2,
            out_channels=[512, 512, 3],
            b_norm=False,
        )
        self.fold_2 = MLP(
            in_feat_dims=latent_dim + 3,
            out_channels=[512, 512, 3],
            b_norm=False,
        )

        range_x = torch.linspace(-1.0, 1.0, self.grid_dim)
        range_y = torch.linspace(-1.0, 1.0, self.grid_dim)
        x_coor, y_coor = torch.meshgrid(range_x, range_y, indexing="ij")
        self.register_buffer(
            "grid", torch.stack([x_coor, y_coor], axis=-1).float().reshape(-1, 2)
        )

    def __call__(self, latent):
        batch_size = latent.shape[0]
        grid = self.grid.unsqueeze(0).expand(batch_size, -1, -1)
        latent_exp = latent.unsqueeze(1).expand(-1, self.actual_points, -1)

        # 1st folding
        in_1 = torch.cat((grid, latent_exp), 2)
        out_1 = self.fold_1(in_1)

        # 2nd folding
        in_2 = torch.cat((out_1, latent_exp), 2)
        out_2 = self.fold_2(in_2)

        return out_2
