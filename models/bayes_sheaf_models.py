# Copyright 2022 Twitter, Inc.
# Modifications 2024 Patrick Gillespie
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import numpy as np

from typing import Tuple
from abc import abstractmethod
from torch import nn
from lib import laplace as lap


class SheafLearner(nn.Module):
    """Base model that learns a sheaf from the features and the graph structure."""
    def __init__(self):
        super(SheafLearner, self).__init__()
        self.L = None

    @abstractmethod
    def forward(self, x, edge_index):
        raise NotImplementedError()

    def set_L(self, weights):
        self.L = weights.clone().detach()


class BayesConcatSheafLearner(SheafLearner):
    """Learns a sheaf distribution parameters (mean & variance) by concatenating the local node features
    and passing them through respective linear layers."""

    def __init__(self, in_channels: int, mean_size: int, var_size: int):
        super(BayesConcatSheafLearner, self).__init__()

        self.lin_mean = torch.nn.Linear(in_channels*2, mean_size, bias=False)
        self.lin_var = torch.nn.Linear(in_channels*2, var_size, bias=False)

    def forward(self, x, edge_index):
        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        maps_mean = self.lin_mean(torch.cat([x_row, x_col], dim=1))
        maps_var = self.lin_var(torch.cat([x_row, x_col], dim=1))

        return maps_mean, maps_var


class BayesConcatSheafLearnerVariant(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer."""

    def __init__(self, d: int, hidden_channels: int, mean_size: int, var_size: int):
        super(BayesConcatSheafLearnerVariant, self).__init__()

        self.mean_size = mean_size
        self.var_size = var_size
        self.d = d
        self.hidden_channels = hidden_channels
        self.lin_mean = torch.nn.Linear(hidden_channels * 2, mean_size, bias=False)
        self.lin_var = torch.nn.Linear(hidden_channels * 2, var_size, bias=False)

    def forward(self, x, edge_index):
        row, col = edge_index

        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        x_cat = torch.cat([x_row, x_col], dim=-1)
        x_cat = x_cat.reshape(-1, self.d, self.hidden_channels * 2).sum(dim=1)

        mean_maps = self.lin_mean(x_cat).view(-1, self.mean_size)
        var_maps = self.lin_var(x_cat).view(-1, self.var_size)

        return mean_maps, var_maps


class EdgeWeightLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, in_channels: int, edge_index):
        super(EdgeWeightLearner, self).__init__()
        self.in_channels = in_channels
        self.linear1 = torch.nn.Linear(in_channels*2, 1, bias=False)
        self.full_left_right_idx, _ = lap.compute_left_right_map_index(edge_index, full_matrix=True)

    def forward(self, x, edge_index):
        _, full_right_idx = self.full_left_right_idx

        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        weights = self.linear1(torch.cat([x_row, x_col], dim=1))
        weights = torch.sigmoid(weights)

        edge_weights = weights * torch.index_select(weights, index=full_right_idx, dim=0)
        return edge_weights

    def update_edge_index(self, edge_index):
        self.full_left_right_idx, _ = lap.compute_left_right_map_index(edge_index, full_matrix=True)


