# Copyright 2022 Twitter, Inc.
# Modifications 2024 Patrick Gillespie
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import torch_sparse

from torch import nn
from torch_geometric.utils import degree
from torch_geometric.nn.conv import GCNConv
from models.sheaf_base import SheafDiffusion
from models import bayes_laplacian_builders as lb
from models.bayes_sheaf_models import BayesConcatSheafLearner, BayesConcatSheafLearnerVariant, EdgeWeightLearner
from models.bayes_kl import KLDivergence


class BayesDiagSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(BayesDiagSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 0

        self.kldiv = KLDivergence('normal')

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        if self.sparse_learner:
            self.sheaf_learner = BayesConcatSheafLearnerVariant(
                self.final_d, self.hidden_channels, mean_size=self.d, var_size=self.d)
        else:
            self.sheaf_learner = BayesConcatSheafLearner(
                self.hidden_dim, mean_size=self.d, var_size=self.d)
        self.laplacian_builder = lb.BayesDiagLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, normalised=self.normalised,
            deg_normalised=self.deg_normalised, add_hp=self.add_hp, add_lp=self.add_lp)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x_maps = x
        mean_params, log_var_params = self.sheaf_learner(x_maps.reshape(self.graph_size, -1), self.edge_index)
        mean_params = torch.tanh(mean_params)
        log_var_params = -F.elu(-log_var_params)
        std_params = torch.exp(log_var_params)

        kl = self.kldiv(mean_params, log_var_params)

        Ls = []
        for layer in range(self.layers):
            L, trans_maps = self.laplacian_builder(mean_params, std_params)
            Ls.append(L)

        x0 = x
        for layer in range(self.layers):
            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[layer](x)

            x = torch_sparse.spmm(Ls[layer][0], Ls[layer][1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))
            x0 = coeff * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1), kl


class BayesBundleSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(BayesBundleSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 1
        assert not self.deg_normalised

        self.kldiv = KLDivergence('cayley', self.d)

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        if self.sparse_learner:
            self.sheaf_learner = BayesConcatSheafLearnerVariant(
                self.final_d, self.hidden_channels, mean_size=self.get_param_size(), var_size=1)
        else:
            self.sheaf_learner = BayesConcatSheafLearner(
                self.hidden_dim, mean_size=self.get_param_size(), var_size=1)
            
        if self.use_edge_weights:
            self.weight_learner = EdgeWeightLearner(self.hidden_dim, edge_index)

        self.laplacian_builder = lb.BayesNormConnectionLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_hp=self.add_hp,
            add_lp=self.add_lp, orth_map=self.orth_trans)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        maps_input_dim = self.input_dim + 1 if self.sheaf_use_deg else self.input_dim
        self.lin_maps = nn.Linear(maps_input_dim, self.hidden_dim)

    def get_param_size(self):
        return self.d * (self.d - 1) // 2

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        for weight_learner in self.weight_learners:
            weight_learner.update_edge_index(edge_index)

    def forward(self, x):

        # Construct the sheaf Laplacians
        x_maps = x
        if self.sheaf_use_deg:
            deg = degree(self.edge_index[0], self.graph_size)
            x_maps = torch.concat((x_maps, deg.unsqueeze(-1)), dim=-1)

        x_maps = F.elu(self.lin_maps(x_maps))
        mean_params, var_params = self.sheaf_learner(x_maps, self.edge_index)
        mean_params = torch.tanh(mean_params)
        var_params = torch.sigmoid(var_params) * (1 - 0.05) + (0.05 / 2)  # bound away from 0 and 1 for stability
        edge_weights = self.weight_learner(x_maps, self.edge_index) if self.use_edge_weights else None

        kl = self.kldiv(None, var_params)

        Ls = []
        for layer in range(self.layers):
            L, trans_maps = self.laplacian_builder(mean_params, var_params, edge_weights)
            Ls.append(L)

        # Pass data through network
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        for layer in range(self.layers):

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])

            # Use the adjacency matrix rather than the diagonal
            x = torch_sparse.spmm(Ls[layer][0], Ls[layer][1], x.size(0), x.size(0), x)
            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1), kl


class BayesGeneralSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(BayesGeneralSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 1

        self.kldiv = KLDivergence('normal')

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        if self.sparse_learner:
            self.sheaf_learner = BayesConcatSheafLearnerVariant(self.final_d,
                self.hidden_channels, mean_size=self.d**2, var_size=self.d**2)
        else:
            self.sheaf_learner = BayesConcatSheafLearner(
                self.hidden_dim, mean_size=self.d**2, var_size=self.d**2)

        self.laplacian_builder = lb.BayesGeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_lp=self.add_lp, add_hp=self.add_hp,
            normalised=self.normalised, deg_normalised=self.deg_normalised)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x_maps = x
        mean_params, log_var_params = self.sheaf_learner(x_maps.reshape(self.graph_size, -1), self.edge_index)
        mean_params = torch.tanh(mean_params)
        log_var_params = -F.elu(-log_var_params)
        std_params = torch.exp(log_var_params)

        kl = self.kldiv(mean_params, log_var_params)

        Ls = []
        for layer in range(self.layers):
            L, trans_maps = self.laplacian_builder(mean_params, std_params)
            Ls.append(L)

        x0 = x
        for layer in range(self.layers):

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])

            # Use the adjacency matrix rather than the diagonal
            x = torch_sparse.spmm(Ls[layer][0], Ls[layer][1], x.size(0), x.size(0), x)
            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))

        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1), kl


class GCN(SheafDiffusion):
    # GCN model for comparison
    def __init__(self, edge_index, args):
        super(GCN, self).__init__(edge_index, args)
        assert args['d'] > 0

        self.final_d = self.d
        self.hidden_dim = self.d * self.hidden_channels

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.gcn = GCNConv(self.hidden_dim, self.hidden_dim)

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        for layer in range(self.layers):
            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[layer](x)

            x = x.reshape(self.graph_size, -1)
            x = self.gcn(x, self.edge_index)
            x = x.view(self.graph_size * self.final_d, -1)

            if self.use_act:
                x = F.elu(x)

        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)
