"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data, Batch
# from dgl.nn.pytorch import GATConv
from torch_geometric.nn import VGAE, GATConv
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
import torch
import parms_setting as ps
import copy


class GAT_NodeWeightEncoder(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 num_head_NCLA,
                 num_head_Auto,
                 activation,
                 feat_drop,
                 negative_slope,
                 add_mask=False,
                 ):
        super().__init__()
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.num_dim = out_dim
        self.activate = activation
        self.num_head_NCLA = num_head_NCLA
        self.num_head_Auto = num_head_Auto

        # Initialize GATConv layers
        self.conv1 = GATConv(in_dim, num_hidden, heads=self.num_head_NCLA, dropout=feat_drop,
                             negative_slope=negative_slope, add_self_loops=False, concat=True)

        self.bn1 = nn.BatchNorm1d(num_hidden * self.num_head_NCLA)

        self.gat_layers = nn.ModuleList()
        for _ in range(self.num_head_NCLA):
            self.gat_layers.append(GATConv(num_hidden, out_dim, heads=self.num_head_Auto, dropout=feat_drop,
                                           negative_slope=negative_slope, add_self_loops=False, concat=True))

        self.bns = nn.ModuleList()
        for _ in range(self.num_head_NCLA):
            self.bns.append(nn.BatchNorm1d(out_dim))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_1, a = self.conv1(x, edge_index, return_attention_weights=True)
        x_2 = self.bn1(x_1)
        x_ = self.bn1(F.elu(x_2))

        # 计算每个列表应该包含的数据数量
        split_size = x_.shape[1] // self.num_head_NCLA

        # 按列进行切片，将 x 分成四个列表
        heads = [x_[:, i * split_size:(i + 1) * split_size] for i in range(self.num_head_NCLA)]
        a_adj = a[1]

        return heads, a_adj
class GATViewGenerator(VGAE):
    def __init__(self, dataset, dim, enc_, args, add_mask=False):
        self.add_mask = add_mask
        encoder = enc_(dataset.x.shape[1],  # Correct the input feature dimension
                       args.num_hidden,
                       args.out_dim,
                       args.num_layers,
                       args.num_head_NCLA,
                       args.num_head_Auto,
                       nn.ELU(),
                       args.in_drop,
                       args.negative_slope)
        # encoder = encoder(dataset, dim, self.add_mask)
        super().__init__(encoder=encoder)

    def forward(self, args, data_in, requires_grad):
        # with torch.no_grad():
        #     data = copy.deepcopy(data_in)
        data = copy.deepcopy(data_in).to('cuda')

        x, edge_index, e_type = data.x, data.edge_index, data.edge_type
        e_type = torch.tensor(e_type, dtype=torch.int64)

        num_s, dim_s = x.shape[0], x.shape[1]
        edge_attr = True
        if data.edge_type is not None:
            edge_attr = data.edge_type
            edge_attr = torch.tensor(edge_attr)

        data.x = data.x.float()
        x = x.float()
        x.requires_grad = requires_grad

        heads, attention = self.encoder(data)  # 调用GAT_NodeWeightEncoder对data_o的边进行增强;encode对数据data进行编码  p=[572,2]

        # get augmented data_ncla
        aug_data_ncla = []
        for i in range(args.num_head_NCLA):
            sub_data_ncla = Data(x=heads[i], edge_index=data.edge_index, y=data.y, edge_type=data.edge_type)   # 使用gatconv得到的邻接矩阵构造ncla增强的数据
            aug_data_ncla.append(sub_data_ncla)

        return heads, aug_data_ncla , attention    # data: generate sample; aug_data has two augmentions