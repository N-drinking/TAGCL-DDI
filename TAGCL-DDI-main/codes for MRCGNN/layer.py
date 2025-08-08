import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
import numpy as np
import csv
import os
import random
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_random_seed(1, deterministic=True)
def reset_parameters(w):
    stdv = 1. / math.sqrt(w.size(0))
    w.data.uniform_(-stdv, stdv)


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(32, 32, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 0) / torch.sum(msk)

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(in_channels, 2 * out_channels)
        self.linear2 = nn.Linear(2 * out_channels, out_channels)

    pass


class MRCGNN(nn.Module):
    def __init__(self, feature, hidden1, hidden2, decoder1, dropout,zhongzi,momentum=0.999):
        super(MRCGNN, self).__init__()

        self.num_layers = 4
        self.encoder_1 = RGCNConv(feature, hidden1, num_relations=65)
        self.encoder_2 = RGCNConv(hidden1, hidden2, num_relations=65)
        # 初始化列表来存储编码器层
        self.encoder_o1_ = nn.ModuleList()
        self.encoder_o2_ = nn.ModuleList()

        # 第一层的初始化 (可以根据需求调整)
        self.encoder_o1_.append(RGCNConv(feature, hidden1, num_relations=65))
        self.encoder_o2_.append(RGCNConv(hidden1, hidden2, num_relations=65))
        for l in range(1, self.num_layers):
            self.encoder_o1_.append(RGCNConv(feature, hidden1, num_relations=65))
            self.encoder_o2_.append(RGCNConv(hidden1, hidden2, num_relations=65))

        self.attt = torch.zeros(2)
        self.attt[0] = 0.8
        self.attt[1] = 0.2
        self.attt = nn.Parameter(self.attt)
        self.disc = Discriminator(hidden2 * 2)

        self.dropout = dropout
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        self.mlp = nn.ModuleList([nn.Linear(576, 256),
                                  nn.ELU(),
                                  nn.Dropout(p=0.1),
                                  nn.Linear(256, 128),
                                  nn.ELU(),
                                  nn.Dropout(p=0.1),
                                  nn.Linear(128, 65)
                                  ])

        drug_list = []
        with open('data/drug_listxiao.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                drug_list.append(row[0])
        features = np.load('trimnet/drug_emb_trimnet' + str(zhongzi) + '.npy')

        ids = np.load('trimnet/drug_idsxiao.npy')
        ids = ids.tolist()
        features1 = []
        for i in range(len(drug_list)):
            features1.append(features[ids.index(drug_list[i])])
        features1 = np.array(features1)
        self.features1 = torch.from_numpy(features1).cuda()

    def MLP(self, vectors, layer):
            for i in range(layer):
                vectors = self.mlp[i](vectors)

            return vectors

    def forward(self, data_o, data_s, idx):
        x2_os = []


        # **********
        x_oo, adj_o, e_type_o = data_o.x, data_o.edge_index, data_o.edge_type
        e_type_o = torch.tensor(e_type_o, dtype=torch.int64)
        x1_oo = F.relu(self.encoder_1(x_oo, adj_o, e_type_o))
        x1_oo = F.dropout(x1_oo, self.dropout, training=self.training)
        x2_oo = self.encoder_2(x1_oo, adj_o, e_type_o)  # data_ncla feature
        x2_os.append(x2_oo)        # ****
        for l in range(self.num_layers):
            # RGCN for DDI event graph and two corrupted graph
            # 提取ncla增强数据的特征
            x_o, adj, e_type = data_s[l].x, data_s[l].edge_index, data_s[l].edge_type
            e_type = torch.tensor(e_type, dtype=torch.int64)
            x1_o = F.relu(self.encoder_o1_[l](x_o, adj, e_type))
            x1_o = F.dropout(x1_o, self.dropout, training=self.training)
            x1_os = x1_o
            x2_o = self.encoder_o2_[l](x1_os, adj, e_type)
            x2_os.append(x2_o)  # data_ncla feature

        a = [int(i) for i in list(idx[0])]
        b = [int(i) for i in list(idx[1])]

        drug1_list = torch.tensor(a, dtype=torch.long)
        drug2_list = torch.tensor(b, dtype=torch.long)
        # layer attention
        embeds = torch.cat(x2_os, axis=1)   # +data_o feature

        drug1_embedding = embeds[drug1_list]
        drug2_embedding = embeds[drug2_list]

        # 输入x 特征
        drug1_o_embedding = self.features1[drug1_list].to('cuda')
        drug2_o_embedding = self.features1[drug2_list].to('cuda')

        input_features = torch.cat((drug1_embedding, drug1_o_embedding, drug2_embedding, drug2_o_embedding), dim=1)
        mlp_out = self.MLP(input_features, len(self.mlp))
        return x2_os, mlp_out
