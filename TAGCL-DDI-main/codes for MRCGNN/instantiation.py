from torch import optim
from torch.optim import Adam
from layer import MRCGNN
import numpy as np
import os
import random
import torch
from augmentation import GATViewGenerator, GAT_NodeWeightEncoder

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

def Create_model(args, data_o):

    model = MRCGNN(feature=args.dimensions, hidden1=args.hidden1, hidden2=args.hidden2, decoder1=args.decoder1, dropout=args.dropout, zhongzi=args.zhongzi)
    view_ncla = GATViewGenerator(data_o, args.num_hidden, GAT_NodeWeightEncoder, args)      # 创建ncla增强对象

    args.lr = 0.0015
    optimizer = Adam([{'params': model.parameters()}, {'params': view_ncla.parameters()},], args.lr, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch//10))

    return model, view_ncla, optimizer, scheduler